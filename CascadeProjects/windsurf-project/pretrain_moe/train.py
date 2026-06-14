#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py  —  Multi-session pretraining loop for a from-scratch MoE GPT on a free
             Google Colab T4. nanoGPT-style, pure PyTorch, single-file-runnable.

WINNER CONFIG (mem obs #1848 / #1869, "B-core hardened with A+C grafts"):
    9 layers x 640d, head_dim=64, 8 experts top-2, d_ff=1920, ctx=1024,
    vocab=16384 byte-level BPE, ~202.9M total / ~69.5M active, ~6.4GB VRAM on a T4.

This file owns ONLY the training loop + the atomic Google-Drive multi-session
resume bundle. The architecture lives in model.py; the packed uint16 train.bin /
val.bin + tokenizer are produced by prepare_data.py. RANDOM INIT ONLY — train.py
never loads pretrained MODEL weights; on resume it reloads OUR OWN checkpoint.

WHAT THIS LOOP DOES
    * AdamW (fused when available) with model-provided decay / no-decay groups.
    * Cosine LR decay with linear warmup, indexed by GLOBAL step (so the schedule
      is identical across the ~9 Colab reconnects — it is a pure function of step).
    * Gradient accumulation: many micro-batches per optimizer step to reach a large
      effective token batch on one T4 without OOM.
    * AMP autocast (bf16 preferred on T4-class+; fp16 + GradScaler fallback).
    * MoE load-balancing aux loss + router z-loss are summed into the CE loss
      *inside the model*; train.py just logs the components and anneals the router
      exploration noise (1.0 -> 0.0) as a function of global step.
    * Periodic eval loss on the held-out val.bin (model.eval(), no_grad, no router
      noise) — a clean LM-CE number for tracking divergence across sessions.
    * MULTI-SESSION checkpointing every N steps to a Google-Drive dir, saving
      model + optimizer + scaler + global_step + token_count + scheduler position
      + torch/numpy/python RNG + config hash, written .tmp-then-os.replace so a
      mid-write Colab kill never corrupts the bundle.
    * On startup: auto-resume from the latest VALID checkpoint in the Drive dir,
      restoring everything exactly (step, tokens, optimizer, scaler, RNG, schedule)
      and fast-forwarding the data sampler by total tokens so the multi-session
      schedule is honored end-to-end. Designed to be KILLED at the 12h cap and
      resumed N times until the 6B-token budget is met.

VRAM PLAN (honest, the stance): bf16 weights(2)+grad(2)+fp32 master(4)+Adam m(4)+
    v(4) = 16 B/param * 202.9M = 3.25GB resident (all 8 experts + their full AdamW
    state stay on the GPU — MoE saves FLOPs, not memory). Gradient checkpointing +
    micro-batch B=48 @ T=1024 keeps activations ~2.9GB, with the (B,T,16384) logits
    buffer the single largest term. ~6.4GB of 16GB total -> a >9GB margin that
    absorbs cuBLAS workspace and eval-time logits spikes so we never OOM mid-run.

8B-ACTIVE SCALE-UP PATH (documented, NOT targeted): the SAME train.py reaches a
    Mixtral-class ~8B-active / ~47B-total model with a CONFIG-ONLY change to
    model.py (n_embd~4096, n_layer~32, d_ff~14336, head_dim 128, vocab ~64k,
    ctx 4096+). That is categorically impossible on one 16GB T4 — all 8 expert
    weights + their fp32 AdamW state can no longer be resident on a single device.
    It requires a multi-GPU cluster with expert-parallel + tensor-parallel sharding
    (Megatron / DeepSpeed-MoE), ZeRO-3 optimizer-state sharding, and a token budget
    in the trillions (~15-20x params). Only the config + parallelism strategy
    change — never this loop or the architecture family.

USAGE (Colab T4, multi-session):
    # one-time data prep (see prepare_data.py), then per session just run:
    python train.py --data-dir /content/drive/MyDrive/moe_pretrain/data \
                    --ckpt-dir /content/drive/MyDrive/moe_pretrain/ckpts \
                    --config   /content/drive/MyDrive/moe_pretrain/winner_config.json
    # Kill at the 12h cap; re-run the SAME command next session -> auto-resume.

    # quick CPU smoke test (tiny model, a handful of steps, no Drive needed):
    python train.py --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# model.py is the sibling architecture module (random init only).
try:
    from model import MoEGPT, MoEGPTConfig
except Exception as _e:  # pragma: no cover - import path edge cases
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import MoEGPT, MoEGPTConfig  # type: ignore


# ===========================================================================
# 1. TRAIN CONFIG  (all the loop hyper-parameters; JSON-serialisable)
# ===========================================================================
@dataclass
class TrainConfig:
    # --- where the packed data + tokenizer live (prepare_data.py output) ---
    data_dir: str = field(default_factory=lambda: os.environ.get(
        "PRETRAIN_DATA_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")))
    # --- where the multi-session resume bundles are WRITTEN (a Drive mount, or
    #     /kaggle/working/ckpts under --storage kaggle). ---
    ckpt_dir: str = field(default_factory=lambda: os.environ.get(
        "PRETRAIN_CKPT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts")))
    # --- where a resume bundle is READ on startup. Defaults to ckpt_dir so the
    #     classic single-dir (Colab/Drive) behavior is unchanged. On Kaggle the
    #     write dir (/kaggle/working/ckpts) and the read dir (/kaggle/input/
    #     weaver-moe-ckpt, a read-only mounted dataset) differ — set this to the
    #     mounted input so a fresh kernel resumes the previous version's output. ---
    resume_dir: Optional[str] = None
    # --- where the multi-session bundles live. "drive"/"local" keep the legacy
    #     single-dir behavior (resume_dir == ckpt_dir). "kaggle" wires the
    #     read-only-input / writable-working split + the wall-clock guard. The
    #     mode itself sets NO paths it isn't told (kaggle_entry.py passes explicit
    #     --ckpt-dir/--resume-dir), it only records intent + tunes guard logging. ---
    storage: str = "drive"
    # --- wall-clock guard (Kaggle hard-kills a kernel at ~12h). When > 0, the loop
    #     writes a final atomic checkpoint and exits 0 cleanly once this many
    #     seconds of WALL TIME have elapsed inside train(). 0 disables the guard
    #     (Colab/Drive path relies on the platform's own ~12h kill instead). ---
    max_seconds: float = 0.0
    # --- optional path to the winner_config.json (architecture). If absent we
    #     use model.py's defaults, which ARE the winning config. ---
    config_path: Optional[str] = None

    # --- batch geometry (the T4 VRAM plan: B=48 micro-batch, T=ctx_len) ---
    micro_batch_size: int = 48          # per-forward micro-batch (the VRAM knob)
    grad_accum_steps: int = 8           # micro-batches per optimizer step
    # effective tokens/optimizer-step = micro_batch_size * grad_accum * ctx_len
    #   = 48 * 8 * 1024 = ~393k tokens/step (a healthy large batch on one T4).

    # --- token budget / schedule length ---
    token_budget_b: float = 6.0         # total tokens across ALL sessions (6B plan)
    # max_steps is DERIVED from the budget + tokens-per-step unless overridden.
    max_steps: Optional[int] = None

    # --- optimizer (AdamW) ---
    learning_rate: float = 6e-4         # peak LR (post-warmup)
    min_lr: float = 6e-5                # cosine floor (== lr/10, Chinchilla-style)
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0              # global-norm clip (divergence insurance)

    # --- LR schedule (cosine w/ linear warmup, indexed by GLOBAL step) ---
    warmup_steps: int = 2000
    # decay_steps defaults to max_steps; cosine floors at min_lr afterwards.
    decay_steps: Optional[int] = None

    # --- router exploration noise anneal (1.0 -> 0.0 over these steps) ---
    router_noise_anneal_steps: int = 6000

    # --- precision ---
    dtype: str = "bf16"                 # "bf16" | "fp16" | "fp32"; auto-fallback if unsupported

    # --- eval ---
    eval_interval: int = 1000           # run eval every N optimizer steps
    eval_iters: int = 100               # micro-batches averaged per eval
    eval_at_start: bool = True          # one eval right after (re)load, for a baseline

    # --- logging ---
    log_interval: int = 10              # print a train-loss line every N steps

    # --- checkpointing (multi-session resume) ---
    ckpt_interval: int = 1500           # write a resume bundle every N steps (~spec)
    keep_last_k: int = 3                # how many timestamped bundles to retain
    always_save_latest: bool = True     # also maintain a stable ckpt_latest.pt symlink/copy

    # --- misc ---
    seed: int = 1337
    device: str = "auto"                # "auto" | "cuda" | "cpu"
    compile: bool = False               # torch.compile (off by default; T4 graphs are finicky)
    gradient_checkpointing: bool = True # recompute blocks in backward to fit B=48 (VRAM)

    # ---- derived / serialisation helpers ----
    def tokens_per_step(self, ctx_len: int) -> int:
        return self.micro_batch_size * self.grad_accum_steps * ctx_len

    def resolved_max_steps(self, ctx_len: int) -> int:
        if self.max_steps is not None:
            return int(self.max_steps)
        budget = int(self.token_budget_b * 1e9)
        return max(1, budget // self.tokens_per_step(ctx_len))

    def resolved_decay_steps(self, ctx_len: int) -> int:
        return int(self.decay_steps) if self.decay_steps is not None \
            else self.resolved_max_steps(ctx_len)

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        import hashlib
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


# ===========================================================================
# 2. LR SCHEDULE  (pure function of global step -> identical across sessions)
# ===========================================================================
def cosine_lr(step: int, peak_lr: float, min_lr: float,
              warmup_steps: int, decay_steps: int) -> float:
    """Linear warmup -> cosine decay -> flat floor. Deterministic in `step`, so
    a resumed run lands on EXACTLY the right LR with no extra state to persist."""
    if step < warmup_steps:
        # linear warmup from 0 -> peak (avoid div-by-zero at warmup_steps==0)
        return peak_lr * (step + 1) / max(1, warmup_steps)
    if step >= decay_steps:
        return min_lr
    # cosine from peak -> min over [warmup_steps, decay_steps)
    progress = (step - warmup_steps) / max(1, (decay_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return min_lr + coeff * (peak_lr - min_lr)


def router_noise_scale(step: int, anneal_steps: int) -> float:
    """Anneal the MoE exploration noise 1.0 -> 0.0 linearly over `anneal_steps`,
    then hold at 0. Pure function of step (no persisted state needed)."""
    if anneal_steps <= 0:
        return 0.0
    return max(0.0, 1.0 - step / float(anneal_steps))


# ===========================================================================
# 3. DATA LOADER  (uint16 memmap of packed tokens -> random ctx windows)
#    nanoGPT-style: re-mmap each batch (cheap, OS page cache), draw random
#    starting offsets. A per-step torch.Generator seeded by (base_seed, step)
#    makes the sample stream a PURE FUNCTION OF STEP, so resume fast-forwards
#    by simply restoring global_step — no shard cursor to thread through here.
# ===========================================================================
class PackedBinDataset:
    """Reads a uint16 .bin produced by prepare_data.py as flat packed tokens.

    get_batch(step) returns (x, y) of shape (B, T) int64 on the target device,
    sampled at random offsets whose RNG is keyed by the global step. Keying by
    step is what makes the multi-session schedule reproducible: session 3 picking
    up at step 41,000 draws the same windows it would have in one long run, so no
    window is systematically repeated or skipped across the ~9 reconnects."""

    def __init__(self, bin_path: str, ctx_len: int, dtype: str = "uint16"):
        self.bin_path = bin_path
        self.ctx_len = ctx_len
        self.np_dtype = np.dtype(dtype)
        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"Missing {bin_path}. Run prepare_data.py (--build) first to pack "
                f"train.bin / val.bin.")
        nbytes = os.path.getsize(bin_path)
        self.n_tokens = nbytes // self.np_dtype.itemsize
        # need at least one full (ctx_len + 1) window to form (x, y).
        self.max_start = self.n_tokens - (ctx_len + 1)
        if self.max_start <= 0:
            raise ValueError(
                f"{bin_path} holds only {self.n_tokens} tokens — need > {ctx_len+1} "
                f"for a single ctx window. Pack more data first.")

    def _mmap(self) -> np.memmap:
        # re-open per batch (nanoGPT pattern): avoids a leaked handle / stale page
        # cache across the long multi-session run, and is effectively free.
        return np.memmap(self.bin_path, dtype=self.np_dtype, mode="r")

    def get_batch(self, batch_size: int, step: int, device: torch.device,
                  base_seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self._mmap()
        # RNG keyed by (base_seed, step): same step -> same windows, every session.
        g = torch.Generator()
        g.manual_seed((base_seed * 1_000_003 + step) & 0x7FFFFFFFFFFFFFFF)
        ix = torch.randint(low=0, high=self.max_start, size=(batch_size,), generator=g)
        # gather windows (numpy slice into int64 to be safe with cross_entropy).
        x = torch.stack([
            torch.from_numpy(data[i:i + self.ctx_len].astype(np.int64)) for i in ix.tolist()
        ])
        y = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + self.ctx_len].astype(np.int64)) for i in ix.tolist()
        ])
        if device.type == "cuda":
            # pin + non-blocking async copy overlaps H2D with compute.
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y


# ===========================================================================
# 4. RNG STATE  (capture / restore torch + cuda + numpy + python for resume)
# ===========================================================================
def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            t = state["torch"]
            # torch wants a ByteTensor on CPU.
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.uint8)
            torch.set_rng_state(t.to("cpu", torch.uint8))
        if "cuda" in state and torch.cuda.is_available():
            cuda_states = state["cuda"]
            fixed = []
            for s in cuda_states:
                if not isinstance(s, torch.Tensor):
                    s = torch.tensor(s, dtype=torch.uint8)
                fixed.append(s.to("cpu", torch.uint8))
            torch.cuda.set_rng_state_all(fixed)
    except Exception as e:  # never let RNG restore crash a resume.
        print(f"[resume] WARNING: could not fully restore RNG state ({e}). "
              f"Continuing with best-effort restore.", file=sys.stderr)


# ===========================================================================
# 5. CHECKPOINT BUNDLE  (atomic .tmp -> os.replace; full resume of everything)
# ===========================================================================
CKPT_PREFIX = "ckpt_step"
CKPT_LATEST = "ckpt_latest.pt"


def _atomic_torch_save(obj: dict, path: str) -> None:
    """Write to <path>.tmp, fsync, then os.replace -> path. os.replace is atomic
    on a single filesystem, so a Colab kill mid-write leaves the OLD checkpoint
    intact and the .tmp is simply discarded on the next run."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint(ckpt_dir: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scaler: Optional[torch.cuda.amp.GradScaler],
                    global_step: int, token_count: int,
                    model_cfg: MoEGPTConfig, train_cfg: TrainConfig,
                    best_val: float, keep_last_k: int,
                    always_latest: bool) -> str:
    """Persist the COMPLETE resume bundle (the spec's atomic Drive bundle)."""
    bundle = {
        # --- the model itself (OUR weights — never pretrained) ---
        "model": model.state_dict(),
        # --- full AdamW state (m, v, step per-param-group) ---
        "optimizer": optimizer.state_dict(),
        # --- AMP scaler (fp16 path); None on bf16/fp32 ---
        "scaler": scaler.state_dict() if scaler is not None else None,
        # --- exact scheduler position (LR is a pure fn of this) ---
        "global_step": int(global_step),
        # --- token counter (honors the 6B budget end-to-end) ---
        "token_count": int(token_count),
        # --- best eval loss so far (for logging continuity) ---
        "best_val": float(best_val),
        # --- ALL RNG streams (torch/cuda/numpy/python) ---
        "rng_state": capture_rng_state(),
        # --- config + hashes so a resume refuses a mismatched architecture ---
        "model_config": model_cfg.to_dict(),
        "model_config_hash": model_cfg.config_hash(),
        "train_config": train_cfg.to_dict(),
        "train_config_hash": train_cfg.config_hash(),
        # --- provenance ---
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": torch.__version__,
        "format_version": 1,
    }

    step_path = os.path.join(ckpt_dir, f"{CKPT_PREFIX}{global_step:09d}.pt")
    _atomic_torch_save(bundle, step_path)
    if always_latest:
        # a stable filename the next session loads first (also atomic).
        _atomic_torch_save(bundle, os.path.join(ckpt_dir, CKPT_LATEST))
    _prune_old_checkpoints(ckpt_dir, keep_last_k)
    return step_path


def _list_step_checkpoints(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return []
    out = []
    for fn in os.listdir(ckpt_dir):
        if fn.startswith(CKPT_PREFIX) and fn.endswith(".pt") and not fn.endswith(".tmp"):
            try:
                step = int(fn[len(CKPT_PREFIX):-3])
                out.append((step, os.path.join(ckpt_dir, fn)))
            except ValueError:
                continue
    out.sort(key=lambda t: t[0])
    return out


def _prune_old_checkpoints(ckpt_dir: str, keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return
    ckpts = _list_step_checkpoints(ckpt_dir)
    for _, path in ckpts[:-keep_last_k]:
        try:
            os.remove(path)
        except OSError:
            pass


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Prefer the stable ckpt_latest.pt; fall back to the highest-step bundle.
    Validate by attempting a metadata-only load so a half-written file (despite
    the atomic write) is skipped rather than crashing the resume."""
    candidates = []
    latest = os.path.join(ckpt_dir, CKPT_LATEST)
    if os.path.exists(latest):
        candidates.append(latest)
    candidates += [p for _, p in reversed(_list_step_checkpoints(ckpt_dir))]
    for path in candidates:
        try:
            # cheap-ish validity probe: load to CPU; torch.load raises on truncation.
            _ = torch.load(path, map_location="cpu", weights_only=False)
            return path
        except Exception as e:
            print(f"[resume] checkpoint {path} unreadable ({e}); trying older one.",
                  file=sys.stderr)
            continue
    return None


# ===========================================================================
# 6. DEVICE / DTYPE SETUP
# ===========================================================================
def resolve_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            print("[device] cuda requested but unavailable -> falling back to cpu.",
                  file=sys.stderr)
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_amp(dtype_pref: str, device: torch.device):
    """Return (autocast_dtype, use_scaler). bf16 is preferred on T4-class+ GPUs;
    fp16 falls back to a GradScaler; fp32 disables autocast entirely."""
    if device.type != "cuda":
        return None, False  # CPU: run fp32, no autocast
    if dtype_pref == "fp32":
        return None, False
    if dtype_pref == "bf16":
        # CAUTION: torch.cuda.is_bf16_supported() returns True on Turing (T4, sm_75)
        # because PyTorch 2.x counts the emulation fallback as "supported" — but
        # Turing has NO bf16 tensor cores, so bf16 matmuls drop off the fast path and
        # run ~3-4x slower (this is what capped the Kaggle T4 run at ~3.3k tok/s).
        # Require REAL bf16 tensor cores (Ampere sm_80+); otherwise use fp16, which
        # DOES hit the T4's fp16 tensor cores. Gate on compute capability, not the
        # emulation-aware helper.
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return torch.bfloat16, False  # bf16 needs no loss scaler
        print(f"[amp] no bf16 tensor cores on sm_{major}x (< sm_80) -> "
              "using fp16 + GradScaler for tensor-core throughput.",
              file=sys.stderr)
        return torch.float16, True
    # explicit fp16
    return torch.float16, True


# ===========================================================================
# 7. EVAL  (clean LM-CE on the val split; no router noise, no grad)
# ===========================================================================
@torch.no_grad()
def evaluate(model: torch.nn.Module, val_ds: PackedBinDataset, cfg: TrainConfig,
             device: torch.device, amp_dtype, step: int) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    # Eval uses noise_scale 0 so the gate is the deterministic, deployed router.
    if hasattr(model, "set_router_noise_scale"):
        model.set_router_noise_scale(0.0)

    losses = torch.zeros(cfg.eval_iters)
    lm_losses = torch.zeros(cfg.eval_iters)
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if (device.type == "cuda" and amp_dtype is not None) else nullcontext())
    for i in range(cfg.eval_iters):
        # eval RNG keyed off a disjoint step namespace so it never aliases train.
        x, y = val_ds.get_batch(cfg.micro_batch_size, step + 10_000_000 + i,
                                device, base_seed=cfg.seed + 777)
        with autocast_ctx:
            _, loss, aux = model(x, y)
        losses[i] = loss.item()
        lm_losses[i] = aux["lm_loss"].item()

    if was_training:
        model.train()
    mean_lm = lm_losses.mean().item()
    return {
        "val_loss": losses.mean().item(),
        "val_lm_loss": mean_lm,
        "val_ppl": math.exp(min(20.0, mean_lm)),  # clamp to avoid inf early
    }


# ===========================================================================
# 8. BUILD MODEL + OPTIMIZER
# ===========================================================================
def build_model_config(train_cfg: TrainConfig, meta: Optional[dict]) -> MoEGPTConfig:
    """Architecture from --config JSON if given, else model.py winner defaults.
    The data meta.json (vocab_size / ctx_len) is the source of truth for the two
    geometry fields that MUST match the packed .bin, so we override them from meta
    and assert agreement to avoid a silent shape mismatch."""
    if train_cfg.config_path and os.path.exists(train_cfg.config_path):
        with open(train_cfg.config_path) as f:
            d = json.load(f)
        # tolerate the full winner-config object (rationale/est fields ignored).
        mcfg = MoEGPTConfig.from_dict(d)
    else:
        mcfg = MoEGPTConfig()  # the winning config defaults

    if meta:
        mv, mc = meta.get("vocab_size"), meta.get("ctx_len")
        if mv and mv != mcfg.vocab_size:
            print(f"[config] overriding vocab_size {mcfg.vocab_size} -> {mv} "
                  f"(from data meta.json).")
            mcfg.vocab_size = int(mv)
        if mc and mc != mcfg.ctx_len:
            print(f"[config] overriding ctx_len {mcfg.ctx_len} -> {mc} "
                  f"(from data meta.json).")
            mcfg.ctx_len = int(mc)
        # re-run dataclass validation after the overrides.
        mcfg.__post_init__()
    return mcfg


def build_optimizer(model: MoEGPT, cfg: TrainConfig, device: torch.device):
    groups = model.configure_param_groups(weight_decay=cfg.weight_decay)
    # fused AdamW is a big throughput win on CUDA; gracefully fall back if absent.
    use_fused = False
    if device.type == "cuda":
        try:
            use_fused = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        except Exception:
            use_fused = False
    extra = {"fused": True} if use_fused else {}
    optim = torch.optim.AdamW(
        groups, lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2), eps=1e-8, **extra,
    )
    if use_fused:
        print("[optim] using fused AdamW.")
    return optim


# ===========================================================================
# 9. TRAIN
# ===========================================================================
def train(train_cfg: TrainConfig) -> None:
    # --- seed everything (resume will overwrite RNG from the bundle) ---
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed & 0xFFFFFFFF)
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)
    # TF32 matmuls: a free throughput win on Ampere+; harmless on T4.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = resolve_device(train_cfg.device)
    amp_dtype, use_scaler = resolve_amp(train_cfg.dtype, device)
    print(f"[setup] device={device}  amp_dtype={amp_dtype}  scaler={use_scaler}")

    # --- data meta (vocab/ctx/dtype/paths) from prepare_data.py ---
    meta_path = os.path.join(train_cfg.data_dir, "meta.json")
    meta = None
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        print(f"[data] meta: vocab={meta.get('vocab_size')} ctx={meta.get('ctx_len')} "
              f"train_tok={meta.get('train_tokens')} val_tok={meta.get('val_tokens')}")
    else:
        print(f"[data] no meta.json at {meta_path}; using model.py config defaults. "
              f"(Run prepare_data.py to produce train.bin/val.bin/meta.json.)",
              file=sys.stderr)

    # --- model (RANDOM INIT) ---
    model_cfg = build_model_config(train_cfg, meta)
    model = MoEGPT.from_config(model_cfg).to(device)
    ctx_len = model_cfg.ctx_len
    print(f"[model] {model.num_params()/1e6:.2f}M total / "
          f"{model.num_active_params()/1e6:.2f}M active  "
          f"config_hash={model_cfg.config_hash()}")

    # Gradient checkpointing: recompute each transformer block in the backward
    # pass so the B=48 activation footprint fits the T4. We monkey-wrap the block
    # forwards via torch.utils.checkpoint without touching model.py.
    if train_cfg.gradient_checkpointing and device.type == "cuda":
        _enable_gradient_checkpointing(model)
        print("[model] gradient checkpointing ENABLED (T4 activation budget).")

    if train_cfg.compile and hasattr(torch, "compile"):
        print("[model] torch.compile(model) ... (first step will be slow)")
        model = torch.compile(model)  # type: ignore

    # --- optimizer + AMP scaler ---
    optimizer = build_optimizer(model, train_cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler) if use_scaler else None

    # --- schedule geometry ---
    max_steps = train_cfg.resolved_max_steps(ctx_len)
    decay_steps = train_cfg.resolved_decay_steps(ctx_len)
    tokens_per_step = train_cfg.tokens_per_step(ctx_len)
    print(f"[sched] max_steps={max_steps}  warmup={train_cfg.warmup_steps}  "
          f"decay_steps={decay_steps}  tokens/step={tokens_per_step/1e3:.0f}k  "
          f"budget={train_cfg.token_budget_b}B")

    # --- AUTO-RESUME from the latest valid bundle ---
    # The READ dir defaults to ckpt_dir (legacy single-dir Drive/Colab path). On
    # Kaggle the read dir is a read-only mounted dataset distinct from the
    # writable working ckpt_dir, so we look in resume_dir first, then fall back to
    # the write dir (covers a same-session re-launch that already wrote there).
    global_step = 0
    token_count = 0
    best_val = float("inf")
    resume_dir = train_cfg.resume_dir or train_cfg.ckpt_dir
    ckpt_path = find_latest_checkpoint(resume_dir)
    if ckpt_path is None and resume_dir != train_cfg.ckpt_dir:
        ckpt_path = find_latest_checkpoint(train_cfg.ckpt_dir)
    if ckpt_path is not None:
        global_step, token_count, best_val = _resume_from_checkpoint(
            ckpt_path, model, optimizer, scaler, model_cfg)
        print(f"[resume] RESUMED from {ckpt_path} at "
              f"step={global_step}  tokens={token_count/1e6:.1f}M  "
              f"best_val={best_val:.4f}")
    else:
        print(f"[resume] no checkpoint in {resume_dir}"
              + (f" or {train_cfg.ckpt_dir}" if resume_dir != train_cfg.ckpt_dir else "")
              + " -> FRESH RANDOM-INIT run.")

    if global_step >= max_steps:
        print(f"[train] global_step {global_step} >= max_steps {max_steps}. "
              f"Budget already met — nothing to do.")
        return

    # --- datasets ---
    bin_dtype = (meta or {}).get("dtype", "uint16")
    train_ds = PackedBinDataset(os.path.join(train_cfg.data_dir, "train.bin"),
                                ctx_len, bin_dtype)
    val_path = os.path.join(train_cfg.data_dir, "val.bin")
    val_ds = PackedBinDataset(val_path, ctx_len, bin_dtype) if os.path.exists(val_path) else None
    print(f"[data] train tokens on disk: {train_ds.n_tokens/1e6:.1f}M"
          + (f"  val tokens: {val_ds.n_tokens/1e6:.1f}M" if val_ds else "  (no val.bin)"))

    autocast_ctx_factory = (
        (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype))
        if (device.type == "cuda" and amp_dtype is not None) else (lambda: nullcontext())
    )

    # --- optional baseline eval right after (re)load ---
    if train_cfg.eval_at_start and val_ds is not None:
        m = evaluate(model, val_ds, train_cfg, device, amp_dtype, global_step)
        best_val = min(best_val, m["val_loss"])
        print(f"[eval@start] step={global_step}  val_loss={m['val_loss']:.4f}  "
              f"val_lm={m['val_lm_loss']:.4f}  ppl={m['val_ppl']:.2f}")

    # ----------------------- THE LOOP -----------------------
    # Wall-clock guard: on Kaggle the kernel is HARD-KILLED at ~12h, so we set a
    # deadline (default disabled; kaggle_entry.py passes ~11.5h) and break out to
    # write a final atomic checkpoint + exit 0 BEFORE the platform kill. The guard
    # also halts at the START of a step (never mid-optimizer-step) so the bundle is
    # always a clean step boundary.
    t_start = time.time()
    deadline = (t_start + train_cfg.max_seconds) if train_cfg.max_seconds > 0 else None
    if deadline is not None:
        print(f"[guard] wall-clock guard ARMED: {train_cfg.max_seconds/3600.0:.2f}h "
              f"budget -> final checkpoint + clean exit before the platform kill.")
    hit_deadline = False

    model.train()
    t_step = time.time()
    running_lm = running_aux = running_z = 0.0
    n_running = 0

    while global_step < max_steps:
        # --- wall-clock guard: stop cleanly at a step boundary before the kill ---
        if deadline is not None and time.time() >= deadline:
            print(f"[guard] wall-clock budget reached at step {global_step} "
                  f"(tokens={token_count/1e6:.1f}M) -> writing final checkpoint and "
                  f"exiting 0 for the next session to resume.")
            hit_deadline = True
            break
        # set LR + router noise for THIS step (both pure fns of global_step).
        lr = cosine_lr(global_step, train_cfg.learning_rate, train_cfg.min_lr,
                       train_cfg.warmup_steps, decay_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        if hasattr(model, "set_router_noise_scale"):
            model.set_router_noise_scale(
                router_noise_scale(global_step, train_cfg.router_noise_anneal_steps))

        optimizer.zero_grad(set_to_none=True)
        step_lm = step_aux = step_z = 0.0

        # ---- gradient accumulation over micro-batches ----
        for micro in range(train_cfg.grad_accum_steps):
            # key the data RNG by a unique (step, micro) so accumulation micro-
            # batches differ AND the whole stream is reproducible on resume.
            data_step = global_step * train_cfg.grad_accum_steps + micro
            x, y = train_ds.get_batch(train_cfg.micro_batch_size, data_step,
                                      device, base_seed=train_cfg.seed)
            with autocast_ctx_factory():
                _, loss, aux = model(x, y)
                # scale so the accumulated grad == mean over micro-batches.
                loss = loss / train_cfg.grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_lm += aux["lm_loss"].item() / train_cfg.grad_accum_steps
            step_aux += aux["aux_loss"].item() / train_cfg.grad_accum_steps
            step_z += aux["z_loss"].item() / train_cfg.grad_accum_steps

        # ---- optimizer step (unscale -> clip -> step -> update scaler) ----
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

        global_step += 1
        token_count += tokens_per_step
        running_lm += step_lm; running_aux += step_aux; running_z += step_z
        n_running += 1

        # ---- periodic train log ----
        if global_step % train_cfg.log_interval == 0:
            dt = time.time() - t_step
            tok_s = train_cfg.log_interval * tokens_per_step / max(1e-9, dt)
            print(f"step {global_step:>7d}/{max_steps}  "
                  f"lm {running_lm/n_running:.4f}  "
                  f"aux {running_aux/n_running:.4f}  "
                  f"z {running_z/n_running:.4f}  "
                  f"lr {lr:.2e}  "
                  f"{tok_s/1e3:.1f}k tok/s  "
                  f"tok {token_count/1e6:.0f}M/{train_cfg.token_budget_b*1e3:.0f}M  "
                  f"noise {router_noise_scale(global_step, train_cfg.router_noise_anneal_steps):.2f}")
            running_lm = running_aux = running_z = 0.0
            n_running = 0
            t_step = time.time()

        # ---- periodic eval ----
        if val_ds is not None and global_step % train_cfg.eval_interval == 0:
            m = evaluate(model, val_ds, train_cfg, device, amp_dtype, global_step)
            improved = m["val_loss"] < best_val
            best_val = min(best_val, m["val_loss"])
            print(f"[eval] step={global_step}  val_loss={m['val_loss']:.4f}  "
                  f"val_lm={m['val_lm_loss']:.4f}  ppl={m['val_ppl']:.2f}"
                  f"{'  <- best' if improved else ''}")
            t_step = time.time()  # don't count eval time in throughput

        # ---- periodic atomic checkpoint to Drive (multi-session resume) ----
        if global_step % train_cfg.ckpt_interval == 0:
            path = save_checkpoint(
                train_cfg.ckpt_dir, model, optimizer, scaler,
                global_step, token_count, model_cfg, train_cfg,
                best_val, train_cfg.keep_last_k, train_cfg.always_save_latest)
            print(f"[ckpt] step={global_step}  tokens={token_count/1e6:.1f}M  "
                  f"-> {os.path.basename(path)} (atomic; resumable across sessions)")
            t_step = time.time()

    # ---- final checkpoint at budget completion OR at the wall-clock guard ----
    final_path = save_checkpoint(
        train_cfg.ckpt_dir, model, optimizer, scaler,
        global_step, token_count, model_cfg, train_cfg,
        best_val, train_cfg.keep_last_k, train_cfg.always_save_latest)
    if hit_deadline:
        print(f"[done] SESSION CAP (wall-clock guard) at step {global_step}/{max_steps}  "
              f"tokens={token_count/1e6:.1f}M  best_val={best_val:.4f}  "
              f"-> {final_path}. Push this as the new ckpt-dataset version and re-run "
              f"to resume EXACTLY here next session.")
    else:
        print(f"[done] reached step {global_step}/{max_steps}  "
              f"tokens={token_count/1e6:.1f}M  best_val={best_val:.4f}  "
              f"final checkpoint -> {final_path}. 6B-budget run complete "
              f"(or this session's cap).")


# ===========================================================================
# 9b. RESUME + GRADIENT-CHECKPOINT HELPERS
# ===========================================================================
def _resume_from_checkpoint(path: str, model: torch.nn.Module,
                            optimizer: torch.optim.Optimizer,
                            scaler: Optional[torch.cuda.amp.GradScaler],
                            model_cfg: MoEGPTConfig) -> Tuple[int, int, float]:
    """Restore model + optimizer + scaler + step + tokens + RNG from a bundle.
    Refuses to load a mismatched architecture (would corrupt the run)."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)

    saved_hash = bundle.get("model_config_hash")
    if saved_hash and saved_hash != model_cfg.config_hash():
        raise RuntimeError(
            f"Checkpoint architecture hash {saved_hash} != current "
            f"{model_cfg.config_hash()}. Refusing to resume a mismatched model "
            f"(point --config at the original winner_config.json, or start fresh "
            f"in a clean --ckpt-dir).")

    # handle a possible torch.compile() "_orig_mod." prefix on either side.
    state = bundle["model"]
    target = model
    missing, unexpected = _load_state_dict_flexible(target, state)
    if missing or unexpected:
        print(f"[resume] state_dict: {len(missing)} missing, {len(unexpected)} "
              f"unexpected keys (tolerated).", file=sys.stderr)

    optimizer.load_state_dict(bundle["optimizer"])
    # move optimizer state tensors to the model's device (load was map_location=cpu).
    _optimizer_state_to_device(optimizer, next(model.parameters()).device)

    if scaler is not None and bundle.get("scaler") is not None:
        try:
            scaler.load_state_dict(bundle["scaler"])
        except Exception as e:
            print(f"[resume] scaler restore failed ({e}); continuing with fresh scaler.",
                  file=sys.stderr)

    restore_rng_state(bundle.get("rng_state"))

    return (int(bundle.get("global_step", 0)),
            int(bundle.get("token_count", 0)),
            float(bundle.get("best_val", float("inf"))))


def _load_state_dict_flexible(model: torch.nn.Module, state: dict):
    """Load a state dict, stripping/adding a torch.compile '_orig_mod.' prefix as
    needed so a checkpoint saved compiled loads uncompiled and vice-versa."""
    model_keys = list(model.state_dict().keys())
    model_has_prefix = any(k.startswith("_orig_mod.") for k in model_keys)
    state_has_prefix = any(k.startswith("_orig_mod.") for k in state.keys())
    if model_has_prefix and not state_has_prefix:
        state = {f"_orig_mod.{k}": v for k, v in state.items()}
    elif state_has_prefix and not model_has_prefix:
        state = {k[len("_orig_mod."):]: v for k, v in state.items()}
    return model.load_state_dict(state, strict=False)


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device) -> None:
    for st in optimizer.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device)


def _enable_gradient_checkpointing(model: MoEGPT) -> None:
    """Wrap each transformer Block.forward with torch.utils.checkpoint so the
    block's activations are recomputed in backward instead of stored. This is the
    knob that lets micro-batch B=48 @ T=1024 fit the T4. We avoid editing model.py
    by rebinding forward on each block; use_reentrant=False is the modern, safe
    variant that plays well with AMP autocast."""
    from torch.utils.checkpoint import checkpoint

    try:
        blocks = model.transformer.h
    except AttributeError:
        print("[ckpt] could not find transformer.h blocks; gradient checkpointing "
              "skipped.", file=sys.stderr)
        return

    for block in blocks:
        if getattr(block, "_grad_ckpt_wrapped", False):
            continue
        orig_forward = block.forward

        def make_wrapper(fwd):
            def wrapped(x, *args, **kwargs):
                if torch.is_grad_enabled() and x.requires_grad:
                    return checkpoint(fwd, x, *args, use_reentrant=False, **kwargs)
                return fwd(x, *args, **kwargs)
            return wrapped

        block.forward = make_wrapper(orig_forward)  # type: ignore[assignment]
        block._grad_ckpt_wrapped = True  # type: ignore[attr-defined]


# ===========================================================================
# 10. SMOKE TEST  (tiny model, a few steps, CPU/GPU, no Drive needed)
# ===========================================================================
def _smoke_test() -> None:
    """End-to-end sanity: build tiny data, run a few steps, checkpoint, and verify
    a from-scratch resume restores global_step/token_count/optimizer exactly."""
    import tempfile

    print("=" * 70)
    print("train.py SMOKE TEST — tiny MoE, packed random bin, resume round-trip")
    print("=" * 70)

    tmp = tempfile.mkdtemp(prefix="moe_smoke_")
    data_dir = os.path.join(tmp, "data")
    ckpt_dir = os.path.join(tmp, "ckpts")
    os.makedirs(data_dir, exist_ok=True)

    # tiny architecture so this runs on CPU in seconds.
    tiny = MoEGPTConfig(
        vocab_size=512, ctx_len=64, n_layer=2, n_head=2, n_embd=64,
        head_dim=32, n_experts=4, top_k=2, d_ff=128,
    )
    # write a config.json + meta.json + random packed bins.
    cfg_path = os.path.join(tmp, "tiny_config.json")
    json.dump(tiny.to_dict(), open(cfg_path, "w"))
    meta = {"vocab_size": tiny.vocab_size, "ctx_len": tiny.ctx_len,
            "dtype": "uint16", "eos_id": 0,
            "train_tokens": 200_000, "val_tokens": 20_000}
    json.dump(meta, open(os.path.join(data_dir, "meta.json"), "w"))
    rng = np.random.default_rng(0)
    rng.integers(0, tiny.vocab_size, size=200_000, dtype=np.uint16).tofile(
        os.path.join(data_dir, "train.bin"))
    rng.integers(0, tiny.vocab_size, size=20_000, dtype=np.uint16).tofile(
        os.path.join(data_dir, "val.bin"))

    common = dict(
        data_dir=data_dir, ckpt_dir=ckpt_dir, config_path=cfg_path,
        micro_batch_size=4, grad_accum_steps=2, max_steps=12,
        warmup_steps=2, decay_steps=12, router_noise_anneal_steps=8,
        eval_interval=6, eval_iters=3, ckpt_interval=6, log_interval=2,
        dtype="fp32", device="cpu", gradient_checkpointing=False,
        keep_last_k=2, eval_at_start=True,
    )

    print("\n--- PHASE 1: train 12 steps from scratch ---")
    train(TrainConfig(**common))

    ckpt = find_latest_checkpoint(ckpt_dir)
    assert ckpt is not None, "no checkpoint written"
    b = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert b["global_step"] == 12, f"expected step 12, got {b['global_step']}"
    assert b["token_count"] == 12 * 4 * 2 * tiny.ctx_len, "token_count mismatch"
    assert b["optimizer"]["state"], "optimizer state empty (no AdamW m/v saved)"
    assert "rng_state" in b and "model_config_hash" in b
    print(f"  checkpoint OK: step={b['global_step']} tokens={b['token_count']} "
          f"hash={b['model_config_hash']}")

    print("\n--- PHASE 2: extend max_steps -> 18, AUTO-RESUME from step 12 ---")
    common2 = dict(common)
    common2["max_steps"] = 18
    common2["decay_steps"] = 18
    common2["eval_at_start"] = False
    train(TrainConfig(**common2))
    b2 = torch.load(find_latest_checkpoint(ckpt_dir), map_location="cpu", weights_only=False)
    assert b2["global_step"] == 18, f"resume should reach 18, got {b2['global_step']}"
    print(f"  resume OK: continued 12 -> {b2['global_step']} "
          f"(tokens {b2['token_count']}); optimizer + RNG carried across sessions.")

    # idempotent: re-running at/after budget should no-op.
    print("\n--- PHASE 3: re-run at budget -> should no-op ---")
    train(TrainConfig(**common2))
    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED — random init, AdamW+cosine+accum+AMP-path, eval,")
    print("atomic multi-session checkpoint, and exact resume of step/tokens/optim.")
    print("=" * 70)


# ===========================================================================
# 11. CLI
# ===========================================================================
def parse_args(argv) -> TrainConfig:
    ap = argparse.ArgumentParser(
        description="Multi-session pretraining loop for a from-scratch MoE GPT "
                    "on a free Colab T4 (resumable across the ~12h session cap).")
    ap.add_argument("--smoke", action="store_true",
                    help="run the tiny CPU smoke test (no Drive / no real data needed).")
    ap.add_argument("--data-dir", default=None, help="dir with train.bin/val.bin/meta.json.")
    ap.add_argument("--ckpt-dir", default=None,
                    help="dir resume bundles are WRITTEN to (Drive mount, or "
                         "/kaggle/working/ckpts under --storage kaggle).")
    ap.add_argument("--resume-dir", default=None,
                    help="dir a resume bundle is READ from on startup (defaults to "
                         "--ckpt-dir). On Kaggle point this at the read-only mounted "
                         "checkpoint dataset, e.g. /kaggle/input/weaver-moe-ckpt.")
    ap.add_argument("--storage", choices=["drive", "local", "kaggle"], default=None,
                    help="storage/runtime mode. 'drive'/'local' = legacy single-dir "
                         "resume (read==write dir). 'kaggle' = read-only-input / "
                         "writable-working split + wall-clock guard (see --max-hours).")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="wall-clock guard: write a final atomic checkpoint and exit 0 "
                         "after this many seconds (0 = disabled). Mutually informs "
                         "--max-hours.")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock guard in HOURS (convenience; converted to seconds). "
                         "Use ~11.5 on Kaggle to beat the ~12h hard kill.")
    ap.add_argument("--config", dest="config_path", default=None,
                    help="winner_config.json (architecture). Omit to use model.py defaults.")
    ap.add_argument("--micro-batch-size", type=int, default=None)
    ap.add_argument("--grad-accum-steps", type=int, default=None)
    ap.add_argument("--token-budget-b", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--min-lr", type=float, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    ap.add_argument("--eval-interval", type=int, default=None)
    ap.add_argument("--eval-iters", type=int, default=None)
    ap.add_argument("--ckpt-interval", type=int, default=None)
    ap.add_argument("--log-interval", type=int, default=None)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    ap.add_argument("--compile", action="store_true", help="enable torch.compile.")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing (uses more VRAM).")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    if args.smoke:
        return None  # handled in main()

    cfg = TrainConfig()
    # only override fields the user explicitly set.
    for name, attr in [
        ("data_dir", "data_dir"), ("ckpt_dir", "ckpt_dir"),
        ("resume_dir", "resume_dir"), ("storage", "storage"),
        ("max_seconds", "max_seconds"),
        ("config_path", "config_path"),
        ("micro_batch_size", "micro_batch_size"), ("grad_accum_steps", "grad_accum_steps"),
        ("token_budget_b", "token_budget_b"), ("max_steps", "max_steps"),
        ("learning_rate", "learning_rate"), ("min_lr", "min_lr"),
        ("warmup_steps", "warmup_steps"), ("weight_decay", "weight_decay"),
        ("grad_clip", "grad_clip"), ("dtype", "dtype"),
        ("eval_interval", "eval_interval"), ("eval_iters", "eval_iters"),
        ("ckpt_interval", "ckpt_interval"), ("log_interval", "log_interval"),
        ("device", "device"), ("seed", "seed"),
    ]:
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, attr, v)
    # --max-hours is a convenience that overrides --max-seconds when given.
    if args.max_hours is not None:
        cfg.max_seconds = float(args.max_hours) * 3600.0
    if args.compile:
        cfg.compile = True
    if args.no_grad_ckpt:
        cfg.gradient_checkpointing = False
    return cfg


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--smoke" in argv:
        _smoke_test()
        return 0
    cfg = parse_args(argv)
    print("=" * 70)
    print(f"from-scratch MoE GPT pretrain — multi-session (storage={cfg.storage})")
    print(f"  python {platform.python_version()}  torch {torch.__version__}")
    print(f"  data_dir   = {os.path.abspath(cfg.data_dir)}")
    print(f"  ckpt_dir   = {os.path.abspath(cfg.ckpt_dir)}  (writes)")
    _rdir = cfg.resume_dir or cfg.ckpt_dir
    print(f"  resume_dir = {os.path.abspath(_rdir)}  (reads)")
    print(f"  config     = {cfg.config_path or '(model.py winner defaults)'}")
    if cfg.max_seconds > 0:
        print(f"  guard      = {cfg.max_seconds/3600.0:.2f}h wall-clock "
              f"(final atomic ckpt + exit 0 before the kill)")
    print("=" * 70)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
