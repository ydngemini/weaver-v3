#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kaggle_train.py  —  HEADLESS GPU training kernel for the from-scratch MoE GPT.

THREE PROVEN KAGGLE FACTS THIS SCRIPT IS BUILT AROUND:
    1. Kaggle's torch is 2.10.0+cu128 — it supports sm_70/75/80/86/90 but NOT
       sm_60. A P100 IS sm_60, so it fails with "no kernel image is available".
       FIX: kernel-metadata.json forces a T4 via "machine_shape":"NvidiaTeslaT4"
       and run_kaggle.sh pushes with --accelerator NvidiaTeslaT4. This script
       ALSO guards at runtime: if cuda is unavailable OR the device is a P100, it
       prints a LOUD warning so the log makes the cause obvious.
    2. A SCRIPT kernel runs only its single code_file; sibling .py files are NOT
       importable. So model.py / prepare_data.py / train.py ship as the Kaggle
       DATASET `weaver-moe-code`. Kaggle does NOT reliably mount an attached
       dataset at /kaggle/input/<slug> — it can nest it (e.g. under
       /kaggle/input/datasets/...), which broke the prep kernel twice. So we
       AUTO-DISCOVER each mounted dataset by walking /kaggle/input and matching on
       sentinel files, instead of trusting a hardcoded slug path.
    3. The smoke kernel proved build + autograd + optimizer + output retrieval all
       work on a T4 (Tesla T4, 15.64GB, loss decreasing, SMOKE_OK).

THE CROSS-SESSION RESUME LOOP (the whole point):
    * Data: read-only from the weaver-moe-data mount (train.bin / val.bin /
      meta.json / tokenizer — packed ONCE by the prep kernel). Never re-tokenize.
    * Resume: read the latest atomic checkpoint from the read-only weaver-moe-ckpt
      mount (or start FRESH random-init if empty — the bootstrap session). train.py
      writes the NEW bundle to /kaggle/working/ckpts.
    * Wall-clock guard: train.py runs with --max-hours 11.5 so it writes a final
      atomic checkpoint and exits 0 cleanly BEFORE Kaggle's ~12h hard kill.
    * The launcher pulls /kaggle/working and versions the weaver-moe-ckpt dataset,
      so the next kernel run resumes EXACTLY here.

This is a thin orchestration wrapper: it reports the GPU, guards the device,
auto-discovers the three mounts, then invokes the code dataset's train.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time


# --- Preferred mount paths (used first; auto-discovery is the fallback) ---
CODE_INPUT = os.environ.get("KAGGLE_CODE_INPUT", "/kaggle/input/weaver-moe-code")
DATA_INPUT = os.environ.get("KAGGLE_DATA_INPUT", "/kaggle/input/weaver-moe-data")
CKPT_INPUT = os.environ.get("KAGGLE_CKPT_INPUT", "/kaggle/input/weaver-moe-ckpt")
WORKING = os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")

CKPT_OUT = os.path.join(WORKING, "ckpts")   # writable -> saved as version output

# Wall-clock guard: stay under Kaggle's ~12h hard kill. Override via env.
MAX_HOURS = float(os.environ.get("KAGGLE_MAX_HOURS", "11.5"))  # full session ~11.5h: final atomic ckpt + clean exit before Kaggle's ~12h kill.

# DTYPE / THROUGHPUT (read before changing): we do NOT pass --dtype to train.py.
# train.py's resolve_amp() auto-selects the dtype by compute capability:
#   T4 / Turing (sm_75) -> fp16 + GradScaler  (T4 has fp16 tensor cores but NO bf16)
#   Ampere+ (sm_80+)    -> bf16
# This is the fix for the 3.3k tok/s validation run: PyTorch 2.x is_bf16_supported()
# returns True on T4 via emulation, so the old code ran bf16 with no tensor-core path
# (~3-4x slow). Forcing --dtype fp16 here would also hurt a future A100/L4 upgrade, so
# leave it to the capability gate. fp16 should restore throughput toward the ~19k est.

# T4 VRAM reality: the architect's B=48 estimate OOM'd on a real T4 (14.56 GiB usable).
# Shrink the micro-batch and raise grad-accum to keep the SAME 384-seq effective batch
# (12*32 == 48*8), cutting per-forward activation memory ~4x. Override via env.
MICRO_BATCH = os.environ.get("KAGGLE_MICRO_BATCH", "12")
GRAD_ACCUM = os.environ.get("KAGGLE_GRAD_ACCUM", "32")
# Reclaim the large "reserved but unallocated" pool (fragmentation) seen at OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ===========================================================================
# 0. MOUNT AUTO-DISCOVERY  (Kaggle nests attached datasets unpredictably)
# ===========================================================================
def _input_tree(maxdepth: int = 6) -> str:
    lines = []
    try:
        for dp, dns, fns in os.walk("/kaggle/input"):
            if dp.count("/") <= maxdepth:
                lines.append(f"  {dp}  dirs={sorted(dns)[:8]} files={sorted(fns)[:8]}")
    except Exception as e:
        return f"  <could not walk /kaggle/input: {e}>"
    return "\n".join(lines) or "  <empty>"


def _find_dir_with(explicit: str, needed_files) -> str | None:
    """Return the mount dir that contains ALL of needed_files. Prefer the explicit
    slug path (fast path / local dry-run), else walk /kaggle/input to find it."""
    if os.path.isdir(explicit) and all(
            os.path.exists(os.path.join(explicit, f)) for f in needed_files):
        return explicit
    for dp, _dns, fns in os.walk("/kaggle/input"):
        if all(f in fns for f in needed_files):
            return dp
    return None


def _find_ckpt_dir(explicit: str) -> str | None:
    """Return a mount dir holding a usable .pt checkpoint, or None for fresh init."""
    def has_pt(d: str) -> bool:
        try:
            return any(f.endswith(".pt") and not f.endswith(".tmp")
                       for f in os.listdir(d))
        except Exception:
            return False
    if os.path.isdir(explicit) and has_pt(explicit):
        return explicit
    for dp, _dns, fns in os.walk("/kaggle/input"):
        if any(f.endswith(".pt") and not f.endswith(".tmp") for f in fns):
            return dp
    return None


# ===========================================================================
# 1. GPU REPORT + DEVICE GUARD  (PRINT GPU FIRST so the log confirms a T4)
# ===========================================================================
def report_gpu_and_guard() -> bool:
    """Print the GPU up front (so we can confirm T4 in the kernel log), then guard.
    Returns True if the device looks trainable (CUDA + not a P100), else False —
    the caller prints a loud warning but still proceeds to let train.py report the
    concrete failure (it will fall back to CPU / surface the 'no kernel image')."""
    print("=" * 70)
    print("kaggle_train.py — headless MoE GPT GPU training session")
    print("=" * 70)
    import platform
    print(f"[env] python {platform.python_version()}  cwd={os.getcwd()}")
    print(f"[env] CKPT_OUT={CKPT_OUT}  MAX_HOURS={MAX_HOURS}")

    gpu_name = ""
    # nvidia-smi is the most reliable probe before torch is up.
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                print(f"[gpu] {line.strip()}")
                if not gpu_name:
                    gpu_name = line.split(",")[0].strip()
        else:
            print("[gpu] nvidia-smi returned non-zero (no GPU? check enable_gpu).")
    except Exception as e:
        print(f"[gpu] nvidia-smi unavailable ({e}).")

    # torch-level probe: confirm CUDA + capability (P100 == sm_60 == cu128 fails).
    cuda_ok = False
    is_p100 = False
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        print(f"[gpu] torch {torch.__version__}  cuda_available={cuda_ok}")
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(f"[gpu] device 0: {name}  compute_capability=sm_{cap[0]}{cap[1]}")
            gpu_name = gpu_name or name
            is_p100 = ("P100" in name) or (cap == (6, 0))
    except Exception as e:
        print(f"[gpu] torch CUDA probe failed ({e}).")

    if not cuda_ok:
        print("!" * 70)
        print("[guard] WARNING: CUDA is NOT available. This kernel was meant to run "
              "on a T4 GPU. Confirm enable_gpu=true AND machine_shape=NvidiaTeslaT4 "
              "in kernel-metadata.json, and push with --accelerator NvidiaTeslaT4.")
        print("!" * 70)
        return False
    if is_p100:
        print("!" * 70)
        print("[guard] WARNING: detected a Tesla P100 (sm_60). Kaggle's torch "
              "2.10.0+cu128 has NO sm_60 kernel image -> training WILL fail with "
              "'no kernel image is available'. FORCE a T4: set machine_shape="
              "NvidiaTeslaT4 in kernel-metadata.json and push with "
              "--accelerator NvidiaTeslaT4. Re-push to get a T4.")
        print("!" * 70)
        return False

    print(f"[guard] OK: trainable GPU detected ({gpu_name or 'CUDA device'}).")
    return True


# ===========================================================================
# 2. DEPS  (torch + numpy preinstalled on Kaggle GPU image; do NOT touch torch)
# ===========================================================================
def ensure_deps() -> None:
    try:
        import torch  # noqa: F401
        import numpy  # noqa: F401
        print(f"[deps] torch {torch.__version__}  numpy ok (preinstalled — not "
              f"touching torch; reinstalling risks the CUDA build).")
    except Exception as e:
        print(f"[deps] WARNING: torch/numpy import failed ({e}). Kaggle's GPU image "
              f"should provide them; not attempting a torch reinstall.")


# ===========================================================================
# 3. RUN TRAIN  (subprocess the code dataset's train.py, --storage kaggle)
# ===========================================================================
def run_train(code_dir: str, data_dir: str, resume_dir: str) -> int:
    os.makedirs(CKPT_OUT, exist_ok=True)
    train_py = os.path.join(code_dir, "train.py")

    cmd = [
        sys.executable, train_py,
        "--storage", "kaggle",
        "--data-dir", data_dir,            # READ packed data (read-only mount)
        "--ckpt-dir", CKPT_OUT,            # WRITE new bundle into the version output
        "--resume-dir", resume_dir,        # READ previous bundle (may be empty -> fresh)
        "--max-hours", str(MAX_HOURS),     # wall-clock guard: final ckpt + exit 0 < 12h
        "--micro-batch-size", MICRO_BATCH, # T4-safe per-forward batch (was 48 -> OOM)
        "--grad-accum-steps", GRAD_ACCUM,  # keep 384-seq effective batch (12*32 == 48*8)
    ]
    # Optional architecture config passthrough (if shipped in the code dataset).
    cfg = os.path.join(code_dir, "winner_config.json")
    if os.path.exists(cfg):
        cmd += ["--config", cfg]

    print("=" * 70)
    print(f"[train] launching: {' '.join(cmd)}")
    print("=" * 70)
    t0 = time.time()
    # cwd = code_dir so train.py's `from model import ...` resolves to the sibling
    # in the code dataset; inherit stdio so Kaggle captures the full feed.
    rc = subprocess.run(cmd, cwd=code_dir).returncode
    print(f"[train] train.py exited rc={rc} after {(time.time()-t0)/3600.0:.2f}h.")
    return rc


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> int:
    trainable = report_gpu_and_guard()
    if not trainable:
        print("[main] proceeding despite the guard warning so train.py surfaces the "
              "concrete failure in the log (then fix machine_shape and re-push).")
    ensure_deps()

    # --- auto-discover the three mounts (Kaggle nests them under /kaggle/input) ---
    code_dir = _find_dir_with(CODE_INPUT, ["train.py", "model.py"])
    if code_dir is None:
        print("[FATAL] code dataset (weaver-moe-code: train.py + model.py) not found "
              "anywhere under /kaggle/input. Attach 'nathaniellockwood/weaver-moe-code' "
              "as a dataset_source and re-push. Tree:\n" + _input_tree())
        return 2
    print(f"[code] located code dataset at: {code_dir}")
    sys.path.insert(0, code_dir)   # Kaggle script-kernel fix: code is a DATASET

    data_dir = _find_dir_with(DATA_INPUT, ["train.bin"])
    if data_dir is None:
        print("[FATAL] data dataset (weaver-moe-data: train.bin) not found under "
              "/kaggle/input. Run the prep kernel first and attach "
              "'nathaniellockwood/weaver-moe-data' as a dataset_source. Tree:\n"
              + _input_tree())
        return 2
    print(f"[data] located data dataset at: {data_dir}")

    resume_dir = _find_ckpt_dir(CKPT_INPUT)
    if resume_dir:
        n = sum(1 for f in os.listdir(resume_dir)
                if f.endswith(".pt") and not f.endswith(".tmp"))
        print(f"[resume] found {n} checkpoint bundle(s) in {resume_dir}: resuming.")
    else:
        resume_dir = CKPT_INPUT  # non-existent path -> train.py does a fresh init
        print("[resume] no .pt checkpoint mounted -> FRESH random-init (bootstrap "
              "session). train.py tolerates the absent resume dir.")

    rc = run_train(code_dir, data_dir, resume_dir)

    if rc == 0:
        print("=" * 70)
        print("[done] session complete. /kaggle/working/ckpts holds the latest atomic "
              "checkpoint — version the weaver-moe-ckpt dataset from it and re-run to "
              "resume exactly here.")
        print("=" * 70)
    else:
        print(f"[done] session ended rc={rc}. The previous ckpt dataset version is "
              f"untouched and still resumable; inspect the log above.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
