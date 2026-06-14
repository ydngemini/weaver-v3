#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py  —  Data preparation stage for a from-scratch MoE GPT pretrain.

WINNER CONFIG (mem obs #1848, "B-core hardened with A+C grafts"):
    9 layers x 640d, head_dim=64, 8 experts top-2, d_ff=1920, ctx=1024,
    vocab=16384 byte-level BPE, ~202.9M total / ~69.5M active, ~6.4GB VRAM on a T4.

WHAT THIS FILE DOES (and ONLY this file — it does NOT train the model):
    1. Streams the corpus shards from HuggingFace with streaming=True (never
       materializes the multi-TB sources to disk).
    2. Trains (or loads) a 16,384-vocab byte-level BPE tokenizer with a GPT-2-style
       regex pretokenizer on a *sampled* slice of the real corpus.
    3. Tokenizes every document, appends an EOS/document separator, and PACKS the
       contiguous token stream into ctx-sized windows written to two uint16
       memmaps: train.bin and val.bin, honoring the configured mix weights.
    4. Maintains a RESUMABLE shard cursor (per-stream, atomic .tmp->rename) so a
       re-run after a Colab disconnect CONTINUES where it stopped instead of
       restarting — and never repeats or skips a shard across the ~9 reconnects.

Design constraints that drive every decision here:
    * Target HW = ONE free Colab T4 (16GB), ~12h cap, multi-session. So:
        - Streaming only. No local copy of The Stack v2 / codeparrot / StarCoder.
        - The byte-level BPE *must* be trained on the real distribution (code-heavy)
          BEFORE the run so the small 16k vocab fits — that tight vocab is what
          keeps the (B,T,16384) logits activation small and buys the T4 headroom.
        - The shard cursor is PER-STREAM (lang/dataset, sample index, byte progress)
          NOT a flat global index, because interleave order must be reproduced
          exactly for the model-training stage's "fast-forward by total_tokens"
          resume to land on the same data.
    * RANDOM INIT downstream: nothing here loads pretrained MODEL weights. (The
      BPE tokenizer is *fit on our corpus*, not a borrowed pretrained vocab.)

Single-file-runnable:
    python prepare_data.py --train-tokenizer        # fit the 16k BPE once
    python prepare_data.py --build                  # pack train.bin / val.bin
    python prepare_data.py                           # do both (tokenizer then build)
Resume is automatic: re-running --build continues from the saved cursor.

Outputs (all under DATA_DIR, default ./data, override with --data-dir or
PRETRAIN_DATA_DIR, point it at a Google Drive mount for multi-session persistence):
    data/tokenizer/  (tokenizer.json + meta.json)   — the trained BPE
    data/train.bin   (uint16 memmap of packed token ids)
    data/val.bin     (uint16 memmap of packed token ids, ctx-aligned holdout)
    data/cursor.json (atomic per-stream resume cursor + running token counts)
    data/meta.json   (vocab_size, ctx_len, dtype, counts — read by train.py)

8B-ACTIVE SCALE-UP PATH (documented, NOT targeted): the same pipeline scales by
CONFIG ONLY — bump vocab to ~64k, ctx to 4096+, and point at the *full* (non-sample)
splits. That regime (~8B active / ~47B total Mixtral-class) cannot train on one
16GB T4: all 8 expert weights + fp32 AdamW state stop being resident on a single
device. It needs multi-GPU expert-parallel + tensor-parallel sharding
(Megatron / DeepSpeed-MoE), ZeRO-3, and a token budget in the trillions. The DATA
stage here is unchanged; only the parallelism strategy and the config change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Lazy / optional imports. We import the heavy deps inside functions so that
# `python prepare_data.py --help` works in a bare environment and so the file
# stays importable as a module by the training script.
# ---------------------------------------------------------------------------


# ===========================================================================
# 1. CONFIG
#    Prefer a sibling config.py (shared with model/train); otherwise fall back
#    to the inline WINNER defaults so this file is genuinely single-file-runnable.
# ===========================================================================

@dataclass
class ShardSpec:
    """One corpus stream. mix_weight is a *relative* probability; the packer
    samples shards in proportion to these weights so the final .bin matches the
    intended corpus mix without materializing per-shard buffers."""
    name: str
    hf_path: str
    # datasets.load_dataset positional config / name (e.g. "sample-10BT", a lang).
    hf_config: Optional[str] = None
    # data_dir for repos that split subsets by directory (StarCoderData).
    data_dir: Optional[str] = None
    split: str = "train"
    # which row field holds the text; tried in order, first non-empty wins.
    text_fields: Tuple[str, ...] = ("content", "text", "body")
    mix_weight: float = 1.0
    # streaming shuffle buffer (0 = no shuffle; keeps order deterministic+cheap).
    shuffle_buffer: int = 10_000
    seed: int = 42
    # optional row predicate name -> resolved to a callable in _build_filter().
    row_filter: Optional[str] = None
    # pin a revision for reproducibility across the multi-session run.
    revision: Optional[str] = None
    # The Stack v2 rows are POINTERS (blob_id), not code: fetch from SH S3.
    needs_sh_s3: bool = False
    gated: bool = False
    notes: str = ""


@dataclass
class DataConfig:
    # --- tokenizer / packing geometry (from the winner config) ---
    vocab_size: int = 16384
    ctx_len: int = 1024
    # dtype for the packed .bin. uint16 holds vocab up to 65535 -> 16k fits.
    bin_dtype: str = "uint16"

    # --- token budget (the 6B-token plan) ---
    token_budget_b: float = 6.0           # total tokens to pack across all sessions
    val_tokens_m: float = 10.0            # held-out validation tokens (memmap val.bin)

    # --- special tokens for packing ---
    eos_token: str = "<|endoftext|>"      # document separator, GPT-2 style

    # --- tokenizer training sample budget (NOT the full corpus) ---
    tokenizer_train_docs: int = 400_000   # docs sampled to fit the 16k BPE
    tokenizer_train_max_chars: int = 600_000_000  # ~600MB cap on the sample

    # --- io / resume ---
    data_dir: str = field(default_factory=lambda: os.environ.get(
        "PRETRAIN_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")))
    cursor_flush_every_docs: int = 2000   # atomically persist cursor this often
    bin_flush_every_tokens: int = 5_000_000

    # --- wall-clock guard (multi-session prep) ---
    # Kaggle CPU kernels are HARD-KILLED at ~12h and a timed-out run is marked FAILED
    # -> its /kaggle/working output is NOT committed, so you'd lose the whole session.
    # Set this below the platform cap so build() breaks the loop, flushes atomically,
    # writes meta, and exits 0 -> the partial (resumable) output IS saved. The cursor
    # already flushes every 2000 docs, so at most a few MB is re-streamed next session.
    # None = unbounded (local / Colab, where the runtime persists between sessions).
    max_seconds: Optional[float] = None

    # --- the corpus mix (weights normalized at runtime) ---
    shards: List[ShardSpec] = field(default_factory=lambda: [
        ShardSpec(
            name="the-stack-v2-dedup",
            hf_path="bigcode/the-stack-v2-dedup",
            # per-language configs interleaved; we expand these at load time.
            hf_config=None,
            text_fields=("content",),
            mix_weight=0.78,
            row_filter="permissive_licenses",
            needs_sh_s3=True,
            notes=("rows are blob_id POINTERS; content fetched from Software "
                   "Heritage S3 (needs AWS creds + SH/INRIA agreement). Falls "
                   "back/skips cleanly if creds absent."),
        ),
        ShardSpec(
            name="codeparrot-clean",
            hf_path="codeparrot/codeparrot-clean",
            text_fields=("content",),
            mix_weight=0.62,
            row_filter="drop_gpl",
            notes="Python code spine; ~14.7B raw tok available, ~0.25 epoch at 0.62.",
        ),
        ShardSpec(
            name="starcoderdata",
            hf_path="bigcode/starcoderdata",
            # expanded into per-data_dir streams at load time.
            data_dir=None,
            text_fields=("content", "text", "body"),
            mix_weight=0.18,
            gated=True,
            notes="gated; run `huggingface-cli login` with accepted ToS token.",
        ),
        ShardSpec(
            name="fineweb-edu",
            hf_path="HuggingFaceFW/fineweb-edu",
            hf_config="sample-10BT",
            text_fields=("text",),
            mix_weight=0.12,
            row_filter="edu_score_ge_3",
            notes="thin NL slice (ODC-BY); the 'not pure syntax' explanatory prose.",
        ),
    ])

    # The Stack v2: which language configs to interleave.
    stack_v2_langs: Tuple[str, ...] = (
        "Python", "JavaScript", "TypeScript", "Go", "Rust", "C", "C++", "Java",
    )
    # StarCoderData: code data_dirs + a thin NL tail.
    starcoder_code_dirs: Tuple[str, ...] = (
        "python", "javascript", "java", "go", "c", "cpp", "rust",
        "typescript", "html", "shell",
    )
    starcoder_nl_dirs: Tuple[str, ...] = ("markdown", "github-issues-filtered-structured")

    def config_hash(self) -> str:
        """Stable hash of the CORPUS-AFFECTING data config — written into meta +
        checkpoint bundle so train.py can refuse a mismatched .bin / cursor.

        CRITICAL: deliberately EXCLUDES io/location fields (`data_dir` and the
        flush-cadence knobs). Multi-session resume on Colab routinely moves the
        output between local scratch and a Google Drive mount; if `data_dir` were
        hashed, every relocation would invalidate the resume cursor and silently
        restart the 6B-token build. Only fields that change WHICH TOKENS land in
        the .bin (vocab, ctx, mix, shard set, budget, special tokens) belong here."""
        d = asdict(self)
        for io_field in ("data_dir", "cursor_flush_every_docs",
                         "bin_flush_every_tokens", "max_seconds"):
            d.pop(io_field, None)
        payload = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def load_config() -> DataConfig:
    """Use a sibling config.py if present (so model + data + train agree),
    else the inline winner defaults."""
    try:
        import config as _sibling  # type: ignore
        if hasattr(_sibling, "DATA_CONFIG"):
            cfg = _sibling.DATA_CONFIG
            if isinstance(cfg, DataConfig):
                return cfg
        if hasattr(_sibling, "get_data_config"):
            return _sibling.get_data_config()
    except Exception:
        pass
    return DataConfig()


# ===========================================================================
# 1b. CORPUS PRESETS
#    A preset swaps the shard list (and any geometry it needs) WITHOUT touching
#    the rest of the config. The default ("winner") keeps the inline shards.
#    "kaggle" is the HEADLESS-SAFE corpus: every source is DIRECTLY streamable
#    with only `enable_internet=true` — no AWS creds, no Software Heritage S3
#    agreement, no gated ToU token. It is code-heavy with a thin NL slice.
#
#    WHY this differs from the winner corpus: The-Stack-v2 rows are blob_id
#    POINTERS (content needs AWS + a Software Heritage S3 agreement) and
#    starcoderdata is GATED (needs an accepted-ToU HF token) — neither runs on a
#    headless Kaggle kernel. The kaggle preset drops both and packs from sources
#    that download with no auth at all.
# ===========================================================================

def _kaggle_shards() -> List[ShardSpec]:
    """Directly-streamable, non-gated corpus for a headless Kaggle kernel.

    Sources (all load with only enable_internet=true — no AWS / SH-S3 / gated ToU):
      * codeparrot/codeparrot-clean   — Python code spine (PRIMARY code source).
      * bigcode/the-stack-smol         — small permissive multi-language code sample
                                          (SECONDARY code; optional, skips cleanly).
      * HuggingFaceFW/fineweb-edu      — sample-10BT streaming NL slice (thin tail).

    Any source that 404s / needs auth is skipped with a warning by iter_shard_text;
    it never hard-fails the prep.
    """
    return [
        ShardSpec(
            name="codeparrot-clean",
            hf_path="codeparrot/codeparrot-clean",
            text_fields=("content",),
            mix_weight=0.62,
            row_filter="drop_gpl",
            notes="PRIMARY code (Python). Directly streamable, no auth.",
        ),
        ShardSpec(
            name="the-stack-smol",
            hf_path="bigcode/the-stack-smol",
            # the-stack-smol ships per-language configs under data/<lang>/ ; the
            # default config streams the full small permissive multi-language sample.
            text_fields=("content",),
            mix_weight=0.20,
            notes=("SECONDARY code (multi-language, permissive small sample). "
                   "Directly downloadable, no SH-S3 pointers. Optional — skips "
                   "cleanly if the config 404s."),
        ),
        ShardSpec(
            name="fineweb-edu",
            hf_path="HuggingFaceFW/fineweb-edu",
            hf_config="sample-10BT",
            text_fields=("text",),
            mix_weight=0.12,
            row_filter="edu_score_ge_3",
            notes="thin NL slice (ODC-BY); explanatory prose, streamed.",
        ),
    ]


# Registry of named corpus presets. The default is the inline winner shards.
CORPUS_PRESETS = {
    "kaggle": _kaggle_shards,
}


def apply_corpus_preset(cfg: DataConfig, preset: Optional[str]) -> DataConfig:
    """Swap cfg.shards for a named preset's shard list. `None` or 'winner'/'default'
    leaves the inline winner shards intact (never breaks existing presets)."""
    if not preset or preset in ("winner", "default"):
        return cfg
    builder = CORPUS_PRESETS.get(preset)
    if builder is None:
        valid = ", ".join(["winner"] + sorted(CORPUS_PRESETS))
        raise SystemExit(
            f"[corpus] unknown preset '{preset}'. Valid: {valid}.")
    cfg.shards = builder()
    print(f"[corpus] preset '{preset}': "
          f"{', '.join(s.name for s in cfg.shards)}")
    return cfg


# ===========================================================================
# 2. PERMISSIVE-LICENSE / ROW FILTERS
# ===========================================================================

PERMISSIVE_LICENSES = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC",
    "Unlicense", "MPL-2.0",
}


def _build_filter(name: Optional[str]):
    """Resolve a named row predicate to a callable (or None for pass-through)."""
    if not name:
        return None

    if name == "permissive_licenses":
        def _f(row: Dict[str, Any]) -> bool:
            lic = row.get("detected_licenses") or []
            return len(lic) > 0 and all(l in PERMISSIVE_LICENSES for l in lic)
        return _f

    if name == "drop_gpl":
        def _f(row: Dict[str, Any]) -> bool:
            lic = (row.get("license") or "")
            return not str(lic).lower().startswith("gpl")
        return _f

    if name == "edu_score_ge_3":
        def _f(row: Dict[str, Any]) -> bool:
            return int(row.get("int_score", 3) or 3) >= 3
        return _f

    # Unknown filter name: be permissive (don't silently drop everything).
    return None


# ===========================================================================
# 3. STREAMING SHARD READERS  (HuggingFace datasets, streaming=True)
#    Each reader yields plain text strings. A *cursor* of already-consumed
#    document counts lets us fast-forward (.skip) on resume.
# ===========================================================================

def _first_text(row: Dict[str, Any], fields: Tuple[str, ...]) -> str:
    for f in fields:
        v = row.get(f)
        if v:
            return v if isinstance(v, str) else str(v)
    return ""


def _maybe_sh_s3_client():
    """Build a Software Heritage S3 client for The Stack v2 content fetch.
    Returns None if creds / libs are missing — caller then skips that shard
    rather than crashing a 12h session."""
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        return None
    try:
        import boto3  # type: ignore
        session = boto3.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        return session.client("s3")
    except Exception as e:  # pragma: no cover - env dependent
        print(f"[stack-v2] boto3/S3 unavailable ({e}); skipping Stack v2.", file=sys.stderr)
        return None


def _fetch_stack_v2_content(row: Dict[str, Any], s3) -> str:
    """The Stack v2 rows are pointers: pull the gzip blob from SH S3."""
    try:
        from smart_open import open as s_open  # type: ignore
    except Exception:
        return ""
    blob_id = row.get("blob_id")
    if not blob_id or s3 is None:
        return ""
    url = f"s3://softwareheritage/content/{blob_id}"
    try:
        with s_open(url, "rb", compression=".gz", transport_params={"client": s3}) as f:
            enc = row.get("src_encoding") or "utf-8"
            return f.read().decode(enc, errors="replace")
    except Exception:
        return ""  # survive non-UTF8 / missing blobs without killing the stream


def _expand_streams(cfg: DataConfig, shard: ShardSpec) -> List[Tuple[str, Dict[str, Any]]]:
    """Expand a logical shard into one-or-more concrete (sub_id, load_kwargs)
    sub-streams. Sub-streams that need separate cursors (per-language Stack v2,
    per-data_dir StarCoder) are returned individually so the cursor is per-stream."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    common = {"path": shard.hf_path, "split": shard.split, "streaming": True}
    if shard.revision:
        common["revision"] = shard.revision

    if shard.name == "the-stack-v2-dedup":
        for lang in cfg.stack_v2_langs:
            out.append((f"{shard.name}:{lang}", {**common, "name": lang}))
        return out

    if shard.name == "starcoderdata":
        for d in cfg.starcoder_code_dirs + cfg.starcoder_nl_dirs:
            out.append((f"{shard.name}:{d}", {**common, "data_dir": d}))
        return out

    # Single concrete stream.
    kw = dict(common)
    if shard.hf_config:
        kw["name"] = shard.hf_config
    if shard.data_dir:
        kw["data_dir"] = shard.data_dir
    out.append((shard.name, kw))
    return out


def iter_shard_text(
    cfg: DataConfig,
    shard: ShardSpec,
    sub_id: str,
    load_kwargs: Dict[str, Any],
    skip_docs: int = 0,
) -> Iterator[str]:
    """Yield text documents for one concrete sub-stream, fast-forwarding by
    `skip_docs` (resume). Yields ("" -> skipped) are filtered out by caller."""
    from datasets import load_dataset  # local import (heavy)

    try:
        ds = load_dataset(**load_kwargs)
    except Exception as e:
        msg = str(e)
        if shard.gated and ("401" in msg or "gated" in msg.lower() or "authentication" in msg.lower()):
            print(f"[{sub_id}] GATED — `huggingface-cli login` with an accepted-ToS "
                  f"token, then re-run. Skipping for now.", file=sys.stderr)
        else:
            print(f"[{sub_id}] load failed ({e}); skipping shard.", file=sys.stderr)
        return

    row_filter = _build_filter(shard.row_filter)
    if shard.shuffle_buffer and shard.shuffle_buffer > 0:
        try:
            ds = ds.shuffle(seed=shard.seed, buffer_size=shard.shuffle_buffer)
        except Exception:
            pass
    if row_filter is not None:
        try:
            ds = ds.filter(row_filter)
        except Exception:
            # Some streaming sets dislike .filter; fall back to inline filtering.
            row_filter_inline = row_filter
            row_filter = None
        else:
            row_filter_inline = None
    else:
        row_filter_inline = None

    # Resume: skip already-consumed docs for THIS sub-stream.
    if skip_docs > 0:
        try:
            ds = ds.skip(skip_docs)
        except Exception:
            # If .skip unsupported, fall back to manual skipping below.
            _manual_skip = skip_docs
        else:
            _manual_skip = 0
    else:
        _manual_skip = 0

    s3 = _maybe_sh_s3_client() if shard.needs_sh_s3 else None
    if shard.needs_sh_s3 and s3 is None:
        print(f"[{sub_id}] Software Heritage S3 creds/libs missing — these rows are "
              f"blob_id POINTERS, not code. Skipping Stack v2 sub-stream "
              f"(set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY + SH agreement).",
              file=sys.stderr)
        return

    seen = 0
    for row in ds:
        if _manual_skip > 0:
            _manual_skip -= 1
            continue
        if row_filter_inline is not None and not row_filter_inline(row):
            continue
        if shard.needs_sh_s3:
            text = _fetch_stack_v2_content(row, s3)
        else:
            text = _first_text(row, shard.text_fields)
        seen += 1
        if text:
            yield text


# ===========================================================================
# 4. TOKENIZER  —  train OR load a 16k byte-level BPE
#    GPT-2-style regex pretokenizer, ByteLevel, trained on a sampled slice of
#    the REAL corpus so the small vocab fits the code+text distribution.
# ===========================================================================

def tokenizer_paths(cfg: DataConfig) -> Tuple[str, str]:
    tdir = os.path.join(cfg.data_dir, "tokenizer")
    return os.path.join(tdir, "tokenizer.json"), os.path.join(tdir, "meta.json")


def _gpt2_split_regex() -> str:
    # The GPT-2 pretokenization regex (contractions, words, numbers, punct, ws).
    return (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )


def train_tokenizer(cfg: DataConfig) -> str:
    """Fit a byte-level BPE tokenizer (vocab=cfg.vocab_size) on a sampled,
    mix-weighted slice of the streamed corpus. Idempotent: returns existing
    tokenizer if already trained for this config geometry."""
    tok_json, tok_meta = tokenizer_paths(cfg)
    if os.path.exists(tok_json) and os.path.exists(tok_meta):
        try:
            meta = json.load(open(tok_meta))
            if meta.get("vocab_size") == cfg.vocab_size:
                print(f"[tokenizer] reuse existing {tok_json} (vocab={cfg.vocab_size})")
                return tok_json
        except Exception:
            pass

    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, Regex  # type: ignore
    from tokenizers.pre_tokenizers import ByteLevel, Split, Sequence  # type: ignore

    os.makedirs(os.path.dirname(tok_json), exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = Sequence([
        Split(pattern=Regex(_gpt2_split_regex()), behavior="isolated"),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()

    special = [cfg.eos_token]
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=special,
        initial_alphabet=ByteLevel.alphabet(),  # full 256-byte alphabet -> nothing OOV
        show_progress=True,
        min_frequency=2,
    )

    print(f"[tokenizer] training {cfg.vocab_size}-vocab byte-level BPE on a sampled "
          f"corpus slice (<= {cfg.tokenizer_train_docs} docs, "
          f"<= {cfg.tokenizer_train_max_chars/1e6:.0f}M chars)...")

    def _sample_iter() -> Iterator[str]:
        # Mix-weighted round-robin sample across shards so the BPE sees the real
        # code-heavy distribution (and the thin NL tail). Bounded by doc + char caps.
        weights = _normalized_weights(cfg)
        gens = _open_all_substreams(cfg, cursor=None)  # fresh, no resume for sampling
        if not gens:
            raise RuntimeError(
                "No streamable shards available for tokenizer training. Check "
                "network / HF auth / AWS creds, or pass --shards to subset.")
        rng = random.Random(1234)
        ids = [g["weight_key"] for g in gens]
        wts = [weights[g["weight_key"]] for g in gens]
        n_docs = 0
        n_chars = 0
        # Round-robin biased by weight: pick a shard, pull one doc.
        alive = list(range(len(gens)))
        while alive and n_docs < cfg.tokenizer_train_docs and n_chars < cfg.tokenizer_train_max_chars:
            live_w = [wts[i] for i in alive]
            choice = rng.choices(alive, weights=live_w, k=1)[0]
            try:
                doc = next(gens[choice]["it"])
            except StopIteration:
                alive.remove(choice)
                continue
            if not doc:
                continue
            doc = doc[:200_000]  # cap any single pathological doc
            n_docs += 1
            n_chars += len(doc)
            if n_docs % 5000 == 0:
                print(f"  [tokenizer] sampled {n_docs} docs, {n_chars/1e6:.0f}M chars")
            yield doc

    tokenizer.train_from_iterator(_sample_iter(), trainer=trainer)
    tokenizer.save(tok_json)

    eos_id = tokenizer.token_to_id(cfg.eos_token)
    meta = {
        "vocab_size": cfg.vocab_size,
        "real_vocab_size": tokenizer.get_vocab_size(),
        "eos_token": cfg.eos_token,
        "eos_id": eos_id,
        "ctx_len": cfg.ctx_len,
        "pretokenizer": "gpt2-regex+bytelevel",
        "config_hash": cfg.config_hash(),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json.dump(meta, open(tok_meta, "w"), indent=2)
    print(f"[tokenizer] saved {tok_json} (real_vocab={tokenizer.get_vocab_size()}, "
          f"eos_id={eos_id})")
    return tok_json


def load_tokenizer(cfg: DataConfig):
    from tokenizers import Tokenizer  # type: ignore
    tok_json, tok_meta = tokenizer_paths(cfg)
    if not os.path.exists(tok_json):
        raise FileNotFoundError(
            f"No tokenizer at {tok_json}. Run `python prepare_data.py --train-tokenizer` first.")
    tok = Tokenizer.from_file(tok_json)
    meta = json.load(open(tok_meta)) if os.path.exists(tok_meta) else {}
    eos_id = meta.get("eos_id")
    if eos_id is None:
        eos_id = tok.token_to_id(cfg.eos_token)
    return tok, eos_id


# ===========================================================================
# 5. MIX WEIGHTS + STREAM ORCHESTRATION
# ===========================================================================

def _normalized_weights(cfg: DataConfig) -> Dict[str, float]:
    raw = {s.name: max(0.0, float(s.mix_weight)) for s in cfg.shards}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def _open_all_substreams(cfg: DataConfig, cursor: Optional["Cursor"]) -> List[Dict[str, Any]]:
    """Open every concrete sub-stream as a Python generator, fast-forwarding by
    the cursor's per-substream doc count. Returns list of dicts with the live
    iterator, the sub_id (cursor key), and the logical shard's weight key."""
    out: List[Dict[str, Any]] = []
    for shard in cfg.shards:
        for sub_id, load_kwargs in _expand_streams(cfg, shard):
            skip = cursor.docs_consumed.get(sub_id, 0) if cursor is not None else 0
            it = iter_shard_text(cfg, shard, sub_id, load_kwargs, skip_docs=skip)
            out.append({
                "sub_id": sub_id,
                "weight_key": shard.name,   # mix weight applies per logical shard
                "it": _counting_iter(it, sub_id, cursor),
                "shard": shard,
            })
    return out


def _counting_iter(it: Iterator[str], sub_id: str, cursor: Optional["Cursor"]) -> Iterator[str]:
    """Wrap a text iterator so each yielded doc bumps the cursor's per-substream
    counter (used for resume fast-forward)."""
    for doc in it:
        if cursor is not None:
            cursor.docs_consumed[sub_id] = cursor.docs_consumed.get(sub_id, 0) + 1
        yield doc


# ===========================================================================
# 6. RESUMABLE CURSOR  (atomic .tmp -> os.rename)
# ===========================================================================

@dataclass
class Cursor:
    config_hash: str
    # per-substream documents consumed (the resume fast-forward key).
    docs_consumed: Dict[str, int] = field(default_factory=dict)
    # running output position in train.bin / val.bin (token offsets).
    train_tokens: int = 0
    val_tokens: int = 0
    total_docs: int = 0
    rng_state: Optional[int] = None       # python RNG seed-state proxy
    finished: bool = False
    updated_at: str = ""

    @staticmethod
    def path(cfg: DataConfig) -> str:
        return os.path.join(cfg.data_dir, "cursor.json")

    @classmethod
    def load_or_new(cls, cfg: DataConfig) -> "Cursor":
        p = cls.path(cfg)
        chash = cfg.config_hash()
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                if d.get("config_hash") == chash:
                    c = cls(config_hash=chash)
                    c.__dict__.update(d)
                    c.config_hash = chash
                    print(f"[cursor] RESUMING: {c.total_docs} docs, "
                          f"{c.train_tokens/1e6:.1f}M train tok packed so far.")
                    return c
                else:
                    print("[cursor] config hash changed -> starting fresh "
                          "(old cursor incompatible).", file=sys.stderr)
            except Exception as e:
                print(f"[cursor] unreadable ({e}) -> starting fresh.", file=sys.stderr)
        return cls(config_hash=chash)

    def save_atomic(self, cfg: DataConfig) -> None:
        p = self.path(cfg)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)  # atomic on POSIX; survives a mid-write Colab kill


# ===========================================================================
# 7. APPENDING UINT16 MEMMAP WRITERS
#    np.memmap can't grow, so we append to a raw .bin via a buffered file
#    handle and keep an exact token count in the cursor. The training loader
#    re-opens these as np.memmap(dtype=uint16) and reshapes by ctx_len.
# ===========================================================================

class BinWriter:
    """Append-only uint16 token writer with periodic flush. Resumes by trusting
    the cursor's token count and truncating any partial tail beyond it (so a
    crash mid-flush never corrupts ctx alignment)."""

    def __init__(self, path: str, dtype: str, resume_tokens: int):
        self.path = path
        self.np_dtype = np.dtype(dtype)
        self.itemsize = self.np_dtype.itemsize
        os.makedirs(os.path.dirname(path), exist_ok=True)
        valid_bytes = resume_tokens * self.itemsize
        if os.path.exists(path):
            cur = os.path.getsize(path)
            if cur > valid_bytes:
                # truncate stale tail written after the last cursor flush.
                with open(path, "r+b") as f:
                    f.truncate(valid_bytes)
            elif cur < valid_bytes:
                # cursor claims more than file has -> trust the file (rare).
                resume_tokens = cur // self.itemsize
        else:
            open(path, "wb").close()
            resume_tokens = 0
        self.tokens = resume_tokens
        self._fh = open(path, "ab", buffering=1024 * 1024)

    def write(self, ids: np.ndarray) -> int:
        if ids.dtype != self.np_dtype:
            ids = ids.astype(self.np_dtype, copy=False)
        self._fh.write(ids.tobytes())
        self.tokens += int(ids.size)
        return int(ids.size)

    def flush(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._fh.close()


# ===========================================================================
# 8. THE PACKER  —  tokenize, EOS-delimit, pack contiguous ctx windows
# ===========================================================================

def build(cfg: DataConfig) -> None:
    """Stream -> tokenize -> EOS-delimit -> pack into train.bin / val.bin until
    the token budget is hit, honoring mix weights and resuming from the cursor.

    Packing model (nanoGPT-style): we maintain a flat token stream; documents are
    separated by EOS; the stream is sliced into ctx_len windows downstream. We do
    NOT pad — packing is dense, which is the correct, FLOP-efficient layout. We
    only ensure the val.bin holdout is filled FIRST (a fixed early budget) so it
    is stable across re-runs, then everything else goes to train.bin."""
    tok, eos_id = load_tokenizer(cfg)
    if eos_id is None:
        raise RuntimeError("Tokenizer has no EOS id; retrain with --train-tokenizer.")

    cursor = Cursor.load_or_new(cfg)
    if cursor.finished:
        print("[build] cursor marks build FINISHED — nothing to do. "
              "Delete data/cursor.json to rebuild.")
        return

    # Restore python RNG so the weighted shard sampling reproduces on resume.
    rng = random.Random()
    if cursor.rng_state is not None:
        rng.seed(cursor.rng_state)
    else:
        cursor.rng_state = 1234
        rng.seed(cursor.rng_state)

    weights = _normalized_weights(cfg)
    print("[build] normalized mix weights:")
    for k, v in weights.items():
        print(f"    {k:24s} {v:0.3f}")

    budget_tokens = int(cfg.token_budget_b * 1e9)
    val_target = int(cfg.val_tokens_m * 1e6)

    train_bin = os.path.join(cfg.data_dir, "train.bin")
    val_bin = os.path.join(cfg.data_dir, "val.bin")
    train_w = BinWriter(train_bin, cfg.bin_dtype, resume_tokens=cursor.train_tokens)
    val_w = BinWriter(val_bin, cfg.bin_dtype, resume_tokens=cursor.val_tokens)
    # keep cursor consistent with what's actually on disk after truncation.
    cursor.train_tokens = train_w.tokens
    cursor.val_tokens = val_w.tokens

    gens = _open_all_substreams(cfg, cursor)
    if not gens:
        print("[build] No streamable shards available — check HF auth / AWS creds "
              "/ network. Nothing written.", file=sys.stderr)
        return

    alive = list(range(len(gens)))
    docs_since_flush = 0
    tokens_since_flush = 0
    t0 = time.time()
    last_report = t0

    eos_arr = np.array([eos_id], dtype=np.dtype(cfg.bin_dtype))

    def _encode(doc: str) -> np.ndarray:
        ids = tok.encode(doc).ids
        arr = np.fromiter(ids, dtype=np.dtype(cfg.bin_dtype), count=len(ids))
        # append EOS document separator (nanoGPT style)
        return np.concatenate([arr, eos_arr])

    try:
        while alive and cursor.train_tokens < budget_tokens:
            # weighted pick among still-live sub-streams (by logical shard weight)
            live_w = [weights.get(gens[i]["weight_key"], 0.0) for i in alive]
            if sum(live_w) <= 0:
                live_w = [1.0] * len(alive)
            idx = rng.choices(alive, weights=live_w, k=1)[0]
            try:
                doc = next(gens[idx]["it"])
            except StopIteration:
                alive.remove(idx)
                continue
            except Exception as e:
                # A TRANSIENT stream/network error (HF CDN 408/503, reset, decode
                # hiccup) must NOT abort a multi-hour build. Real incident: a single
                # HTTP 408 propagated out of next() and the native layer called
                # terminate() -> SIGABRT, killing a 3.3h / 2.29B-token run. Here we
                # flush progress atomically, drop this sub-stream for the REST OF THIS
                # SESSION, and keep packing from the others. The per-substream cursor
                # (docs_consumed) means the NEXT session re-opens this stream from
                # exactly here, so no data is lost — only this session's mix skews.
                sub_id = gens[idx].get("sub_id", gens[idx].get("weight_key", "?"))
                print(f"[build] sub-stream '{sub_id}' errored "
                      f"({type(e).__name__}: {e}); flushing + dropping it for this "
                      f"session (resumes next run).", file=sys.stderr)
                train_w.flush(); val_w.flush()
                cursor.save_atomic(cfg)
                alive.remove(idx)
                continue
            if not doc:
                continue

            ids = _encode(doc)
            cursor.total_docs += 1
            docs_since_flush += 1

            # Fill the validation holdout FIRST (fixed early budget, stable on re-run).
            if cursor.val_tokens < val_target:
                need = val_target - cursor.val_tokens
                if ids.size <= need:
                    val_w.write(ids)
                    cursor.val_tokens = val_w.tokens
                else:
                    val_w.write(ids[:need])
                    train_w.write(ids[need:])
                    cursor.val_tokens = val_w.tokens
                    cursor.train_tokens = train_w.tokens
                    tokens_since_flush += ids.size
            else:
                n = train_w.write(ids)
                cursor.train_tokens = train_w.tokens
                tokens_since_flush += n

            # --- atomic, periodic checkpoint of cursor + bin flush ---
            if (docs_since_flush >= cfg.cursor_flush_every_docs
                    or tokens_since_flush >= cfg.bin_flush_every_tokens):
                train_w.flush()
                val_w.flush()
                cursor.save_atomic(cfg)
                docs_since_flush = 0
                tokens_since_flush = 0

                # --- wall-clock guard: stop cleanly BEFORE the platform kill so the
                # partial (resumable) output is committed. Checked at flush boundaries
                # so cursor + bins are already consistent on disk when we break. ---
                if cfg.max_seconds is not None and (time.time() - t0) >= cfg.max_seconds:
                    print(f"[build] wall-clock guard hit ({cfg.max_seconds/3600:.2f}h) "
                          f"at {cursor.train_tokens/1e6:.1f}M/{budget_tokens/1e6:.0f}M "
                          f"train tok -> flushing + exiting 0 so this session COMMITS a "
                          f"resumable output. Re-run to continue from here.",
                          file=sys.stderr)
                    break

            now = time.time()
            if now - last_report > 30:
                rate = cursor.train_tokens / max(1e-9, now - t0)
                pct = 100.0 * cursor.train_tokens / budget_tokens
                eta_h = (budget_tokens - cursor.train_tokens) / max(1.0, rate) / 3600.0
                print(f"[build] {cursor.train_tokens/1e6:8.1f}M / "
                      f"{budget_tokens/1e6:.0f}M train tok ({pct:5.1f}%)  "
                      f"val {cursor.val_tokens/1e6:.1f}M  "
                      f"{rate/1e3:6.1f}k tok/s  ETA {eta_h:5.1f}h  "
                      f"docs={cursor.total_docs}  live_streams={len(alive)}")
                last_report = now

        cursor.finished = cursor.train_tokens >= budget_tokens
    except KeyboardInterrupt:
        print("\n[build] interrupted — flushing + saving cursor atomically "
              "so the next run RESUMES.", file=sys.stderr)
    finally:
        train_w.flush(); val_w.flush()
        train_w.close(); val_w.close()
        cursor.save_atomic(cfg)
        _write_meta(cfg, cursor, eos_id)

    status = "FINISHED" if cursor.finished else "PAUSED (resumable)"
    print(f"[build] {status}: train={cursor.train_tokens/1e6:.1f}M tok, "
          f"val={cursor.val_tokens/1e6:.1f}M tok, docs={cursor.total_docs}")
    print(f"[build] outputs: {train_bin}  {val_bin}")
    if not cursor.finished and not alive:
        print("[build] NOTE: all streams exhausted before hitting the token budget. "
              "Add shards / lift mix, or pin a larger split.", file=sys.stderr)


def _write_meta(cfg: DataConfig, cursor: Cursor, eos_id: int) -> None:
    meta = {
        "vocab_size": cfg.vocab_size,
        "ctx_len": cfg.ctx_len,
        "dtype": cfg.bin_dtype,
        "eos_id": eos_id,
        "train_tokens": cursor.train_tokens,
        "val_tokens": cursor.val_tokens,
        "total_docs": cursor.total_docs,
        "config_hash": cfg.config_hash(),
        "train_bin": os.path.abspath(os.path.join(cfg.data_dir, "train.bin")),
        "val_bin": os.path.abspath(os.path.join(cfg.data_dir, "val.bin")),
        "finished": cursor.finished,
        # the number of full ctx windows currently available for sampling
        "train_windows": cursor.train_tokens // cfg.ctx_len,
        "val_windows": cursor.val_tokens // cfg.ctx_len,
    }
    with open(os.path.join(cfg.data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ===========================================================================
# 9. CLI
# ===========================================================================

def main(argv: Optional[List[str]] = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(
        description="Prepare packed uint16 train/val .bin for from-scratch MoE GPT pretrain.")
    ap.add_argument("--data-dir", default=None,
                    help="output dir (defaults to PRETRAIN_DATA_DIR or ./data; "
                         "point at a Drive mount for multi-session persistence).")
    ap.add_argument("--train-tokenizer", action="store_true",
                    help="fit the 16k byte-level BPE on a sampled corpus slice.")
    ap.add_argument("--build", action="store_true",
                    help="stream + tokenize + pack train.bin / val.bin (resumable).")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--ctx-len", type=int, default=None)
    ap.add_argument("--token-budget-b", type=float, default=None,
                    help="override total token budget (in billions).")
    ap.add_argument("--corpus", type=str, default=None,
                    help="named corpus preset to swap in the shard list (e.g. "
                         "'kaggle' = directly-streamable, non-gated code+NL for a "
                         "headless kernel). Omit / 'winner' keeps the inline shards.")
    ap.add_argument("--shards", type=str, default=None,
                    help="comma-separated shard NAMES to restrict to (e.g. "
                         "codeparrot-clean,fineweb-edu for a CPU smoke test). Applied "
                         "AFTER --corpus, so it subsets the chosen preset.")
    ap.add_argument("--reset-cursor", action="store_true",
                    help="delete the resume cursor and rebuild from scratch.")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock guard (hours): break the build at a flush "
                         "boundary, commit a resumable partial, exit 0. Use on Kaggle "
                         "to stay under the ~12h kill. Omit for unbounded local runs.")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="wall-clock guard in seconds (overrides --max-hours).")
    args = ap.parse_args(argv)

    # corpus preset (swap shard list) BEFORE any --shards subsetting / overrides.
    cfg = apply_corpus_preset(cfg, args.corpus)

    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.vocab_size:
        cfg.vocab_size = args.vocab_size
    if args.ctx_len:
        cfg.ctx_len = args.ctx_len
    if args.token_budget_b:
        cfg.token_budget_b = args.token_budget_b
    if args.max_seconds is not None:
        cfg.max_seconds = args.max_seconds
    elif args.max_hours is not None:
        cfg.max_seconds = args.max_hours * 3600.0
    if args.shards:
        keep = {s.strip() for s in args.shards.split(",") if s.strip()}
        valid_names = [s.name for s in cfg.shards]
        cfg.shards = [s for s in cfg.shards if s.name in keep]
        if not cfg.shards:
            print(f"[main] --shards matched nothing; valid for this corpus: "
                  f"{valid_names}", file=sys.stderr)
            return 2

    os.makedirs(cfg.data_dir, exist_ok=True)
    print(f"[main] data_dir = {os.path.abspath(cfg.data_dir)}")
    print(f"[main] config_hash = {cfg.config_hash()}  vocab={cfg.vocab_size} "
          f"ctx={cfg.ctx_len} budget={cfg.token_budget_b}B")

    if args.reset_cursor:
        for fn in ("cursor.json",):
            p = os.path.join(cfg.data_dir, fn)
            if os.path.exists(p):
                os.remove(p)
                print(f"[main] removed {p}")

    # Default (no action flags) = do the full pipeline: tokenizer then build.
    do_tok = args.train_tokenizer or not (args.train_tokenizer or args.build)
    do_build = args.build or not (args.train_tokenizer or args.build)

    if do_tok:
        train_tokenizer(cfg)
    if do_build:
        build(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
