#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kaggle_prep.py  —  HEADLESS CPU data-prep kernel for the from-scratch MoE GPT.

WHY THIS EXISTS (the two Kaggle facts that shape it):
    1. A Kaggle SCRIPT kernel runs ONLY its single `code_file`. Sibling .py files
       pushed in the same folder are NOT importable. So the project code
       (model.py / prepare_data.py / train.py) is shipped as a separate Kaggle
       DATASET (`nathaniellockwood/weaver-moe-code`) and mounted read-only at
       /kaggle/input/weaver-moe-code/. We put THAT on sys.path before importing.
    2. The original corpus (The-Stack-v2 + starcoderdata) cannot run headless —
       Stack-v2 rows are blob_id POINTERS (need AWS + a Software Heritage S3
       agreement) and starcoderdata is GATED. We use prepare_data.py's
       `--corpus kaggle` preset: codeparrot-clean + the-stack-smol + fineweb-edu,
       all DIRECTLY streamable with only enable_internet=true (no auth at all).

WHAT THIS KERNEL DOES:
    * Puts /kaggle/input/weaver-moe-code on sys.path.
    * Runs prepare_data.py with --corpus kaggle to pack train.bin / val.bin +
      the trained 16k BPE tokenizer into /kaggle/working (which becomes this
      kernel version's output -> the `weaver-moe-data` dataset).
    * FIRST-RUN BUDGET is modest by default (~0.3B tokens) so it finishes inside
      one CPU session. Override with env KAGGLE_PREP_BUDGET_B or arg --budget-b.

OUTPUT (in /kaggle/working, saved as the kernel version output):
    train.bin  val.bin  meta.json  tokenizer/   cursor.json
After it completes, `kaggle kernels output` pulls these and
`kaggle datasets create -p <out>` makes the `weaver-moe-data` dataset.

CPU-SESSION NOTE: a Kaggle CPU kernel also has a wall-clock cap (~12h). The
default ~0.3B budget streams + tokenizes well inside that. prepare_data.py is
RESUMABLE (atomic cursor.json), so if you raise the budget and a session is
killed, a re-push continues where it stopped instead of restarting.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


# --- Kaggle mount layout (overridable via env for local dry-runs) ---
CODE_INPUT = os.environ.get("KAGGLE_CODE_INPUT", "/kaggle/input/weaver-moe-code")
# Prior prep output (the weaver-moe-data dataset) mounted read-only so a later CPU
# session RESUMES instead of restarting. Absent on the first (cold) session — handled.
DATA_INPUT = os.environ.get("KAGGLE_DATA_INPUT", "/kaggle/input/weaver-moe-data")
WORKING = os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")

# Wall-clock guard handed to prepare_data.py so a session commits a resumable partial
# before Kaggle's ~12h CPU kill (a timed-out kernel is FAILED -> no output saved).
PREP_MAX_HOURS = float(os.environ.get("KAGGLE_PREP_MAX_HOURS", "11.5"))

# First-run token budget (billions). Modest by default so one CPU session
# finishes; raise it for a larger data dataset (prep is resumable across pushes).
PREP_BUDGET_B = float(os.environ.get("KAGGLE_PREP_BUDGET_B", "6.0"))  # FULL PLAN: 6.0B tokens == model budget (15258 steps x 393k tok/step).
# prepare_data.py is RESUMABLE (atomic cursor.json) — a CPU session won't pack 6B at
# once, so run this kernel across multiple sessions, re-versioning weaver-moe-data each
# time, until meta.json reports finished:true. (Was 0.03 == 30M for the validation run;
# at that size an 11.5h fp16 train would loop the corpus ~15x and overfit.)
# The headless-safe corpus preset defined in prepare_data.py.
CORPUS = os.environ.get("KAGGLE_PREP_CORPUS", "kaggle")


def _parse_budget_arg(argv) -> float:
    """Allow `--budget-b X` on the kernel argv to override the env/default."""
    budget = PREP_BUDGET_B
    for i, a in enumerate(argv):
        if a == "--budget-b" and i + 1 < len(argv):
            try:
                budget = float(argv[i + 1])
            except ValueError:
                pass
        elif a.startswith("--budget-b="):
            try:
                budget = float(a.split("=", 1)[1])
            except ValueError:
                pass
    return budget


def _find_prior_data_dir() -> "str | None":
    """Locate a prior prep output (the weaver-moe-data mount) to RESUME from.
    Prefer the conventional mount path; otherwise auto-discover by walking
    /kaggle/input for a directory holding the resume sentinel (cursor.json), which
    Kaggle sometimes nests under <owner>/<slug>/. Returns None on a cold session."""
    sentinel = "cursor.json"
    if os.path.isfile(os.path.join(DATA_INPUT, sentinel)):
        return DATA_INPUT
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None
    for dirpath, _dirs, files in os.walk(root):
        # don't mistake the code dataset for a data mount
        if os.path.abspath(dirpath) == os.path.abspath(CODE_INPUT):
            continue
        if sentinel in files:
            return dirpath
    return None


def seed_resume_from_mount() -> bool:
    """Copy a prior partial (cursor.json + train.bin/val.bin + meta.json + tokenizer/)
    from the read-only data mount into WORKING so prepare_data.py RESUMES and APPENDS.
    The bins must live in the writable data_dir (BinWriter opens them append+truncate),
    so a copy is required — the mount can't be written. No-op (cold start) if absent."""
    prior = _find_prior_data_dir()
    if prior is None:
        print("[resume] no prior weaver-moe-data mount found -> COLD start "
              "(fresh tokenizer + empty bins). This is correct for session 1.")
        return False
    print(f"[resume] found prior prep output at: {prior}")
    os.makedirs(WORKING, exist_ok=True)
    for fname in ("cursor.json", "train.bin", "val.bin", "meta.json"):
        src = os.path.join(prior, fname)
        if os.path.isfile(src):
            dst = os.path.join(WORKING, fname)
            print(f"[resume] copying {fname} ({os.path.getsize(src)/1e6:.1f} MB) -> WORKING")
            # copyfile (NOT copy2): Kaggle input mounts are READ-ONLY and copy2 would
            # replicate those mode bits -> BinWriter's r+b/ab open and _write_meta's "w"
            # open would then raise PermissionError on the first resume. copyfile gives
            # default (writable) perms; chmod makes it explicit/bulletproof.
            shutil.copyfile(src, dst)
            os.chmod(dst, 0o644)
    src_tok = os.path.join(prior, "tokenizer")
    if os.path.isdir(src_tok):
        dst_tok = os.path.join(WORKING, "tokenizer")
        if os.path.isdir(dst_tok):
            shutil.rmtree(dst_tok)
        shutil.copytree(src_tok, dst_tok)
        print("[resume] copied tokenizer/ -> WORKING (stable BPE preserved).")
    print("[resume] seed complete -> prepare_data.py will resume from the cursor.")
    return True


def seed_tokenizer_from_code(code_dir: str) -> bool:
    """Copy the pre-trained tokenizer bundled in the code dataset into WORKING so
    prepare_data.py reuses it (train_tokenizer is idempotent on vocab_size) and skips
    the crash-prone, non-resumable BPE re-fit. No-op if WORKING already has one (a
    resume mount wins — it's the same stable BPE) or the code dataset ships none."""
    dst = os.path.join(WORKING, "tokenizer")
    if os.path.isfile(os.path.join(dst, "tokenizer.json")):
        print("[tokenizer] WORKING already has a tokenizer (from resume mount) -> keep it.")
        return False
    src = os.path.join(code_dir, "tokenizer")
    if not os.path.isfile(os.path.join(src, "tokenizer.json")):
        print("[tokenizer] no bundled tokenizer in code dataset -> prepare_data.py will "
              "train one (cold path). Bundle kaggle/code_dataset/tokenizer/ to skip this.")
        return False
    os.makedirs(WORKING, exist_ok=True)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[tokenizer] seeded STABLE pre-trained BPE from code dataset -> {dst} "
          "(prepare_data.py will REUSE it, skipping the re-fit).")
    return True


def report_env() -> None:
    print("=" * 70)
    print("kaggle_prep.py — headless CPU data-prep (pack train.bin/val.bin)")
    print("=" * 70)
    import platform
    print(f"[env] python {platform.python_version()}  cwd={os.getcwd()}")
    print(f"[env] CODE_INPUT={CODE_INPUT}  exists={os.path.isdir(CODE_INPUT)}")
    print(f"[env] DATA_INPUT={DATA_INPUT}  exists={os.path.isdir(DATA_INPUT)} (prior partial; absent on session 1)")
    print(f"[env] WORKING={WORKING}")
    print(f"[env] CORPUS={CORPUS}  PREP_BUDGET_B={PREP_BUDGET_B}  PREP_MAX_HOURS={PREP_MAX_HOURS}")


def ensure_deps() -> None:
    """The prep needs `datasets` + `tokenizers`. Kaggle's CPU image usually has
    them; install only if missing. numpy ships preinstalled."""
    missing = []
    for mod, pkg in (("datasets", "datasets"), ("tokenizers", "tokenizers"),
                     ("numpy", "numpy")):
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        print(f"[deps] installing missing: {missing}")
        rc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-input", *missing]
        ).returncode
        if rc != 0:
            print(f"[deps] pip install returned {rc}; continuing (HF streaming may "
                  f"fail if datasets is truly absent).")
    else:
        print("[deps] datasets + tokenizers + numpy already importable.")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    report_env()

    # Locate the code dataset. Kaggle does NOT always mount an attached dataset at
    # /kaggle/input/<slug> — it can nest it (e.g. under /kaggle/input/datasets/...),
    # so auto-discover the dir that actually holds prepare_data.py rather than
    # hardcoding a path.
    code_dir = None
    if os.path.isdir(CODE_INPUT) and os.path.exists(os.path.join(CODE_INPUT, "prepare_data.py")):
        code_dir = CODE_INPUT
    else:
        for dp, _dns, fns in os.walk("/kaggle/input"):
            if "prepare_data.py" in fns and "model.py" in fns:
                code_dir = dp
                break
    if code_dir is None:
        tree = []
        try:
            for dp, dns, fns in os.walk("/kaggle/input"):
                if dp.count("/") <= 6:
                    tree.append(f"  {dp}  dirs={dns[:8]} files={fns[:8]}")
        except Exception as e:
            tree = [f"<could not walk /kaggle/input: {e}>"]
        print("[FATAL] could not find prepare_data.py anywhere under /kaggle/input. "
              "Is 'nathaniellockwood/weaver-moe-code' attached as a dataset_source? Tree:\n"
              + "\n".join(tree))
        return 2
    print(f"[code] located code dataset at: {code_dir}")

    # Make the project code importable + locatable (Kaggle script-kernel fix).
    sys.path.insert(0, code_dir)
    prep_py = os.path.join(code_dir, "prepare_data.py")

    ensure_deps()

    budget = _parse_budget_arg(argv)
    os.makedirs(WORKING, exist_ok=True)

    # Cross-session resume: pull a prior partial into WORKING before building so this
    # session continues toward the 6B budget instead of restarting from zero.
    seed_resume_from_mount()
    # If no tokenizer arrived via the resume mount, seed the STABLE pre-trained BPE
    # bundled in the code dataset so prepare_data.py REUSES it (idempotent on
    # vocab_size) instead of re-fitting. Re-fitting is the slow, crash-prone,
    # NON-resumable step (a prior session segfaulted in it), and a different BPE would
    # silently corrupt already-packed bins. Reuse = stable tokenizer across all sessions.
    seed_tokenizer_from_code(code_dir)

    cmd = [
        sys.executable, prep_py,
        "--corpus", CORPUS,
        "--data-dir", WORKING,
        "--token-budget-b", str(budget),
        "--max-hours", str(PREP_MAX_HOURS),
    ]
    print("=" * 70)
    print(f"[prep] launching: {' '.join(cmd)}")
    print("=" * 70)
    # Hardened env: unbuffered stdout (a buffered crash hid the last segfault's
    # location) + single-threaded tokenizers (the datasets<->tokenizers native thread
    # interaction is a known segfault source on headless kernels).
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # cwd = CODE_INPUT so prepare_data.py's relative defaults resolve there;
    # --data-dir WORKING pins the OUTPUT into the saved kernel output.
    rc = subprocess.run(cmd, cwd=code_dir, env=env).returncode

    train_bin = os.path.join(WORKING, "train.bin")
    if not os.path.exists(train_bin):
        # Nothing was packed -> nothing to resume. Truly fatal.
        print(f"[FATAL] prep produced no train.bin at {train_bin} (rc={rc}). "
              f"Check enable_internet=true + HF reachability. A single source 404 "
              f"should be SKIPPED, not fatal — inspect the log.")
        return 3
    if rc != 0:
        # prepare_data.py crashed/exited non-zero (e.g. the flaky native segfault) BUT
        # a partial train.bin + atomic cursor.json exist. Commit them and EXIT 0 so this
        # session's progress is SAVED — a FAILED kernel discards /kaggle/working
        # entirely, losing hours of streaming. The bin tail is truncated back to the
        # cursor on the next run, so the committed partial is self-healing + resumable.
        print(f"[WARN] prepare_data.py exited rc={rc} but a partial train.bin exists "
              f"({os.path.getsize(train_bin)/1e6:.1f} MB) -> committing a RESUMABLE "
              f"partial and exiting 0. Re-run (add weaver-moe-data source) to continue.")

    # Report what we packed so the log confirms a usable data dataset.
    try:
        size_mb = os.path.getsize(train_bin) / 1e6
        print(f"[prep] DONE. train.bin = {size_mb:.1f} MB at {train_bin}")
    except OSError:
        pass
    print("=" * 70)
    print("[done] /kaggle/working now holds train.bin / val.bin / meta.json / "
          "tokenizer/. Pull it with `kaggle kernels output` and create the "
          "`weaver-moe-data` dataset from it (run_kaggle.sh prep does both).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
