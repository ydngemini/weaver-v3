#!/usr/bin/env python3
"""oncompute.ai C2D entrypoint — fine-tune the Weaver MoE base on the 480-trace
distill corpus, emit the trained SFT checkpoint as the downloadable output.

Contract (Ocean Compute-to-Data):
  * input  — the base checkpoint is mounted under /data/inputs/ (publish the
             2.3GB ckpt_latest.pt as the job's input dataset). Override locally
             with BASE_CKPT=/path/to/ckpt.pt.
  * output — the trained checkpoint is written to /data/outputs/ (Ocean collects
             this dir and makes it downloadable). Override with OUT_DIR=...
  * corpus + tokenizer are baked into the image (small); model.py rides along.

Env knobs: EPOCHS (default 3), LR (default 2e-4), BASE_CKPT, OUT_DIR.

Portable: identical behaviour on any NVIDIA GPU box — just set BASE_CKPT/OUT_DIR.
"""
import glob
import os
import subprocess
import sys

HERE = "/app/pretrain_moe"
DISTILL = os.path.join(HERE, "distill")
DATA = os.path.join(DISTILL, "agentic_traces.jsonl")
TOKENIZER = os.path.join(HERE, "tokenizer.json")
SFT = os.path.join(DISTILL, "sft_train.py")


def find_base_ckpt():
    env = os.environ.get("BASE_CKPT")
    if env and os.path.exists(env):
        return env
    # Ocean mounts published input datasets under /data/inputs/<did>/<index>
    for root in ("/data/inputs", "/data/ddo", "/app/inputs"):
        hits = glob.glob(os.path.join(root, "**", "*.pt"), recursive=True)
        if hits:
            return max(hits, key=os.path.getsize)  # biggest .pt = the model bundle
    return None


def main():
    out_dir = os.environ.get("OUT_DIR") or ("/data/outputs" if os.path.isdir("/data/outputs") else "./outputs")
    os.makedirs(out_dir, exist_ok=True)

    base = find_base_ckpt()
    if not base:
        print("[run_job] WARNING: no base checkpoint found under /data/inputs — "
              "running the trainer's --smoke self-test only (no real fine-tune).", flush=True)
        subprocess.check_call([sys.executable, SFT, "--smoke"])
        return

    epochs = os.environ.get("EPOCHS", "3")
    lr = os.environ.get("LR", "2e-4")
    print(f"[run_job] base ckpt: {base} ({os.path.getsize(base) / 1e9:.2f} GB)", flush=True)
    cmd = [sys.executable, SFT, "--base-ckpt", base, "--data", DATA,
           "--tokenizer", TOKENIZER, "--out", out_dir, "--epochs", epochs, "--lr", lr]
    print("[run_job] " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

    outs = glob.glob(os.path.join(out_dir, "**", "*.pt"), recursive=True)
    for pt in outs:
        print(f"[run_job] OUTPUT {pt} ({os.path.getsize(pt) / 1e9:.2f} GB)", flush=True)
    print(f"[run_job] done — {len(outs)} checkpoint(s) in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
