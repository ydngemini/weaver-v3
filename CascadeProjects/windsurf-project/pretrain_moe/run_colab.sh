#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_colab.sh — one-session launcher for the from-scratch MoE GPT pretrain
# ═══════════════════════════════════════════════════════════════════════════
# This is the per-session command. Run it INSIDE a Colab cell (after Drive is
# mounted) or in any GPU shell with the Drive dir reachable. It is SAFE TO
# RE-RUN every session: prepare_data.py resumes its packing cursor and train.py
# auto-resumes from the latest atomic Drive checkpoint (step / tokens / AdamW /
# RNG / LR schedule all restored). Kill it at the ~12h Colab cap, reconnect,
# run it again — repeat ~9 times until the 6B-token budget is met.
#
#   FREE COLAB HAS NO HEADLESS CLI. There is no command that submits this to a
#   free T4 from a terminal — colab.research.google.com is browser/notebook only,
#   and `gcloud colab` is the PAID Colab Enterprise (Vertex) runtime, not the
#   free tier. So: open colab_pretrain.ipynb in the browser, pick a T4 runtime,
#   Run all. That notebook calls exactly this script's logic. See README.md.
#
# USAGE (from a Colab cell, repo cloned to /content/pretrain_moe):
#   !bash /content/pretrain_moe/run_colab.sh
#
# USAGE (local GPU box / sanity run), env-overridable:
#   DRIVE_ROOT=/path/to/moe_pretrain CODE_DIR=/path/to/pretrain_moe \
#       bash run_colab.sh
#
# Env knobs (all optional; defaults target the Colab Drive layout):
#   DRIVE_ROOT  persistent root for data + ckpts (default: Drive moe_pretrain)
#   CODE_DIR    where model.py/prepare_data.py/train.py live (default: this dir)
#   CONFIG      winner_config.json path (default: write the embedded winner config)
#   TOKEN_BUDGET_B   total token budget in billions (default 6)
#   SKIP_PREP   set to 1 to skip data prep (e.g. data already packed)
#   SKIP_INSTALL set to 1 to skip pip install (e.g. deps already present)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# --- locate the code dir (defaults to the folder this script lives in) -------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$SCRIPT_DIR}"

# --- persistent root: a Google Drive mount on Colab, else a local dir --------
if [[ -z "${DRIVE_ROOT:-}" ]]; then
  if [[ -d "/content/drive/MyDrive" ]]; then
    DRIVE_ROOT="/content/drive/MyDrive/moe_pretrain"
  else
    DRIVE_ROOT="$CODE_DIR/run"
    echo "[run_colab] no Drive mount found -> using local DRIVE_ROOT=$DRIVE_ROOT"
    echo "[run_colab] (on Colab, mount Drive first so the run survives reconnects)"
  fi
fi

DATA_DIR="$DRIVE_ROOT/data"
CKPT_DIR="$DRIVE_ROOT/ckpts"
CONFIG="${CONFIG:-$DRIVE_ROOT/winner_config.json}"
TOKEN_BUDGET_B="${TOKEN_BUDGET_B:-6}"
# Interpreter as a single-element array so a path containing spaces (e.g. a venv
# under a directory with spaces, like this dev box) is passed as ONE argv
# element, not word-split. Quote "${PY[@]}" at every call site below.
PY=("${PYTHON:-python}")

mkdir -p "$DATA_DIR" "$CKPT_DIR"

echo "═══════════════════════════════════════════════════════════════════════"
echo " from-scratch MoE GPT pretrain — one session (re-run to resume)"
echo "   CODE_DIR   = $CODE_DIR"
echo "   DRIVE_ROOT = $DRIVE_ROOT"
echo "   DATA_DIR   = $DATA_DIR"
echo "   CKPT_DIR   = $CKPT_DIR"
echo "   CONFIG     = $CONFIG"
echo "   BUDGET     = ${TOKEN_BUDGET_B}B tokens"
echo "═══════════════════════════════════════════════════════════════════════"

# --- 1/4  write the winner config to Drive (idempotent) ----------------------
if [[ ! -f "$CONFIG" ]]; then
  echo "▶ 1/4  writing winner config -> $CONFIG"
  cat > "$CONFIG" <<'JSON'
{
  "vocab_size": 16384,
  "ctx_len": 1024,
  "n_layer": 9,
  "n_head": 10,
  "n_embd": 640,
  "head_dim": 64,
  "n_experts": 8,
  "top_k": 2,
  "d_ff": 1920,
  "capacity_factor": 1.25,
  "drop_overflow_tokens": true,
  "router_noise_std": 1.0,
  "aux_loss_coef": 0.01,
  "router_z_loss_coef": 0.001,
  "use_qk_norm": true,
  "dropout": 0.0,
  "bias": false,
  "init_std": 0.02
}
JSON
else
  echo "▶ 1/4  winner config already present -> $CONFIG"
fi

# --- 2/4  install deps (skippable) -------------------------------------------
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  echo "▶ 2/4  installing python deps"
  "${PY[@]}" -m pip install -q --upgrade \
    "torch" "numpy" "datasets" "tokenizers" "huggingface_hub" \
    "smart_open[s3]" "boto3" || {
      echo "[run_colab] pip install hit an error; on Colab torch is preinstalled — continuing." >&2
    }
else
  echo "▶ 2/4  SKIP_INSTALL=1 -> skipping pip install"
fi

# --- 3/4  prepare data (resumable; skippable) --------------------------------
if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  echo "▶ 3/4  prepare_data.py (train 16k BPE, pack train.bin/val.bin — RESUMABLE)"
  "${PY[@]}" "$CODE_DIR/prepare_data.py" \
      --data-dir "$DATA_DIR" \
      --token-budget-b "$TOKEN_BUDGET_B"
else
  echo "▶ 3/4  SKIP_PREP=1 -> skipping data prep (assuming train.bin already packed)"
fi

# --- 4/4  train (auto-resumes from the latest Drive checkpoint) ---------------
echo "▶ 4/4  train.py (multi-session; auto-resumes from $CKPT_DIR)"
"${PY[@]}" "$CODE_DIR/train.py" \
    --data-dir "$DATA_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --config   "$CONFIG" \
    --token-budget-b "$TOKEN_BUDGET_B"

echo "═══════════════════════════════════════════════════════════════════════"
echo " session ended. If Colab killed it at the cap, RECONNECT and run this"
echo " again — it resumes from the latest atomic checkpoint in $CKPT_DIR."
echo "═══════════════════════════════════════════════════════════════════════"
