#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# setup_oracle.sh — provision the Weaver headless stack on Oracle ARM
# ═══════════════════════════════════════════════════════════════════
# Run ON the Oracle A1 box (Ubuntu 22.04/24.04 aarch64), from inside
# the windsurf-project/ directory after cloning the repo:
#
#   cd ~/weaver/CascadeProjects/windsurf-project
#   bash deploy/setup_oracle.sh
#
# Assumes: weaver_merged_1B_Q4_K_M.gguf has been scp'd here already,
# and deploy/env.oracle.example has been copied to .env and filled in.
set -euo pipefail
cd "$(dirname "$0")/.."
PROJ="$(pwd)"

echo "▶ System packages (build tools, BLAS, python venv)"
sudo apt-get update -y
sudo apt-get install -y build-essential cmake git python3-venv python3-dev \
    libopenblas-dev pkg-config curl

echo "▶ Python venv"
[ -d venv ] || python3 -m venv venv
venv/bin/pip install --upgrade pip wheel setuptools

echo "▶ Core Python deps (ARM-safe — no bitsandbytes/torch/vision/audio)"
# Strip packages that don't build or aren't needed on a headless ARM box.
BAD='bitsandbytes|^torch|torchvision|torchaudio|deepface|insightface|opencv|pyaudio|sounddevice|onnxruntime-gpu'
if [ -f requirements.txt ]; then
    grep -vEi "$BAD" requirements.txt > /tmp/req.arm.txt || true
    venv/bin/pip install -r /tmp/req.arm.txt || \
        echo "   ⚠ some optional deps failed — that's OK for headless; check log"
fi

# Core brain runtime deps installed EXPLICITLY — a single `pip install -r` aborts on
# the first unbuildable optional package, silently skipping everything after it, which
# left Nexus Bus / MoE router / Obsidian bridge dark on the first cloud run. These are
# pure-Python (ARM wheels) and non-negotiable for the headless brain.
echo "▶ Core brain runtime deps (explicit — survive a partial requirements failure)"
venv/bin/pip install --upgrade websockets scikit-learn aiohttp watchdog httpx numpy boto3 \
    "aws-sdk-bedrock-runtime>=0.7.0" "smithy-aws-core>=0.7.0"

echo "▶ Inference deps: OpenAI client + llama-cpp-python (built with OpenBLAS)"
venv/bin/pip install --upgrade "openai>=1.40" python-dotenv
# [server] extra is REQUIRED — weaver-llm.service runs `python -m llama_cpp.server`,
# which imports uvicorn/fastapi/sse-starlette (crash-loops without them).
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
    venv/bin/pip install --upgrade "llama-cpp-python[server]"

echo "▶ Sanity: GGUF present?"
GGUF="$PROJ/weaver_merged_1B_Q4_K_M.gguf"
[ -f "$GGUF" ] && ls -lh "$GGUF" || \
    echo "   ⚠ $GGUF missing — build it locally (deploy/build_soul_gguf.sh) and scp it here."

echo "▶ .env present?"
[ -f .env ] || { echo "   ⚠ copy deploy/env.oracle.example → .env and fill it in"; }

echo ""
echo "✅ Setup done. Start it with:"
echo "   ./start_weaver.sh --headless        # foreground test"
echo "   sudo cp deploy/weaver.service /etc/systemd/system/ && \\"
echo "     sudo systemctl daemon-reload && sudo systemctl enable --now weaver   # persistent"
echo ""
echo "🔒 Keep Oracle's security list CLOSED. Reach dashboards via SSH tunnel:"
echo "   ssh -L 9996:localhost:9996 -L 9990:localhost:9990 ubuntu@<ORACLE_IP>"
