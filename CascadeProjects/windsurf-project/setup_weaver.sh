#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# setup_weaver.sh — Edge Orchestrator Environment Setup
# ═══════════════════════════════════════════════════════════════
# Source this file before working:  source setup_weaver.sh
#
# Target: Intel i5-6360U (Skylake), AVX2, no AVX-512, no GPU
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Rust (QPY serialization speedup) ────────────────────────
export RUSTUP_HOME="/media/ydn/SYPHER_CORE/.rustup"
export CARGO_HOME="/media/ydn/SYPHER_CORE/.cargo"
source "$CARGO_HOME/env" 2>/dev/null

# ── Qiskit Rust Backend ─────────────────────────────────────
export QISKIT_USE_RUST=1

# ── OpenVINO CPU config (AVX2 only, no AVX-512) ────────────
export OPENVINO_NUM_THREADS=$(nproc)
export OV_CPU_BACKEND_CONFIG="ARCH=AVX2"

# ── Activate unified venv ───────────────────────────────────
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "✅ venv activated ($(python3 --version))"
else
    echo "⚠️  venv not found — run: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
fi

# ── Summary ─────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "  Weaver Edge Orchestrator Environment"
echo "═══════════════════════════════════════════════════"
echo "  QISKIT_USE_RUST=$QISKIT_USE_RUST"
echo "  OPENVINO_NUM_THREADS=$OPENVINO_NUM_THREADS"
echo "  OV_CPU_BACKEND_CONFIG=$OV_CPU_BACKEND_CONFIG"
echo "  RUSTUP_HOME=$RUSTUP_HOME"
echo "  Rust: $(rustc --version 2>/dev/null || echo 'not in PATH')"
echo "═══════════════════════════════════════════════════"
