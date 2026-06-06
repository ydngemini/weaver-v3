#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# build_soul_gguf.sh — bake the Weaver LoRA into a quantized GGUF
# ═══════════════════════════════════════════════════════════════════
# RUN THIS LOCALLY on your x86 box (has transformers + the llama.cpp repo).
# Produces  weaver_merged_1B_Q4_K_M.gguf  (~0.8 GB) to upload to Oracle,
# so the ARM box never has to run the heavy transformers merge itself.
#
#   ./deploy/build_soul_gguf.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."                      # → windsurf-project/
PROJ="$(pwd)"

LLAMA_CPP="${LLAMA_CPP:-/media/ydn/SYPHER_CORE/Untitled Folder/llama.cpp}"
MERGED_DIR="${MERGED_DIR:-$PROJ/weaver_merged_1B}"
OUT_GGUF="$PROJ/weaver_merged_1B_Q4_K_M.gguf"
F16_GGUF="$PROJ/weaver_merged_1B_f16.gguf"

echo "▶ 1/3  Merge LoRA adapter → standalone HF model ($MERGED_DIR)"
if [ ! -f "$MERGED_DIR/config.json" ]; then
    venv/bin/python3 merge_lora.py --output "$MERGED_DIR"
else
    echo "   (already merged — skipping)"
fi

echo "▶ 2/3  Convert HF → f16 GGUF"
venv/bin/python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED_DIR" \
    --outfile "$F16_GGUF" --outtype f16

echo "▶ 3/3  Quantize f16 → Q4_K_M"
QUANT_BIN="$LLAMA_CPP/build/bin/llama-quantize"
[ -x "$QUANT_BIN" ] || QUANT_BIN="$LLAMA_CPP/build/bin/quantize"
"$QUANT_BIN" "$F16_GGUF" "$OUT_GGUF" Q4_K_M

rm -f "$F16_GGUF"
echo ""
echo "✅ Soul Voice GGUF ready:"
ls -lh "$OUT_GGUF"
echo ""
echo "Next: scp it to the Oracle box into windsurf-project/, e.g."
echo "   scp '$OUT_GGUF' ubuntu@<ORACLE_IP>:~/weaver/CascadeProjects/windsurf-project/"
