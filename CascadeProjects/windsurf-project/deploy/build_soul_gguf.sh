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

# Auto-detect the llama.cpp checkout (the convert script + quantize live here).
# Override with LLAMA_CPP=/path/to/llama.cpp. Candidates cover both partitions and
# the nested clone layout seen in this workspace.
if [ -z "${LLAMA_CPP:-}" ]; then
    for _c in \
        "/media/ydn/SYPHER_CORE2/Untitled Folder/llama.cpp/llama.cpp" \
        "/media/ydn/SYPHER_CORE2/Untitled Folder/llama.cpp" \
        "/media/ydn/SYPHER_CORE/Untitled Folder/llama.cpp/llama.cpp"; do
        [ -f "$_c/convert_hf_to_gguf.py" ] && LLAMA_CPP="$_c" && break
    done
fi
: "${LLAMA_CPP:?could not find llama.cpp — set LLAMA_CPP=/path/to/llama.cpp (needs convert_hf_to_gguf.py)}"
MERGED_DIR="${MERGED_DIR:-$PROJ/weaver_merged_1B}"
OUT_GGUF="$PROJ/weaver_merged_1B_Q4_K_M.gguf"
F16_GGUF="$PROJ/weaver_merged_1B_f16.gguf"

echo "▶ 1/3  Merge LoRA adapter → standalone HF model ($MERGED_DIR)"
# Skip ONLY if the merged model actually has weights — a config.json alone is a
# stub from an aborted merge and would make convert_hf_to_gguf.py fail downstream.
if ls "$MERGED_DIR"/*.safetensors >/dev/null 2>&1 || [ -f "$MERGED_DIR/pytorch_model.bin" ]; then
    echo "   (already merged — skipping)"
else
    [ -d "$MERGED_DIR" ] && rm -rf "$MERGED_DIR"   # clear any config-only stub
    venv/bin/python3 merge_lora.py --output "$MERGED_DIR"
fi

echo "▶ 2/3  Convert HF → f16 GGUF"
venv/bin/python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED_DIR" \
    --outfile "$F16_GGUF" --outtype f16

echo "▶ 3/3  Quantize f16 → Q4_K_M"
QUANT_BIN=""
for _b in "$LLAMA_CPP/build/bin/llama-quantize" "$LLAMA_CPP/build/bin/quantize" \
          "$LLAMA_CPP/llama-quantize"; do
    [ -x "$_b" ] && QUANT_BIN="$_b" && break
done
if [ -z "$QUANT_BIN" ]; then
    echo "   llama-quantize not built — compiling llama.cpp (one-time, needs cmake + a compiler)…"
    cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$LLAMA_CPP/build" --target llama-quantize -j"$(nproc)"
    QUANT_BIN="$LLAMA_CPP/build/bin/llama-quantize"
fi
"$QUANT_BIN" "$F16_GGUF" "$OUT_GGUF" Q4_K_M

rm -f "$F16_GGUF"
echo ""
echo "✅ Soul Voice GGUF ready:"
ls -lh "$OUT_GGUF"
echo ""
echo "Next: scp it to the Oracle box into windsurf-project/, e.g."
echo "   scp '$OUT_GGUF' ubuntu@<ORACLE_IP>:~/weaver/CascadeProjects/windsurf-project/"
