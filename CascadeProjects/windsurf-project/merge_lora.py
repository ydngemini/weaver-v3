#!/usr/bin/env python3
"""
merge_lora.py — Merge Weaver's LoRA adapter into the base model
================================================================
Eliminates adapter overhead during inference by baking the LoRA
weights directly into the base model parameters.

After merging:
  - No PEFT runtime needed at inference time
  - Zero additional latency from adapter application
  - Model can be converted to GGUF for ollama/llama.cpp serving
  - File size equals full model (~2GB for 1B params)

Usage:
    python3 merge_lora.py                          # merge to ./weaver_merged_1B/
    python3 merge_lora.py --output ./my_merged/    # custom output path
    python3 merge_lora.py --quantize                # also quantize to 4-bit GGUF
"""

import argparse
import os
import sys
import json
import time

PROJ = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(PROJ, "weaver_fracture_1B_lora")
DEFAULT_OUTPUT = os.path.join(PROJ, "weaver_merged_1B")


def merge(output_path: str) -> str:
    """Merge LoRA adapter into base model and save standalone."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # Read adapter config for base model name
    with open(os.path.join(ADAPTER_PATH, "adapter_config.json")) as f:
        cfg = json.load(f)
    base_name = cfg["base_model_name_or_path"]

    print(f"[MERGE] Base model:  {base_name}")
    print(f"[MERGE] LoRA adapter: {ADAPTER_PATH}")
    print(f"[MERGE] Output:      {output_path}")
    print(f"[MERGE] LoRA rank:   {cfg.get('r', '?')}, alpha={cfg.get('lora_alpha', '?')}")

    t0 = time.monotonic()

    # Load base model in float32 for clean merge
    print("[MERGE] Loading base model (float32)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Apply LoRA adapter
    print("[MERGE] Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # Merge weights — this bakes LoRA into the base weights
    print("[MERGE] Merging LoRA weights into base model...")
    merged = model.merge_and_unload()

    # Save merged model
    print(f"[MERGE] Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    merged.save_pretrained(output_path)

    # Save tokenizer from adapter directory
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(output_path)

    elapsed = time.monotonic() - t0
    size_mb = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if os.path.isfile(os.path.join(output_path, f))
    ) / 1_000_000

    print(f"[MERGE] ✅ Merged model saved ({size_mb:.0f} MB, {elapsed:.1f}s)")
    print(f"[MERGE]")
    print(f"[MERGE] Next steps:")
    print(f"[MERGE]   1. Update lora_server.py to load from {output_path} directly")
    print(f"[MERGE]   2. Or convert to GGUF: python3 merge_lora.py --quantize")
    print(f"[MERGE]   3. Or create Ollama model:")
    print(f"[MERGE]      echo 'FROM {output_path}' > Modelfile")
    print(f"[MERGE]      ollama create weaver-soul -f Modelfile")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge Weaver LoRA into base model")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--quantize", action="store_true", help="Also create GGUF quantized version")
    args = parser.parse_args()

    merged_path = merge(args.output)

    if args.quantize:
        print("\n[MERGE] To quantize to GGUF, run:")
        print(f"  git clone https://github.com/ggerganov/llama.cpp")
        print(f"  cd llama.cpp && pip install -r requirements.txt")
        print(f"  python convert_hf_to_gguf.py {merged_path} --outfile weaver-soul.gguf")
        print(f"  ./llama-quantize weaver-soul.gguf weaver-soul-q4_k_m.gguf Q4_K_M")


if __name__ == "__main__":
    main()
