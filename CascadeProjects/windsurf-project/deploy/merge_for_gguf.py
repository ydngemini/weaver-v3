#!/usr/bin/env python3
"""Merge the Weaver LoRA onto a NON-quantized fp16 base for clean GGUF export.

The adapter (weaver_fracture_1B_lora) was trained on the unsloth *bnb-4bit* base,
but merging into a 4-bit base and calling save_pretrained() raises
NotImplementedError under transformers >=5 (revert_weight_conversion has no
reverse op for the bnb weight conversion). Merging onto the equivalent fp16 base
avoids that path entirely and produces a higher-fidelity standalone model that
llama.cpp's convert_hf_to_gguf.py reads directly.

Base defaults to unsloth/Llama-3.2-1B-Instruct (open, ungated); override with
WEAVER_MERGE_BASE. Output dir via --output (default ./weaver_merged_1B).

Usage:  venv/bin/python3 deploy/merge_for_gguf.py --output ./weaver_merged_1B
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
ADAPTER = os.path.join(_PROJ, "weaver_fracture_1B_lora")
BASE = os.environ.get("WEAVER_MERGE_BASE", "unsloth/Llama-3.2-1B-Instruct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(_PROJ, "weaver_merged_1B"))
    args = ap.parse_args()

    print(f"[merge] base   : {BASE}", flush=True)
    print(f"[merge] adapter: {ADAPTER}", flush=True)
    print(f"[merge] output : {args.output}", flush=True)

    # float32 on CPU for a clean, op-complete merge (fp16 CPU ops are spotty).
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(base, ADAPTER)
    print("[merge] merging LoRA weights into base…", flush=True)
    merged = model.merge_and_unload()
    # Cast to fp16 BEFORE saving: halves save-time memory (a fp32 1B save OOM-killed
    # an 8 GB box) and is lossless for the pipeline — the next step converts to f16
    # GGUF anyway. Small shards cap the serialization buffer.
    del model, base
    merged = merged.half()
    print("[merge] saving fp16 shards…", flush=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="1GB")
    AutoTokenizer.from_pretrained(ADAPTER).save_pretrained(args.output)
    print("[merge] done — standalone HF model written", flush=True)


if __name__ == "__main__":
    main()
