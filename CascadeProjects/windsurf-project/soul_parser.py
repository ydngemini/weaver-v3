"""
soul_parser.py — Data Ingestion for Cloud Training Phase

Reads weaver_soul_dataset.jsonl and prepares it for the LoRA fine-tuning
pipeline. Validates structure, computes statistics, and outputs a
cloud-ready training manifest.
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field

DATASET_PATH = Path(__file__).parent / "weaver_soul_dataset.jsonl"
OUTPUT_PATH = Path(__file__).parent / "training_manifest.json"

GOVERNOR_CONFIG = {
    "qpu_time_box_seconds": 600,
    "tau_base": 0.5,
    "lambda_rate": 0.01,
    "adaptive_threshold_formula": "tau_t = tau_base + lambda * (N_calls / delta_t)",
    "description": (
        "T_max = 600s (10-minute QPU session cap). "
        "Adaptive threshold tau_t rises with call density to prevent "
        "rate-limiting during entropy spikes."
    ),
}


@dataclass
class DatasetStats:
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    avg_input_len: float = 0.0
    avg_output_len: float = 0.0
    errors: list = field(default_factory=list)


def validate_sample(sample: dict, idx: int) -> tuple[bool, str]:
    if "messages" not in sample:
        return False, f"Line {idx}: missing 'messages' key"
    msgs = sample["messages"]
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False, f"Line {idx}: 'messages' must have at least 2 entries"
    roles = [m.get("role") for m in msgs]
    if "user" not in roles or "assistant" not in roles:
        return False, f"Line {idx}: need both 'user' and 'assistant' roles"
    for m in msgs:
        if not m.get("content", "").strip():
            return False, f"Line {idx}: empty content in {m.get('role')} message"
    return True, ""


def parse_dataset() -> DatasetStats:
    stats = DatasetStats()

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    input_lens = []
    output_lens = []

    with open(DATASET_PATH, "r") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            stats.total_samples += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                stats.invalid_samples += 1
                stats.errors.append(f"Line {idx}: invalid JSON — {e}")
                continue

            valid, err = validate_sample(sample, idx)
            if valid:
                stats.valid_samples += 1
                msgs = sample["messages"]
                user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
                asst_msgs = [m["content"] for m in msgs if m["role"] == "assistant"]
                if user_msgs:
                    input_lens.append(sum(len(m) for m in user_msgs))
                if asst_msgs:
                    output_lens.append(sum(len(m) for m in asst_msgs))
            else:
                stats.invalid_samples += 1
                stats.errors.append(err)

    if input_lens:
        stats.avg_input_len = sum(input_lens) / len(input_lens)
    if output_lens:
        stats.avg_output_len = sum(output_lens) / len(output_lens)

    return stats


def write_manifest(stats: DatasetStats):
    manifest = {
        "dataset_path": str(DATASET_PATH),
        "total_samples": stats.total_samples,
        "valid_samples": stats.valid_samples,
        "invalid_samples": stats.invalid_samples,
        "avg_input_chars": round(stats.avg_input_len, 1),
        "avg_output_chars": round(stats.avg_output_len, 1),
        "governor_config": GOVERNOR_CONFIG,
        "errors": stats.errors[:20],
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {OUTPUT_PATH}")


def main():
    print("═══════════════════════════════════════════════")
    print("  Soul Parser — Dataset Ingestion")
    print("═══════════════════════════════════════════════")
    print(f"  Source: {DATASET_PATH}")
    print()

    stats = parse_dataset()

    print(f"  Total samples:   {stats.total_samples}")
    print(f"  Valid:           {stats.valid_samples}")
    print(f"  Invalid:         {stats.invalid_samples}")
    print(f"  Avg input len:   {stats.avg_input_len:.0f} chars")
    print(f"  Avg output len:  {stats.avg_output_len:.0f} chars")

    if stats.errors:
        print(f"\n  First errors:")
        for err in stats.errors[:5]:
            print(f"    ⚠️  {err}")

    print()
    print("  Governor Config:")
    print(f"    QPU Time-Box:        T_max = {GOVERNOR_CONFIG['qpu_time_box_seconds']}s")
    print(f"    Adaptive Threshold:  {GOVERNOR_CONFIG['adaptive_threshold_formula']}")
    print(f"    tau_base = {GOVERNOR_CONFIG['tau_base']}, lambda = {GOVERNOR_CONFIG['lambda_rate']}")
    print()

    write_manifest(stats)
    print("  ✅ Dataset ready for cloud training phase.")


if __name__ == "__main__":
    main()
