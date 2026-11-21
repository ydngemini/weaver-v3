"""Measure warm CPU inference and record tensor digests for output parity."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

import torch

SEED = 4481


def tensor_digests(value, prefix=""):
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous().numpy()
        header = repr((str(data.dtype), data.shape)).encode()
        return {prefix: sha256(header + data.tobytes()).hexdigest()}
    result = {}
    for name, child in value.items():
        result.update(tensor_digests(child, f"{prefix}/{name}"))
    return result


def measure(fn, repetitions):
    for _ in range(10):
        fn()
    elapsed = []
    for _ in range(repetitions):
        begin = perf_counter_ns()
        latest = fn()
        elapsed.append((perf_counter_ns() - begin) / 1e6)
    p95 = sorted(elapsed)[(95 * len(elapsed) + 99) // 100 - 1]
    return {"p50_ms": median(elapsed), "p95_ms": p95, "tensors": tensor_digests(latest)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=80)
    p.add_argument("--optional-diagnostics-off", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(args.source_dir.resolve()))
    from adaptive_inputs import select_tokens
    from model import WeaverSOMAU3

    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    results = {}
    with torch.inference_mode():
        for width in (128, 2048, 8192):
            tokens = torch.randn(4, width, 64)
            valid = torch.rand(4, width) > .2
            valid[0].zero_()
            valid[1, :width // 2].zero_()
            salience = torch.randint(0, 12, (4, width)).float() / 12

            def route():
                r = select_tokens(tokens, valid, salience, 48)
                return {"tokens": r.tokens, "mask": r.valid, "evidence": r.evidence}

            results[f"router_{width}"] = measure(route, args.repetitions)

        inputs = (torch.randn(1, 8, 7), torch.ones(1, 8, 3), torch.zeros(1, 8))
        voice = {"audio": {"values": torch.randn(1, 16000) * .1},
                 "text": {"values": torch.randint(0, 128, (1, 80))}}
        mixed = {**voice, "rf": {"values": torch.randn(1, 2048, 8)},
                 "touch": {"values": torch.randn(1, 64, 6)},
                 "vision": {"values": torch.rand(1, 8, 3, 48, 48),
                            "timestamps": torch.arange(8).unsqueeze(0) / 10}}
        torch.manual_seed(SEED)
        opts = dict(fusion_dim=64, vision_encoder="motion", audio_encoder="logmel",
                    enable_adaptive_inputs=True, max_modality_tokens=48)
        model = WeaverSOMAU3(core_dim=64, ff_dim=128, depth=2, **opts).eval()
        results["voice"] = measure(lambda: model(*inputs, modalities=voice), args.repetitions)
        results["mixed_zero_gain"] = measure(lambda: model(*inputs, modalities=mixed), args.repetitions)
        if args.optional_diagnostics_off:
            results["mixed_zero_gain_diagnostics_off"] = measure(
                lambda: model(*inputs, modalities=mixed, include_quantum_diagnostics=False), args.repetitions)
        model.multimodal_gain.fill_(.25)
        model.quantum_gain.fill_(.05)
        results["mixed_nonzero_gains"] = measure(lambda: model(*inputs, modalities=mixed), args.repetitions)
        if args.optional_diagnostics_off:
            results["mixed_nonzero_gains_diagnostics_off"] = measure(
                lambda: model(*inputs, modalities=mixed, include_quantum_diagnostics=False), args.repetitions)
        torch.manual_seed(SEED)
        full = WeaverSOMAU3(**opts).eval()
        results["default_body_mixed"] = measure(lambda: full(*inputs, modalities=mixed), args.repetitions)
    report = {"seed": SEED, "torch": str(torch.__version__), "cpu_threads": 1,
              "repetitions": args.repetitions, "results": results}
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
