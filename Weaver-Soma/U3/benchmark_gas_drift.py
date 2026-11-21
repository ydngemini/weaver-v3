"""Train gas-array heads and measure transfer over chronological drift batches."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from zipfile import ZipFile

import torch
from torch.nn import functional as F

from gas_summary import GAS_CLASSES, GasSummaryHead, compress
from model import WeaverSOMAU3

SEED = 1626


def read_batches(path):
    batches = {}
    with ZipFile(path) as archive:
        for batch in range(1, 11):
            suffix = f"batch{batch}.dat"
            filename = next(name for name in archive.namelist() if Path(name).name == suffix)
            samples, labels = [], []
            for line in archive.read(filename).decode().splitlines():
                if not line.strip():
                    continue
                label, *columns = line.split()
                row = [0.] * 128
                for col in columns:
                    key, value = col.split(":")
                    row[int(key) - 1] = float(value)
                samples.append(row)
                labels.append(int(label) - 1)
            batches[batch] = (torch.tensor(samples).reshape(-1, 16, 8), torch.tensor(labels))
    return batches


def fit(training, validation, dropout):
    torch.manual_seed(SEED)
    x, y = training
    model = GasSummaryHead()
    features = compress(x)
    model.center.copy_(features.mean(0))
    model.scale.copy_(features.std(0).clamp_min(.01))
    counts = torch.bincount(y, minlength=len(GAS_CLASSES))
    class_weight = counts.clamp_min(1).float().rsqrt()
    opt = torch.optim.AdamW(model.parameters(), lr=.012)
    rng = torch.Generator().manual_seed(SEED)
    best = (-1, float("-inf"))
    for epoch in range(80):
        for indices in torch.randperm(len(x), generator=rng).split(128):
            keep = None
            if dropout:
                keep = (torch.rand((len(indices), 16, 1), generator=rng) > .2).expand(-1, -1, 8)
            logits = model(x[indices], keep)["logits"]
            loss = F.cross_entropy(logits, y[indices], weight=class_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits = model(validation[0])["logits"]
            key = (int((logits.argmax(-1) == validation[1]).sum()), -float(F.cross_entropy(logits, validation[1])))
            if key > best:
                best = key
                weights = deepcopy(model.state_dict())
                selected = epoch + 1
    model.load_state_dict(weights)
    return model.eval(), selected


@torch.inference_mode()
def measure(model, batch, masked=False):
    x, y = batch
    valid = torch.ones_like(x, dtype=torch.bool)
    if masked:
        valid[:, ::4] = False
    out = model(x, valid)
    logits = out["logits"]
    torch.testing.assert_close(logits, out["sensor_contributions"].sum(1) + out["bias"])
    pred = logits.argmax(-1)
    right = pred == y
    return {"total": len(y), "correct": int(right.sum()), "scored": int(out["score_available"].sum()),
            "mean_absolute_standardized_deviation": float(out["mean_absolute_standardized_deviation"].mean()),
            "per_class_total": torch.bincount(y, minlength=len(GAS_CLASSES)).tolist(),
            "per_class_correct": torch.bincount(y[right], minlength=len(GAS_CLASSES)).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    batches = read_batches(args.dataset)
    training = tuple(torch.cat(columns) for columns in zip(*(batches[n] for n in (1, 2, 3, 4))))
    validation = batches[5]
    args.run_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, dropout in (("clean_training", False), ("sensor_dropout_training", True)):
        model, epoch = fit(training, validation, dropout)
        cases = {}
        for i in range(6, 11):
            cases[i] = {"intact": measure(model, batches[i]), "four_masked_sensors": measure(model, batches[i], True)}
        results[name] = {"selected_epoch": epoch, "batches": cases}
        torch.save({"format_version": 1, "classes": GAS_CLASSES, "weights": model.state_dict()},
                   args.run_dir / f"gas_{name}.pt")
        print(name, {n: (r["intact"]["correct"], r["four_masked_sensors"]["correct"], r["intact"]["total"])
                     for n, r in cases.items()}, flush=True)
    report = {"source_sha256": WeaverSOMAU3._file_hash(args.dataset), "training_batches": [1, 2, 3, 4],
              "validation_batch": 5, "seed": SEED, "results": results}
    (args.run_dir / "gas_drift_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
