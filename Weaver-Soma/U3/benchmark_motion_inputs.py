"""Fit native camera-motion and phone-IMU readouts on CPU."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import struct
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model import LEGACY_OUTPUTS, WeaverSOMAU3
from motion_inputs import IMU_ACTIVITY_CLASSES, VISION_MOTION_CLASSES, InertialActivityStem, MotionVisionStem

DIM = 64
SEED = 1819
IMU_AXES = [f"{signal}_{axis}" for signal in ("total_acc", "body_gyro") for axis in "xyz"]
TRAIN_DIGITS = "train-images-idx3-ubyte.gz"
TEST_DIGITS = "t10k-images-idx3-ubyte.gz"
FRAMES = 8


def load_digits(path):
    with gzip.open(path, "rb") as stream:
        magic, count, rows, cols = struct.unpack(">4I", stream.read(16))
        if magic != 2051 or (rows, cols) != (28, 28):
            raise ValueError("expected MNIST image IDX file")
        return np.frombuffer(stream.read(), dtype=np.uint8).reshape(count, rows, cols)


def moving_digits(digits, size, seed):
    rng = np.random.default_rng(seed)
    clips = np.zeros((size, FRAMES, 48, 48), dtype=np.uint8)
    labels = np.empty(size, dtype=np.int64)
    choice = rng.choice(len(digits), size=size // 2, replace=False)
    for i, index in enumerate(choice):
        glyph = digits[index, 4:24, 4:24]
        horizontal = i % 2 == 0
        step = int(rng.integers(1, 4))
        offset = int(rng.integers(0, 29 - step * (FRAMES - 1)))
        other = int(rng.integers(0, 29))
        for frame in range(FRAMES):
            moving = offset + step * frame
            x, y = (moving, other) if horizontal else (other, moving)
            clips[2 * i, frame, y:y + 20, x:x + 20] = glyph
        clips[2 * i + 1] = clips[2 * i, ::-1]
        labels[2 * i:2 * i + 2] = (1, 0) if horizontal else (3, 2)
    return torch.from_numpy(clips), torch.from_numpy(labels)


def vision_args(data, indices, *, corrupt=False):
    values = data[indices, :, None].float().expand(-1, -1, 3, -1, -1) / 255
    time = (torch.arange(values.shape[1], dtype=torch.float64) * .1).expand(len(values), -1)
    args = {"values": values, "timestamps": time}
    if corrupt:
        keep = torch.ones(values.shape[:2], dtype=torch.bool)
        keep[:, 2:4] = False
        args["valid"] = keep
        args["values"] = torch.where(keep[..., None, None, None], values, float("nan"))
    return args


def load_har(directory, split):
    base = directory / split
    signals = [base / "Inertial Signals" / f"{axis}_{split}.txt" for axis in IMU_AXES]
    labels = base / f"y_{split}.txt"
    subjects = base / f"subject_{split}.txt"
    values = np.stack([np.loadtxt(path, dtype=np.float32) for path in signals], axis=-1)
    return (torch.from_numpy(values), torch.from_numpy(np.loadtxt(labels, dtype=np.int64) - 1),
            torch.from_numpy(np.loadtxt(subjects, dtype=np.int64)), signals + [labels, subjects])


def summary_only(values):
    return torch.cat((values.mean(1), values.std(1, correction=0)), dim=-1)


def fit(model, training, validation, batch, *, epochs, rate):
    opt = torch.optim.AdamW(model.parameters(), lr=rate)
    generator = torch.Generator().manual_seed(SEED)
    best = (-1, float("inf"))
    for epoch in range(epochs):
        model.train()
        for indices in training[torch.randperm(len(training), generator=generator)].split(96):
            logits, target = batch(model, indices)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            logits, target = zip(*(batch(model, indices) for indices in validation.split(96)))
            logits, target = torch.cat(logits), torch.cat(target)
            key = (int((logits.argmax(-1) == target).sum()), -float(F.cross_entropy(logits, target)))
        if key > best:
            best = key
            weights = deepcopy(model.state_dict())
            best_epoch = epoch + 1
    model.load_state_dict(weights)
    return best_epoch


def metrics(pred, target, count, available=None):
    if available is None:
        available = torch.ones_like(target, dtype=torch.bool)
    correct = available & (pred == target)
    matrix = torch.bincount(count * target[available] + pred[available], minlength=count ** 2).reshape(count, count)
    f1 = (2 * matrix.diag() / (matrix.sum(0) + torch.bincount(target, minlength=count)).clamp_min(1)).mean()
    return {"windows": len(target), "scoreable": int(available.sum()), "correct": int(correct.sum()),
            "accuracy": float(correct.float().mean()), "macro_f1": float(f1), "confusion": matrix.tolist()}


def verify_integration(vision, imu, video, phone):
    torch.manual_seed(SEED)
    native = WeaverSOMAU3(fusion_dim=DIM, core_dim=64, ff_dim=128, depth=2, quantum_qubits=3,
                         vision_encoder="motion", enable_imu=True, max_modality_tokens=32).eval()
    native.motion_vision.load_state_dict(vision.state_dict(), strict=True)
    native.inertial.load_state_dict(imu.state_dict(), strict=True)
    legacy = WeaverSOMAU3(fusion_dim=DIM, core_dim=64, ff_dim=128, depth=2, quantum_qubits=3,
                         max_modality_tokens=32).eval()
    legacy.load_state_dict({key: val for key, val in native.state_dict().items() if key in legacy.state_dict()})
    obs, qual, context = torch.zeros(2, 8, 7), torch.ones(2, 8, 3), torch.zeros(2, 8)
    streams = {"vision": video, "imu": phone}
    with torch.inference_mode():
        original = legacy(obs, qual, context)
        combined = native(obs, qual, context, modalities=streams)
        torch.testing.assert_close(combined["modality_logits"]["vision"], vision(**video).logits)
        torch.testing.assert_close(combined["modality_logits"]["imu"], imu(**phone).logits)
        vision_evidence = combined["motion_evidence"]["vision"]
        reconstructed = (vision_evidence["class_velocity_contributions"].sum(-1) + vision_evidence["class_bias"])
        torch.testing.assert_close(combined["modality_logits"]["vision"], reconstructed)
        assert all(torch.equal(original[k], combined[k]) for k in LEGACY_OUTPUTS)
        for sample in streams.values():
            sample["valid"] = torch.zeros(sample["values"].shape[:2], dtype=torch.bool)
            sample["values"] = torch.full_like(sample["values"], float("nan"))
        native.multimodal_gain.fill_(.25)
        empty = native(obs, qual, context, modalities=streams)
        assert all(torch.equal(original[k], empty[k]) for k in LEGACY_OUTPUTS)
        assert not empty["sensory_token_mask"].any() and not empty["modality_logits"]
    return {"legacy_bit_identical_outputs": len(LEGACY_OUTPUTS), "missing_inputs_return_logits": False,
            "native_matches_isolated_logits": True, "vision_logit_reconstruction": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist-dir", type=Path, required=True)
    parser.add_argument("--har-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    started = perf_counter()
    train_digits = args.mnist_dir / TRAIN_DIGITS
    test_digits = args.mnist_dir / TEST_DIGITS
    tiles = load_digits(train_digits)
    # Partition by source glyph before deriving reversed clips.
    train_video, train_vlabel = moving_digits(tiles[:55000], 960, SEED)
    val_video, val_vlabel = moving_digits(tiles[55000:], 240, SEED + 1)
    dev_video = torch.cat((train_video, val_video))
    dev_vlabel = torch.cat((train_vlabel, val_vlabel))
    vision = MotionVisionStem(DIM)
    vision_epoch = fit(
        vision, torch.arange(len(train_video)), torch.arange(len(train_video), len(dev_video)),
        lambda model, ids: (model(**vision_args(dev_video, ids)).logits, dev_vlabel[ids]),
        epochs=10, rate=.03,
    )
    test_video, vlabels = moving_digits(load_digits(test_digits), 480, SEED + 2)
    video_results = {}
    with torch.inference_mode():
        for name, damaged in (("clean", False), ("two_missing_frames", True)):
            scores = [vision(**vision_args(test_video, ids, corrupt=damaged))
                      for ids in torch.arange(len(test_video)).split(64)]
            prediction = torch.cat([entry.logits.argmax(-1) for entry in scores])
            available = torch.cat([entry.evidence["score_available"] for entry in scores])
            video_results[name] = metrics(prediction, vlabels, len(VISION_MOTION_CLASSES), available)
            video_results[name]["both_directions_correct"] = int(
                (available & (prediction == vlabels)).reshape(-1, 2).all(-1).sum())
    assert torch.equal(test_video[0], test_video[1].flip(0))

    train_x, train_y, subjects, train_files = load_har(args.har_dir, "train")
    validation_people = subjects.unique()[::5]
    validation = torch.isin(subjects, validation_people)
    train_ids = (~validation).nonzero().flatten()
    val_ids = validation.nonzero().flatten()
    imu = InertialActivityStem(DIM)
    imu_epoch = fit(imu, train_ids, val_ids, lambda model, ids: (model(train_x[ids]).logits, train_y[ids]),
                    epochs=16, rate=.0015)
    summary = summary_only(train_x)
    shift = summary[train_ids].mean(0)
    scale = summary[train_ids].std(0).clamp_min(.00001)
    summary = (summary - shift) / scale
    baseline = nn.Sequential(nn.Linear(12, 40), nn.SiLU(), nn.Linear(40, len(IMU_ACTIVITY_CLASSES)))
    baseline_epoch = fit(baseline, train_ids, val_ids, lambda model, ids: (model(summary[ids]), train_y[ids]),
                         epochs=16, rate=.0015)
    test_x, ilabels, test_subjects, test_files = load_har(args.har_dir, "test")
    assert not torch.isin(test_subjects, subjects.unique()).any()
    imu_results = {}
    with torch.inference_mode():
        imu_results["unordered_mean_std_baseline"] = metrics(
            baseline((summary_only(test_x) - shift) / scale).argmax(-1), ilabels, len(IMU_ACTIVITY_CLASSES))
        for view in ("clean", "drop_gyro", "drop_middle_32_samples"):
            results = []
            for values in test_x.split(96):
                quality = torch.ones_like(values)
                if view == "drop_gyro":
                    quality[..., 3:] = 0
                elif view == "drop_middle_32_samples":
                    quality[:, 48:80, :] = 0
                packed = torch.where(quality > 0, values, float("nan"))
                results.append(imu(packed, quality=quality))
            imu_results[view] = metrics(torch.cat([r.logits.argmax(-1) for r in results]), ilabels,
                                         len(IMU_ACTIVITY_CLASSES),
                                         torch.cat([r.evidence["score_available"] for r in results]))

    provenance = {path.name: WeaverSOMAU3._file_hash(path)
                  for path in [train_digits, test_digits] + train_files + test_files}
    report = {
        "seed": SEED, "torch_version": str(torch.__version__), "hardware": "CPU",
        "wall_seconds": perf_counter() - started,
        "vision": {"classes": VISION_MOTION_CLASSES, "training_clips": len(train_video),
                   "validation_clips": len(val_video), "best_epoch": vision_epoch, "results": video_results},
        "imu": {"classes": IMU_ACTIVITY_CLASSES, "sample_rate_hz": 50, "axes": IMU_AXES,
                "training_windows": len(train_ids), "validation_windows": len(val_ids),
                "validation_people": validation_people.tolist(), "test_people": test_subjects.unique().tolist(),
                "best_epoch": imu_epoch, "baseline_epoch": baseline_epoch, "results": imu_results},
        "source_sha256": provenance,
        "native_integration": verify_integration(vision, imu, vision_args(test_video, torch.arange(2)),
                                                 {"values": test_x[:2]}),
    }
    args.run_dir.mkdir(parents=True, exist_ok=False)
    for stem, model, labels in (("vision_motion", vision, VISION_MOTION_CLASSES),
                                ("imu_activity", imu, IMU_ACTIVITY_CLASSES)):
        torch.save({"format_version": 1, "stem": stem, "fusion_dim": DIM, "classes": labels,
                    "weights": model.state_dict()}, args.run_dir / f"{stem}.pt")
    (args.run_dir / "motion_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    for stem, results in (("vision", video_results), ("imu", imu_results)):
        print(stem, {case: f"{metrics['correct']}/{metrics['windows']}" for case, metrics in results.items()})


if __name__ == "__main__":
    main()
