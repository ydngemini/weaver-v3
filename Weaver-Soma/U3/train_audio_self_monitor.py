"""Train and test a SOMA audio correctness monitor on CPU using ESC-50."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

from audio_self_monitor import AUDIO_CLASSES, AudioSelfMonitor, monitor_features
from benchmark_esc50_audio import BATCH_SIZE, DIM, fit, extract, new_stem, read_audio, read_waveforms
from train_esc50 import CLASSES, SOURCE, fetch, prepare_data, sha256

FIT_CONDITIONS = ("clean", "noise", "clipping", "packet_loss")
TEST_CONDITIONS = FIT_CONDITIONS + ("echo_bandlimit",)
UNKNOWN_CLASSES = ("clock_tick", "crackling_fire", "keyboard_typing", "rooster", "sea_waves")
BASE_SEED = 2501
MONITOR_SEED = 841
TARGET_COVERAGE = .70


def distort(waveforms, condition, *, noise_seed):
    valid = torch.ones_like(waveforms, dtype=torch.bool)
    if condition == "noise":
        gen = torch.Generator().manual_seed(noise_seed)
        noise = torch.randn(waveforms.shape, generator=gen)
        strength = waveforms.square().mean(dim=1, keepdim=True).sqrt()
        waveforms = (waveforms + 1.25 * strength * noise).clamp(-1, 1)
    elif condition == "clipping":
        waveforms = (6 * waveforms).clamp(-1, 1)
    elif condition == "packet_loss":
        for start in (8000, 24000, 40000, 56000, 72000):
            valid[:, start:start + 6400] = False
        waveforms = waveforms.masked_fill(~valid, 0)
    elif condition == "echo_bandlimit":
        echo = F.pad(waveforms[:, :-1920], (1920, 0))
        combined = (waveforms + .65 * echo).clamp(-1, 1)
        waveforms = F.avg_pool1d(combined.unsqueeze(1), 15, stride=1, padding=7)[:, 0]
    elif condition != "clean":
        raise ValueError(f"unknown replay condition: {condition}")
    return waveforms, valid


@torch.inference_mode()
def predict_recordings(stem, waveforms, labels, conditions, *, noise_seed):
    stem.eval()
    rows = []
    for condition in conditions:
        waveforms_variant, masks = distort(waveforms, condition, noise_seed=noise_seed)
        per_condition = []
        for waveform, mask in zip(waveforms_variant.split(BATCH_SIZE), masks.split(BATCH_SIZE)):
            tokens, valid = stem(waveform, mask)
            logits = stem.classify(tokens, valid)[:, :len(AUDIO_CLASSES)]
            features = monitor_features(logits, waveform, mask, tokens, valid)
            per_condition.append((logits, features))
        all_logits = torch.cat([item[0] for item in per_condition])
        all_features = torch.cat([item[1] for item in per_condition])
        rows.append((condition, all_logits, all_features, labels))
    return rows


def train_monitor(rows):
    features = torch.cat([row[2] for row in rows])
    target = torch.cat([row[1].argmax(1) == row[3] for row in rows]).float()
    torch.manual_seed(MONITOR_SEED)
    monitor = AudioSelfMonitor()
    monitor.center.copy_(features.mean(0))
    monitor.scale.copy_(features.std(0, unbiased=False).clamp_min(.025))
    optimizer = torch.optim.LBFGS(monitor.parameters(), max_iter=80, line_search_fn="strong_wolfe")

    def objective():
        optimizer.zero_grad(set_to_none=True)
        nll = F.binary_cross_entropy_with_logits(monitor.log_odds(features), target)
        loss = nll + .025 * monitor.linear.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(objective)
    return monitor.eval()


def fit_calibration(rows, monitor):
    logits = torch.cat([row[1] for row in rows])
    features = torch.cat([row[2] for row in rows])
    labels = torch.cat([row[3] for row in rows])
    grid = torch.logspace(-1, 1, 101)
    all_logits = logits.unsqueeze(0) / grid[:, None, None]
    losses = -all_logits.log_softmax(2)[:, torch.arange(len(labels)), labels].mean(1)
    temperature = float(grid[losses.argmin()])
    with torch.no_grad():
        # Correct the intercept on the fixed deployment classifier's dev predictions.
        offsets = torch.linspace(-3, 3, 121)
        scores = monitor.log_odds(features)
        correctness = (logits.argmax(1) == labels).float()
        terms = scores.unsqueeze(0) + offsets.unsqueeze(1)
        costs = F.binary_cross_entropy_with_logits(
            terms, correctness.expand_as(terms), reduction="none",
        ).mean(dim=1)
        monitor.linear.bias.add_(offsets[costs.argmin()])
    return temperature


@torch.inference_mode()
def confidence_scores(logits, features, monitor, temperature):
    return {
        "raw_softmax": logits.softmax(dim=1).amax(dim=1).numpy(),
        "temperature_scaled": (logits / temperature).softmax(dim=1).amax(dim=1).numpy(),
        "learned_monitor": monitor.log_odds(features).sigmoid().numpy(),
    }


def metrics_at(scores, correct, cutoff):
    accept = scores >= cutoff
    order = np.argsort(-scores, kind="stable")
    risk = np.cumsum(~correct[order]) / np.arange(1, len(scores) + 1)
    k = int(np.ceil(TARGET_COVERAGE * len(scores)))
    return {
        "brier": float(np.mean((scores - correct) ** 2)),
        "auroc_correctness": float(roc_auc_score(correct, scores))
        if np.unique(correct).size == 2 else None,
        "aurc": float(risk.mean()),
        "risk_at_70pct_ranked_coverage": float(risk[k - 1]),
        "fixed_threshold_coverage": float(accept.mean()),
        "fixed_threshold_risk": float(np.mean(~correct[accept])) if accept.any() else None,
        "accepted": int(accept.sum()),
        "errors_accepted": int((accept & ~correct).sum()),
        "mean_confidence": float(scores.mean()),
    }


def measure(rows, monitor, temperature, thresholds):
    report = {}
    pooled_scores = {key: [] for key in thresholds}
    pooled_correct = []
    for name, logits, features, labels in rows:
        scores = confidence_scores(logits, features, monitor, temperature)
        correct = (logits.argmax(1) == labels).numpy()
        report[name] = {"n": len(labels), "classifier_accuracy": float(correct.mean()),
                        "confidence": {key: metrics_at(value, correct, thresholds[key])
                                       for key, value in scores.items()}}
        for key, value in scores.items():
            pooled_scores[key].append(value)
        pooled_correct.append(correct)
    correct = np.concatenate(pooled_correct)
    report["in_scope_pooled"] = {"n": len(correct), "classifier_accuracy": float(correct.mean()),
                                 "confidence": {key: metrics_at(np.concatenate(values), correct, thresholds[key])
                                                for key, values in pooled_scores.items()}}
    return report


def off_label_rows(root, known):
    with (root / "meta" / "esc50.csv").open(newline="", encoding="utf-8") as metadata:
        rows = [row for row in csv.DictReader(metadata)
                if row["fold"] == "5" and row["category"] in UNKNOWN_CLASSES]
    if len(rows) != 40 or {r["src_file"] for r in rows} & {r["src_file"] for r in known}:
        raise ValueError("unexpected or overlapping off-label ESC-50 sources")
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(fetch, f"{SOURCE}/audio/{row['filename']}",
                            root / "audio" / row["filename"]) for row in rows]
        for job in jobs:
            job.result()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audio-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    if AUDIO_CLASSES != CLASSES:
        raise RuntimeError("class order differs from the audio benchmark")
    released = torch.load(args.audio_checkpoint, map_location="cpu", weights_only=True)
    if (released["architecture"] != "SOMA-U3-native-logmel-audio-v1"
            or tuple(released["classes"]) != AUDIO_CLASSES or released["fusion_dim"] != DIM
            or released["sample_rate"] != 16000 or released["seed"] != BASE_SEED):
        raise ValueError("expected the SOMA audio5 seed 2501 checkpoint")
    known = prepare_data(args.data_dir, 8)
    source_folds = {}
    for row in known:
        fold = source_folds.setdefault(row["src_file"], row["fold"])
        if fold != row["fold"]:
            raise ValueError("an original recording crosses folds")
    fit_rows = [row for row in known if int(row["fold"]) <= 3]
    waveform, labels, folds = read_audio(args.data_dir, fit_rows)
    features = extract(new_stem(BASE_SEED), waveform)
    out_of_fold = []
    for fold in (1, 2, 3):
        train = folds != fold
        stem, _ = fit(BASE_SEED, features[train], labels[train], epochs=24, augment=False)
        out_of_fold.extend(predict_recordings(
            stem, waveform[~train], labels[~train], FIT_CONDITIONS, noise_seed=3800 + fold,
        ))
        print(f"FIT: generated independent classifier errors for fold {fold}", flush=True)
    monitor = train_monitor(out_of_fold)
    frozen = new_stem(BASE_SEED).eval()
    frozen.load_state_dict(released["weights"], strict=True)
    cal_rows = [row for row in known if row["fold"] == "4"]
    cal_wave, cal_y, _ = read_audio(args.data_dir, cal_rows)
    calibration = predict_recordings(frozen, cal_wave, cal_y, FIT_CONDITIONS, noise_seed=3804)
    temperature = fit_calibration(calibration, monitor)
    dev_scores = confidence_scores(
        torch.cat([row[1] for row in calibration]), torch.cat([row[2] for row in calibration]),
        monitor, temperature,
    )
    thresholds = {key: float(np.quantile(value, 1 - TARGET_COVERAGE, method="lower"))
                  for key, value in dev_scores.items()}
    monitor.threshold.fill_(thresholds["learned_monitor"])
    print(f"LOCKED BEFORE FOLD 5: {json.dumps(thresholds)}", flush=True)

    known_test = [row for row in known if row["fold"] == "5"]
    test_wave, test_y, _ = read_audio(args.data_dir, known_test)
    test_rows = predict_recordings(frozen, test_wave, test_y, TEST_CONDITIONS, noise_seed=3805)
    result = measure(test_rows, monitor, temperature, thresholds)
    unknown = off_label_rows(args.data_dir, known)
    unknown_wave = read_waveforms(args.data_dir, unknown)
    _, logits, stats, _ = predict_recordings(
        frozen, unknown_wave, torch.full((len(unknown),), -1), ("clean",), noise_seed=3806,
    )[0]
    unknown_scores = confidence_scores(logits, stats, monitor, temperature)
    meta = {
        "architecture": "SOMA-U3-audio-self-monitor-v1",
        "classes": AUDIO_CLASSES, "off_label_classes": UNKNOWN_CLASSES,
        "unique_train_clips": len(fit_rows), "train_replays": sum(len(row[3]) for row in out_of_fold),
        "unique_calibration_clips": len(cal_rows), "calibration_replays": sum(len(row[3]) for row in calibration),
        "unique_in_scope_test_clips": len(known_test), "unique_off_label_test_clips": len(unknown),
        "fit_folds": [1, 2, 3], "calibration_fold": 4, "test_fold": 5,
        "test_fold_previously_used_for_audio_classification": True,
        "known_fit_conditions": FIT_CONDITIONS, "held_out_corruption": "echo_bandlimit",
        "noise_seeds": {"fit_by_fold": [3801, 3802, 3803], "calibration": 3804, "test": 3805},
        "temperature": temperature, "thresholds": thresholds, "target_dev_coverage": TARGET_COVERAGE,
        "audio_checkpoint_sha256": sha256(args.audio_checkpoint),
    }
    checkpoint = args.run_dir / "soma_u3_audio_self_monitor.pt"
    torch.save({
        "architecture": meta["architecture"], "classes": AUDIO_CLASSES, "sample_rate": 16000,
        "fusion_dim": DIM, "audio_checkpoint_sha256": meta["audio_checkpoint_sha256"],
        "audio_weights": frozen.state_dict(), "monitor_weights": monitor.state_dict(),
    }, checkpoint)
    report = {**meta, "monitor_parameters": sum(p.numel() for p in monitor.parameters()),
              "checkpoint_sha256": sha256(checkpoint), "test": result,
              "off_label_false_accept_rate": {key: float(np.mean(value >= thresholds[key]))
                                              for key, value in unknown_scores.items()}}
    (args.run_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"in_scope": result["in_scope_pooled"],
                      "off_label_false_accept_rate": report["off_label_false_accept_rate"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
