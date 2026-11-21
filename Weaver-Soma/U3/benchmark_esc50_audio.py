"""Select and benchmark SOMA's native log-mel encoder on the five ESC-50 classes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
import wave

import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch.nn import functional as F

from audio_spectral import SAMPLE_RATE, SpectralAudioStem, masked_statistics
from model import WeaverSOMAU3
from train_esc50 import CLASSES, prepare_data, sha256


DIM = 128
SEEDS = (2501, 2502, 2503)
EPOCHS = (8, 16, 24)
BATCH_SIZE = 16
LABELS = list(range(len(CLASSES)))


def new_stem(seed):
    torch.manual_seed(seed)
    return SpectralAudioStem(DIM, WeaverSOMAU3.LABELS["audio"])


def read_audio(folder, rows):
    clips = []
    for row in rows:
        with wave.open(str(folder / "audio" / row["filename"]), "rb") as source:
            pcm = source.readframes(source.getnframes())
        waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        samples = resample_poly(waveform, 160, 441)
        if samples.shape != (5 * SAMPLE_RATE,):
            raise ValueError(f"expected a five-second clip: {row['filename']}")
        clips.append(samples)
    waveform = torch.from_numpy(np.stack(clips))
    labels = torch.tensor([CLASSES.index(row["category"]) for row in rows])
    folds = torch.tensor([int(row["fold"]) for row in rows])
    return waveform, labels, folds


def extract(stem, waveforms):
    with torch.no_grad():
        return torch.cat([
            stem.frontend(batch, torch.ones_like(batch, dtype=torch.bool))[0]
            for batch in waveforms.split(BATCH_SIZE)
        ])


def mask_features(features):
    # Masks simulate short missing intervals and narrow missing frequency bands.
    batch, frames, bands = features.shape
    start = torch.randint(frames - 18 + 1, (batch, 1))
    frame = torch.arange(frames)[None, :]
    drop_time = (frame >= start) & (frame < start + 18)
    start = torch.randint(bands - 5 + 1, (batch, 1))
    band = torch.arange(bands)[None, :]
    drop_band = (band >= start) & (band < start + 5)
    return features.masked_fill(drop_time[:, :, None] | drop_band[:, None, :], 0)


def logits(stem, features):
    valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
    return stem.classify(*stem.encode_features(features, valid))[:, :len(CLASSES)]


def evaluate(truth, prediction):
    matrix = confusion_matrix(truth, prediction, labels=LABELS)
    recall = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    return {
        "n": int(matrix.sum()),
        "accuracy": float(matrix.trace() / matrix.sum()),
        "macro_f1": float(f1_score(truth, prediction, labels=LABELS, average="macro",
                                  zero_division=0)),
        "recall_by_class": dict(zip(CLASSES, recall.tolist())),
        "confusion": matrix.tolist(),
    }


def predict(stem, features):
    stem.eval()
    with torch.inference_mode():
        return torch.cat([logits(stem, batch).argmax(dim=1) for batch in features.split(BATCH_SIZE)])


def fit(seed, features, labels, *, epochs, augment, validation=None):
    stem = new_stem(seed)
    optimizer = torch.optim.AdamW(stem.parameters(), lr=0.002, weight_decay=0.002)
    scores = {}
    for epoch in range(1, epochs + 1):
        stem.train()
        for ids in torch.randperm(len(labels)).split(BATCH_SIZE):
            batch = features[ids]
            if augment:
                batch = mask_features(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(logits(stem, batch), labels[ids])
            loss.backward()
            optimizer.step()
        if validation is not None and epoch in EPOCHS:
            val_features, val_labels = validation
            scores[epoch] = evaluate(val_labels, predict(stem, val_features))
    return stem, scores


def select(features, labels, folds):
    candidates = {(augment, epoch): [] for augment in (False, True) for epoch in EPOCHS}
    for augment in (False, True):
        for fold in range(1, 5):
            train, valid = folds != fold, folds == fold
            for seed in SEEDS:
                _, scores = fit(
                    seed, features[train], labels[train], epochs=max(EPOCHS), augment=augment,
                    validation=(features[valid], labels[valid]),
                )
                for epoch, score in scores.items():
                    candidates[augment, epoch].append({
                        "fold": fold, "seed": seed, "accuracy": score["accuracy"],
                        "macro_f1": score["macro_f1"],
                    })
            print(f"DEV: fold {fold}, masking={augment} finished", flush=True)
    summaries = [
        {"masking": augment, "epochs": epoch, "runs": runs,
         "mean_macro_f1": float(np.mean([r["macro_f1"] for r in runs])),
         "mean_accuracy": float(np.mean([r["accuracy"] for r in runs]))}
        for (augment, epoch), runs in candidates.items()
    ]
    winner = max(summaries, key=lambda r: (r["mean_macro_f1"], -r["epochs"], not r["masking"]))
    return winner, summaries


def mfcc_stats(features):
    cepstra = torch.from_numpy(dct(features.numpy(), norm="ortho", axis=-1)[..., :20].copy())
    delta = cepstra[:, 1:] - cepstra[:, :-1]
    values = torch.cat([
        masked_statistics(t, torch.ones(t.shape[:2], dtype=torch.bool)) for t in (cepstra, delta)
    ], dim=1)
    return values.numpy()


def checked_manifest(root, rows, expected):
    manifest = [
        {"filename": row["filename"], "category": row["category"], "fold": int(row["fold"]),
         "sha256": sha256(root / "audio" / row["filename"])}
        for row in rows
    ]
    counts = Counter((row["category"], row["fold"]) for row in manifest)
    if len(counts) != 25 or set(counts.values()) != {8}:
        raise ValueError("expected eight clips per class in each official fold")
    sources = {}
    for row in rows:
        fold = sources.setdefault(row["src_file"], row["fold"])
        if fold != row["fold"]:
            raise ValueError("a source recording crosses folds")
    if expected is not None:
        original = json.loads(expected.read_text())
        if sorted(original, key=lambda row: row["filename"]) != sorted(
            manifest, key=lambda row: row["filename"]
        ):
            raise ValueError("dataset manifest differs from the reference run")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--download-workers", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1 or args.download_workers < 1:
        parser.error("thread and worker counts must be positive")
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    rows = prepare_data(args.data_dir, args.download_workers)
    manifest = checked_manifest(args.data_dir, rows, args.expected_manifest)
    print(f"DATA: {len(manifest)} clips verified, with source-disjoint folds", flush=True)
    dev_rows = [row for row in rows if row["fold"] != "5"]
    dev_audio, dev_labels, dev_folds = read_audio(args.data_dir, dev_rows)
    dev_features = extract(new_stem(SEEDS[0]), dev_audio)
    winner, dev_results = select(dev_features, dev_labels, dev_folds)
    selected = {k: v for k, v in winner.items() if k != "runs"}
    print(f"LOCKED BEFORE FOLD 5: {json.dumps(selected)}", flush=True)
    (args.run_dir / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")

    # Folds 1–3 match the original run's training set size and exact membership.
    train = dev_folds <= 3
    fitted = [
        fit(seed, dev_features[train], dev_labels[train],
            epochs=winner["epochs"], augment=winner["masking"])[0]
        for seed in SEEDS
    ]
    baseline = make_pipeline(StandardScaler(), SVC(C=3))
    baseline.fit(mfcc_stats(dev_features[train]), dev_labels[train].numpy())
    test_rows = [row for row in rows if row["fold"] == "5"]
    test_audio, test_labels, _ = read_audio(args.data_dir, test_rows)
    test_features = extract(fitted[0], test_audio)
    fold5 = [
        {"seed": seed, **evaluate(test_labels, predict(stem, test_features))}
        for seed, stem in zip(SEEDS, fitted)
    ]
    mfcc = evaluate(test_labels, baseline.predict(mfcc_stats(test_features)))
    checkpoint = args.run_dir / "soma_u3_audio5_logmel.pt"
    torch.save({
        "architecture": "SOMA-U3-native-logmel-audio-v1", "fusion_dim": DIM,
        "classes": CLASSES, "sample_rate": SAMPLE_RATE, "seed": SEEDS[0],
        "encoder": "logmel", "weights": fitted[0].state_dict(), "selection": selected,
    }, checkpoint)
    results = {
        "classes": CLASSES, "sample_rate": SAMPLE_RATE,
        "dataset": "ESC-50 five-class subset, CC BY-NC 3.0",
        "reference_manifest_matched": args.expected_manifest is not None,
        "selection_folds": [1, 2, 3, 4], "final_fit_folds": [1, 2, 3],
        "evaluation_fold": 5, "evaluation_fold_previously_benchmarked": True,
        "selection": selected, "development_candidates": dev_results,
        "primary_seed": SEEDS[0], "fold5_by_seed": fold5, "mfcc_rbf_fold5": mfcc,
        "adapter": checkpoint.name, "adapter_sha256": sha256(checkpoint),
        "trainable_parameters": sum(p.numel() for p in fitted[0].parameters()),
        "seconds": round(time.monotonic() - started, 2),
    }
    (args.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.run_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print("FOLD 5:", [round(row["accuracy"], 4) for row in fold5], "MFCC/RBF:", mfcc["accuracy"])


if __name__ == "__main__":
    main()
