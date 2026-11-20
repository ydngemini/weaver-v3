"""Train SOMA U3's *native* audio stem on genuine ESC-50 recordings.

Real-world small-subset experiment, NOT 50-class ESC-50 benchmark, not production.
Official folds 1-3 train, fold 4 validation, fold 5 never seen until final.
Dataset: https://github.com/karolpiczak/ESC-50 (CC BY-NC 3.0).
This run trains the U3 audio subgraph in a small-core variant; saved sensory
adapter tensors are shape-compatible with default U3 fusion_dim=128.
The existing U1/U2 core is NOT trained, loaded, or claimed restored here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from model import WeaverSOMAU3

SOURCE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
CLASSES = ("dog", "rain", "siren", "coughing", "sneezing")
BASE_KEYS = ("audio_conv.", "label_heads.audio.", "token_norm.", "position_embed.", "type_embed.audio")
SRC_SAMPLE_RATE = 44100


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 500:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SOMA-U3-research/1.0"})
            with urllib.request.urlopen(request, timeout=28) as r, open(tmp, "wb") as out:
                while True:
                    block = r.read(1024 * 512)
                    if not block:
                        break
                    out.write(block)
            if tmp.stat().st_size < 500:
                raise ValueError(f"tiny download: {url}")
            os.replace(tmp, dest)
            return
        except Exception as exc:
            error = exc
            if tmp.exists():
                tmp.unlink()
            time.sleep(attempt + 1)
    raise RuntimeError(f"Could not download {url}: {error}")


def prepare_data(root, workers):
    meta = root / "meta" / "esc50.csv"
    fetch(f"{SOURCE}/meta/esc50.csv", meta)
    rows = list(csv.DictReader(meta.open(newline="", encoding="utf-8")))
    selected = [r for r in rows if r["category"] in CLASSES]
    if not selected or len(selected) != 40 * len(CLASSES):
        raise RuntimeError("ESC-50 class/fold metadata count mismatch")
    targets = [r["filename"] for r in selected]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(fetch, f"{SOURCE}/audio/{name}", root / "audio" / name): name
                for name in targets}
        for future in as_completed(jobs):
            future.result()
    for row in selected:
        path = root / "audio" / row["filename"]
        with wave.open(str(path), "rb") as f:
            if f.getnchannels() != 1 or f.getframerate() != SRC_SAMPLE_RATE or f.getsampwidth() != 2:
                raise RuntimeError(f"unexpected ESC-50 audio format: {path}")
    return selected


def waveform(path, samples):
    with wave.open(str(path), "rb") as f:
        sig = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype("float32") / 32768.0
    # Low-pass before changing sample rate, using an inexpensive Hann-window FIR.
    # For accurate high-frequency benchmarks, upgrade to a documented polyphase resampler.
    source_len = len(sig)
    output_rate = samples / (source_len / SRC_SAMPLE_RATE)
    cutoff = min(0.45, 0.44 * (output_rate / SRC_SAMPLE_RATE))
    taps = 101
    n = np.arange(taps, dtype="float64") - (taps - 1) / 2
    kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hanning(taps)
    kernel /= kernel.sum()
    clean = np.convolve(sig, kernel.astype("float32"), mode="same")
    # Target sample times in the original waveform.
    positions = np.linspace(0, len(clean) - 1, samples, dtype="float64")
    result = np.interp(positions, np.arange(len(clean)), clean).astype("float32")
    return result


def load_audio(rows, root, samples):
    sounds = []
    labels = []
    folds = []
    for row in rows:
        sounds.append(waveform(root / "audio" / row["filename"], samples))
        labels.append(CLASSES.index(row["category"]))
        folds.append(int(row["fold"]))
    return torch.from_numpy(np.stack(sounds)), torch.tensor(labels), torch.tensor(folds)


def model_audio(model, audio):
    # Audio subgraph of the exact U3 model, no third-party or pretrained model.
    b = audio.shape[0]
    obs = torch.zeros(b, 8, 7)
    quality = torch.ones(b, 8, 3)
    context = torch.zeros(b, 8)
    model.deep_cognitive_core.train_mode = "adapter"  # bypass *random, frozen* U2 deep blocks
    out = model(obs, quality, context, modalities={
        "audio": {"values": audio, "valid": torch.ones_like(audio, dtype=torch.bool)}})
    return out["modality_logits"]["audio"][:, :len(CLASSES)]


def scores(model, x, y, ids, batch):
    model.eval()
    preds = []
    with torch.inference_mode():
        for chunk in ids.split(batch):
            logits = model_audio(model, x[chunk])
            preds.append(logits.argmax(1))
    pred = torch.cat(preds)
    truth = y[ids]
    matrix = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int64)
    for t, p in zip(truth, pred):
        matrix[t, p] += 1
    accuracy = float((pred == truth).float().mean())
    recall = (matrix.diag() / matrix.sum(1).clamp(min=1)).tolist()
    return {"n": int(len(ids)), "accuracy": accuracy, "balanced_accuracy": sum(recall)/len(recall),
            "recall_by_class": dict(zip(CLASSES, recall)), "confusion": matrix.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--samples", type=int, default=12000,
                   help="Samples per 5 second clip after filtered resampling (2400Hz default)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--download-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=2501)
    p.add_argument("--download-only", action="store_true")
    args = p.parse_args()
    if args.samples < 2048 or args.epochs < 1:
        p.error("samples >=2048; epochs >=1")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    print("DATA: downloading/verifying official ESC-50 WAV subset", flush=True)
    rows = prepare_data(args.data_dir, args.download_workers)
    print(f"DATA OK: {len(rows)} original audio files, classes={CLASSES}", flush=True)
    if args.download_only:
        return
    x, y, fold = load_audio(rows, args.data_dir, args.samples)
    print(f"DECODED: shape={list(x.shape)}, effective rate={args.samples / 5:.0f}Hz", flush=True)
    train_idx = (fold <= 3).nonzero()[:, 0]
    valid_idx = (fold == 4).nonzero()[:, 0]
    test_idx = (fold == 5).nonzero()[:, 0]
    assert (len(train_idx), len(valid_idx), len(test_idx)) == (120, 40, 40)
    model = WeaverSOMAU3(core_dim=64, depth=2, ff_dim=128,
                         fusion_dim=128, quantum_qubits=4)
    # Full U3's future audio adapter shape matches. All original base weights frozen.
    for name, param in model.named_parameters():
        param.requires_grad_(any(name.startswith(prefix) for prefix in BASE_KEYS))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.5e-3,
                            weight_decay=0.002)
    baseline = scores(model, x, y, valid_idx, args.batch_size)
    print(f"DEV BEFORE: accuracy={baseline['accuracy']:.3f}", flush=True)
    best_acc = -1.
    best = None
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(args.seed + epoch)
        order = train_idx[torch.randperm(len(train_idx), generator=generator)]
        losses = []
        for batch_ids in order.split(args.batch_size):
            xb, yb = x[batch_ids], y[batch_ids]
            logits = model_audio(model, xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        dev = scores(model, x, y, valid_idx, args.batch_size)
        entry = {"epoch": epoch, "training_loss": sum(losses)/len(losses),
                 "dev_accuracy": dev["accuracy"]}
        history.append(entry)
        print(f"EPOCH {epoch:02d} train_loss={entry['training_loss']:.4f} dev_accuracy={dev['accuracy']:.3f}",
              flush=True)
        if dev["accuracy"] > best_acc:
            best_acc = dev["accuracy"]
            best = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()
                    if any(k.startswith(prefix) for prefix in BASE_KEYS)}
    model.load_state_dict(best, strict=False)
    dev_final = scores(model, x, y, valid_idx, args.batch_size)
    heldout = scores(model, x, y, test_idx, args.batch_size)   # test only ONCE after selection
    ckpt = args.run_dir / "soma_u3_ESC50_5class_audio_adapter.pt"
    torch.save({"architecture": "SOMA-U3-audio-stem-ESC50-research-v1",
                "weights": best, "classes": CLASSES, "fusion_dim": 128,
                "samples_per_clip": args.samples, "seed": args.seed,
                "dataset": "ESC-50 (CC BY-NC 3.0), folds 1-3/4/5"}, ckpt)
    manifest = [{"filename": row["filename"], "fold": int(row["fold"]),
                 "category": row["category"], "sha256": sha256(args.data_dir/"audio"/row["filename"])}
                for row in rows]
    result = {"run_status": "COMPLETE", "synthetic_only": False,
              "real_audio": True, "stage": "U3-native-audio-stem-research",
              "dataset": "ESC-50 CC BY-NC 3.0; source github.com/karolpiczak/ESC-50",
              "classes": CLASSES,
              "folds": {"train": [1, 2, 3], "validation": [4], "test": [5]},
              "samples": {"train": 120, "validation": 40, "test": 40},
              "model_config": {"U2_core_dim": 64, "U2_depth": 2, "U2_ff_dim": 128,
                               "U3_fusion_dim": 128, "U3_quantum_qubits": 4,
                               "U2_base": "random frozen, NOT Stage4 trained"},
              "trained_modules": list(BASE_KEYS),
              "fixed_seed": args.seed, "epochs": args.epochs,
              "sample_rate_hz": args.samples / 5, "dev_before": baseline,
              "best_dev": dev_final, "heldout_test": heldout,
              "history": history, "elapsed_seconds": round(time.monotonic()-started, 2),
              "adapter_file": str(ckpt), "adapter_sha256": sha256(ckpt),
              "limitations": ["only a 5-class subset, not 50-class ESC-50 benchmark",
                              "the inherited 100M U2 core was NOT trained/loaded",
                              "quantum layer NOT trained or used to improve classifier",
                              "no live microphone or physical sensor",
                              "dataset is noncommercial; no unrestricted commercial usage"]}
    (args.run_dir / "metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    (args.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("FINAL JSON SUMMARY: "+json.dumps({"best_dev_accuracy": dev_final["accuracy"],
          "heldout_test_accuracy": heldout["accuracy"], "model_path": str(ckpt),
          "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
