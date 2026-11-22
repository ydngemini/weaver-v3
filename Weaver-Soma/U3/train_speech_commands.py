"""Train and compare native SOMA streaming keyword heads on Speech Commands v0.02."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import file_digest, sha256
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import resource
import tarfile
from time import perf_counter
import wave

import numpy as np
import torch
from torch.nn import functional as F

from native_speech_commands import (
    COMMANDS, LABELS, SAMPLE_RATE, STRIDE, WINDOW, CommandStream, KeywordRecognizer,
)


SEED = 6207
UNKNOWN_LIMIT = .10
SPEAKER = re.compile(r"([0-9a-f]{8})_nohash_")


@dataclass
class Recording:
    name: str
    speaker: str | None
    label: int
    samples: np.ndarray


def _wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError("corpus clip is not mono 16 kHz PCM16")
        samples = np.frombuffer(source.readframes(SAMPLE_RATE), dtype="<i2")
    return np.pad(samples, (0, SAMPLE_RATE - len(samples)))


def _official_lists(path: Path) -> dict[str, set[str]]:
    files = {"validation_list.txt", "testing_list.txt"}
    result = {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name).name
            if name in files:
                result[name] = set(archive.extractfile(member).read().decode().splitlines())
                if result.keys() == files:
                    return result
    raise ValueError("training archive is missing the official split lists")


def _train_validation(path: Path) -> dict[str, list[Recording]]:
    manifests = _official_lists(path)
    result = {"train": [], "validation": []}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if (not member.isfile() or len(parts) != 2
                    or not parts[1].endswith(".wav") or parts[0].startswith("_")):
                continue
            name = "/".join(parts)
            if name in manifests["testing_list.txt"]:
                continue
            split = "validation" if name in manifests["validation_list.txt"] else "train"
            label = COMMANDS.index(parts[0]) if parts[0] in COMMANDS else LABELS.index("unknown")
            fraction = int.from_bytes(sha256(name.encode()).digest()[:4], "big") / 2**32
            if split == "train" and fraction >= (.45 if label < len(COMMANDS) else .07):
                continue
            if split == "validation" and label == LABELS.index("unknown") and fraction >= .20:
                continue
            speaker = SPEAKER.search(parts[1])
            if speaker is None:
                raise ValueError(f"missing speaker ID in {name}")
            result[split].append(Recording(name, speaker[1], label,
                                           _wav(archive.extractfile(member).read())))
    for split in result:
        result[split].sort(key=lambda row: row.name)
    return result


def _test(path: Path) -> list[Recording]:
    events = []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) != 2 or not parts[1].endswith(".wav"):
                continue
            name = "/".join(parts)
            label = parts[0].strip("_")
            speaker = SPEAKER.search(parts[1])
            if speaker is None and label != "silence":
                raise ValueError(f"missing speaker ID in {name}")
            events.append(Recording(name, speaker[1] if speaker else None, LABELS.index(label),
                                    _wav(archive.extractfile(member).read())))
    return sorted(events, key=lambda row: row.name)


def _verify_speakers(folds: dict[str, list[Recording]]) -> dict[str, int]:
    speakers = {}
    owners = {}
    for split, rows in folds.items():
        unique = {row.speaker for row in rows if row.speaker}
        for speaker in unique:
            if owners.setdefault(speaker, split) != split:
                raise ValueError(f"speaker {speaker} appears across official partitions")
        speakers[split] = len(unique)
    return speakers


def _artificial_quiet(rows: list[Recording], count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for i in range(count):
        gain = 10 ** rng.uniform(-4.4, -1.7)
        samples = (rng.standard_normal(SAMPLE_RATE) * gain * 32768).astype(np.int16)
        rows.append(Recording(f"generated-quiet/{i}", None, LABELS.index("silence"), samples))


def _prepare(rows: list[Recording], frontend: KeywordRecognizer, seed: int):
    rng = np.random.default_rng(seed)
    acoustic, noise_10db = [], []
    for start in range(0, len(rows), 128):
        selected = rows[start:start + 128]
        pcm = np.stack([row.samples for row in selected]).astype(np.float32) / 32768
        rms = np.sqrt(np.mean(pcm ** 2, axis=1, keepdims=True))
        white = rng.standard_normal(pcm.shape, dtype=np.float32)
        white *= rms / (np.sqrt(np.mean(white ** 2, axis=1, keepdims=True)) * math.sqrt(10))
        # Mix spoken clips only. Preserve the independent test background recordings.
        silence = np.array([row.label == LABELS.index("silence") for row in selected])
        white[silence] = 0
        with torch.no_grad():
            acoustic.append(frontend.features(torch.from_numpy(pcm)))
            noise_10db.append(frontend.features(torch.from_numpy(np.clip(pcm + white, -1, 1))))
    return torch.cat(acoustic), torch.cat(noise_10db), torch.tensor([row.label for row in rows])


@torch.no_grad()
def _predict(model, features):
    model.eval()
    return torch.cat([model(batch).softmax(dim=-1) for batch in features.split(128)])


def _macro(probabilities, labels):
    predicted = probabilities.argmax(dim=-1)
    return float(torch.stack([(predicted[labels == i] == i).float().mean()
                              for i in range(len(LABELS))]).mean())


def _threshold(probabilities, labels):
    confidence, best = probabilities.max(dim=-1)
    unsupported = labels >= len(COMMANDS)
    candidates = confidence[unsupported & (best < len(COMMANDS))].sort(descending=True).values
    budget = math.floor(int(unsupported.sum()) * UNKNOWN_LIMIT)
    return math.nextafter(float(candidates[budget]), math.inf) if len(candidates) > budget else 0.


def _score(probabilities, labels, cutoff):
    confidence, best = probabilities.max(dim=-1)
    known = labels < len(COMMANDS)
    accept = (best < len(COMMANDS)) & (confidence.double() >= cutoff)
    correct = best == labels
    by_label = {}
    for index, label in enumerate(LABELS):
        mask = labels == index
        by_label[label] = {"clips": int(mask.sum()), "top_label_correct": int((mask & correct).sum()),
                           "accepted_as_command": int((mask & accept).sum())}
    return {
        "macro_top_label_accuracy": _macro(probabilities, labels),
        "known_top_label_accuracy": float(correct[known].float().mean()),
        "known_coverage": float(accept[known].float().mean()),
        "known_correct_and_accepted": float((correct & accept)[known].float().mean()),
        "unsupported_false_acceptance": float(accept[~known].float().mean()),
        "by_label": by_label,
    }, best, confidence, accept


def _prefix_scores(model, features, labels, cutoff):
    results = {}
    for ms in (250, 500, 750):
        frames = (ms * SAMPLE_RATE // 1000 - WINDOW) // STRIDE + 1
        scores, _, _, _ = _score(_predict(model, features[:, :frames]), labels, cutoff)
        results[str(ms)] = {name: scores[name] for name in
                            ("known_coverage", "known_correct_and_accepted",
                             "unsupported_false_acceptance")}
    return results


def _fit(data, temporal: bool, epochs: int):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    model = KeywordRecognizer(temporal)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
    clean, noisy, labels = data["train"]
    weight = torch.bincount(labels).float().pow(-.5)
    best = float("-inf")
    history = []
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.
        for index in torch.randperm(len(labels), generator=rng).split(128):
            mix = (torch.rand(len(index), generator=rng) < .30)[:, None, None]
            batch = torch.where(mix, noisy[index], clean[index])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch), labels[index], weight=weight)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(index)
        valid = _macro(_predict(model, data["validation"][0]), data["validation"][2])
        history.append({"epoch": epoch, "train_loss": loss_total / len(labels), "validation_macro": valid})
        if valid > best:
            best = valid
            best_weights = {name: state.detach().clone() for name, state in model.state_dict().items()}
            best_epoch = epoch
        print(f"temporal={temporal} epoch={epoch:d} macro_validation={valid:.4f}", flush=True)
    model.load_state_dict(best_weights)
    return model, {"history": history, "selected_epoch": best_epoch,
                   "train_and_validation_seconds": perf_counter() - started}


def _stream_timing(model, clips):
    timings = []
    max_error = 0.
    torch.set_num_threads(1)
    for index in np.linspace(0, len(clips) - 1, min(41, len(clips)), dtype=int):
        row = clips[index]
        stream = CommandStream(model)
        raw = row.samples.astype("<i2").tobytes()
        for offset in range(0, len(raw), 2 * STRIDE):
            start = perf_counter()
            result = stream.push(raw[offset:offset + 2 * STRIDE])
            duration = perf_counter() - start
            if index and result is not None:
                timings.append(1000 * duration)
        with torch.no_grad():
            pcm = torch.from_numpy(row.samples.astype(np.float32) / 32768)[None, :]
            offline = model(model.features(pcm))[0]
        max_error = max(max_error, float((result - offline).abs().max()))
    torch.set_num_threads(2)
    return {"warm_scored_chunks": len(timings), "p50_ms": float(np.quantile(timings, .5)),
            "p95_ms": float(np.quantile(timings, .95)), "max_offline_logit_difference": max_error}


def run(train: Path, test: Path, out: Path, epochs: int) -> dict:
    torch.set_num_threads(2)
    folds = {**_train_validation(train), "test": _test(test)}
    speakers = _verify_speakers(folds)
    _artificial_quiet(folds["train"], 1200, SEED)
    _artificial_quiet(folds["validation"], 400, SEED + 1)
    if any(len({row.label for row in rows}) != len(LABELS) for rows in folds.values()):
        raise ValueError("each split must contain all keyword and reject labels")
    counts = {key: dict(Counter(LABELS[row.label] for row in rows)) for key, rows in folds.items()}
    print(f"Verified split speakers and counts: {speakers} {counts}", flush=True)
    frontend = KeywordRecognizer(False)
    data = {key: _prepare(rows, frontend, SEED + 2 + index)
            for index, (key, rows) in enumerate(folds.items())}
    hashes = {}
    for path in (train, test):
        with path.open("rb") as stream:
            hashes[path.name] = file_digest(stream, "sha256").hexdigest()
    results = {"speaker_counts": speakers, "class_counts": counts, "archive_sha256": hashes,
               "validation_unsupported_false_acceptance_limit": UNKNOWN_LIMIT, "epochs": epochs,
               "seed": SEED, "torch_version": torch.__version__, "models": {}}
    out.mkdir(parents=True, exist_ok=False)
    for name, temporal in (("pooling_baseline", False), ("soma_recurrent", True)):
        model, training = _fit(data, temporal, epochs)
        valid = _predict(model, data["validation"][0])
        cutoff = _threshold(valid, data["validation"][2])
        metrics = {**training, "confidence_threshold": cutoff,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "validation": _score(valid, data["validation"][2], cutoff)[0]}
        checkpoint = out / f"{name}.pt"
        torch.save({"version": 1, "state_dict": model.state_dict(), "temporal": temporal,
                    "labels": LABELS, "threshold": cutoff}, checkpoint)
        metrics["checkpoint_bytes"] = checkpoint.stat().st_size
        with (out / f"{name}_test_decisions.jsonl").open("w", encoding="utf-8") as trace:
            for index, condition in enumerate(("clean", "white_noise_10db")):
                posterior = _predict(model, data["test"][index])
                metrics[condition], best, confidence, accept = _score(posterior, data["test"][2], cutoff)
                for row, guess, conf, accepted in zip(folds["test"], best, confidence, accept):
                    trace.write(json.dumps({"clip": row.name, "condition": condition,
                                            "expected": LABELS[row.label], "top_label": LABELS[int(guess)],
                                            "confidence": float(conf), "accept": bool(accepted)}) + "\n")
        metrics["streaming"] = _stream_timing(model, folds["test"])
        metrics["clean_prefixes_ms"] = _prefix_scores(model, data["test"][0], data["test"][2], cutoff)
        if metrics["streaming"]["max_offline_logit_difference"] >= 1e-4:
            raise RuntimeError("streaming inference diverged from offline scoring")
        results["models"][name] = metrics
    results["peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-archive", required=True, type=Path)
    parser.add_argument("--test-archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=18)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    run(args.train_archive, args.test_archive, args.out, args.epochs)


if __name__ == "__main__":
    main()
