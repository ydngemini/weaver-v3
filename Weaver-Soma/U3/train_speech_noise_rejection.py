"""Evaluate hard background negatives for the native SOMA command recognizer."""
from __future__ import annotations

import argparse
from collections import defaultdict
from csv import DictReader
from dataclasses import dataclass
from hashlib import file_digest
import json
import math
from pathlib import Path
import resource
from time import perf_counter
import wave

import numpy as np
from scipy.signal import resample_poly
import torch
from torch.nn import functional as F

from native_speech_commands import COMMANDS, LABELS, SAMPLE_RATE, KeywordRecognizer
from train_speech_commands import (
    SEED, Recording, _artificial_quiet, _predict, _prepare, _score, _stream_timing,
    _test, _train_validation, _verify_speakers,
)


NOISE_SEED = 42133
MAX_FALSE_ACCEPTANCE = .10
ENVIRONMENT_SPLIT = {1: "train", 2: "train", 3: "train", 4: "validation", 5: "test"}
SILENCE = LABELS.index("silence")


@dataclass(frozen=True)
class Environment:
    file: str
    source: str
    category: str
    windows: np.ndarray


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _environment(directory: Path):
    owners = {}
    meta = directory / "meta" / "esc50.csv"
    if not meta.is_file():
        meta = directory / "esc50.csv"
    hashes = {"esc50.csv": _digest(meta)}
    result = {split: [] for split in ENVIRONMENT_SPLIT.values()}
    with meta.open(newline="", encoding="utf-8") as stream:
        entries = sorted(DictReader(stream), key=lambda row: row["filename"])
    for row in entries:
        if row["esc10"] != "True":
            continue
        name = row["filename"]
        source = row["src_file"]
        split = ENVIRONMENT_SPLIT[int(row["fold"])]
        if owners.setdefault(source, split) != split:
            raise ValueError(f"original recording {source} crosses environment partitions")
        path = directory / "audio" / name
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (44100, 1, 2):
                raise ValueError(f"unexpected ESC-10 file format: {name}")
            raw = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        pcm = resample_poly(raw.astype(np.float32) / 32768, 160, 441)
        if len(pcm) != 5 * SAMPLE_RATE:
            raise ValueError(f"unexpected ESC-10 clip duration: {name}")
        hashes[name] = _digest(path)
        result[split].append(Environment(name, source, row["category"], pcm.reshape(5, SAMPLE_RATE)))
    if any(not rows for rows in result.values()):
        raise ValueError("ESC-10 training, validation, and test folds must all be present")
    return result, hashes


def _rms(samples):
    return np.sqrt(np.mean(samples ** 2, axis=-1, keepdims=True)).clip(min=1e-7)


def _pcm16(samples):
    return np.clip(np.rint(samples * 32768), -32768, 32767).astype(np.int16)


def _paired_negatives(environments: list[Environment]):
    rng = np.random.default_rng(NOISE_SEED)
    recorded, synthetic = [], []
    for item in environments:
        for index, pcm in enumerate(item.windows):
            name = f"{item.file}/{index}"
            scaled = pcm
            if index % 2:
                scaled = pcm / _rms(pcm) * 10 ** rng.uniform(-2.5, -.9)
            actual = _pcm16(scaled)
            white = rng.standard_normal(SAMPLE_RATE, dtype=np.float32)
            white *= _rms(actual.astype(np.float32) / 32768) / _rms(white)
            recorded.append(Recording(name, None, SILENCE, actual))
            synthetic.append(Recording(name, None, SILENCE, _pcm16(white)))
    return recorded, synthetic


def _environment_features(frontend, environments):
    pcm = np.concatenate([item.windows for item in environments])
    batches = torch.from_numpy(pcm.clip(-1, 32767 / 32768)).split(128)
    with torch.no_grad():
        return torch.cat([frontend.features(batch) for batch in batches])


def _mixed_command_features(frontend, commands, environments):
    noises = np.concatenate([item.windows for item in environments])
    rng = np.random.default_rng(NOISE_SEED + 12)
    batched = []
    for offset in range(0, len(commands), 128):
        selected = commands[offset:offset + 128]
        speech = np.stack([row.samples for row in selected]).astype(np.float32) / 32768
        noise = noises[rng.integers(0, len(noises), len(selected))]
        noise = noise / _rms(noise) * (_rms(speech) / math.sqrt(10))
        mixture = np.clip(speech + noise, -1, 32767 / 32768)
        with torch.no_grad():
            batched.append(frontend.features(torch.from_numpy(mixture)))
    return torch.cat(batched)


def _command_scores(probabilities):
    confidence, best = probabilities.max(dim=-1)
    return confidence.double().masked_fill(best >= len(COMMANDS), -1.)


def _cutoff(scores):
    allowed = math.floor(len(scores) * MAX_FALSE_ACCEPTANCE)
    candidates = scores[scores >= 0].sort(descending=True).values
    return math.nextafter(float(candidates[allowed]), math.inf) if len(candidates) > allowed else 0.


def _calibrate(speech_posterior, labels, environment_posterior):
    speech_scores = _command_scores(speech_posterior)
    slices = {"other_words": speech_scores[labels == LABELS.index("unknown")],
              "generated_quiet": speech_scores[labels == SILENCE],
              "recorded_environment": _command_scores(environment_posterior)}
    if any(not len(group) for group in slices.values()):
        raise ValueError("all validation rejection slices must be present")
    cutoff = max(_cutoff(group) for group in slices.values())
    rates = {name: float((scores >= cutoff).double().mean()) for name, scores in slices.items()}
    return cutoff, rates


def _from_checkpoint(original):
    model = KeywordRecognizer()
    model.load_state_dict(original["state_dict"])
    return model


def _fine_tune(original, train, extra, validation, ambient_validation, epochs, name):
    model = _from_checkpoint(original)
    rng = torch.Generator().manual_seed(NOISE_SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.001)
    clean, noisy, labels = (torch.cat((base, added)) for base, added in zip(train, extra))
    weights = torch.bincount(labels).float().pow(-.5)
    best = -1.
    history = []
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for ids in torch.randperm(len(labels), generator=rng).split(128):
            mix = (torch.rand(len(ids), generator=rng) < .30)[:, None, None]
            x = torch.where(mix, noisy[ids], clean[ids])
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), labels[ids], weight=weights).backward()
            optimizer.step()
        posterior = _predict(model, validation[0])
        cutoff, rejected = _calibrate(posterior, validation[2], _predict(model, ambient_validation))
        accepted = _score(posterior, validation[2], cutoff)[0]["known_correct_and_accepted"]
        history.append({"epoch": epoch, "validation_correct_accepted": accepted,
                        "threshold": cutoff, "validation_false_accepts": rejected})
        if accepted > best:
            best = accepted
            selected = epoch
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(f"{name} epoch {epoch}: known correct/accepted {accepted:.3f}, cutoff {cutoff:.3f}", flush=True)
    model.load_state_dict(state)
    return model, {"selected_epoch": selected, "history": history,
                   "train_and_validation_seconds": perf_counter() - start}


def _environment_result(posterior, clips, cutoff):
    accepted = (_command_scores(posterior) >= cutoff).reshape(len(clips), 5)
    per_source = defaultdict(bool)
    categories = defaultdict(lambda: {"windows": 0, "accepted": 0})
    for clip, scores in zip(clips, accepted):
        per_source[clip.source] |= bool(scores.any())
        categories[clip.category]["windows"] += len(scores)
        categories[clip.category]["accepted"] += int(scores.sum())
    return {"accepted_windows": int(accepted.sum()), "windows": accepted.numel(),
            "sources_with_a_false_accept": sum(per_source.values()), "sources": len(per_source),
            "clips": len(clips), "by_category": dict(categories)}


def _speech_decisions(posteriors, rows, labels, cutoff, trace, mode, name):
    result, best, confidence, accept = _score(posteriors, labels, cutoff)
    for row, guess, score, accepted in zip(rows, best, confidence, accept):
        trace.write(json.dumps({"model": name, "condition": mode, "clip": row.name,
                                "expected": LABELS[row.label], "top_label": LABELS[int(guess)],
                                "confidence": float(score), "accept": bool(accepted)}) + "\n")
    return result


def _mixed_results(posteriors, rows, cutoff):
    expected = torch.tensor([row.label for row in rows])
    best = posteriors.argmax(dim=-1)
    accepted = _command_scores(posteriors) >= cutoff
    return {"clips": len(rows), "accepted": int(accepted.sum()),
            "correct_and_accepted": int((accepted & (best == expected)).sum())}


def run(train_archive, test_archive, esc, starting_checkpoint, out, epochs):
    torch.set_num_threads(2)
    original = torch.load(starting_checkpoint, map_location="cpu", weights_only=True)
    if original["version"] != 1 or not original["temporal"] or tuple(original["labels"]) != LABELS:
        raise ValueError("expected the recurrent SOMA speech command checkpoint")
    model = _from_checkpoint(original)
    speech = {**_train_validation(train_archive), "test": _test(test_archive)}
    speakers = _verify_speakers(speech)
    _artificial_quiet(speech["train"], 1200, SEED)
    _artificial_quiet(speech["validation"], 400, SEED + 1)
    env, hashes = _environment(esc)
    print("Checked official speech speakers and ESC-10 source folds.", flush=True)
    live, matched = _paired_negatives(env["train"])
    data = {key: _prepare(rows, model, SEED + index + 2)
            for index, (key, rows) in enumerate(speech.items())}
    extras = {"synthetic_negatives": _prepare(matched, model, NOISE_SEED),
              "recorded_negatives": _prepare(live, model, NOISE_SEED)}
    validation_environment = _environment_features(model, env["validation"])
    test_environment = _environment_features(model, env["test"])
    commands = [row for row in speech["test"] if row.label < len(COMMANDS)]
    test_mixture = _mixed_command_features(model, commands, env["test"])

    result = {"models": {}, "speaker_counts": speakers, "validation_slice_limit": MAX_FALSE_ACCEPTANCE,
              "training_clips_per_epoch": len(speech["train"]) + len(live),
              "added_background_windows_per_epoch": len(live), "epochs": epochs, "seed": NOISE_SEED,
              "esc10_folds": ENVIRONMENT_SPLIT, "esc10_sha256": hashes,
              "input_sha256": {p.name: _digest(p) for p in
                               (train_archive, test_archive, starting_checkpoint)}}
    out.mkdir(parents=True, exist_ok=False)
    mode_names = ("original", "threshold_only", "synthetic_negatives", "recorded_negatives")
    with (out / "speech_test_decisions.jsonl").open("w", encoding="utf-8") as trace:
        for name in mode_names:
            model = _from_checkpoint(original)
            training = {}
            if name in extras:
                model, training = _fine_tune(original, data["train"], extras[name], data["validation"],
                                             validation_environment, epochs, name)
            val = _predict(model, data["validation"][0])
            threshold, val_ood = _calibrate(val, data["validation"][2], _predict(model, validation_environment))
            # The first row retains the previously published threshold.
            cutoff = original["threshold"] if name == "original" else threshold
            entry = {"threshold": cutoff, "training": training,
                     "validation_calibration_cutoff": threshold,
                     "validation_false_accepts_at_calibration_cutoff": val_ood}
            entry["speech_commands_test"] = _speech_decisions(
                _predict(model, data["test"][0]), speech["test"], data["test"][2], cutoff,
                trace, "original_test_audio", name)
            entry["esc10_only_test"] = _environment_result(
                _predict(model, test_environment), env["test"], cutoff)
            entry["known_commands_plus_esc10_10db"] = _mixed_results(
                _predict(model, test_mixture), commands, cutoff)
            entry["streaming"] = _stream_timing(model, speech["test"])
            if entry["streaming"]["max_offline_logit_difference"] >= 1e-4:
                raise RuntimeError("streaming recognizer diverged from offline logits")
            if name != "original":
                torch.save({"version": 1, "temporal": True, "labels": LABELS, "threshold": cutoff,
                            "state_dict": model.state_dict()}, out / f"{name}.pt")
            result["models"][name] = entry
    result["peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (out / "noise_rejection_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-archive", "test-archive", "esc", "starting-checkpoint", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    run(args.train_archive, args.test_archive, args.esc, args.starting_checkpoint, args.out, args.epochs)


if __name__ == "__main__":
    main()
