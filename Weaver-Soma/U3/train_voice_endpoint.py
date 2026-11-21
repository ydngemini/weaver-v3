"""Train the transcript endpoint model and compare silence policies on held-out data."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from audio_spectral import SAMPLE_RATE
from demo_realtime_voice import synthesize
from realtime_voice import RealtimeVoiceSession, VoiceConfig
from voice_endpoint import TranscriptEndpointModel, endpoint_features, words

FRAME = SAMPLE_RATE // 50
BASE_FRAMES = 15


def read_gsm(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line)["question"] for line in stream]


def read_webgpt(path):
    from pyarrow.parquet import ParquetFile

    return [row["question"]["full_text"]
            for batch in ParquetFile(path).iter_batches(columns=["question"])
            for row in batch.to_pylist()]


def examples(token_groups):
    """Pair full utterances with both near-end and interior cuts."""
    for tokens in token_groups:
        if len(tokens) < 4:
            continue
        entropy = int.from_bytes(sha256(" ".join(tokens).encode()).digest()[:4])
        yield endpoint_features(tokens), 1
        near_cut = len(tokens) - 1 - entropy % min(4, len(tokens) - 2)
        middle_cut = 2 + entropy % (len(tokens) - 3)
        for cut in {near_cut, middle_cut}:
            yield endpoint_features(tokens[:cut]), 0


def encode(rows, indices):
    features = torch.zeros(len(rows), len(indices))
    labels = torch.tensor([label for _, label in rows], dtype=torch.float32)
    for i, (active, _) in enumerate(rows):
        for key, value in active.items():
            index = indices.get(key)
            if index is not None:
                features[i, index] = value
    return features, labels


def fit(training, validation):
    frequency = Counter(key for active, _ in training for key in active)
    names = sorted(key for key, count in frequency.items() if count >= 24)
    indices = {name: i for i, name in enumerate(names)}
    x, y = encode(training, indices)
    layer = torch.nn.Linear(len(names), 1)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()
    optimizer = torch.optim.LBFGS(layer.parameters(), max_iter=100, line_search_fn="strong_wolfe")
    positive = (1 - y).sum() / y.sum()

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(layer(x).squeeze(1), y, pos_weight=positive)
        loss = loss + .00015 * layer.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    x_val, y_val = encode(validation, indices)
    with torch.no_grad():
        val_logits = layer(x_val).flatten().tolist()
        # Fix a 10% maximum early-close rate on validation prefixes before opening the test sets.
        incomplete = sorted(score for score, label in zip(val_logits, y_val.tolist()) if label == 0)
        threshold = incomplete[int(.9 * len(incomplete))] + .00001
        weights = dict(zip(names, layer.weight.flatten().tolist()))
    return TranscriptEndpointModel(layer.bias.item(), weights, threshold), indices


def classify(model, rows, indices):
    x, y = encode(rows, indices)
    prediction = (x @ torch.tensor(list(model.weights.values())) + model.bias) >= model.threshold
    return {
        "complete_called_complete": int((prediction & y.bool()).sum()),
        "complete_examples": int(y.sum()),
        "incomplete_called_complete": int((prediction & ~y.bool()).sum()),
        "incomplete_examples": int((1 - y).sum()),
    }


def audio_samples(path):
    pcm = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-f", "s16le", "pipe:1",
    ], timeout=15)
    return np.frombuffer(pcm, "<i2")


def voiced_frames(samples):
    samples = np.pad(samples, (0, -len(samples) % FRAME)).reshape(-1, FRAME)
    return np.flatnonzero(np.mean((samples.astype(np.float32) / 32768) ** 2, axis=1) >= VoiceConfig().speech_rms ** 2)


def replay(samples, mark, model, end_frames=BASE_FRAMES, decoder=None):
    session = RealtimeVoiceSession(VoiceConfig(end_frames=end_frames), endpoint_model=model)
    padded = np.pad(samples, (0, -len(samples) % FRAME + FRAME * 45)).astype("<i2")
    turn = None
    stops = []
    for start in range(0, len(padded), FRAME):
        end = start + FRAME
        session.push_pcm16le(start, padded[start:end].tobytes())
        for event in session.seal_input(end):
            if event.type == "speech_started":
                turn = event.data["turn_id"]
                if decoder is not None:
                    decoder.start_utt()
            elif event.type == "speech_frame" and decoder is not None:
                frame = event.data["frame"]
                pcm = (frame.samples.numpy() * 32768).astype("<i2").tobytes()
                decoder.process_raw(pcm, False, False)
                hypothesis = decoder.hyp()
                if hypothesis is not None:
                    session.report_transcript(turn, hypothesis.hypstr, frame.end_sample)
            elif event.type == "speech_stopped":
                stops.append({"at_sample": event.at_sample, **event.data})
                turn = None
                if decoder is not None:
                    decoder.end_utt()
        if end in mark and turn is not None:
            session.report_transcript(turn, mark[end], end)
    return stops


def pause_replay(prefix, rest, gap_frames, voice):
    def clip(tokens):
        data = np.frombuffer(synthesize(" ".join(tokens), voice), "<i2")
        end = (voiced_frames(data)[-1] + 1) * FRAME
        return np.pad(data, (0, max(0, end - len(data))))[:end]

    first = clip(prefix)
    samples = np.concatenate((first, np.zeros(FRAME * gap_frames, dtype="<i2"), clip(rest)))
    return samples, {len(first): " ".join(prefix), len(samples): " ".join(prefix + rest)}


def compare_audio(clips, model, decoder=None):
    rows = []
    for ident, samples, mark in clips:
        last_speech = (int(voiced_frames(samples)[-1]) + 1) * FRAME
        variants = {"fixed_300ms": (None, BASE_FRAMES), "fixed_600ms": (None, BASE_FRAMES * 2),
                    "learned": (model, BASE_FRAMES)}
        if decoder is not None:
            variants["learned_streaming_asr"] = (model, BASE_FRAMES)
        results = {}
        for policy, (endpoint, frames) in variants.items():
            if policy == "learned_streaming_asr":
                stops = replay(samples, {}, endpoint, frames, decoder)
            else:
                stops = replay(samples, mark, endpoint, frames)
            after = [event for event in stops if event["at_sample"] > last_speech]
            results[policy] = {
                "premature_stops": len(stops) - len(after),
                "end_delay_ms": None if not after else (after[0]["at_sample"] - last_speech) * 1000 // SAMPLE_RATE,
                "last_endpoint": after[0].get("endpoint") if after else None,
            }
        rows.append({"clip": ident, "policies": results})
    return rows


def summarize(rows):
    result = {}
    for policy in rows[0]["policies"]:
        cases = [row["policies"][policy] for row in rows]
        delays = [case["end_delay_ms"] for case in cases if case["end_delay_ms"] is not None]
        uncut_delays = [case["end_delay_ms"] for case in cases
                        if case["end_delay_ms"] is not None and not case["premature_stops"]]
        result[policy] = {
            "clips": len(cases),
            "clips_with_premature_stop": sum(case["premature_stops"] > 0 for case in cases),
            "missing_final_stop": len(cases) - len(delays),
            "median_final_delay_ms_on_uncut_clips": median(uncut_delays) if uncut_delays else None,
            "fast_final_stops": sum(delay == 300 for delay in uncut_delays),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm-dir", type=Path, required=True)
    parser.add_argument("--webgpt", type=Path, required=True)
    parser.add_argument("--cookbook-walk", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare-asr", action="store_true", help="also test streaming PocketSphinx partials")
    args = parser.parse_args()
    torch.set_num_threads(1)

    gsm_train = args.gsm_dir / "train.jsonl"
    gsm_test = args.gsm_dir / "test.jsonl"
    csv_path = args.cookbook_walk / "customer_service_synthetic.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        support = list(csv.DictReader(stream))

    train, val = [], []
    seen = set()
    counts = {}
    for name, utterances in (("gsm8k", read_gsm(gsm_train)), ("webgpt", read_webgpt(args.webgpt))):
        counts[name] = {"train_utterances": 0, "validation_utterances": 0}
        for text in utterances:
            tokens = words(text)
            normalized = " ".join(tokens)
            if len(tokens) < 4 or normalized in seen:
                continue
            seen.add(normalized)
            validation = sha256(normalized.encode()).digest()[0] < 26
            (val if validation else train).append(tokens)
            counts[name]["validation_utterances" if validation else "train_utterances"] += 1

    training, validation = list(examples(train)), list(examples(val))
    model, indices = fit(training, validation)
    held_out_gsm = [words(text) for text in read_gsm(gsm_test) if " ".join(words(text)) not in seen]
    support_text = [words(row["user_text"]) for row in support]
    original = []
    pauses = []
    for i, (row, tokens) in enumerate(zip(support, support_text)):
        pcm = audio_samples(args.cookbook_walk / row["audio_path"])
        last_speech = int(voiced_frames(pcm)[-1] + 1) * FRAME
        original.append((row["example_id"], pcm, {last_speech: row["user_text"]}))
        split = 2 + i % (len(tokens) - 3)
        # A second condition puts identical audio into all three policies.
        pauses.append((row["example_id"], *pause_replay(
            tokens[:split], tokens[split:], 17 + i % 7, ("slt", "kal16")[i % 2])))

    decoder = None
    if args.compare_asr:
        from pocketsphinx import Decoder

        decoder = Decoder(samprate=SAMPLE_RATE)
    original_results = compare_audio(original, model, decoder)
    pause_results = compare_audio(pauses, model)
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = {"version": 1, "bias": model.bias, "weights": dict(model.weights), "threshold": model.threshold}
    (args.out / "voice_endpoint_model.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    source_hashes = {name: sha256(path.read_bytes()).hexdigest()
                     for name, path in (("gsm8k_train", gsm_train), ("gsm8k_test", gsm_test),
                                        ("webgpt_parquet", args.webgpt), ("cookbook_csv", csv_path))}
    report = {
        "source_sha256": source_hashes, "corpus_counts": counts,
        "labeled_train_examples": len(training), "labeled_validation_examples": len(validation),
        "model_features": len(indices), "threshold": model.threshold,
        "validation": classify(model, validation, indices),
        "held_out_gsm8k": classify(model, list(examples(held_out_gsm)), indices),
        "held_out_cookbook_text": classify(model, list(examples(support_text)), indices),
        "original_cookbook_audio": summarize(original_results),
        "synthesized_mid_turn_pauses_with_reference_transcript": summarize(pause_results),
        "original_samples": original_results, "injected_pause_samples": pause_results,
        "streaming_asr": "PocketSphinx unrestricted English model" if args.compare_asr else None,
    }
    (args.out / "voice_endpoint_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"original_samples", "injected_pause_samples"}}, indent=2))


if __name__ == "__main__":
    main()
