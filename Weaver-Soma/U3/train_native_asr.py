"""Train an English character CTC model using pinned LibriSpeech Parquet shards."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import file_digest, sha256
import io
import json
import math
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import pyarrow.parquet as parquet
from rapidfuzz.distance import Levenshtein
import soundfile
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from native_asr import (
    ALPHABET, SAMPLE_RATE, STRIDE, TOKEN_ID, ASRStream, CausalTranscriber, GreedyCTC,
    normalize, transcript,
)


REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
SHARDS = {
    "train": ("train.100-0000.parquet", "train.100-0007.parquet"),
    "validation": ("validation-0000.parquet",),
    "test": ("test-0000.parquet",),
}
SEED = 19711
BATCH = 32


@dataclass(frozen=True)
class Clip:
    identifier: str
    speaker: int
    reference: str
    samples: int
    features: torch.Tensor
    noisy_features: torch.Tensor | None
    stream_pcm: bytes | None


def _seed(identifier: str) -> int:
    return int.from_bytes(sha256(identifier.encode()).digest()[:8], "big")


@torch.no_grad()
def _read(path: Path, model: CausalTranscriber, test: bool) -> list[Clip]:
    result = []
    file = parquet.ParquetFile(path)
    for batch in file.iter_batches(batch_size=64, columns=["id", "speaker_id", "text", "audio"]):
        for row in batch.to_pylist():
            identifier = row["id"]
            reference = normalize(row["text"])
            pcm, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
            if rate != SAMPLE_RATE or pcm.ndim != 1 or not reference:
                raise ValueError(f"invalid LibriSpeech recording or reference: {identifier}")
            audio = pcm.astype(np.float32) / 32768
            features = model.features(torch.from_numpy(audio))
            if not len(features):
                raise ValueError(f"too few audio samples for an acoustic frame: {identifier}")
            noisy = raw = None
            if test:
                seed = _seed(identifier)
                white = np.random.default_rng(seed).standard_normal(len(audio), dtype=np.float32)
                rms = np.sqrt(np.mean(audio ** 2))
                white *= rms / (math.sqrt(10) * np.sqrt(np.mean(white ** 2)))
                noisy = model.features(torch.from_numpy(np.clip(audio + white, -1, 32767 / 32768)))
                if seed % 64 == 0:
                    raw = pcm.astype("<i2").tobytes()
            result.append(Clip(identifier, row["speaker_id"], reference, len(pcm), features, noisy, raw))
    return result


def _audit(folds) -> dict:
    seen = {}
    ids = set()
    summary = {}
    for label, rows in folds.items():
        speakers = {row.speaker for row in rows}
        for speaker in speakers:
            if seen.setdefault(speaker, label) != label:
                raise ValueError(f"LibriSpeech speaker {speaker} overlaps corpus partitions")
        for row in rows:
            if row.identifier in ids or int(row.identifier.split("-")[0]) != row.speaker:
                raise ValueError(f"invalid or duplicate utterance ID: {row.identifier}")
            ids.add(row.identifier)
        summary[label] = {"utterances": len(rows), "speakers": len(speakers),
                          "hours": sum(row.samples for row in rows) / 3600 / SAMPLE_RATE}
    return summary


def _min_alignment(reference: str) -> int:
    return len(reference) + sum(a == b for a, b in zip(reference, reference[1:]))


def _batches(rows) -> list:
    ordered = sorted(rows, key=lambda row: (len(row.features), row.identifier))
    return [ordered[i:i + BATCH] for i in range(0, len(ordered), BATCH)]


def _training_step(model, loss, rows):
    features = pad_sequence([row.features for row in rows], batch_first=True)
    audio_lengths = torch.tensor([len(row.features) for row in rows])
    text_lengths = torch.tensor([len(row.reference) for row in rows])
    targets = torch.tensor([TOKEN_ID[char] for row in rows for char in row.reference])
    logits, _ = model(features)
    return loss(logits.log_softmax(dim=-1).transpose(0, 1), targets, audio_lengths, text_lengths)


@torch.no_grad()
def _evaluate(model, rows, noisy=False):
    model.eval()
    predictions = []
    words = defaultdict(int)
    errors = defaultdict(int)
    error_types = Counter()
    characters = character_errors = 0
    for group in _batches(rows):
        features = [row.noisy_features if noisy else row.features for row in group]
        logits, _ = model(pad_sequence(features, batch_first=True))
        codes = logits.argmax(dim=-1).tolist()
        for row, frames, indices in zip(group, features, codes):
            hypothesis = transcript(indices[:len(frames)])
            ref = row.reference.split()
            hyp = hypothesis.split()
            operations = Levenshtein.editops(ref, hyp)
            error_types.update(operation.tag for operation in operations)
            words[row.speaker] += len(ref)
            errors[row.speaker] += len(operations)
            expected = row.reference.replace(" ", "")
            found = hypothesis.replace(" ", "")
            characters += len(expected)
            character_errors += Levenshtein.distance(expected, found)
            predictions.append({"id": row.identifier, "speaker_id": row.speaker, "reference": row.reference,
                                "hypothesis": hypothesis, "word_errors": len(operations),
                                "reference_words": len(ref)})
    stats = {"wer": sum(errors.values()) / sum(words.values()),
             "cer_without_spaces": character_errors / characters,
             "reference_words": sum(words.values()), "word_errors": sum(errors.values()),
             "word_error_types": dict(error_types),
             "blank_hypotheses": sum(not row["hypothesis"] for row in predictions),
             "speaker_macro_wer": float(np.mean([errors[speaker] / count for speaker, count in words.items()])),
             "speaker_wer": {str(speaker): errors[speaker] / words[speaker] for speaker in sorted(words)}}
    return stats, predictions


@torch.no_grad()
def _stream_check(model, rows):
    torch.set_num_threads(1)
    clips = [row for row in rows if row.stream_pcm is not None]
    if not clips:
        raise ValueError("no audio selected for the streaming parity check")
    durations = []
    max_difference = 0.
    for utterance, row in enumerate(clips):
        stream, decoder = ASRStream(model), GreedyCTC()
        packet_logits, packet_text = [], []
        # Deliver each clip in 20 ms packets, which may arrive between 40 ms model frames.
        for offset in range(0, len(row.stream_pcm), STRIDE):
            now = perf_counter()
            logits = stream.push(row.stream_pcm[offset:offset + STRIDE])
            letters = decoder.push(logits.argmax(dim=-1).tolist())
            if utterance and len(logits):
                durations.append(1000 * (perf_counter() - now))
            packet_logits.append(logits)
            packet_text.extend(letter for letter, _ in letters)
        offline, _ = model(row.features[None])
        offline = offline[0]
        diff = float((torch.cat(packet_logits) - offline).abs().max())
        max_difference = max(diff, max_difference)
        if " ".join("".join(packet_text).split()) != transcript(offline.argmax(dim=-1).tolist()):
            raise RuntimeError(f"streaming emissions disagree with batch output on {row.identifier}")
    return {"checked_utterances": len(clips), "max_logit_difference": max_difference,
            "timed_packets": len(durations), "warm_cpu_packet_p50_ms": float(np.quantile(durations, .5)),
            "warm_cpu_packet_p95_ms": float(np.quantile(durations, .95))}


def run(data: Path, out: Path, epochs: int):
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = CausalTranscriber()
    files = [data / name for names in SHARDS.values() for name in names]
    hashes = {}
    for filename in files:
        with filename.open("rb") as stream:
            hashes[filename.name] = file_digest(stream, "sha256").hexdigest()
    folds = {split: [row for name in names for row in _read(data / name, model, split == "test")]
             for split, names in SHARDS.items()}
    provenance = _audit(folds)
    train = [row for row in folds["train"] if _min_alignment(row.reference) <= len(row.features)]
    provenance["train"]["ctc_unalignable_excluded"] = len(folds["train"]) - len(train)
    print(f"Corpus partitions: {provenance}", flush=True)
    loader = _batches(train)
    rng = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0018, weight_decay=.001)
    loss = nn.CTCLoss(blank=0)
    out.mkdir(parents=True, exist_ok=False)
    history = []
    best_wer = math.inf
    for epoch in range(1, epochs + 1):
        started = perf_counter()
        model.train()
        cumulative = 0.
        for index in torch.randperm(len(loader), generator=rng).tolist():
            rows = loader[index]
            optimizer.zero_grad(set_to_none=True)
            value = _training_step(model, loss, rows)
            if not math.isfinite(float(value.detach())):
                raise RuntimeError("nonfinite CTC loss")
            value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            cumulative += float(value.detach()) * len(rows)
        validation, _ = _evaluate(model, folds["validation"])
        record = {"epoch": epoch, "ctc_loss": cumulative / len(train),
                  "validation_wer": validation["wer"], "validation_cer": validation["cer_without_spaces"],
                  "train_and_validation_seconds": perf_counter() - started}
        history.append(record)
        if validation["wer"] < best_wer:
            best_wer, best_epoch = validation["wer"], epoch
            torch.save({"version": 1, "alphabet": ALPHABET, "state_dict": model.state_dict()},
                       out / "soma_ctc.pt")
        (out / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)

    selected = torch.load(out / "soma_ctc.pt", weights_only=True)
    model.load_state_dict(selected["state_dict"])
    metrics = {"source_revision": REVISION, "parquet_sha256": hashes, "corpus": provenance,
               "selected_epoch": best_epoch, "parameters": sum(param.numel() for param in model.parameters()),
               "seed": SEED, "training_threads": 4, "torch_version": torch.__version__, "tests": {}}
    for condition, noisy in (("librispeech_test_clean", False), ("test_clean_with_white_noise_10db", True)):
        result, predictions = _evaluate(model, folds["test"], noisy=noisy)
        metrics["tests"][condition] = result
        with (out / f"{condition}.jsonl").open("w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row) + "\n")
    metrics["streaming"] = _stream_check(model, folds["test"])
    metrics["peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({key: {k: v for k, v in row.items() if k != "speaker_wer"}
                      for key, row in metrics["tests"].items()}), flush=True)
    (out / "asr_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    run(args.data, args.out, args.epochs)


if __name__ == "__main__":
    main()
