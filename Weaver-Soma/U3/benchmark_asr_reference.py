"""A fixed speaker-balanced LibriSpeech sample for the bundled PocketSphinx reference."""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import io
import json
from pathlib import Path

from pocketsphinx import Decoder
import pyarrow.parquet as parquet
from rapidfuzz.distance import Levenshtein
import soundfile

from native_asr import SAMPLE_RATE, normalize


def _sample(file: parquet.ParquetFile) -> set[str]:
    speakers = defaultdict(list)
    for batch in file.iter_batches(columns=["id", "speaker_id"]):
        for row in batch.to_pylist():
            speakers[row["speaker_id"]].append(row["id"])
    if any(len(utterances) < 2 for utterances in speakers.values()):
        raise ValueError("the comparison needs two utterances per test speaker")
    return {utt for rows in speakers.values()
            for utt in sorted(rows, key=lambda identifier: sha256(identifier.encode()).digest())[:2]}


def benchmark(path: Path) -> dict:
    corpus = parquet.ParquetFile(path)
    ids = _sample(corpus)
    engine = Decoder(samprate=SAMPLE_RATE, loglevel="ERROR")
    predictions = []
    for batch in corpus.iter_batches(batch_size=48, columns=["id", "speaker_id", "audio", "text"]):
        for row in batch.to_pylist():
            if row["id"] not in ids:
                continue
            samples, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
            if rate != SAMPLE_RATE or samples.ndim != 1:
                raise ValueError(f"unexpected reference PCM for {row['id']}")
            engine.start_utt()
            engine.process_raw(samples.astype("<i2").tobytes(), no_search=False, full_utt=True)
            engine.end_utt()
            hypothesis = engine.hyp()
            predictions.append({"id": row["id"], "speaker_id": row["speaker_id"],
                                "reference": normalize(row["text"]),
                                "hypothesis": normalize(hypothesis.hypstr) if hypothesis else ""})
    if {row["id"] for row in predictions} != ids:
        raise ValueError("incomplete fixed reference sample")
    errors = sum(Levenshtein.distance(row["reference"].split(), row["hypothesis"].split())
                 for row in predictions)
    words = sum(len(row["reference"].split()) for row in predictions)
    return {"model": "PocketSphinx bundled English acoustic and language models",
            "version": version("pocketsphinx"), "utterances": len(ids),
            "speakers": len({row["speaker_id"] for row in predictions}),
            "wer": errors / words, "word_errors": errors, "reference_words": words,
            "predictions": predictions}


def score_native(comparison: dict, path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        native = {row["id"]: row for row in map(json.loads, stream)}
    total = 0
    for row in comparison["predictions"]:
        result = native[row["id"]]
        if result["reference"] != row["reference"]:
            raise ValueError(f"different reference text for {row['id']}")
        total += Levenshtein.distance(row["reference"].split(), result["hypothesis"].split())
    return {"utterances": comparison["utterances"], "reference_words": comparison["reference_words"],
            "word_errors": total, "wer": total / comparison["reference_words"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-parquet", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--native-predictions", type=Path, help="score SOMA on the same sampled utterances")
    args = parser.parse_args()
    result = benchmark(args.test_parquet)
    if args.native_predictions:
        result["paired_soma"] = score_native(result, args.native_predictions)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}))


if __name__ == "__main__":
    main()
