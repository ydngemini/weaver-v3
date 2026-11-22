"""Compare frozen CTC posteriors with greedy, beam and train-text language decoding."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as parquet
from rapidfuzz.distance import Levenshtein
import torch
from torch.nn.utils.rnn import pad_sequence

from asr_prefix_beam import BeamSettings, CharacterModel, PrefixBeam
from native_asr import load_checkpoint, transcript
from train_native_asr import REVISION, SHARDS, _audit, _batches, _read


LANGUAGE_GRID = ((.20, 0.), (.20, .30), (.40, 0.), (.40, .55),
                 (.65, 0.), (.65, .90), (.90, 0.), (.90, 1.25))
WORKER_LM = None


@dataclass(frozen=True)
class Utterance:
    identifier: str
    speaker: int
    reference: str
    posterior: np.ndarray


def _init_worker(lm):
    global WORKER_LM
    WORKER_LM = lm


def _decode(task):
    row, settings = task
    beam = PrefixBeam(settings, WORKER_LM)
    beam.push(row.posterior)
    return beam.text()


def _metadata(path):
    corpus = parquet.ParquetFile(path)
    return [row for batch in corpus.iter_batches(columns=["id", "speaker_id", "text"])
            for row in batch.to_pylist()]


def _language_data(data: Path):
    rows = [row for filename in SHARDS["train"] for row in _metadata(data / filename)]
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError("a language model training transcript is listed twice")
    speakers = {row["speaker_id"] for row in rows}
    return CharacterModel.fit(row["text"] for row in rows), ids, speakers


def _validation_sample(rows, per_speaker=3):
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row.speaker].append(row)
    if any(len(group) < per_speaker for group in by_speaker.values()):
        raise ValueError("insufficient clips for speaker-balanced selection")
    return [row for group in by_speaker.values()
            for row in sorted(group, key=lambda row: sha256(row.identifier.encode()).digest())[:per_speaker]]


@torch.no_grad()
def _posteriors(model, rows, noisy=False):
    result = []
    for batch in _batches(rows):
        feats = [row.noisy_features if noisy else row.features for row in batch]
        logits, _ = model(pad_sequence(feats, batch_first=True))
        posterior = logits.log_softmax(dim=-1)
        for row, f, probs in zip(batch, feats, posterior):
            result.append(Utterance(row.identifier, row.speaker, row.reference, probs[:len(f)].numpy().copy()))
    return result


def _score(rows, hyps):
    words = chars = word_errors = char_errors = empties = 0
    details = []
    speaker_errors = defaultdict(lambda: [0, 0])
    word_error_types = Counter()
    for row, hypothesis in zip(rows, hyps, strict=True):
        ref_words = row.reference.split()
        operations = Levenshtein.editops(ref_words, hypothesis.split())
        edits = len(operations)
        word_error_types.update(operation.tag for operation in operations)
        ref_chars = row.reference.replace(" ", "")
        words += len(ref_words)
        chars += len(ref_chars)
        word_errors += edits
        char_errors += Levenshtein.distance(ref_chars, hypothesis.replace(" ", ""))
        empties += not hypothesis
        scores = speaker_errors[row.speaker]
        scores[0] += edits
        scores[1] += len(ref_words)
        details.append({"id": row.identifier, "speaker_id": row.speaker, "reference": row.reference,
                        "hypothesis": hypothesis, "reference_words": len(ref_words), "word_errors": edits})
    return ({"utterances": len(rows), "speakers": len(speaker_errors), "reference_words": words,
             "word_errors": word_errors, "wer": word_errors / words, "cer_without_spaces": char_errors / chars,
             "word_error_types": dict(word_error_types),
             "blank_hypotheses": empties,
             "speaker_macro_wer": float(np.mean([a / b for a, b in speaker_errors.values()]))}, details)


def _greedy(rows):
    return [transcript(row.posterior.argmax(axis=-1).tolist()) for row in rows]


def _beam(pool, rows, settings):
    return pool.map(_decode, [(row, settings) for row in rows], chunksize=10)


def _latency(rows, settings, lm):
    timings = []
    for row in rows[:25]:
        beam = PrefixBeam(settings, lm)
        bucket = []
        for frame in row.posterior:
            started = perf_counter()
            beam.push((frame,))
            bucket.append(1000 * (perf_counter() - started))
        if row is not rows[0]:
            timings.extend(bucket)
    return {"frames": len(timings), "search_only_p50_ms_per_40ms_frame": float(np.quantile(timings, .5)),
            "search_only_p95_ms_per_40ms_frame": float(np.quantile(timings, .95))}


def _digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def run(data, checkpoint, reference, out, workers):
    torch.set_num_threads(4)
    model = load_checkpoint(checkpoint)
    lm, train_ids, train_speakers = _language_data(data)
    validation = _read(data / SHARDS["validation"][0], model, False)
    test = _read(data / SHARDS["test"][0], model, True)
    corpus = _audit({"validation": validation, "test": test})
    if any(row.identifier in train_ids or row.speaker in train_speakers for row in validation + test):
        raise ValueError("language model training overlaps development or test")
    paired = json.loads(reference.read_text(encoding="utf-8"))
    paired_ids = {row["id"] for row in paired["predictions"]}
    paired_test = [row for row in test if row.identifier in paired_ids]
    if len(paired_test) != len(paired_ids):
        raise ValueError("some PocketSphinx sample IDs are absent from the test partition")
    validation = _validation_sample(validation)
    val_posts = _posteriors(model, validation)
    out.mkdir(parents=True, exist_ok=False)
    lm.save(out / "training_character_lm.json")

    settings = BeamSettings()
    grid = [BeamSettings(lm_weight=alpha, character_bonus=bonus) for alpha, bonus in LANGUAGE_GRID]
    select = []
    results = {}
    torch.set_num_threads(2)
    with multiprocessing.get_context("spawn").Pool(workers, _init_worker, (lm,)) as pool:
        val_greedy, _ = _score(val_posts, _greedy(val_posts))
        val_beam, _ = _score(val_posts, _beam(pool, val_posts, settings))
        for candidate in grid:
            score, _ = _score(val_posts, _beam(pool, val_posts, candidate))
            select.append({"settings": asdict(candidate), **score})
            print({"validation_wer": score["wer"], **asdict(candidate)}, flush=True)
        chosen = BeamSettings(**min(select, key=lambda row: (row["wer"], row["cer_without_spaces"]))["settings"])
        (out / "selected_decoder.json").write_text(json.dumps(asdict(chosen), indent=2) + "\n")
        (out / "decoder_calibration.json").write_text(json.dumps(select, indent=2) + "\n")
        print({"selected": asdict(chosen)}, flush=True)
        clean_posts = _posteriors(model, test)
        test_noisy = _posteriors(model, paired_test, noisy=True)
        conditions = {"test_clean": clean_posts, "matched_noisy_10db": test_noisy}
        modes = (("greedy", None), ("acoustic_beam", settings), ("fused", chosen))
        with (out / "decoder_predictions.jsonl").open("w", encoding="utf-8") as trace:
            for condition, rows in conditions.items():
                results[condition] = {}
                for mode, config in modes:
                    hypotheses = _greedy(rows) if config is None else _beam(pool, rows, config)
                    metrics, output = _score(rows, hypotheses)
                    results[condition][mode] = metrics
                    for row in output:
                        trace.write(json.dumps({"condition": condition, "decoder": mode, **row}) + "\n")
                    if condition == "test_clean":
                        matched = [(row, text) for row, text in zip(rows, hypotheses)
                                   if row.identifier in paired_ids]
                        selected_rows, selected_hypotheses = zip(*matched)
                        metric, _ = _score(selected_rows, selected_hypotheses)
                        if metric["reference_words"] != paired["reference_words"]:
                            raise ValueError("paired reference word totals disagree")
                        results.setdefault("matched_clean", {})[mode] = metric
                    print({"condition": condition, "mode": mode, "wer": metrics["wer"]}, flush=True)
    cpu_sample = _validation_sample(clean_posts, per_speaker=1)
    metrics = {
        "checkpoint_sha256": _digest(checkpoint), "corpus_revision": REVISION,
        "shards_sha256": {name: _digest(data / name) for names in SHARDS.values() for name in names},
        "acoustic_checkpoint_training": {"utterances": len(train_ids), "speakers": len(train_speakers)},
        "language_model": {"order": lm.order, "training_utterances": len(train_ids), "contexts": len(lm.counts)},
        "corpus": corpus, "selection": {"sample_size": len(validation), "greedy": val_greedy,
                                                "acoustic_beam": val_beam},
        "winner": asdict(chosen), "results": results, "pocketsphinx_matched_wer": paired["wer"],
        "search_latency_acoustic": _latency(cpu_sample, settings, None),
        "search_latency_fused": _latency(cpu_sample, chosen, lm),
    }
    (out / "decoder_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path, help="previously recorded PocketSphinx comparison")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    run(args.data, args.checkpoint, args.reference, args.out, args.workers)


if __name__ == "__main__":
    main()
