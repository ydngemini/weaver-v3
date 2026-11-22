"""Continue the same native ASR checkpoint with matched steps and more speakers."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from hashlib import file_digest
import json
import math
import multiprocessing
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import torch
from torch import nn

from asr_prefix_beam import BeamSettings, CharacterModel
from benchmark_asr_beam import _beam, _greedy, _init_worker, _posteriors, _score
from native_asr import ALPHABET, SAMPLE_RATE, load_checkpoint
from train_native_asr import (
    BATCH, REVISION, SHARDS, _audit, _batches, _evaluate, _min_alignment, _read,
    _stream_check, _training_step,
)


ADDED_SHARDS = tuple(f"train.100-{n:04d}.parquet" for n in (2, 5, 9, 12))
FRESH_TEST = "other-test-0000.parquet"
SEED = 34973
LEARNING_RATE = .0009


def _match(requested: list[int], pool: list[int]) -> list[int]:
    """Give each requested padded frame load the nearest unused candidate."""
    if not pool:
        raise ValueError("at least one expanded batch is required")
    ordered = sorted((frames, index) for index, frames in enumerate(pool))
    available = []
    chosen = []
    for target in requested:
        if not available:
            available = ordered.copy()
        insert = bisect_left(available, (target, -1))
        neighbors = range(max(0, insert - 1), min(len(available), insert + 1))
        best = min(neighbors, key=lambda i: (abs(available[i][0] - target), available[i][1]))
        chosen.append(available.pop(best)[1])
    return chosen


def _frame_load(batch) -> int:
    return len(batch) * max(len(row.features) for row in batch)


def _schedule(original, expanded, updates):
    eligible = lambda rows: [row for row in rows if _min_alignment(row.reference) <= len(row.features)]
    reference = [batch for batch in _batches(eligible(original)) if len(batch) == BATCH]
    candidate = [batch for batch in _batches(eligible(expanded)) if len(batch) == BATCH]
    if not reference or not candidate:
        raise ValueError("training partitions must have at least one complete batch")
    rng = np.random.default_rng(SEED)
    order = []
    while len(order) < updates:
        order.extend(rng.permutation(len(reference)).tolist())
    replay = [reference[index] for index in order[:updates]]
    matched = _match([_frame_load(batch) for batch in replay], [_frame_load(batch) for batch in candidate])
    return replay, [candidate[index] for index in matched]


def _exposure(schedule) -> dict:
    return {"updates": len(schedule),
            "clip_exposures": sum(map(len, schedule)),
            "audio_hours_seen": sum(row.samples for batch in schedule for row in batch) / (3600 * SAMPLE_RATE),
            "padded_frames_seen": sum(map(_frame_load, schedule)),
            "speakers_seen": len({row.speaker for batch in schedule for row in batch}),
            "unique_utterances_seen": len({row.identifier for batch in schedule for row in batch})}


def _save(model, out):
    torch.save({"version": 1, "alphabet": ALPHABET, "state_dict": model.state_dict()}, out)


def _continue(checkpoint, path, schedule, validation, initial, validation_every):
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    model = load_checkpoint(checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=.001)
    objective = nn.CTCLoss(blank=0)
    history = [{"update": 0, "validation_wer": initial["wer"], "validation_cer": initial["cer_without_spaces"]}]
    best = (initial["wer"], initial["cer_without_spaces"])
    selected = 0
    _save(model, path)
    start = perf_counter()
    loss_sum = 0.
    interval = 0
    model.train()
    for update, batch in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = _training_step(model, objective, batch)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise ValueError(f"nonfinite CTC loss after update {update}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        loss_sum += value
        interval += 1
        if update % validation_every and update != len(schedule):
            continue
        score, _ = _evaluate(model, validation)
        key = score["wer"], score["cer_without_spaces"]
        record = {"update": update, "mean_ctc_loss": loss_sum / interval,
                  "validation_wer": key[0], "validation_cer": key[1]}
        history.append(record)
        print(path.stem, json.dumps(record), flush=True)
        if key < best:
            best, selected = key, update
            _save(model, path)
        model.train()
        loss_sum = 0.
        interval = 0
    return {**_exposure(schedule), "selected_update": selected, "history": history,
            "training_and_dev_seconds": perf_counter() - start, "checkpoint_bytes": path.stat().st_size}


def _speaker_delta(control, candidate) -> dict:
    """Resample speaker clusters to estimate the paired change in pooled WER."""
    old = {row["id"]: row for row in control}
    if len(old) != len(control) or set(old) != {row["id"] for row in candidate}:
        raise ValueError("the paired WER comparison requires identical unique utterance IDs")
    paired = defaultdict(lambda: [0, 0])
    for row in candidate:
        prior = old[row["id"]]
        if any(row[key] != prior[key] for key in ("speaker_id", "reference", "reference_words")):
            raise ValueError("the paired WER comparison requires identical references")
        pair = paired[row["speaker_id"]]
        pair[0] += row["word_errors"] - prior["word_errors"]
        pair[1] += row["reference_words"]
    total = np.asarray(list(paired.values()))
    random = np.random.default_rng(SEED)
    draws = random.integers(0, len(total), size=(5000, len(total)))
    sums = total[draws].sum(axis=1)
    intervals = 100 * np.quantile(sums[:, 0] / sums[:, 1], (.025, .975))
    return {"speakers": len(total), "draws": len(draws),
            "difference_wer_points": 100 * total[:, 0].sum() / total[:, 1].sum(),
            "speaker_bootstrap_interval95_wer_points": intervals.tolist()}


def _digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def run(data, checkpoint, decoder_dir, out, updates, cadence, workers):
    torch.set_num_threads(4)
    model = load_checkpoint(checkpoint)
    original = [row for name in SHARDS["train"] for row in _read(data / name, model, False)]
    extra = [row for name in ADDED_SHARDS for row in _read(data / name, model, False)]
    expanded = original + extra
    validation = _read(data / SHARDS["validation"][0], model, False)
    tests = {"test_clean": _read(data / SHARDS["test"][0], model, True),
             "test_other": _read(data / FRESH_TEST, model, False)}
    corpus = _audit({"train": expanded, "validation": validation, **tests})
    corpus["original_training"] = _audit({"original_training": original})["original_training"]
    original_plan, expanded_plan = _schedule(original, expanded, updates)
    print({"corpus": corpus, "original_schedule": _exposure(original_plan),
           "expanded_schedule": _exposure(expanded_plan)}, flush=True)
    initial, _ = _evaluate(model, validation)

    out.mkdir(parents=True, exist_ok=False)
    paths = {"before_continuation": checkpoint, "original_data": out / "original_data_asr.pt",
             "expanded_data": out / "expanded_data_asr.pt"}
    training = {"initial_validation": initial}
    for name, schedule in (("original_data", original_plan), ("expanded_data", expanded_plan)):
        training[name] = _continue(checkpoint, paths[name], schedule, validation, initial, cadence)
        (out / "continuation_history.json").write_text(json.dumps(training, indent=2) + "\n")
    del original_plan, expanded_plan, original, extra, expanded, validation

    decoder_config = decoder_dir / "selected_decoder.json"
    lm_path = decoder_dir / "training_character_lm.json"
    settings = BeamSettings(**json.loads(decoder_config.read_text(encoding="utf-8")))
    lm = CharacterModel.load(lm_path)
    results = defaultdict(dict)
    comparison = {}
    traces = defaultdict(dict)
    torch.set_num_threads(2)
    with multiprocessing.get_context("spawn").Pool(workers, _init_worker, (lm,)) as pool:
        with (out / "continuation_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for label, rows in tests.items():
                for name, path in paths.items():
                    model = load_checkpoint(path)
                    posteriors = _posteriors(model, rows)
                    results[label][name] = {}
                    for decoder, hypothesis in (("greedy", _greedy(posteriors)),
                                                ("fused", _beam(pool, posteriors, settings))):
                        score, details = _score(posteriors, hypothesis)
                        results[label][name][decoder] = score
                        traces[label, decoder][name] = details
                        for row in details:
                            stream.write(json.dumps({"partition": label, "model": name,
                                                     "decoder": decoder, **row}) + "\n")
                        print({"partition": label, "model": name, "decoder": decoder, "wer": score["wer"]},
                              flush=True)
                comparison[label] = {}
                for decoder in ("greedy", "fused"):
                    transcript_rows = traces[label, decoder]
                    comparison[label][decoder] = _speaker_delta(transcript_rows["original_data"],
                                                                 transcript_rows["expanded_data"])
    streaming = _stream_check(load_checkpoint(paths["expanded_data"]), tests["test_clean"])
    names = [name for files in SHARDS.values() for name in files] + list(ADDED_SHARDS) + [FRESH_TEST]
    summary = {"source_revision": REVISION, "checkpoint_sha256": _digest(checkpoint),
               "decoder_config_sha256": _digest(decoder_config), "fixed_lm_sha256": _digest(lm_path),
               "checkpoints_sha256": {name: _digest(path) for name, path in paths.items()},
               "shards_sha256": {name: _digest(data / name) for name in names}, "corpus": corpus,
               "batch_size": BATCH, "learning_rate": LEARNING_RATE, "seed": SEED,
               "torch_version": torch.__version__, "training_threads": 4, "results": results,
               "expanded_minus_original": comparison, "expanded_greedy_streaming": streaming,
               "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out / "continuation_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoint", "decoder-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=384)
    parser.add_argument("--validation-every", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if min(args.updates, args.validation_every, args.workers) < 1:
        parser.error("updates, validation interval, and workers must be positive")
    run(args.data, args.checkpoint, args.decoder_dir, args.out, args.updates,
        args.validation_every, args.workers)


if __name__ == "__main__":
    main()
