"""Train SOMA's native acoustic checkpoint across LibriSpeech with rollback evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from pyarrow.parquet import ParquetFile
import torch
from torch import nn

from asr_prefix_beam import BeamSettings, CharacterModel, PrefixBeam
from benchmark_asr_beam import _beam, _greedy, _init_worker, _posteriors, _score, _validation_sample
from native_asr import ALPHABET, SAMPLE_RATE, load_checkpoint
from train_native_asr import REVISION, SHARDS, _audit, _batches, _evaluate, _min_alignment, _read, _training_step
from train_asr_data_scale import FRESH_TEST, _digest


PARTITIONS = ("clean/train.100/", "clean/train.360/", "other/train.500/")
RATES = (.00065, .00052, .00035)
SEED = 39871
REPLAY_PERIOD = 10


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _acoustic(state):
    return {"version": 1, "alphabet": ALPHABET, "state_dict": state}


def _snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _source_paths(catalog, partitions):
    record = json.loads(catalog.read_text(encoding="utf-8"))
    if record["sha"] != REVISION:
        raise ValueError("unexpected LibriSpeech catalog revision")
    files = [item["rfilename"] for item in record["siblings"]]
    selected = {}
    for partition in PARTITIONS[:partitions]:
        entries = sorted(path for path in files if path.startswith(partition) and path.endswith(".parquet"))
        if not entries:
            raise ValueError(f"no Parquet files for {partition}")
        selected[partition] = entries
    return selected


def _shard(data, scratch, source):
    local = data / source.removeprefix("clean/").replace("/", "-")
    if local.is_file():
        return local, False
    target = scratch / source.replace("/", "-")
    partial = target.with_suffix(".part")
    base = f"https://huggingface.co/datasets/openslr/librispeech_asr/resolve/{REVISION}/{source}"
    for attempt in range(3):
        try:
            subprocess.run(["curl", "-fsSL", "--retry", "1", "--max-time", "140",
                            f"{base}?download=true&attempt={attempt}", "-o", str(partial)], check=True)
            partial.replace(target)
            return target, True
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
    raise AssertionError("unreachable shard download state")


@contextmanager
def _prefetch(data, out, sources, depth=3):
    """Keep only a small rolling window of remotely retrieved shards."""
    with TemporaryDirectory(prefix=".prefetch-", dir=out) as temporary:
        with ThreadPoolExecutor(max_workers=depth) as network:
            def stream():
                pending = {}
                for offset, source in enumerate(sources):
                    if offset == 0:
                        for item in sources[:depth]:
                            pending[item] = network.submit(_shard, data, Path(temporary), item)
                    path, created = pending.pop(source).result()
                    try:
                        yield source, path
                    finally:
                        if created:
                            path.unlink(missing_ok=True)
                    if offset + depth < len(sources):
                        following = sources[offset + depth]
                        pending[following] = network.submit(_shard, data, Path(temporary), following)
            items = stream()
            try:
                yield items
            finally:
                items.close()


def _replay(data, model):
    clips = [row for name in SHARDS["train"] for row in _read(data / name, model, False)]
    clips.sort(key=lambda row: sha256(row.identifier.encode()).digest())
    counts = {}
    selected = []
    for row in clips:
        count = counts.get(row.speaker, 0)
        if count < 16 and _min_alignment(row.reference) <= len(row.features):
            counts[row.speaker] = count + 1
            selected.append(row)
    return _batches(selected)


def _fused(model, recording, settings, lm):
    posts = _posteriors(model, recording)
    results = []
    for item in posts:
        beam = PrefixBeam(settings, lm)
        beam.push(item.posterior)
        results.append(beam.text())
    return _score(posts, results)[0]


def _guard(model, validation, clean_sample, other_sample, settings, lm):
    greedy, _ = _evaluate(model, validation)
    clean = _fused(model, clean_sample, settings, lm)
    other = _fused(model, other_sample, settings, lm)
    return {"clean_greedy_wer": greedy["wer"], "clean_greedy_cer": greedy["cer_without_spaces"],
            "clean_fused_wer": clean["wer"], "other_fused_wer": other["wer"]}


def _dominates(candidate, incumbent):
    return (all(candidate[key] <= incumbent[key] for key in incumbent)
            and any(candidate[key] < incumbent[key] for key in incumbent))


def _decision(candidate, anchor):
    failures = []
    for split, decoders in anchor.items():
        for decoder, reference in decoders.items():
            for metric in ("wer", "cer_without_spaces"):
                if candidate[split][decoder][metric] > reference[metric]:
                    failures.append(f"{split}.{decoder}.{metric}")
    return failures


def _final_evaluation(data, out, base, selected, settings, lm, workers):
    base_models = {"anchor": base, "development_candidate": selected}
    tests = {"test_clean": _read(data / SHARDS["test"][0], load_checkpoint(base), False),
             "test_other": _read(data / FRESH_TEST, load_checkpoint(base), False)}
    _audit(tests)
    stats = {}
    torch.set_num_threads(2)
    with multiprocessing.get_context("spawn").Pool(workers, _init_worker, (lm,)) as pool:
        with (out / "heldout_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for name, checkpoint in base_models.items():
                model = load_checkpoint(checkpoint)
                stats[name] = {}
                for split, rows in tests.items():
                    posts = _posteriors(model, rows)
                    stats[name][split] = {}
                    for method, hyps in (("greedy", _greedy(posts)), ("fused", _beam(pool, posts, settings))):
                        scores, details = _score(posts, hyps)
                        stats[name][split][method] = scores
                        for row in details:
                            stream.write(json.dumps({"model": name, "split": split,
                                                     "decoder": method, **row}) + "\n")
                    print("held-out", name, split, {k: round(v["wer"], 5)
                                                  for k, v in stats[name][split].items()}, flush=True)
    regressions = _decision(stats["development_candidate"], stats["anchor"])
    return stats, regressions


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    base = out / "anchor.pt"
    if not base.exists():
        shutil.copyfile(args.checkpoint, base)
    config = args.decoder_dir / "selected_decoder.json"
    lm_path = args.decoder_dir / "training_character_lm.json"
    hashes = {"checkpoint": _digest(args.checkpoint), "decoder": _digest(config), "language_model": _digest(lm_path)}
    if _digest(base) != hashes["checkpoint"]:
        raise ValueError("the resume anchor differs from the supplied checkpoint")
    settings = BeamSettings(**json.loads(config.read_text()))
    lm = CharacterModel.load(lm_path)
    selected = out / "development_candidate.pt"
    restart = out / "resume.pt"
    groups = _source_paths(args.data / "hf_catalog.json", args.partitions)
    model = load_checkpoint(base)

    # Development and test speaker IDs are excluded before a shard can train.
    validation = _read(args.data / SHARDS["validation"][0], model, False)
    other = _read(args.data / "other-validation-0000.parquet", model, False)
    _audit({"validation": validation, "validation_other": other})
    forbidden = {row.speaker for row in validation + other}
    for filename in (SHARDS["test"][0], FRESH_TEST):
        file = ParquetFile(args.data / filename)
        for batch in file.iter_batches(columns=["speaker_id"]):
            forbidden.update(batch.column(0).to_pylist())
    clean_sample = _validation_sample(validation)
    other_sample = _validation_sample(other)
    del other

    optimizer = torch.optim.AdamW(model.parameters(), lr=RATES[0], weight_decay=.001)
    objective = nn.CTCLoss(blank=0)
    replay = [] if args.finish else _replay(args.data, model)
    if restart.exists():
        state = torch.load(restart, map_location="cpu", weights_only=True)
        if (state["input_sha256"] != hashes or state["source_revision"] != REVISION
                or state["validation_shards"] != args.validation_shards):
            raise ValueError("resume settings differ from the original run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        journal = state["journal"]
        seen = set(state["train_ids"])
        best = state["best_model"]
        _save(selected, _acoustic(best))
    else:
        if args.finish:
            raise ValueError("a completed training shard is required to finish")
        initial = _guard(model, validation, clean_sample, other_sample, settings, lm)
        journal = {"source_revision": REVISION, "input_sha256": hashes, "guard": initial,
                   "best_guard": initial, "best_hours": 0., "corpus_hours": 0., "replay_hours": 0.,
                   "updates": 0, "shards": [], "validation_checks": []}
        seen = set()
        best = _snapshot(model)
        _save(selected, _acoustic(best))
        _write_json(out / "training_journal.json", journal)
        print("initial gate", json.dumps(initial), flush=True)

    done = {item["file"] for item in journal["shards"]}
    for index, (partition, sources) in enumerate(groups.items()):
        if args.finish or (args.max_hours is not None and journal["corpus_hours"] >= args.max_hours):
            break
        remaining = [source for source in sources if source not in done]
        with _prefetch(args.data, out, remaining) as stream:
            for source, path in stream:
                content = _digest(path)
                rows = _read(path, model, False)
                ids = {row.identifier for row in rows}
                speakers = {row.speaker for row in rows}
                if len(ids) != len(rows) or seen.intersection(ids) or speakers.intersection(forbidden):
                    raise ValueError(f"duplicate utterance or evaluation speaker in {source}")
                seen.update(ids)

                aligned = [row for row in rows if _min_alignment(row.reference) <= len(row.features)]
                hours = sum(row.samples for row in aligned) / (3600 * SAMPLE_RATE)
                batches = _batches(aligned)
                tag = int.from_bytes(sha256(source.encode()).digest()[:8], "big")
                rng = np.random.default_rng(SEED + tag)
                start = perf_counter()
                losses = []
                for group in optimizer.param_groups:
                    group["lr"] = RATES[index]
                model.train()
                for batch_id in rng.permutation(len(batches)):
                    batches_for_step = [batches[batch_id]]
                    if journal["updates"] and journal["updates"] % REPLAY_PERIOD == 0:
                        added = replay[int(rng.integers(len(replay)))]
                        batches_for_step.append(added)
                        journal["replay_hours"] += sum(clip.samples for clip in added) / (3600 * SAMPLE_RATE)
                    for batch in batches_for_step:
                        optimizer.zero_grad(set_to_none=True)
                        loss = _training_step(model, objective, batch)
                        value = float(loss.detach())
                        if not math.isfinite(value):
                            raise ValueError(f"nonfinite training CTC loss in {source}")
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                        journal["updates"] += 1
                        losses.append(value)

                journal["corpus_hours"] += hours
                record = {"file": source, "sha256": content, "hours": hours,
                          "trained_utterances": len(aligned), "unalignable": len(rows) - len(aligned),
                          "training_seconds": perf_counter() - start,
                          "mean_ctc_loss": float(np.mean(losses))}
                journal["shards"].append(record)
                last = source == sources[-1]
                if len(journal["shards"]) % args.validation_shards == 0 or last:
                    score = _guard(model, validation, clean_sample, other_sample, settings, lm)
                    advance = _dominates(score, journal["best_guard"])
                    if advance:
                        journal["best_guard"], journal["best_hours"] = score, journal["corpus_hours"]
                        best = _snapshot(model)
                        _save(selected, _acoustic(best))
                    journal["validation_checks"].append({"hours": journal["corpus_hours"], "file": source,
                                                         "metrics": score, "accepted_on_dev": advance})
                    print("development gate", json.dumps(journal["validation_checks"][-1]), flush=True)

                _save(restart, {"source_revision": REVISION, "input_sha256": hashes,
                                "validation_shards": args.validation_shards, "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                                "journal": journal, "best_model": best, "train_ids": sorted(seen)})
                _write_json(out / "training_journal.json", journal)
                print("shard complete", json.dumps({"source": source, "shards": len(journal["shards"]),
                      "hours": round(journal["corpus_hours"], 2), "best_hours": round(journal["best_hours"], 2),
                      "mean_ctc_loss": record["mean_ctc_loss"], "training_seconds": record["training_seconds"]}),
                      flush=True)
                if args.max_hours is not None and journal["corpus_hours"] >= args.max_hours:
                    break
        if args.max_hours is not None and journal["corpus_hours"] >= args.max_hours:
            break

    if not args.finish and journal["shards"] and (
        not journal["validation_checks"] or journal["validation_checks"][-1]["hours"] != journal["corpus_hours"]
    ):
        score = _guard(model, validation, clean_sample, other_sample, settings, lm)
        advance = _dominates(score, journal["best_guard"])
        if advance:
            journal["best_guard"], journal["best_hours"] = score, journal["corpus_hours"]
            best = _snapshot(model)
            _save(selected, _acoustic(best))
        journal["validation_checks"].append({"hours": journal["corpus_hours"], "metrics": score,
                                             "accepted_on_dev": advance})
        # Keep the checkpoint used by future resumes consistent with the final validation.
        state = torch.load(restart, map_location="cpu", weights_only=True)
        state.update(journal=journal, best_model=best)
        _save(restart, state)
        _write_json(out / "training_journal.json", journal)
    print("training ended", json.dumps({"corpus_hours": journal["corpus_hours"],
                                        "best_hours": journal["best_hours"]}), flush=True)
    del validation, clean_sample, other_sample, replay, model, optimizer
    stats, failures = _final_evaluation(args.data, out, base, selected, settings, lm, args.workers)
    keep_anchor = bool(failures or not journal["best_hours"])
    shutil.copyfile(base if keep_anchor else selected, out / "retained.pt")
    report = {"training": journal, "heldout": stats, "heldout_regressions": failures,
              "retained_checkpoint": "anchor" if keep_anchor else "development_candidate"}
    _write_json(out / "training_report.json", report)
    print("retention", json.dumps({"selected": report["retained_checkpoint"], "regressions": failures}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoint", "decoder-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--partitions", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--validation-shards", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--finish", action="store_true", help="evaluate a saved run without updating weights")
    args = parser.parse_args()
    if min(args.validation_shards, args.threads, args.workers) < 1:
        parser.error("validation interval, threads, and workers must be positive")
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("the optional exposure limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
