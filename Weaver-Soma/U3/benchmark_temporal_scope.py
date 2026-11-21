"""Compare temporal class support against the frozen audio scope detector."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from audio_scope import ARCHITECTURE, SCOPE_COLUMNS, AudioScopeMonitor
from benchmark_audio_scope import (
    CALIBRATION, COVERAGE, EXPOSURE, TEST, UNKNOWN_CLASSES, audio_scope_trace,
    evaluate, extra_recordings, observe, operating_threshold, record, scope_readout,
)
from benchmark_esc50_audio import read_waveforms
from model import WeaverSOMAU3
from train_audio_self_monitor import TEST_CONDITIONS
from train_esc50 import CLASSES, prepare_data, sha256

LOOKALIKE_CLASSES = (
    "cat", "crow", "sheep", "car_horn", "water_drops", "drinking_sipping", "snoring", "door_wood_creaks",
)
CANDIDATES = ("temporal_prototype", "temporal_neighbors")
SECOND_OPINION_CLASSES = (
    "cow", "pig", "frog", "crying_baby", "laughing", "church_bells", "fireworks", "toilet_flush",
)


def sources(rows):
    return {row["src_file"] for row in rows}


@torch.no_grad()
def fit_candidates(observed, rows):
    dynamics, available = observed["temporal"]
    if not available.all():
        raise ValueError("all fitting clips must contain temporal evidence")
    x = torch.cat((observed["features"][:, SCOPE_COLUMNS], dynamics), dim=1)
    center = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(.025)
    transformed = ((x - center) / scale).clamp(-8, 8)
    labels = torch.tensor([CLASSES.index(row["category"]) for row in rows])
    files = tuple(row["filename"] for row in rows)
    prototypes = AudioScopeMonitor("temporal_prototype")
    neighbors = AudioScopeMonitor("temporal_neighbors", files)
    for probe in (prototypes, neighbors):
        probe.center.copy_(center)
        probe.scale.copy_(scale)
    prototypes.prototypes.copy_(torch.stack([
        transformed[labels == index].mean(0) for index in range(len(CLASSES))
    ]))
    neighbors.references.copy_(transformed)
    return dict(zip(CANDIDATES, (prototypes, neighbors)))


def readouts(recording, methods):
    return {name: scope_readout(recording, probe) for name, probe in methods.items()}


def scope_scores(result):
    return {name: row["score"] for name, row in result.items()}


def compare(known, unknown, thresholds):
    return {name: evaluate(values, unknown[name], thresholds[name]) for name, values in known.items()}


def joint_margin(arrays, candidate, thresholds, scales):
    baseline = (arrays["incumbent"] - thresholds["incumbent"]) / scales["incumbent"]
    temporal = (arrays[candidate] - thresholds[candidate]) / scales[candidate]
    return torch.minimum(baseline, temporal), (baseline <= temporal)


def selected_trace(readout):
    trace = audio_scope_trace(readout)
    audit = readout["audit"]
    if "reference_files" in audit:
        trace["reference_files"] = audit["reference_files"]
        trace["nearest_reference_indices"] = audit["nearest_reference_indices"].tolist()
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--monitor-checkpoint", type=Path, required=True)
    parser.add_argument("--incumbent-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(921)
    torch.set_num_threads(3)
    used_categories = set(CLASSES + EXPOSURE + CALIBRATION + TEST + UNKNOWN_CLASSES)
    if (used_categories.intersection(LOOKALIKE_CLASSES + SECOND_OPINION_CLASSES)
            or set(LOOKALIKE_CLASSES).intersection(SECOND_OPINION_CLASSES)):
        raise ValueError("new test categories must be disjoint from earlier experiments and each other")

    model = WeaverSOMAU3(audio_encoder="logmel", core_dim=64, depth=1, ff_dim=128).eval()
    monitor = model.load_audio_self_monitor(args.monitor_checkpoint)
    incumbent = model.load_audio_scope_monitor(args.incumbent_checkpoint)
    if incumbent.method != "prototype":
        raise ValueError("expected the frozen prototype scope checkpoint")
    known = prepare_data(args.data_dir, 8)
    fit_rows = [row for row in known if int(row["fold"]) <= 3]
    cal_rows = [row for row in known if row["fold"] == "4"]
    test_rows = [row for row in known if row["fold"] == "5"]
    unknown = extra_recordings(args.data_dir, (
        ("calibration", {4}, CALIBRATION), ("fresh", {5}, LOOKALIKE_CLASSES),
        ("previous_hard", {5}, TEST), ("previous_original", {5}, UNKNOWN_CLASSES),
        ("second_opinion", {5}, SECOND_OPINION_CLASSES),
    ))
    with (args.data_dir / "meta" / "esc50.csv").open(newline="", encoding="utf-8") as handle:
        prior_exposure = [
            row for row in csv.DictReader(handle)
            if int(row["fold"]) <= 3 and row["category"] in EXPOSURE
        ]
    former_sources = sources(known + prior_exposure + unknown["calibration"]
                             + unknown["previous_hard"] + unknown["previous_original"])
    excluded_fresh = [row for row in unknown["fresh"] if row["src_file"] in former_sources]
    unknown["fresh"] = [row for row in unknown["fresh"] if row["src_file"] not in former_sources]
    used_sources = former_sources | sources(unknown["fresh"])
    excluded_second = [row for row in unknown["second_opinion"] if row["src_file"] in used_sources]
    unknown["second_opinion"] = [
        row for row in unknown["second_opinion"] if row["src_file"] not in used_sources
    ]
    lock_sources = sources(fit_rows + cal_rows)
    heldout_sources = sources(test_rows + unknown["fresh"] + unknown["previous_hard"]
                              + unknown["previous_original"])
    excluded_cal = [
        row for row in unknown["calibration"] if row["src_file"] in heldout_sources
    ]
    unknown["calibration"] = [
        row for row in unknown["calibration"] if row["src_file"] not in heldout_sources
    ]
    if lock_sources & heldout_sources or sources(fit_rows) & sources(cal_rows + unknown["calibration"]):
        raise ValueError("underlying source recordings cross an experiment split")
    for split, classes in (("fresh", LOOKALIKE_CLASSES), ("second_opinion", SECOND_OPINION_CLASSES)):
        if set(classes) != {row["category"] for row in unknown[split]}:
            raise ValueError(f"at least one {split} category has no source-disjoint test clip")

    def get(rows, condition="clean"):
        waves = read_waveforms(args.data_dir, rows)
        return observe(model, waves, condition, include_temporal=True)

    fitted = fit_candidates(get(fit_rows), fit_rows)
    methods = {"incumbent": incumbent, **fitted}
    known_cal = scope_scores(readouts(get(cal_rows), methods))
    unknown_cal = scope_scores(readouts(get(unknown["calibration"]), methods))
    thresholds = {name: operating_threshold(known_cal[name]) for name in CANDIDATES}
    thresholds["incumbent"] = float(incumbent.threshold)
    for name in CANDIDATES:
        fitted[name].threshold.fill_(thresholds[name])
    calibration = compare(known_cal, unknown_cal, thresholds)
    pick = min(CANDIDATES, key=lambda name: (calibration[name]["unknown_accepted"],
                                            -calibration[name]["auroc"], name))
    selected = fitted[pick]
    joint_scales = {
        key: float((torch.quantile(known_cal[key], .75) - torch.quantile(known_cal[key], .25)).clamp_min(1e-6))
        for key in ("incumbent", pick)
    }
    joint_cal, _ = joint_margin(known_cal, pick, thresholds, joint_scales)
    joint_cutoff = operating_threshold(joint_cal)

    checkpoint = args.run_dir / "soma_u3_temporal_scope.pt"
    torch.save({
        "architecture": ARCHITECTURE, "method": pick, "weights": selected.state_dict(),
        "monitor_sha256": monitor.loaded_checkpoint_sha256, "reference_files": selected.reference_files,
    }, checkpoint)
    print(f"LOCKED before test scoring: {pick}, {thresholds}, joint={joint_cutoff}", flush=True)

    conditions = {
        condition: readouts(get(test_rows, condition), methods) for condition in TEST_CONDITIONS
    }
    unfamiliar = {
        split: readouts(get(unknown[split]), methods)
        for split in ("fresh", "previous_hard", "previous_original", "second_opinion")
    }
    familiar_clean = scope_scores(conditions["clean"])
    fresh = scope_scores(unfamiliar["fresh"])
    clean_test = compare(familiar_clean, fresh, thresholds)
    retrospective = {
        split: compare(familiar_clean, scope_scores(result), thresholds)
        for split, result in unfamiliar.items() if split in ("previous_hard", "previous_original")
    }
    by_category = {}
    for category in LOOKALIKE_CLASSES:
        positions = [i for i, row in enumerate(unknown["fresh"]) if row["category"] == category]
        by_category[category] = {
            "count": len(positions),
            "accepted": {name: int((values[positions] >= thresholds[name]).sum())
                         for name, values in fresh.items()},
        }
    candidate = clean_test[pick]
    baseline = clean_test["incumbent"]
    passed = {
        "familiar_acceptance_floor": candidate["known_accepted"] >= math.ceil(.8 * len(test_rows)),
        "improves_fresh_unknowns_at_matched_coverage":
            candidate["unknown_accepted_at_matched_known_coverage"]
            < baseline["unknown_accepted_at_matched_known_coverage"],
        "no_additional_fresh_unknown_acceptance_at_fixed_cutoff":
            candidate["unknown_accepted"] <= baseline["unknown_accepted"],
    }
    second_scores = scope_scores(unfamiliar["second_opinion"])
    joint_known, joint_known_limiter = joint_margin(familiar_clean, pick, thresholds, joint_scales)
    joint_unknown, joint_unknown_limiter = joint_margin(second_scores, pick, thresholds, joint_scales)
    second_test = {
        "incumbent": evaluate(familiar_clean["incumbent"], second_scores["incumbent"], thresholds["incumbent"]),
        "joint": evaluate(joint_known, joint_unknown, joint_cutoff),
    }
    summary = {
        "second_opinion_test": second_test, "second_opinion_unknown_count": len(unknown["second_opinion"]),
        "second_opinion_rule": {"cutoff": joint_cutoff, "iqr_scales": joint_scales},
        "second_opinion_categories": SECOND_OPINION_CLASSES,
        "excluded_second_opinion_for_source_overlap": [row["filename"] for row in excluded_second],
        "locked_temporal_candidate": pick, "thresholds": thresholds, "calibration_target_coverage": COVERAGE,
        "calibration": calibration, "clean_test": clean_test, "primary_gates": passed,
        "fresh_heldout_familiar": len(test_rows), "fresh_heldout_unfamiliar": len(unknown["fresh"]),
        "fit_known": len(fit_rows), "calibration_known": len(cal_rows),
        "calibration_unknown": len(unknown["calibration"]),
        "fresh_unknown_by_category": by_category,
        "historical_unknown_regression": retrospective,
        "hen_regression_accepted": {
            name: int(sum(bool(row["within_scope"][i])
                          for i, metadata in enumerate(unknown["previous_hard"]) if metadata["category"] == "hen"))
            for name, row in unfamiliar["previous_hard"].items()
        },
        "known_accepted_under_corruption": {
            condition: {name: int(row["within_scope"].sum()) for name, row in results.items()}
            for condition, results in conditions.items()
        },
        "previously_evaluated_familiar_test_fold": True,
        "fit_sources": [1, 2, 3], "calibration_fold": 4, "test_fold": 5,
        "excluded_calibration_for_source_overlap": [row["filename"] for row in excluded_cal],
        "excluded_fresh_for_source_overlap": [row["filename"] for row in excluded_fresh],
        "audio_monitor_sha256": monitor.loaded_checkpoint_sha256,
        "incumbent_sha256": incumbent.loaded_checkpoint_sha256,
        "temporal_checkpoint_sha256": sha256(checkpoint),
    }
    evidence = {
        "familiar": record(test_rows, True, familiar_clean, thresholds),
        "fresh_unfamiliar": record(unknown["fresh"], False, fresh, thresholds),
        "second_opinion_unfamiliar": record(
            unknown["second_opinion"], False, second_scores, thresholds,
        ),
        "second_opinion_trace": {
            "familiar_score": joint_known.tolist(), "familiar_limited_by_incumbent": joint_known_limiter.tolist(),
            "unfamiliar_score": joint_unknown.tolist(),
            "unfamiliar_limited_by_incumbent": joint_unknown_limiter.tolist(),
        },
        "selected_traces": {
            "familiar": selected_trace(conditions["clean"][pick]),
            "fresh_unfamiliar": selected_trace(unfamiliar["fresh"][pick]),
        },
        "recording_manifest": [
            {"file": row["filename"], "source": row["src_file"], "fold": int(row["fold"]),
             "category": row["category"], "sha256": sha256(args.data_dir / "audio" / row["filename"])}
            for row in fit_rows + cal_rows + test_rows + [row for group in unknown.values() for row in group]
        ],
    }
    for filename, data in (("temporal_metrics.json", summary), ("temporal_evidence.json", evidence)):
        (args.run_dir / filename).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
