"""Fit inspectable audio scope probes and evaluate on disjoint ESC-50 categories."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch

from audio_scope import ARCHITECTURE, SCOPE_COLUMNS, AudioScopeMonitor, temporal_scope_features
from benchmark_esc50_audio import BATCH_SIZE, read_waveforms
from model import WeaverSOMAU3
from train_audio_self_monitor import TEST_CONDITIONS, UNKNOWN_CLASSES, distort
from train_esc50 import CLASSES, SOURCE, fetch, prepare_data, sha256

EXPOSURE = ("helicopter", "hand_saw", "glass_breaking", "insects",
            "engine", "breathing", "clapping", "wind")
CALIBRATION = ("chirping_birds", "vacuum_cleaner", "door_wood_knock", "clock_alarm", "pouring_water")
TEST = ("hen", "thunderstorm", "washing_machine", "can_opening", "footsteps")
COVERAGE = 0.90


def extra_recordings(root, entries):
    with (root / "meta" / "esc50.csv").open(newline="", encoding="utf-8") as file:
        metadata = list(csv.DictReader(file))
    result = {}
    for name, folds, categories in entries:
        rows = [row for row in metadata if int(row["fold"]) in folds and row["category"] in categories]
        if len(rows) != 8 * len(folds) * len(categories):
            raise ValueError(f"unexpected number of ESC-50 recordings for {name}")
        result[name] = rows
    with ThreadPoolExecutor(max_workers=12) as pool:
        jobs = [pool.submit(fetch, f"{SOURCE}/audio/{row['filename']}",
                            root / "audio" / row["filename"])
                for rows in result.values() for row in rows]
        for job in jobs:
            job.result()
    return result


@torch.inference_mode()
def observe(model, sounds, condition="clean", *, include_temporal=False):
    waves, valid = distort(sounds, condition, noise_seed=1201)
    features, maximum, correctness, labels = [], [], [], []
    temporal_values, temporal_valid = [], []
    for wav, mask in zip(waves.split(BATCH_SIZE), valid.split(BATCH_SIZE)):
        tokens, token_valid = model.spectral_audio(wav, mask)
        logits = model.spectral_audio.classify(tokens, token_valid)[:, :len(CLASSES)]
        readout = model.audio_self_monitor(logits, wav, mask, tokens, token_valid)
        if not readout["score_available"].all():
            raise ValueError("the evaluation expects scoreable recordings")
        features.append(readout["audit"]["features"])
        maximum.append(logits.softmax(1).amax(1))
        correctness.append(readout["p_correct"])
        labels.append(logits.argmax(1))
        if include_temporal:
            changes, available = temporal_scope_features(tokens, token_valid)
            temporal_values.append(changes)
            temporal_valid.append(available)
    result = {"features": torch.cat(features), "softmax": torch.cat(maximum),
              "correctness": torch.cat(correctness), "labels": torch.cat(labels)}
    if include_temporal:
        result["temporal"] = (torch.cat(temporal_values), torch.cat(temporal_valid))
    return result


@torch.no_grad()
def fit_detectors(known, labels, exposure):
    x = known["features"][:, SCOPE_COLUMNS]
    y = exposure["features"][:, SCOPE_COLUMNS]
    center = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(.025)
    fit_x = ((x - center) / scale).clamp(-8, 8)
    fit_y = ((y - center) / scale).clamp(-8, 8)
    proto = AudioScopeMonitor("prototype")
    linear = AudioScopeMonitor("exposure")
    for probe in (proto, linear):
        probe.center.copy_(center)
        probe.scale.copy_(scale)
    proto.prototypes.copy_(torch.stack([fit_x[labels == idx].mean(0) for idx in range(len(CLASSES))]))
    fit = LogisticRegression(C=.2, class_weight="balanced", max_iter=500)
    features = torch.cat((fit_x, fit_y))
    fit.fit(features.numpy(), np.r_[np.ones(len(x)), np.zeros(len(y))])
    linear.weights.copy_(torch.from_numpy(fit.coef_[0]))
    linear.intercept.fill_(float(fit.intercept_[0]))
    reconstructed = features @ linear.weights + linear.intercept
    np.testing.assert_allclose(reconstructed.numpy(), fit.decision_function(features.numpy()),
                               atol=2e-6, rtol=1e-5)
    return {"prototype": proto, "exposure": linear}


def scope_readout(recording, probe):
    features = recording["features"]
    return probe({"audit": {"features": features},
                  "score_available": torch.ones(len(features), dtype=torch.bool)}, recording.get("temporal"))


def scores(recording, probes):
    return {"max_softmax": recording["softmax"], "correctness": recording["correctness"],
            **{key: scope_readout(recording, probe)["score"] for key, probe in probes.items()}}


def audio_scope_trace(readout):
    audit = readout["audit"]
    return {
        "feature_names": list(audit["feature_names"]), "features": audit["features"].tolist(),
        "contributions": audit["contributions"].tolist(), "intercept": audit["intercept"].tolist(),
        "score": readout["score"].tolist(), "cutoff": float(audit["threshold"]),
        **({"nearest_prototype": audit["nearest_prototype"].tolist(), "prototype_classes": CLASSES}
           if "nearest_prototype" in audit else {}),
    }


def operating_threshold(known):
    count = math.ceil(COVERAGE * len(known))
    return float(known.sort(descending=True).values[count - 1])


def evaluate(known, unfamiliar, threshold):
    choose = operating_threshold(known)
    labels = np.r_[np.ones(len(known)), np.zeros(len(unfamiliar))]
    return {
        "auroc": float(roc_auc_score(labels, torch.cat((known, unfamiliar)).numpy())),
        "known_accepted": int((known >= threshold).sum()),
        "unknown_accepted": int((unfamiliar >= threshold).sum()),
        "matched_known_accepted": int((known >= choose).sum()),
        "unknown_accepted_at_matched_known_coverage": int((unfamiliar >= choose).sum()),
    }


def record(rows, known, arrays, thresholds):
    return [
        {"filename": row["filename"], "category": row["category"], "in_scope": known,
         "scores": {name: float(vals[i]) for name, vals in arrays.items()},
         "accepted": {name: bool(arrays[name][i] >= cutoff) for name, cutoff in thresholds.items()}}
        for i, row in enumerate(rows)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--monitor-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(921)
    torch.set_num_threads(3)
    unknown_groups = (set(EXPOSURE), set(CALIBRATION), set(TEST), set(UNKNOWN_CLASSES))
    if len(set.union(*unknown_groups, set(CLASSES))) != sum(map(len, unknown_groups)) + len(CLASSES):
        raise ValueError("off-label categories must be disjoint")
    model = WeaverSOMAU3(audio_encoder="logmel", core_dim=64, depth=1, ff_dim=128).eval()
    monitor = model.load_audio_self_monitor(args.monitor_checkpoint)
    known = prepare_data(args.data_dir, 8)
    fit_rows = [r for r in known if int(r["fold"]) <= 3]
    cal_rows = [r for r in known if r["fold"] == "4"]
    test_rows = [r for r in known if r["fold"] == "5"]
    external = extra_recordings(args.data_dir, (
        ("exposure", {1, 2, 3}, EXPOSURE), ("calibration", {4}, CALIBRATION), ("test", {5}, TEST),
        ("previous_test", {5}, UNKNOWN_CLASSES),
    ))
    # Some names in different ESC-50 folds reuse an original source recording.
    # Keep the historical replication cohort intact and omit conflicting calibration clips.
    historical_sources = {row["src_file"] for row in external["previous_test"]}
    excluded_calibration = [row for row in external["calibration"] if row["src_file"] in historical_sources]
    external["calibration"] = [row for row in external["calibration"]
                               if row["src_file"] not in historical_sources]
    development = fit_rows + cal_rows + external["exposure"] + external["calibration"]
    checked = test_rows + external["test"] + external["previous_test"]
    if {r["src_file"] for r in development} & {r["src_file"] for r in checked}:
        raise ValueError("test source recordings overlap the development sources")
    fit_set = fit_rows + external["exposure"]
    cal_set = cal_rows + external["calibration"]
    if {r["src_file"] for r in fit_set} & {r["src_file"] for r in cal_set}:
        raise ValueError("fit and calibration source recordings overlap")

    def get(rows, condition="clean"):
        return observe(model, read_waveforms(args.data_dir, rows), condition)

    labels = torch.tensor([CLASSES.index(r["category"]) for r in fit_rows])
    probes = fit_detectors(get(fit_rows), labels, get(external["exposure"]))
    known_cal = scores(get(cal_rows), probes)
    unknown_cal = scores(get(external["calibration"]), probes)
    cutoffs = {name: operating_threshold(vals) for name, vals in known_cal.items()}
    calibration = {
        name: evaluate(vals, unknown_cal[name], cutoffs[name]) for name, vals in known_cal.items()
    }
    method = min(probes, key=lambda name: (calibration[name]["unknown_accepted"],
                                         -calibration[name]["auroc"], name))
    selected = probes[method]
    selected.threshold.fill_(cutoffs[method])
    candidate = args.run_dir / "soma_u3_audio_scope.pt"
    torch.save({"architecture": ARCHITECTURE, "method": method,
                "monitor_sha256": monitor.loaded_checkpoint_sha256,
                "weights": selected.state_dict()}, candidate)
    print(f"LOCKED before test scoring: {method}, thresholds={cutoffs}", flush=True)

    test_wave = read_waveforms(args.data_dir, test_rows)
    previous_observations = get(external["previous_test"])
    fresh_observations = get(external["test"])
    familiar_observations = {condition: observe(model, test_wave, condition)
                             for condition in TEST_CONDITIONS}
    old_unknown = scores(previous_observations, probes)
    new_unknown = scores(fresh_observations, probes)
    known_test = {condition: scores(observed, probes)
                  for condition, observed in familiar_observations.items()}
    test = {
        name: evaluate(vals, new_unknown[name], cutoffs[name])
        for name, vals in known_test["clean"].items()
    }
    baseline = min(("max_softmax", "correctness"), key=lambda name: (
        calibration[name]["unknown_accepted"], -calibration[name]["auroc"], name,
    ))
    def joint_accepted(values):
        return int(((values[method] >= cutoffs[method])
                    & (values["correctness"] >= monitor.threshold)).sum())

    report = {
        "paired_monitor_sha256": monitor.loaded_checkpoint_sha256,
        "scope_checkpoint_sha256": sha256(candidate), "method_locked_before_testing": method,
        "baseline_selected_on_calibration": baseline, "calibration_target_known_coverage": COVERAGE,
        "thresholds": cutoffs, "calibration": calibration, "test_clean": test,
        "calibration_known_count": len(cal_rows), "calibration_unknown_count": len(external["calibration"]),
        "calibration_clips_excluded_for_shared_historical_source": [r["filename"] for r in excluded_calibration],
        "clean_test_known_count": len(test_rows), "fresh_unknown_count": len(external["test"]),
        "fresh_unknown_classes": TEST, "exposure_classes": EXPOSURE, "calibration_unknown_classes": CALIBRATION,
        "historical_unknown": {name: int((values >= cutoffs[name]).sum()) for name, values in old_unknown.items()},
        "test_known_accepted_under_corruption": {
            condition: {name: int((values >= cutoffs[name]).sum()) for name, values in methods.items()}
            for condition, methods in known_test.items()
        },
        "fresh_unknown_by_class": {
            category: {
                name: int(sum(bool(values[i] >= cutoffs[name])
                              for i, row in enumerate(external["test"]) if row["category"] == category))
                for name, values in new_unknown.items()
            }
            for category in TEST
        },
        "joint_threshold_secondary_diagnostic": {
            "original_correctness_cutoff": float(monitor.threshold),
            "known_accepted": {condition: joint_accepted(values) for condition, values in known_test.items()},
            "fresh_unknown_accepted": joint_accepted(new_unknown),
        },
        "test_is_from_same_dataset": True, "known_test_fold_previously_evaluated": True,
        "fixed_familiar_floor_met": test[method]["known_accepted"] >= math.ceil(.8 * len(test_rows)),
        "matched_coverage_improves_over_selected_baseline":
            test[method]["unknown_accepted_at_matched_known_coverage"]
            < test[baseline]["unknown_accepted_at_matched_known_coverage"],
    }
    evidence = {
        "known_heldout": record(test_rows, True, known_test["clean"], cutoffs),
        "fresh_unknown": record(external["test"], False, new_unknown, cutoffs),
        "historical_unknown": record(external["previous_test"], False, old_unknown, cutoffs),
        "selected_scope_traces": {
            "known_heldout": audio_scope_trace(scope_readout(familiar_observations["clean"], selected)),
            "fresh_unknown": audio_scope_trace(scope_readout(fresh_observations, selected)),
            "historical_unknown": audio_scope_trace(scope_readout(previous_observations, selected)),
        },
        "manifest": [
            {"file": row["filename"], "source": row["src_file"], "fold": int(row["fold"]),
             "category": row["category"], "sha256": sha256(args.data_dir / "audio" / row["filename"])}
            for row in development + checked
        ],
    }
    for filename, payload in (("scope_metrics.json", report), ("scope_evidence.json", evidence)):
        (args.run_dir / filename).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
