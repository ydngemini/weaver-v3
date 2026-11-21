"""Score paired live voice trials against independently annotated audio."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import wave


ENDPOINTS = (300, 600)
INTERRUPTION = "true_interruption"
PAUSE = "within_turn_pause"
CONTROLS = {"echo_only", "clean_control"}
SCENARIOS = CONTROLS | {INTERRUPTION, PAUSE}


def _wave_header(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as audio:
        return audio.getframerate(), audio.getnframes()


def _sample(value, frames: int) -> int:
    if type(value) is not int or not 0 <= value < frames:
        raise ValueError(f"sample mark {value!r} lies outside the recording")
    return value


def _trial(trial: dict, directory: Path, physical: bool) -> dict:
    scenario = trial["scenario"]
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    with (directory / trial["trace"]).open(encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    mode = "device" if physical else "synthetic_loopback"
    if not events or events[0].get("type") != "run" or events[0].get("mode") != mode:
        raise ValueError(f"expected trace mode {mode}")
    endpoint = events[0]["endpoint_ms"]
    if endpoint not in ENDPOINTS:
        raise ValueError("expected the 300 or 600 ms endpoint control")
    blocks = [event for event in events if event["type"] == "device_block"]
    interrupts = [event for event in events if event["type"] == "response_interrupted"]
    reasons = []
    if not blocks or any(event["type"] == "device_error" for event in events):
        reasons.append("device_trace_error")
    if any(block["missing_input"] for block in blocks):
        reasons.append("missing_input")
    if scenario != PAUSE and not any(block["output_samples"] > 0 for block in blocks):
        reasons.append("no_assistant_playback")
    row = {"trial_id": trial["trial_id"], "speaker_id": trial["speaker_id"],
           "pair_id": trial["pair_id"], "scenario": scenario, "endpoint_ms": endpoint,
           "invalid_reasons": reasons, "interruptions": len(interrupts)}
    if scenario == INTERRUPTION:
        rate, frames = _wave_header(directory / trial["reference_wav"])
        onset = _sample(trial["speech_onset_ref_sample"], frames)
        mute = trial["audible_mute_ref_sample"]
        if mute is not None and _sample(mute, frames) < onset:
            raise ValueError("audible mute precedes the interruption")
        row["audible_mute_ms"] = None if mute is None or not interrupts else (
            1000 * (mute - onset) / rate
        )
        row["scheduled_mutes"] = [event["dac_s"] for event in events
                                  if event["type"] == "silence_scheduled"]
    elif scenario == PAUSE:
        rate, frames = _wave_header(directory / trial["input_wav"])
        if rate != events[0].get("sample_rate_hz", 16000):
            raise ValueError("pause marks require the runner's capture sample rate")
        start = _sample(trial["pause_start_input_sample"], frames)
        resume = _sample(trial["resume_input_sample"], frames)
        if resume <= start:
            raise ValueError("pause resume must follow pause start")
        if not blocks or blocks[-1]["offset"] + events[0].get("frame_samples", 320) != frames:
            reasons.append("capture_trace_length_mismatch")
        started = any(event["type"] == "speech_started"
                      and event["onset_sample"] < start for event in events)
        if not started:
            reasons.append("no_speech_detected_before_pause")
        row["pause_ms"] = (resume - start) * 1000 / rate
        row["premature_endpoint"] = any(
            event["type"] == "speech_stopped" and start <= event["sample"] < resume
            for event in events
        )
    return row


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    result = sorted(values)[math.ceil(fraction * len(values)) - 1]
    return round(result, 3) if math.isfinite(result) else None


def _wilson(count: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = count / total
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return [round(max(0., center - radius), 6), round(min(1., center + radius), 6)]


def _summary(rows: list[dict]) -> dict:
    cuts = [row for row in rows if row["scenario"] == INTERRUPTION]
    waits = [math.inf if row["audible_mute_ms"] is None else row["audible_mute_ms"]
             for row in cuts]
    controls = [row for row in rows if row["scenario"] in CONTROLS]
    false_count = sum(row["interruptions"] > 0 for row in controls)
    pauses = [row for row in rows if row["scenario"] == PAUSE]
    return {
        "counts_by_scenario": dict(Counter(row["scenario"] for row in rows)),
        "true_interruptions": len(cuts),
        "missed_or_unmeasured_interruptions": sum(not math.isfinite(wait) for wait in waits),
        "audible_mute_p50_ms": _percentile(waits, .50),
        "audible_mute_p95_ms": _percentile(waits, .95),
        "false_interruption_events": false_count,
        "control_events": len(controls),
        "false_interruption_rate": false_count / len(controls) if controls else None,
        "false_interruption_wilson95_trialwise": _wilson(false_count, len(controls)),
        "premature_pause_endpoints": sum(row["premature_endpoint"] for row in pauses),
        "pause_events": len(pauses),
    }


def score(manifest: Path) -> dict:
    labels = json.loads(manifest.read_text(encoding="utf-8"))
    evidence = labels["evidence"]
    if evidence not in {"physical", "synthetic"}:
        raise ValueError("expected physical or synthetic evidence")
    rows = []
    ids = set()
    traces = set()
    pairs: dict[tuple[str, str, str], set[int]] = {}
    for label in labels["trials"]:
        trial_id = label["trial_id"]
        trace = (manifest.parent / label["trace"]).resolve()
        if trial_id in ids or trace in traces:
            raise ValueError(f"{trial_id}: trial IDs and trace files must be unique")
        ids.add(trial_id)
        traces.add(trace)
        try:
            row = _trial(label, manifest.parent, physical=evidence == "physical")
        except (KeyError, ValueError, OSError, wave.Error) as error:
            raise ValueError(f"{trial_id}: {error}") from error
        key = row["speaker_id"], row["pair_id"], row["scenario"]
        variants = pairs.setdefault(key, set())
        if row["endpoint_ms"] in variants:
            raise ValueError(f"{trial_id}: repeated endpoint for the same speaker, pair and scenario")
        variants.add(row["endpoint_ms"])
        rows.append(row)
    valid = [row for row in rows if not row["invalid_reasons"]]
    invalid = len(rows) - len(valid)
    speakers = len({row["speaker_id"] for row in valid})
    pair_count = sum(variants == set(ENDPOINTS) for variants in pairs.values())
    by_endpoint = {ms: _summary([row for row in valid if row["endpoint_ms"] == ms])
                   for ms in ENDPOINTS}
    coverage_ready = (evidence == "physical" and len(valid) >= 200 and speakers >= 10
                      and not invalid and pair_count == len(pairs))
    by_endpoint_gate = {}
    for ms, result in by_endpoint.items():
        p95 = result["audible_mute_p95_ms"]
        false_rate = result["false_interruption_rate"]
        by_endpoint_gate[ms] = {
            "all_scenarios_observed": set(result["counts_by_scenario"]) == SCENARIOS,
            "latency_target_met": p95 is not None and p95 <= 150,
            "false_interruption_target_met": false_rate is not None and false_rate < .02,
        }
    return {
        "evidence": evidence,
        "valid_trials": len(valid), "invalid_trials": invalid, "speakers": speakers,
        "complete_pairs": pair_count, "pair_groups": len(pairs),
        "physical_pilot_coverage_met": coverage_ready, "by_endpoint": by_endpoint,
        "physical_gate_components_by_endpoint": by_endpoint_gate,
        "trials": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = score(args.manifest)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"voice pilot: {error}\n")


if __name__ == "__main__":
    main()
