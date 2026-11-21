"""Replay the production-path transport with controlled pauses, echo, and interruption."""
import argparse
import json
from pathlib import Path

from live_voice import write_loopback


def summarize(events):
    fixture = events[0]
    cancels = [row for row in events if row["type"] == "response_interrupted"]
    muted = [row for row in events if row["type"] == "silence_scheduled"]
    if muted:
        stop = muted[0]
        stale = [row for row in events if row["type"] == "device_block"
                 and row["response_id"] == stop["response_id"] and row["dac_s"] >= stop["dac_s"]]
        assert not stale, stale
    onset = fixture["injected_user_second_start_adc_s"]
    energy_onset = fixture["reference_energy_onset_adc_s"]
    resume = fixture["resume_sample"]
    stopped = [row for row in events if row["type"] == "speech_stopped"]
    return {
        "transcripts": [row["text"] for row in events if row["type"] == "transcript_final"],
        "input_turns": sum(row["type"] == "speech_started" for row in events),
        "premature_pause_endpoints": None if resume is None else sum(row["sample"] < resume for row in stopped),
        "interruptions": cancels,
        "fixture_insertion_to_scheduled_mute_ms": None if onset is None or not muted else
            round(1000 * (muted[0]["dac_s"] - onset), 3),
        "fixture_energy_onset_to_scheduled_mute_ms": None if energy_onset is None or not muted else
            round(1000 * (muted[0]["dac_s"] - energy_onset), 3),
        "stale_output_after_scheduled_mute": 0 if muted else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    variants = {
        "interruption_300ms": {"endpoint_ms": 300, "interrupt": True},
        "interruption_600ms": {"endpoint_ms": 600, "interrupt": True},
        "pause_300ms": {"endpoint_ms": 300, "interrupt": False, "pause_ms": 420},
        "pause_600ms": {"endpoint_ms": 600, "interrupt": False, "pause_ms": 420},
        "echo_only": {"endpoint_ms": 300, "interrupt": False, "echo_gain": .8},
        "echo_fixture_subtraction": {"endpoint_ms": 300, "interrupt": False,
                                     "echo_gain": .8, "fixture_cancel": True},
    }
    result = {name: summarize(write_loopback(args.run_dir / f"{name}.jsonl", **kwargs))
              for name, kwargs in variants.items()}
    for name in ("interruption_300ms", "interruption_600ms"):
        scenario = result[name]
        assert len(scenario["interruptions"]) == 1 and len(scenario["transcripts"]) == 2, scenario
        assert scenario["interruptions"][0]["heard_text"] == "The local voice session is running.", scenario
        assert scenario["fixture_energy_onset_to_scheduled_mute_ms"] <= 150, scenario
    assert result["pause_300ms"]["premature_pause_endpoints"] == 1, result
    assert result["pause_600ms"]["premature_pause_endpoints"] == 0, result
    assert result["echo_only"]["input_turns"] > 1, result
    assert result["echo_fixture_subtraction"]["input_turns"] == 1, result
    (args.run_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
