"""Check trial timing, missing cutoffs, and evidence separation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave

from score_voice_device_pilot import _summary, _trial, score


class VoiceDevicePilotTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        with wave.open(str(self.root / "room.wav"), "wb") as audio:
            audio.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            audio.writeframes(bytes(2 * 24000))
        with wave.open(str(self.root / "input.wav"), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(bytes(2 * 16000))

    def trace(self, *events, mode="device", endpoint=300):
        rows = [{"type": "run", "mode": mode, "endpoint_ms": endpoint},
                {"type": "device_block", "offset": 15680, "missing_input": False,
                 "output_samples": 320}, *events]
        (self.root / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def label(self, scenario, **fields):
        return {"trial_id": "one", "speaker_id": "p01", "pair_id": "script01",
                "scenario": scenario, "trace": "trace.jsonl", **fields}

    def test_independent_recording_controls_audible_latency(self):
        self.trace({"type": "response_interrupted"}, {"type": "silence_scheduled", "dac_s": 9.1})
        label = self.label("true_interruption", reference_wav="room.wav",
                           speech_onset_ref_sample=4800, audible_mute_ref_sample=10560)
        row = _trial(label, self.root, physical=True)
        self.assertEqual(row["audible_mute_ms"], 120)
        self.assertEqual(row["scheduled_mutes"], [9.1])

    def test_unmeasured_cutoff_counts_toward_tail(self):
        self.trace()
        label = self.label("true_interruption", reference_wav="room.wav",
                           speech_onset_ref_sample=4800, audible_mute_ref_sample=9600)
        missed = _trial(label, self.root, physical=True)
        ordinary = dict(missed, audible_mute_ms=100)
        result = _summary([ordinary] * 9 + [missed])
        self.assertEqual(result["missed_or_unmeasured_interruptions"], 1)
        self.assertEqual(result["audible_mute_p50_ms"], 100)
        self.assertIsNone(result["audible_mute_p95_ms"])

    def test_pause_marks_use_runner_capture_samples(self):
        self.trace({"type": "speech_started", "onset_sample": 1000},
                   {"type": "speech_stopped", "sample": 8000})
        label = self.label("within_turn_pause", input_wav="input.wav",
                           pause_start_input_sample=3200, resume_input_sample=9920)
        row = _trial(label, self.root, physical=True)
        self.assertEqual(row["pause_ms"], 420)
        self.assertTrue(row["premature_endpoint"])
        self.assertEqual(row["invalid_reasons"], [])

    def test_synthetic_trace_cannot_count_as_physical_recording(self):
        self.trace({"type": "response_interrupted"}, mode="synthetic_loopback")
        label = self.label("echo_only")
        manifest = self.root / "trials.json"
        manifest.write_text(json.dumps({"evidence": "physical", "trials": [label]}))
        with self.assertRaisesRegex(ValueError, "expected trace mode device"):
            score(manifest)
        manifest.write_text(json.dumps({"evidence": "synthetic", "trials": [label]}))
        result = score(manifest)
        self.assertFalse(result["physical_pilot_coverage_met"])
        self.assertEqual(result["by_endpoint"][300]["false_interruption_rate"], 1)


if __name__ == "__main__":
    unittest.main()
