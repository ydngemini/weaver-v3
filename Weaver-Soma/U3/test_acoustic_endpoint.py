"""Regression checks for temporal leakage and cutoff calibration."""
import unittest

import numpy as np

from benchmark_acoustic_endpoint import (
    EARLY_MS, LATE_MS, LOOKBACK, SAMPLE_RATE, Event, _metrics, _operating_point,
    acoustic_features,
)


class AcousticEndpointTests(unittest.TestCase):
    def test_future_audio_cannot_change_features(self):
        rng = np.random.default_rng(17)
        pcm = rng.integers(-32000, 32000, size=2 * LOOKBACK, dtype=np.int16)
        original = acoustic_features(pcm, LOOKBACK)
        pcm[LOOKBACK:] = 0
        self.assertEqual(original, acoustic_features(pcm, LOOKBACK))

    def test_cutoff_counts_only_new_short_pause_cuts_and_respects_ties(self):
        short = Event("test", LOOKBACK, SAMPLE_RATE // 2, False, {})
        long = Event("test", LOOKBACK, SAMPLE_RATE, False, {})
        terminal = Event("test", LOOKBACK, SAMPLE_RATE, True, {})
        rows = [short] * 50 + [long, terminal]
        scores = np.array([2.0, 2.0] + [0.0] * 48 + [5.0, 3.0])
        cutoff = _operating_point(rows, scores)
        self.assertGreater(cutoff, 2.0)
        waits = np.where(scores >= cutoff, EARLY_MS, LATE_MS)
        results = _metrics(rows, waits)
        self.assertEqual(results["short_pause_false_cuts"], 0)
        self.assertEqual(results["all_pause_false_cuts"], 1)
        self.assertEqual(results["early_terminal_events"], 1)

    def test_empty_meeting_keeps_zero_counts_and_undefined_wait(self):
        results = _metrics([], np.array([], dtype=int))
        self.assertEqual(results["terminal_events"], 0)
        self.assertEqual(results["internal_pauses"], 0)
        self.assertIsNone(results["terminal_wait_p95_ms"])


if __name__ == "__main__":
    unittest.main()
