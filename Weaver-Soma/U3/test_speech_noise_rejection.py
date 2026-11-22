"""Check the validity of calibration and source-level false-accept counts."""
import unittest

import numpy as np
import torch

from native_speech_commands import LABELS
from train_speech_noise_rejection import Environment, _calibrate, _environment_result


def posterior(scores):
    result = torch.zeros(len(scores), len(LABELS))
    for row, (label, confidence) in enumerate(scores):
        result[row, LABELS.index(label)] = confidence
        result[row, LABELS.index("go")] = 1 - confidence
    return result


class NoiseRejectionTests(unittest.TestCase):
    def test_calibration_bounds_the_hardest_negative_group_and_ties(self):
        spoken = posterior([("yes", .9)] * 10 + [("silence", .9)] * 10)
        labels = torch.tensor([LABELS.index("unknown")] * 10 + [LABELS.index("silence")] * 10)
        environment = posterior([("yes", .95)] * 2 + [("silence", .8)] * 8)
        cutoff, rates = _calibrate(spoken, labels, environment)
        self.assertGreater(cutoff, float(environment[0, LABELS.index("yes")]))
        self.assertEqual(set(rates.values()), {0.})

    def test_sources_count_multiple_false_windows_only_once(self):
        empty = np.empty((0, 0))
        clips = [Environment("a.wav", "same", "rain", empty),
                 Environment("b.wav", "same", "rain", empty),
                 Environment("c.wav", "different", "clock_tick", empty)]
        scores = posterior([("yes", .9)] * 3 + [("silence", .9)] * 12)
        report = _environment_result(scores, clips, cutoff=.8)
        self.assertEqual(report["accepted_windows"], 3)
        self.assertEqual(report["windows"], 15)
        self.assertEqual(report["sources_with_a_false_accept"], 1)
        self.assertEqual(report["sources"], 2)


if __name__ == "__main__":
    unittest.main()
