"""Protect causal scoring and the streaming/replay match."""
import unittest

import numpy as np
import torch

from native_speech_commands import COMMANDS, LABELS, WINDOW, CommandStream, KeywordRecognizer, decision
from train_speech_commands import _score, _threshold


class StreamingSpeechTests(unittest.TestCase):
    def test_packetized_pcm_matches_the_batch_model(self):
        samples = np.random.default_rng(12).integers(-10000, 10000, size=3700, dtype=np.int16)
        pcm = samples.astype("<i2").tobytes()
        for temporal in (False, True):
            with self.subTest(temporal=temporal):
                model = KeywordRecognizer(temporal).eval()
                stream = CommandStream(model)
                last = None
                for offset in range(0, len(pcm), 386):
                    scores = stream.push(pcm[offset:offset + 386])
                    last = scores if scores is not None else last
                    self.assertLess(len(stream.pending), WINDOW)
                with torch.no_grad():
                    value = torch.from_numpy(samples.astype(np.float32) / 32768)
                    expected = model(model.features(value[None, :]))[0]
                torch.testing.assert_close(last, expected, atol=1e-5, rtol=1e-5)

    def test_sub_window_input_and_reset_are_causal(self):
        model = KeywordRecognizer()
        stream = CommandStream(model)
        self.assertIsNone(stream.push(bytes(2 * (WINDOW - 1))))
        self.assertEqual(stream.frame_count, 0)
        first = stream.push(bytes(2))
        stream.push(b"\xfe\x1f" * 640)
        fresh = CommandStream(model)
        torch.testing.assert_close(first, fresh.push(bytes(2 * WINDOW)))

    def test_rejection_threshold_preserves_tied_unsupported_budget(self):
        count = len(LABELS)
        probabilities = torch.full((12, count), .01)
        probabilities[:, 0] = .70
        probabilities[10:, 0] = .90
        labels = torch.tensor([10] * 5 + [11] * 5 + [0, 0])
        cutoff = _threshold(probabilities, labels)
        _, _, _, accepted = _score(probabilities, labels, cutoff)
        self.assertEqual(accepted.tolist(), [False] * 10 + [True, True])
        unsupported = torch.zeros(count)
        unsupported[len(COMMANDS)] = 10
        self.assertIsNone(decision(unsupported, 0.)["command"])


if __name__ == "__main__":
    unittest.main()
