"""Verify causality, packetized acoustic state, and CTC boundary rules."""
import unittest

import numpy as np
import torch

from native_asr import (
    SPAN, STRIDE, TOKEN_ID, ASRStream, CausalTranscriber, GreedyCTC, normalize, transcript,
)


class NativeASRTests(unittest.TestCase):
    def test_stream_logits_match_whole_clip(self):
        pcm = np.random.default_rng(19).integers(-20000, 20000, size=5811, dtype=np.int16)
        raw = pcm.astype("<i2").tobytes()
        ref = torch.from_numpy(pcm.astype(np.float32) / 32768)
        model = CausalTranscriber().eval()
        stream = ASRStream(model)
        pieces = []
        for offset in range(0, len(raw), 426):
            pieces.append(stream.push(raw[offset:offset + 426]))
            self.assertLess(len(stream.pending), SPAN)
        with torch.no_grad():
            whole, _ = model(model.features(ref[None]))
        torch.testing.assert_close(torch.cat(pieces), whole[0], atol=1e-5, rtol=1e-5)

    def test_acoustic_frame_waits_for_its_final_sample(self):
        stream = ASRStream(CausalTranscriber())
        self.assertEqual(len(stream.push(bytes(2 * (SPAN - 1)))), 0)
        self.assertEqual(len(stream.push(bytes(2))), 1)
        self.assertEqual(len(stream.push(bytes(2 * STRIDE))), 1)
        self.assertEqual(len(stream.push(b"")), 0)

    def test_ctc_blank_resets_repetition_across_packets(self):
        a, space, b = map(TOKEN_ID.__getitem__, "A B")
        decoder = GreedyCTC()
        self.assertEqual(decoder.push([a, a]), [("A", SPAN)])
        self.assertEqual(decoder.push([a, 0]), [])
        reference = [("A", SPAN + 4 * STRIDE), (" ", SPAN + 5 * STRIDE),
                     ("B", SPAN + 6 * STRIDE)]
        self.assertEqual(decoder.push([a, space, b]), reference)
        self.assertEqual(transcript([a, a, a, 0, a, space, b]), "AA B")
        self.assertEqual(normalize("  don't—STOP.  "), "DON'T STOP")


if __name__ == "__main__":
    unittest.main()
