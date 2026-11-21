"""Motion input regressions for missing evidence and causal token construction."""
import unittest

import torch

from motion_inputs import InertialActivityStem, MotionVisionStem, retain_observed_tokens


class MotionInputTests(unittest.TestCase):
    def test_video_reversal_and_no_direction_on_static_video(self):
        stem = MotionVisionStem(32)
        frames = torch.zeros(1, 4, 3, 24, 24)
        for i in range(4):
            frames[:, i, :, 5:13, 3 + i:11 + i] = 1
        time = torch.arange(4, dtype=torch.float64)[None] / 10
        right = stem(frames, timestamps=time)
        left = stem(frames.flip(1), timestamps=time)
        self.assertTrue(right.evidence["score_available"].item())
        self.assertGreater(right.evidence["mean_xy_per_second"][0, 0].item(), 0.)
        torch.testing.assert_close(left.evidence["mean_xy_per_second"], -right.evidence["mean_xy_per_second"])
        self.assertFalse(stem(frames[:, :1].expand_as(frames), timestamps=time).evidence["score_available"].item())
        self.assertFalse(stem(frames, timestamps=time, valid=torch.tensor([[1, 0, 1, 0]])).evidence[
            "score_available"].item())
        self.assertFalse(stem(frames, timestamps=time * 3).evidence["score_available"].item())

    def test_masked_imu_values_do_not_change_predictions_and_tokens_are_causal(self):
        torch.manual_seed(19)
        stem = InertialActivityStem(32)
        values = torch.randn(1, 64, 6)
        quality = torch.ones_like(values)
        quality[:, 8:16] = 0
        before = stem(values, quality=quality)
        masked = torch.where(quality.bool(), values, float("nan"))
        after = stem(masked, quality=quality)
        self.assertTrue(torch.equal(before.logits, after.logits))
        self.assertTrue(torch.equal(before.tokens, after.tokens))
        self.assertFalse(after.valid[0, 8:16].any())
        prefix = stem(masked[:, :48], quality=quality[:, :48])
        torch.testing.assert_close(after.tokens[:, :48], prefix.tokens)
        self.assertFalse(stem(values[:, :31]).evidence["score_available"].item())

    def test_budget_preserves_sparse_observations(self):
        tokens = torch.arange(10.).reshape(1, 10, 1).expand(2, -1, -1)
        valid = torch.tensor([[False, True, False, False, False, True, False, False, False, False],
                              [True] * 10])
        packed, present = retain_observed_tokens(tokens, valid, 3)
        self.assertEqual(packed[0, present[0], 0].tolist(), [1., 5.])
        self.assertEqual(packed[1, present[1], 0].tolist(), [0., 4., 9.])
