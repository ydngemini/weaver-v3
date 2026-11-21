"""Regression tests for evidence masking, adaptive budgets and native fusion."""
import unittest

import torch

from adaptive_inputs import AdaptiveSignalEncoder, select_tokens, spatial_point_features, tactile_grid_features
from model import LEGACY_OUTPUTS, WeaverSOMAU3


class AdaptiveInputsTests(unittest.TestCase):
    def test_router_reserves_coverage_and_keeps_an_off_grid_change(self):
        tokens = torch.arange(60.).reshape(1, 60, 1).expand(2, -1, -1)
        valid = torch.ones(2, 60, dtype=torch.bool)
        valid[1] = False
        valid[1, [7, 31]] = True
        changes = torch.zeros(2, 60)
        changes[:, 31] = 1
        routed = select_tokens(tokens, valid, changes, 8)
        ids = routed.evidence["selected_input_indices"]
        self.assertEqual(ids[0, 0], 0)
        self.assertEqual(ids[0, -1], 59)
        self.assertIn(31, ids[0].tolist())
        self.assertEqual(ids[1, routed.valid[1]].tolist(), [7, 31])
        self.assertEqual(routed.evidence["retained_tokens"].tolist(), [8, 2])

    def test_filtered_sensor_tokens_are_causal_and_mask_gaps(self):
        torch.manual_seed(179)
        encoder = AdaptiveSignalEncoder(6, 32)
        signal = torch.randn(2, 40, 6)
        quality = torch.ones_like(signal)
        quality[:,  :5, :2] = 0
        quality[:, 20, :] = 0
        masked = torch.where(quality > 0, signal, float("nan"))
        time = torch.arange(40).float().expand(2, -1)
        full = encoder(masked, quality=quality, timestamps=time, limit=40)
        prefix = encoder(masked[:, :28], quality=quality[:, :28], timestamps=time[:, :28], limit=40)
        torch.testing.assert_close(full.tokens[:, :28], prefix.tokens)
        torch.testing.assert_close(full.tokens, encoder(signal, quality=quality, timestamps=time, limit=40).tokens)
        self.assertEqual(full.evidence["per_channel_delta_pairs"][0, 0].item(), 32)

    def test_grid_and_geometry_ignore_missing_measurements(self):
        grid = torch.tensor([[[[0., 0.], [0., 1.]]]])
        valid = torch.tensor([[[[False, True], [True, True]]]])
        features, count = tactile_grid_features(grid, valid)
        self.assertEqual(count.item(), 3)
        torch.testing.assert_close(features[0, 0, :4], torch.tensor([1/3, 1/3, 1., 1.]))
        points = torch.tensor([[[[1., 2., 3.], [3., 4., 5.], [100., 100., 100.]]]])
        features, count = spatial_point_features(points, torch.tensor([[[True, True, False]]]))
        self.assertEqual(count.item(), 2)
        torch.testing.assert_close(features[0, 0], torch.tensor([2., 3., 4., 1., 1., 1.]))

    def test_all_adaptive_profiles_through_u3(self):
        torch.manual_seed(91)
        model = WeaverSOMAU3(core_dim=64, ff_dim=128, depth=2, fusion_dim=32,
                            max_modality_tokens=8, enable_adaptive_inputs=True).eval()
        modalities = {name: {"values": torch.rand(2, 24, channels)}
                      for name, channels in (("touch", 6), ("olfaction", 8), ("rf", 8), ("spatial", 6))}
        modalities["touch_grid"] = {"values": torch.rand(2, 24, 3, 4)}
        modalities["spatial_points"] = {"values": torch.rand(2, 24, 6, 3)}
        modalities["gas_summary"] = {"values": torch.rand(2, 16, 8)}
        modalities["text"] = {"values": torch.randint(0, 255, (2, 24)),
                              "valid": torch.tensor([[True] * 24, [False] * 24])}
        args = (torch.zeros(2, 8, 7), torch.ones(2, 8, 3), torch.zeros(2, 8))
        with torch.inference_mode():
            before = model(*args)
            out = model(*args, modalities=modalities)
            self.assertEqual(len(out["adaptive_evidence"]), 7)
            self.assertEqual(int(out["sensory_token_mask"][0].sum()), 56)
            self.assertTrue(all(torch.equal(before[k], out[k]) for k in LEGACY_OUTPUTS))
            intent = out["text_intent"]
            torch.testing.assert_close(intent["logits"], intent["token_contributions"].sum(1) + intent["bias"])
            self.assertEqual(intent["score_available"].tolist(), [True, False])
            gas = out["gas_summary"]
            torch.testing.assert_close(gas["logits"], gas["sensor_contributions"].sum(1) + gas["bias"])
            for name, sample in modalities.items():
                if name == "text":
                    sample["valid"].fill_(False)
                else:
                    sample["values"].fill_(float("nan"))
            model.multimodal_gain.fill_(.5)
            missing = model(*args, modalities=modalities)
            self.assertTrue(all(torch.equal(before[k], missing[k]) for k in LEGACY_OUTPUTS))
            self.assertFalse(missing["sensory_token_mask"].any())
            self.assertFalse(missing["gas_summary"]["score_available"].any())
