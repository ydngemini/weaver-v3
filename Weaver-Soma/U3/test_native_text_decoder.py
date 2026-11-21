"""Causal and cache parity regressions for the native text decoder."""
import unittest

import torch

from native_text_decoder import DecoderConfig, NativeTextDecoder, complete


class NativeTextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = NativeTextDecoder(DecoderConfig(vocab_size=23, context=16, width=24, heads=4)).eval()
        self.tokens = torch.randint(0, 23, (2, 11))

    def test_future_tokens_do_not_change_past_scores(self):
        with torch.inference_mode():
            original, _ = self.model(self.tokens)
            edited = self.tokens.clone()
            edited[:, 7:] = (edited[:, 7:] + 1) % 23
            changed, _ = self.model(edited)
        torch.testing.assert_close(original[:, :7], changed[:, :7], rtol=0, atol=0)

    def test_cache_matches_full_prefix_with_chunked_and_single_token_appends(self):
        with torch.inference_mode():
            expected, _ = self.model(self.tokens)
            first, cache = self.model(self.tokens[:, :4], use_cache=True)
            second, cache = self.model(self.tokens[:, 4:10], cache=cache, use_cache=True)
            tail, cache = self.model(self.tokens[:, 10:], cache=cache, use_cache=True)
        torch.testing.assert_close(torch.cat((first, second, tail), dim=1), expected)
        self.assertEqual(cache[0][0].shape[-2], 11)

    def test_completion_enforces_context_bound_and_does_not_mutate_prompt(self):
        prompt = [12, 8, 2]
        pieces = complete(self.model, prompt, max_new=10)
        self.assertEqual(prompt, [12, 8, 2])
        self.assertEqual(len(pieces), 10)
        with self.assertRaises(ValueError):
            complete(self.model, prompt, max_new=14)
