"""Adapter and response-only training regressions."""
from copy import deepcopy
import unittest

import torch

from decoder_lora import add_lora, merge_lora
from native_text_decoder import DecoderConfig, NativeTextDecoder
from train_grounded_text import fixture, intervene, minibatch, training_rows
from train_native_text import Bytes


class GroundedTextTests(unittest.TestCase):
    def test_lora_identity_merge_and_trainable_parameters(self):
        torch.manual_seed(4)
        base = NativeTextDecoder(DecoderConfig(17, context=12, width=24, layers=1, heads=4)).eval()
        tokens = torch.randint(0, 17, (2, 8))
        adapted = add_lora(deepcopy(base), rank=2, alpha=4)
        with torch.no_grad():
            torch.testing.assert_close(adapted(tokens)[0], base(tokens)[0], rtol=0, atol=0)
        adaptable = dict(adapted.named_parameters())
        trainable = {name for name, param in adaptable.items() if param.requires_grad}
        self.assertEqual(len(trainable), 8)
        self.assertTrue(all(name.endswith((".a", ".b")) for name in trainable))
        optimizer = torch.optim.SGD([adaptable[name] for name in trainable], lr=.2)
        adapted(tokens)[0].square().mean().backward()
        optimizer.step()
        for name, value in base.state_dict().items():
            direct = adaptable.get(name)
            with_base = ".".join(name.split(".")[:-1] + ["base", name.split(".")[-1]])
            self.assertTrue(torch.equal(value, direct if direct is not None else adaptable[with_base]))
        with torch.no_grad():
            self.assertFalse(torch.equal(adapted(tokens)[0], base(tokens)[0]))
            torch.testing.assert_close(adapted(tokens)[0], merge_lora(adapted)(tokens)[0],
                                       rtol=1e-5, atol=1e-6)

    def test_response_loss_ignores_prompt_and_padding(self):
        codec = Bytes()
        rows = [{"prompt": "Q: city?\nA:", "expected": "unknown"},
                {"prompt": "Q: tool?\nA:", "expected": "pen [A]"}]
        features = training_rows(rows, codec, context=40)
        batch, target = minibatch(features, codec.eos)
        for row, labels in zip(rows, target):
            prefix_len = 1 + len(codec.encode(row["prompt"]))
            self.assertTrue((labels[:prefix_len - 1] == -100).all())
            payload = labels[labels != -100].tolist()
            self.assertEqual(payload, codec.encode(" " + row["expected"]) + [codec.eos])
        self.assertEqual(batch.shape, target.shape)

    def test_intervention_changes_answer_or_removes_evidence(self):
        used = set()
        rows = fixture(123, 24, used)
        groups = list(intervene(rows, used))
        self.assertGreater(len(groups), 0)
        for clean, swapped, removed in groups:
            self.assertNotEqual(clean["expected"], swapped["expected"])
            self.assertEqual(clean["expected"][-3:], swapped["expected"][-3:])
            self.assertEqual(removed["expected"], "unknown")
            self.assertEqual({r for r in clean["records"] if r[1] != clean["query"]}, set(removed["records"]))
