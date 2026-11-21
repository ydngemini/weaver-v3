"""Grounded response and original-character provenance regressions."""
import unittest

import torch

from grounded_pointer import (
    UNSUPPORTED, FieldRanker, Route, answer_prompt, features, matching_fields, parse_prompt, select_evidence,
)


class GroundedPointerTests(unittest.TestCase):
    def test_unicode_value_is_copied_with_original_line_endings(self):
        prompt = "[A] food: arroz.\r\n[B] city: São Paulo 東京.\r\nQuestion: Which city?\r\nAnswer:"
        question, evidence = parse_prompt(prompt)
        self.assertEqual(question, "Which city?")
        reply = select_evidence(Route("city", .9, .8), evidence)
        self.assertEqual(reply.text(), "São Paulo 東京 [B]")
        span = reply.evidence[0]
        self.assertEqual(prompt[span.start:span.end], span.value)

    def test_missing_agreement_and_conflict_are_distinct(self):
        tail = "\nQuestion: Which city?\nAnswer:"
        city = Route("city", .9, .8)
        evidence = parse_prompt("[A] city: Rome.\n[B] city: Rome." + tail)[1]
        self.assertEqual(len(select_evidence(city, evidence).evidence), 2)
        conflict = parse_prompt("[A] city: Rome.\n[B] city: Paris." + tail)[1]
        denied = select_evidence(city, conflict)
        self.assertEqual(denied.status, "conflicting_evidence")
        self.assertEqual(denied.text(), "unknown")
        self.assertEqual(select_evidence(city, ()).status, "missing_evidence")
        self.assertEqual(select_evidence(Route(None, .7, .6), evidence).status, "unsupported_question")

    def test_scope_check_can_reject_a_confident_wrong_ranker(self):
        lexicon = {"city": ("city", "town"), "food": ("food", "dish")}
        vocabulary = sorted(features("Which city and dish are recorded?"))
        model = FieldRanker(("city", "food", UNSUPPORTED), vocabulary, lexicon=lexicon)
        with torch.no_grad():
            model.linear.weight.zero_()
            model.linear.bias.copy_(torch.tensor([9., -4., -4.]))
        self.assertEqual(matching_fields("town and dish", lexicon), ("city", "food"))
        question = "Which city and dish are recorded?"
        self.assertEqual(model.routes([question], .4, enforce_scope=False)[0].field, "city")
        self.assertIsNone(model.routes([question], .4)[0].field)
        prompt = "[A] city: Rome.\nQuestion: Which city?\nAnswer:"
        self.assertEqual(answer_prompt(prompt, model, .4).text(), "Rome [A]")

    def test_malformed_or_duplicate_evidence_is_rejected(self):
        tail = "\nQuestion: Which city?\nAnswer:"
        for cards in ("[A] city: Rome.\n[A] city: Paris.", "[A] city Rome."):
            with self.subTest(cards=cards), self.assertRaises(ValueError):
                parse_prompt(cards + tail)
