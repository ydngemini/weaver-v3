"""Verify prefix probability, packet boundaries, and language smoothing."""
import itertools
import math
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from asr_prefix_beam import BeamSettings, CharacterModel, PrefixBeam
from native_asr import ALPHABET, TOKEN_ID, transcript


def frame(blank, a, b):
    row = [-math.inf] * len(ALPHABET)
    for key, value in (("", blank), ("A", a), ("B", b)):
        if value:
            row[TOKEN_ID[key]] = math.log(value)
    return row


class PrefixSearchTests(unittest.TestCase):
    def test_prefix_mass_matches_enumeration_of_tiny_alignments(self):
        audible = [frame(.3, .6, .1), frame(.4, .3, .3), frame(.2, .7, .1)]
        total = defaultdict(float)
        for alignment in itertools.product((0, TOKEN_ID["A"], TOKEN_ID["B"]), repeat=3):
            text = transcript(list(alignment))
            total[text] += math.exp(sum(row[token] for row, token in zip(audible, alignment)))
        beam = PrefixBeam(BeamSettings(width=128))
        beam.push(audible)
        self.assertEqual(beam.text(), max(total, key=total.get))
        for text, probability in total.items():
            self.assertAlmostEqual(sum(math.exp(p) for p in beam.mass[text]), probability, places=12)

    def test_repeat_crosses_chunks_only_after_a_blank(self):
        audible = [frame(0., 1., 0.), frame(0., 1., 0.), frame(1., 0., 0.), frame(0., 1., 0.)]
        stream = PrefixBeam(BeamSettings())
        for row in audible:
            stream.push([row])
        whole = PrefixBeam(BeamSettings())
        whole.push(audible)
        self.assertEqual(stream.text(), "AA")
        self.assertEqual(stream.mass, whole.mass)

    def test_language_model_persists_and_weights_an_ambiguous_token(self):
        lm = CharacterModel.fit(["A"] * 20)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "lm.json"
            lm.save(path)
            restored = CharacterModel.load(path)
        self.assertEqual(lm.log_distribution("PA"), restored.log_distribution("PA"))
        unseen = lm.log_distribution("UNKNOWN")
        self.assertAlmostEqual(sum(math.exp(logp) for logp in unseen.values()), 1.)
        acoustic = PrefixBeam(BeamSettings())
        language = PrefixBeam(BeamSettings(lm_weight=.5), restored)
        for decoder in (acoustic, language):
            decoder.push([frame(.01, .49, .50)])
        self.assertEqual(acoustic.text(), "B")
        self.assertEqual(language.text(), "A")


if __name__ == "__main__":
    unittest.main()
