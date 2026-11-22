"""Prefix beam search and a transcript-only character model for SOMA CTC."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
import wave

import torch

from native_asr import ALPHABET, SAMPLE_RATE, STRIDE, ASRStream, load_checkpoint, normalize


START = "\x02"
END = "\x03"
SYMBOLS = ALPHABET[1:] + (END,)
NEG_INF = -math.inf


def _logadd(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    return a if b == NEG_INF else a + math.log1p(math.exp(b - a))


class CharacterModel:
    """Suffix backoff with Witten-Bell interpolation and finite unseen scores."""

    def __init__(self, order: int, counts: dict[str, dict[str, int]]):
        self.order = order
        self.counts = counts
        self.cache: dict[str, dict[str, float]] = {}
        self.totals = {key: (sum(row.values()), len(row)) for key, row in counts.items()}

    @classmethod
    def fit(cls, utterances, order: int = 5) -> CharacterModel:
        if order < 2:
            raise ValueError("character model order must be at least two")
        counts = defaultdict(Counter)
        for text in utterances:
            prefix = START * (order - 1)
            for symbol in normalize(text) + END:
                counts[""][symbol] += 1
                for n in range(1, order):
                    counts[prefix[-n:]][symbol] += 1
                prefix = (prefix + symbol)[-(order - 1):]
        if not counts:
            raise ValueError("character model needs training transcripts")
        return cls(order, {context: dict(row) for context, row in counts.items()})

    def log_distribution(self, prefix: str) -> dict[str, float]:
        n = self.order - 1
        history = (START * n + prefix[-n:])[-n:]
        while history not in self.counts:
            history = history[1:]
        if history not in self.cache:
            unigram = self.counts[""]
            base_sum, _ = self.totals[""]
            p = {c: (unigram.get(c, 0) + .1) / (base_sum + .1 * len(SYMBOLS)) for c in SYMBOLS}
            for width in range(1, len(history) + 1):
                suffix = history[-width:]
                counts = self.counts[suffix]
                total, escape = self.totals[suffix]
                p = {c: (counts.get(c, 0) + escape * prior) / (total + escape)
                     for c, prior in p.items()}
            self.cache[history] = {c: math.log(value) for c, value in p.items()}
        return self.cache[history]

    def save(self, path: Path) -> None:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"version": 1, "order": self.order, "counts": self.counts}, stream)
            stream.write("\n")

    @classmethod
    def load(cls, path: Path) -> CharacterModel:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["version"] != 1:
            raise ValueError("unsupported character model version")
        return cls(data["order"], data["counts"])


@dataclass(frozen=True)
class BeamSettings:
    width: int = 12
    top_tokens: int = 7
    lm_weight: float = 0.
    character_bonus: float = 0.

    def __post_init__(self):
        if self.width < 1 or not 1 <= self.top_tokens < len(ALPHABET):
            raise ValueError("invalid beam width or token pruning limit")
        if not math.isfinite(self.lm_weight) or self.lm_weight < 0 or not math.isfinite(self.character_bonus):
            raise ValueError("language weight and character bonus must be finite, with nonnegative weight")


class PrefixBeam:
    """Maintain separate blank and nonblank mass for each surviving collapsed string."""

    def __init__(self, settings: BeamSettings, lm: CharacterModel | None = None):
        if settings.lm_weight and lm is None:
            raise ValueError("positive language weight needs a character model")
        self.settings = settings
        self.lm = lm if settings.lm_weight else None
        self.mass = {"": [0., NEG_INF]}
        self.frames = 0

    def push(self, log_probs) -> None:
        for frame in log_probs:
            # Keep blank and the best acoustic nonblank candidates.
            row = frame.tolist() if hasattr(frame, "tolist") else frame
            order = heapq.nlargest(self.settings.top_tokens, range(1, len(ALPHABET)), key=row.__getitem__)
            candidates = [(ALPHABET[i], row[i]) for i in order if row[i] != NEG_INF]
            distribution = dict(zip(ALPHABET, row))
            updated: dict[str, list[float]] = {}
            weight = self.settings.lm_weight
            bonus = self.settings.character_bonus
            for prefix, (blank, spoken) in self.mass.items():
                total = _logadd(blank, spoken)
                last = prefix[-1:]
                same = updated.setdefault(prefix, [NEG_INF, NEG_INF])
                same[0] = _logadd(same[0], total + row[0])
                if last:
                    same[1] = _logadd(same[1], spoken + distribution[last])
                language = self.lm.log_distribution(prefix) if self.lm else None
                for character, score in candidates:
                    # Only a blank-terminated path can append a second identical label.
                    source = blank if last == character else total
                    if source == NEG_INF:
                        continue
                    score = source + score + bonus
                    if language is not None:
                        score += weight * language[character]
                    entry = updated.setdefault(prefix + character, [NEG_INF, NEG_INF])
                    entry[1] = _logadd(entry[1], score)
            winners = heapq.nlargest(self.settings.width, updated.items(),
                                     key=lambda entry: _logadd(*entry[1]))
            self.mass = dict(winners)
            self.frames += 1

    def _final_score(self, entry) -> float:
        text, mass = entry
        score = _logadd(*mass)
        if self.lm is not None:
            score += self.settings.lm_weight * self.lm.log_distribution(text)[END]
        return score

    def text(self) -> str:
        best = max(self.mass.items(), key=self._final_score)[0]
        return " ".join(best.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--settings", type=Path, help="validation-selected decoder settings")
    parser.add_argument("--lm", type=Path, help="character model trained on the training transcripts")
    args = parser.parse_args()
    torch.set_num_threads(1)
    settings = BeamSettings(**json.loads(args.settings.read_text())) if args.settings else BeamSettings()
    lm = CharacterModel.load(args.lm) if args.lm else None
    beam = PrefixBeam(settings, lm)
    stream = ASRStream(load_checkpoint(args.checkpoint))
    with wave.open(str(args.wav), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, SAMPLE_RATE):
            parser.error("expected mono 16 kHz PCM16 WAV")
        while pcm := source.readframes(STRIDE // 2):
            logits = stream.push(pcm)
            if len(logits):
                beam.push(logits.log_softmax(dim=-1).numpy())
    if not beam.frames:
        parser.error("audio is shorter than one acoustic frame")
    print(json.dumps({"text": beam.text(), "frames": beam.frames}, indent=2))


if __name__ == "__main__":
    main()
