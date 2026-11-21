"""Small, inspectable transcript model for optional speech endpointing."""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+", re.ASCII)
CONTINUATIONS = set("a an the if when while because or and but with to for on at from by of in".split())
FILLERS = {"um", "uh", "erm", "hmm"}


def words(text: str) -> list[str]:
    return TOKEN.findall(text.lower().replace("’", "'"))


def endpoint_features(tokens: list[str]) -> dict[str, float]:
    # Live partials often arrive without punctuation or case.
    if not tokens:
        return {}
    normalized = ["#" if token.isdecimal() else token for token in tokens]
    last = normalized[-1]
    result = {
        f"tail={last}": 1.0,
        f"length={min(len(tokens) // 5, 6)}": 1.0,
    }
    if len(tokens) > 1:
        result["tail2=" + " ".join(normalized[-2:])] = 1.0
    if last in CONTINUATIONS:
        result["unfinished_tail"] = 1.0
    if last in FILLERS:
        result["filler_tail"] = 1.0
    return result


@dataclass(frozen=True)
class TranscriptEndpointModel:
    bias: float
    weights: Mapping[str, float]
    threshold: float

    def __post_init__(self):
        weights = dict(self.weights)
        if (not isfinite(self.bias) or not isfinite(self.threshold)
                or not all(isinstance(name, str) and isfinite(weight) for name, weight in weights.items())):
            raise ValueError("endpoint weights and threshold must be finite")
        object.__setattr__(self, "weights", MappingProxyType(weights))

    @classmethod
    def load(cls, path: Path) -> TranscriptEndpointModel:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["version"] != 1:
            raise ValueError("unsupported transcript endpoint model version")
        return cls(data["bias"], data["weights"], data["threshold"])

    def evidence(self, text: str) -> tuple[float, dict[str, float]] | None:
        features = endpoint_features(words(text))
        if not features:
            return None
        contributions = {"bias": self.bias}
        contributions.update({name: value * self.weights[name]
                              for name, value in features.items() if name in self.weights})
        return sum(contributions.values()), contributions
