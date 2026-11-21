"""Resolve a structured evidence question and copy its source span exactly."""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import torch
from torch import nn

UNSUPPORTED = "__unsupported__"
CARD = re.compile(r"\[([A-Z])\] ([a-z][a-z_]*): (.+)\.")
WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class EvidenceSpan:
    source_id: str
    field: str
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class Route:
    field: str | None
    score: float
    margin: float
    matched_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundedAnswer:
    status: str
    route: Route
    evidence: tuple[EvidenceSpan, ...] = ()

    def text(self):
        if self.status != "answered":
            return "unknown"
        primary = self.evidence[0]
        return f"{primary.value} [{primary.source_id}]"


def parse_prompt(prompt):
    """Return question and provenance spans in Python string offsets."""
    if not isinstance(prompt, str):
        raise TypeError("evidence prompt must be a string")
    lines = prompt.splitlines(keepends=True)
    if len(lines) < 2 or lines[-1].rstrip("\r\n") != "Answer:":
        raise ValueError("expected a final Answer: line")
    question = lines[-2].rstrip("\r\n").removeprefix("Question: ")
    if question == lines[-2].rstrip("\r\n") or not question.strip():
        raise ValueError("expected a nonempty Question: line before Answer:")
    used = set()
    evidence = []
    offset = 0
    for line in lines[:-2]:
        match = CARD.fullmatch(line.rstrip("\r\n"))
        if match is None:
            raise ValueError("expected an evidence line of the form [A] field: value.")
        ident, field, value = match.groups()
        if ident in used or not value.strip():
            raise ValueError("evidence IDs must be unique and values nonempty")
        used.add(ident)
        evidence.append(EvidenceSpan(ident, field, value, offset + match.start(3), offset + match.end(3)))
        offset += len(line)
    return question, tuple(evidence)


def words(question):
    return WORD.findall(unicodedata.normalize("NFKC", question).casefold())


def features(question):
    tokens = words(question)
    result = {f"word={word}" for word in tokens}
    for word in tokens:
        bounded = "^" + word + "$"
        for size in (3, 4, 5):
            result.update(f"char={bounded[i:i + size]}" for i in range(len(bounded) - size + 1))
    return result


def matching_fields(question, lexicon):
    tokens = set(words(question))
    return tuple(field for field, aliases in lexicon.items() if tokens.intersection(aliases))


class FieldRanker(nn.Module):
    """A trained PyTorch ranker whose input is the question alone."""

    def __init__(self, labels, vocabulary, *, lexicon=None):
        super().__init__()
        self.labels = tuple(labels)
        self.vocabulary = tuple(vocabulary)
        self.indices = {word: i for i, word in enumerate(vocabulary)}
        self.lexicon = dict(lexicon) if lexicon is not None else None
        self.linear = nn.Linear(len(vocabulary), len(labels))

    def vectors(self, questions):
        data = self.linear.weight.new_zeros(len(questions), len(self.indices))
        for row, question in enumerate(questions):
            active = [self.indices[k] for k in features(question) if k in self.indices]
            if active:
                data[row, active] = len(active) ** -.5
        return data

    def forward(self, data):
        return self.linear(data)

    @torch.inference_mode()
    def routes(self, questions, threshold, *, enforce_scope=True):
        x = self.vectors(questions)
        scores, ids = self(x).softmax(-1).topk(2)
        routes = []
        for question, seen, top, order in zip(questions, x.any(-1).tolist(), scores.tolist(), ids.tolist()):
            best, second = top
            label = self.labels[order[0]]
            margin = best - second
            selected = label if seen and margin >= threshold and label != UNSUPPORTED else None
            matched = matching_fields(question, self.lexicon) if self.lexicon is not None else ()
            if enforce_scope and self.lexicon is not None and (selected is None or matched != (selected,)):
                selected = None
            routes.append(Route(selected, best, margin, matched))
        return routes


def select_evidence(route, evidence):
    if route.field is None:
        return GroundedAnswer("unsupported_question", route)
    matching = tuple(item for item in evidence if item.field == route.field)
    if not matching:
        return GroundedAnswer("missing_evidence", route)
    if len({item.value for item in matching}) != 1:
        return GroundedAnswer("conflicting_evidence", route, matching)
    return GroundedAnswer("answered", route, matching)


def answer_prompt(prompt, ranker, threshold):
    question, evidence = parse_prompt(prompt)
    return select_evidence(ranker.routes([question], threshold)[0], evidence)


def load_pointer(checkpoint):
    bundle = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if bundle.get("architecture") != "soma-u3-structured-evidence-v1":
        raise ValueError("unsupported evidence checkpoint")
    ranker = FieldRanker(bundle["labels"], bundle["vocabulary"], lexicon=bundle["lexicon"])
    ranker.load_state_dict(bundle["weights"], strict=True)
    return ranker.eval(), float(bundle["margin_threshold"])
