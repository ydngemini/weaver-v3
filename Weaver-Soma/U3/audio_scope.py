"""Inspectable, advisory scope scores for the five-class audio research stem."""
from __future__ import annotations

from enum import IntEnum

import torch
from torch import nn

from audio_self_monitor import AUDIO_CLASSES, FEATURE_NAMES

SCOPE_FEATURES = tuple(
    name for name in FEATURE_NAMES
    if name in ("log_rms", "normalized_roughness") or name.startswith("spectral_group_")
)
SCOPE_COLUMNS = tuple(FEATURE_NAMES.index(name) for name in SCOPE_FEATURES)
ARCHITECTURE = "SOMA-U3-audio-scope-v1"


class ScopeReason(IntEnum):
    WITHIN_SCOPE = 0
    OUTSIDE_SCOPE = 1
    UNAVAILABLE = 2


class AudioScopeMonitor(nn.Module):
    """Higher scores indicate more similarity to the fitted in-scope examples."""

    def __init__(self, method: str):
        super().__init__()
        if method not in ("prototype", "exposure"):
            raise ValueError("unknown scope scoring method")
        self.method = method
        self.loaded_checkpoint_sha256: str | None = None
        width = len(SCOPE_FEATURES)
        self.register_buffer("center", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        self.register_buffer("threshold", torch.zeros(()))
        if method == "prototype":
            self.register_buffer("prototypes", torch.zeros(len(AUDIO_CLASSES), width))
        else:
            self.register_buffer("weights", torch.zeros(width))
            self.register_buffer("intercept", torch.zeros(()))

    def forward(self, correctness):
        features = correctness["audit"]["features"][:, SCOPE_COLUMNS]
        z = ((features - self.center) / self.scale).clamp(-8, 8)
        audit = {"feature_names": SCOPE_FEATURES, "checkpoint_sha256": self.loaded_checkpoint_sha256}
        if self.method == "prototype":
            terms = -(z[:, None, :] - self.prototypes).square() / len(SCOPE_FEATURES)
            class_scores = terms.sum(dim=2)
            nearest = class_scores.argmax(dim=1)
            rows = torch.arange(len(z), device=z.device)
            contributions = terms[rows, nearest]
            intercept = z.new_zeros(len(z))
        else:
            contributions = z * self.weights
            intercept = self.intercept.expand(len(z))
        scores = contributions.sum(dim=1) + intercept
        available = correctness["score_available"] & torch.isfinite(scores)
        within_scope = available & (scores >= self.threshold)
        if self.method == "prototype":
            audit["nearest_prototype"] = nearest.masked_fill(~available, -1)
            audit["prototype_classes"] = AUDIO_CLASSES
        reason = torch.where(within_scope, ScopeReason.WITHIN_SCOPE, ScopeReason.OUTSIDE_SCOPE)
        reason = reason.masked_fill(~available, ScopeReason.UNAVAILABLE)
        audit.update({
            "features": features.masked_fill(~available[:, None], 0),
            "contributions": contributions.masked_fill(~available[:, None], 0),
            "intercept": intercept.masked_fill(~available, 0),
            "threshold": self.threshold,
        })
        return {
            "method": self.method, "score": scores.masked_fill(~available, 0),
            "score_available": available, "within_scope": within_scope,
            "recommend_abstain": ~within_scope, "decision_reason": reason, "audit": audit,
        }
