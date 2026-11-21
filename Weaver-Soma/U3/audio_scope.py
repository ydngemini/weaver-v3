"""Inspectable, advisory scope scores for the five-class audio research stem."""
from __future__ import annotations

from enum import IntEnum

import torch
from torch import nn
from torch.nn import functional as F

from audio_self_monitor import AUDIO_CLASSES, FEATURE_NAMES, SPECTRAL_GROUPS

SCOPE_FEATURES = tuple(
    name for name in FEATURE_NAMES
    if name in ("log_rms", "normalized_roughness") or name.startswith("spectral_group_")
)
SCOPE_COLUMNS = tuple(FEATURE_NAMES.index(name) for name in SCOPE_FEATURES)
ARCHITECTURE = "SOMA-U3-audio-scope-v1"
TEMPORAL_METHODS = ("temporal_prototype", "temporal_neighbors")
TEMPORAL_LAGS = (1, 5)
NEIGHBOR_COUNT = 3
TEMPORAL_FEATURES = tuple(
    f"spectral_group_{band}_mean_change_lag_{lag}"
    for lag in TEMPORAL_LAGS for band in range(SPECTRAL_GROUPS)
)


class ScopeReason(IntEnum):
    WITHIN_SCOPE = 0
    OUTSIDE_SCOPE = 1
    UNAVAILABLE = 2


def temporal_scope_features(tokens, valid):
    """Summarize short and longer spectral changes across consecutive valid tokens."""
    bands = min(64, tokens.shape[-1] // 2)
    spectral = F.adaptive_avg_pool1d(tokens[..., :bands].float(), SPECTRAL_GROUPS)
    summaries = []
    available = valid.any(dim=1)
    for lag in TEMPORAL_LAGS:
        if spectral.shape[1] <= lag:
            summaries.append(spectral.new_zeros((len(spectral), SPECTRAL_GROUPS)))
            available = torch.zeros_like(available)
            continue
        intervals = valid.unfold(1, lag + 1, 1).all(dim=-1)
        change = (spectral[:, lag:] - spectral[:, :-lag]).abs()
        change = change.masked_fill(~intervals[:, :, None], 0)
        count = intervals.sum(dim=1)
        summaries.append(change.sum(dim=1) / count.clamp_min(1)[:, None])
        available = available & (count > 0)
    return torch.cat(summaries, dim=1), available


class AudioScopeMonitor(nn.Module):
    """Higher scores indicate more similarity to the fitted in-scope examples."""

    def __init__(self, method: str, reference_files: tuple[str, ...] = ()):
        super().__init__()
        if method not in ("prototype", "exposure", *TEMPORAL_METHODS):
            raise ValueError("unknown scope scoring method")
        if method == "temporal_neighbors" and (
            not isinstance(reference_files, tuple)
            or len(reference_files) < NEIGHBOR_COUNT
            or not all(isinstance(name, str) and name for name in reference_files)
            or len(set(reference_files)) != len(reference_files)
        ):
            raise ValueError("invalid temporal neighbor reference files")
        self.method = method
        self.uses_temporal_features = method in TEMPORAL_METHODS
        self.reference_files = reference_files
        self.feature_names = SCOPE_FEATURES + (TEMPORAL_FEATURES if self.uses_temporal_features else ())
        self.loaded_checkpoint_sha256: str | None = None
        width = len(self.feature_names)
        self.register_buffer("center", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        self.register_buffer("threshold", torch.zeros(()))
        if method in ("prototype", "temporal_prototype"):
            self.register_buffer("prototypes", torch.zeros(len(AUDIO_CLASSES), width))
        elif method == "temporal_neighbors":
            self.register_buffer("references", torch.zeros(len(reference_files), width))
        else:
            self.register_buffer("weights", torch.zeros(width))
            self.register_buffer("intercept", torch.zeros(()))

    def forward(self, correctness, temporal=None):
        features = correctness["audit"]["features"][:, SCOPE_COLUMNS]
        available = correctness["score_available"]
        if self.uses_temporal_features:
            if temporal is None:
                raise ValueError("temporal scope scoring requires token evidence")
            temporal_features, temporal_available = temporal
            features = torch.cat((features, temporal_features), dim=1)
            available = available & temporal_available
        z = ((features - self.center) / self.scale).clamp(-8, 8)
        audit = {"feature_names": self.feature_names, "checkpoint_sha256": self.loaded_checkpoint_sha256}
        if self.method in ("prototype", "temporal_prototype", "temporal_neighbors"):
            by_neighbor = self.method == "temporal_neighbors"
            centers = self.references if by_neighbor else self.prototypes
            terms = -(z[:, None, :] - centers).square() / len(self.feature_names)
            nearest = terms.sum(dim=2).argsort(dim=1, descending=True, stable=True)
            rows = torch.arange(len(z), device=z.device)
            if by_neighbor:
                nearest = nearest[:, :NEIGHBOR_COUNT]
                contributions = terms[rows[:, None], nearest].mean(dim=1)
                audit["reference_files"] = self.reference_files
            else:
                nearest = nearest[:, :1]
                contributions = terms[rows, nearest[:, 0]]
                audit["prototype_classes"] = AUDIO_CLASSES
            intercept = z.new_zeros(len(z))
        else:
            contributions = z * self.weights
            intercept = self.intercept.expand(len(z))
        scores = contributions.sum(dim=1) + intercept
        available = available & torch.isfinite(scores)
        within_scope = available & (scores >= self.threshold)
        if "prototype_classes" in audit:
            audit["nearest_prototype"] = nearest[:, 0].masked_fill(~available, -1)
        if "reference_files" in audit:
            audit["nearest_reference_indices"] = nearest.masked_fill(~available[:, None], -1)
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
