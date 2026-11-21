"""Read-only audio metacognition for the native five-class SOMA encoder."""
from __future__ import annotations

from enum import IntEnum

import torch
from torch import nn
from torch.nn import functional as F

AUDIO_CLASSES = ("dog", "rain", "siren", "coughing", "sneezing")
SPECTRAL_GROUPS = 8
FEATURE_NAMES = (
    *(f"probability_{name}" for name in AUDIO_CLASSES),
    "top_probability", "probability_margin", "entropy", "valid_sample_fraction",
    "log_rms", "clipped_fraction", "normalized_roughness",
    *(f"spectral_group_{index}_mean" for index in range(SPECTRAL_GROUPS)),
    *(f"spectral_group_{index}_spread" for index in range(SPECTRAL_GROUPS)),
)
FEATURE_COUNT = len(FEATURE_NAMES)


class AudioDecisionReason(IntEnum):
    AT_OR_ABOVE_THRESHOLD = 0
    BELOW_THRESHOLD = 1
    MISSING_SAMPLES = 2
    ZERO_SIGNAL = 3
    NO_ENCODED_EVIDENCE = 4
    NONFINITE_EVIDENCE = 5


def spectral_input(values, valid, quality=None):
    """Respect explicit validity, signal quality, and finite readings."""
    usable = valid.bool() & torch.isfinite(values)
    if quality is not None:
        usable = usable & torch.isfinite(quality) & (quality > 0)
        safe_quality = quality.masked_fill(~usable, 0).clamp(0, 1)
        return values.masked_fill(~usable, 0) * safe_quality, usable
    return values.masked_fill(~usable, 0), usable


def monitor_features(class_logits, waveform, sample_valid, tokens, token_valid):
    """Use only information available when the classifier makes its decision."""
    probs = class_logits.float().softmax(dim=-1)
    top = probs.topk(2, dim=-1).values
    safe = waveform.float().masked_fill(~sample_valid, 0)
    count = sample_valid.sum(dim=1).clamp_min(1)
    rms = (safe.square().sum(dim=1) / count).clamp_min(1e-8).sqrt()
    clipped = ((safe.abs() >= 0.95) & sample_valid).sum(dim=1) / count
    adjacent = sample_valid[:, 1:] & sample_valid[:, :-1]
    differences = (safe[:, 1:] - safe[:, :-1]).masked_fill(~adjacent, 0)
    roughness = differences.square().sum(dim=1) / adjacent.sum(dim=1).clamp_min(1)
    health = torch.stack((
        top[:, 0], top[:, 0] - top[:, 1],
        -(probs * probs.clamp_min(1e-8).log()).sum(dim=1),
        sample_valid.float().mean(dim=1), rms.log(), clipped,
        roughness.clamp_min(1e-8).sqrt() / rms,
    ), dim=1)
    bands = min(64, tokens.shape[-1] // 2)
    spectral = F.adaptive_avg_pool1d(tokens[..., :bands].float(), SPECTRAL_GROUPS)
    mask = token_valid.unsqueeze(-1)
    frames = mask.sum(dim=1).clamp_min(1)
    mean = spectral.masked_fill(~mask, 0).sum(dim=1) / frames
    centered = (spectral - mean.unsqueeze(1)).masked_fill(~mask, 0)
    spread = (centered.square().sum(dim=1) / frames + 1e-8).sqrt()
    return torch.cat((probs, health, mean, spread), dim=1)


class AudioSelfMonitor(nn.Module):
    """Learned probability that a fixed native classifier chose the correct label."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(FEATURE_COUNT, 1)
        self.register_buffer("center", torch.zeros(FEATURE_COUNT))
        self.register_buffer("scale", torch.ones(FEATURE_COUNT))
        self.register_buffer("threshold", torch.tensor(0.5))
        self.loaded_checkpoint_sha256: str | None = None

    def standardize(self, features):
        return ((features - self.center) / self.scale).clamp(-8, 8).to(self.linear.weight.dtype)

    def log_odds(self, features):
        return self.linear(self.standardize(features)).squeeze(-1)

    def forward(self, class_logits, waveform, sample_valid, tokens, token_valid):
        observed = sample_valid.any(dim=1)
        token_present = token_valid.any(dim=1)
        has_signal = (waveform.ne(0) & sample_valid).any(dim=1)
        finite_logits = torch.isfinite(class_logits).all(dim=1)
        features = monitor_features(
            class_logits.masked_fill(~finite_logits.unsqueeze(1), 0),
            waveform, sample_valid, tokens, token_valid,
        )
        finite = finite_logits & torch.isfinite(features).all(dim=1)
        score_available = observed & has_signal & token_present & finite
        features = features.masked_fill(~score_available.unsqueeze(1), 0)
        standardized = self.standardize(features)
        log_odds = self.linear(standardized).squeeze(-1)
        scores = log_odds.sigmoid()
        reason = torch.full_like(observed, AudioDecisionReason.AT_OR_ABOVE_THRESHOLD, dtype=torch.long)
        reason.masked_fill_(scores < self.threshold, AudioDecisionReason.BELOW_THRESHOLD)
        reason.masked_fill_(~finite, AudioDecisionReason.NONFINITE_EVIDENCE)
        reason.masked_fill_(~token_present, AudioDecisionReason.NO_ENCODED_EVIDENCE)
        reason.masked_fill_(~has_signal, AudioDecisionReason.ZERO_SIGNAL)
        reason.masked_fill_(~observed, AudioDecisionReason.MISSING_SAMPLES)
        return {
            "p_correct": scores.masked_fill(~score_available, 0),
            "samples_observed": observed,
            "score_available": score_available,
            "recommend_abstain": reason != AudioDecisionReason.AT_OR_ABOVE_THRESHOLD,
            "decision_reason": reason,
            "audit": {
                "feature_names": FEATURE_NAMES,
                "features": features,
                "log_odds_contributions": (standardized * self.linear.weight).masked_fill(
                    ~score_available.unsqueeze(1), 0,
                ),
                "intercept": self.linear.bias.expand_as(log_odds).masked_fill(~score_available, 0),
                "log_odds": log_odds.masked_fill(~score_available, 0),
                "threshold": self.threshold,
                "loaded_checkpoint_sha256": self.loaded_checkpoint_sha256,
            },
        }
