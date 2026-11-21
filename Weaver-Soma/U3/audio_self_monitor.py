"""Read-only audio metacognition for the native five-class SOMA encoder."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

AUDIO_CLASSES = ("dog", "rain", "siren", "coughing", "sneezing")
SPECTRAL_GROUPS = 8
FEATURE_COUNT = len(AUDIO_CLASSES) + 7 + 2 * SPECTRAL_GROUPS


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

    def log_odds(self, features):
        standardized = ((features - self.center) / self.scale).clamp(-8, 8)
        return self.linear(standardized.to(self.linear.weight.dtype)).squeeze(-1)

    def forward(self, class_logits, waveform, sample_valid, tokens, token_valid):
        observed = sample_valid.any(dim=1) & token_valid.any(dim=1)
        has_signal = (waveform.ne(0) & sample_valid).any(dim=1)
        finite = torch.isfinite(class_logits).all(dim=1)
        score_available = observed & has_signal & finite
        features = monitor_features(
            class_logits.masked_fill(~finite.unsqueeze(1), 0),
            waveform, sample_valid, tokens, token_valid,
        )
        scores = self.log_odds(features).sigmoid()
        return {
            "p_correct": scores.masked_fill(~score_available, 0),
            "samples_observed": observed,
            "score_available": score_available,
            "recommend_abstain": ~score_available | (scores < self.threshold),
        }
