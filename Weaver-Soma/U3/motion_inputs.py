"""Optional camera motion and inertial encoders for SOMA U3."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

VISION_MOTION_CLASSES = ("left", "right", "up", "down")
IMU_ACTIVITY_CLASSES = ("walking", "upstairs", "downstairs", "sitting", "standing", "lying")


@dataclass
class MotionEncoding:
    tokens: torch.Tensor
    valid: torch.Tensor
    logits: torch.Tensor
    evidence: dict[str, torch.Tensor]


def observed_rows(values, valid, name):
    if valid is None:
        return torch.ones(values.shape[:2], device=values.device, dtype=torch.bool)
    if valid.shape != values.shape[:2]:
        raise ValueError(f"{name} validity must match [batch, time]")
    return valid.bool()


def masked_mean(values, valid):
    return (values * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1).clamp_min(1).unsqueeze(-1)


def retain_observed_tokens(tokens, valid, limit):
    """Preserve valid tokens first and evenly subsample them only when over budget."""
    if tokens.shape[1] <= limit:
        return tokens, valid
    count = valid.sum(dim=1, keepdim=True)
    slots = torch.arange(limit, device=tokens.device)[None, :]
    ordered = (~valid).to(torch.int64).argsort(dim=1, stable=True)
    if limit == 1:
        ranks = (count - 1).clamp_min(0)
    else:
        spread = slots * (count - 1) // (limit - 1)
        ranks = torch.where(count > limit, spread, slots)
    select = ordered.gather(1, ranks)
    return tokens.gather(1, select[..., None].expand(-1, -1, tokens.shape[-1])), valid.gather(1, select)


class MotionVisionStem(nn.Module):
    """Compact causal encoder for one dominant contrasting image region.

    Centroids describe image-plane movement. They are not object tracks and
    can move when the camera or lighting moves.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.direction = nn.Linear(2, len(VISION_MOTION_CLASSES))

    def forward(self, values, *, valid=None, timestamps=None):
        if (values.ndim != 5 or values.shape[1] == 0 or values.shape[2] != 3
                or min(values.shape[-2:]) < 4 or not values.is_floating_point()):
            raise ValueError("camera motion expects float RGB [B,F,3,H>=4,W>=4]")
        observed = observed_rows(values, valid, "camera")
        if timestamps is None or timestamps.shape != observed.shape:
            raise ValueError("camera timestamps must be supplied in seconds as [B,F]")
        observed = observed & torch.isfinite(timestamps) & torch.isfinite(values).flatten(2).all(dim=-1)
        rgb = torch.where(observed[:, :, None, None, None], values, 0).clamp(0, 1)
        gray = rgb.mean(dim=2)
        height, width = gray.shape[-2:]
        border = torch.cat((gray[..., 0, :], gray[..., -1, :],
                            gray[..., 1:-1, 0], gray[..., 1:-1, -1]), dim=-1)
        contrast = (gray - border.mean(dim=-1)[..., None, None]).abs()
        mass = contrast.sum(dim=(-2, -1))
        x = torch.linspace(-1, 1, width, device=rgb.device, dtype=rgb.dtype)
        y = torch.linspace(-1, 1, height, device=rgb.device, dtype=rgb.dtype)
        cx = (contrast.sum(dim=-2) * x).sum(-1) / mass.clamp_min(1e-6)
        cy = (contrast.sum(dim=-1) * y).sum(-1) / mass.clamp_min(1e-6)
        center = torch.stack((cx, cy), dim=-1)
        usable = observed & (mass / (height * width) >= .008)
        time = torch.where(observed, timestamps, 0)
        dt = time[:, 1:] - time[:, :-1]
        pair = usable[:, 1:] & usable[:, :-1] & (dt > 0) & (dt <= .25)
        displacement = center[:, 1:] - center[:, :-1]
        velocity = F.pad(torch.where(pair[..., None], displacement / dt.to(rgb.dtype).clamp_min(.001)[..., None], 0),
                         (0, 0, 1, 0))
        paired = F.pad(pair, (1, 0), value=False)
        mean_velocity = masked_mean(velocity, paired)
        mean_speed = masked_mean(torch.linalg.vector_norm(velocity, dim=-1, keepdim=True), paired).squeeze(-1)
        net_speed = torch.linalg.vector_norm(mean_velocity, dim=-1)
        coherence = net_speed / mean_speed.clamp_min(1e-6)
        features = torch.cat((rgb.mean(dim=(-2, -1)), center, (mass / (height * width))[..., None],
                              velocity, paired[..., None], usable[..., None],
                              gray.std(dim=(-2, -1), correction=0)[..., None]), dim=-1)
        tokens = F.pad(features, (0, self.dim - features.shape[-1])) * observed[..., None]
        available = paired.any(dim=-1) & (net_speed >= .05) & (coherence >= .55)
        logits = self.direction(mean_velocity) * available[:, None]
        return MotionEncoding(tokens, observed, logits, {
            "score_available": available, "observed_frames": observed.sum(-1),
            "motion_pairs": paired.sum(-1), "pair_mask": paired,
            "mean_xy_per_second": mean_velocity,
            "motion_coherence": coherence,
            "class_velocity_contributions": mean_velocity[:, None, :] * self.direction.weight,
            "class_bias": self.direction.bias,
        })


class InertialActivityStem(nn.Module):
    """Causal residual convolutions for fixed-rate accel and gyro samples."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        width = min(dim, 40)
        self.project = nn.Conv1d(18, width, 1)
        self.layers = nn.ModuleList(nn.Conv1d(width, width, 5, dilation=d) for d in (1, 3, 9))
        self.classifier = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                        nn.Linear(width, len(IMU_ACTIVITY_CLASSES)))

    def forward(self, values, *, valid=None, quality=None):
        if values.ndim != 3 or values.shape[1] == 0 or values.shape[2] != 6 or not values.is_floating_point():
            raise ValueError("IMU expects float [B,T,6]: accel xyz in g, gyro xyz in rad/s")
        observed = observed_rows(values, valid, "IMU")[..., None] & torch.isfinite(values)
        if quality is None:
            quality = torch.ones_like(values)
        elif quality.shape != values.shape:
            raise ValueError("IMU quality must match [B,T,6]")
        observed = observed & torch.isfinite(quality) & (quality > 0)
        reliability = torch.where(observed, quality, 0).to(values.dtype).clamp(0, 1)
        safe = (torch.where(observed, values, 0) * reliability).clamp(-16, 16)
        pair = observed[:, 1:] & observed[:, :-1]
        change = F.pad(torch.where(pair, safe[:, 1:] - safe[:, :-1], 0), (0, 0, 1, 0))
        mask = observed.any(dim=-1)
        features = torch.cat((safe, 5 * change, reliability), dim=-1).transpose(1, 2)
        hidden = F.silu(self.project(features)) * mask[:, None]
        for conv in self.layers:
            past = conv.dilation[0] * (conv.kernel_size[0] - 1)
            hidden = (hidden + F.silu(conv(F.pad(hidden, (past, 0))))) * mask[:, None]
        hidden = hidden.transpose(1, 2)
        mean = masked_mean(hidden, mask)
        stdev = (masked_mean((hidden - mean[:, None]).square(), mask) + 1e-6).sqrt()
        count = mask.sum(-1)
        available = count >= 32
        logits = self.classifier(torch.cat((mean, stdev), dim=-1)) * available[:, None]
        return MotionEncoding(F.pad(hidden, (0, self.dim - hidden.shape[-1])), mask, logits, {
            "score_available": available, "observed_samples": count,
            "per_axis_samples": observed.sum(dim=1), "per_axis_adjacent_pairs": pair.sum(dim=1),
        })
