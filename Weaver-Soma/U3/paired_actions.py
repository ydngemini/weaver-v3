"""Auditable trial-level fusion for Kinect joint coordinates and wearable IMU."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

PHASES = 8
ACTIONS = (
    "swipe left", "swipe right", "wave", "clap", "throw", "cross arms",
    "basketball shot", "draw X", "draw circle clockwise", "draw circle counterclockwise",
    "draw triangle", "bowl", "box", "baseball swing", "tennis forehand", "arm curl",
    "tennis serve", "push", "knock", "catch", "pick up and throw", "jog in place",
    "walk in place", "sit to stand", "stand to sit", "lunge", "squat",
)


def phase_mean(values, valid):
    """Summarize equal fractions of a segmented trial, with per-axis coverage."""
    safe = torch.where(valid, values, 0)
    mean = F.adaptive_avg_pool1d(safe.transpose(1, 2), PHASES)
    coverage = F.adaptive_avg_pool1d(valid.transpose(1, 2).to(values.dtype), PHASES)
    return (mean / coverage.clamp_min(1e-6)).transpose(1, 2), coverage.transpose(1, 2)


def skeleton_phases(values, valid=None):
    if values.ndim != 4 or values.shape[1] < PHASES or values.shape[2:] != (20, 3):
        raise ValueError("skeleton expects [B,T,20,3] Kinect v1 joint positions and T >= 8")
    known = torch.isfinite(values).all(-1)
    if valid is not None:
        if valid.shape != known.shape:
            raise ValueError("skeleton validity must match [B,T,20]")
        known = known & valid.bool()
    known = known & known[:, :, :1]
    root = torch.where(known[:, :, :1], values[:, :, 0], 0)
    first = known[:, :, 0].to(torch.int64).argmax(dim=1)
    origin = root[torch.arange(len(root), device=root.device), first]
    relative = torch.cat(((root - origin[:, None]).unsqueeze(2),
                          values[:, :, 1:] - root[:, :, None],), dim=2)
    axis_mask = known[..., None].expand_as(relative)
    return phase_mean(relative.flatten(2), axis_mask.flatten(2))


def wearable_phases(values, valid=None):
    if values.ndim != 3 or values.shape[1] < PHASES or values.shape[2] != 6:
        raise ValueError("wearable expects [B,T,6] UTD-MHAD readings and T >= 8")
    known = torch.isfinite(values)
    if valid is not None:
        if valid.shape != known.shape:
            raise ValueError("wearable validity must match [B,T,6]")
        known = known & valid.bool()
    return phase_mean(values, known)


class PairedActionHead(nn.Module):
    """Add separate, inspectable logits weighted by measured stream coverage."""

    def __init__(self):
        super().__init__()
        self.readouts = nn.ModuleList(nn.Linear(PHASES * size, len(ACTIONS)) for size in (60, 6))
        self.register_buffer("skeleton_center", torch.zeros(PHASES, 60))
        self.register_buffer("skeleton_scale", torch.ones(PHASES, 60))
        self.register_buffer("wearable_center", torch.zeros(PHASES, 6))
        self.register_buffer("wearable_scale", torch.ones(PHASES, 6))

    def forward(self, skeleton=None, wearable=None, *, skeleton_valid=None, wearable_valid=None):
        pose = skeleton_phases(skeleton, skeleton_valid) if skeleton is not None else None
        imu = wearable_phases(wearable, wearable_valid) if wearable is not None else None
        return self.score_phases(pose, imu)

    def score_phases(self, pose, imu, *, coverage_weighting=True):
        present = pose if pose is not None else imu
        if present is None:
            raise ValueError("provide at least one sensor tensor")
        size = present[0].shape[0]
        evidence = []
        for item, layer, center, scale in zip(
                (pose, imu), self.readouts, (self.skeleton_center, self.wearable_center),
                (self.skeleton_scale, self.wearable_scale)):
            if item is None:
                logits = center.new_zeros(size, len(ACTIONS))
                coverage = center.new_zeros(size, *center.shape)
            else:
                values, coverage = item
                if len(values) != size:
                    raise ValueError("sensor batch sizes must agree")
                normalized = torch.where(coverage > 0, ((values - center) / scale).clamp(-8, 8), 0)
                logits = layer(normalized.flatten(1))
            fraction = coverage.mean(dim=(1, 2))
            available = fraction >= .5
            evidence.append((logits, available, coverage, fraction))
        camera, wearable = evidence
        available = torch.stack((camera[1], wearable[1]), dim=-1)
        fractions = torch.stack((camera[3], wearable[3]), dim=-1)
        weights = fractions.square() * available if coverage_weighting else available.to(fractions.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        camera_term = camera[0] * weights[:, :1]
        wearable_term = wearable[0] * weights[:, 1:]
        disagreement = camera[1] & wearable[1] & (camera[0].argmax(-1) != wearable[0].argmax(-1))
        return {
            "logits": camera_term + wearable_term, "score_available": available.any(-1),
            "stream_weights": weights,
            "skeleton_contribution": camera_term, "wearable_contribution": wearable_term,
            "skeleton_coverage": camera[2], "wearable_coverage": wearable[2],
            "streams_disagree": disagreement,
        }
