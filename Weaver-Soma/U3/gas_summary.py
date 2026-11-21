"""Interpretable UCI gas-array readout for archived sensor descriptors."""
from __future__ import annotations

import torch
from torch import nn

GAS_CLASSES = ("ethanol", "ethylene", "ammonia", "acetaldehyde", "acetone", "toluene")


def compress(values):
    return values.sign() * values.abs().log1p()


class GasSummaryHead(nn.Module):
    """Group 128 measured descriptors by their 16 physical sensors."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(16, 8, len(GAS_CLASSES)))
        self.bias = nn.Parameter(torch.zeros(len(GAS_CLASSES)))
        self.register_buffer("center", torch.zeros(16, 8))
        self.register_buffer("scale", torch.ones(16, 8))
        nn.init.normal_(self.weight, std=.015)

    def forward(self, values, valid=None):
        if values.ndim != 3 or values.shape[1:] != (16, 8) or not values.is_floating_point():
            raise ValueError("gas summary expects float [B,16 sensors,8 descriptors]")
        observed = torch.isfinite(values)
        if valid is not None:
            if valid.shape != values.shape:
                raise ValueError("gas summary validity must match the input shape")
            observed = observed & valid.bool()
        safe = torch.where(observed, values, 0)
        scaled = (compress(safe) - self.center) / self.scale
        evidence = torch.where(observed, scaled.clamp(-8, 8), 0)
        available = observed.any(-1).sum(-1) >= 8
        terms = torch.einsum("bsc,sck->bsk", evidence, self.weight) * available[:, None, None]
        bias = self.bias * available[:, None]
        return {
            "logits": terms.sum(dim=1) + bias, "score_available": available,
            "sensor_contributions": terms, "bias": bias, "observed_features_per_sensor": observed.sum(-1),
            "mean_absolute_standardized_deviation": (
                evidence.abs().sum((1, 2)) / observed.sum((1, 2)).clamp_min(1)),
        }
