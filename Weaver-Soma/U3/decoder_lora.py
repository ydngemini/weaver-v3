"""Low-rank adapters for the native SOMA decoder's linear blocks."""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

BLOCK_TARGETS = ("qkv", "projection", "ff.1", "ff.3")


class LowRankLinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError("LoRA needs a linear layer and a rank within its dimensions")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be positive and finite")
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.a), self.b)

    @torch.no_grad()
    def merged(self):
        layer = deepcopy(self.base)
        layer.weight.add_((self.b @ self.a) * self.scale)
        return layer


def adapter_targets(model):
    for block in model.blocks:
        for path in BLOCK_TARGETS:
            parent, _, name = path.rpartition(".")
            module = block.get_submodule(parent) if parent else block
            yield module, name


def add_lora(model, *, rank=8, alpha=16):
    targets = [(module, name, LowRankLinear(getattr(module, name), rank, alpha))
               for module, name in adapter_targets(model)]
    model.requires_grad_(False)
    for module, name, layer in targets:
        layer.a.requires_grad_(True)
        layer.b.requires_grad_(True)
        setattr(module, name, layer)
    return model


def merge_lora(model):
    merged = deepcopy(model)
    for module, name in adapter_targets(merged):
        layer = getattr(module, name)
        if not isinstance(layer, LowRankLinear):
            raise ValueError("expected a LoRA-adapted decoder")
        setattr(module, name, layer.merged())
    return merged


def adapter_weights(model):
    return {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
