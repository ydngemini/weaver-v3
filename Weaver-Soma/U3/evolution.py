"""SOMA U3 experimental structural plasticity operators, research only.

All network changes create fresh candidate modules. No live deployment or
optimizer state is mutated; promotion requires external evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
import torch
from torch import nn

_ACTIVATIONS = (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.LeakyReLU)


@dataclass(frozen=True)
class GrowthManifest:
    old_hidden: int
    new_hidden: int
    duplicate_sources: tuple[int, ...]
    parameter_delta: int
    note: str = "fresh candidate: recreate optimizer and independently evaluate"


def widen_feedforward(ff: nn.Sequential, sources: list[int] | tuple[int, ...]):
    """Net2Wider-style function-preserving hidden-neuron replication.

    Supports Linear -> elementwise activation -> Linear with optional biases.
    New network computes the same mathematical function, modulo FP rounding.
    `sources` selects which existing hidden neurons to duplicate.
    The original module is not changed.
    """
    if not isinstance(ff, nn.Sequential) or len(ff) != 3:
        raise TypeError("expected Sequential(Linear, elementwise activation, Linear)")
    a, activation, b = ff
    if not isinstance(a, nn.Linear) or not isinstance(b, nn.Linear):
        raise TypeError("first and last layers must be Linear")
    if not isinstance(activation, _ACTIVATIONS):
        raise TypeError("only elementwise activations are supported")
    if a.out_features != b.in_features:
        raise ValueError("hidden dimensions do not match")
    old = a.out_features
    if not sources:
        raise ValueError("at least one source neuron must be selected")
    if len(sources) > old:
        raise ValueError("growth limited to at most 2x hidden width per change")
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= old for i in sources):
        raise ValueError("source indices must reference existing hidden neurons")
    old_device, old_dtype = a.weight.device, a.weight.dtype
    if b.weight.device != old_device or b.weight.dtype != old_dtype:
        raise ValueError("both linears must share device and dtype")
    index = torch.tensor([*range(old), *sources], device=old_device, dtype=torch.long)
    multiplicity = torch.bincount(index, minlength=old).to(dtype=old_dtype)
    new_a = nn.Linear(a.in_features, len(index), bias=a.bias is not None,
                      device=old_device, dtype=old_dtype)
    new_b = nn.Linear(len(index), b.out_features, bias=b.bias is not None,
                      device=old_device, dtype=old_dtype)
    with torch.no_grad():
        new_a.weight.copy_(a.weight.index_select(0, index))
        if a.bias is not None:
            new_a.bias.copy_(a.bias.index_select(0, index))
        new_b.weight.copy_(b.weight.index_select(1, index) / multiplicity[index].unsqueeze(0))
        if b.bias is not None:
            new_b.bias.copy_(b.bias)
    grown = nn.Sequential(new_a, deepcopy(activation), new_b)
    if not ff.training:
        grown.eval()
    before = sum(p.numel() for p in ff.parameters())
    after = sum(p.numel() for p in grown.parameters())
    return grown, GrowthManifest(old, len(index), tuple(sources), after-before)


class EvolutionProposalHead(nn.Module):
    """TRAINABLE model component proposing bounded mutation categories.

    Input normalized, finite learning diagnostics [B,8], not raw credentials.
    Output uncalibrated action logits and bounded hyperparameter suggestions.
    This does not apply changes or evaluate whether proposals are safe.
    """
    ACTIONS = ("no_change", "widen_ffn", "rewire_sparse", "learning_rate_trial")

    def __init__(self, signal_count: int = 8, width: int = 64):
        super().__init__()
        if signal_count < 1 or width < 8:
            raise ValueError("invalid proposal-head dimensions")
        self.signal_count = signal_count
        self.trunk = nn.Sequential(nn.Linear(signal_count, width), nn.SiLU(),
                                   nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU())
        self.action = nn.Linear(width, len(self.ACTIONS))
        self.growth = nn.Linear(width, 1)
        self.learning_rate = nn.Linear(width, 1)

    def forward(self, diagnostics: torch.Tensor):
        if diagnostics.ndim != 2 or diagnostics.shape[-1] != self.signal_count:
            raise ValueError(f"diagnostics must be [B,{self.signal_count}]")
        if not bool(torch.isfinite(diagnostics).all()):
            raise ValueError("nonfinite diagnostics")
        hidden = self.trunk(diagnostics)
        return {
            "action_logits": self.action(hidden),
            "growth_fraction": (0.25 * torch.sigmoid(self.growth(hidden))).squeeze(-1),
            "lr_multiplier": (0.5 + 1.5 * torch.sigmoid(self.learning_rate(hidden))).squeeze(-1),
        }
