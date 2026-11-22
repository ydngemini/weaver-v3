"""SOMA's randomly initialized causal language core for native pretraining."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class PretrainConfig:
    vocab_size: int = 32_768
    context: int = 2048
    width: int = 3072
    layers: int = 40
    query_heads: int = 24
    kv_heads: int = 8
    intermediate: int = 7936

    def __post_init__(self):
        dimensions = (
            self.vocab_size, self.context, self.width, self.layers,
            self.query_heads, self.kv_heads, self.intermediate,
        )
        if min(dimensions) < 1 or self.vocab_size > 65_536:
            raise ValueError("require positive dimensions and a uint16 vocabulary")
        if self.width % (2 * self.query_heads) or self.query_heads % self.kv_heads:
            raise ValueError("require even head dimensions and complete query groups")

    @classmethod
    def smoke(cls, vocab_size: int = 512):
        return cls(vocab_size, 256, 64, 2, 4, 2, 128)


def rotate(values, cos, sin):
    first, second = values.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class PretrainBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.width // config.query_heads
        self.query_heads = config.query_heads
        self.kv_heads = config.kv_heads
        self.attn_norm = nn.RMSNorm(config.width, eps=1e-5)
        self.ff_norm = nn.RMSNorm(config.width, eps=1e-5)
        self.qkv = nn.Linear(config.width, (self.query_heads + 2 * self.kv_heads) * self.head_dim,
                             bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)
        self.gate_up = nn.Linear(config.width, 2 * config.intermediate, bias=False)
        self.down = nn.Linear(config.intermediate, config.width, bias=False)

    def forward(self, hidden, cos, sin):
        batch, length, width = hidden.shape
        qkv = self.qkv(self.attn_norm(hidden)).reshape(batch, length, -1, self.head_dim)
        query, key, value = qkv.transpose(1, 2).split(
            (self.query_heads, self.kv_heads, self.kv_heads), dim=1
        )
        cos, sin = cos.to(query.dtype), sin.to(query.dtype)
        attended = F.scaled_dot_product_attention(
            rotate(query, cos, sin), rotate(key, cos, sin), value,
            is_causal=True, enable_gqa=self.query_heads != self.kv_heads,
        )
        hidden = hidden + self.out(attended.transpose(1, 2).reshape(batch, length, width))
        gate, up = self.gate_up(self.ff_norm(hidden)).chunk(2, dim=-1)
        return hidden + self.down(F.silu(gate) * up)


class SomaPretrainLM(nn.Module):
    def __init__(self, config: PretrainConfig, *, activation_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.activation_checkpointing = activation_checkpointing
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(PretrainBlock(config) for _ in range(config.layers))
        self.norm = nn.RMSNorm(config.width, eps=1e-5)
        half_dim = config.width // (2 * config.query_heads)
        frequency = 10_000.0 ** -(torch.arange(half_dim, dtype=torch.float32) / half_dim)
        angles = torch.arange(config.context, dtype=torch.float32)[:, None] * frequency
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        for block in self.blocks:
            for projection in (block.out, block.down):
                nn.init.normal_(projection.weight, std=0.02 / math.sqrt(2 * config.layers))

    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.context:
            raise ValueError("expected [batch, time] tokens within the configured context")
        length = tokens.shape[1]
        cos, sin = self.rope_cos[:length], self.rope_sin[:length]
        hidden = self.embedding(tokens)
        for block in self.blocks:
            if self.training and self.activation_checkpointing:
                hidden = checkpoint(block, hidden, cos, sin, use_reentrant=False)
            else:
                hidden = block(hidden, cos, sin)
        return F.linear(self.norm(hidden), self.embedding.weight)
