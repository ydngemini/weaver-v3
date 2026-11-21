"""SOMA's compact native causal text decoder."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DecoderConfig:
    vocab_size: int
    context: int = 128
    width: int = 96
    layers: int = 2
    heads: int = 4

    def __post_init__(self):
        if min(self.vocab_size, self.context, self.width, self.layers, self.heads) < 1 or self.width % self.heads:
            raise ValueError("positive decoder dimensions and a head-aligned width are required")


class CausalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.attention_norm = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.projection = nn.Linear(config.width, config.width, bias=False)
        self.ff = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, 3 * config.width),
                                nn.GELU(), nn.Linear(3 * config.width, config.width))

    def forward(self, values, previous, mask, *, save_cache):
        batch, length, width = values.shape
        heads = self.qkv(self.attention_norm(values)).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = heads.permute(2, 0, 3, 1, 4).unbind(0)
        if previous is not None:
            k = torch.cat((previous[0], k), dim=-2)
            v = torch.cat((previous[1], v), dim=-2)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=previous is None)
        values = values + self.projection(attended.transpose(1, 2).reshape(batch, length, width))
        values = values + self.ff(values)
        return values, (k, v) if save_cache else None


class NativeTextDecoder(nn.Module):
    """A from-scratch causal decoder with an explicit append-only KV cache."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList(CausalBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.width)
        nn.init.normal_(self.embedding.weight, std=.02)
        nn.init.normal_(self.position.weight, std=.02)

    def forward(self, tokens, *, cache=None, use_cache=False):
        if tokens.ndim != 2 or tokens.shape[-1] < 1:
            raise ValueError("expected a nonempty token batch with shape [batch, time]")
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError("cache must contain one entry per decoder block")
        past = 0 if cache is None else cache[0][0].shape[-2]
        end = past + tokens.shape[-1]
        if end > self.config.context:
            raise ValueError("token sequence exceeds the decoder context")
        positions = torch.arange(past, end, device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
        mask = None
        if cache is not None and tokens.shape[-1] > 1:
            keys = torch.arange(end, device=tokens.device)
            mask = positions[:, None] >= keys[None, :]
        updated = []
        for index, layer in enumerate(self.blocks):
            prior = None if cache is None else cache[index]
            hidden, state = layer(hidden, prior, mask, save_cache=use_cache)
            if use_cache:
                updated.append(state)
        logits = F.linear(self.final_norm(hidden), self.embedding.weight)
        return logits, tuple(updated) if use_cache else None


@torch.inference_mode()
def complete(model, prompt, *, max_new, stop=None, generator=None):
    """Sample one continuation. The caller supplies a prompt including BOS."""
    if len(prompt) < 1 or max_new < 1 or len(prompt) + max_new > model.config.context:
        raise ValueError("prompt and continuation must fit the decoder context")
    device = model.embedding.weight.device
    current = torch.tensor([prompt], device=device)
    cache = None
    pieces = []
    for _ in range(max_new):
        logits, cache = model(current, cache=cache, use_cache=True)
        last = logits[0, -1]
        if generator is None:
            token = last.argmax().item()
        else:
            score, ids = last.topk(min(32, len(last)))
            token = ids[torch.multinomial((score / .85).softmax(-1), 1, generator=generator)].item()
        if token == stop:
            break
        pieces.append(token)
        current = torch.tensor([[token]], device=device)
    return pieces
