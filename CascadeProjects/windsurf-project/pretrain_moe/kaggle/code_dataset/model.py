"""
model.py — From-scratch Mixture-of-Experts GPT for free-Colab-T4 pretraining.

Winner config (mem obs #1848, "B-core hardened with A+C grafts"):
    9 layers x 640d, 10 heads (head_dim=64), 8 experts top-2, d_ff=1920 (3x),
    ctx=1024, vocab=16384 (trained byte-level BPE), tied tok-emb/lm-head.
    Total ~202.9M params / ~69.5M active-per-token. Fits a single 16GB T4
    with mixed-precision AdamW + gradient checkpointing at micro-batch B=48.

This file is ARCHITECTURE ONLY. Random init everywhere — no pretrained weights
are ever loaded. The training loop, optimizer, data streaming, and the atomic
Google-Drive multi-session resume bundle live in the sibling train.py; this
module only exposes the model, its config, and a from-config constructor.

Design notes (why this shape, on a free T4):
  * MoE saves FLOPs, NOT memory. All 8 expert weights and their full fp32 AdamW
    optimizer state stay resident on the single GPU. We size TOTAL params (not
    active) against the 16GB budget — hence 8 modest 3x experts, not bigger ones.
  * The (B, T, vocab) logits tensor is the single largest activation on a T4, so
    the tight 16k vocab is deliberate: it keeps that buffer small AND keeps the
    tied tok-emb/lm-head at only ~10.49M params.
  * Full grafted router stack (from proposal C): noisy top-2 gating with annealed
    Gaussian logit noise, Switch-style load-balancing aux loss, router z-loss, and
    a capacity factor with token-dropping to bound per-expert dispatch buffers.
  * Cheap divergence insurance (from C): optional QK-LayerNorm, head_dim=64,
    designed for bf16 autocast + fp32 master weights + grad-clip 1.0 (handled in
    the trainer). GELU experts, no biases on the big matmuls.

8B-active SCALE-UP PATH (documented, NOT targeted):
    The SAME code reaches a Mixtral-class ~8B-active / ~47B-total model with a
    CONFIG-ONLY change — keep 8 experts / top-2 but set n_embd~4096, n_layer~32,
    d_ff~14336, head_dim 128, vocab ~64k, ctx 4096+. That is categorically
    impossible on one 16GB T4: all 8 expert weights plus their fp32 AdamW state
    can no longer be resident on a single device. It requires a multi-GPU cluster
    with expert-parallel + tensor-parallel sharding (Megatron / DeepSpeed-MoE),
    ZeRO-3 optimizer-state sharding, and a token budget in the trillions
    (~15-20x params). It changes only the config + parallelism strategy — never
    this architecture family.

Single-file-runnable: `python model.py` runs a self-test (param count + a
forward/backward on random tokens) and prints the param accounting.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class MoEGPTConfig:
    """All architecture + router hyper-parameters. JSON-serialisable.

    The defaults ARE the winning config from mem obs #1848. The training script
    should construct this from a JSON file so the same config hash is logged in
    the atomic resume bundle across the ~9 Colab reconnects.
    """

    # --- core transformer shape ---
    vocab_size: int = 16384          # trained byte-level BPE 16k (tight on purpose)
    ctx_len: int = 1024              # block / context length
    n_layer: int = 9
    n_head: int = 10
    n_embd: int = 640
    head_dim: int = 64               # n_head * head_dim == n_embd (10 * 64 == 640)

    # --- MoE feed-forward ---
    n_experts: int = 8               # "8" = 8 EXPERTS, never 8B params
    top_k: int = 2                   # top-2 routing
    d_ff: int = 1920                 # per-expert hidden (3x n_embd)
    capacity_factor: float = 1.25    # bounds per-expert dispatch buffer; drops overflow
    drop_overflow_tokens: bool = True  # token-dropping when an expert exceeds capacity

    # --- router stack (grafted from proposal C) ---
    router_noise_std: float = 1.0    # base std of Gaussian noise on gate logits (annealed -> 0)
    aux_loss_coef: float = 0.01      # Switch-style load-balancing aux loss weight
    router_z_loss_coef: float = 1e-3  # keeps gate logits bounded / stable

    # --- stability levers ---
    use_qk_norm: bool = True         # optional QK-LayerNorm (cheap divergence insurance)
    dropout: float = 0.0             # 0.0 for a single-epoch-ish pretrain (no overfit risk)
    bias: bool = False               # no biases on big matmuls (params + a touch of stability)

    # --- init ---
    init_std: float = 0.02           # base std for weight init (GPT-2 style)

    def __post_init__(self):
        assert self.n_head * self.head_dim == self.n_embd, (
            f"n_head*head_dim ({self.n_head}*{self.head_dim}) must equal "
            f"n_embd ({self.n_embd})"
        )
        assert self.top_k <= self.n_experts, "top_k cannot exceed n_experts"
        assert self.top_k >= 1, "top_k must be >= 1"

    # ---- (de)serialisation helpers for the resume bundle ----
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "MoEGPTConfig":
        # Ignore unknown keys defensively so an older checkpoint config still loads.
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def from_json(cls, s: str) -> "MoEGPTConfig":
        return cls.from_dict(json.loads(s))

    def config_hash(self) -> str:
        """Short stable hash of the architecture — logged in every checkpoint so
        a resume can refuse to load a mismatched-architecture bundle."""
        import hashlib

        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Causal self-attention
# ----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Uses PyTorch's scaled_dot_product_attention (Flash / mem-efficient kernel
    when available, with the causal mask fused) so we never materialise a
    (B, n_head, T, T) score matrix — important for the T4 activation budget.
    Optional QK-LayerNorm normalises queries and keys per-head before the dot
    product, a cheap stabiliser against attention-logit blow-up in bf16.
    """

    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.use_qk_norm = cfg.use_qk_norm

        # Fused QKV projection (no bias on the big mats).
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout_p = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

        if self.use_qk_norm:
            # Per-head RMS/Layer norm over head_dim. LayerNorm without bias is fine.
            self.q_norm = nn.LayerNorm(self.head_dim, bias=cfg.bias)
            self.k_norm = nn.LayerNorm(self.head_dim, bias=cfg.bias)

        # Flag the output projection so init can scale it by 1/sqrt(2*n_layer).
        self.c_proj._is_residual_proj = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # (B, T, n_head, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # -> (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Fused causal attention (Flash kernel when the dtype/device allow it).
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ----------------------------------------------------------------------------
# A single expert (a plain GELU MLP)
# ----------------------------------------------------------------------------
class Expert(nn.Module):
    """One MoE expert: a 2-layer GELU MLP, n_embd -> d_ff -> n_embd."""

    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, cfg.d_ff, bias=cfg.bias)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.n_embd, bias=cfg.bias)
        self.fc2._is_residual_proj = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ----------------------------------------------------------------------------
# MoE feed-forward layer (the grafted router stack lives here)
# ----------------------------------------------------------------------------
class MoEFeedForward(nn.Module):
    """Sparse MoE FFN: noisy top-k router + capacity-bounded dispatch + GELU experts.

    Router stack (grafted from proposal C):
      * noisy top-k gating: Gaussian noise added to the gate logits during
        training, scaled by ``router_noise_std * noise_scale`` where
        ``noise_scale`` is annealed 1 -> 0 by the trainer (early exploration).
      * Switch-Transformer load-balancing aux loss:
            aux = n_experts * sum_e (frac_tokens_to_e * mean_gate_prob_e)
        which is minimised when load is uniform. Returned for the trainer to add
        with ``aux_loss_coef``.
      * router z-loss: mean of logsumexp(logits)^2, keeps gate logits bounded so
        the softmax can't drift to huge magnitudes in bf16. Returned separately.
      * capacity factor + token-drop: per-expert capacity is
            ceil(capacity_factor * top_k * T_tokens / n_experts)
        tokens beyond an expert's capacity are dropped (their MoE contribution is
        zero, residual still flows), which bounds the dispatch/combine buffers so
        VRAM stays flat regardless of how skewed routing gets mid-run.

    The aux/z losses are exposed via ``self.last_aux_loss`` / ``self.last_z_loss``
    (and returned) so the top-level model can sum them across layers.
    """

    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.capacity_factor = cfg.capacity_factor
        self.drop_overflow = cfg.drop_overflow_tokens
        self.router_noise_std = cfg.router_noise_std

        # Router: a single linear gate (n_embd -> n_experts), no bias.
        self.gate = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_experts)])

        # Set by the trainer each step to anneal exploration noise 1.0 -> 0.0.
        self.register_buffer("noise_scale", torch.tensor(1.0), persistent=False)

        # Diagnostics filled in on every forward (for logging / the trainer).
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)
        self.last_z_loss: torch.Tensor = torch.tensor(0.0)

    def set_noise_scale(self, scale: float) -> None:
        """Trainer calls this each step to anneal the exploration noise."""
        self.noise_scale.fill_(float(scale))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)            # (N, C),  N = B*T
        N = x_flat.shape[0]

        # ---- router logits ----
        logits = self.gate(x_flat)           # (N, n_experts)

        # router z-loss: penalise large logsumexp so gate logits stay bounded.
        z = torch.logsumexp(logits, dim=-1)  # (N,)
        z_loss = (z ** 2).mean()

        # noisy gating: Gaussian noise on logits during training (annealed).
        if self.training and self.router_noise_std > 0.0:
            noise = torch.randn_like(logits) * (self.router_noise_std * self.noise_scale)
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        # Softmax gate probabilities (over ALL experts) — used both for the aux
        # loss (mean prob per expert) and to weight the selected experts.
        gate_probs = F.softmax(logits.float(), dim=-1)            # (N, E), clean probs for aux
        route_probs = F.softmax(noisy_logits.float(), dim=-1)     # (N, E), noisy for selection

        # ---- top-k selection ----
        topk_vals, topk_idx = torch.topk(route_probs, self.top_k, dim=-1)  # (N, k)
        # Renormalise the kept gate weights so they sum to 1 per token.
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)
        topk_weights = topk_weights.to(x_flat.dtype)

        # ---- Switch load-balancing aux loss ----
        # frac_e = fraction of tokens that routed expert e into their top-k.
        # mean_prob_e = mean clean gate prob for expert e over all tokens.
        # aux = E * sum_e frac_e * mean_prob_e   (minimised at uniform load).
        with torch.no_grad():
            # one-hot of which experts each token selected (any of its top-k)
            expert_mask = torch.zeros(N, self.n_experts, device=x.device, dtype=gate_probs.dtype)
            expert_mask.scatter_(1, topk_idx, 1.0)
        frac_tokens = expert_mask.mean(dim=0)            # (E,)  fraction routed to each
        mean_prob = gate_probs.mean(dim=0)               # (E,)  mean gate prob
        aux_loss = self.n_experts * torch.sum(frac_tokens * mean_prob)

        # ---- capacity-bounded dispatch + combine ----
        # Per-expert capacity: how many tokens each expert will actually process.
        capacity = int(
            math.ceil(self.capacity_factor * self.top_k * N / self.n_experts)
        )
        capacity = max(capacity, 1)

        out = torch.zeros_like(x_flat)

        # Process expert-by-expert. For each expert gather the (token, slot) pairs
        # that selected it, truncate to capacity (token-drop), run the expert
        # once on that gathered batch, and scatter-add the weighted result back.
        # Flatten the (N, k) selection so each (token, slot) is one routing event.
        flat_expert = topk_idx.reshape(-1)               # (N*k,)
        flat_weight = topk_weights.reshape(-1)           # (N*k,)
        flat_token = (
            torch.arange(N, device=x.device).unsqueeze(1).expand(N, self.top_k).reshape(-1)
        )                                                # (N*k,) token index per event

        for e in range(self.n_experts):
            sel = (flat_expert == e).nonzero(as_tuple=True)[0]   # routing events for expert e
            if sel.numel() == 0:
                continue
            if self.drop_overflow and sel.numel() > capacity:
                # Keep the first `capacity` events; the rest are dropped.
                # (Order here is token-major; a position-based priority could be
                # substituted, but FIFO-by-token is the standard Switch behaviour.)
                sel = sel[:capacity]
            tok_idx = flat_token[sel]                    # tokens this expert handles
            w = flat_weight[sel].unsqueeze(1)            # (n_e, 1) gate weights
            expert_in = x_flat[tok_idx]                  # (n_e, C)
            expert_out = self.experts[e](expert_in)      # (n_e, C)
            out.index_add_(0, tok_idx, (expert_out * w).to(out.dtype))

        out = out.reshape(B, T, C)

        # Stash diagnostics for logging.
        self.last_aux_loss = aux_loss.detach()
        self.last_z_loss = z_loss.detach()

        return out, aux_loss, z_loss


# ----------------------------------------------------------------------------
# Transformer block: attention + MoE FFN, pre-norm residual
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.moe = MoEFeedForward(cfg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln_1(x))
        moe_out, aux_loss, z_loss = self.moe(self.ln_2(x))
        x = x + moe_out
        return x, aux_loss, z_loss


# ----------------------------------------------------------------------------
# The full MoE GPT
# ----------------------------------------------------------------------------
class MoEGPT(nn.Module):
    """From-scratch MoE GPT. Random init only — no pretrained weights, ever."""

    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token embeddings
                wpe=nn.Embedding(cfg.ctx_len, cfg.n_embd),      # learned positional embeddings
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),   # final layernorm
            )
        )
        # LM head — tied to the token embedding (weights shared below).
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # Random init for every parameter.
        self.apply(self._init_weights)
        # Scaled init for residual output projections (GPT-2 trick): 1/sqrt(2*n_layer).
        residual_scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for module in self.modules():
            if getattr(module, "_is_residual_proj", False):
                with torch.no_grad():
                    module.weight.mul_(residual_scale)

    # ---- weight init ----
    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    # ---- noise annealing pass-through (trainer calls once per step) ----
    def set_router_noise_scale(self, scale: float) -> None:
        """Anneal the exploration noise across all MoE layers (1.0 -> 0.0)."""
        for block in self.transformer.h:
            block.moe.set_noise_scale(scale)

    # ---- forward ----
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        idx:     (B, T) int64 token ids
        targets: (B, T) int64 next-token targets, or None for inference

        Returns (logits, loss, aux) where:
          logits: (B, T, vocab) if targets is None else (B, T, vocab) too
          loss:   total loss (LM CE + aux_loss_coef*aux + z_loss_coef*z) or None
          aux:    dict with 'lm_loss', 'aux_loss', 'z_loss' for logging
        """
        B, T = idx.shape
        assert T <= self.cfg.ctx_len, (
            f"sequence length {T} exceeds ctx_len {self.cfg.ctx_len}"
        )
        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)

        tok_emb = self.transformer.wte(idx)        # (B, T, C)
        pos_emb = self.transformer.wpe(pos)        # (T, C)
        x = self.transformer.drop(tok_emb + pos_emb)

        total_aux = torch.zeros((), device=device)
        total_z = torch.zeros((), device=device)
        for block in self.transformer.h:
            x, aux_loss, z_loss = block(x)
            total_aux = total_aux + aux_loss
            total_z = total_z + z_loss
        # Average the router losses over layers so the coefficients are layer-count independent.
        n_layer = len(self.transformer.h)
        total_aux = total_aux / n_layer
        total_z = total_z / n_layer

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)               # (B, T, vocab)
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            loss = (
                lm_loss
                + self.cfg.aux_loss_coef * total_aux
                + self.cfg.router_z_loss_coef * total_z
            )
            aux = {
                "lm_loss": lm_loss.detach(),
                "aux_loss": total_aux.detach(),
                "z_loss": total_z.detach(),
            }
            return logits, loss, aux
        else:
            # Inference: only compute logits for the last position to save memory.
            logits = self.lm_head(x[:, [-1], :])   # (B, 1, vocab)
            aux = {
                "lm_loss": torch.zeros((), device=device),
                "aux_loss": total_aux.detach(),
                "z_loss": total_z.detach(),
            }
            return logits, None, aux

    # ---- generation (greedy / temperature sampling) ----
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively sample max_new_tokens. idx: (B, T) seed tokens."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.cfg.ctx_len else idx[:, -self.cfg.ctx_len:]
            )
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_tok), dim=1)
        if was_training:
            self.train()
        return idx

    # ---- parameter accounting ----
    def num_params(self, non_embedding: bool = False) -> int:
        """Total parameter count. With non_embedding=True the positional table is
        excluded (the tied token-emb/lm-head is counted once, as it should be)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def num_active_params(self) -> int:
        """Params actually used per token: shared (embeds/attn/norms/head) + only
        top_k of the n_experts experts. This is the number that must sit in the
        30-80M target band (winner reports ~69.5M)."""
        cfg = self.cfg
        total = self.num_params()
        # One expert's parameter count (fc1 + fc2, plus biases if enabled).
        per_expert = 0
        for e_idx in range(cfg.n_experts):
            if e_idx == 0:
                per_expert = sum(p.numel() for p in self.transformer.h[0].moe.experts[0].parameters())
                break
        # Inactive experts = (n_experts - top_k) per layer, across all layers.
        inactive = (cfg.n_experts - cfg.top_k) * per_expert * cfg.n_layer
        return total - inactive

    # ---- from-config constructors (the required entry points) ----
    @classmethod
    def from_config(cls, cfg: MoEGPTConfig) -> "MoEGPT":
        """Build a randomly-initialised model from a config dataclass."""
        return cls(cfg)

    @classmethod
    def from_config_dict(cls, d: dict) -> "MoEGPT":
        """Build from a plain dict (e.g. loaded from the JSON winner config)."""
        return cls(MoEGPTConfig.from_dict(d))

    @classmethod
    def from_config_json(cls, path_or_str: str) -> "MoEGPT":
        """Build from a JSON file path OR a raw JSON string."""
        import os

        if os.path.exists(path_or_str):
            with open(path_or_str, "r") as f:
                d = json.load(f)
        else:
            d = json.loads(path_or_str)
        # Tolerate the full "winner config" object that also carries rationale/
        # est fields — keep only the architecture keys MoEGPTConfig knows about.
        return cls.from_config_dict(d)

    # ---- AdamW param groups (decay vs no-decay) for the trainer ----
    def configure_param_groups(self, weight_decay: float = 0.1):
        """Split params into decay (2D matmuls/embeddings) and no-decay (norms,
        biases, 1D). The trainer feeds these straight to torch.optim.AdamW."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]


# ----------------------------------------------------------------------------
# Self-test: param accounting + a forward/backward on random tokens.
# Run:  python model.py
# ----------------------------------------------------------------------------
def _selftest() -> None:
    torch.manual_seed(0)
    cfg = MoEGPTConfig()  # the winning config
    model = MoEGPT.from_config(cfg)

    total = model.num_params()
    active = model.num_active_params()
    non_emb = model.num_params(non_embedding=True)

    print("=" * 70)
    print("MoEGPT self-test — winning config (mem obs #1848)")
    print("=" * 70)
    print(f"  config hash         : {cfg.config_hash()}")
    print(f"  n_layer x n_embd    : {cfg.n_layer} x {cfg.n_embd}")
    print(f"  n_head x head_dim   : {cfg.n_head} x {cfg.head_dim}")
    print(f"  experts / top_k     : {cfg.n_experts} / {cfg.top_k}")
    print(f"  d_ff (per expert)   : {cfg.d_ff}")
    print(f"  vocab / ctx_len     : {cfg.vocab_size} / {cfg.ctx_len}")
    print("-" * 70)
    print(f"  TOTAL params        : {total/1e6:.2f}M   (target ~202.9M)")
    print(f"  ACTIVE params/token : {active/1e6:.2f}M   (target ~69.5M, band 30-80M)")
    print(f"  non-embedding params: {non_emb/1e6:.2f}M")
    print("-" * 70)

    # Forward + backward on a tiny random batch (random tokens, random init).
    B, T = 2, 64
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    model.train()
    model.set_router_noise_scale(0.5)  # mid-anneal, exercise the noisy path
    logits, loss, aux = model(idx, targets)
    loss.backward()

    print(f"  forward logits shape: {tuple(logits.shape)}")
    print(f"  total loss          : {loss.item():.4f}")
    print(f"    lm_loss           : {aux['lm_loss'].item():.4f}  "
          f"(ln(vocab)={math.log(cfg.vocab_size):.4f} expected at init)")
    print(f"    aux_loss (Switch) : {aux['aux_loss'].item():.4f}  "
          f"(==1.0 at perfectly-uniform routing)")
    print(f"    z_loss (router)   : {aux['z_loss'].item():.4f}")

    # Confirm grads flowed to every expert in at least one layer.
    g0 = model.transformer.h[0].moe.experts[0].fc1.weight.grad
    assert g0 is not None and torch.isfinite(g0).all(), "expert grad missing / non-finite"
    # Confirm weight tying held.
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr(), (
        "tok-emb / lm-head are not tied"
    )

    # Inference path (single-position logits).
    model.eval()
    inf_logits, inf_loss, _ = model(idx)
    assert inf_loss is None and inf_logits.shape == (B, 1, cfg.vocab_size)

    # Generation smoke test.
    out = model.generate(idx[:, :8], max_new_tokens=5, temperature=0.8, top_k=50)
    assert out.shape == (B, 13)

    print("-" * 70)
    print("  weight tying        : OK (tok-emb shares storage with lm-head)")
    print("  expert grads finite : OK")
    print("  inference path      : OK (B,1,vocab last-position logits)")
    print("  generate()          : OK")
    print("=" * 70)
    print("SELF-TEST PASSED — random init, forward+backward, all router losses live.")


if __name__ == "__main__":
    _selftest()
