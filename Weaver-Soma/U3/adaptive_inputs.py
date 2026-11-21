"""Causal encoders and bounded token selection for the remaining U3 input paths."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

SIGNAL_CHANNELS = {"touch": 6, "olfaction": 8, "rf": 8, "spatial": 6}


@dataclass
class AdaptiveEncoding:
    tokens: torch.Tensor
    valid: torch.Tensor
    evidence: dict[str, torch.Tensor]
    intent_logits: torch.Tensor | None = None


def time_mask(values, valid):
    if values.shape[1] < 1:
        raise ValueError("empty input sequence")
    if valid is None:
        return torch.ones(values.shape[:2], device=values.device, dtype=torch.bool)
    if valid.shape != values.shape[:2]:
        raise ValueError("validity must match [B,T]")
    return valid.bool()


def masked_mean(tokens, valid):
    return (tokens * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


def _stable_cpu_selection(priority, limit):
    """Select by priority, resolving boundary ties in input order."""
    boundary = priority.topk(limit, dim=-1, sorted=False).values.amin(-1, keepdim=True)
    above = priority > boundary
    tied = priority == boundary
    slots = limit - above.sum(-1, keepdim=True)
    chosen = above | (tied & (tied.cumsum(-1, dtype=torch.int32) <= slots))
    width = priority.shape[-1]
    ordered = torch.where(chosen, torch.arange(width, device=priority.device), width)
    return ordered.topk(limit, dim=-1, largest=False).values


def select_tokens(tokens, valid, salience, limit):
    """Spend half the budget on time coverage and use the rest on observed changes."""
    width = valid.shape[1]
    if width <= limit:
        indices = torch.arange(width, device=valid.device).expand(len(valid), -1)
    else:
        fast_cpu = valid.device.type == "cpu" and width >= max(2048, 4 * limit)
        if fast_cpu:
            running = valid.cumsum(dim=-1, dtype=torch.int32)
            count = running[:, -1:]
        else:
            count = valid.sum(dim=-1, keepdim=True)
        anchor_count = max(1, limit // 2)
        rank = torch.arange(anchor_count, device=valid.device).unsqueeze(0)
        # With one anchor, use the latest sample. Multiple anchors include both ends.
        positions = (count - 1).clamp_min(0)
        spread = positions if anchor_count == 1 else rank * positions // (anchor_count - 1)
        if fast_cpu:
            anchor = torch.searchsorted(running, (spread + 1).int().contiguous()).clamp_max(width - 1)
        else:
            chronological = (~valid).long().argsort(dim=-1, stable=True)
            anchor = chronological.gather(-1, spread)
        priority = salience.clamp(0, 1).masked_fill(~valid, -1)
        anchor_valid = valid.gather(-1, anchor)
        priority.scatter_(-1, anchor, torch.where(anchor_valid, 2., -1.).to(priority.dtype))
        if fast_cpu:
            indices = _stable_cpu_selection(priority, limit)
        else:
            indices = priority.argsort(dim=-1, descending=True, stable=True)[:, :limit].sort(dim=-1).values
    kept = valid.gather(-1, indices)
    gathered = tokens.gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))
    return AdaptiveEncoding(gathered * kept[..., None], kept, {
        "selected_input_indices": torch.where(kept, indices, -1),
        "selected_salience": torch.where(kept, salience.gather(-1, indices), 0),
        "observed_steps": valid.sum(-1),
        "retained_tokens": kept.sum(-1),
    })


class CausalBlock(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.past = 2 * dilation
        self.depthwise = nn.Conv1d(dim, dim, 3, dilation=dilation, groups=dim)
        self.mix = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, valid):
        local = self.depthwise(F.pad(x.transpose(1, 2), (self.past, 0))).transpose(1, 2)
        return self.norm(x + self.mix(F.silu(local))) * valid[..., None]


class AdaptiveTextEncoder(nn.Module):
    """UTF-8 byte context and an optional native intent probe."""
    def __init__(self, dim, *, contextual=True):
        super().__init__()
        self.embedding = nn.Embedding(256, dim)
        self.local = nn.ModuleList(CausalBlock(dim, d) for d in (1, 4) if contextual)
        self.intent = nn.Linear(dim, 150)

    def forward(self, values, *, valid=None, limit=128):
        if values.ndim != 2 or values.dtype not in (torch.int32, torch.int64, torch.uint8):
            raise ValueError("text expects integer UTF-8 bytes [B,T]")
        mask = time_mask(values, valid)
        safe = torch.where(mask, values, 0)
        if ((safe < 0) | (safe > 255)).any():
            raise ValueError("observed text values must be bytes 0..255")
        x = self.embedding(safe.long()) * mask[..., None]
        for block in self.local:
            x = block(x, mask)
        # Prefer digits and UTF-8 lead bytes when long text exceeds its token cap.
        digits = (safe >= 48) & (safe <= 57)
        utf8_lead = safe >= 192
        salience = (digits | utf8_lead).float()
        out = select_tokens(x, mask, salience, limit)
        out.intent_logits = self.intent(masked_mean(out.tokens, out.valid)) * mask.any(-1, keepdim=True)
        return out


class AdaptiveSignalEncoder(nn.Module):
    """Filter short and slow variations from calibrated numeric measurements."""
    def __init__(self, channels, dim):
        super().__init__()
        self.channels = channels
        self.project = nn.Linear(3 * channels, dim)
        self.short = CausalBlock(dim, 1)
        self.slow = CausalBlock(dim, 4)

    def forward(self, values, *, valid=None, quality=None, timestamps=None, limit=128):
        if (values.ndim != 3 or values.shape[2] != self.channels
                or not values.is_floating_point()):
            raise ValueError(f"expected calibrated float [B,T,{self.channels}]")
        observed = time_mask(values, valid)[..., None] & torch.isfinite(values)
        if quality is None:
            quality = torch.ones_like(values)
        elif quality.shape != values.shape:
            raise ValueError("quality must match sensor values")
        observed = observed & torch.isfinite(quality) & (quality > 0)
        if timestamps is not None:
            if timestamps.shape != values.shape[:2]:
                raise ValueError("timestamps must match [B,T]")
            observed = observed & torch.isfinite(timestamps)[..., None]
        q = torch.where(observed, quality, 0).to(values.dtype).clamp(0, 1)
        safe = torch.where(observed, values, 0).clamp(-16, 16)
        pair = observed[:, 1:] & observed[:, :-1]
        if timestamps is not None:
            ordered = timestamps[:, 1:] > timestamps[:, :-1]
            pair = pair & ordered[..., None]
        delta = F.pad(torch.where(pair, safe[:, 1:] - safe[:, :-1], 0), (0, 0, 1, 0))
        pair_q = F.pad(torch.where(pair, torch.minimum(q[:, 1:], q[:, :-1]), 0), (0, 0, 1, 0))
        activity = (delta.abs() * pair_q).sum(-1) / (pair_q > 0).sum(-1).clamp_min(1)
        salience = activity / (1 + activity)
        valid_steps = observed.any(-1)
        x = self.project(torch.cat((safe * q, delta * pair_q, q), dim=-1)) * valid_steps[..., None]
        fast = self.short(x, valid_steps)
        slow = self.slow(fast, valid_steps)
        x = torch.lerp(slow, fast, salience[..., None])
        out = select_tokens(x, valid_steps, salience, limit)
        out.evidence["per_channel_observations"] = observed.sum(1)
        out.evidence["per_channel_delta_pairs"] = pair.sum(1)
        return out


def tactile_grid_features(values, valid=None):
    """Convert calibrated pressure grids to force, footprint, center, and spread."""
    if (values.ndim != 4 or not values.is_floating_point()
            or min(values.shape[-2:]) < 2):
        raise ValueError("touch grid expects float pressure [B,T,H>=2,W>=2]")
    if valid is not None and valid.shape == values.shape:
        known = valid.bool() & torch.isfinite(values)
    else:
        known = time_mask(values, valid)[..., None, None] & torch.isfinite(values)
    force = torch.where(known, values, 0).clamp(0, 1)
    h, w = force.shape[-2:]
    gx = torch.linspace(-1, 1, w, dtype=force.dtype, device=force.device)
    gy = torch.linspace(-1, 1, h, dtype=force.dtype, device=force.device)
    total = force.sum((-2, -1)).clamp_min(1e-6)
    cx = (force.sum(-2) * gx).sum(-1) / total
    cy = (force.sum(-1) * gy).sum(-1) / total
    count = known.sum((-2, -1))
    mean = force.sum((-2, -1)) / count.clamp_min(1)
    variance = torch.where(known, (force - mean[..., None, None]).square(), 0).sum((-2, -1))
    spread = (variance / count.clamp_min(1)).clamp_min(1e-12).sqrt()
    features = torch.stack((mean, (force > .02).sum((-2, -1)) / count.clamp_min(1),
                            cx, cy, force.amax((-2, -1)), spread), dim=-1)
    return features.to(force.dtype), count


def spatial_point_features(values, valid=None):
    """Summarize metric XYZ points in a world or device coordinate frame."""
    if (values.ndim != 4 or values.shape[-1] != 3 or values.shape[-2] < 1
            or not values.is_floating_point()):
        raise ValueError("spatial points expect float [B,T,P,3] in metric coordinates")
    known = torch.isfinite(values).all(-1)
    if valid is not None:
        if valid.shape != known.shape:
            raise ValueError("point validity must match [B,T,P]")
        known = known & valid.bool()
    count = known.sum(-1, keepdim=True)
    points = torch.where(known[..., None], values, 0)
    centroid = points.sum(-2) / count.clamp_min(1)
    offsets = torch.where(known[..., None], points - centroid.unsqueeze(-2), 0)
    spread = (offsets.square().sum(-2) / count.clamp_min(1)).clamp_min(1e-12).sqrt()
    return torch.cat((centroid, spread), dim=-1), count.squeeze(-1)


class AdaptiveSensoryBank(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.text = AdaptiveTextEncoder(dim)
        self.signals = nn.ModuleDict({name: AdaptiveSignalEncoder(channels, dim)
                                      for name, channels in SIGNAL_CHANNELS.items()})
        # Alternate measured profiles have their own parameters.
        self.grid = AdaptiveSignalEncoder(6, dim)
        self.points = AdaptiveSignalEncoder(6, dim)

    def forward(self, name, sample, limit):
        values = sample["values"]
        valid = sample.get("valid")
        if name == "text":
            return self.text(values, valid=valid, limit=limit)
        ts = sample.get("timestamps")
        if name == "touch_grid":
            features, count = tactile_grid_features(values, valid)
            out = self.grid(features, valid=count > 0, timestamps=ts, limit=limit)
            out.evidence["pressure_features"] = features
            out.evidence["observed_taxels"] = count
            return out
        if name == "spatial_points":
            features, count = spatial_point_features(values, valid)
            out = self.points(features / 10, valid=count > 0, timestamps=ts, limit=limit)
            out.evidence["observed_points"] = count
            out.evidence["point_features_in_meters"] = features
            return out
        return self.signals[name](values, valid=valid, quality=sample.get("quality"), timestamps=ts, limit=limit)
