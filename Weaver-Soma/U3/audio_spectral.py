"""Native SOMA audio encoder for 16 kHz mono PCM, with no pretrained weights."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


SAMPLE_RATE = 16_000
WINDOW = 400
HOP = 160
FFT_SIZE = 512


def _mel_filters(bands: int) -> torch.Tensor:
    mel_max = 2595 * torch.log10(torch.tensor(1.0 + (SAMPLE_RATE / 2) / 700))
    edges = 700 * (10 ** (torch.linspace(0, float(mel_max), bands + 2) / 2595) - 1)
    hz = torch.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
    lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    slopes = torch.minimum((hz - lower) / (center - lower), (upper - hz) / (upper - center))
    return slopes.clamp_min(0)


def masked_statistics(tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Summarize sustained energy, variability, and brief peaks."""
    mask = valid.unsqueeze(-1)
    count = mask.sum(dim=1).clamp_min(1)
    clean = tokens.masked_fill(~mask, 0)
    mean = clean.sum(dim=1) / count
    centered = (clean - mean.unsqueeze(1)).masked_fill(~mask, 0)
    deviation = (centered.square().sum(dim=1) / count + 1e-8).sqrt()
    peak = tokens.masked_fill(~mask, torch.finfo(tokens.dtype).min).amax(dim=1)
    present = valid.any(dim=1, keepdim=True)
    return torch.cat((mean, deviation, peak), dim=-1).masked_fill(~present, 0)


class SpectralAudioStem(nn.Module):
    """Log-mel frames, temporal convolutions, and an audio-only classifier."""

    def __init__(self, dim: int, classes: int):
        super().__init__()
        bands = min(64, dim // 2)
        hidden = dim - bands
        self.register_buffer("window", torch.hann_window(WINDOW), persistent=False)
        self.register_buffer("mel_filters", _mel_filters(bands), persistent=False)
        self.local = nn.Sequential(
            nn.Conv1d(bands, hidden, 5, padding=2), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * dim, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(64, classes),
        )

    def frontend(self, waveform: torch.Tensor, valid: torch.Tensor):
        """Return normalized log-mel magnitudes and valid frame positions."""
        pad = max(0, WINDOW - waveform.shape[1])
        pad += (-(waveform.shape[1] + pad - WINDOW)) % HOP
        samples = waveform.float().masked_fill(~valid, 0)
        windows = F.pad(samples, (0, pad)).unfold(1, WINDOW, HOP)
        mask = F.pad(valid, (0, pad)).unfold(1, WINDOW, HOP).any(dim=-1)
        spectrum = torch.fft.rfft(windows * self.window.float(), n=FFT_SIZE).abs().square()
        mel = spectrum @ self.mel_filters.float().T
        mel = mel.masked_fill(~mask.unsqueeze(-1), 0)
        peak = mel.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        level = (mel / peak).clamp_min(1e-8).log10()
        features = (level / 8 + 1).masked_fill(~mask.unsqueeze(-1), 0)
        return features.to(self.local[0].weight.dtype), mask

    def encode_features(self, features: torch.Tensor, valid: torch.Tensor):
        spectral = features.masked_fill(~valid.unsqueeze(-1), 0).transpose(1, 2)
        learned = self.local(spectral)
        tokens = torch.cat((spectral, learned), dim=1)
        tokens = tokens.masked_fill(~valid.unsqueeze(1), torch.finfo(tokens.dtype).min)
        tokens = F.max_pool1d(tokens, kernel_size=4, stride=4, ceil_mode=True).transpose(1, 2)
        mask = F.max_pool1d(valid.unsqueeze(1).float(), 4, stride=4, ceil_mode=True)[:, 0] > 0
        return tokens.masked_fill(~mask.unsqueeze(-1), 0), mask

    def forward(self, waveform: torch.Tensor, valid: torch.Tensor):
        return self.encode_features(*self.frontend(waveform, valid))

    def classify(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.head(masked_statistics(tokens, valid))
