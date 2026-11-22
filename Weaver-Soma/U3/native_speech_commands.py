"""A causal, clip-local SOMA keyword recognizer with a bounded PCM state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import wave

import numpy as np
import torch
from torch import nn

from audio_spectral import FFT_SIZE, HOP, SAMPLE_RATE, WINDOW, _mel_filters


COMMANDS = ("down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes")
LABELS = COMMANDS + ("unknown", "silence")
STRIDE = 2 * HOP
BANDS = 40
HIDDEN = 64


class KeywordRecognizer(nn.Module):
    def __init__(self, temporal: bool = True):
        super().__init__()
        self.register_buffer("window", torch.hann_window(WINDOW), persistent=False)
        self.register_buffer("mel_filters", _mel_filters(BANDS), persistent=False)
        if temporal:
            self.project = nn.Sequential(nn.Linear(BANDS, HIDDEN), nn.SiLU())
            self.recurrent = nn.GRU(HIDDEN, HIDDEN, batch_first=True)
        else:
            self.project = nn.Identity()
            self.recurrent = None
        size = 2 * (HIDDEN if temporal else BANDS)
        self.head = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, len(LABELS)))

    def features(self, pcm: torch.Tensor) -> torch.Tensor:
        """Transform complete left-aligned windows; each window uses only received PCM."""
        if pcm.shape[-1] < WINDOW:
            return pcm.new_empty((*pcm.shape[:-1], 0, BANDS))
        frames = pcm.unfold(-1, WINDOW, STRIDE)
        power = torch.fft.rfft(frames * self.window, n=FFT_SIZE).abs().square()
        mel = (power @ self.mel_filters.T).clamp_min(1e-8).log10()
        return mel.clamp(max=4).mul(.25).add(1)

    def encode(self, features: torch.Tensor, state=None):
        hidden = self.project(features)
        if self.recurrent is None:
            return hidden, None
        return self.recurrent(hidden, state)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encode(features)
        return self.head(torch.cat((hidden.mean(dim=1), hidden.amax(dim=1)), dim=-1))


class CommandStream:
    """Serve one utterance. Start a new stream at the next utterance boundary."""

    def __init__(self, model: KeywordRecognizer):
        self.model = model.eval()
        self.pending = model.window.new_empty(0)
        self.state = self.total = self.peak = None
        self.frame_count = 0

    @torch.no_grad()
    def push(self, pcm16le: bytes) -> torch.Tensor | None:
        """Add complete PCM16 samples and score if a new frame is ready."""
        if not pcm16le:
            return None
        pcm = np.frombuffer(pcm16le, dtype="<i2").astype(np.float32) / 32768
        samples = torch.cat((self.pending, torch.from_numpy(pcm).to(self.pending.device)))
        frames = self.model.features(samples)
        count = len(frames)
        self.pending = samples[count * STRIDE:].clone()
        if not count:
            return None
        hidden, self.state = self.model.encode(frames.unsqueeze(0), self.state)
        total = hidden[0].sum(dim=0)
        peak = hidden[0].amax(dim=0)
        self.total = total if self.total is None else self.total + total
        self.peak = peak if self.peak is None else torch.maximum(self.peak, peak)
        self.frame_count += count
        return self.model.head(torch.cat((self.total / self.frame_count, self.peak)))


def decision(logits: torch.Tensor, threshold: float) -> dict:
    probabilities = logits.softmax(dim=-1)
    best = int(probabilities.argmax())
    confidence = float(probabilities[best])
    accepted = best < len(COMMANDS) and confidence >= threshold
    return {
        "command": LABELS[best] if accepted else None,
        "top_label": LABELS[best],
        "confidence": confidence,
        "threshold": threshold,
        "reason": "accepted" if accepted else "unsupported_or_low_confidence",
        "scores": dict(zip(LABELS, probabilities.tolist())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["version"] != 1 or tuple(checkpoint["labels"]) != LABELS:
        parser.error("incompatible keyword checkpoint")
    model = KeywordRecognizer(checkpoint["temporal"])
    model.load_state_dict(checkpoint["state_dict"])
    stream = CommandStream(model)
    logits = None
    with wave.open(str(args.wav), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, SAMPLE_RATE):
            parser.error("expected mono 16 kHz PCM16 WAV")
        while pcm := source.readframes(STRIDE):
            scores = stream.push(pcm)
            if scores is not None:
                logits = scores
    if logits is None:
        parser.error("audio is shorter than one acoustic frame")
    print(json.dumps({"frames": stream.frame_count, **decision(logits, checkpoint["threshold"])}, indent=2))


if __name__ == "__main__":
    main()
