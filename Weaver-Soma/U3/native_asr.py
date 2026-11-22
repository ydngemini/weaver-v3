"""Causal CTC transcription from a stream of 16 kHz mono PCM16."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import wave

import numpy as np
import torch
from torch import nn

from audio_spectral import FFT_SIZE, HOP, SAMPLE_RATE, WINDOW, _mel_filters


ALPHABET = ("", " ", "'", *"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
TOKEN_ID = {symbol: i for i, symbol in enumerate(ALPHABET)}
STRIDE = 4 * HOP
SPAN = WINDOW + 2 * HOP
BANDS = 40
HIDDEN = 96


def normalize(text: str) -> str:
    """Apply the same case, punctuation, and spacing rules to the entire corpus."""
    return " ".join(re.findall(r"[A-Z']+", text.replace("’", "'").upper()))


class CausalTranscriber(nn.Module):
    """Two acoustic windows per tick and a unidirectional GRU acoustic encoder."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(WINDOW), persistent=False)
        self.register_buffer("mel", _mel_filters(BANDS), persistent=False)
        self.project = nn.Sequential(nn.Linear(2 * BANDS, HIDDEN), nn.LayerNorm(HIDDEN), nn.SiLU())
        self.rec = nn.GRU(HIDDEN, HIDDEN, num_layers=2, dropout=.10, batch_first=True)
        self.output = nn.Linear(HIDDEN, len(ALPHABET))

    def features(self, pcm: torch.Tensor) -> torch.Tensor:
        if pcm.size(-1) < SPAN:
            return pcm.new_empty((*pcm.shape[:-1], 0, 2 * BANDS))
        span = pcm.unfold(-1, SPAN, STRIDE)
        frames = torch.stack((span[..., :WINDOW], span[..., 2 * HOP:]), dim=-2)
        power = torch.fft.rfft(frames * self.window, n=FFT_SIZE).abs().square()
        energy = (power @ self.mel.T).clamp_min(1e-8).log10().clamp(max=4)
        return energy.mul(.25).add(1).flatten(-2)

    def forward(self, features: torch.Tensor, state=None):
        hidden, state = self.rec(self.project(features), state)
        return self.output(hidden), state


class ASRStream:
    """Carry only the unfinished acoustic window and the recurrent state."""

    def __init__(self, model: CausalTranscriber):
        self.model = model.eval()
        self.pending = model.window.new_empty(0)
        self.state = None

    @torch.no_grad()
    def push(self, pcm16le: bytes) -> torch.Tensor:
        samples = np.frombuffer(pcm16le, dtype="<i2").astype(np.float32) / 32768
        audio = torch.cat((self.pending, torch.from_numpy(samples).to(self.pending.device)))
        frames = self.model.features(audio)
        self.pending = audio[len(frames) * STRIDE:].clone()
        if not len(frames):
            return frames.new_empty((0, len(ALPHABET)))
        logits, self.state = self.model(frames[None], self.state)
        return logits[0]


class GreedyCTC:
    """Collapse frame repetitions across packets and retain blank boundaries."""

    def __init__(self):
        self.previous = 0
        self.frames = 0

    def push(self, codes: list[int]) -> list[tuple[str, int]]:
        emitted = []
        for index, token in enumerate(codes, self.frames):
            if token and token != self.previous:
                emitted.append((ALPHABET[token], SPAN + index * STRIDE))
            self.previous = token
        self.frames += len(codes)
        return emitted


def transcript(codes: list[int]) -> str:
    return " ".join("".join(symbol for symbol, _ in GreedyCTC().push(codes)).split())


def load_checkpoint(path: Path) -> CausalTranscriber:
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record["version"] != 1 or tuple(record["alphabet"]) != ALPHABET:
        raise ValueError("incompatible SOMA transcription checkpoint")
    model = CausalTranscriber()
    model.load_state_dict(record["state_dict"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--events", action="store_true", help="include per-character causal emission offsets")
    args = parser.parse_args()
    torch.set_num_threads(1)
    stream = ASRStream(load_checkpoint(args.checkpoint))
    decoder = GreedyCTC()
    emissions = []
    with wave.open(str(args.wav), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, SAMPLE_RATE):
            parser.error("expected mono 16 kHz PCM16 WAV")
        while pcm := source.readframes(STRIDE // 2):
            emissions.extend(decoder.push(stream.push(pcm).argmax(dim=-1).tolist()))
    if not decoder.frames:
        parser.error("audio is shorter than one acoustic frame")
    result = {"text": " ".join("".join(char for char, _ in emissions).split()),
              "frames": decoder.frames}
    if args.events:
        result["events"] = [{"character": char, "through_sample": sample} for char, sample in emissions]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
