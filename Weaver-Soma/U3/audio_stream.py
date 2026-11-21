"""Assemble timestamped mono PCM into bounded, auditable U3 audio windows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from audio_spectral import SAMPLE_RATE


def _sample_index(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("sample offsets must be non-negative integers")
    return value


@dataclass(frozen=True)
class PacketReceipt:
    accepted_samples: int
    duplicate_samples: int
    late_samples: int


@dataclass(frozen=True)
class AudioWindow:
    start_sample: int
    end_sample: int
    samples: torch.Tensor
    valid: torch.Tensor
    received: torch.Tensor

    def modalities(self):
        """Build one audio observation for the normal full U3 inference path."""
        return {"audio": {"values": self.samples.unsqueeze(0), "valid": self.valid.unsqueeze(0)}}


class AudioWindowBuffer:
    """Use one buffer per source with a caller-controlled sample watermark.

    A seal fixes all samples below its offset. Overlapping windows retain
    the sealed portion needed by the next window. Serialize calls per buffer.
    """

    sample_rate = SAMPLE_RATE

    def __init__(
        self, *, start_sample: int = 0, window_samples: int = 5 * SAMPLE_RATE,
        hop_samples: int | None = None, max_pending_samples: int | None = None,
    ):
        self._next_start = _sample_index(start_sample)
        self._sealed_until = self._next_start
        self.window_samples = _sample_index(window_samples)
        self.hop_samples = self.window_samples if hop_samples is None else _sample_index(hop_samples)
        self.capacity = 2 * self.window_samples if max_pending_samples is None else _sample_index(max_pending_samples)
        if not 1 <= self.hop_samples <= self.window_samples <= self.capacity:
            raise ValueError("require 1 <= hop_samples <= window_samples <= max_pending_samples")
        self._samples = torch.zeros(self.capacity, dtype=torch.float32)
        self._valid = torch.zeros(self.capacity, dtype=torch.bool)
        self._received = torch.zeros(self.capacity, dtype=torch.bool)

    @property
    def next_window_start(self) -> int:
        return self._next_start

    @property
    def sealed_until(self) -> int:
        return self._sealed_until

    def _indices(self, start: int, count: int) -> torch.Tensor:
        ring_offset = start % self.capacity
        return torch.arange(ring_offset, ring_offset + count) % self.capacity

    def push(self, start_sample: int, samples: torch.Tensor, valid: torch.Tensor | None = None) -> PacketReceipt:
        """Store observed mono floats. Identical unsealed overlaps are idempotent."""
        start = _sample_index(start_sample)
        if samples.ndim != 1 or not samples.is_floating_point():
            raise ValueError("audio packet must contain a 1D floating-point PCM tensor")
        if valid is not None and (valid.shape != samples.shape or valid.dtype != torch.bool):
            raise ValueError("audio packet validity must be a matching boolean tensor")
        size = len(samples)
        stop = start + size
        late = min(size, max(0, self.sealed_until - start))
        if late == size:
            return PacketReceipt(0, 0, late)
        start += late
        count = stop - start
        if stop > self.next_window_start + self.capacity:
            raise BufferError("audio packet exceeds the pending sample capacity")
        values = samples[late:].detach().to(device="cpu", dtype=torch.float32)
        usable = torch.isfinite(values)
        if valid is not None:
            usable = usable & valid[late:].detach().to(device="cpu")
        values = values.masked_fill(~usable, 0)

        slots = self._indices(start, count)
        occupied = self._received[slots]
        conflict = occupied & ((self._valid[slots] != usable) | (self._samples[slots] != values))
        if conflict.any():
            raise ValueError("conflicting overlap in unsealed audio packet")
        duplicates = int(occupied.sum())
        self._samples[slots] = values
        self._valid[slots] = usable
        self._received[slots] = True
        return PacketReceipt(count - duplicates, duplicates, late)

    def push_pcm16le(self, start_sample: int, pcm: bytes) -> PacketReceipt:
        """Decode mono, signed little-endian 16-bit PCM."""
        if len(pcm) % 2:
            raise ValueError("PCM16 packet must contain whole samples")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return self.push(start_sample, torch.from_numpy(samples))

    def seal(self, end_sample: int) -> None:
        """Declare packets below this exclusive offset final, including any gaps."""
        end = _sample_index(end_sample)
        if end < self.sealed_until:
            raise ValueError("the audio watermark cannot move backwards")
        self._sealed_until = end

    def pop_window(self) -> AudioWindow | None:
        """Return the next sealed window and release samples outside future windows."""
        start = self.next_window_start
        end = start + self.window_samples
        if end > self.sealed_until:
            return None
        slots = self._indices(start, self.window_samples)
        window = AudioWindow(
            start, end, self._samples[slots], self._valid[slots], self._received[slots],
        )
        expired = slots[:self.hop_samples]
        self._samples[expired] = 0
        self._valid[expired] = False
        self._received[expired] = False
        self._next_start += self.hop_samples
        return window
