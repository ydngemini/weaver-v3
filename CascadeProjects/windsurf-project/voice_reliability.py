#!/usr/bin/env python3
"""Bounded reliability primitives for Weaver's realtime voice transport."""

from __future__ import annotations

import hashlib
import math
import secrets
import struct
import time
import uuid
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


VOICE_PROTOCOL_VERSION = 2
VOICE_FRAME_MAGIC = b"WVR2"
VOICE_FRAME_HEADER = struct.Struct(">4sQQ")
MAX_SEQUENCE = (1 << 63) - 1
RECONNECT_POLICY = {
    "initialDelayMs": 250,
    "factor": 2.0,
    "maxDelayMs": 8_000,
    "jitterRatio": 0.2,
    "maxAttempts": 8,
}


class VoiceProtocolError(ValueError):
    """Raised when a voice transport message violates a hard bound."""


@dataclass(frozen=True)
class VoiceFrame:
    sequence: int
    captured_at_ms: int
    audio: bytes


def encode_voice_frame(sequence: int, captured_at_ms: int, audio: bytes) -> bytes:
    """Encode the efficient v2 binary envelope used by native clients."""

    if isinstance(sequence, bool) or not 1 <= int(sequence) <= MAX_SEQUENCE:
        raise VoiceProtocolError("audio sequence is invalid")
    if isinstance(captured_at_ms, bool) or not 0 <= int(captured_at_ms) <= 10_000_000_000_000:
        raise VoiceProtocolError("capture timestamp is invalid")
    if not isinstance(audio, bytes) or not audio:
        raise VoiceProtocolError("audio frame is empty")
    return VOICE_FRAME_HEADER.pack(VOICE_FRAME_MAGIC, int(sequence), int(captured_at_ms)) + audio


def decode_voice_frame(value: bytes, *, max_audio_bytes: int) -> VoiceFrame:
    if not isinstance(value, bytes) or len(value) <= VOICE_FRAME_HEADER.size:
        raise VoiceProtocolError("audio frame is incomplete")
    if len(value) - VOICE_FRAME_HEADER.size > max_audio_bytes:
        raise VoiceProtocolError("audio frame is too large")
    magic, sequence, captured_at_ms = VOICE_FRAME_HEADER.unpack_from(value)
    if magic != VOICE_FRAME_MAGIC or not 1 <= sequence <= MAX_SEQUENCE:
        raise VoiceProtocolError("audio frame envelope is invalid")
    if captured_at_ms > 10_000_000_000_000:
        raise VoiceProtocolError("capture timestamp is invalid")
    return VoiceFrame(sequence, captured_at_ms, value[VOICE_FRAME_HEADER.size :])


def _finite(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


class VoiceIngressSequencer:
    """Reorder a small PCM window, acknowledge contiguity, and bound gaps."""

    def __init__(
        self,
        *,
        expected_sequence: int = 1,
        max_buffered_frames: int = 24,
        max_forward_gap: int = 96,
        max_jitter_ms: int = 120,
    ) -> None:
        self.expected_sequence = min(max(int(expected_sequence), 1), MAX_SEQUENCE)
        self.max_buffered_frames = min(max(int(max_buffered_frames), 4), 64)
        self.max_forward_gap = min(max(int(max_forward_gap), 8), 256)
        self.max_jitter_ms = min(max(int(max_jitter_ms), 20), 500)
        self._buffer: dict[int, VoiceFrame] = {}
        self._gap_started_ms: float | None = None
        self._last_arrival_ms: float | None = None
        self._last_capture_ms: int | None = None
        self._jitter_ms = 0.0
        self.received = 0
        self.released = 0
        self.duplicates = 0
        self.lost = 0
        self.rejected = 0
        self.max_buffer_depth = 0

    @property
    def ack_sequence(self) -> int:
        return max(0, self.expected_sequence - 1)

    @property
    def jitter_ms(self) -> float:
        return round(min(max(self._jitter_ms, 0.0), 60_000.0), 2)

    def _update_jitter(self, frame: VoiceFrame, arrival_ms: float) -> None:
        if self._last_arrival_ms is not None and self._last_capture_ms is not None:
            arrival_delta = max(0.0, arrival_ms - self._last_arrival_ms)
            capture_delta = max(0.0, float(frame.captured_at_ms - self._last_capture_ms))
            variation = abs(arrival_delta - capture_delta)
            self._jitter_ms += (variation - self._jitter_ms) / 16.0
        self._last_arrival_ms = arrival_ms
        self._last_capture_ms = frame.captured_at_ms

    def _drain(self) -> list[VoiceFrame]:
        released: list[VoiceFrame] = []
        while self.expected_sequence in self._buffer:
            released.append(self._buffer.pop(self.expected_sequence))
            self.expected_sequence += 1
        self.released += len(released)
        self._gap_started_ms = None if not self._buffer else self._gap_started_ms
        return released

    def ingest(self, frame: VoiceFrame, *, arrival_ms: float | None = None) -> dict[str, Any]:
        current_ms = float(arrival_ms if arrival_ms is not None else time.monotonic() * 1_000)
        sequence = frame.sequence
        self.received += 1
        self._update_jitter(frame, current_ms)

        if sequence < self.expected_sequence or sequence in self._buffer:
            self.duplicates += 1
            return self._result([], duplicate=True)
        if sequence - self.expected_sequence > self.max_forward_gap:
            self.rejected += 1
            raise VoiceProtocolError("audio sequence gap is too large")

        self._buffer[sequence] = frame
        self.max_buffer_depth = max(self.max_buffer_depth, len(self._buffer))
        released = self._drain()
        if released:
            return self._result(released)

        if self._gap_started_ms is None:
            self._gap_started_ms = current_ms
        gap_age = current_ms - self._gap_started_ms
        if len(self._buffer) >= self.max_buffered_frames or gap_age >= self.max_jitter_ms:
            next_sequence = min(self._buffer)
            self.lost += max(0, next_sequence - self.expected_sequence)
            self.expected_sequence = next_sequence
            self._gap_started_ms = None
            released = self._drain()
        return self._result(released)

    def _result(self, frames: list[VoiceFrame], *, duplicate: bool = False) -> dict[str, Any]:
        return {
            "frames": frames,
            "ack_sequence": self.ack_sequence,
            "received_sequence": frames[-1].sequence if frames else None,
            "buffered": len(self._buffer),
            "missing": self.lost,
            "duplicate": duplicate,
            "jitter_ms": self.jitter_ms,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "ack_sequence": self.ack_sequence,
            "expected_sequence": self.expected_sequence,
            "received": self.received,
            "released": self.released,
            "duplicates": self.duplicates,
            "lost": self.lost,
            "rejected": self.rejected,
            "buffered": len(self._buffer),
            "max_buffer_depth": self.max_buffer_depth,
            "jitter_ms": self.jitter_ms,
        }


class VoiceSessionReliability:
    """Session-local sequencing, interruption generations, and safe telemetry."""

    _TELEMETRY_FIELDS = {
        "rttMs",
        "uplinkKbps",
        "packetLoss",
        "captureJitterMs",
        "audioRoute",
        "thermalState",
        "lowPowerMode",
        "deviceClass",
    }
    _AUDIO_ROUTES = {"built-in", "wired", "bluetooth", "unknown"}
    _THERMAL_STATES = {"nominal", "fair", "serious", "critical", "unknown"}
    _DEVICE_CLASSES = {"iphone", "iphone-16e", "ipad", "web", "unknown"}

    def __init__(
        self,
        *,
        expected_sequence: int = 1,
        last_output_ack: int = 0,
        max_jitter_ms: int = 120,
    ) -> None:
        self.session_id = f"voice-{uuid.uuid4().hex[:24]}"
        self.ingress = VoiceIngressSequencer(
            expected_sequence=expected_sequence,
            max_jitter_ms=max_jitter_ms,
        )
        self.server_sequence = max(0, int(last_output_ack))
        self.last_output_ack = max(0, int(last_output_ack))
        self.generation = 0
        self.interruptions = 0
        self.reconnects = 0
        self.telemetry: dict[str, Any] = {}

    def next_server_sequence(self) -> int:
        self.server_sequence += 1
        return self.server_sequence

    def acknowledge_output(self, sequence: Any) -> int:
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise VoiceProtocolError("output acknowledgement is invalid")
        if sequence < self.last_output_ack or sequence > self.server_sequence:
            raise VoiceProtocolError("output acknowledgement is outside the session window")
        self.last_output_ack = sequence
        return sequence

    def interrupt(self) -> int:
        self.generation += 1
        self.interruptions += 1
        return self.generation

    def record_telemetry(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) - self._TELEMETRY_FIELDS:
            raise VoiceProtocolError("device telemetry contains unsupported fields")
        bounded: dict[str, Any] = {}
        for name, maximum in (
            ("rttMs", 30_000.0),
            ("uplinkKbps", 1_000_000.0),
            ("captureJitterMs", 2_000.0),
        ):
            if name in value:
                number = _finite(value[name], default=-1)
                if not 0 <= number <= maximum:
                    raise VoiceProtocolError("device telemetry value is invalid")
                bounded[name] = round(number, 2)
        if "packetLoss" in value:
            loss = _finite(value["packetLoss"], default=-1)
            if not 0 <= loss <= 1:
                raise VoiceProtocolError("device packet loss is invalid")
            bounded["packetLoss"] = round(loss, 4)
        if "lowPowerMode" in value:
            if not isinstance(value["lowPowerMode"], bool):
                raise VoiceProtocolError("low power mode must be boolean")
            bounded["lowPowerMode"] = value["lowPowerMode"]
        for name, allowed in (
            ("audioRoute", self._AUDIO_ROUTES),
            ("thermalState", self._THERMAL_STATES),
            ("deviceClass", self._DEVICE_CLASSES),
        ):
            if name in value:
                item = str(value[name]).strip().lower()
                if item not in allowed:
                    raise VoiceProtocolError("device telemetry category is invalid")
                bounded[name] = item
        self.telemetry = bounded
        return dict(bounded)

    def resume_state(self) -> dict[str, Any]:
        return {
            "expected_sequence": self.ingress.expected_sequence,
            "last_output_ack": self.last_output_ack,
            "reconnects": self.reconnects + 1,
        }

    def snapshot(self) -> dict[str, Any]:
        ingress = self.ingress.snapshot()
        return {
            "protocol_version": VOICE_PROTOCOL_VERSION,
            "session_id": self.session_id,
            "input_ack_sequence": ingress["ack_sequence"],
            "output_ack_sequence": self.last_output_ack,
            "frames_received": ingress["received"],
            "frames_released": ingress["released"],
            "frames_duplicate": ingress["duplicates"],
            "frames_lost": ingress["lost"],
            "frames_rejected": ingress["rejected"],
            "buffer_depth": ingress["buffered"],
            "max_buffer_depth": ingress["max_buffer_depth"],
            "jitter_ms": ingress["jitter_ms"],
            "interruptions": self.interruptions,
            "reconnects": self.reconnects,
            "telemetry": deepcopy(self.telemetry),
        }


class VoiceResumeRegistry:
    """Single-use, bounded reconnect tickets; the brain key is still required."""

    def __init__(self, *, ttl_seconds: int = 600, max_entries: int = 128, clock=time.time) -> None:
        self.ttl_seconds = min(max(int(ttl_seconds), 30), 900)
        self.max_entries = min(max(int(max_entries), 16), 512)
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("ascii", "ignore")).hexdigest()

    def _prune(self) -> None:
        now = float(self._clock())
        expired = [digest for digest, (expiry, _) in self._entries.items() if expiry <= now]
        for digest in expired:
            self._entries.pop(digest, None)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def issue(self, state: dict[str, Any]) -> str:
        self._prune()
        token = secrets.token_urlsafe(32)
        self._entries[self._digest(token)] = (
            float(self._clock()) + self.ttl_seconds,
            deepcopy(state),
        )
        self._prune()
        return token

    def update(self, token: str, state: dict[str, Any]) -> bool:
        self._prune()
        digest = self._digest(str(token or ""))
        existing = self._entries.get(digest)
        if existing is None:
            return False
        expiry, _ = existing
        self._entries[digest] = (expiry, deepcopy(state))
        self._entries.move_to_end(digest)
        return True

    def consume(self, token: Any) -> dict[str, Any] | None:
        self._prune()
        text = str(token or "")
        if not 32 <= len(text) <= 64:
            return None
        stored = self._entries.pop(self._digest(text), None)
        if stored is None or stored[0] <= float(self._clock()):
            return None
        return deepcopy(stored[1])

    def __len__(self) -> int:
        self._prune()
        return len(self._entries)
