"""Time-aligned sensor windows matching U1/U2 obs[B,8,7] and q[B,8,3].

No model may interpret an untrusted channel's last plausible value as healthy.
The separate supervisor must enforce an independent safe state when stop=True.
"""
from collections import deque
from dataclasses import dataclass
import numpy as np
from .integrity import DEFAULT_SPECS, SensorPacket, HealthMonitor, HealthReport

CHANNELS = ("pressure", "temperature", "joint")


@dataclass(frozen=True)
class AssembledFrame:
    obs: np.ndarray
    quality: np.ndarray
    stop_requested: bool
    reasons: dict[str, tuple[str, ...]]


class SOMASensorWindow:
    """Read-only bridge for SOMA U1 and U2 numerical input contracts.

    Keep sensor acquisition/health separate from SOMA neural confidence; this
    layer never commands motors, signs intents, or sets physical safety limits.
    """
    def __init__(self, monitor: HealthMonitor | None = None, history: int = 8,
                 safety_boundaries: tuple[float, float, float] | None = None):
        if history != 8:
            raise ValueError("current SOMA U1/U2 require exactly 8 history frames")
        self.monitor = monitor or HealthMonitor()
        self.history = history
        if safety_boundaries is not None:
            bounds = np.asarray(safety_boundaries, dtype=np.float32)
            if bounds.shape != (3,) or not np.isfinite(bounds).all():
                raise ValueError("invalid research safety boundaries")
        self.safety_boundaries = safety_boundaries
        self._frames: deque[tuple[np.ndarray, np.ndarray, int]] = deque(maxlen=history)
        self._last_good_value: dict[str, tuple[int, float]] = {}
        self._cache: dict[str, tuple[int, float]] = {}
        self._reacquire: dict[str, int] = {}

    def push(self, packets: dict[str, SensorPacket], frame_ns: int, *,
             independently_verified_contact: bool | None = None,
             witness_changes: dict[str, float] | None = None,
             independent_references: dict[str, float] | None = None) -> AssembledFrame:
        if not isinstance(frame_ns, int) or frame_ns < 0:
            raise ValueError("frame clock must be host monotonic nanoseconds")
        if self._frames and frame_ns <= self._frames[-1][2]:
            raise ValueError("frame timestamp not strictly increasing")
        witness_changes = witness_changes or {}
        independent_references = independent_references or {}
        values = np.zeros(3, dtype=np.float32)
        slopes = np.zeros(3, dtype=np.float32)
        flags = np.zeros(3, dtype=np.float32)
        reasons: dict[str, tuple[str, ...]] = {}
        for j, channel in enumerate(CHANNELS):
            fresh_sample = channel in packets
            if fresh_sample:
                report = self.monitor.inspect(packets[channel], frame_ns,
                    witness_change=witness_changes.get(channel),
                    independent_reference=independent_references.get(channel))
                if report.trusted and report.value is not None:
                    self._cache[channel] = (packets[channel].captured_ns, report.value)
                    self._reacquire.pop(channel, None)
                elif report.reasons == ("reacquiring",):
                    # Valid first sample, but not enough independent samples
                    # to grant trust; don't reset on an expected 5-Hz gap.
                    self._cache.pop(channel, None)
                    self._reacquire[channel] = packets[channel].captured_ns
                else:
                    self._cache.pop(channel, None)
                    self._reacquire.pop(channel, None)
            else:
                cached = self._cache.get(channel)
                ttl = self.monitor.specs[channel].max_age_ns
                if cached is not None and 0 <= frame_ns - cached[0] <= ttl:
                    # Different physical sensors have different acquisition rates.
                    # Reuse the LAST independently checked reading only within TTL.
                    report = HealthReport(channel, True, cached[1], ("fresh_cache",), 0)
                elif (channel in self._reacquire and
                      0 <= frame_ns - self._reacquire[channel] <= ttl):
                    report = HealthReport(channel, False, None,
                                          ("awaiting_next_sample",), 0)
                else:
                    report = self.monitor.mark_missing(channel)
                    self._cache.pop(channel, None)
                    self._reacquire.pop(channel, None)
            reasons[channel] = report.reasons
            if report.trusted and report.value is not None:
                flags[j] = 1
                values[j] = report.value
                last = self._last_good_value.get(channel)
                if fresh_sample and last is not None and frame_ns > last[0]:
                    # Do not compute derivatives across a missing or faulty frame.
                    prior_trusted = bool(self._frames and self._frames[-1][1][j])
                    if prior_trusted:
                        slopes[j] = (report.value - last[1]) / ((frame_ns - last[0]) * 1e-9)
                if fresh_sample:
                    self._last_good_value[channel] = (frame_ns, report.value)
        # Contact state must be an independently authenticated/calibrated switch;
        # do not infer from pressure if its integrity is unknown.
        contact = 1.0 if independently_verified_contact is True else 0.0
        row = np.concatenate([values, slopes, [contact]]).astype(np.float32)
        self._frames.append((row, flags, frame_ns))
        obs = np.zeros((8, 7), dtype=np.float32)
        quality = np.zeros((8, 3), dtype=np.float32)
        latest = list(self._frames)
        for offset, (a, q, _) in enumerate(latest, start=8-len(latest)):
            obs[offset, :] = a
            quality[offset, :] = q
        stop = not bool(np.all(flags == 1))
        if self.safety_boundaries is not None:
            # These are NOT certified limits: caller provides fixture-specific
            # approved thresholds, and the actual MCU enforces its own limits.
            limit = np.asarray(self.safety_boundaries, dtype=np.float32)
            if limit.shape != (3,) or not np.isfinite(limit).all():
                raise ValueError("invalid research safety boundaries")
            if np.any((values >= limit) & (flags > 0)):
                stop = True
                reasons["supervisor"] = ("illustrative_boundary_exceeded",)
        return AssembledFrame(obs, quality, stop, reasons)
