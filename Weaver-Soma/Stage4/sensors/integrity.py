"""Independent, fail-closed *synthetic sensor* integrity checks for SOMA.

No motor or physical-safety control. Does not prove readings are authentic or true:
CRC is accidental-corruption detection, not a signature. A smooth biased sensor
without a second independent witness cannot generally be detected.
Microcontroller timestamps MUST be translated to host monotonic time before
calling this code, with bounded synchronization error.
"""
from dataclasses import dataclass
import math
import struct
import zlib


@dataclass(frozen=True)
class ChannelSpec:
    channel: str
    sensor_id: str
    unit: str
    gain: float                   # physical units = raw * gain + offset
    offset: float
    valid_min: float              # plausible physical range, NOT safe limit
    valid_max: float
    max_rate_per_second: float    # generously bounded engineering assumption
    max_age_ns: int = 250_000_000
    stuck_tolerance: float = 0.00001
    witness_min_change: float = 0.005
    independent_tolerance: float = 0.15
    recover_good_samples: int = 2


DEFAULT_SPECS = {
    "pressure": ChannelSpec("pressure", "skin.pressure.0", "kPa", 1.0, 0.0, 0.0, 160.0, 2000.0, independent_tolerance=8.0, witness_min_change=1.0),
    "temperature": ChannelSpec("temperature", "skin.temp.0", "degC", 1.0, 0.0, -20.0, 120.0, 120.0, independent_tolerance=4.0, witness_min_change=0.5),
    "joint": ChannelSpec("joint", "joint.load.0", "fraction", 1.0, 0.0, -0.1, 1.5, 10.0, independent_tolerance=0.08, witness_min_change=0.015),
}


@dataclass(frozen=True)
class SensorPacket:
    sensor_id: str
    channel: str
    sequence: int
    captured_ns: int
    raw_value: float
    checksum: int

    @staticmethod
    def checksum_for(sensor_id: str, channel: str, sequence: int, captured_ns: int, raw_value: float) -> int:
        # Deterministic encoding; non-cryptographic, no security against forgery.
        payload = (sensor_id + "\x00" + channel + "\x00").encode("utf-8")
        payload += struct.pack("<qqd", sequence, captured_ns, raw_value)
        return zlib.crc32(payload) & 0xFFFFFFFF

    @classmethod
    def create(cls, spec: ChannelSpec, sequence: int, captured_ns: int, physical_value: float):
        raw = (physical_value - spec.offset) / spec.gain
        crc = cls.checksum_for(spec.sensor_id, spec.channel, sequence, captured_ns, raw)
        return cls(spec.sensor_id, spec.channel, sequence, captured_ns, raw, crc)


@dataclass(frozen=True)
class HealthReport:
    channel: str
    trusted: bool
    value: float | None
    reasons: tuple[str, ...]
    consecutive_good: int


class HealthMonitor:
    """Conservative hardware-independent checks; NOT an AI health estimator.

    Each integrity check is deterministic. Inputs `witness_change` and
    `independent_reference` must originate from distinct hardware; never
    supply predicted SOMA model output or unverified motor *commands*.
    """
    def __init__(self, specs: dict[str, ChannelSpec] | None = None):
        self.specs = dict(DEFAULT_SPECS if specs is None else specs)
        for s in self.specs.values():
            if s.gain == 0 or s.recover_good_samples < 1 or s.valid_min >= s.valid_max:
                raise ValueError("invalid channel calibration/config")
        self._last_sequence: dict[str, int] = {}
        self._last_accepted: dict[str, tuple[int, float]] = {}
        self._good: dict[str, int] = {name: 0 for name in self.specs}
        self._faulted: dict[str, bool] = {name: False for name in self.specs}

    def mark_missing(self, channel: str) -> HealthReport:
        if channel not in self.specs:
            raise ValueError("unknown sensor")
        self._good[channel] = 0
        self._faulted[channel] = True
        return HealthReport(channel, False, None, ("missing",), 0)

    def inspect(self, packet: SensorPacket, arrival_ns: int, *,
                witness_change: float | None = None,
                independent_reference: float | None = None) -> HealthReport:
        channel = packet.channel
        if channel not in self.specs:
            raise ValueError("unknown sensor channel")
        spec = self.specs[channel]
        reasons: list[str] = []
        if packet.sensor_id != spec.sensor_id:
            reasons.append("sensor_id_mismatch")
        if not (isinstance(packet.sequence, int) and packet.sequence >= 0 and
                isinstance(packet.captured_ns, int) and packet.captured_ns >= 0 and
                isinstance(arrival_ns, int) and arrival_ns >= 0):
            reasons.append("invalid_timestamp_or_sequence")
        if not isinstance(packet.raw_value, (float, int)) or not math.isfinite(packet.raw_value):
            reasons.append("nonfinite")
        if not reasons:
            checksum = SensorPacket.checksum_for(packet.sensor_id, channel, packet.sequence, packet.captured_ns, float(packet.raw_value))
            if packet.checksum != checksum:
                reasons.append("checksum")
            if packet.sequence <= self._last_sequence.get(channel, -1):
                reasons.append("sequence_replay")
            age = arrival_ns - packet.captured_ns
            if age < 0:
                reasons.append("future_timestamp")
            if age > spec.max_age_ns:
                reasons.append("stale_timestamp")
        value = (float(packet.raw_value * spec.gain + spec.offset)
                 if isinstance(packet.raw_value, (int, float)) and math.isfinite(packet.raw_value) else None)
        if value is not None:
            if not spec.valid_min <= value <= spec.valid_max:
                reasons.append("implausible_range")
            last = self._last_accepted.get(channel)
            if last is not None:
                prev_ns, prev_value = last
                dt = (packet.captured_ns - prev_ns) * 1e-9
                if dt <= 0:
                    reasons.append("nonmonotonic_clock")
                elif abs(value - prev_value) / dt > spec.max_rate_per_second:
                    reasons.append("implausible_slew")
                # A constant signal is NOT a fault on its own. Only flag if
                # separate, independently measured change is expected.
                if (witness_change is not None and math.isfinite(witness_change)
                        and witness_change >= spec.witness_min_change
                        and abs(value - prev_value) <= spec.stuck_tolerance):
                    reasons.append("stuck_with_witness")
            if independent_reference is not None:
                if not math.isfinite(independent_reference):
                    reasons.append("invalid_reference")
                elif abs(value - independent_reference) > spec.independent_tolerance:
                    reasons.append("redundancy_disagreement")
        if not reasons:
            self._last_sequence[channel] = packet.sequence
            self._last_accepted[channel] = (packet.captured_ns, value)
            self._good[channel] += 1
            trusted = self._good[channel] >= spec.recover_good_samples
            if trusted:
                self._faulted[channel] = False
            else:
                reasons.append("reacquiring")
            return HealthReport(channel, trusted, value if trusted else None, tuple(reasons), self._good[channel])
        self._good[channel] = 0
        self._faulted[channel] = True
        # Still note a monotonically newer sequence when the payload is sound;
        # prevents a rejected plausible frame being replayed as a fresh sample.
        if not any(r in reasons for r in ("checksum", "sensor_id_mismatch", "invalid_timestamp_or_sequence")):
            self._last_sequence[channel] = max(packet.sequence, self._last_sequence.get(channel, -1))
        return HealthReport(channel, False, None, tuple(sorted(set(reasons))), 0)
