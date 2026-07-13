#!/usr/bin/env python3
"""Atomic, revisioned, privacy-safe state for Weaver headless v2."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from awareness_fusion import fuse_awareness
from headless_privacy import (
    degraded_reasons,
    ensure_public_state_safe,
    private_cognition_metadata,
)
from headless_schemas import (
    HEADLESS_SCHEMA_VERSION,
    AwarenessChannels,
    AwarenessPublicState,
    CognitionPublicState,
    FabricLaneCounters,
    FabricLanePublicState,
    FabricLanes,
    FabricPublicState,
    FreshnessState,
    HeadlessDelta,
    HeadlessPublicState,
    HeadlessSnapshot,
    SourceFreshness,
    SystemPublicState,
    VoicePublicState,
)
from weaver_neural_fabric import PENTHOUSE_ZONES


MAX_SNAPSHOT_BYTES = 65_536
MAX_DELTA_BYTES = 16_384
MAX_DELTA_HISTORY = 256
PUBLIC_STATE_FIELDS = ("freshness", "system", "awareness", "voice", "cognition", "fabric")


class HeadlessStateError(RuntimeError):
    """Raised when a public state payload violates a hard safety budget."""


def _bounded_int(value: Any, *, default: int = 0, maximum: int = 1_000_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return min(max(int(value), 0), maximum)


def _bounded_float(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not math.isfinite(number):
        return default
    return round(min(max(number, 0.0), 1.0), 4)


def _bounded_metric(value: Any, *, maximum: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(min(max(number, 0.0), maximum), 4)


def _freshness_from_channel(channel: Any, *, now_ms: int) -> SourceFreshness:
    data = channel if isinstance(channel, dict) else {}
    raw_age = data.get("age_ms")
    age_ms = None if raw_age is None else _bounded_int(raw_age, maximum=3_600_000)
    observed_at_ms = None if age_ms is None else max(0, now_ms - age_ms)
    return SourceFreshness(
        fresh=bool(data.get("fresh", False)),
        age_ms=age_ms,
        confidence=_bounded_float(data.get("confidence")),
        observed_at_ms=observed_at_ms,
    )


def _freshness_from_unix(value: Any, *, now_ms: int, ttl_ms: int) -> SourceFreshness:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return SourceFreshness(fresh=False, age_ms=None, confidence=0.0, observed_at_ms=None)
    observed_at_ms = _bounded_int(float(value) * 1_000, maximum=10_000_000_000_000)
    age_ms = min(max(now_ms - observed_at_ms, 0), 3_600_000)
    confidence = max(0.0, 1.0 - age_ms / max(ttl_ms, 1))
    return SourceFreshness(
        fresh=age_ms <= ttl_ms,
        age_ms=age_ms,
        confidence=round(confidence, 4),
        observed_at_ms=observed_at_ms,
    )


def _lane_state(value: Any) -> FabricLanePublicState:
    lane = value if isinstance(value, dict) else {}
    counters = lane.get("counters") if isinstance(lane.get("counters"), dict) else {}
    return FabricLanePublicState(
        active=_bounded_int(lane.get("active")),
        queued=_bounded_int(lane.get("queued")),
        counters=FabricLaneCounters(
            submitted=_bounded_int(counters.get("submitted")),
            completed=_bounded_int(counters.get("completed")),
            failed=_bounded_int(counters.get("failed")),
            deadlines=_bounded_int(counters.get("deadlines")),
            shed=_bounded_int(counters.get("shed")),
            cancelled=_bounded_int(counters.get("cancelled")),
        ),
    )


def _safe_status(value: Any, *, default: str = "no-data") -> str:
    text = str(value or default).strip().lower()[:32]
    safe = "".join(character for character in text if character.isalnum() or character in "_.:-")
    return safe or default


def build_public_state(
    legacy: dict[str, Any],
    fabric: dict[str, Any],
    cognition: dict[str, Any],
    *,
    now: float | None = None,
) -> HeadlessPublicState:
    """Project internal state into an allowlisted public model.

    Raw cognition, transcripts, model routes, prompts, credentials, ledger tail
    events, and full environment object records are intentionally unreachable.
    """

    current = float(now if now is not None else time.time())
    now_ms = int(current * 1_000)
    perception = cognition.get("perception") if isinstance(cognition.get("perception"), dict) else {}
    raw_channels = perception.get("channels") if isinstance(perception.get("channels"), dict) else {}
    channels = AwarenessChannels(
        body=_freshness_from_channel(raw_channels.get("body"), now_ms=now_ms),
        environment=_freshness_from_channel(raw_channels.get("environment"), now_ms=now_ms),
        camera=_freshness_from_channel(raw_channels.get("camera"), now_ms=now_ms),
        microphone=_freshness_from_channel(raw_channels.get("microphone"), now_ms=now_ms),
    )
    headless_freshness = _freshness_from_unix(
        legacy.get("last_tick_at"), now_ms=now_ms, ttl_ms=15_000
    )
    fusion = fuse_awareness(
        channels=channels.model_dump(mode="python"),
        fabric=fabric,
        cognition=cognition,
        dependencies=(
            legacy.get("dependency_health")
            if isinstance(legacy.get("dependency_health"), dict)
            else None
        ),
        headless_fresh=headless_freshness.fresh,
        now_ms=now_ms,
    )
    fused_sources = fusion["sources"]
    freshness = FreshnessState(
        headless=headless_freshness,
        fabric=fused_sources["fabric"],
        cognition=fused_sources["cognition"],
        dependencies=fused_sources["dependencies"],
        body=channels.body,
        environment=channels.environment,
        camera=channels.camera,
        microphone=channels.microphone,
    )

    reasons = degraded_reasons(legacy, fabric, cognition)
    if fusion["status"] in {"degraded", "no-data"}:
        reasons.append("awareness-degraded")
    active = bool(legacy.get("active", False))
    ledger = fabric.get("ledger") if isinstance(fabric.get("ledger"), dict) else {}
    ready = active and not reasons and bool(ledger.get("valid", False))
    system = SystemPublicState(
        active=active,
        ready=ready,
        status="inactive" if not active else ("degraded" if reasons else "nominal"),
        uptime_seconds=_bounded_int(current - float(legacy.get("started_at", current) or current)),
        degraded_reasons=reasons,
    )

    body = perception.get("body") if isinstance(perception.get("body"), dict) else {}
    world = perception.get("world") if isinstance(perception.get("world"), dict) else {}
    zone_value = str(world.get("zone") or "").strip().lower()
    zone = zone_value if zone_value in PENTHOUSE_ZONES else None
    objects = world.get("objects") if isinstance(world.get("objects"), dict) else {}
    awareness = AwarenessPublicState(
        fusion_version=fusion["fusion_version"],
        status=fusion["status"],
        confidence=fusion["confidence"],
        degraded_reasons=fusion["degraded_reasons"],
        body_revision=_bounded_int(perception.get("body_revision")),
        world_revision=_bounded_int(perception.get("world_revision")),
        awake=bool(body.get("awake", False)),
        zone=zone,
        visible_objects=min(
            sum(1 for item in objects.values() if isinstance(item, dict) and item.get("visible", False)),
            20,
        ),
        channels=channels,
        sources=fused_sources,
        dependencies=fusion["dependencies"],
    )

    voice_data = (
        legacy.get("voice_realtime")
        if isinstance(legacy.get("voice_realtime"), dict)
        else {}
    )
    prewarm = voice_data.get("prewarm") if isinstance(voice_data.get("prewarm"), dict) else {}
    slo = voice_data.get("slo") if isinstance(voice_data.get("slo"), dict) else {}
    prewarm_status = _safe_status(prewarm.get("status"), default="pending")
    slo_status = _safe_status(slo.get("status"), default="no-data")
    if voice_data.get("last_error"):
        voice_status = "degraded"
    elif prewarm_status in {"pending", "warming", "prewarming"}:
        voice_status = "warming"
    elif slo_status == "no-data":
        voice_status = "no-data"
    else:
        voice_status = "ready"
    transport_data = (
        voice_data.get("last_transport")
        if isinstance(voice_data.get("last_transport"), dict)
        else {}
    )
    device_data = (
        transport_data.get("telemetry")
        if isinstance(transport_data.get("telemetry"), dict)
        else {}
    )
    audio_route = str(device_data.get("audioRoute") or "unknown").lower()
    if audio_route not in {"built-in", "wired", "bluetooth", "unknown"}:
        audio_route = "unknown"
    thermal_state = str(device_data.get("thermalState") or "unknown").lower()
    if thermal_state not in {"nominal", "fair", "serious", "critical", "unknown"}:
        thermal_state = "unknown"
    device_class = str(device_data.get("deviceClass") or "unknown").lower()
    if device_class not in {"iphone", "iphone-16e", "ipad", "web", "unknown"}:
        device_class = "unknown"
    voice = VoicePublicState(
        configured=bool(voice_data),
        status=voice_status,
        prewarm_status=prewarm_status,
        slo_status=slo_status,
        sessions_started=_bounded_int(voice_data.get("sessions_started")),
        reaction_target_ms=min(max(_bounded_int(slo.get("reaction_target_ms"), default=200), 50), 1_000),
        queue_target_ms=min(max(_bounded_int(slo.get("queue_target_ms"), default=120), 20), 2_000),
        semantic_soft_target_ms=min(
            max(_bounded_int(slo.get("semantic_target_ms"), default=3_000), 500),
            15_000,
        ),
        transport={
            "protocol_version": min(
                max(_bounded_int(transport_data.get("protocol_version"), default=1), 1),
                2,
            ),
            "input_ack_sequence": _bounded_int(transport_data.get("input_ack_sequence")),
            "output_ack_sequence": _bounded_int(transport_data.get("output_ack_sequence")),
            "frames_received": _bounded_int(transport_data.get("frames_received")),
            "frames_released": _bounded_int(transport_data.get("frames_released")),
            "frames_duplicate": _bounded_int(transport_data.get("frames_duplicate")),
            "frames_lost": _bounded_int(transport_data.get("frames_lost")),
            "frames_rejected": _bounded_int(transport_data.get("frames_rejected")),
            "buffer_depth": _bounded_int(transport_data.get("buffer_depth"), maximum=64),
            "max_buffer_depth": _bounded_int(
                transport_data.get("max_buffer_depth"), maximum=64
            ),
            "jitter_ms": _bounded_metric(
                transport_data.get("jitter_ms"), maximum=60_000
            ) or 0.0,
            "interruptions": _bounded_int(transport_data.get("interruptions")),
            "reconnects": _bounded_int(transport_data.get("reconnects")),
            "device": {
                "present": bool(device_data),
                "rtt_ms": _bounded_metric(device_data.get("rttMs"), maximum=30_000),
                "packet_loss": _bounded_metric(device_data.get("packetLoss"), maximum=1.0),
                "capture_jitter_ms": _bounded_metric(
                    device_data.get("captureJitterMs"), maximum=2_000
                ),
                "audio_route": audio_route,
                "thermal_state": thermal_state,
                "low_power_mode": (
                    device_data.get("lowPowerMode")
                    if isinstance(device_data.get("lowPowerMode"), bool)
                    else None
                ),
                "device_class": device_class,
            },
        },
    )

    private = private_cognition_metadata(legacy)
    embodiment = (
        cognition.get("embodiment") if isinstance(cognition.get("embodiment"), dict) else {}
    )
    cognition_public = CognitionPublicState(
        status="guarded" if cognition.get("status") == "guarded" else "nominal",
        phase="idle",
        **private,
        observations=_bounded_int(perception.get("observations")),
        intent_evaluations=_bounded_int(embodiment.get("intent_evaluations")),
    )

    lanes = fabric.get("lanes") if isinstance(fabric.get("lanes"), dict) else {}
    fabric_public = FabricPublicState(
        status=(
            str(fabric.get("status"))
            if fabric.get("status") in {"nominal", "watch", "guarded"}
            else "guarded"
        ),
        pressure=_bounded_float(
            (fabric.get("accelerator") or {}).get("pressure")
            if isinstance(fabric.get("accelerator"), dict)
            else 0.0
        ),
        ledger_valid=bool(ledger.get("valid", False)),
        ledger_sequence=_bounded_int(ledger.get("sequence")),
        lanes=FabricLanes(
            realtime=_lane_state(lanes.get("realtime")),
            interactive=_lane_state(lanes.get("interactive")),
            embodiment=_lane_state(lanes.get("embodiment")),
            background=_lane_state(lanes.get("background")),
        ),
    )
    public_state = HeadlessPublicState(
        freshness=freshness,
        system=system,
        awareness=awareness,
        voice=voice,
        cognition=cognition_public,
        fabric=fabric_public,
    )
    ensure_public_state_safe(public_state.model_dump(mode="json"))
    return public_state


def _canonical_bytes(value: Any) -> bytes:
    if isinstance(value, (HeadlessPublicState, HeadlessSnapshot, HeadlessDelta)):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class HeadlessStateStore:
    """Serialize public updates into monotonic snapshots and bounded deltas."""

    def __init__(
        self,
        *,
        max_history: int = MAX_DELTA_HISTORY,
        max_snapshot_bytes: int = MAX_SNAPSHOT_BYTES,
        max_delta_bytes: int = MAX_DELTA_BYTES,
    ) -> None:
        self.max_history = min(max(int(max_history), 1), MAX_DELTA_HISTORY)
        self.max_snapshot_bytes = min(max(int(max_snapshot_bytes), 1_024), MAX_SNAPSHOT_BYTES)
        self.max_delta_bytes = min(max(int(max_delta_bytes), 512), MAX_DELTA_BYTES)
        self._lock = asyncio.Lock()
        self._changed = asyncio.Condition(self._lock)
        self._revision = 0
        self._content_digest: bytes | None = None
        self._snapshot: HeadlessSnapshot | None = None
        self._history: deque[HeadlessDelta] = deque(maxlen=self.max_history)

    @property
    def revision(self) -> int:
        return self._revision

    async def publish(self, state: HeadlessPublicState | dict[str, Any]) -> HeadlessSnapshot:
        public = state if isinstance(state, HeadlessPublicState) else HeadlessPublicState.model_validate(state)
        content = public.model_dump(mode="json")
        ensure_public_state_safe(content)
        digest = _canonical_bytes(content)
        async with self._changed:
            if self._snapshot is not None and digest == self._content_digest:
                return self._snapshot
            base_revision = self._revision
            previous = (
                self._snapshot.model_dump(mode="json", include=set(PUBLIC_STATE_FIELDS))
                if self._snapshot is not None
                else {}
            )
            changes = {
                field: content[field]
                for field in PUBLIC_STATE_FIELDS
                if previous.get(field) != content[field]
            }
            revision = base_revision + 1
            generated_at = datetime.now(timezone.utc)
            snapshot = HeadlessSnapshot(
                schema_version=HEADLESS_SCHEMA_VERSION,
                revision=revision,
                generated_at=generated_at,
                **public.model_dump(),
            )
            delta = HeadlessDelta(
                schema_version=HEADLESS_SCHEMA_VERSION,
                base_revision=base_revision,
                revision=revision,
                generated_at=generated_at,
                changes=changes,
            )
            if len(_canonical_bytes(snapshot)) > self.max_snapshot_bytes:
                raise HeadlessStateError("public snapshot exceeds its hard size budget")
            if len(_canonical_bytes(delta)) > self.max_delta_bytes:
                raise HeadlessStateError("public delta exceeds its hard size budget")
            self._revision = revision
            self._content_digest = digest
            self._snapshot = snapshot
            self._history.append(delta)
            self._changed.notify_all()
            return snapshot

    async def snapshot(self) -> HeadlessSnapshot | None:
        async with self._lock:
            return self._snapshot

    async def changes_since(self, revision: int) -> list[HeadlessDelta] | None:
        """Return a contiguous delta chain or ``None`` when resync is required."""

        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            return None
        async with self._lock:
            if revision == self._revision:
                return []
            if revision > self._revision or not self._history:
                return None
            selected = [delta for delta in self._history if delta.revision > revision]
            if not selected or selected[0].base_revision != revision:
                return None
            expected = revision
            for delta in selected:
                if delta.base_revision != expected:
                    return None
                expected = delta.revision
            return selected if expected == self._revision else None

    async def wait_for_revision(
        self,
        revision: int,
        *,
        timeout: float | None = None,
    ) -> HeadlessSnapshot | None:
        """Wait for a newer snapshot without polling or inventing revisions."""

        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            return None
        async with self._changed:
            if self._revision <= revision:
                try:
                    await asyncio.wait_for(
                        self._changed.wait_for(lambda: self._revision > revision),
                        timeout=timeout,
                    )
                except TimeoutError:
                    return None
            return self._snapshot
