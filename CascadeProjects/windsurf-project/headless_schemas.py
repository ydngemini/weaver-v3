#!/usr/bin/env python3
"""Strict public contracts for Weaver headless v2.

The realtime protocol is data-only.  In particular, ``capsule_submit`` accepts
the exact output shape produced by :class:`IntentCompiler`; it does not define
an alternate command or execution format.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from weaver_neural_fabric import (
    MEMORY_CATEGORIES,
    PENTHOUSE_INTERACTIONS,
    PENTHOUSE_ZONES,
    POSE_SLOTS,
    SENSOR_CAPABILITIES,
)


HEADLESS_SCHEMA_VERSION = 2
N8N_CONTRACT_VERSION = "weaver-headless-n8n-v1"
MAX_CAPSULE_ACTIONS = 8
MAX_CAPSULE_TTL_MS = 60_000
MAX_SUBSCRIPTIONS = 8


class StrictContract(BaseModel):
    """Closed, immutable protocol object."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


def _safe_text(value: str, *, limit: int, field: str) -> str:
    text = " ".join(value.split())
    if (
        not text
        or len(text) > limit
        or any(not character.isprintable() for character in text)
        or "<" in text
        or ">" in text
    ):
        raise ValueError(f"{field} must be bounded printable text")
    return text


class BoneRotation(StrictContract):
    x: float = Field(ge=-1.05, le=1.05)
    y: float = Field(ge=-1.05, le=1.05)
    z: float = Field(ge=-1.05, le=1.05)

    @field_validator("x", "y", "z")
    @classmethod
    def finite_rotation(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("bone rotation must be finite")
        return value


class PoseAction(StrictContract):
    id: str = Field(pattern=r"^a[1-8]$")
    type: Literal["pose"]
    values: dict[str, float]

    @field_validator("values")
    @classmethod
    def bounded_pose(cls, values: dict[str, float]) -> dict[str, float]:
        if not 1 <= len(values) <= len(POSE_SLOTS) or set(values) - POSE_SLOTS:
            raise ValueError("pose controls are outside the IntentCompiler allowlist")
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values.values()):
            raise ValueError("pose controls must be finite and within -1..1")
        return values


class BonesAction(StrictContract):
    id: str = Field(pattern=r"^a[1-8]$")
    type: Literal["bones"]
    bones: dict[str, BoneRotation]

    @field_validator("bones")
    @classmethod
    def bounded_bones(cls, bones: dict[str, BoneRotation]) -> dict[str, BoneRotation]:
        if not 1 <= len(bones) <= 64:
            raise ValueError("bones must contain 1..64 controls")
        for name in bones:
            if not name or len(name) > 80 or not all(
                character.isalnum() or character in "_.:-" for character in name
            ):
                raise ValueError("bone name is invalid")
        return bones


class NavigateAction(StrictContract):
    id: str = Field(pattern=r"^a[1-8]$")
    type: Literal["navigate"]
    zone: Literal["center", "window", "kitchen", "gallery", "lounge"]


class InteractAction(StrictContract):
    id: str = Field(pattern=r"^a[1-8]$")
    type: Literal["interact"]
    interaction: Literal[
        "window_glazing",
        "balcony_rail",
        "lounge_sofa",
        "coffee_table",
        "serving_tray",
        "reading_book",
        "lounge_chair",
        "floor_lamp",
        "fireplace",
        "sculpture_orb",
        "gallery_art",
        "kitchen_island",
        "bar_stool",
        "pendant_lights",
        "fruit_bowl",
        "wine_bottle",
        "back_bar",
        "bed",
        "night_lamp",
        "indoor_plant",
    ]


class SpeakAction(StrictContract):
    id: str = Field(pattern=r"^a[1-8]$")
    type: Literal["speak"]
    text: str = Field(min_length=1, max_length=800)

    @field_validator("text")
    @classmethod
    def bounded_speech(cls, value: str) -> str:
        return _safe_text(value, limit=800, field="speak.text")


class ObserveAction(StrictContract):
    id: str = Field(pattern=r"^a[1-8]$")
    type: Literal["observe"]
    sensor: Literal["body", "environment", "camera", "microphone"]


class RememberAction(StrictContract):
    id: str = Field(pattern=r"^a[1-8]$")
    type: Literal["remember"]
    category: Literal["preference", "constraint", "observation", "relationship"]
    note: str = Field(min_length=1, max_length=500)

    @field_validator("note")
    @classmethod
    def bounded_note(cls, value: str) -> str:
        return _safe_text(value, limit=500, field="remember.note")


CapsuleAction = Annotated[
    Union[
        PoseAction,
        BonesAction,
        NavigateAction,
        InteractAction,
        SpeakAction,
        ObserveAction,
        RememberAction,
    ],
    Field(discriminator="type"),
]


class IntentPreconditions(StrictContract):
    world_revision: int = Field(ge=0, le=1_000_000_000)
    body_revision: int = Field(ge=0, le=1_000_000_000)
    requires_awake: bool
    max_duration_ms: int = Field(ge=500, le=30_000)


class IntentIntegrity(StrictContract):
    algorithm: Literal["hmac-sha256"]
    key_id: str = Field(pattern=r"^[0-9a-f]{12}$")
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")


class SignedIntentCapsule(StrictContract):
    version: Literal[1]
    capsule_id: str = Field(pattern=r"^cap-[0-9a-f]{24}$")
    issued_at_ms: int = Field(ge=0)
    expires_at_ms: int = Field(ge=0)
    priority: Literal["realtime", "interactive", "embodiment"]
    goal: str = Field(min_length=1, max_length=160)
    preconditions: IntentPreconditions
    actions: list[CapsuleAction] = Field(min_length=1, max_length=MAX_CAPSULE_ACTIONS)
    rollback: list[
        Literal[
            "reset_pose",
            "reset_bones",
            "stop_locomotion",
            "cancel_interaction",
            "stop_speech",
        ]
    ] = Field(max_length=5)
    integrity: IntentIntegrity

    @field_validator("goal")
    @classmethod
    def bounded_goal(cls, value: str) -> str:
        return _safe_text(value, limit=160, field="goal")

    @model_validator(mode="after")
    def exact_compiler_shape(self) -> "SignedIntentCapsule":
        ttl_ms = self.expires_at_ms - self.issued_at_ms
        if not 1_000 <= ttl_ms <= MAX_CAPSULE_TTL_MS:
            raise ValueError("capsule lifetime is outside the compiler bounds")
        if self.preconditions.max_duration_ms > min(ttl_ms, 30_000):
            raise ValueError("maximum duration exceeds the capsule lifetime")
        if [action.id for action in self.actions] != [
            f"a{index}" for index in range(1, len(self.actions) + 1)
        ]:
            raise ValueError("action identifiers must follow compiler order")
        rollback_map = {
            "pose": "reset_pose",
            "bones": "reset_bones",
            "navigate": "stop_locomotion",
            "interact": "cancel_interaction",
            "speak": "stop_speech",
        }
        expected: list[str] = []
        for action in reversed(self.actions):
            operation = rollback_map.get(action.type)
            if operation and operation not in expected:
                expected.append(operation)
        if list(self.rollback) != expected:
            raise ValueError("rollback does not match the compiled actions")
        return self


class SourceFreshness(StrictContract):
    fresh: bool
    age_ms: int | None = Field(default=None, ge=0, le=3_600_000)
    confidence: float = Field(ge=0.0, le=1.0)
    observed_at_ms: int | None = Field(default=None, ge=0)


class FreshnessState(StrictContract):
    headless: SourceFreshness
    fabric: SourceFreshness
    cognition: SourceFreshness
    dependencies: SourceFreshness
    body: SourceFreshness
    environment: SourceFreshness
    camera: SourceFreshness
    microphone: SourceFreshness


class SystemPublicState(StrictContract):
    active: bool
    ready: bool
    status: Literal["nominal", "degraded", "inactive"]
    uptime_seconds: int = Field(ge=0)
    degraded_reasons: list[
        Literal[
            "headless-degraded",
            "fabric-pressure",
            "fabric-ledger-invalid",
            "cognition-guarded",
            "voice-degraded",
            "awareness-degraded",
        ]
    ] = Field(max_length=8)


class AwarenessChannels(StrictContract):
    body: SourceFreshness
    environment: SourceFreshness
    camera: SourceFreshness
    microphone: SourceFreshness


class AwarenessFusionSources(StrictContract):
    body: SourceFreshness
    world: SourceFreshness
    cognition: SourceFreshness
    fabric: SourceFreshness
    dependencies: SourceFreshness


class DependencyServiceState(StrictContract):
    enabled: bool
    required: bool
    status: Literal["ready", "busy", "warming", "degraded", "unknown", "disabled"]
    fresh: bool
    age_ms: int | None = Field(default=None, ge=0, le=3_600_000)
    confidence: float = Field(ge=0.0, le=1.0)
    observed_at_ms: int | None = Field(default=None, ge=0)


class DependencyServices(StrictContract):
    cortex: DependencyServiceState
    n8n: DependencyServiceState
    voice: DependencyServiceState


class DependencyAwarenessState(StrictContract):
    status: Literal["nominal", "busy", "limited", "degraded"]
    confidence: float = Field(ge=0.0, le=1.0)
    degraded_count: int = Field(ge=0, le=3)
    services: DependencyServices


class AwarenessPublicState(StrictContract):
    fusion_version: Literal[1]
    status: Literal["nominal", "limited", "degraded", "no-data"]
    confidence: float = Field(ge=0.0, le=1.0)
    degraded_reasons: list[
        Literal[
            "headless-stale",
            "body-stale",
            "environment-stale",
            "camera-stale",
            "microphone-stale",
            "cognition-guarded",
            "fabric-pressure",
            "fabric-ledger-invalid",
            "dependency-degraded",
            "dependency-limited",
        ]
    ] = Field(max_length=10)
    body_revision: int = Field(ge=0)
    world_revision: int = Field(ge=0)
    awake: bool
    zone: Literal["center", "window", "kitchen", "gallery", "lounge"] | None
    visible_objects: int = Field(ge=0, le=20)
    channels: AwarenessChannels
    sources: AwarenessFusionSources
    dependencies: DependencyAwarenessState


class VoicePublicState(StrictContract):
    configured: bool
    status: Literal["ready", "warming", "degraded", "no-data"]
    prewarm_status: str = Field(min_length=1, max_length=32, pattern=r"^[a-z0-9_.:-]+$")
    slo_status: str = Field(min_length=1, max_length=32, pattern=r"^[a-z0-9_.:-]+$")
    sessions_started: int = Field(ge=0)
    reaction_target_ms: int = Field(ge=50, le=1_000)
    queue_target_ms: int = Field(ge=20, le=2_000)
    semantic_soft_target_ms: int = Field(ge=500, le=15_000)
    transport: "VoiceTransportPublicState"


class VoiceDeviceTelemetryPublicState(StrictContract):
    present: bool
    rtt_ms: float | None = Field(default=None, ge=0, le=30_000)
    packet_loss: float | None = Field(default=None, ge=0.0, le=1.0)
    capture_jitter_ms: float | None = Field(default=None, ge=0, le=2_000)
    audio_route: Literal["built-in", "wired", "bluetooth", "unknown"]
    thermal_state: Literal["nominal", "fair", "serious", "critical", "unknown"]
    low_power_mode: bool | None
    device_class: Literal["iphone", "iphone-16e", "ipad", "web", "unknown"]


class VoiceTransportPublicState(StrictContract):
    protocol_version: int = Field(ge=1, le=2)
    input_ack_sequence: int = Field(ge=0)
    output_ack_sequence: int = Field(ge=0)
    frames_received: int = Field(ge=0)
    frames_released: int = Field(ge=0)
    frames_duplicate: int = Field(ge=0)
    frames_lost: int = Field(ge=0)
    frames_rejected: int = Field(ge=0)
    buffer_depth: int = Field(ge=0, le=64)
    max_buffer_depth: int = Field(ge=0, le=64)
    jitter_ms: float = Field(ge=0, le=60_000)
    interruptions: int = Field(ge=0)
    reconnects: int = Field(ge=0)
    device: VoiceDeviceTelemetryPublicState


class CognitionPublicState(StrictContract):
    status: Literal["nominal", "guarded"]
    phase: Literal[
        "idle",
        "accepted",
        "queued",
        "thinking",
        "synthesizing",
        "completed",
        "cancelled",
        "failed",
    ]
    thought_count: int = Field(ge=0)
    dream_count: int = Field(ge=0)
    last_thought_at: datetime | None
    last_dream_at: datetime | None
    private_thought_available: bool
    private_dream_available: bool
    thought_topics: list[
        Literal["attention", "body", "environment", "latency", "memory", "privacy", "safety", "voice"]
    ] = Field(max_length=4)
    dream_topics: list[
        Literal["attention", "body", "environment", "latency", "memory", "privacy", "safety", "voice"]
    ] = Field(max_length=4)
    private_content_hidden: Literal[True] = True
    observations: int = Field(ge=0)
    intent_evaluations: int = Field(ge=0)


class FabricLaneCounters(StrictContract):
    submitted: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    deadlines: int = Field(ge=0)
    shed: int = Field(ge=0)
    cancelled: int = Field(ge=0)


class FabricLanePublicState(StrictContract):
    active: int = Field(ge=0)
    queued: int = Field(ge=0)
    counters: FabricLaneCounters


class FabricLanes(StrictContract):
    realtime: FabricLanePublicState
    interactive: FabricLanePublicState
    embodiment: FabricLanePublicState
    background: FabricLanePublicState


class FabricPublicState(StrictContract):
    status: Literal["nominal", "watch", "guarded"]
    pressure: float = Field(ge=0.0, le=1.0)
    ledger_valid: bool
    ledger_sequence: int = Field(ge=0)
    lanes: FabricLanes


class HeadlessPublicState(StrictContract):
    freshness: FreshnessState
    system: SystemPublicState
    awareness: AwarenessPublicState
    voice: VoicePublicState
    cognition: CognitionPublicState
    fabric: FabricPublicState


class HeadlessSnapshot(HeadlessPublicState):
    schema_version: Literal[2] = HEADLESS_SCHEMA_VERSION
    revision: int = Field(ge=1)
    generated_at: datetime


class HeadlessDelta(StrictContract):
    schema_version: Literal[2] = HEADLESS_SCHEMA_VERSION
    base_revision: int = Field(ge=0)
    revision: int = Field(ge=1)
    generated_at: datetime
    changes: dict[str, Any]

    @field_validator("changes")
    @classmethod
    def public_top_level_only(cls, value: dict[str, Any]) -> dict[str, Any]:
        allowed = {"freshness", "system", "awareness", "voice", "cognition", "fabric"}
        if not value or set(value) - allowed:
            raise ValueError("delta contains a non-public state field")
        return value


class SubscribeMessage(StrictContract):
    type: Literal["subscribe"]
    channels: list[Literal["state", "progress", "capsule_receipts"]] = Field(
        min_length=1,
        max_length=MAX_SUBSCRIPTIONS,
    )

    @field_validator("channels")
    @classmethod
    def unique_channels(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("subscriptions must be unique")
        return value


class ResumeMessage(StrictContract):
    type: Literal["resume"]
    revision: int = Field(ge=0)


class PingMessage(StrictContract):
    type: Literal["ping"]
    nonce: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")


class CapsuleSubmitMessage(StrictContract):
    type: Literal["capsule_submit"]
    capsule: SignedIntentCapsule


ClientMessage = Annotated[
    Union[SubscribeMessage, ResumeMessage, PingMessage, CapsuleSubmitMessage],
    Field(discriminator="type"),
]


class HelloMessage(StrictContract):
    type: Literal["hello"] = "hello"
    schema_version: Literal[2] = HEADLESS_SCHEMA_VERSION
    correlation_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    heartbeat_interval_ms: int = Field(ge=1_000, le=10_000)
    revision: int = Field(ge=0)


class SnapshotMessage(StrictContract):
    type: Literal["snapshot"] = "snapshot"
    snapshot: HeadlessSnapshot


class DeltaMessage(StrictContract):
    type: Literal["delta"] = "delta"
    delta: HeadlessDelta


class ProgressMessage(StrictContract):
    type: Literal["progress"] = "progress"
    turn_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    phase: Literal[
        "accepted",
        "queued",
        "thinking",
        "synthesizing",
        "completed",
        "cancelled",
        "failed",
    ]
    elapsed_ms: int = Field(ge=0, le=180_000)


class CapsuleReceiptMessage(StrictContract):
    type: Literal["capsule_receipt"] = "capsule_receipt"
    capsule_id: str = Field(pattern=r"^cap-[0-9a-f]{24}$")
    status: Literal["verified", "evaluated", "rejected"]
    decision: Literal["allow", "revise", "block", "not-evaluated"]
    correlation_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")


class HeartbeatMessage(StrictContract):
    type: Literal["heartbeat"] = "heartbeat"
    sent_at: datetime
    revision: int = Field(ge=0)


class PublicErrorMessage(StrictContract):
    type: Literal["error"] = "error"
    code: Literal[
        "authentication-required",
        "invalid-message",
        "capsule-invalid",
        "capsule-expired",
        "capsule-replayed",
        "capsule-blocked",
        "rate-limited",
        "state-resync-required",
        "service-unavailable",
    ]
    retryable: bool
    correlation_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")


class SessionBootstrapResponse(StrictContract):
    schema_version: Literal[2] = HEADLESS_SCHEMA_VERSION
    csrf_token: str = Field(min_length=32, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    expires_at: datetime
    expires_in_seconds: int = Field(ge=1, le=3_600)


class SessionRevokedResponse(StrictContract):
    revoked: Literal[True] = True


class PublicHTTPError(StrictContract):
    code: Literal[
        "authentication-required",
        "feature-disabled",
        "invalid-request",
        "rate-limited",
        "request-too-large",
        "state-unavailable",
    ]
    retryable: bool
    correlation_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")


class PublicHTTPErrorEnvelope(StrictContract):
    error: PublicHTTPError


HealthComponentName = Literal[
    "process",
    "fabric",
    "cognition",
    "state",
    "cortex",
    "bedrock",
    "n8n",
    "local-cortex",
    "voice",
    "memory",
    "codebase",
]
HealthReason = Literal[
    "startup-incomplete",
    "fabric-ledger-invalid",
    "fabric-pressure",
    "cognition-guarded",
    "state-unavailable",
    "state-stale",
    "cortex-unavailable",
    "bedrock-degraded",
    "n8n-degraded",
    "n8n-unobserved",
    "local-cortex-degraded",
    "voice-degraded",
    "voice-warming",
    "memory-degraded",
    "codebase-degraded",
]


class HealthComponent(StrictContract):
    enabled: bool
    required: bool
    status: Literal["ready", "busy", "warming", "degraded", "unknown", "disabled"]
    source: Literal["local", "control-plane", "active-probe"]
    reason: HealthReason | None = None
    latency_ms: float | None = Field(default=None, ge=0.0, le=10_000.0)
    checked_at: datetime


class HealthReport(StrictContract):
    schema_version: Literal[1] = 1
    kind: Literal["liveness", "readiness", "deep"]
    status: Literal["alive", "ready", "degraded", "not-ready"]
    ready: bool
    checked_at: datetime
    duration_ms: float = Field(ge=0.0, le=10_000.0)
    reasons: list[HealthReason] = Field(max_length=15)
    components: dict[HealthComponentName, HealthComponent] = Field(
        min_length=1,
        max_length=11,
    )

    @model_validator(mode="after")
    def internally_consistent(self) -> "HealthReport":
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("health reasons must be unique")
        if self.kind == "liveness":
            if self.status != "alive" or not self.ready or set(self.components) != {"process"}:
                raise ValueError("liveness must contain only the live process check")
        elif self.ready and self.status not in {"ready", "degraded"}:
            raise ValueError("a ready report cannot be not-ready")
        elif not self.ready and self.status != "not-ready":
            raise ValueError("an unavailable report must be not-ready")
        return self


class GoldenMetric(StrictContract):
    operation: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_.:-]+$")
    requests: int = Field(ge=0)
    successes: int = Field(ge=0)
    server_errors: int = Field(ge=0)
    client_errors: int = Field(ge=0)
    cancelled: int = Field(ge=0)
    in_flight: int = Field(ge=0, le=100_000)
    max_in_flight: int = Field(ge=0, le=100_000)
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    duration_p50_ms: float | None = Field(default=None, ge=0.0, le=600_000.0)
    duration_p95_ms: float | None = Field(default=None, ge=0.0, le=600_000.0)


class ErrorBudgetView(StrictContract):
    operation: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_.:-]+$")
    objective: float = Field(ge=0.5, le=0.9999)
    latency_target_ms: float | None = Field(default=None, ge=1.0, le=600_000.0)
    samples: int = Field(ge=0, le=4_096)
    good: int = Field(ge=0, le=4_096)
    bad: int = Field(ge=0, le=4_096)
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    error_budget_remaining_pct: float = Field(ge=0.0, le=100.0)
    burn_rate: float = Field(ge=0.0, le=10_000.0)
    status: Literal["no-data", "healthy", "watch", "exhausted"]


class BoundedTrace(StrictContract):
    trace_id: str = Field(pattern=r"^trc-[0-9a-f]{24}$")
    correlation_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    operation: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_.:-]+$")
    started_at: datetime
    duration_ms: float = Field(ge=0.0, le=600_000.0)
    outcome: Literal["success", "client-error", "server-error", "cancelled", "rejected"]
    result_code: int | None = Field(default=None, ge=0, le=4_999)
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict, max_length=8)

    @field_validator("attributes")
    @classmethod
    def categorical_attributes_only(
        cls,
        value: dict[str, str | int | float | bool],
    ) -> dict[str, str | int | float | bool]:
        allowed = {
            "method", "route", "phase", "reason_code", "protocol", "revision",
            "lane", "speaker", "frames", "bytes", "queue_depth", "retryable",
        }
        if set(value) - allowed:
            raise ValueError("trace attributes are outside the telemetry allowlist")
        for item in value.values():
            if isinstance(item, str) and (
                len(item) > 96
                or not re.fullmatch(r"[A-Za-z0-9_./:{}-]+", item)
            ):
                raise ValueError("trace attributes must be bounded categories")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("trace numeric attributes must be finite")
        return value


class VoiceSLOView(StrictContract):
    status: Literal["no-data", "nominal", "watch", "breached"]
    samples: int = Field(ge=0, le=4_096)
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    error_budget_remaining_pct: float = Field(ge=0.0, le=100.0)
    reaction_target_ms: int = Field(ge=50, le=1_000)
    queue_target_ms: int = Field(ge=20, le=2_000)
    semantic_target_ms: int = Field(ge=500, le=15_000)
    reaction_p95_ms: float | None = Field(default=None, ge=0.0, le=600_000.0)
    queue_p95_ms: float | None = Field(default=None, ge=0.0, le=600_000.0)
    semantic_p95_ms: float | None = Field(default=None, ge=0.0, le=600_000.0)


class ObservabilityReport(StrictContract):
    schema_version: Literal[1] = 1
    generated_at: datetime
    retention_traces: int = Field(ge=16, le=1_024)
    retention_samples_per_operation: int = Field(ge=16, le=4_096)
    total_requests: int = Field(ge=0)
    total_server_errors: int = Field(ge=0)
    current_in_flight: int = Field(ge=0, le=100_000)
    metrics: list[GoldenMetric] = Field(max_length=64)
    error_budgets: list[ErrorBudgetView] = Field(max_length=16)
    voice_slo: VoiceSLOView
    recent_traces: list[BoundedTrace] = Field(max_length=32)


class N8NCognitionContext(StrictContract):
    awareness_confidence: float = Field(ge=0.0, le=1.0)
    fabric_pressure: float = Field(ge=0.0, le=1.0)
    immune_status: Literal["nominal", "guarded"]
    open_components: list[str] = Field(max_length=8)

    @field_validator("open_components")
    @classmethod
    def bounded_components(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(
            not re.fullmatch(r"[a-z0-9_.:-]{1,64}", item) for item in value
        ):
            raise ValueError("n8n open components must be unique safe identifiers")
        return value


class N8NHeadlessRequest(StrictContract):
    contract_version: Literal["weaver-headless-n8n-v1"] = N8N_CONTRACT_VERSION
    correlation_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    deadline_ms: Literal[115_000] = 115_000
    text: str = Field(min_length=1, max_length=4_000)
    self_check: bool
    introspect: bool
    path_glob: str = Field(default="**/*", min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_./*?{}-]+$")
    search_query: str = Field(max_length=240)
    codebase_context: str = Field(max_length=12_000)
    quantum_pathway: str = Field(max_length=500)
    cognition_context: N8NCognitionContext

    @field_validator("text", "search_query", "codebase_context", "quantum_pathway")
    @classmethod
    def bounded_contract_text(cls, value: str) -> str:
        if any(ord(character) < 32 and character not in "\n\r\t" for character in value):
            raise ValueError("n8n contract text contains a control character")
        return value


def _parse_n8n_timestamp(value: Any) -> Any:
    """Normalize the workflow's JSON timestamp without relaxing strict models."""
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    return value


class N8NPublicSuccess(StrictContract):
    contract_version: Literal["weaver-headless-n8n-v1"] = N8N_CONTRACT_VERSION
    status: Literal["ok"]
    error: Literal[False]
    correlation_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    manifested_response: str = Field(min_length=1, max_length=12_000)
    speaker: Literal["weaver"]
    speaker_boundary_applied: Literal[True]
    speaker_model: Literal["qwen.qwen3-235b-a22b-2507"]
    internal_draft_hidden: Literal[True]
    reflection_applied: Literal[True]
    soul_voice_active: bool
    codebase_grounded: bool
    expert_parallel: Literal[True]
    expert_count: Literal[5]
    experts_completed: int = Field(ge=0, le=5)
    expert_errors: int = Field(ge=0, le=5)
    expert_fanout_elapsed_ms: int = Field(ge=0, le=115_000)
    execution_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.:-]+$")
    timestamp: datetime
    pipeline_architecture: Literal["parallel-fanout-barrier"]
    pipeline_version: Literal["v6-parallel-cognition"]

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, value: Any) -> Any:
        return _parse_n8n_timestamp(value)

    @field_validator("manifested_response")
    @classmethod
    def bounded_manifestation(cls, value: str) -> str:
        text = value.strip()
        if not text or any(
            ord(character) < 32 and character not in "\n\r\t" for character in text
        ):
            raise ValueError("n8n manifestation is empty or contains control characters")
        return text


class N8NPublicRejection(StrictContract):
    contract_version: Literal["weaver-headless-n8n-v1"] = N8N_CONTRACT_VERSION
    status: Literal["rejected"]
    error: Literal[True]
    error_code: Literal["invalid-request", "speaker-boundary-failed", "pipeline-unavailable"]
    correlation_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    execution_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.:-]+$")
    timestamp: datetime
    pipeline_version: Literal["v6-parallel-cognition"]

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, value: Any) -> Any:
        return _parse_n8n_timestamp(value)


N8NPublicResponse = Annotated[
    Union[N8NPublicSuccess, N8NPublicRejection],
    Field(discriminator="status"),
]


class MemoryLifecyclePublicState(StrictContract):
    status: Literal["connected"]
    version: Literal[1]
    active_records: int = Field(ge=0, le=16_384)
    deleted_records: int = Field(ge=0, le=16_384)
    max_records: int = Field(ge=64, le=16_384)
    duplicates_consolidated: int = Field(ge=0)
    fresh_records: int = Field(ge=0)
    stale_records: int = Field(ge=0)
    freshness_score: float = Field(ge=0.0, le=1.0)
    retention_due: int = Field(ge=0)
    deletion_audit_events: int = Field(ge=0)


class MemoryDeletionResponse(StrictContract):
    memory_id: str = Field(pattern=r"^mem-[0-9a-f]{24}$")
    deleted: Literal[True]
    already_deleted: bool
    audit_id: str = Field(pattern=r"^del-[0-9a-f]{24}$")
    storage_records_removed: int = Field(ge=0)
    storage_complete: bool


def _bounded_chat_text(value: str, *, limit: int) -> str:
    text = value.strip()
    if (
        not text
        or len(text) > limit
        or any(ord(character) < 32 and character not in "\n\r\t" for character in text)
    ):
        raise ValueError("chat text is empty, oversized, or contains control characters")
    return text


class HeadlessChatHistoryMessage(StrictContract):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def bounded_content(cls, value: str) -> str:
        return _bounded_chat_text(value, limit=8_000)


class HeadlessChatRequest(StrictContract):
    message: str
    history: list[HeadlessChatHistoryMessage] = Field(default_factory=list, max_length=20)
    client_turn_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )
    max_tokens: int = Field(default=512, ge=32, le=1_024)

    @field_validator("message")
    @classmethod
    def bounded_message(cls, value: str) -> str:
        return _bounded_chat_text(value, limit=12_000)

    @model_validator(mode="after")
    def bounded_total_context(self) -> "HeadlessChatRequest":
        total = len(self.message) + sum(len(item.content) for item in self.history)
        if total > 24_000:
            raise ValueError("chat context exceeds its hard character budget")
        return self


class HeadlessChatCancelledResponse(StrictContract):
    turn_id: str = Field(pattern=r"^turn-[0-9a-f]{24}$")
    cancelled: Literal[True]


class HeadlessVoiceSynthesisRequest(StrictContract):
    text: str

    @field_validator("text")
    @classmethod
    def bounded_text(cls, value: str) -> str:
        return _bounded_chat_text(value, limit=800)


ServerMessage = Annotated[
    Union[
        HelloMessage,
        SnapshotMessage,
        DeltaMessage,
        ProgressMessage,
        CapsuleReceiptMessage,
        HeartbeatMessage,
        PublicErrorMessage,
    ],
    Field(discriminator="type"),
]


# These assertions protect the duplicated Literal declarations above from
# drifting away from the compiler's runtime allowlists.
assert set(NavigateAction.model_fields["zone"].annotation.__args__) == PENTHOUSE_ZONES
assert set(InteractAction.model_fields["interaction"].annotation.__args__) == PENTHOUSE_INTERACTIONS
assert set(ObserveAction.model_fields["sensor"].annotation.__args__) == SENSOR_CAPABILITIES
assert set(RememberAction.model_fields["category"].annotation.__args__) == MEMORY_CATEGORIES
