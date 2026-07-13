#!/usr/bin/env python3
"""Weaver Seven-Angle Cognition Mesh.

This module adds seven deterministic control-plane capabilities above the
Neural Fabric and below model-generated behavior:

1. perception: freshness-aware body/environment sensor fusion;
2. embodiment: a reflex kernel that checks signed intent preconditions;
3. prediction: a counterfactual digital twin that scores plans before action;
4. compute: an adaptive, deadline-aware model and accelerator governor;
5. memory: a metadata-only, multi-timescale salience pyramid;
6. resilience: circuit breakers and anomaly-aware component immunity;
7. evolution: advisory-only shadow policies that can never self-deploy.

The Mesh accepts no raw image/audio payloads, executable code, arbitrary URLs,
commands, secrets, or unbounded text.  It does not execute Intent Capsules.  It
only observes bounded state, evaluates already-signed capsules, recommends
routes, records scalar outcomes, and emits shadow proposals.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from weaver_neural_fabric import (
    PENTHOUSE_INTERACTIONS,
    PENTHOUSE_ZONES,
    POSE_SLOTS,
)


class CognitionValidationError(ValueError):
    """Raised when a bounded Cognition Mesh schema is violated."""


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def _number(value: Any, low: float, high: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CognitionValidationError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise CognitionValidationError(f"{field} is out of range")
    return round(number, 4)


def _integer(value: Any, low: int, high: int, *, field: str) -> int:
    number = _number(value, low, high, field=field)
    if not number.is_integer():
        raise CognitionValidationError(f"{field} must be an integer")
    return int(number)


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise CognitionValidationError(f"{field} must be boolean")
    return value


def _identifier(value: Any, *, field: str, limit: int = 64) -> str:
    text = str(value or "").strip().lower()
    if not text or len(text) > limit or not all(
        character.isalnum() or character in "_.:-" for character in text
    ):
        raise CognitionValidationError(f"{field} is invalid")
    return text


def _strict_object(
    value: Any,
    *,
    field: str,
    allowed: set[str],
    allow_empty: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, dict) or (not allow_empty and not value):
        raise CognitionValidationError(f"{field} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise CognitionValidationError(f"{field} contains unknown fields")
    return value


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    position = (len(clean) - 1) * _clamp(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(clean[lower], 1)
    weight = position - lower
    return round(clean[lower] * (1 - weight) + clean[upper] * weight, 1)


def _ewma(previous: float | None, value: float, alpha: float = 0.22) -> float:
    return float(value) if previous is None else alpha * float(value) + (1 - alpha) * previous


SENSOR_CHANNELS = {"body", "environment", "camera", "microphone"}
CHANNEL_TTL_MS = {
    "body": 1_200,
    "environment": 4_000,
    "camera": 1_800,
    "microphone": 800,
}
GROUND_CONTACTS = {"left_foot", "right_foot"}
TASKS = {"voice", "embodiment", "chat", "code", "vision", "dream"}
MEMORY_KINDS = {"observation", "route", "outcome", "safety", "interaction"}
MEMORY_TAGS = {
    "voice", "body", "environment", "camera", "microphone", "chat", "code",
    "vision", "dream", "model", "latency", "quality", "safety", "navigation",
    "interaction", "memory", "n8n", "user",
}
COMPONENTS = {
    "weaver-speed", "weaver-fast-aws", "weaver-brain", "weaver-code",
    "weaver-vision", "weaver-headless", "local-lora", "n8n", "voice",
    "embodiment", "memory", "sensors",
}


class SituationalGraph:
    """Fuses bounded scalar observations into a freshness-aware world/body graph."""

    def __init__(self) -> None:
        self.body_revision = 0
        self.world_revision = 0
        self.last_observed_at_ms = 0
        self.body: dict[str, Any] = {
            "awake": True,
            "balance": 1.0,
            "velocity_mps": 0.0,
            "ground_contacts": ["left_foot", "right_foot"],
            "pose": {},
        }
        self.world: dict[str, Any] = {
            "zone": "center",
            "ambient_light": 0.5,
            "noise": 0.0,
            "obstacle_distance_m": 10.0,
            "objects": {},
        }
        self.channels: dict[str, dict[str, Any]] = {}
        self.observations = 0

    def _record_channel(self, channel: str, confidence: float, observed_at_ms: int) -> None:
        self.channels[channel] = {
            "reported_confidence": round(confidence, 4),
            "observed_at_ms": observed_at_ms,
        }
    def observe(self, payload: Any, *, now_ms: int | None = None) -> dict[str, Any]:
        data = _strict_object(
            payload,
            field="observation",
            allowed={"observed_at_ms", "body", "environment", "sensors"},
            allow_empty=False,
        )
        current_ms = int(now_ms if now_ms is not None else time.time() * 1000)
        observed_at_ms = _integer(
            data.get("observed_at_ms", current_ms),
            0,
            9_999_999_999_999,
            field="observed_at_ms",
        )
        if observed_at_ms > current_ms + 5_000 or observed_at_ms < current_ms - 300_000:
            raise CognitionValidationError("observation timestamp is outside the accepted window")
        if self.last_observed_at_ms and observed_at_ms < self.last_observed_at_ms - 2_000:
            raise CognitionValidationError("observation is older than the fused state")

        changed = False
        if "body" in data:
            body = _strict_object(
                data["body"],
                field="body",
                allowed={
                    "awake", "balance", "velocity_mps", "ground_contacts", "pose", "confidence",
                },
                allow_empty=False,
            )
            if "awake" in body:
                self.body["awake"] = _boolean(body["awake"], field="body.awake")
            if "balance" in body:
                self.body["balance"] = _number(body["balance"], 0, 1, field="body.balance")
            if "velocity_mps" in body:
                self.body["velocity_mps"] = _number(
                    body["velocity_mps"], 0, 8, field="body.velocity_mps"
                )
            if "ground_contacts" in body:
                contacts = body["ground_contacts"]
                if not isinstance(contacts, list) or len(contacts) > 2:
                    raise CognitionValidationError("body.ground_contacts must contain at most two items")
                normalized = []
                for contact in contacts:
                    item = str(contact or "").strip().lower()
                    if item not in GROUND_CONTACTS or item in normalized:
                        raise CognitionValidationError("body.ground_contacts contains an invalid item")
                    normalized.append(item)
                self.body["ground_contacts"] = normalized
            if "pose" in body:
                pose = _strict_object(
                    body["pose"], field="body.pose", allowed=set(POSE_SLOTS), allow_empty=True
                )
                if len(pose) > len(POSE_SLOTS):
                    raise CognitionValidationError("body.pose is too large")
                self.body["pose"].update({
                    key: _number(value, -1, 1, field=f"body.pose.{key}")
                    for key, value in pose.items()
                })
            confidence = _number(body.get("confidence", 0.9), 0, 1, field="body.confidence")
            self._record_channel("body", confidence, observed_at_ms)
            self.body_revision += 1
            changed = True

        if "environment" in data:
            environment = _strict_object(
                data["environment"],
                field="environment",
                allowed={
                    "zone", "ambient_light", "noise", "obstacle_distance_m", "objects", "confidence",
                },
                allow_empty=False,
            )
            if "zone" in environment:
                zone = str(environment["zone"] or "").strip().lower()
                if zone not in PENTHOUSE_ZONES:
                    raise CognitionValidationError("environment.zone is invalid")
                self.world["zone"] = zone
            if "ambient_light" in environment:
                self.world["ambient_light"] = _number(
                    environment["ambient_light"], 0, 1, field="environment.ambient_light"
                )
            if "noise" in environment:
                self.world["noise"] = _number(environment["noise"], 0, 1, field="environment.noise")
            if "obstacle_distance_m" in environment:
                self.world["obstacle_distance_m"] = _number(
                    environment["obstacle_distance_m"],
                    0,
                    50,
                    field="environment.obstacle_distance_m",
                )
            if "objects" in environment:
                objects = environment["objects"]
                if not isinstance(objects, list) or len(objects) > len(PENTHOUSE_INTERACTIONS):
                    raise CognitionValidationError("environment.objects is too large")
                compiled: dict[str, Any] = {}
                for index, raw in enumerate(objects):
                    item = _strict_object(
                        raw,
                        field=f"environment.objects[{index}]",
                        allowed={"id", "zone", "distance_m", "visible", "confidence"},
                        allow_empty=False,
                    )
                    object_id = str(item.get("id") or "").strip().lower()
                    if object_id not in PENTHOUSE_INTERACTIONS or object_id in compiled:
                        raise CognitionValidationError("environment object id is invalid or duplicated")
                    zone = str(item.get("zone", self.world["zone"]) or "").strip().lower()
                    if zone not in PENTHOUSE_ZONES:
                        raise CognitionValidationError("environment object zone is invalid")
                    compiled[object_id] = {
                        "zone": zone,
                        "distance_m": _number(
                            item.get("distance_m", 10), 0, 50, field="environment.object.distance_m"
                        ),
                        "visible": _boolean(item.get("visible", True), field="environment.object.visible"),
                        "confidence": _number(
                            item.get("confidence", 0.8), 0, 1, field="environment.object.confidence"
                        ),
                    }
                self.world["objects"] = compiled
            confidence = _number(
                environment.get("confidence", 0.85), 0, 1, field="environment.confidence"
            )
            self._record_channel("environment", confidence, observed_at_ms)
            self.world_revision += 1
            changed = True

        if "sensors" in data:
            sensors = _strict_object(
                data["sensors"], field="sensors", allowed=set(SENSOR_CHANNELS), allow_empty=False
            )
            for channel, raw in sensors.items():
                sensor = _strict_object(
                    raw,
                    field=f"sensors.{channel}",
                    allowed={"confidence", "sample_age_ms"},
                    allow_empty=True,
                )
                confidence = _number(
                    sensor.get("confidence", 1.0), 0, 1, field=f"sensors.{channel}.confidence"
                )
                sample_age_ms = _integer(
                    sensor.get("sample_age_ms", 0),
                    0,
                    CHANNEL_TTL_MS[channel] * 4,
                    field=f"sensors.{channel}.sample_age_ms",
                )
                self._record_channel(channel, confidence, observed_at_ms - sample_age_ms)
            changed = True

        if not changed:
            raise CognitionValidationError("observation contains no usable state")
        self.observations += 1
        self.last_observed_at_ms = max(self.last_observed_at_ms, observed_at_ms)
        return self.snapshot(now_ms=current_ms)

    def snapshot(self, *, now_ms: int | None = None) -> dict[str, Any]:
        current_ms = int(now_ms if now_ms is not None else time.time() * 1000)
        channel_state: dict[str, Any] = {}
        confidences: list[float] = []
        for channel in sorted(SENSOR_CHANNELS):
            stored = self.channels.get(channel)
            if not stored:
                channel_state[channel] = {"fresh": False, "age_ms": None, "confidence": 0.0}
                continue
            age_ms = max(0, current_ms - int(stored["observed_at_ms"]))
            freshness = _clamp(1 - age_ms / CHANNEL_TTL_MS[channel])
            confidence = round(float(stored["reported_confidence"]) * freshness, 4)
            channel_state[channel] = {
                "fresh": age_ms <= CHANNEL_TTL_MS[channel],
                "age_ms": age_ms,
                "confidence": confidence,
            }
            if confidence > 0:
                confidences.append(confidence)
        return {
            "body_revision": self.body_revision,
            "world_revision": self.world_revision,
            "last_observed_at_ms": self.last_observed_at_ms or None,
            "observations": self.observations,
            "awareness_confidence": round(sum(confidences) / len(confidences), 4)
            if confidences else 0.0,
            "body": {
                **self.body,
                "pose": dict(self.body["pose"]),
                "ground_contacts": list(self.body["ground_contacts"]),
            },
            "world": {
                **self.world,
                "objects": {key: dict(value) for key, value in self.world["objects"].items()},
            },
            "channels": channel_state,
        }


class ReflexKernel:
    """Deterministic safety/interlock layer for signed embodiment intents."""

    @staticmethod
    def evaluate(capsule: dict[str, Any], awareness: dict[str, Any]) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []

        def check(name: str, passed: bool, severity: str, detail: str) -> None:
            checks.append({"name": name, "passed": bool(passed), "severity": severity, "detail": detail})

        preconditions = capsule.get("preconditions") or {}
        body = awareness["body"]
        world = awareness["world"]
        requested_body = int(preconditions.get("body_revision", 0) or 0)
        requested_world = int(preconditions.get("world_revision", 0) or 0)
        check(
            "body_revision",
            requested_body in {0, awareness["body_revision"]},
            "block",
            "body state must match the compiled intent",
        )
        check(
            "world_revision",
            requested_world in {0, awareness["world_revision"]},
            "block",
            "world state must match the compiled intent",
        )
        requires_awake = bool(preconditions.get("requires_awake", True))
        check("awake", not requires_awake or bool(body["awake"]), "block", "body must be awake")

        action_types = {str(action.get("type") or "") for action in capsule.get("actions", [])}
        needs_body = bool(action_types & {"pose", "bones", "navigate", "interact"})
        needs_world = bool(action_types & {"navigate", "interact"})
        check(
            "body_freshness",
            not needs_body or awareness["channels"]["body"]["confidence"] >= 0.25,
            "revise",
            "refresh body telemetry before motion",
        )
        check(
            "world_freshness",
            not needs_world or awareness["channels"]["environment"]["confidence"] >= 0.25,
            "revise",
            "refresh environment telemetry before movement",
        )

        current_pose = body.get("pose", {})
        largest_delta = 0.0
        largest_bone_rotation = 0.0
        interactions: list[str] = []
        for action in capsule.get("actions", []):
            action_type = action.get("type")
            if action_type == "pose":
                for slot, target in (action.get("values") or {}).items():
                    largest_delta = max(largest_delta, abs(float(target) - float(current_pose.get(slot, 0))))
            elif action_type == "bones":
                for rotation in (action.get("bones") or {}).values():
                    largest_bone_rotation = max(
                        largest_bone_rotation,
                        *(abs(float(rotation.get(axis, 0))) for axis in ("x", "y", "z")),
                    )
            elif action_type == "interact":
                interactions.append(str(action.get("interaction") or ""))

        check(
            "pose_delta",
            largest_delta <= 0.7,
            "revise",
            "segment large pose changes into smaller trajectories",
        )
        check(
            "bone_delta",
            largest_bone_rotation <= 0.9,
            "revise",
            "segment large bone rotations into smaller trajectories",
        )
        if "navigate" in action_types:
            check("balance", float(body["balance"]) >= 0.35, "block", "balance is below locomotion floor")
            check(
                "collision_clearance",
                float(world["obstacle_distance_m"]) >= 0.45,
                "block",
                "navigation clearance is below the hard safety floor",
            )
            check(
                "ground_contact",
                bool(body.get("ground_contacts")),
                "block",
                "locomotion requires at least one ground contact",
            )
        for interaction in interactions:
            target = world.get("objects", {}).get(interaction)
            check(
                f"target_{interaction}",
                bool(
                    target
                    and target.get("visible")
                    and float(target.get("confidence", 0)) >= 0.35
                    and float(target.get("distance_m", 99)) <= 2.5
                ),
                "revise",
                "observe or approach the interaction target before contact",
            )

        failed = [item for item in checks if not item["passed"]]
        decision = "approve"
        if any(item["severity"] == "block" for item in failed):
            decision = "block"
        elif failed:
            decision = "revise"
        return {
            "technology": "weaver-reflex-kernel",
            "decision": decision,
            "approved": decision == "approve",
            "checks": checks,
            "failed_checks": [item["name"] for item in failed],
            "largest_pose_delta": round(largest_delta, 4),
            "largest_bone_rotation": round(largest_bone_rotation, 4),
        }


class CounterfactualTwin:
    """Fast scalar digital twin for plan risk, energy, and outcome estimates."""

    @staticmethod
    def simulate(
        capsule: dict[str, Any], awareness: dict[str, Any], reflex: dict[str, Any]
    ) -> dict[str, Any]:
        body = awareness["body"]
        world = awareness["world"]
        energy = 0.0
        duration_ms = 0
        lower_body_motion = 0.0
        destination = world["zone"]
        interventions: list[str] = []
        for action in capsule.get("actions", []):
            action_type = action.get("type")
            if action_type == "navigate":
                energy += 0.28
                duration_ms += 2_500
                destination = action.get("zone", destination)
            elif action_type == "interact":
                energy += 0.16
                duration_ms += 1_500
            elif action_type == "pose":
                values = action.get("values") or {}
                magnitude = sum(abs(float(value)) for value in values.values()) / max(len(values), 1)
                energy += 0.06 + magnitude * 0.12
                duration_ms += 550
                lower_body_motion = max(
                    lower_body_motion,
                    *(abs(float(values.get(slot, 0))) for slot in ("leftKnee", "rightKnee", "leftAnkle", "rightAnkle")),
                )
            elif action_type == "bones":
                rotations = [
                    abs(float(value))
                    for rotation in (action.get("bones") or {}).values()
                    for value in rotation.values()
                ]
                energy += 0.05 + (sum(rotations) / max(len(rotations), 1)) * 0.1
                duration_ms += 450
            elif action_type == "speak":
                energy += 0.03
                duration_ms += min(8_000, max(400, len(str(action.get("text") or "")) * 38))
            elif action_type == "observe":
                energy += 0.01
                duration_ms += 250
            elif action_type == "remember":
                energy += 0.01
                duration_ms += 100

        clearance = float(world["obstacle_distance_m"])
        collision_risk = _clamp(1 - clearance / 2.0) if destination != world["zone"] else 0.0
        stability = _clamp(
            float(body["balance"])
            - lower_body_motion * 0.28
            - min(float(body["velocity_mps"]) / 8, 1) * 0.12
        )
        confidence = float(awareness["awareness_confidence"])
        reflex_factor = {"approve": 1.0, "revise": 0.68, "block": 0.0}[reflex["decision"]]
        success_probability = _clamp(
            (0.35 + confidence * 0.65) * stability * (1 - collision_risk) * reflex_factor
        )
        max_duration = int((capsule.get("preconditions") or {}).get("max_duration_ms", 30_000))
        if duration_ms > max_duration:
            interventions.append("split_plan_to_fit_duration_budget")
        if reflex["largest_pose_delta"] > 0.7 or reflex["largest_bone_rotation"] > 0.9:
            interventions.append("segment_motion_trajectory")
        if awareness["channels"]["environment"]["confidence"] < 0.25:
            interventions.append("refresh_environment_observation")
        if collision_risk > 0.35:
            interventions.append("replan_for_clearance")

        recommendation = "execute"
        if reflex["decision"] == "block":
            recommendation = "block"
        elif reflex["decision"] == "revise" or duration_ms > max_duration or success_probability < 0.55:
            recommendation = "revise"
        return {
            "technology": "weaver-counterfactual-twin",
            "recommendation": recommendation,
            "predicted_zone": destination,
            "estimated_duration_ms": duration_ms,
            "energy_cost": round(_clamp(energy), 4),
            "stability": round(stability, 4),
            "collision_risk": round(collision_risk, 4),
            "success_probability": round(success_probability, 4),
            "interventions": sorted(set(interventions)),
        }


@dataclass(frozen=True)
class ModelProfile:
    alias: str
    capabilities: tuple[str, ...]
    base_latency_ms: int
    base_quality: float
    relative_cost: float
    accelerator_units: int


MODEL_PROFILES = (
    ModelProfile("weaver-speed", ("voice", "embodiment", "chat"), 140, 0.66, 0.12, 2),
    ModelProfile("weaver-fast-aws", ("voice", "embodiment", "chat"), 220, 0.73, 0.24, 3),
    ModelProfile("weaver-brain", ("chat",), 900, 0.92, 0.82, 6),
    ModelProfile("weaver-code", ("code",), 1_000, 0.94, 0.88, 7),
    ModelProfile("weaver-vision", ("vision", "chat"), 1_200, 0.90, 0.94, 7),
    ModelProfile("weaver-headless", ("dream", "chat"), 800, 0.84, 0.62, 5),
    ModelProfile("local-lora", ("voice", "embodiment", "chat", "code"), 350, 0.58, 0.04, 2),
)


class ResilienceImmuneSystem:
    """Per-component circuit breakers with bounded anomaly telemetry."""

    def __init__(self, *, failure_threshold: int = 3, cooldown_seconds: float = 30) -> None:
        self.failure_threshold = min(max(int(failure_threshold), 2), 10)
        self.cooldown_seconds = min(max(float(cooldown_seconds), 5), 300)
        self.components: dict[str, dict[str, Any]] = {}

    def _component(self, name: str) -> dict[str, Any]:
        return self.components.setdefault(name, {
            "state": "closed",
            "failure_streak": 0,
            "successes": 0,
            "failures": 0,
            "anomalies": 0,
            "opened_until": 0.0,
            "latency_ewma_ms": None,
            "latencies": deque(maxlen=64),
        })

    def state_for(self, name: str, *, now: float | None = None) -> str:
        item = self._component(name)
        current = float(now if now is not None else time.monotonic())
        if item["state"] == "open" and current >= item["opened_until"]:
            item["state"] = "half-open"
        return str(item["state"])

    def allow(self, name: str) -> bool:
        return self.state_for(name) != "open"

    def record(self, name: str, *, success: bool, latency_ms: float, target_ms: float) -> dict[str, Any]:
        item = self._component(name)
        latency = float(latency_ms)
        item["latencies"].append(latency)
        item["latency_ewma_ms"] = _ewma(item["latency_ewma_ms"], latency)
        if latency > max(float(target_ms) * 2, float(target_ms) + 250):
            item["anomalies"] += 1
        current_state = self.state_for(name)
        if success:
            item["successes"] += 1
            item["failure_streak"] = 0
            if current_state == "half-open":
                item["state"] = "closed"
        else:
            item["failures"] += 1
            item["failure_streak"] += 1
            if current_state == "half-open" or item["failure_streak"] >= self.failure_threshold:
                item["state"] = "open"
                item["opened_until"] = time.monotonic() + self.cooldown_seconds
        return self.component_snapshot(name)

    def component_snapshot(self, name: str) -> dict[str, Any]:
        item = self._component(name)
        state = self.state_for(name)
        return {
            "state": state,
            "failure_streak": item["failure_streak"],
            "successes": item["successes"],
            "failures": item["failures"],
            "anomalies": item["anomalies"],
            "latency_ewma_ms": round(item["latency_ewma_ms"], 1)
            if item["latency_ewma_ms"] is not None else None,
            "latency_p95_ms": _percentile(item["latencies"], 0.95),
        }

    def snapshot(self) -> dict[str, Any]:
        components = {name: self.component_snapshot(name) for name in sorted(self.components)}
        return {
            "technology": "weaver-immune-system",
            "status": "guarded" if any(item["state"] == "open" for item in components.values()) else "nominal",
            "failure_threshold": self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "open_components": [name for name, item in components.items() if item["state"] == "open"],
            "components": components,
        }


class InferenceGovernor:
    """Deadline, quality, pressure, cost, and health-aware model governor."""

    def __init__(self) -> None:
        self.profiles = {profile.alias: profile for profile in MODEL_PROFILES}
        self.telemetry: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"samples": 0, "latency_ewma_ms": None, "quality_ewma": None, "success_ewma": None}
        )
        self.decisions = 0

    def record(self, component: str, *, success: bool, latency_ms: float, quality: float) -> None:
        if component not in self.profiles:
            return
        item = self.telemetry[component]
        item["samples"] += 1
        item["latency_ewma_ms"] = _ewma(item["latency_ewma_ms"], latency_ms)
        item["quality_ewma"] = _ewma(item["quality_ewma"], quality)
        item["success_ewma"] = _ewma(item["success_ewma"], 1.0 if success else 0.0)

    def plan(
        self,
        *,
        task: str,
        deadline_ms: int,
        quality_priority: float,
        fabric_pressure: float,
        immune: ResilienceImmuneSystem,
    ) -> dict[str, Any]:
        scored: list[dict[str, Any]] = []
        for profile in self.profiles.values():
            if task not in profile.capabilities or not immune.allow(profile.alias):
                continue
            telemetry = self.telemetry.get(profile.alias, {})
            latency = float(telemetry.get("latency_ewma_ms") or profile.base_latency_ms)
            quality = float(telemetry.get("quality_ewma") or profile.base_quality)
            success = float(telemetry.get("success_ewma") if telemetry.get("success_ewma") is not None else 1.0)
            deadline_fit = _clamp(deadline_ms / max(latency, 1))
            pressure_penalty = fabric_pressure * profile.accelerator_units / 8
            task_boost = 0.0
            if (task, profile.alias) in {
                ("voice", "weaver-speed"), ("embodiment", "weaver-speed"),
                ("chat", "weaver-brain"), ("code", "weaver-code"),
                ("vision", "weaver-vision"), ("dream", "weaver-headless"),
            }:
                task_boost = 0.16
            score = (
                quality * (0.35 + quality_priority * 0.45)
                + deadline_fit * (0.55 - quality_priority * 0.25)
                + success * 0.18
                + task_boost
                - profile.relative_cost * 0.12
                - pressure_penalty * 0.28
            )
            scored.append({
                "alias": profile.alias,
                "score": round(score, 5),
                "estimated_latency_ms": round(latency),
                "estimated_quality": round(quality, 4),
                "accelerator_units": profile.accelerator_units,
                "deadline_fit": round(deadline_fit, 4),
            })
        scored.sort(key=lambda item: (-item["score"], item["estimated_latency_ms"], item["alias"]))
        if not scored:
            raise CognitionValidationError("no healthy inference route supports this task")
        self.decisions += 1
        primary = scored[0]
        return {
            "technology": "weaver-inference-governor",
            "task": task,
            "deadline_ms": deadline_ms,
            "fabric_pressure": round(fabric_pressure, 4),
            "primary": primary,
            "fallbacks": scored[1:3],
            "candidate_count": len(scored),
            "advisory": True,
        }

    def snapshot(self) -> dict[str, Any]:
        telemetry = {}
        for alias, item in sorted(self.telemetry.items()):
            telemetry[alias] = {
                "samples": item["samples"],
                "latency_ewma_ms": round(item["latency_ewma_ms"], 1)
                if item["latency_ewma_ms"] is not None else None,
                "quality_ewma": round(item["quality_ewma"], 4)
                if item["quality_ewma"] is not None else None,
                "success_ewma": round(item["success_ewma"], 4)
                if item["success_ewma"] is not None else None,
            }
        return {
            "technology": "weaver-inference-governor",
            "profiles": [
                {**asdict(profile), "capabilities": list(profile.capabilities)}
                for profile in MODEL_PROFILES
            ],
            "decisions": self.decisions,
            "telemetry": telemetry,
        }


class SalienceMemoryPyramid:
    """Hot events plus warm aggregates; stores metadata and digests, never content."""

    def __init__(self, *, hot_events: int = 64) -> None:
        self.hot: deque[dict[str, Any]] = deque()
        self.hot_events = min(max(int(hot_events), 16), 256)
        self.warm: dict[str, dict[str, Any]] = {}
        self.total_events = 0
        self.consolidations = 0

    def _consolidate(self, event: dict[str, Any]) -> None:
        for tag in event["tags"]:
            aggregate = self.warm.setdefault(tag, {
                "events": 0,
                "successes": 0,
                "reward_ewma": None,
                "salience_peak": 0.0,
                "last_at_ms": 0,
            })
            aggregate["events"] += 1
            aggregate["successes"] += int(event["success"])
            aggregate["reward_ewma"] = _ewma(aggregate["reward_ewma"], event["reward"], 0.15)
            aggregate["salience_peak"] = max(aggregate["salience_peak"], event["salience"])
            aggregate["last_at_ms"] = event["at_ms"]
        self.consolidations += 1

    def record(
        self,
        *,
        kind: str,
        tags: list[str],
        reward: float,
        surprise: float,
        risk: float,
        success: bool,
        at_ms: int | None = None,
    ) -> dict[str, Any]:
        salience = _clamp(
            abs(float(reward)) * 0.35
            + float(surprise) * 0.25
            + float(risk) * 0.3
            + (0 if success else 0.25)
        )
        body = {
            "at_ms": int(at_ms if at_ms is not None else time.time() * 1000),
            "kind": kind,
            "tags": sorted(set(tags)),
            "reward": round(float(reward), 4),
            "surprise": round(float(surprise), 4),
            "risk": round(float(risk), 4),
            "success": bool(success),
            "salience": round(salience, 4),
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        event = {
            "event_id": f"mem-{uuid.uuid4().hex[:16]}",
            "digest": hashlib.sha256(canonical).hexdigest()[:24],
            **body,
        }
        if len(self.hot) >= self.hot_events:
            self._consolidate(self.hot.popleft())
        self.hot.append(event)
        self.total_events += 1
        return {"event_id": event["event_id"], "digest": event["digest"], "salience": event["salience"]}

    def recall(self, tags: Iterable[str], *, limit: int = 8) -> dict[str, Any]:
        wanted = set(tags)
        matching = [event for event in self.hot if wanted.intersection(event["tags"])]
        matching.sort(key=lambda event: (-event["salience"], -event["at_ms"]))
        warm = {tag: dict(self.warm[tag]) for tag in sorted(wanted) if tag in self.warm}
        return {"hot": [dict(event) for event in matching[:limit]], "warm": warm}

    def snapshot(self) -> dict[str, Any]:
        return {
            "technology": "weaver-salience-pyramid",
            "privacy": "metadata-only",
            "total_events": self.total_events,
            "hot_events": len(self.hot),
            "hot_capacity": self.hot_events,
            "warm_patterns": len(self.warm),
            "consolidations": self.consolidations,
            "high_salience_events": sum(1 for event in self.hot if event["salience"] >= 0.7),
        }


class ShadowPolicyLab:
    """Produces bounded advisory experiments; it has no apply or deploy operation."""

    def __init__(self) -> None:
        self.outcomes = 0
        self.failures = 0
        self.last_proposals: list[dict[str, Any]] = []

    def observe(self, *, success: bool) -> None:
        self.outcomes += 1
        self.failures += int(not success)

    @staticmethod
    def _proposal(kind: str, rationale: str, expected_effect: str) -> dict[str, Any]:
        digest = hashlib.sha256(f"{kind}|{rationale}".encode("utf-8")).hexdigest()[:12]
        return {
            "proposal_id": f"shadow-{digest}",
            "kind": kind,
            "rationale": rationale,
            "expected_effect": expected_effect,
            "status": "shadow-only",
            "automatic_mutation": False,
        }

    def propose(
        self,
        *,
        fabric: dict[str, Any],
        awareness: dict[str, Any],
        immune: dict[str, Any],
    ) -> list[dict[str, Any]]:
        proposals: list[dict[str, Any]] = []
        realtime = fabric.get("lanes", {}).get("realtime", {})
        counters = realtime.get("counters", {})
        p95 = realtime.get("latency_p95_ms")
        if (p95 is not None and float(p95) > 200) or counters.get("deadlines", 0):
            proposals.append(self._proposal(
                "increase-realtime-reservation",
                "realtime p95 or deadline pressure exceeded the 200 ms control-plane budget",
                "reduce voice and reflex admission contention",
            ))
        if awareness.get("awareness_confidence", 0) < 0.45:
            proposals.append(self._proposal(
                "raise-sensor-cadence",
                "fused awareness confidence is below 0.45",
                "improve body and environment freshness before motion",
            ))
        if immune.get("open_components"):
            proposals.append(self._proposal(
                "route-around-open-circuits",
                "one or more components are circuit-open",
                "prefer healthy fallback routes until half-open recovery",
            ))
        if self.outcomes >= 5 and self.failures / self.outcomes > 0.2:
            proposals.append(self._proposal(
                "reduce-action-horizon",
                "recent bounded outcome failure ratio exceeds 20 percent",
                "split plans into shorter observable steps",
            ))
        if not proposals:
            proposals.append(self._proposal(
                "hold-baseline",
                "no measured guardrail currently requires a policy change",
                "preserve stable behavior while collecting more outcomes",
            ))
        self.last_proposals = proposals[:4]
        return [dict(item) for item in self.last_proposals]

    def snapshot(self) -> dict[str, Any]:
        return {
            "technology": "weaver-shadow-policy-lab",
            "mode": "advisory-only",
            "outcomes": self.outcomes,
            "failures": self.failures,
            "proposals": [dict(item) for item in self.last_proposals],
        }


class CognitionMesh:
    """Facade joining all seven angles into one consistent control plane."""

    angles = (
        "perception", "embodiment", "prediction", "compute", "memory", "resilience", "evolution"
    )

    def __init__(self) -> None:
        self.awareness = SituationalGraph()
        self.reflex = ReflexKernel()
        self.twin = CounterfactualTwin()
        self.immune = ResilienceImmuneSystem()
        self.governor = InferenceGovernor()
        self.memory = SalienceMemoryPyramid()
        self.policy = ShadowPolicyLab()
        self.started_at = time.time()
        self.intent_evaluations = 0
        self.outcomes = 0

    @staticmethod
    def _fabric_pressure(fabric: dict[str, Any]) -> float:
        return _clamp(float(fabric.get("accelerator", {}).get("pressure", 0) or 0))

    def observe(self, payload: Any) -> dict[str, Any]:
        before = self.awareness.snapshot()["awareness_confidence"]
        snapshot = self.awareness.observe(payload)
        surprise = _clamp(abs(float(snapshot["awareness_confidence"]) - float(before)))
        tags = []
        if isinstance(payload, dict):
            tags.extend(key for key in ("body", "environment") if key in payload)
            if isinstance(payload.get("sensors"), dict):
                tags.extend(key for key in payload["sensors"] if key in SENSOR_CHANNELS)
        receipt = self.memory.record(
            kind="observation",
            tags=sorted(set(tags)) or ["environment"],
            reward=0,
            surprise=surprise,
            risk=1 - float(snapshot["awareness_confidence"]),
            success=True,
        )
        return {
            "technology": "weaver-cognition-mesh",
            "angle": "perception",
            "awareness": snapshot,
            "memory_receipt": receipt,
        }

    def evaluate_intent(self, capsule: Any, *, fabric: dict[str, Any]) -> dict[str, Any]:
        data = _strict_object(
            capsule,
            field="capsule",
            allowed={
                "version", "capsule_id", "issued_at_ms", "expires_at_ms", "priority", "goal",
                "preconditions", "actions", "rollback", "integrity",
            },
            allow_empty=False,
        )
        awareness = self.awareness.snapshot()
        reflex = self.reflex.evaluate(data, awareness)
        twin = self.twin.simulate(data, awareness, reflex)
        route = self.governor.plan(
            task="embodiment",
            deadline_ms=min(
                45_000,
                max(50, int((data.get("preconditions") or {}).get("max_duration_ms", 20_000))),
            ),
            quality_priority=0.45,
            fabric_pressure=self._fabric_pressure(fabric),
            immune=self.immune,
        )
        memory_receipt = self.memory.record(
            kind="safety",
            tags=["body", "safety"],
            reward=1 if reflex["approved"] else -0.4,
            surprise=0.2 if reflex["decision"] != "approve" else 0,
            risk=float(twin["collision_risk"]),
            success=reflex["approved"],
        )
        self.intent_evaluations += 1
        immune = self.immune.snapshot()
        proposals = self.policy.propose(fabric=fabric, awareness=awareness, immune=immune)
        return {
            "technology": "weaver-seven-angle-plan",
            "version": 1,
            "capsule_id": str(data.get("capsule_id") or "")[:40],
            "decision": twin["recommendation"],
            "angles": {
                "perception": awareness,
                "embodiment": reflex,
                "prediction": twin,
                "compute": route,
                "memory": memory_receipt,
                "resilience": immune,
                "evolution": proposals,
            },
        }

    def plan_inference(self, payload: Any, *, fabric: dict[str, Any]) -> dict[str, Any]:
        data = _strict_object(
            payload,
            field="route request",
            allowed={"task", "deadline_ms", "quality_priority"},
            allow_empty=False,
        )
        task = str(data.get("task") or "").strip().lower()
        if task not in TASKS:
            raise CognitionValidationError("task is invalid")
        deadline = _integer(data.get("deadline_ms", 2_000), 50, 180_000, field="deadline_ms")
        quality = _number(data.get("quality_priority", 0.5), 0, 1, field="quality_priority")
        route = self.governor.plan(
            task=task,
            deadline_ms=deadline,
            quality_priority=quality,
            fabric_pressure=self._fabric_pressure(fabric),
            immune=self.immune,
        )
        receipt = self.memory.record(
            kind="route",
            tags=[task if task in MEMORY_TAGS else "model", "model", "latency"],
            reward=0,
            surprise=0,
            risk=1 - float(route["primary"]["deadline_fit"]),
            success=True,
        )
        return {**route, "memory_receipt": receipt}

    def record_outcome(self, payload: Any) -> dict[str, Any]:
        data = _strict_object(
            payload,
            field="outcome",
            allowed={
                "component", "task", "success", "latency_ms", "target_ms", "quality",
                "reward", "surprise", "risk", "tags",
            },
            allow_empty=False,
        )
        component = _identifier(data.get("component"), field="component")
        if component not in COMPONENTS:
            raise CognitionValidationError("component is not registered")
        task = str(data.get("task") or "chat").strip().lower()
        if task not in TASKS:
            raise CognitionValidationError("task is invalid")
        success = _boolean(data.get("success"), field="success")
        latency = _number(data.get("latency_ms"), 0, 180_000, field="latency_ms")
        target = _number(data.get("target_ms", 2_000), 20, 180_000, field="target_ms")
        quality = _number(data.get("quality", 0.5), 0, 1, field="quality")
        reward = _number(data.get("reward", 1 if success else -1), -1, 1, field="reward")
        surprise = _number(data.get("surprise", 0), 0, 1, field="surprise")
        risk = _number(data.get("risk", 0), 0, 1, field="risk")
        raw_tags = data.get("tags", [])
        if not isinstance(raw_tags, list) or len(raw_tags) > 8:
            raise CognitionValidationError("tags must contain at most eight items")
        tags = []
        for raw in raw_tags:
            tag = _identifier(raw, field="tag", limit=24)
            if tag not in MEMORY_TAGS:
                raise CognitionValidationError("tag is not registered")
            if tag not in tags:
                tags.append(tag)
        default_tag = task if task in MEMORY_TAGS else "model"
        tags = tags or [default_tag, "model"]
        immune = self.immune.record(component, success=success, latency_ms=latency, target_ms=target)
        self.governor.record(component, success=success, latency_ms=latency, quality=quality)
        memory = self.memory.record(
            kind="outcome",
            tags=tags,
            reward=reward,
            surprise=surprise,
            risk=risk,
            success=success,
        )
        self.policy.observe(success=success)
        self.outcomes += 1
        return {
            "technology": "weaver-cognition-outcome",
            "accepted": True,
            "component": component,
            "immune": immune,
            "memory_receipt": memory,
            "samples": self.outcomes,
        }

    def snapshot(self, *, fabric: dict[str, Any]) -> dict[str, Any]:
        awareness = self.awareness.snapshot()
        immune = self.immune.snapshot()
        proposals = self.policy.propose(fabric=fabric, awareness=awareness, immune=immune)
        return {
            "technology": "weaver-cognition-mesh",
            "version": 1,
            "status": "guarded" if immune["status"] == "guarded" else "nominal",
            "uptime_seconds": round(time.time() - self.started_at),
            "angles": list(self.angles),
            "perception": awareness,
            "embodiment": {
                "technology": "weaver-reflex-kernel",
                "intent_evaluations": self.intent_evaluations,
            },
            "prediction": {"technology": "weaver-counterfactual-twin", "mode": "pre-execution"},
            "compute": self.governor.snapshot(),
            "memory": self.memory.snapshot(),
            "resilience": immune,
            "evolution": {**self.policy.snapshot(), "proposals": proposals},
        }
