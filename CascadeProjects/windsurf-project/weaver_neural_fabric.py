#!/usr/bin/env python3
"""Weaver Neural Fabric: deadline-aware cognition scheduling and intent capsules.

The Fabric is deliberately model-agnostic.  It sits above Bedrock, local models,
voice, embodiment, and background cognition to provide four properties:

* reserved accelerator capacity for live voice;
* bounded per-lane concurrency, deadlines, admission, and load shedding;
* a metadata-only tamper-evident operational ledger;
* signed declarative Intent Capsules with expiry, preconditions, and rollback.

No prompt, transcript, secret, model output, URL, or arbitrary command is ever
accepted by the Fabric ledger.  The scheduler executes coroutine factories that
the trusted application supplies; an Intent Capsule is data, never executable
code.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import secrets
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Generic, TypeVar


T = TypeVar("T")


class FabricError(RuntimeError):
    """Base class for safe, expected Fabric failures."""


class FabricOverloaded(FabricError):
    """Raised when bounded admission or load shedding rejects work."""


class FabricDeadlineExceeded(FabricError):
    """Raised when a request cannot finish inside its declared deadline."""


class IntentValidationError(ValueError):
    """Raised when an Intent Capsule request violates the capability schema."""


class WorkClass(str, Enum):
    REALTIME = "realtime"
    INTERACTIVE = "interactive"
    EMBODIMENT = "embodiment"
    BACKGROUND = "background"

    @classmethod
    def parse(cls, value: "WorkClass | str") -> "WorkClass":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value or "").strip().lower())
        except ValueError as exc:
            raise FabricOverloaded("unknown work class") from exc


@dataclass(frozen=True)
class LanePolicy:
    concurrency: int
    max_queue: int
    default_deadline_ms: int
    max_deadline_ms: int
    max_cost_units: int


DEFAULT_LANE_POLICIES: dict[WorkClass, LanePolicy] = {
    WorkClass.REALTIME: LanePolicy(2, 8, 30_000, 180_000, 8),
    WorkClass.INTERACTIVE: LanePolicy(4, 24, 90_000, 180_000, 8),
    WorkClass.EMBODIMENT: LanePolicy(2, 16, 15_000, 45_000, 4),
    WorkClass.BACKGROUND: LanePolicy(1, 4, 120_000, 240_000, 6),
}


@dataclass(frozen=True)
class FabricExecution(Generic[T]):
    value: T
    receipt: dict[str, Any]


def _percentile(values: list[float], quantile: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return round(clean[0], 1)
    position = (len(clean) - 1) * min(max(float(quantile), 0.0), 1.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(clean[lower], 1)
    weight = position - lower
    return round(clean[lower] * (1 - weight) + clean[upper] * weight, 1)


def _safe_identifier(value: Any, limit: int = 56) -> str:
    return "".join(
        character if character.isalnum() or character in "_.:-" else "-"
        for character in str(value or "").strip().lower()
    )[:limit]


class ProofOfSequenceLedger:
    """Bounded hash chain containing operational metadata only."""

    _allowed_fields = {
        "name", "result", "reason", "queue_ms", "run_ms", "total_ms",
        "cost_units", "deadline_ms", "active", "queued", "pressure",
    }

    def __init__(self, max_events: int = 256) -> None:
        self.max_events = min(max(int(max_events), 32), 2048)
        self._events: deque[dict[str, Any]] = deque()
        self._sequence = 0
        self._anchor_digest = "0" * 64
        self._head_digest = self._anchor_digest

    @staticmethod
    def _safe_field(key: str, value: Any) -> Any:
        if key not in ProofOfSequenceLedger._allowed_fields:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return round(float(value), 3)
        if isinstance(value, str):
            return _safe_identifier(value)
        return None

    def record(
        self,
        kind: str,
        lane: WorkClass | str,
        request_id: str,
        **fields: Any,
    ) -> dict[str, Any]:
        self._sequence += 1
        event: dict[str, Any] = {
            "sequence": self._sequence,
            "at_ms": int(time.time() * 1000),
            "kind": _safe_identifier(kind, 40),
            "lane": WorkClass.parse(lane).value,
            "request_id": _safe_identifier(request_id, 48),
            "previous_digest": self._head_digest,
        }
        for key, value in fields.items():
            safe = self._safe_field(key, value)
            if safe is not None:
                event[key] = safe
        canonical = json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
        event["digest"] = hashlib.sha256(bytes.fromhex(self._head_digest) + canonical).hexdigest()
        self._head_digest = event["digest"]
        self._events.append(event)
        if len(self._events) > self.max_events:
            removed = self._events.popleft()
            self._anchor_digest = removed["digest"]
        return dict(event)

    def verify(self) -> bool:
        previous = self._anchor_digest
        for stored in self._events:
            event = dict(stored)
            digest = str(event.pop("digest", ""))
            if event.get("previous_digest") != previous:
                return False
            canonical = json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
            expected = hashlib.sha256(bytes.fromhex(previous) + canonical).hexdigest()
            if not hmac.compare_digest(digest, expected):
                return False
            previous = digest
        return hmac.compare_digest(previous, self._head_digest)

    def snapshot(self, limit: int = 32) -> dict[str, Any]:
        return {
            "algorithm": "sha256-chain-v1",
            "events": len(self._events),
            "sequence": self._sequence,
            "anchor_digest": self._anchor_digest,
            "head_digest": self._head_digest,
            "valid": self.verify(),
            "tail": [dict(event) for event in list(self._events)[-max(1, min(int(limit), 64)):]],
        }


class WeightedAcceleratorCapacity:
    """Weighted capacity that permanently reserves units for real-time work."""

    def __init__(self, total_units: int, realtime_reserved_units: int) -> None:
        self.total_units = min(max(int(total_units), 4), 128)
        self.realtime_reserved_units = min(
            max(int(realtime_reserved_units), 1), self.total_units - 1
        )
        self.in_use = 0
        self.realtime_in_use = 0
        self._condition = asyncio.Condition()

    def _limit_for(self, lane: WorkClass) -> int:
        return self.total_units if lane is WorkClass.REALTIME else (
            self.total_units - self.realtime_reserved_units
        )

    async def acquire(self, lane: WorkClass, units: int, deadline_at: float) -> None:
        limit = self._limit_for(lane)
        if units > limit:
            raise FabricOverloaded("work cost exceeds lane capacity")
        async with self._condition:
            while self.in_use + units > limit:
                remaining = deadline_at - time.monotonic()
                if remaining <= 0:
                    raise FabricDeadlineExceeded("accelerator admission deadline exceeded")
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except TimeoutError as exc:
                    raise FabricDeadlineExceeded("accelerator admission deadline exceeded") from exc
            self.in_use += units
            if lane is WorkClass.REALTIME:
                self.realtime_in_use += units

    async def release(self, lane: WorkClass, units: int) -> None:
        async with self._condition:
            self.in_use = max(0, self.in_use - units)
            if lane is WorkClass.REALTIME:
                self.realtime_in_use = max(0, self.realtime_in_use - units)
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_units": self.total_units,
            "realtime_reserved_units": self.realtime_reserved_units,
            "in_use": self.in_use,
            "realtime_in_use": self.realtime_in_use,
            "pressure": round(self.in_use / self.total_units, 4),
        }


class NeuralFabric:
    """Bulkheaded, deadline-aware scheduler for all Weaver cognition classes."""

    def __init__(
        self,
        *,
        capacity_units: int = 16,
        realtime_reserved_units: int = 4,
        lane_policies: dict[WorkClass, LanePolicy] | None = None,
        ledger_events: int = 256,
    ) -> None:
        self.policies = dict(lane_policies or DEFAULT_LANE_POLICIES)
        self.capacity = WeightedAcceleratorCapacity(capacity_units, realtime_reserved_units)
        self.ledger = ProofOfSequenceLedger(ledger_events)
        self._lane_semaphores = {
            lane: asyncio.Semaphore(policy.concurrency)
            for lane, policy in self.policies.items()
        }
        self._state_lock = asyncio.Lock()
        self._queued = {lane: 0 for lane in self.policies}
        self._active = {lane: 0 for lane in self.policies}
        self._counters = {
            lane: {
                "submitted": 0,
                "completed": 0,
                "failed": 0,
                "deadlines": 0,
                "shed": 0,
                "cancelled": 0,
            }
            for lane in self.policies
        }
        self._latency_ms = {lane: deque(maxlen=128) for lane in self.policies}
        self._queue_ms = {lane: deque(maxlen=128) for lane in self.policies}
        self.started_at = time.time()

    def _remaining(self, deadline_at: float) -> float:
        remaining = deadline_at - time.monotonic()
        if remaining <= 0:
            raise FabricDeadlineExceeded("fabric deadline exceeded")
        return remaining

    async def _admit(self, lane: WorkClass, request_id: str, name: str) -> None:
        policy = self.policies[lane]
        async with self._state_lock:
            realtime_pressure = self._active[WorkClass.REALTIME] + self._queued[WorkClass.REALTIME]
            if lane is WorkClass.BACKGROUND and realtime_pressure > 0:
                self._counters[lane]["shed"] += 1
                self.ledger.record(
                    "shed", lane, request_id, name=name, reason="realtime-pressure",
                    active=self._active[lane], queued=self._queued[lane],
                )
                raise FabricOverloaded("background work yielded to realtime")
            if self._queued[lane] >= policy.max_queue:
                self._counters[lane]["shed"] += 1
                self.ledger.record(
                    "shed", lane, request_id, name=name, reason="queue-full",
                    active=self._active[lane], queued=self._queued[lane],
                )
                raise FabricOverloaded("fabric lane queue full")
            self._queued[lane] += 1
            self._counters[lane]["submitted"] += 1

    async def _finish_state(self, lane: WorkClass, promoted: bool) -> None:
        async with self._state_lock:
            if promoted:
                self._active[lane] = max(0, self._active[lane] - 1)
            else:
                self._queued[lane] = max(0, self._queued[lane] - 1)

    async def execute(
        self,
        *,
        lane: WorkClass | str,
        name: str,
        factory: Callable[[], Awaitable[T]],
        deadline_ms: int | None = None,
        cost_units: int = 1,
    ) -> FabricExecution[T]:
        work_class = WorkClass.parse(lane)
        policy = self.policies[work_class]
        deadline = int(deadline_ms or policy.default_deadline_ms)
        deadline = min(max(deadline, 50), policy.max_deadline_ms)
        cost = int(cost_units)
        if isinstance(cost_units, bool) or not 1 <= cost <= policy.max_cost_units:
            raise FabricOverloaded("invalid work cost")
        request_id = f"fab-{uuid.uuid4().hex[:20]}"
        safe_name = _safe_identifier(name, 56) or "work"
        submitted_at = time.monotonic()
        deadline_at = submitted_at + deadline / 1000
        lane_acquired = False
        capacity_acquired = False
        promoted = False
        await self._admit(work_class, request_id, safe_name)
        self.ledger.record(
            "submitted", work_class, request_id, name=safe_name,
            cost_units=cost, deadline_ms=deadline,
            active=self._active[work_class], queued=self._queued[work_class],
        )
        try:
            try:
                await asyncio.wait_for(
                    self._lane_semaphores[work_class].acquire(),
                    timeout=self._remaining(deadline_at),
                )
                lane_acquired = True
            except TimeoutError as exc:
                raise FabricDeadlineExceeded("lane admission deadline exceeded") from exc
            async with self._state_lock:
                self._queued[work_class] = max(0, self._queued[work_class] - 1)
                self._active[work_class] += 1
                promoted = True
            queue_ms = (time.monotonic() - submitted_at) * 1000
            await self.capacity.acquire(work_class, cost, deadline_at)
            capacity_acquired = True
            run_started = time.monotonic()
            try:
                value = await asyncio.wait_for(factory(), timeout=self._remaining(deadline_at))
            except TimeoutError as exc:
                raise FabricDeadlineExceeded("work execution deadline exceeded") from exc
            run_ms = (time.monotonic() - run_started) * 1000
            total_ms = (time.monotonic() - submitted_at) * 1000
            self._counters[work_class]["completed"] += 1
            self._latency_ms[work_class].append(total_ms)
            self._queue_ms[work_class].append(queue_ms)
            self.ledger.record(
                "completed", work_class, request_id, name=safe_name, result="ok",
                queue_ms=queue_ms, run_ms=run_ms, total_ms=total_ms,
                cost_units=cost, deadline_ms=deadline,
            )
            return FabricExecution(
                value=value,
                receipt={
                    "request_id": request_id,
                    "lane": work_class.value,
                    "name": safe_name,
                    "queue_ms": round(queue_ms, 1),
                    "run_ms": round(run_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "cost_units": cost,
                    "deadline_ms": deadline,
                },
            )
        except FabricDeadlineExceeded:
            self._counters[work_class]["deadlines"] += 1
            self.ledger.record(
                "deadline", work_class, request_id, name=safe_name,
                reason="deadline", total_ms=(time.monotonic() - submitted_at) * 1000,
                cost_units=cost, deadline_ms=deadline,
            )
            raise
        except asyncio.CancelledError:
            self._counters[work_class]["cancelled"] += 1
            self.ledger.record(
                "cancelled", work_class, request_id, name=safe_name,
                reason="caller-cancelled", total_ms=(time.monotonic() - submitted_at) * 1000,
            )
            raise
        except Exception:
            self._counters[work_class]["failed"] += 1
            self.ledger.record(
                "failed", work_class, request_id, name=safe_name,
                reason="work-failed", total_ms=(time.monotonic() - submitted_at) * 1000,
                cost_units=cost,
            )
            raise
        finally:
            if capacity_acquired:
                await asyncio.shield(self.capacity.release(work_class, cost))
            if lane_acquired:
                self._lane_semaphores[work_class].release()
            await asyncio.shield(self._finish_state(work_class, promoted))

    def snapshot(self) -> dict[str, Any]:
        lanes: dict[str, Any] = {}
        for lane, policy in self.policies.items():
            latency = list(self._latency_ms[lane])
            queue = list(self._queue_ms[lane])
            lanes[lane.value] = {
                "policy": asdict(policy),
                "active": self._active[lane],
                "queued": self._queued[lane],
                "counters": dict(self._counters[lane]),
                "latency_p50_ms": _percentile(latency, 0.50),
                "latency_p95_ms": _percentile(latency, 0.95),
                "queue_p95_ms": _percentile(queue, 0.95),
            }
        pressure = self.capacity.snapshot()
        realtime = lanes[WorkClass.REALTIME.value]
        status = "guarded" if pressure["pressure"] >= 0.85 else "nominal"
        if realtime["counters"]["deadlines"] or realtime["counters"]["shed"]:
            status = "watch"
        return {
            "technology": "weaver-neural-fabric",
            "version": 1,
            "status": status,
            "uptime_seconds": round(time.time() - self.started_at),
            "accelerator": pressure,
            "lanes": lanes,
            "ledger": self.ledger.snapshot(),
        }


POSE_SLOTS = {
    "presence", "openness", "lean", "turn", "hipShift", "spineTwist",
    "headPitch", "headYaw", "headRoll", "leftArm", "rightArm",
    "leftForearm", "rightForearm", "leftElbow", "rightElbow", "leftHand",
    "rightHand", "leftStep", "rightStep", "leftKnee", "rightKnee",
    "leftAnkle", "rightAnkle", "gazeX", "gazeY",
}
PENTHOUSE_ZONES = {"center", "window", "kitchen", "gallery", "lounge"}
PENTHOUSE_INTERACTIONS = {
    "window_glazing", "balcony_rail", "lounge_sofa", "coffee_table",
    "serving_tray", "reading_book", "lounge_chair", "floor_lamp",
    "fireplace", "sculpture_orb", "gallery_art", "kitchen_island",
    "bar_stool", "pendant_lights", "fruit_bowl", "wine_bottle",
    "back_bar", "bed", "night_lamp", "indoor_plant",
}
SENSOR_CAPABILITIES = {"body", "environment", "camera", "microphone"}
MEMORY_CATEGORIES = {"preference", "constraint", "observation", "relationship"}


def _bounded_text(value: Any, limit: int, *, field: str) -> str:
    text = " ".join(str(value or "").split())
    text = "".join(character for character in text if character.isprintable())
    text = text.replace("<", "").replace(">", "")
    if not text or len(text) > limit:
        raise IntentValidationError(f"{field} must be 1..{limit} characters")
    return text


def _bounded_number(value: Any, low: float, high: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IntentValidationError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise IntentValidationError(f"{field} is out of range")
    return round(number, 4)


class IntentCompiler:
    """Compiles bounded declarative actions into signed, expiring capsules."""

    action_types = {"pose", "bones", "navigate", "interact", "speak", "observe", "remember"}

    def __init__(self, signing_secret: str | bytes | None = None) -> None:
        if isinstance(signing_secret, str):
            secret = signing_secret.encode("utf-8")
        else:
            secret = signing_secret or secrets.token_bytes(32)
        if len(secret) < 16:
            secret = hashlib.sha256(secret).digest()
        self._secret = secret
        self.key_id = hashlib.sha256(secret).hexdigest()[:12]

    def _compile_action(self, raw: Any, index: int) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise IntentValidationError("each action must be an object")
        action_type = str(raw.get("type") or "").strip().lower()
        if action_type not in self.action_types:
            raise IntentValidationError("unsupported action type")
        action: dict[str, Any] = {"id": f"a{index + 1}", "type": action_type}
        if action_type == "pose":
            values = raw.get("values", raw.get("pose"))
            if not isinstance(values, dict) or not 1 <= len(values) <= len(POSE_SLOTS):
                raise IntentValidationError("pose values must be a bounded object")
            unknown = set(values) - POSE_SLOTS
            if unknown:
                raise IntentValidationError("pose contains unknown controls")
            action["values"] = {
                key: _bounded_number(value, -1.0, 1.0, field=f"pose.{key}")
                for key, value in values.items()
            }
        elif action_type == "bones":
            bones = raw.get("bones")
            if not isinstance(bones, dict) or not 1 <= len(bones) <= 64:
                raise IntentValidationError("bones must contain 1..64 controls")
            compiled: dict[str, Any] = {}
            for name, rotation in bones.items():
                safe_name = str(name or "")
                if not safe_name or len(safe_name) > 80 or not all(
                    character.isalnum() or character in "_.:-" for character in safe_name
                ):
                    raise IntentValidationError("invalid bone name")
                if not isinstance(rotation, dict) or not rotation or set(rotation) - {"x", "y", "z"}:
                    raise IntentValidationError("bone rotation must contain x/y/z only")
                compiled[safe_name] = {
                    axis: _bounded_number(rotation.get(axis, 0), -1.05, 1.05, field=f"bone.{axis}")
                    for axis in ("x", "y", "z")
                }
            action["bones"] = compiled
        elif action_type == "navigate":
            zone = str(raw.get("zone") or "").strip().lower()
            if zone not in PENTHOUSE_ZONES:
                raise IntentValidationError("invalid navigation zone")
            action["zone"] = zone
        elif action_type == "interact":
            interaction = str(raw.get("interaction") or "").strip().lower()
            if interaction not in PENTHOUSE_INTERACTIONS:
                raise IntentValidationError("invalid interaction")
            action["interaction"] = interaction
        elif action_type == "speak":
            action["text"] = _bounded_text(raw.get("text"), 800, field="speak.text")
        elif action_type == "observe":
            sensor = str(raw.get("sensor") or "").strip().lower()
            if sensor not in SENSOR_CAPABILITIES:
                raise IntentValidationError("invalid sensor capability")
            action["sensor"] = sensor
        elif action_type == "remember":
            category = str(raw.get("category") or "").strip().lower()
            if category not in MEMORY_CATEGORIES:
                raise IntentValidationError("invalid memory category")
            action["category"] = category
            action["note"] = _bounded_text(raw.get("note"), 500, field="remember.note")
        return action

    @staticmethod
    def _rollback_for(actions: list[dict[str, Any]]) -> list[str]:
        rollback: list[str] = []
        mapping = {
            "pose": "reset_pose",
            "bones": "reset_bones",
            "navigate": "stop_locomotion",
            "interact": "cancel_interaction",
            "speak": "stop_speech",
        }
        for action in reversed(actions):
            operation = mapping.get(action["type"])
            if operation and operation not in rollback:
                rollback.append(operation)
        return rollback

    def _signature(self, body: dict[str, Any]) -> str:
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hmac.new(self._secret, canonical, hashlib.sha256).hexdigest()

    def compile(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise IntentValidationError("intent request must be an object")
        goal = _bounded_text(payload.get("goal"), 160, field="goal")
        raw_actions = payload.get("actions")
        if not isinstance(raw_actions, list) or not 1 <= len(raw_actions) <= 8:
            raise IntentValidationError("actions must contain 1..8 items")
        actions = [self._compile_action(action, index) for index, action in enumerate(raw_actions)]
        raw_ttl = payload.get("ttl_ms", 15_000)
        if isinstance(raw_ttl, bool) or not isinstance(raw_ttl, (int, float)):
            raise IntentValidationError("ttl_ms must be numeric")
        ttl_ms = int(raw_ttl)
        if not 1_000 <= ttl_ms <= 60_000:
            raise IntentValidationError("ttl_ms must be between 1000 and 60000")
        try:
            priority = WorkClass.parse(payload.get("priority", WorkClass.EMBODIMENT.value))
        except FabricOverloaded as exc:
            raise IntentValidationError("invalid intent priority") from exc
        if priority is WorkClass.BACKGROUND:
            raise IntentValidationError("intent capsules cannot use background priority")
        preconditions = payload.get("preconditions") or {}
        if not isinstance(preconditions, dict):
            raise IntentValidationError("preconditions must be an object")
        allowed_preconditions = {"world_revision", "body_revision", "requires_awake", "max_duration_ms"}
        if set(preconditions) - allowed_preconditions:
            raise IntentValidationError("unknown precondition")
        requires_awake = preconditions.get("requires_awake", True)
        if not isinstance(requires_awake, bool):
            raise IntentValidationError("requires_awake must be boolean")
        compiled_preconditions = {
            "world_revision": int(_bounded_number(
                preconditions.get("world_revision", 0), 0, 1_000_000_000,
                field="preconditions.world_revision",
            )),
            "body_revision": int(_bounded_number(
                preconditions.get("body_revision", 0), 0, 1_000_000_000,
                field="preconditions.body_revision",
            )),
            "requires_awake": requires_awake,
            "max_duration_ms": int(_bounded_number(
                preconditions.get("max_duration_ms", min(ttl_ms, 30_000)),
                500, 30_000, field="preconditions.max_duration_ms",
            )),
        }
        issued_at = int(time.time() * 1000)
        body: dict[str, Any] = {
            "version": 1,
            "capsule_id": f"cap-{uuid.uuid4().hex[:24]}",
            "issued_at_ms": issued_at,
            "expires_at_ms": issued_at + ttl_ms,
            "priority": priority.value,
            "goal": goal,
            "preconditions": compiled_preconditions,
            "actions": actions,
            "rollback": self._rollback_for(actions),
        }
        return {
            **body,
            "integrity": {
                "algorithm": "hmac-sha256",
                "key_id": self.key_id,
                "signature": self._signature(body),
            },
        }

    def verify(self, capsule: Any, *, check_expiry: bool = True) -> bool:
        if not isinstance(capsule, dict) or not isinstance(capsule.get("integrity"), dict):
            return False
        integrity = capsule["integrity"]
        if integrity.get("algorithm") != "hmac-sha256" or integrity.get("key_id") != self.key_id:
            return False
        body = {key: value for key, value in capsule.items() if key != "integrity"}
        supplied = str(integrity.get("signature") or "")
        if not hmac.compare_digest(supplied, self._signature(body)):
            return False
        if check_expiry:
            try:
                if int(body.get("expires_at_ms", 0)) < int(time.time() * 1000):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def capabilities(self) -> dict[str, Any]:
        return {
            "technology": "weaver-intent-capsules",
            "version": 1,
            "action_types": sorted(self.action_types),
            "pose_slots": sorted(POSE_SLOTS),
            "zones": sorted(PENTHOUSE_ZONES),
            "interactions": sorted(PENTHOUSE_INTERACTIONS),
            "sensors": sorted(SENSOR_CAPABILITIES),
            "max_actions": 8,
            "max_ttl_ms": 60_000,
            "signing_algorithm": "hmac-sha256",
            "key_id": self.key_id,
        }


class SlidingWindowRateLimiter:
    """Small global limiter suitable for a single authenticated operator key."""

    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = min(max(int(limit), 1), 10_000)
        self.window_seconds = min(max(float(window_seconds), 1.0), 3600.0)
        self._events: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        now = time.monotonic()
        async with self._lock:
            while self._events and now - self._events[0] >= self.window_seconds:
                self._events.popleft()
            if len(self._events) >= self.limit:
                return False
            self._events.append(now)
            return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "window_seconds": self.window_seconds,
            "used": len(self._events),
        }
