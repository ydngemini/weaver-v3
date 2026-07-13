#!/usr/bin/env python3
"""Deterministic, privacy-safe awareness fusion for Weaver headless v2.

This layer does not ingest raw sensor frames and cannot execute actions.  It
combines already-bounded freshness metadata from the Situational Graph with
Neural Fabric, Cognition Mesh, and dependency health signals so clients do not
mistake one healthy subsystem for complete situational awareness.
"""

from __future__ import annotations

import math
from typing import Any


FUSION_VERSION = 1
DEPENDENCY_NAMES = ("cortex", "n8n", "voice")
DEPENDENCY_STATES = {"ready", "busy", "warming", "degraded", "unknown", "disabled"}
DEPENDENCY_TTL_MS = 600_000


def _number(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _confidence(value: Any, *, default: float = 0.0) -> float:
    return round(min(max(_number(value, default=default), 0.0), 1.0), 4)


def _integer(value: Any, *, default: int = 0, maximum: int = 10_000_000_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return min(max(int(value), 0), maximum)


def _source(value: Any) -> dict[str, Any]:
    data = value if isinstance(value, dict) else {}
    raw_age = data.get("age_ms")
    age_ms = None if raw_age is None else _integer(raw_age, maximum=3_600_000)
    observed = data.get("observed_at_ms")
    observed_at_ms = None if observed is None else _integer(observed)
    return {
        "fresh": bool(data.get("fresh", False)),
        "age_ms": age_ms,
        "confidence": _confidence(data.get("confidence")),
        "observed_at_ms": observed_at_ms,
    }


def _control_source(*, confidence: float, now_ms: int) -> dict[str, Any]:
    return {
        "fresh": True,
        "age_ms": 0,
        "confidence": _confidence(confidence),
        "observed_at_ms": now_ms,
    }


def _dependency_service(name: str, value: Any, *, now_ms: int) -> dict[str, Any]:
    data = value if isinstance(value, dict) else {}
    enabled = bool(data.get("enabled", False))
    required = bool(data.get("required", name == "cortex")) and enabled
    raw_status = str(data.get("status") or ("unknown" if enabled else "disabled")).lower()
    status = raw_status if raw_status in DEPENDENCY_STATES else "unknown"
    if not enabled:
        status = "disabled"

    raw_observed = data.get("observed_at_ms")
    observed_at_ms = (
        _integer(raw_observed)
        if isinstance(raw_observed, (int, float)) and not isinstance(raw_observed, bool)
        else None
    )
    age_ms = None if observed_at_ms is None else min(max(now_ms - observed_at_ms, 0), 3_600_000)
    ttl_ms = min(max(_integer(data.get("ttl_ms"), default=DEPENDENCY_TTL_MS), 1_000), 3_600_000)
    fresh = bool(
        enabled
        and status in {"ready", "busy", "warming"}
        and (status == "busy" or (age_ms is not None and age_ms <= ttl_ms))
    )
    base = {
        "ready": 1.0,
        "busy": 1.0,
        "warming": 0.65,
        "unknown": 0.35,
        "degraded": 0.15,
        "disabled": 1.0,
    }[status]
    if enabled and age_ms is not None and status != "busy":
        freshness_factor = max(0.0, 1.0 - age_ms / max(ttl_ms * 2, 1))
        base *= freshness_factor
    elif enabled and age_ms is None and status in {"ready", "warming"}:
        base *= 0.7
    return {
        "enabled": enabled,
        "required": required,
        "status": status,
        "fresh": fresh,
        "age_ms": age_ms,
        "confidence": _confidence(base),
        "observed_at_ms": observed_at_ms,
    }


def _dependencies(value: Any, *, now_ms: int) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    services = {
        name: _dependency_service(name, raw.get(name), now_ms=now_ms)
        for name in DEPENDENCY_NAMES
    }
    enabled = [service for service in services.values() if service["enabled"]]
    required = [service for service in enabled if service["required"]]
    scored = required or enabled
    confidence = (
        sum(float(service["confidence"]) for service in scored) / len(scored)
        if scored
        else 1.0
    )
    stale_required = sum(1 for service in required if not service["fresh"])
    degraded = sum(
        1
        for service in enabled
        if service["status"] in {"degraded", "unknown"} or not service["fresh"]
    )
    if stale_required:
        status = "degraded"
    elif degraded:
        status = "limited"
    elif any(service["status"] in {"busy", "warming"} for service in enabled):
        status = "busy"
    else:
        status = "nominal"
    observed_values = [
        int(service["observed_at_ms"])
        for service in scored
        if service["observed_at_ms"] is not None
    ]
    observed_at_ms = min(observed_values) if observed_values else now_ms
    age_ms = min(max(now_ms - observed_at_ms, 0), 3_600_000)
    aggregate = {
        "fresh": stale_required == 0,
        "age_ms": age_ms,
        "confidence": _confidence(confidence),
        "observed_at_ms": observed_at_ms,
    }
    return {
        "status": status,
        "confidence": _confidence(confidence),
        "degraded_count": degraded,
        "services": services,
        "aggregate": aggregate,
    }


def fuse_awareness(
    *,
    channels: dict[str, Any],
    fabric: dict[str, Any],
    cognition: dict[str, Any],
    dependencies: dict[str, Any] | None,
    headless_fresh: bool,
    now_ms: int,
) -> dict[str, Any]:
    """Return one bounded awareness verdict with stable explanations."""

    body = _source(channels.get("body"))
    world = _source(channels.get("environment"))
    camera = _source(channels.get("camera"))
    microphone = _source(channels.get("microphone"))

    cognition_nominal = cognition.get("status") == "nominal"
    cognition_source = _control_source(
        confidence=1.0 if cognition_nominal else 0.4,
        now_ms=now_ms,
    )
    ledger = fabric.get("ledger") if isinstance(fabric.get("ledger"), dict) else {}
    pressure = _confidence(
        (fabric.get("accelerator") or {}).get("pressure")
        if isinstance(fabric.get("accelerator"), dict)
        else 0.0
    )
    fabric_status = str(fabric.get("status") or "guarded")
    fabric_base = {"nominal": 1.0, "watch": 0.65, "guarded": 0.25}.get(fabric_status, 0.25)
    if not bool(ledger.get("valid", False)):
        fabric_base = 0.0
    fabric_source = _control_source(
        confidence=fabric_base * max(0.4, 1.0 - pressure * 0.6),
        now_ms=now_ms,
    )
    dependency_state = _dependencies(dependencies, now_ms=now_ms)
    dependency_source = dependency_state.pop("aggregate")

    sources = {
        "body": body,
        "world": world,
        "cognition": cognition_source,
        "fabric": fabric_source,
        "dependencies": dependency_source,
    }
    weighted = (
        body["confidence"] * 0.30
        + world["confidence"] * 0.30
        + camera["confidence"] * 0.05
        + microphone["confidence"] * 0.05
        + cognition_source["confidence"] * 0.12
        + fabric_source["confidence"] * 0.10
        + dependency_source["confidence"] * 0.08
    )
    critical_ceiling = min(1.0, (body["confidence"] + world["confidence"]) / 2 + 0.25)
    confidence = _confidence(min(weighted, critical_ceiling))

    reasons: list[str] = []
    if not headless_fresh:
        reasons.append("headless-stale")
    if not body["fresh"]:
        reasons.append("body-stale")
    if not world["fresh"]:
        reasons.append("environment-stale")
    if not camera["fresh"]:
        reasons.append("camera-stale")
    if not microphone["fresh"]:
        reasons.append("microphone-stale")
    if not cognition_nominal:
        reasons.append("cognition-guarded")
    if fabric_status != "nominal":
        reasons.append("fabric-pressure")
    if not bool(ledger.get("valid", False)):
        reasons.append("fabric-ledger-invalid")
    if dependency_state["status"] == "degraded":
        reasons.append("dependency-degraded")
    elif dependency_state["status"] == "limited":
        reasons.append("dependency-limited")

    if body["confidence"] == 0 and world["confidence"] == 0:
        status = "no-data"
    elif (
        not headless_fresh
        or not body["fresh"]
        or not world["fresh"]
        or not cognition_nominal
        or fabric_source["confidence"] < 0.35
        or dependency_state["status"] == "degraded"
        or confidence < 0.45
    ):
        status = "degraded"
    elif reasons or confidence < 0.8:
        status = "limited"
    else:
        status = "nominal"

    return {
        "fusion_version": FUSION_VERSION,
        "status": status,
        "confidence": confidence,
        "degraded_reasons": reasons,
        "sources": sources,
        "dependencies": dependency_state,
    }
