#!/usr/bin/env python3
"""Strict, model-free health aggregation and bounded dependency probes."""

from __future__ import annotations

import asyncio
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from headless_schemas import HealthComponent, HealthReason, HealthReport


UNREADY_STATUSES = frozenset({"warming", "degraded", "unknown", "disabled"})
DEGRADED_STATUSES = frozenset({"warming", "degraded", "unknown"})


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def component(
    *,
    enabled: bool,
    required: bool,
    status: str,
    source: str = "control-plane",
    reason: HealthReason | None = None,
    latency_ms: float | None = None,
    checked_at: datetime | None = None,
) -> HealthComponent:
    return HealthComponent(
        enabled=enabled,
        required=required,
        status=status,
        source=source,
        reason=reason,
        latency_ms=latency_ms,
        checked_at=checked_at or utc_now(),
    )


def report(
    kind: str,
    components: dict[str, HealthComponent],
    *,
    started_at: float,
    checked_at: datetime | None = None,
) -> HealthReport:
    now = checked_at or utc_now()
    if kind == "liveness":
        return HealthReport(
            kind="liveness",
            status="alive",
            ready=True,
            checked_at=now,
            duration_ms=min(max((time.perf_counter() - started_at) * 1_000, 0.0), 10_000.0),
            reasons=[],
            components=components,
        )

    required_unavailable = any(
        item.required and item.status in UNREADY_STATUSES
        for item in components.values()
    )
    degraded = any(
        item.enabled and item.status in DEGRADED_STATUSES
        for item in components.values()
    )
    reasons: list[HealthReason] = []
    for item in components.values():
        if item.reason is not None and item.reason not in reasons:
            reasons.append(item.reason)
    ready = not required_unavailable
    status = "not-ready" if not ready else ("degraded" if degraded else "ready")
    return HealthReport(
        kind=kind,
        status=status,
        ready=ready,
        checked_at=now,
        duration_ms=min(max((time.perf_counter() - started_at) * 1_000, 0.0), 10_000.0),
        reasons=reasons,
        components=components,
    )


def _http_status_sync(url: str, timeout_seconds: float) -> bool:
    if not url.startswith(("http://", "https://")):
        return False
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "WeaverHealth/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = int(getattr(response, "status", 0) or response.getcode() or 0)
        return 200 <= status < 400


async def probe_http(url: str, *, timeout_seconds: float = 1.5) -> tuple[bool, float]:
    """Probe status only; response bodies and exception details are discarded."""

    timeout = min(max(float(timeout_seconds), 0.1), 3.0)
    started = time.perf_counter()
    try:
        healthy = await asyncio.wait_for(
            asyncio.to_thread(_http_status_sync, url, timeout),
            timeout=timeout + 0.25,
        )
    except (asyncio.TimeoutError, OSError, ValueError):
        healthy = False
    latency_ms = min(max((time.perf_counter() - started) * 1_000, 0.0), 10_000.0)
    return bool(healthy), round(latency_ms, 1)


def probe_directory(path: str | os.PathLike[str]) -> tuple[bool, float]:
    started = time.perf_counter()
    try:
        target = Path(path)
        healthy = (
            target.is_dir()
            and os.access(target, os.R_OK)
            and os.access(target, os.W_OK)
        )
    except OSError:
        healthy = False
    return bool(healthy), round((time.perf_counter() - started) * 1_000, 1)


def probe_codebase_manifest() -> tuple[bool, float]:
    started = time.perf_counter()
    try:
        from codebase_api import build_manifest

        manifest: dict[str, Any] = build_manifest(limit=1)
        healthy = manifest.get("policy", {}).get("mode") == "read_only"
    except (ImportError, OSError, TypeError, ValueError):
        healthy = False
    return bool(healthy), round((time.perf_counter() - started) * 1_000, 1)
