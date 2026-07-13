#!/usr/bin/env python3
"""Bounded, content-free observability for Weaver's public runtime."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import math
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from headless_schemas import ObservabilityReport


CORRELATION_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
OPERATION_PATTERN = re.compile(r"^[a-z0-9_.:-]{1,80}$")
ATTRIBUTE_ALLOWLIST = frozenset({
    "method",
    "route",
    "phase",
    "reason_code",
    "protocol",
    "revision",
    "lane",
    "speaker",
    "frames",
    "bytes",
    "queue_depth",
    "retryable",
})
CURRENT_CORRELATION_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "weaver_correlation_id",
    default="",
)


def correlation_id(value: str = "") -> str:
    candidate = str(value or "").strip()
    if CORRELATION_PATTERN.fullmatch(candidate):
        return candidate
    return f"req-{uuid.uuid4().hex[:24]}"


def current_correlation_id() -> str:
    current = CURRENT_CORRELATION_ID.get("")
    return current if CORRELATION_PATTERN.fullmatch(current) else correlation_id()


def bind_correlation(value: str = "") -> contextvars.Token[str]:
    return CURRENT_CORRELATION_ID.set(correlation_id(value))


def reset_correlation(token: contextvars.Token[str]) -> None:
    CURRENT_CORRELATION_ID.reset(token)


def normalized_route(path: str) -> str:
    route = str(path or "/").split("?", 1)[0][:160]
    route = re.sub(r"(?i)(?:mem|turn)-[0-9a-f]{20,32}", "{id}", route)
    route = re.sub(r"(?i)cap-[0-9a-f]{20,32}", "{id}", route)
    route = re.sub(r"(?<![A-Za-z])[0-9]{2,}(?![A-Za-z])", "{n}", route)
    route = re.sub(r"[^A-Za-z0-9_./:{}-]", "-", route)
    return route[:96] or "/"


def operation_for(scope_type: str, method: str, route: str) -> str:
    slug = normalized_route(route).strip("/").lower().replace("/", ".") or "root"
    slug = slug.replace("{id}", "id").replace("{n}", "n")
    candidate = f"{scope_type}.{str(method or 'connect').lower()}.{slug}"[:80]
    return candidate if OPERATION_PATTERN.fullmatch(candidate) else "runtime.other"


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(min(max(float(value), 0.0), 600_000.0) for value in values)
    position = (len(ordered) - 1) * min(max(float(quantile), 0.0), 1.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 1)


def _safe_attributes(values: dict[str, Any] | None) -> dict[str, str | int | float | bool]:
    safe: dict[str, str | int | float | bool] = {}
    for key, raw in (values or {}).items():
        name = str(key)
        if name not in ATTRIBUTE_ALLOWLIST or len(safe) >= 8:
            continue
        if isinstance(raw, bool):
            safe[name] = raw
        elif isinstance(raw, int) and not isinstance(raw, bool):
            safe[name] = min(max(raw, -1_000_000_000), 1_000_000_000)
        elif isinstance(raw, float) and math.isfinite(raw):
            safe[name] = round(min(max(raw, -1_000_000_000.0), 1_000_000_000.0), 3)
        elif isinstance(raw, str):
            value = raw[:96]
            if re.fullmatch(r"[A-Za-z0-9_./:{}-]+", value):
                safe[name] = value
    return safe


@dataclass(frozen=True)
class SLODefinition:
    objective: float
    latency_target_ms: float | None


@dataclass
class _MetricBucket:
    requests: int = 0
    successes: int = 0
    server_errors: int = 0
    client_errors: int = 0
    cancelled: int = 0
    in_flight: int = 0
    max_in_flight: int = 0
    durations: deque[float] = field(default_factory=deque)


@dataclass(frozen=True)
class SpanToken:
    trace_id: str
    correlation_id: str
    operation: str
    started_perf: float
    started_at: datetime
    attributes: dict[str, str | int | float | bool]


DEFAULT_SLOS: dict[str, SLODefinition] = {
    "http.get.headless.v2.state": SLODefinition(0.999, 200.0),
    "http.post.headless.v2.session": SLODefinition(0.99, 200.0),
    "http.post.headless.v2.chat.stream": SLODefinition(0.95, 115_000.0),
    "websocket.connect.headless.v2.stream": SLODefinition(0.995, None),
    "websocket.connect.realtime.voice": SLODefinition(0.95, None),
    "headless.chat.reaction": SLODefinition(0.99, 200.0),
    "headless.chat.semantic": SLODefinition(0.95, 115_000.0),
    "headless.state.publish": SLODefinition(0.999, 50.0),
    "headless.scheduler.tick": SLODefinition(0.99, 1_000.0),
    "voice.reaction": SLODefinition(0.95, 200.0),
    "voice.semantic": SLODefinition(0.95, 3_000.0),
}


class ObservabilityStore:
    """Thread-safe golden metrics and bounded metadata-only traces."""

    def __init__(
        self,
        *,
        trace_limit: int = 256,
        samples_per_operation: int = 256,
        max_operations: int = 64,
        slos: dict[str, SLODefinition] | None = None,
    ) -> None:
        self.trace_limit = min(max(int(trace_limit), 16), 1_024)
        self.samples_per_operation = min(max(int(samples_per_operation), 16), 4_096)
        self.max_operations = min(max(int(max_operations), 8), 64)
        self.slos = dict(slos or DEFAULT_SLOS)
        self._metrics: dict[str, _MetricBucket] = {}
        self._slo_samples: dict[str, deque[bool]] = {
            name: deque(maxlen=self.samples_per_operation) for name in self.slos
        }
        self._traces: deque[dict[str, Any]] = deque(maxlen=self.trace_limit)
        self._lock = threading.Lock()
        self._logger = logging.getLogger("weaver.observability")

    def _operation(self, value: str) -> str:
        candidate = str(value or "").lower()[:80]
        if not OPERATION_PATTERN.fullmatch(candidate):
            candidate = "runtime.other"
        with self._lock:
            if candidate not in self._metrics:
                reserve_other = "runtime.other" not in self._metrics
                if len(self._metrics) >= self.max_operations or (
                    candidate != "runtime.other"
                    and reserve_other
                    and len(self._metrics) >= self.max_operations - 1
                ):
                    candidate = "runtime.other"
                self._metrics.setdefault(
                    candidate,
                    _MetricBucket(durations=deque(maxlen=self.samples_per_operation)),
                )
        return candidate

    def begin(
        self,
        operation: str,
        *,
        correlation: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> SpanToken:
        normalized = self._operation(operation)
        token = SpanToken(
            trace_id=f"trc-{uuid.uuid4().hex[:24]}",
            correlation_id=correlation_id(correlation or CURRENT_CORRELATION_ID.get("")),
            operation=normalized,
            started_perf=time.perf_counter(),
            started_at=datetime.now(timezone.utc),
            attributes=_safe_attributes(attributes),
        )
        with self._lock:
            bucket = self._metrics.setdefault(
                normalized,
                _MetricBucket(durations=deque(maxlen=self.samples_per_operation)),
            )
            bucket.in_flight += 1
            bucket.max_in_flight = max(bucket.max_in_flight, bucket.in_flight)
        return token

    def end(
        self,
        token: SpanToken,
        *,
        outcome: str,
        result_code: int | None = None,
        duration_ms: float | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        allowed_outcomes = {"success", "client-error", "server-error", "cancelled", "rejected"}
        safe_outcome = outcome if outcome in allowed_outcomes else "server-error"
        elapsed = (
            (time.perf_counter() - token.started_perf) * 1_000
            if duration_ms is None
            else float(duration_ms)
        )
        elapsed = min(max(elapsed if math.isfinite(elapsed) else 600_000.0, 0.0), 600_000.0)
        merged_attributes = {**token.attributes, **_safe_attributes(attributes)}
        trace = {
            "trace_id": token.trace_id,
            "correlation_id": token.correlation_id,
            "operation": token.operation,
            "started_at": token.started_at,
            "duration_ms": round(elapsed, 1),
            "outcome": safe_outcome,
            "result_code": (
                min(max(int(result_code), 0), 4_999)
                if isinstance(result_code, int) and not isinstance(result_code, bool)
                else None
            ),
            "attributes": merged_attributes,
        }
        with self._lock:
            bucket = self._metrics.setdefault(
                token.operation,
                _MetricBucket(durations=deque(maxlen=self.samples_per_operation)),
            )
            bucket.in_flight = max(0, bucket.in_flight - 1)
            bucket.requests += 1
            bucket.durations.append(elapsed)
            if safe_outcome == "server-error":
                bucket.server_errors += 1
            elif safe_outcome == "cancelled":
                bucket.cancelled += 1
            elif safe_outcome in {"client-error", "rejected"}:
                bucket.client_errors += 1
                bucket.successes += 1
            else:
                bucket.successes += 1
            definition = self.slos.get(token.operation)
            if definition is not None and safe_outcome != "cancelled":
                good = safe_outcome != "server-error" and (
                    definition.latency_target_ms is None
                    or elapsed <= definition.latency_target_ms
                )
                self._slo_samples[token.operation].append(good)
            self._traces.append(trace)

        log_payload = {
            "event": "runtime-span",
            "trace_id": trace["trace_id"],
            "correlation_id": trace["correlation_id"],
            "operation": trace["operation"],
            "duration_ms": trace["duration_ms"],
            "outcome": trace["outcome"],
            "result_code": trace["result_code"],
            "attributes": trace["attributes"],
        }
        encoded = json.dumps(log_payload, separators=(",", ":"), sort_keys=True)
        if safe_outcome == "server-error":
            self._logger.warning(encoded)
        else:
            self._logger.debug(encoded)

    def record(
        self,
        operation: str,
        *,
        duration_ms: float,
        outcome: str = "success",
        result_code: int | None = None,
        correlation: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        token = self.begin(
            operation,
            correlation=correlation,
            attributes=attributes,
        )
        self.end(
            token,
            outcome=outcome,
            result_code=result_code,
            duration_ms=duration_ms,
        )

    def snapshot(self, *, voice_slo: dict[str, Any]) -> ObservabilityReport:
        with self._lock:
            metrics_copy = {
                name: {
                    "requests": bucket.requests,
                    "successes": bucket.successes,
                    "server_errors": bucket.server_errors,
                    "client_errors": bucket.client_errors,
                    "cancelled": bucket.cancelled,
                    "in_flight": bucket.in_flight,
                    "max_in_flight": bucket.max_in_flight,
                    "durations": list(bucket.durations),
                }
                for name, bucket in self._metrics.items()
            }
            samples_copy = {
                name: list(samples) for name, samples in self._slo_samples.items()
            }
            traces = list(self._traces)[-32:]

        metrics = []
        for name in sorted(metrics_copy):
            values = metrics_copy[name]
            requests = int(values["requests"])
            metrics.append({
                "operation": name,
                "requests": requests,
                "successes": int(values["successes"]),
                "server_errors": int(values["server_errors"]),
                "client_errors": int(values["client_errors"]),
                "cancelled": int(values["cancelled"]),
                "in_flight": int(values["in_flight"]),
                "max_in_flight": int(values["max_in_flight"]),
                "success_rate": (
                    round(int(values["successes"]) / requests, 4) if requests else None
                ),
                "duration_p50_ms": _percentile(values["durations"], 0.50),
                "duration_p95_ms": _percentile(values["durations"], 0.95),
            })

        budgets = []
        for name, definition in self.slos.items():
            samples = samples_copy.get(name, [])
            total = len(samples)
            good = sum(1 for sample in samples if sample)
            bad = total - good
            if total:
                allowed_bad = max(1, math.ceil(total * (1.0 - definition.objective)))
                remaining = max(0.0, (allowed_bad - bad) / allowed_bad * 100.0)
                success_rate = good / total
                bad_rate = bad / total
                burn_rate = bad_rate / max(1.0 - definition.objective, 0.0001)
                status = (
                    "exhausted" if remaining <= 0
                    else ("watch" if burn_rate >= 0.5 or remaining < 50 else "healthy")
                )
            else:
                remaining = 100.0
                success_rate = None
                burn_rate = 0.0
                status = "no-data"
            budgets.append({
                "operation": name,
                "objective": definition.objective,
                "latency_target_ms": definition.latency_target_ms,
                "samples": total,
                "good": good,
                "bad": bad,
                "success_rate": round(success_rate, 4) if success_rate is not None else None,
                "error_budget_remaining_pct": round(remaining, 1),
                "burn_rate": round(min(max(burn_rate, 0.0), 10_000.0), 3),
                "status": status,
            })

        safe_voice = {
            "status": str(voice_slo.get("status") or "no-data"),
            "samples": int(voice_slo.get("samples") or 0),
            "success_rate": voice_slo.get("success_rate"),
            "error_budget_remaining_pct": float(
                voice_slo.get("error_budget_remaining_pct", 100.0)
            ),
            "reaction_target_ms": int(voice_slo.get("reaction_target_ms") or 200),
            "queue_target_ms": int(voice_slo.get("queue_target_ms") or 120),
            "semantic_target_ms": int(voice_slo.get("semantic_target_ms") or 3_000),
            "reaction_p95_ms": voice_slo.get("reaction_p95_ms"),
            "queue_p95_ms": voice_slo.get("queue_p95_ms"),
            "semantic_p95_ms": voice_slo.get("semantic_p95_ms"),
        }
        return ObservabilityReport.model_validate({
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc),
            "retention_traces": self.trace_limit,
            "retention_samples_per_operation": self.samples_per_operation,
            "total_requests": sum(int(values["requests"]) for values in metrics_copy.values()),
            "total_server_errors": sum(
                int(values["server_errors"]) for values in metrics_copy.values()
            ),
            "current_in_flight": sum(
                int(values["in_flight"]) for values in metrics_copy.values()
            ),
            "metrics": metrics,
            "error_budgets": budgets,
            "voice_slo": safe_voice,
            "recent_traces": traces,
        })


OBSERVABILITY = ObservabilityStore()


class ObservabilityMiddleware:
    """Correlate and time HTTP/WebSocket traffic without reading payloads."""

    def __init__(self, app: Any, *, store: ObservabilityStore | None = None) -> None:
        self.app = app
        self.store = store or OBSERVABILITY

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        scope_type = str(scope.get("type") or "")
        if scope_type not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in list(scope.get("headers") or [])
        }
        request_id = correlation_id(headers.get("x-correlation-id", ""))
        scope.setdefault("state", {})["correlation_id"] = request_id
        raw_path = str(scope.get("path") or "/")
        route = normalized_route(raw_path)
        emit_correlation_header = bool(headers.get("x-correlation-id")) or (
            raw_path.startswith("/headless/v2/")
            or raw_path in {
                "/health/live",
                "/health/ready",
                "/health/deep",
                "/health/observability",
            }
        )
        method = str(scope.get("method") or "connect").upper()
        operation = operation_for(scope_type, method, route)
        context_token = bind_correlation(request_id)
        span = self.store.begin(
            operation,
            correlation=request_id,
            attributes={"method": method, "route": route},
        )
        result_code = 500 if scope_type == "http" else 1011

        async def observed_send(message: dict[str, Any]) -> None:
            nonlocal result_code
            message_type = message.get("type")
            if message_type == "http.response.start":
                result_code = int(message.get("status") or 500)
                response_headers = list(message.get("headers") or [])
                if emit_correlation_header and not any(
                    key.lower() == b"x-correlation-id" for key, _ in response_headers
                ):
                    response_headers.append((b"x-correlation-id", request_id.encode("ascii")))
                message["headers"] = response_headers
            elif message_type == "websocket.close":
                result_code = int(message.get("code") or 1000)
            await send(message)

        try:
            await self.app(scope, receive, observed_send)
        except asyncio.CancelledError:
            self.store.end(span, outcome="cancelled", result_code=result_code)
            raise
        except Exception:
            self.store.end(span, outcome="server-error", result_code=result_code)
            raise
        else:
            if scope_type == "http":
                outcome = (
                    "server-error" if result_code >= 500
                    else ("rejected" if result_code in {401, 403, 409, 429}
                          else ("client-error" if result_code >= 400 else "success"))
                )
            else:
                outcome = "server-error" if result_code >= 1011 else "success"
            self.store.end(span, outcome=outcome, result_code=result_code)
        finally:
            reset_correlation(context_token)
