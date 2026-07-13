#!/usr/bin/env python3
"""Bounded resilience primitives for Weaver's dependency and read paths."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar


T = TypeVar("T")


class CircuitOpen(RuntimeError):
    """A dependency is cooling down after its bounded failure threshold."""


@dataclass
class _CircuitState:
    failures: int = 0
    successes: int = 0
    opened_at: float | None = None
    last_failure_at: float | None = None
    last_success_at: float | None = None
    probe_active: bool = False


class AsyncCircuitBreaker:
    """Closed/open/half-open breaker with one recovery probe and no raw errors."""

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 3,
        recovery_seconds: float = 30,
        timeout_seconds: float = 120,
        clock=time.monotonic,
    ) -> None:
        self.name = str(name)[:48]
        self.failure_threshold = min(max(int(failure_threshold), 1), 20)
        self.recovery_seconds = min(max(float(recovery_seconds), 1), 600)
        self.timeout_seconds = min(max(float(timeout_seconds), 0.05), 180)
        self._clock = clock
        self._state = _CircuitState()
        self._lock = asyncio.Lock()

    async def _admit(self) -> bool:
        now = float(self._clock())
        async with self._lock:
            opened_at = self._state.opened_at
            if opened_at is None:
                return False
            if now - opened_at < self.recovery_seconds:
                raise CircuitOpen(f"{self.name} circuit is open")
            if self._state.probe_active:
                raise CircuitOpen(f"{self.name} recovery probe is active")
            self._state.probe_active = True
            return True

    async def _succeeded(self) -> None:
        async with self._lock:
            self._state.successes += 1
            self._state.failures = 0
            self._state.opened_at = None
            self._state.last_success_at = float(self._clock())
            self._state.probe_active = False

    async def _failed(self) -> None:
        now = float(self._clock())
        async with self._lock:
            self._state.failures += 1
            self._state.last_failure_at = now
            self._state.probe_active = False
            if self._state.failures >= self.failure_threshold:
                self._state.opened_at = now

    async def call(
        self,
        factory: Callable[[], Awaitable[T]],
        *,
        timeout_seconds: float | None = None,
    ) -> T:
        await self._admit()
        timeout = min(
            max(float(timeout_seconds or self.timeout_seconds), 0.05),
            self.timeout_seconds,
        )
        try:
            result = await asyncio.wait_for(factory(), timeout=timeout)
        except asyncio.CancelledError:
            async with self._lock:
                self._state.probe_active = False
            raise
        except Exception:
            await self._failed()
            raise
        await self._succeeded()
        return result

    async def snapshot(self) -> dict[str, Any]:
        now = float(self._clock())
        async with self._lock:
            if self._state.opened_at is None:
                status = "closed"
                retry_after_ms = 0
            else:
                remaining = self.recovery_seconds - (now - self._state.opened_at)
                status = "open" if remaining > 0 else "half-open"
                retry_after_ms = max(0, round(remaining * 1_000))
            return {
                "name": self.name,
                "status": status,
                "failures": self._state.failures,
                "successes": self._state.successes,
                "retry_after_ms": retry_after_ms,
                "probe_active": self._state.probe_active,
            }


class RequestCoalescer(Generic[T]):
    """Share identical safe work while shielding it from waiter cancellation."""

    def __init__(self, *, max_keys: int = 64) -> None:
        self.max_keys = min(max(int(max_keys), 1), 512)
        self._lock = asyncio.Lock()
        self._tasks: OrderedDict[str, asyncio.Task[T]] = OrderedDict()
        self.coalesced_waiters = 0

    async def run(self, key: str, factory: Callable[[], Awaitable[T]]) -> T:
        normalized = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        async with self._lock:
            self._tasks = OrderedDict(
                (stored_key, task)
                for stored_key, task in self._tasks.items()
                if not task.done()
            )
            task = self._tasks.get(normalized)
            if task is None:
                while len(self._tasks) >= self.max_keys:
                    _, oldest = self._tasks.popitem(last=False)
                    if not oldest.done():
                        oldest.cancel()
                task = asyncio.create_task(factory())
                self._tasks[normalized] = task
            else:
                self.coalesced_waiters += 1
                self._tasks.move_to_end(normalized)
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                async with self._lock:
                    if self._tasks.get(normalized) is task:
                        self._tasks.pop(normalized, None)

    async def active(self) -> int:
        async with self._lock:
            return sum(1 for task in self._tasks.values() if not task.done())


class BoundedTTLCache(Generic[T]):
    """Small copy-on-read cache for non-secret, immutable derived values."""

    def __init__(self, *, ttl_seconds: float, max_entries: int = 64, clock=time.monotonic) -> None:
        self.ttl_seconds = min(max(float(ttl_seconds), 0.05), 300)
        self.max_entries = min(max(int(max_entries), 1), 512)
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _key(self, value: str) -> str:
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()

    def get(self, key: str) -> T | None:
        digest = self._key(key)
        stored = self._entries.get(digest)
        now = float(self._clock())
        if stored is None or stored[0] <= now:
            self._entries.pop(digest, None)
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(digest)
        return deepcopy(stored[1])

    def put(self, key: str, value: T) -> None:
        digest = self._key(key)
        self._entries[digest] = (float(self._clock()) + self.ttl_seconds, deepcopy(value))
        self._entries.move_to_end(digest)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def snapshot(self) -> dict[str, int]:
        now = float(self._clock())
        self._entries = OrderedDict(
            (key, value) for key, value in self._entries.items() if value[0] > now
        )
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "max_entries": self.max_entries,
        }


def etag_for(value: Any, *, prefix: str = "weaver") -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:20]
    safe_prefix = "".join(character for character in prefix if character.isalnum() or character in "-_")[:24]
    return f'"{safe_prefix or "weaver"}-{digest}"'
