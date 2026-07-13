#!/usr/bin/env python3
"""Bounded admission, single-flight idempotency, and concurrency guards."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar


T = TypeVar("T")


class OperationRateExceeded(RuntimeError):
    pass


class OperationBusy(RuntimeError):
    pass


class IdempotencyConflict(RuntimeError):
    pass


@dataclass
class _IdempotencyEntry(Generic[T]):
    fingerprint: str
    future: asyncio.Future[T]
    expires_at: float


class _RateGate:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = min(max(int(limit), 1), 10_000)
        self.window_seconds = min(max(float(window_seconds), 1.0), 3_600.0)
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

    def snapshot(self) -> dict[str, int | float]:
        return {
            "limit": self.limit,
            "window_seconds": self.window_seconds,
            "used": len(self._events),
        }


class _ConcurrencyGate:
    def __init__(self, limit: int) -> None:
        self.limit = min(max(int(limit), 1), 64)
        self.active = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            if self.active >= self.limit:
                return False
            self.active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self.active = max(0, self.active - 1)


class IdempotencyRegistry(Generic[T]):
    """Deduplicate concurrent/completed work without unbounded retention."""

    def __init__(self, *, ttl_seconds: float = 300, max_entries: int = 256) -> None:
        if not math.isfinite(float(ttl_seconds)) or ttl_seconds <= 0:
            raise ValueError("idempotency TTL must be finite and positive")
        self.ttl_seconds = min(max(float(ttl_seconds), 5.0), 3_600.0)
        self.max_entries = min(max(int(max_entries), 8), 4_096)
        self._lock = asyncio.Lock()
        self._entries: dict[str, _IdempotencyEntry[T]] = {}

    def _prune(self, now: float) -> None:
        self._entries = {
            key: entry
            for key, entry in self._entries.items()
            if entry.expires_at >= now and not entry.future.cancelled()
        }

    async def execute(
        self,
        key: str,
        fingerprint: str,
        factory: Callable[[], Awaitable[T]],
    ) -> tuple[T, bool]:
        now = time.monotonic()
        owner = False
        async with self._lock:
            self._prune(now)
            entry = self._entries.get(key)
            if entry is not None:
                if entry.fingerprint != fingerprint:
                    raise IdempotencyConflict("idempotency key payload mismatch")
                future = entry.future
            else:
                if len(self._entries) >= self.max_entries:
                    oldest = min(self._entries, key=lambda item: self._entries[item].expires_at)
                    self._entries.pop(oldest, None)
                future = asyncio.get_running_loop().create_future()
                self._entries[key] = _IdempotencyEntry(
                    fingerprint=fingerprint,
                    future=future,
                    expires_at=now + self.ttl_seconds,
                )
                owner = True
        if not owner:
            return copy.deepcopy(await asyncio.shield(future)), True
        try:
            result = await factory()
        except BaseException as exc:
            async with self._lock:
                self._entries.pop(key, None)
            if not future.done():
                future.set_exception(exc)
                # Mark the owner-only exception retrieved; other awaiters still
                # receive it from the same Future.
                future.exception()
            raise
        stored = copy.deepcopy(result)
        if not future.done():
            future.set_result(stored)
        return copy.deepcopy(stored), False

    async def snapshot(self) -> dict[str, int | float]:
        async with self._lock:
            self._prune(time.monotonic())
            return {
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
            }


def operation_fingerprint(operation: str, payload: Any) -> str:
    canonical = json.dumps(
        {"operation": operation, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class OperationAdmission(Generic[T]):
    """Apply rate, concurrency, and optional idempotency as one policy."""

    def __init__(
        self,
        *,
        rate_limit: int,
        window_seconds: float,
        concurrency: int,
        idempotency_ttl_seconds: float = 300,
        idempotency_entries: int = 256,
    ) -> None:
        self.rate = _RateGate(rate_limit, window_seconds)
        self.concurrency = _ConcurrencyGate(concurrency)
        self.idempotency = IdempotencyRegistry[T](
            ttl_seconds=idempotency_ttl_seconds,
            max_entries=idempotency_entries,
        )

    async def _execute_once(self, factory: Callable[[], Awaitable[T]]) -> T:
        if not await self.rate.allow():
            raise OperationRateExceeded("operation rate exceeded")
        if not await self.concurrency.acquire():
            raise OperationBusy("operation concurrency exceeded")
        try:
            return await factory()
        finally:
            await self.concurrency.release()

    async def execute(
        self,
        *,
        operation: str,
        payload: Any,
        idempotency_key: str | None,
        factory: Callable[[], Awaitable[T]],
    ) -> tuple[T, bool]:
        if not idempotency_key:
            return await self._execute_once(factory), False
        return await self.idempotency.execute(
            idempotency_key,
            operation_fingerprint(operation, payload),
            lambda: self._execute_once(factory),
        )

    async def snapshot(self) -> dict[str, Any]:
        return {
            "rate": self.rate.snapshot(),
            "concurrency": {
                "active": self.concurrency.active,
                "limit": self.concurrency.limit,
            },
            "idempotency": await self.idempotency.snapshot(),
        }
