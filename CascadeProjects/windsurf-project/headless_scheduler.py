#!/usr/bin/env python3
"""Deterministic lifecycle for Weaver's private headless cadence.

The scheduler decides *when* trusted callbacks may run.  It has no model,
prompt, transport, Intent Capsule, or action execution authority.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable


AsyncReasonCallback = Callable[[str], Awaitable[str]]
AsyncIdleCallback = Callable[[], Awaitable[bool]]
AsyncTickCallback = Callable[[float], Awaitable[None]]
AsyncErrorCallback = Callable[[Exception], Awaitable[None]]


@dataclass(frozen=True)
class HeadlessSchedule:
    thought_seconds: float
    dream_seconds: float
    tick_seconds: float = 5.0
    disabled_seconds: float = 30.0
    jitter_ratio: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "thought_seconds": self.thought_seconds,
            "dream_seconds": self.dream_seconds,
            "tick_seconds": self.tick_seconds,
            "disabled_seconds": self.disabled_seconds,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in values.values()
        ):
            raise ValueError("headless schedule intervals must be finite and positive")
        if (
            isinstance(self.jitter_ratio, bool)
            or not isinstance(self.jitter_ratio, (int, float))
            or not math.isfinite(float(self.jitter_ratio))
            or not 0 <= float(self.jitter_ratio) <= 0.25
        ):
            raise ValueError("headless jitter ratio must be within 0..0.25")


@dataclass(frozen=True)
class HeadlessTokenBudget:
    thought_tokens: int
    dream_tokens: int
    tokens_per_hour: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.thought_tokens, bool)
            or isinstance(self.dream_tokens, bool)
            or isinstance(self.tokens_per_hour, bool)
            or not 1 <= int(self.thought_tokens) <= 4_096
            or not 1 <= int(self.dream_tokens) <= 4_096
            or not max(self.thought_tokens, self.dream_tokens) <= int(self.tokens_per_hour) <= 1_000_000
        ):
            raise ValueError("headless token budget is invalid")


class HeadlessScheduler:
    """Run one bounded thought/dream callback at a time behind an idle gate."""

    def __init__(
        self,
        schedule: HeadlessSchedule,
        *,
        active: Callable[[], bool],
        idle_ready: AsyncIdleCallback,
        run_thought: AsyncReasonCallback,
        run_dream: AsyncReasonCallback,
        on_tick: AsyncTickCallback,
        on_error: AsyncErrorCallback,
        token_budget: HeadlessTokenBudget | None = None,
        priority_event: asyncio.Event | None = None,
        random_unit: Callable[[], float] = random.random,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.schedule = schedule
        self._active = active
        self._idle_ready = idle_ready
        self._run_thought = run_thought
        self._run_dream = run_dream
        self._on_tick = on_tick
        self._on_error = on_error
        self._token_budget = token_budget
        self._priority_event = priority_event
        self._random_unit = random_unit
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._stop = asyncio.Event()
        started = self._monotonic()
        self._next_thought = started + self._jittered(self.schedule.thought_seconds)
        self._next_dream = started + self._jittered(self.schedule.dream_seconds)
        self._token_events: deque[tuple[float, int]] = deque()
        self._running = False
        self._ticks = 0
        self._thought_runs = 0
        self._dream_runs = 0
        self._errors = 0
        self._preemptions = 0
        self._budget_deferrals = 0

    @property
    def running(self) -> bool:
        return self._running

    def snapshot(self) -> dict[str, int | float | bool | str]:
        now = self._monotonic()
        self._prune_token_events(now)
        return {
            "status": "running" if self._running else ("stopping" if self._stop.is_set() else "idle"),
            "running": self._running,
            "ticks": self._ticks,
            "thought_runs": self._thought_runs,
            "dream_runs": self._dream_runs,
            "errors": self._errors,
            "preemptions": self._preemptions,
            "budget_deferrals": self._budget_deferrals,
            "tokens_used_last_hour": sum(cost for _, cost in self._token_events),
            "thought_seconds": float(self.schedule.thought_seconds),
            "dream_seconds": float(self.schedule.dream_seconds),
            "jitter_ratio": float(self.schedule.jitter_ratio),
        }

    def stop(self) -> None:
        self._stop.set()

    def _jittered(self, interval: float) -> float:
        unit = min(max(float(self._random_unit()), 0.0), 1.0)
        multiplier = 1.0 + (unit * 2.0 - 1.0) * float(self.schedule.jitter_ratio)
        return max(float(interval) * multiplier, 0.01)

    def _prune_token_events(self, now: float) -> None:
        while self._token_events and now - self._token_events[0][0] >= 3_600:
            self._token_events.popleft()

    def _charge_tokens(self, kind: str, now: float) -> bool:
        if self._token_budget is None:
            return True
        self._prune_token_events(now)
        cost = (
            self._token_budget.thought_tokens
            if kind == "thought"
            else self._token_budget.dream_tokens
        )
        if sum(value for _, value in self._token_events) + cost > self._token_budget.tokens_per_hour:
            self._budget_deferrals += 1
            return False
        self._token_events.append((now, cost))
        return True

    async def _run_interruptible(self, callback: AsyncReasonCallback) -> bool:
        work = asyncio.create_task(callback("headless-loop"))
        stop_wait = asyncio.create_task(self._stop.wait())
        priority_wait = (
            asyncio.create_task(self._priority_event.wait())
            if self._priority_event is not None
            else None
        )
        waiters = {work, stop_wait}
        if priority_wait is not None:
            waiters.add(priority_wait)
        try:
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if work in done:
                await work
                return True
            work.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await work
            self._preemptions += 1
            return False
        finally:
            for waiter in (stop_wait, priority_wait):
                if waiter is not None and not waiter.done():
                    waiter.cancel()
            for waiter in (stop_wait, priority_wait):
                if waiter is not None:
                    with contextlib.suppress(asyncio.CancelledError):
                        await waiter

    async def _report_error(self, exc: Exception) -> None:
        self._errors += 1
        try:
            await self._on_error(exc)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Error reporting cannot become a recursive failure loop.
            return

    async def run_cycle(self) -> float:
        """Run one scheduling decision and return the bounded next delay."""

        if not self._active():
            return float(self.schedule.disabled_seconds)
        self._ticks += 1
        await self._on_tick(self._wall_clock())
        if (
            (self._priority_event is not None and self._priority_event.is_set())
            or not await self._idle_ready()
        ):
            return float(self.schedule.tick_seconds)

        now = self._monotonic()
        if now >= self._next_thought:
            # Admission is edge-triggered: a failing route does not hot-loop.
            self._next_thought = now + self._jittered(self.schedule.thought_seconds)
            if self._charge_tokens("thought", now):
                try:
                    completed = await self._run_interruptible(self._run_thought)
                    self._thought_runs += int(completed)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._report_error(exc)
            else:
                self._next_thought = now + min(self.schedule.thought_seconds, 60.0)

        now = self._monotonic()
        if (
            not self._stop.is_set()
            and not (self._priority_event is not None and self._priority_event.is_set())
            and now >= self._next_dream
            and await self._idle_ready()
        ):
            self._next_dream = now + self._jittered(self.schedule.dream_seconds)
            if self._charge_tokens("dream", now):
                try:
                    completed = await self._run_interruptible(self._run_dream)
                    self._dream_runs += int(completed)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._report_error(exc)
            else:
                self._next_dream = now + min(self.schedule.dream_seconds, 60.0)
        return float(self.schedule.tick_seconds)

    async def run(self) -> None:
        if self._running:
            raise RuntimeError("headless scheduler is already running")
        self._running = True
        try:
            while not self._stop.is_set():
                try:
                    delay = await self.run_cycle()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._report_error(exc)
                    delay = float(self.schedule.tick_seconds)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=max(delay, 0.01))
                except TimeoutError:
                    continue
        finally:
            self._running = False
