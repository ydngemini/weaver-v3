#!/usr/bin/env python3
"""Cancellable turn registry and privacy-safe framing for headless chat."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any


MAX_ACTIVE_CHAT_TURNS = 4
MAX_STREAM_EVENT_BYTES = 8_192
MAX_DELTA_CHARS = 192


class ChatTurnBusy(RuntimeError):
    """Raised when the bounded interactive turn registry is full."""


@dataclass
class _ActiveTurn:
    task: asyncio.Task[Any]
    cancelled: asyncio.Event


class ChatTurnRegistry:
    """Track cancellable server tasks without retaining prompts or answers."""

    def __init__(self, *, max_active: int = MAX_ACTIVE_CHAT_TURNS) -> None:
        self.max_active = min(max(int(max_active), 1), 16)
        self._lock = asyncio.Lock()
        self._turns: dict[str, _ActiveTurn] = {}

    async def register(self, turn_id: str, task: asyncio.Task[Any]) -> asyncio.Event:
        async with self._lock:
            if turn_id in self._turns or len(self._turns) >= self.max_active:
                raise ChatTurnBusy("interactive turn capacity reached")
            cancelled = asyncio.Event()
            self._turns[turn_id] = _ActiveTurn(task=task, cancelled=cancelled)
            return cancelled

    async def cancel(self, turn_id: str) -> bool:
        async with self._lock:
            turn = self._turns.get(turn_id)
            if turn is None:
                return False
            turn.cancelled.set()
            if not turn.task.done():
                turn.task.cancel()
            return True

    async def forget(self, turn_id: str, task: asyncio.Task[Any]) -> None:
        async with self._lock:
            turn = self._turns.get(turn_id)
            if turn is not None and turn.task is task:
                self._turns.pop(turn_id, None)

    async def active(self) -> int:
        async with self._lock:
            return len(self._turns)


def public_stream_chunks(value: str, *, max_chars: int = MAX_DELTA_CHARS) -> list[str]:
    """Split an already-approved Weaver answer without rewriting its content."""

    text = str(value or "")
    limit = min(max(int(max_chars), 32), MAX_DELTA_CHARS)
    if not text:
        return []
    chunks: list[str] = []
    cursor = 0
    while cursor < len(text):
        boundary = min(cursor + limit, len(text))
        if boundary < len(text):
            preferred = max(
                text.rfind("\n", cursor + 1, boundary + 1),
                text.rfind(" ", cursor + 1, boundary + 1),
            )
            if preferred > cursor + limit // 3:
                boundary = preferred + (1 if text[preferred] == " " else 0)
        chunk = text[cursor:boundary]
        if chunk:
            chunks.append(chunk)
        cursor = boundary
    return chunks


def sse_event(payload: dict[str, Any]) -> bytes:
    """Encode one compact SSE data event under its hard public size budget."""

    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    encoded = f"data: {body}\n\n".encode("utf-8")
    if len(encoded) > MAX_STREAM_EVENT_BYTES:
        raise ValueError("chat stream event exceeds its hard size budget")
    return encoded
