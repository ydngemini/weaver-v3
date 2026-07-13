#!/usr/bin/env python3
"""Bounded WebSocket transport for Weaver's public headless state.

This module can relay state and submit an already-signed Intent Capsule for
verification/evaluation.  It deliberately has no callback capable of applying
an action to Weaver's body, room, shell, filesystem, or process environment.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from fastapi import WebSocket
from pydantic import TypeAdapter, ValidationError
from starlette.websockets import WebSocketDisconnect

from headless_schemas import (
    HEADLESS_SCHEMA_VERSION,
    CapsuleReceiptMessage,
    CapsuleSubmitMessage,
    ClientMessage,
    DeltaMessage,
    HeadlessSnapshot,
    HeartbeatMessage,
    HelloMessage,
    PingMessage,
    PublicErrorMessage,
    ResumeMessage,
    SnapshotMessage,
    SubscribeMessage,
)
from headless_state import HeadlessStateStore
from observability_runtime import correlation_id as normalized_correlation_id


MAX_CLIENT_MESSAGE_BYTES = 32_768
MAX_SERVER_MESSAGE_BYTES = 65_536
MAX_PENDING_MESSAGES = 64
MAX_INVALID_MESSAGES = 3
HEARTBEAT_INTERVAL_MS = 10_000
STATE_WAIT_SECONDS = 1.0

_CLIENT_MESSAGE_ADAPTER = TypeAdapter(ClientMessage)


class TransportBackpressure(RuntimeError):
    """A slow client exhausted its bounded outbound queue."""


class TransportAuthenticationExpired(RuntimeError):
    """The short-lived browser session no longer validates."""


class CapsuleEvaluationFailure(RuntimeError):
    """Stable failure raised by the existing capsule evaluation adapter."""

    def __init__(self, code: str, *, retryable: bool) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class CapsuleReplayGuard:
    """Process-local replay protection bounded by capsule expiry and capacity."""

    def __init__(self, *, max_entries: int = 2_048) -> None:
        self.max_entries = min(max(int(max_entries), 64), 16_384)
        self._lock = asyncio.Lock()
        self._claimed: dict[str, int] = {}

    async def claim(self, capsule_id: str, expires_at_ms: int, *, now_ms: int | None = None) -> bool:
        current = int(now_ms if now_ms is not None else time.time() * 1_000)
        async with self._lock:
            self._claimed = {
                key: expiry for key, expiry in self._claimed.items() if expiry >= current
            }
            if capsule_id in self._claimed or expires_at_ms < current:
                return False
            if len(self._claimed) >= self.max_entries:
                oldest = min(self._claimed, key=self._claimed.get)
                self._claimed.pop(oldest, None)
            self._claimed[capsule_id] = expires_at_ms
            return True

    async def size(self) -> int:
        async with self._lock:
            return len(self._claimed)


@dataclass
class _ConnectionState:
    subscriptions: set[str] = field(
        default_factory=lambda: {"state", "progress", "capsule_receipts"}
    )
    last_revision: int = 0
    invalid_messages: int = 0
    close_code: int = 1000


CapsuleVerifier = Callable[[dict[str, Any]], bool]
CapsuleEvaluator = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
SessionRevalidator = Callable[[], Awaitable[bool]]


class HeadlessTransport:
    """Serve one authenticated, accepted WebSocket connection."""

    def __init__(
        self,
        store: HeadlessStateStore,
        *,
        verify_capsule: CapsuleVerifier,
        evaluate_capsule: CapsuleEvaluator,
        replay_guard: CapsuleReplayGuard,
        revalidate_session: SessionRevalidator | None = None,
        correlation_id: str = "",
        heartbeat_interval_ms: int = HEARTBEAT_INTERVAL_MS,
        max_pending_messages: int = MAX_PENDING_MESSAGES,
    ) -> None:
        self.store = store
        self.verify_capsule = verify_capsule
        self.evaluate_capsule = evaluate_capsule
        self.replay_guard = replay_guard
        self.revalidate_session = revalidate_session
        self.correlation_id = normalized_correlation_id(correlation_id)
        self.heartbeat_interval_ms = min(max(int(heartbeat_interval_ms), 1_000), 10_000)
        self.max_pending_messages = min(
            max(int(max_pending_messages), 4), MAX_PENDING_MESSAGES
        )

    def _correlation_id(self) -> str:
        return self.correlation_id

    def _error(self, code: str, *, retryable: bool) -> PublicErrorMessage:
        return PublicErrorMessage(
            code=code,
            retryable=retryable,
            correlation_id=self._correlation_id(),
        )

    @staticmethod
    def _payload(message: Any) -> dict[str, Any]:
        payload = message.model_dump(mode="json")
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if len(encoded) > MAX_SERVER_MESSAGE_BYTES:
            raise TransportBackpressure("outbound state event exceeds its hard budget")
        return payload

    async def evaluate_submission(
        self,
        message: CapsuleSubmitMessage,
    ) -> CapsuleReceiptMessage | PublicErrorMessage:
        """Verify, replay-guard, and evaluate; never apply capsule actions."""

        capsule = message.capsule.model_dump(mode="json")
        now_ms = int(time.time() * 1_000)
        if int(capsule["expires_at_ms"]) < now_ms:
            return self._error("capsule-expired", retryable=False)
        if not self.verify_capsule(capsule):
            return self._error("capsule-invalid", retryable=False)
        if not await self.replay_guard.claim(
            str(capsule["capsule_id"]),
            int(capsule["expires_at_ms"]),
            now_ms=now_ms,
        ):
            return self._error("capsule-replayed", retryable=False)
        try:
            evaluation = await self.evaluate_capsule(capsule)
        except CapsuleEvaluationFailure as exc:
            return self._error(exc.code, retryable=exc.retryable)
        except Exception:
            return self._error("service-unavailable", retryable=True)
        decision = {
            "execute": "allow",
            "approve": "allow",
            "revise": "revise",
            "block": "block",
        }.get(str(evaluation.get("decision") or "").lower(), "block")
        return CapsuleReceiptMessage(
            capsule_id=str(capsule["capsule_id"]),
            status="evaluated",
            decision=decision,
            correlation_id=self._correlation_id(),
        )

    async def _enqueue(self, queue: asyncio.Queue[dict[str, Any]], message: Any) -> None:
        payload = self._payload(message)
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull as exc:
            raise TransportBackpressure("client outbound queue is full") from exc

    async def _writer(
        self,
        websocket: WebSocket,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        while True:
            payload = await queue.get()
            try:
                await websocket.send_json(payload)
            finally:
                queue.task_done()

    async def _heartbeat(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        connection: _ConnectionState,
    ) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval_ms / 1_000)
            if self.revalidate_session is not None and not await self.revalidate_session():
                raise TransportAuthenticationExpired("headless session expired")
            await self._enqueue(
                queue,
                HeartbeatMessage(
                    sent_at=datetime.now(timezone.utc),
                    revision=self.store.revision,
                ),
            )

    async def _state_pump(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        connection: _ConnectionState,
    ) -> None:
        while True:
            observed_revision = connection.last_revision
            snapshot = await self.store.wait_for_revision(
                observed_revision,
                timeout=STATE_WAIT_SECONDS,
            )
            if snapshot is None:
                continue
            if "state" not in connection.subscriptions:
                connection.last_revision = snapshot.revision
                continue
            deltas = await self.store.changes_since(observed_revision)
            if deltas is None or len(deltas) > 8:
                await self._enqueue(queue, SnapshotMessage(snapshot=snapshot))
            else:
                for delta in deltas:
                    await self._enqueue(queue, DeltaMessage(delta=delta))
            connection.last_revision = snapshot.revision

    async def _resume(
        self,
        message: ResumeMessage,
        queue: asyncio.Queue[dict[str, Any]],
        connection: _ConnectionState,
    ) -> None:
        deltas = await self.store.changes_since(message.revision)
        snapshot = await self.store.snapshot()
        if deltas is None:
            await self._enqueue(
                queue,
                self._error("state-resync-required", retryable=True),
            )
            if snapshot is not None:
                await self._enqueue(queue, SnapshotMessage(snapshot=snapshot))
                connection.last_revision = snapshot.revision
            return
        for delta in deltas:
            await self._enqueue(queue, DeltaMessage(delta=delta))
        connection.last_revision = (
            deltas[-1].revision if deltas else (snapshot.revision if snapshot else message.revision)
        )

    async def _handle_client_message(
        self,
        message: ClientMessage,
        queue: asyncio.Queue[dict[str, Any]],
        connection: _ConnectionState,
    ) -> None:
        if isinstance(message, SubscribeMessage):
            had_state = "state" in connection.subscriptions
            connection.subscriptions = set(message.channels)
            if "state" in connection.subscriptions and not had_state:
                snapshot = await self.store.snapshot()
                if snapshot is not None:
                    await self._enqueue(queue, SnapshotMessage(snapshot=snapshot))
                    connection.last_revision = snapshot.revision
            return
        if isinstance(message, ResumeMessage):
            await self._resume(message, queue, connection)
            return
        if isinstance(message, PingMessage):
            await self._enqueue(
                queue,
                HeartbeatMessage(
                    sent_at=datetime.now(timezone.utc),
                    revision=self.store.revision,
                ),
            )
            return
        if isinstance(message, CapsuleSubmitMessage):
            receipt = await self.evaluate_submission(message)
            await self._enqueue(queue, receipt)

    async def _receiver(
        self,
        websocket: WebSocket,
        queue: asyncio.Queue[dict[str, Any]],
        connection: _ConnectionState,
    ) -> None:
        while True:
            event = await websocket.receive()
            if event.get("type") == "websocket.disconnect":
                return
            raw = event.get("text")
            if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_CLIENT_MESSAGE_BYTES:
                connection.invalid_messages += 1
                await self._enqueue(queue, self._error("invalid-message", retryable=False))
            else:
                try:
                    payload = json.loads(raw)
                    message = _CLIENT_MESSAGE_ADAPTER.validate_python(payload)
                except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
                    connection.invalid_messages += 1
                    await self._enqueue(queue, self._error("invalid-message", retryable=False))
                else:
                    connection.invalid_messages = 0
                    await self._handle_client_message(message, queue, connection)
            if connection.invalid_messages >= MAX_INVALID_MESSAGES:
                connection.close_code = 1008
                return

    async def serve(self, websocket: WebSocket) -> None:
        """Serve after authentication and ``websocket.accept()`` have succeeded."""

        snapshot: HeadlessSnapshot | None = await self.store.snapshot()
        if snapshot is None:
            with contextlib.suppress(Exception):
                await websocket.close(code=1013)
            return
        connection = _ConnectionState(last_revision=snapshot.revision)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self.max_pending_messages
        )
        await self._enqueue(
            queue,
            HelloMessage(
                schema_version=HEADLESS_SCHEMA_VERSION,
                correlation_id=self.correlation_id,
                heartbeat_interval_ms=self.heartbeat_interval_ms,
                revision=snapshot.revision,
            ),
        )
        await self._enqueue(queue, SnapshotMessage(snapshot=snapshot))
        tasks = {
            asyncio.create_task(self._writer(websocket, queue)),
            asyncio.create_task(self._heartbeat(queue, connection)),
            asyncio.create_task(self._state_pump(queue, connection)),
            asyncio.create_task(self._receiver(websocket, queue, connection)),
        }
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exception = task.exception()
                if isinstance(exception, TransportBackpressure):
                    connection.close_code = 1013
                elif isinstance(exception, TransportAuthenticationExpired):
                    connection.close_code = 1008
                elif exception is not None and not isinstance(exception, WebSocketDisconnect):
                    connection.close_code = 1011
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await task
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            with contextlib.suppress(Exception):
                await websocket.close(code=connection.close_code)
