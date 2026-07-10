"""
nexus_client.py — Shared Nexus Bus Client
==========================================
Async WebSocket client that any Weaver lobe can use to register,
publish, and subscribe on the Nexus Bus (ws://localhost:9999).

Features:
  - Auto-reconnect with exponential backoff
  - Publish reconnects on demand and reports delivery-to-socket success
  - Optional listener callback for subscribed topics
  - Validated synchronous one-shot publish for threaded services

Usage:
    from nexus_client import NexusClient

    client = NexusClient("quantum_soul", ["quantum_state"])
    await client.connect()
    await client.publish("quantum_state", {"description": "...", "pathway": "Void"})
    await client.close()
"""

import asyncio
import json
import logging
import os
import random
from typing import Any, Callable, Coroutine, Dict, List, Optional

import websockets

log = logging.getLogger("nexus_client")

NEXUS_URL = os.environ.get("NEXUS_BUS_URL", "ws://localhost:9999")
RECONNECT_BASE = 2.0
RECONNECT_CAP = 30.0
PING_INTERVAL = 20


def publish_once(
    lobe_id: str,
    topic: str,
    payload: Any,
    url: str = NEXUS_URL,
    timeout: float = 5.0,
) -> bool:
    """Publish one event from synchronous code using a full WS handshake."""
    from websockets.sync.client import connect as sync_connect

    with sync_connect(url, open_timeout=timeout, close_timeout=timeout) as ws:
        sync = json.loads(ws.recv(timeout=timeout))
        if sync.get("type") != "sync":
            raise RuntimeError(f"Nexus sync failed: {sync.get('msg', sync)}")

        ws.send(json.dumps({"action": "register", "lobe_id": lobe_id}))
        ack = json.loads(ws.recv(timeout=timeout))
        if ack.get("type") != "ack":
            raise RuntimeError(f"Nexus registration failed: {ack.get('msg', ack)}")

        ws.send(json.dumps({
            "action": "publish",
            "topic": topic,
            "payload": payload,
        }))
    return True


class NexusClient:
    """Async WebSocket client for the Weaver Nexus Bus."""

    def __init__(
        self,
        lobe_id: str,
        topics: Optional[List[str]] = None,
        url: str = NEXUS_URL,
        on_message: Optional[Callable[[str, Dict, str], Coroutine]] = None,
    ):
        self.lobe_id = lobe_id
        self.topics = topics or []
        self.url = url
        self.on_message = on_message
        self._ws = None
        self._connected = False
        self._closing = False
        self._reconnect_task = None
        self._listen_task = None
        self._connect_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._connected and self._ws is not None

    async def connect(self) -> bool:
        if self._closing:
            return False

        async with self._connect_lock:
            if self.connected:
                return True

            ws = None
            try:
                ws = await websockets.connect(
                    self.url, ping_interval=PING_INTERVAL, close_timeout=5
                )
                sync = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
                if sync.get("type") != "sync":
                    raise RuntimeError(f"unexpected Nexus sync frame: {sync}")

                await ws.send(json.dumps({
                    "action": "register",
                    "lobe_id": self.lobe_id,
                }))
                ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
                if ack.get("type") != "ack":
                    raise RuntimeError(ack.get("msg", f"registration rejected: {ack}"))

                if self.topics:
                    await ws.send(json.dumps({
                        "action": "subscribe",
                        "topics": self.topics,
                    }))
                    sub_ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
                    if sub_ack.get("type") != "ack":
                        raise RuntimeError(sub_ack.get("msg", f"subscription rejected: {sub_ack}"))

                self._ws = ws
                self._connected = True
                log.info("[%s] Connected to Nexus Bus at %s", self.lobe_id, self.url)

                # Always drain server frames, even without a message callback.
                if self._listen_task is None or self._listen_task.done():
                    self._listen_task = asyncio.create_task(self._listen_loop())

                return True

            except Exception as e:
                log.warning("[%s] Nexus Bus connect failed: %s", self.lobe_id, e)
                self._connected = False
                if self._ws is ws:
                    self._ws = None
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                if not self._closing:
                    self._schedule_reconnect()
                return False

    async def publish(self, topic: str, payload: Any) -> bool:
        if not self._connected or self._ws is None:
            if self._closing or not await self.connect():
                return False
        try:
            await self._ws.send(json.dumps({
                "action": "publish",
                "topic": topic,
                "payload": payload,
            }))
            return True
        except Exception:
            self._connected = False
            self._schedule_reconnect()
            return False

    async def _listen_loop(self):
        while not self._closing:
            if not self._connected or self._ws is None:
                await asyncio.sleep(1)
                continue
            ws = self._ws
            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("type") == "broadcast" and self.on_message:
                        try:
                            await self.on_message(
                                msg.get("topic", ""),
                                msg.get("payload", {}),
                                msg.get("from", ""),
                            )
                        except Exception as e:
                            log.warning("[%s] on_message error: %s", self.lobe_id, e)
            except asyncio.CancelledError:
                raise
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                log.warning("[%s] Nexus Bus listener failed: %s", self.lobe_id, e)

            # A normal WebSocket close ends ``async for`` without raising. Mark
            # it disconnected here as well; otherwise this loop spins forever.
            if self._ws is ws:
                self._connected = False
                self._ws = None
                if not self._closing:
                    self._schedule_reconnect()
            await asyncio.sleep(0.1)

    def _schedule_reconnect(self):
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self):
        delay = RECONNECT_BASE
        while not self._closing and not self._connected:
            jitter = random.uniform(0, delay * 0.3)
            log.info("[%s] Reconnecting in %.1fs...", self.lobe_id, delay + jitter)
            await asyncio.sleep(delay + jitter)
            if await self.connect():
                return
            delay = min(delay * 2, RECONNECT_CAP)

    async def close(self):
        self._closing = True
        self._connected = False
        current = asyncio.current_task()
        if self._listen_task:
            if self._listen_task is not current:
                self._listen_task.cancel()
                try:
                    await self._listen_task
                except (asyncio.CancelledError, Exception):
                    pass
        if self._reconnect_task:
            if self._reconnect_task is not current:
                self._reconnect_task.cancel()
                try:
                    await self._reconnect_task
                except (asyncio.CancelledError, Exception):
                    pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        log.info("[%s] Disconnected from Nexus Bus.", self.lobe_id)
