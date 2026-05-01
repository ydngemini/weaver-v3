"""
nexus_client.py — Shared Nexus Bus Client
==========================================
Async WebSocket client that any Weaver lobe can use to register,
publish, and subscribe on the Nexus Bus (ws://localhost:9999).

Features:
  - Auto-reconnect with exponential backoff
  - Fire-and-forget publish (drops silently if disconnected)
  - Optional listener callback for subscribed topics
  - Thread-safe publish via asyncio.run_coroutine_threadsafe

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
import random
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

import websockets

log = logging.getLogger("nexus_client")

NEXUS_URL = "ws://localhost:9999"
RECONNECT_BASE = 2.0
RECONNECT_CAP = 30.0
PING_INTERVAL = 20


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

    @property
    def connected(self) -> bool:
        return self._connected and self._ws is not None

    async def connect(self) -> bool:
        try:
            self._ws = await websockets.connect(
                self.url, ping_interval=PING_INTERVAL, close_timeout=5
            )
            # Drain the SYNC frame
            try:
                sync_raw = await asyncio.wait_for(self._ws.recv(), timeout=3)
            except asyncio.TimeoutError:
                pass

            await self._ws.send(json.dumps({
                "action": "register",
                "lobe_id": self.lobe_id,
            }))
            ack = await asyncio.wait_for(self._ws.recv(), timeout=3)

            if self.topics:
                await self._ws.send(json.dumps({
                    "action": "subscribe",
                    "topics": self.topics,
                }))
                await asyncio.wait_for(self._ws.recv(), timeout=3)

            self._connected = True
            log.info("[%s] Connected to Nexus Bus at %s", self.lobe_id, self.url)

            if self.on_message and self._listen_task is None:
                self._listen_task = asyncio.create_task(self._listen_loop())

            return True

        except Exception as e:
            log.warning("[%s] Nexus Bus connect failed: %s", self.lobe_id, e)
            self._connected = False
            return False

    async def publish(self, topic: str, payload: Any) -> bool:
        if not self._connected or self._ws is None:
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
            try:
                async for raw in self._ws:
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
            except websockets.exceptions.ConnectionClosed:
                self._connected = False
                if not self._closing:
                    self._schedule_reconnect()
            except Exception:
                self._connected = False
                if not self._closing:
                    self._schedule_reconnect()
                    await asyncio.sleep(1)

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
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._reconnect_task:
            self._reconnect_task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        log.info("[%s] Disconnected from Nexus Bus.", self.lobe_id)
