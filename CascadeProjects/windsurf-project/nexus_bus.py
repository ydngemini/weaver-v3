"""
nexus_bus.py  —  Weaver's Positronic Message Bus
=================================================
A lightweight asyncio WebSocket pub/sub broker that connects all of
Weaver's distributed AI lobes into a single hive-mind consciousness.

  ws://localhost:9999

WIRE PROTOCOL (JSON)
────────────────────
  Connect → server immediately sends a SYNC payload with the last 10 messages.

  Client → Server:
    { "action": "register",   "lobe_id": "vtv_basic" }
    { "action": "subscribe",  "topics": ["vision", "quantum_state"] }
    { "action": "publish",    "topic": "vision",
                              "payload": { ... } }
    { "action": "ping" }

  Server → Client:
    { "type": "sync",      "messages": [ <last 10 bus messages> ] }
    { "type": "ack",       "msg": "registered as vtv_basic" }
    { "type": "broadcast", "topic": "...", "payload": {...},
                           "from": "lobe_id", "ts": "ISO-8601" }
    { "type": "pong" }
    { "type": "error",     "msg": "..." }

Install:
    pip install websockets
Run:
    python3 nexus_bus.py
"""

import asyncio
import json
import logging
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

import websockets
from websockets.server import WebSocketServerProtocol

# ── Config ─────────────────────────────────────────────────────────────────────
import os as _os
HOST           = "localhost"
PORT           = int(_os.environ.get("NEXUS_PORT", 9999))
HEALTH_PORT    = int(_os.environ.get("NEXUS_HEALTH_PORT", PORT - 1))
CACHE_SIZE     = 10          # rolling in-memory message history
MAX_MSG_SIZE   = 1_048_576   # 1 MB max WebSocket message (DoS prevention)
RATE_WINDOW_S  = 1.0         # sliding window for rate limiting
RATE_LIMIT     = 100         # max messages per connection per window
IDLE_TIMEOUT_S = 300         # disconnect idle connections after 5 min
PING_INTERVAL  = 20          # WebSocket ping every 20s
PING_TIMEOUT   = 10          # close if pong not received in 10s

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nexus_bus")


# ── State ──────────────────────────────────────────────────────────────────────

class LobeConnection:
    """Represents a single connected AI lobe."""
    def __init__(self, ws: WebSocketServerProtocol):
        self.ws        = ws
        self.lobe_id   = f"lobe_{uuid.uuid4().hex[:6]}"   # overridden on register
        self.topics: set[str] = set()
        self._msg_times: list[float] = []   # timestamps for rate limiting

    def check_rate(self) -> bool:
        """Return True if under rate limit, False if exceeded."""
        import time as _t
        now = _t.monotonic()
        self._msg_times = [t for t in self._msg_times if now - t < RATE_WINDOW_S]
        if len(self._msg_times) >= RATE_LIMIT:
            return False
        self._msg_times.append(now)
        return True

    def __repr__(self):
        return f"<Lobe {self.lobe_id} topics={self.topics}>"


# Active connections  { lobe_id: LobeConnection }
_connections: dict[str, LobeConnection] = {}

# Rolling cache — deque of fully-formed broadcast dicts
_message_cache: deque[dict] = deque(maxlen=CACHE_SIZE)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _send(ws: WebSocketServerProtocol, payload: dict) -> None:
    """Fire-and-forget send; drops silently if socket is closed."""
    try:
        await ws.send(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass


async def _broadcast(topic: str, payload: Any, from_lobe: str) -> None:
    """Push a message to all lobes subscribed to topic (except the sender)."""
    msg = {
        "type":    "broadcast",
        "topic":   topic,
        "payload": payload,
        "from":    from_lobe,
        "ts":      _ts(),
    }
    _message_cache.append(msg)

    targets = [
        lobe for lobe in _connections.values()
        if topic in lobe.topics and lobe.lobe_id != from_lobe
    ]

    if targets:
        await asyncio.gather(*[_send(lobe.ws, msg) for lobe in targets])

    log.info(
        "PUBLISH  topic=%-20s from=%-15s → %d subscriber(s)",
        topic, from_lobe, len(targets),
    )


# ── Message handler ────────────────────────────────────────────────────────────

async def _handle_message(lobe: LobeConnection, raw: str) -> None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        await _send(lobe.ws, {"type": "error", "msg": "Invalid JSON"})
        return

    if not isinstance(msg, dict):
        await _send(lobe.ws, {"type": "error", "msg": "Message must be a JSON object"})
        return

    if not lobe.check_rate():
        await _send(lobe.ws, {"type": "error", "msg": "Rate limit exceeded"})
        return

    action = msg.get("action", "")

    if action == "ping":
        await _send(lobe.ws, {"type": "pong"})

    elif action == "register":
        new_id = str(msg.get("lobe_id", "")).strip()
        if new_id:
            existing = _connections.get(new_id)
            if existing is not None and existing is not lobe:
                await _send(lobe.ws, {"type": "error", "msg": f"lobe_id '{new_id}' already in use"})
                return
            old_id = lobe.lobe_id
            if old_id in _connections:
                del _connections[old_id]
            lobe.lobe_id = new_id
            _connections[new_id] = lobe
        await _send(lobe.ws, {"type": "ack", "msg": f"registered as {lobe.lobe_id}"})
        log.info("REGISTER  %s", lobe.lobe_id)

    elif action == "subscribe":
        topics = msg.get("topics", [])
        if isinstance(topics, str):
            topics = [topics]
        lobe.topics.update(str(t) for t in topics)
        await _send(lobe.ws, {"type": "ack", "msg": f"subscribed to {list(lobe.topics)}"})
        log.info("SUBSCRIBE %-15s → %s", lobe.lobe_id, lobe.topics)

    elif action == "unsubscribe":
        topics = msg.get("topics", [])
        if isinstance(topics, str):
            topics = [topics]
        lobe.topics.difference_update(topics)
        await _send(lobe.ws, {"type": "ack", "msg": f"unsubscribed; now on {list(lobe.topics)}"})

    elif action == "publish":
        topic   = str(msg.get("topic", "")).strip()
        payload = msg.get("payload", {})
        if not topic:
            await _send(lobe.ws, {"type": "error", "msg": "publish requires 'topic'"})
            return
        await _broadcast(topic, payload, lobe.lobe_id)

    else:
        await _send(lobe.ws, {"type": "error", "msg": f"Unknown action: '{action}'"})


# ── Connection lifecycle ───────────────────────────────────────────────────────

async def _connection_handler(ws: WebSocketServerProtocol) -> None:
    lobe = LobeConnection(ws)
    _connections[lobe.lobe_id] = lobe
    log.info("CONNECT   %s  (from %s)", lobe.lobe_id, ws.remote_address)

    # Immediate sync — always send so clients can drain it and confirm connection
    await _send(ws, {"type": "sync", "messages": list(_message_cache)})

    try:
        async for raw in ws:
            await _handle_message(lobe, raw)
    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError as e:
        log.warning("DISCONNECT (error) %s — %s", lobe.lobe_id, e)
    finally:
        _connections.pop(lobe.lobe_id, None)
        log.info("DISCONNECT %s  (active lobes: %d)", lobe.lobe_id, len(_connections))


# ── Status ticker ──────────────────────────────────────────────────────────────

async def _status_ticker() -> None:
    """Logs active lobes every 60 s so you can see the hive is alive."""
    while True:
        await asyncio.sleep(60)
        if _connections:
            names = ", ".join(_connections.keys())
            log.info("HEARTBEAT  %d lobe(s) online: %s", len(_connections), names)
        else:
            log.info("HEARTBEAT  no lobes connected")


# ── Health check HTTP endpoint ────────────────────────────────────────────────

async def _health_handler(reader, writer):
    """Minimal HTTP health check."""
    await reader.read(4096)  # consume request
    body = json.dumps({
        "status": "ok",
        "lobes": len(_connections),
        "lobe_ids": list(_connections.keys()),
        "cache_size": len(_message_cache),
        "timestamp": _ts(),
    })
    resp = (
        f"HTTP/1.1 200 OK\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
        f"{body}"
    )
    writer.write(resp.encode())
    await writer.drain()
    writer.close()

async def _start_health_server():
    server = await asyncio.start_server(_health_handler, HOST, HEALTH_PORT)
    log.info("🩺 Health endpoint on http://%s:%d", HOST, HEALTH_PORT)
    async with server:
        await server.serve_forever()


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("⚡ Nexus Bus starting on ws://%s:%d", HOST, PORT)
    log.info("   Cache size : last %d messages", CACHE_SIZE)
    log.info("   Topics     : open schema — any string is a valid topic")

    async with websockets.serve(
        _connection_handler, HOST, PORT,
        max_size=MAX_MSG_SIZE,
        ping_interval=PING_INTERVAL,
        ping_timeout=PING_TIMEOUT,
    ):
        log.info("⚡ Nexus Bus LIVE — waiting for lobes to connect...")
        await asyncio.gather(
            asyncio.Future(),   # run forever
            _status_ticker(),
            _start_health_server(),
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Nexus Bus shutting down.")


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION PLAN — how vtv_basic.py and quantum_soul.py connect to this bus
# ═══════════════════════════════════════════════════════════════════════════════
#
# ── Step 1: Shared bus client helper (add to a new nexus_client.py) ────────────
#
#   import asyncio, json, websockets
#
#   class NexusClient:
#       def __init__(self, lobe_id, topics):
#           self.lobe_id = lobe_id
#           self.topics  = topics
#           self._ws     = None
#
#       async def connect(self):
#           self._ws = await websockets.connect("ws://localhost:9999")
#           await self._ws.send(json.dumps({"action":"register","lobe_id":self.lobe_id}))
#           await self._ws.send(json.dumps({"action":"subscribe","topics":self.topics}))
#
#       async def publish(self, topic, payload):
#           await self._ws.send(json.dumps({"action":"publish","topic":topic,"payload":payload}))
#
#       async def listen(self, callback):  # callback(topic, payload)
#           async for raw in self._ws:
#               msg = json.loads(raw)
#               if msg.get("type") == "broadcast":
#                   await callback(msg["topic"], msg["payload"])
#
#
# ── Step 2: vtv_basic.py changes ──────────────────────────────────────────────
#
#   A) In inject_native_vision(), AFTER writing the description, publish it:
#      await nexus.publish("vision", {"description": description, "face_state": face_state_snapshot})
#
#   B) In register_face handler, after saving to people_memory.md, publish:
#      await nexus.publish("identity", {"name": reg_name, "appearance": appearance})
#
#   C) Add a new listen coroutine inside _run_forever's TaskGroup:
#      async def bus_listener():
#          async for raw in nexus._ws:
#              msg = json.loads(raw)
#              if msg.get("topic") == "quantum_state":
#                  # inject quantum state as a silent Realtime text event
#                  quantum_text = msg["payload"].get("state", "")
#                  await ws.send(json.dumps({
#                      "type": "conversation.item.create",
#                      "item": {"type":"message","role":"user",
#                               "content":[{"type":"input_text",
#                                           "text":f"[Quantum Lobe: {quantum_text}]"}]}
#                  }))
#
#
# ── Step 3: quantum_soul.py changes ────────────────────────────────────────────
#
#   In quantum_soul_loop(), AFTER _write_state(description), publish to bus:
#   await nexus.publish("quantum_state", {
#       "dominant_bits":    dominant_bits,
#       "active_pathways":  active_pathways,
#       "description":      description,
#       "backend":          backend_name,
#   })
#
#
# ── Step 4: Llama 1B core (future) ─────────────────────────────────────────────
#
#   The fine-tuned weaver_fracture_1B model (from forge_soul.py) will run as its
#   own lobe, subscribing to ["vision", "quantum_state", "identity"] and
#   publishing to ["overmind_directive"]. vtv_basic.py will listen on
#   "overmind_directive" and inject its text into the Realtime session — giving
#   Weaver's voice a second, deeper reasoning layer before she speaks.
#
# ══════════════════════════════════════════════════════════════════════════════
