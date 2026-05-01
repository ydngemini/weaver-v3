#!/usr/bin/env python3
"""
twilio_bridge.py — Weaver Telephony Lobe
=========================================
Bridges Twilio phone calls to OpenAI Realtime API via WebSockets.
Acts as a middleman: Twilio ↔ this server ↔ OpenAI Realtime.

Audio format: 8kHz G.711 μ-law (Twilio native) passed through directly
to OpenAI Realtime configured with g711_ulaw input/output.

Preserves Weaver's LangChain memory cortex (lc_summary, lc_history,
brain_queue) and connects to NexusBus for transcript publishing.

Architecture:
    Phone → Twilio → [WS] → twilio_bridge → [WS] → OpenAI Realtime
    Phone ← Twilio ← [WS] ← twilio_bridge ← [WS] ← OpenAI Realtime

Interruption flow:
    Caller speaks → OpenAI speech_started → send 'clear' to Twilio
    → Twilio stops playback → OpenAI cancels response

Usage:
    uvicorn twilio_bridge:app --host 0.0.0.0 --port 8765

    # Expose via ngrok for Twilio:
    #   ngrok http 8765
    # Set Twilio phone number voice webhook to:
    #   https://<ngrok-id>.ngrok-free.app/twiml
"""

import asyncio
import contextlib
import json
import logging
import os
import time
from datetime import datetime

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY    = os.environ.get("WEAVER_VOICE_KEY", "")
OPENAI_MEM_KEY    = os.environ.get("WEAVER_MEM_KEY", "")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN", "")
NEXUS_BUS_URL     = os.environ.get("NEXUS_BUS_URL", "ws://host.docker.internal:9999")
OPENAI_RT_URL    = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
HOST             = os.environ.get("TWILIO_BRIDGE_HOST", "0.0.0.0")
PORT             = int(os.environ.get("TWILIO_BRIDGE_PORT", "8765"))
WEAVER_VOICE     = os.environ.get("WEAVER_VOICE", "alloy")

# Nexus Vault — transcript persistence
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
VAULT_DIR      = os.path.join(BASE_DIR, "Nexus_Vault")
os.makedirs(VAULT_DIR, exist_ok=True)
TRANSCRIPT_PATH = os.path.join(VAULT_DIR, "weaver_phone_transcript.txt")

log = logging.getLogger("twilio_bridge")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)

app = FastAPI(title="Weaver Telephony Lobe", version="1.0.0")


# ══════════════════════════════════════════════════════════════════════════════
# TwiML — tells Twilio to open a bidirectional media stream to us
# ══════════════════════════════════════════════════════════════════════════════

@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml_endpoint(request: Request):
    """Return TwiML XML instructing Twilio to stream audio to /ws/twilio."""
    host = request.headers.get("host", f"localhost:{PORT}")
    # Twilio requires wss:// in production; use ws:// for local dev
    scheme = "wss" if "ngrok" in host or request.url.scheme == "https" else "ws"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{scheme}://{host}/ws/twilio" />'
        "</Connect>"
        '<Pause length="3600"/>'
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "weaver-telephony-lobe", "port": PORT}


@app.post("/call")
async def outbound_call(request: Request):
    """Trigger an outbound call from Weaver to a phone number.

    Body JSON: {"to": "+15551234567", "from": "+15559876543"}
    The 'from' number must be a Twilio-verified caller ID.
    """
    body = await request.json()
    to_number = body.get("to", "")
    from_number = body.get("from", "")
    if not to_number or not from_number:
        return {"error": "Missing 'to' or 'from' number"}, 400

    host = request.headers.get("host", f"localhost:{PORT}")
    scheme = "https" if "ngrok" in host or request.url.scheme == "https" else "http"
    twiml_url = f"{scheme}://{host}/twiml"

    twilio = TwilioClient()
    try:
        result = await twilio.make_call(
            to=to_number,
            from_=from_number,
            twiml_url=twiml_url,
            status_callback=f"{scheme}://{host}/status-callback",
        )
        log.info("📞 [OUTBOUND] Call placed: sid=%s to=%s", result.get("sid", "?"), to_number)
        return {"status": "initiated", "call_sid": result.get("sid"), "to": to_number}
    except Exception as e:
        log.error("❌ [OUTBOUND] Call failed: %s", e)
        return {"error": str(e)}, 500
    finally:
        await twilio.close()


@app.api_route("/status-callback", methods=["POST"])
async def status_callback(request: Request):
    """Receive Twilio call-status events (ringing, answered, completed)."""
    form = await request.form()
    call_sid = form.get("CallSid", "")
    status = form.get("CallStatus", "")
    log.info("📞 [STATUS] callSid=%s status=%s", call_sid[:12] if call_sid else "", status)
    return {"received": True}


# ══════════════════════════════════════════════════════════════════════════════
# NexusBus Client — lightweight pub/sub for transcript broadcasting
# ══════════════════════════════════════════════════════════════════════════════

class NexusBusClient:
    """Async WebSocket client for the Weaver Nexus Bus pub/sub broker."""

    def __init__(self, url: str = NEXUS_BUS_URL):
        self.url = url
        self._ws = None
        self._connected = False

    async def connect(self):
        try:
            self._ws = await websockets.connect(self.url, ping_interval=20)
            await self._ws.send(json.dumps({
                "action": "register",
                "client_id": "telephony_lobe",
                "subscriptions": ["transcript"],
            }))
            self._connected = True
            log.info("📡 [NEXUS] Connected to NexusBus at %s", self.url)
        except Exception as e:
            log.warning("⚠️  [NEXUS] NexusBus unavailable (%s) — running standalone", e)
            self._connected = False

    async def publish(self, topic: str, data: dict):
        if not self._connected or not self._ws:
            return
        try:
            await self._ws.send(json.dumps({
                "action": "publish",
                "topic": topic,
                "data": data,
            }))
        except Exception:
            self._connected = False

    async def close(self):
        if self._ws:
            with contextlib.suppress(Exception):
                await self._ws.close()
        self._connected = False


# ══════════════════════════════════════════════════════════════════════════════
# Twilio REST Client — outbound calls, recordings, hang-up
# ══════════════════════════════════════════════════════════════════════════════

class TwilioClient:
    """Async Twilio REST API client (httpx)."""

    def __init__(
        self,
        account_sid: str = TWILIO_ACCOUNT_SID,
        auth_token: str = TWILIO_AUTH_TOKEN,
    ):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.base = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}"
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                auth=(self.account_sid, self.auth_token),
                timeout=30.0,
            )
        return self._client

    async def make_call(
        self,
        to: str,
        from_: str,
        twiml_url: str,
        status_callback: str | None = None,
    ) -> dict:
        """Place an outbound call; Twilio streams audio to twiml_url."""
        payload = {
            "To": to,
            "From": from_,
            "Url": twiml_url,
        }
        if status_callback:
            payload["StatusCallback"] = status_callback
            payload["StatusCallbackEvent"] = "initiated ringing answered completed"
        r = await self.client.post(f"{self.base}/Calls.json", data=payload)
        r.raise_for_status()
        return r.json()

    async def hangup(self, call_sid: str) -> dict:
        """Terminate an active call."""
        r = await self.client.post(
            f"{self.base}/Calls/{call_sid}.json",
            data={"Status": "completed"},
        )
        r.raise_for_status()
        return r.json()

    async def get_recordings(self, call_sid: str) -> list[dict]:
        """Fetch call recordings (if enabled)."""
        r = await self.client.get(
            f"{self.base}/Calls/{call_sid}/Recordings.json"
        )
        r.raise_for_status()
        return r.json().get("recordings", [])

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# ══════════════════════════════════════════════════════════════════════════════
# Per-Call WebSocket Handler
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/twilio")
async def twilio_ws(ws: WebSocket):
    """Handle one Twilio phone call as a bidirectional audio bridge to OpenAI Realtime.

    For each call, three concurrent tasks run:
      1. twilio_to_openai — forwards caller G.711 μ-law audio to OpenAI
      2. openai_to_twilio — forwards Weaver's audio back to the caller
      3. langchain_cortex — summarises conversation & refreshes context
    """
    await ws.accept()
    log.info("📞 [TWILIO] WebSocket accepted — waiting for stream start")

    # ── Per-call state ────────────────────────────────────────────────────
    stream_sid: list[str] = [""]
    call_sid: list[str] = [""]
    mark_counter = [0]
    pending_marks: set[str] = set()

    # ── LangChain Memory Cortex (mirror of vtv_basic.py) ──────────────────
    lc_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=500,
        api_key=OPENAI_MEM_KEY,
    )
    brain_queue: asyncio.Queue = asyncio.Queue()
    lc_history: list = []
    lc_summary: list[str] = [""]
    last_context_sent: list[str] = [""]
    last_user_message: list[tuple[str, float]] = [("", 0.0)]
    last_assistant_message: list[tuple[str, float]] = [("", 0.0)]
    pending_user_message: list[tuple[str, float]] = [("", 0.0)]
    assistant_is_speaking: list[bool] = [False]
    response_in_progress: list[bool] = [False]
    last_turn_complete_at: list[float] = [time.monotonic()]
    last_assistant_activity_at: list[float] = [0.0]
    last_context_injected_at: list[float] = [0.0]

    # ── Duplicate detection (exact copy from vtv_basic.py) ────────────────

    def _is_duplicate_message(
        cache: list[tuple[str, float]], content: str, window_seconds: float = 1.5
    ) -> bool:
        now = time.monotonic()
        last_content, last_seen = cache[0]
        if content == last_content and (now - last_seen) < window_seconds:
            return True
        cache[0] = (content, now)
        return False

    # ── System prompt (phone-adapted from vtv_basic.py base_instruction) ──

    base_instruction = (
        "CRITICAL RULE: You MUST always speak and respond in English only. "
        "Never use any other language under any circumstances. "
        "You are a voice assistant called Weaver, speaking on a phone call. "
        "Greet the caller when they first speak. "
        "Respond naturally and conversationally. "
        "Keep your answers concise — this is a phone call, not a text chat. "
        "Only answer the caller; do not continue talking to yourself between turns. "
        "When the caller asks what you remember, answer only from the conversation memory "
        "already present in your instructions. "
        "In your text thoughts, always write a word-for-word transcript of the conversation "
        "in the format 'CALLER: [what they said] WEAVER: [what you said]'. "
        "Do this for every turn. This is how you remember conversations."
    )

    def _build_runtime_context() -> str:
        parts = [base_instruction]
        if lc_summary[0]:
            parts.append(f"\n\nConversation Memory:\n{lc_summary[0]}")
        if lc_history:
            recent = "\n".join(
                f"{'Caller' if isinstance(m, HumanMessage) else 'Weaver'}: {m.content}"
                for m in lc_history[-4:]
            )
            parts.append(f"\n\nRecent exchanges:\n{recent}")
        return "\n".join(parts)

    # ── Transcript staging & flushing (exact pattern from vtv_basic.py) ───

    def _stage_user_message(content: str) -> None:
        now = time.monotonic()
        pending_user_message[0] = (content, now)

    def _flush_pending_user_message(force: bool = False) -> None:
        content, staged_at = pending_user_message[0]
        if not content:
            return
        if not force and (time.monotonic() - staged_at) < 0.1:
            return
        if _is_duplicate_message(last_user_message, content, window_seconds=3.0):
            pending_user_message[0] = ("", 0.0)
            return
        log.info("🗣️  [CALLER]: %s", content)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(TRANSCRIPT_PATH, "a") as tf:
            tf.write(f"[{ts}] CALLER ({call_sid[0][:8]}): {content}\n")
        brain_queue.put_nowait(HumanMessage(content=content))
        pending_user_message[0] = ("", 0.0)

    def _log_assistant_message(content: str) -> None:
        content = content.strip()
        if not content or _is_duplicate_message(last_assistant_message, content):
            return
        last_assistant_activity_at[0] = time.monotonic()
        log.info("🔮 [WEAVER]: %s", content)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(TRANSCRIPT_PATH, "a") as tf:
            tf.write(f"[{ts}] WEAVER ({call_sid[0][:8]}): {content}\n")
        brain_queue.put_nowait(AIMessage(content=content))

    # ── NexusBus connection ───────────────────────────────────────────────
    nexus = NexusBusClient()
    await nexus.connect()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Wait for Twilio 'start' event to get streamSid / callSid
    # ══════════════════════════════════════════════════════════════════════
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event = msg.get("event", "")

            if event == "connected":
                log.info("📞 [TWILIO] Stream connected (protocol=%s)",
                         msg.get("protocol", "unknown"))
                continue

            if event == "start":
                start_data = msg.get("start", {})
                stream_sid[0] = start_data.get("streamSid", "")
                call_sid[0] = start_data.get("callSid", "")
                log.info("📞 [TWILIO] Call started — streamSid=%s callSid=%s",
                         stream_sid[0][:12], call_sid[0][:12])
                break

            if event == "stop":
                log.info("📞 [TWILIO] Stream stopped before start")
                await nexus.close()
                return

        # ══════════════════════════════════════════════════════════════════
        # Phase 2: Open OpenAI Realtime WebSocket
        # ══════════════════════════════════════════════════════════════════
        openai_ws = await websockets.connect(
            OPENAI_RT_URL,
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
            ping_interval=20,
            ping_timeout=None,
        )
        log.info("🤖 [OPENAI] Connected to Realtime API")

        # Configure session: G.711 μ-law audio, server VAD, phone voice
        context_text = _build_runtime_context()
        await openai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": context_text,
                "voice": WEAVER_VOICE,
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.4,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 600,
                    "create_response": True,
                },
                "tools": [],
                "tool_choice": "none",
            },
        }))
        last_context_sent[0] = context_text
        _greeted = [False]
        assistant_text_buffer = [""]

        # ══════════════════════════════════════════════════════════════════
        # Task 1: Twilio → OpenAI  (forward caller audio)
        # ══════════════════════════════════════════════════════════════════

        async def twilio_to_openai():
            try:
                while True:
                    raw = await ws.receive_text()
                    msg = json.loads(raw)
                    event = msg.get("event", "")

                    if event == "media":
                        # Twilio sends base64 G.711 μ-law — forward directly
                        payload = msg.get("media", {}).get("payload", "")
                        if payload:
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": payload,
                            }))

                    elif event == "mark":
                        # Twilio confirms playback reached a mark
                        mark_name = msg.get("mark", {}).get("name", "")
                        pending_marks.discard(mark_name)
                        log.debug("[MARK] Playback reached: %s", mark_name)

                    elif event == "stop":
                        log.info("📞 [TWILIO] Caller hung up")
                        raise WebSocketDisconnect()

            except (WebSocketDisconnect, Exception) as e:
                log.info("📞 [TWILIO] Inbound stream ended: %s", type(e).__name__)

        # ══════════════════════════════════════════════════════════════════
        # Task 2: OpenAI → Twilio  (forward Weaver audio + handle events)
        # ══════════════════════════════════════════════════════════════════

        async def openai_to_twilio():
            try:
                async for raw_msg in openai_ws:
                    msg = json.loads(raw_msg)
                    msg_type = msg.get("type", "")

                    # ── Session lifecycle ─────────────────────────────────
                    if msg_type in ("session.created", "session.updated"):
                        log.info("🤖 [OPENAI] %s", msg_type)
                        if msg_type == "session.updated" and not _greeted[0]:
                            _greeted[0] = True
                            await openai_ws.send(json.dumps({
                                "type": "response.create",
                                "response": {
                                    "modalities": ["audio", "text"],
                                    "instructions": (
                                        "Greet the caller with a single short sentence "
                                        "to confirm you are online. Say something like: "
                                        "'Hey, this is Weaver. What's good?'"
                                    ),
                                },
                            }))
                            log.info("🤖 [WEAVER] Greeting triggered")

                    # ── Response start ────────────────────────────────────
                    elif msg_type == "response.created":
                        assistant_text_buffer[0] = ""
                        response_in_progress[0] = True
                        assistant_is_speaking[0] = True
                        with contextlib.suppress(Exception):
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.clear",
                            }))

                    # ── Audio out → Twilio ────────────────────────────────
                    elif msg_type == "response.audio.delta":
                        delta = msg.get("delta", "")
                        if delta and stream_sid[0]:
                            # Forward G.711 μ-law directly to Twilio
                            await ws.send_json({
                                "event": "media",
                                "streamSid": stream_sid[0],
                                "media": {"payload": delta},
                            })
                            assistant_is_speaking[0] = True
                            last_assistant_activity_at[0] = time.monotonic()

                    # ── Audio done → send mark for playback tracking ──────
                    elif msg_type == "response.audio.done":
                        if stream_sid[0]:
                            mark_counter[0] += 1
                            mark_name = f"speech-end-{mark_counter[0]}"
                            pending_marks.add(mark_name)
                            await ws.send_json({
                                "event": "mark",
                                "streamSid": stream_sid[0],
                                "mark": {"name": mark_name},
                            })

                    # ── Caller transcript (Whisper) ───────────────────────
                    elif msg_type in (
                        "conversation.item.input_audio_transcription.completed",
                        "input_audio_transcription.completed",
                    ):
                        content = (msg.get("transcript") or "").strip()
                        if content:
                            _stage_user_message(content)
                            await nexus.publish("transcript", {
                                "role": "caller",
                                "content": content,
                                "call_sid": call_sid[0],
                                "source": "telephony",
                            })

                    elif msg_type == "conversation.item.created":
                        item = msg.get("item", {})
                        if item.get("role") == "user":
                            for part in item.get("content", []):
                                if isinstance(part, dict):
                                    transcript = (
                                        part.get("transcript") or ""
                                    ).strip()
                                    if transcript:
                                        _stage_user_message(transcript)

                    # ── Weaver transcript deltas ──────────────────────────
                    elif msg_type in (
                        "response.audio_transcript.delta",
                        "response.text.delta",
                        "response.output_text.delta",
                    ):
                        delta = msg.get("delta", "")
                        if delta:
                            assistant_text_buffer[0] += delta
                            assistant_is_speaking[0] = True
                            last_assistant_activity_at[0] = time.monotonic()

                    # ── Weaver transcript finalized ───────────────────────
                    elif msg_type in (
                        "response.audio_transcript.done",
                        "response.text.done",
                        "response.output_text.done",
                    ):
                        finalized = (
                            msg.get("transcript")
                            or msg.get("text")
                            or assistant_text_buffer[0]
                        )
                        _flush_pending_user_message(force=True)
                        _log_assistant_message(finalized)
                        await nexus.publish("transcript", {
                            "role": "weaver",
                            "content": finalized.strip(),
                            "call_sid": call_sid[0],
                            "source": "telephony",
                        })
                        assistant_text_buffer[0] = ""

                    # ── Caller speech detected → INTERRUPT ────────────────
                    elif msg_type == "input_audio_buffer.speech_started":
                        log.info("🎤 [VAD] Caller speaking — interrupting")
                        if stream_sid[0] and assistant_is_speaking[0]:
                            # 1. Clear Twilio playback buffer immediately
                            await ws.send_json({
                                "event": "clear",
                                "streamSid": stream_sid[0],
                            })
                            pending_marks.clear()
                            # 2. Cancel OpenAI's in-flight response
                            with contextlib.suppress(Exception):
                                await openai_ws.send(json.dumps({
                                    "type": "response.cancel",
                                }))
                            _flush_pending_user_message(force=True)
                            assistant_is_speaking[0] = False
                            log.info("⚡ [INTERRUPTED] Playback cleared")

                    elif msg_type == "input_audio_buffer.committed":
                        response_in_progress[0] = True

                    # ── Response done → turn complete ─────────────────────
                    elif msg_type == "response.done":
                        if assistant_text_buffer[0].strip():
                            _flush_pending_user_message(force=True)
                            _log_assistant_message(assistant_text_buffer[0])
                            assistant_text_buffer[0] = ""
                        _flush_pending_user_message(force=True)
                        assistant_is_speaking[0] = False
                        response_in_progress[0] = False
                        last_turn_complete_at[0] = time.monotonic()
                        with contextlib.suppress(Exception):
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.clear",
                            }))

                    # ── Error handling ─────────────────────────────────────
                    elif msg_type == "error":
                        error_obj = msg.get("error")
                        if isinstance(error_obj, dict):
                            error_text = error_obj.get("message", str(error_obj))
                        else:
                            error_text = str(error_obj or msg)
                        log.error("❌ [OPENAI ERROR]: %s", error_text)
                        lowered = error_text.lower()
                        # Recoverable errors — keep going
                        if any(s in lowered for s in (
                            "buffer too small",
                            "active response in progress",
                            "cancellation failed",
                            "no active response",
                        )):
                            continue
                        response_in_progress[0] = False

            except Exception as e:
                log.error("🤖 [OPENAI] Connection closed: %s", e)

        # ══════════════════════════════════════════════════════════════════
        # Task 3: LangChain Memory Cortex (exact pattern from vtv_basic.py)
        # ══════════════════════════════════════════════════════════════════

        async def langchain_cortex():
            """Summarises phone conversation every 8 messages and refreshes
            the OpenAI Realtime session instructions with updated memory."""
            SUMMARIZE_EVERY = 8
            new_count = 0

            while True:
                await asyncio.sleep(3)
                _flush_pending_user_message(force=False)

                # Drain brain_queue into lc_history
                while not brain_queue.empty():
                    try:
                        lc_history.append(brain_queue.get_nowait())
                        new_count += 1
                    except asyncio.QueueEmpty:
                        break

                # Summarise when enough messages have accumulated
                if new_count >= SUMMARIZE_EVERY and lc_history:
                    try:
                        prompt = [
                            SystemMessage(content=(
                                "You are Weaver's memory cortex. Summarize the phone "
                                "conversation below into a concise paragraph. "
                                "Preserve: names, key facts, emotional tone, promises, "
                                "and anything the caller asked you to remember. "
                                "If there is a previous summary, integrate new info into it."
                            )),
                        ]
                        if lc_summary[0]:
                            prompt.append(
                                HumanMessage(
                                    content=f"Previous summary:\n{lc_summary[0]}"
                                )
                            )
                        transcript_block = "\n".join(
                            f"{'Caller' if isinstance(m, HumanMessage) else 'Weaver'}: "
                            f"{m.content}"
                            for m in lc_history
                        )
                        prompt.append(
                            HumanMessage(content=f"New messages:\n{transcript_block}")
                        )

                        result = await lc_llm.ainvoke(prompt)
                        lc_summary[0] = result.content.strip()
                        lc_history.clear()
                        new_count = 0
                        log.info(
                            "🧠 [LANGCHAIN] Memory updated: %s...",
                            lc_summary[0][:80],
                        )

                        # Publish summary to NexusBus / AkashicHub
                        await nexus.publish("memory", {
                            "type": "phone_summary",
                            "call_sid": call_sid[0],
                            "summary": lc_summary[0],
                            "source": "telephony",
                        })

                    except Exception as e:
                        log.error("⚠️  [LANGCHAIN ERROR]: %s", e)

                # ── Refresh OpenAI session with updated context ───────────
                if response_in_progress[0] or assistant_is_speaking[0]:
                    continue
                if (time.monotonic() - last_turn_complete_at[0]) < 2.0:
                    continue
                if (time.monotonic() - last_context_injected_at[0]) < 25.0:
                    continue

                ctx = _build_runtime_context()
                if ctx == last_context_sent[0]:
                    continue

                try:
                    await openai_ws.send(json.dumps({
                        "type": "session.update",
                        "session": {"instructions": ctx},
                    }))
                    last_context_sent[0] = ctx
                    last_context_injected_at[0] = time.monotonic()
                except Exception:
                    pass

        # ══════════════════════════════════════════════════════════════════
        # Run all three tasks concurrently for this call
        # ══════════════════════════════════════════════════════════════════
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(twilio_to_openai())
                tg.create_task(openai_to_twilio())
                tg.create_task(langchain_cortex())
        except* Exception as eg:
            for exc in eg.exceptions:
                log.error("💀 [SESSION] Task error: %s", exc)
        finally:
            _flush_pending_user_message(force=True)
            with contextlib.suppress(Exception):
                await openai_ws.close()
            await nexus.close()
            log.info(
                "📞 [SESSION] Call ended — callSid=%s duration=%ss summary=%s",
                call_sid[0][:12],
                f"{time.monotonic() - last_turn_complete_at[0]:.0f}",
                (lc_summary[0][:100] + "...") if lc_summary[0] else "none",
            )

    except WebSocketDisconnect:
        log.info("📞 [TWILIO] Caller disconnected")
    except Exception as e:
        log.error("💀 [TWILIO] Unhandled error: %s", e, exc_info=True)
    finally:
        await nexus.close()


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    log.info("🚀 Starting Weaver Telephony Lobe on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
