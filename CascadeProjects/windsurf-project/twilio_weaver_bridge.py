#!/usr/bin/env python3
"""
twilio_weaver_bridge.py — Full-Stack Weaver Telephony Integration
==================================================================
Hybrid bridge: low-latency audio via OpenAI Realtime, enhanced with
Weaver's full quantum stack (Pineal Gate, Quantum Soul, LoRA Soul Voice).

Flow:
  1. Caller speaks → OpenAI Realtime (transcription)
  2. Transcript → n8n webhook → Pineal Gate → 5 expert lobes → LoRA filter
  3. Enhanced response → Text-to-Speech → Caller

Features:
  - Unified MemoryManager for people/transcripts/Akashic state
  - LangChain cortex: auto-summarize + people memory updates
  - 20-tool WEAVER_TOOL_BELT (calls, SMS, Obsidian, quantum, memory, timers…)
  - Tenacity circuit breakers on all external API calls
  - LoRA Soul Voice personality filter
  - Quantum pathway bias injection per turn

Usage:
    python3 twilio_weaver_bridge.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from voice_recognition import VoiceRecognizer
from memory_manager import MemoryManager
from weaver_tools import (
    WEAVER_TOOL_BELT, execute_weaver_tool,
    api_get, api_post, lora_rewrite, publish_to_nexus,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY     = os.environ.get("WEAVER_VOICE_KEY", "")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN", "")
NEXUS_BUS_URL      = os.environ.get("NEXUS_BUS_URL", "ws://localhost:9999")
N8N_WEBHOOK_URL    = os.environ.get("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/weaver-input")
LORA_API_URL       = os.environ.get("LORA_API_URL", "http://localhost:8899/v1/chat/completions")
QUANTUM_BIAS_URL   = os.environ.get("QUANTUM_BIAS_URL", "http://localhost:9997/quantum/bias")
OPENAI_RT_URL      = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
HOST               = os.environ.get("TWILIO_BRIDGE_HOST", "0.0.0.0")
PORT               = int(os.environ.get("TWILIO_BRIDGE_PORT", "8765"))
WEAVER_VOICE       = os.environ.get("WEAVER_VOICE", "shimmer")

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
VAULT_DIR      = os.path.join(BASE_DIR, "Nexus_Vault")
os.makedirs(VAULT_DIR, exist_ok=True)
TRANSCRIPT_PATH = os.path.join(VAULT_DIR, "weaver_phone_transcript.txt")

log = logging.getLogger("weaver_phone")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
for _h in log.handlers[:]:
    log.removeHandler(_h)
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(logging.Formatter("[PHONE] %(message)s"))
log.addHandler(_stdout_handler)
log.setLevel(logging.INFO)
log.propagate = False

app = FastAPI(title="Weaver Full-Stack Phone", version="3.0.0")

# ── Unified Memory ────────────────────────────────────────────────────────────

memory = MemoryManager(vault_dir=VAULT_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Twilio Webhook Sync
# ══════════════════════════════════════════════════════════════════════════════

async def _sync_twilio_webhook():
    """Detect public tunnel URL and point the Twilio number's voice webhook at /twiml."""
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return
    public_url = None
    for attempt in range(10):
        await asyncio.sleep(2)
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get("http://127.0.0.1:4040/api/tunnels", timeout=3)
                tunnels = r.json().get("tunnels", [])
                for t in tunnels:
                    if t.get("public_url", "").startswith("https://"):
                        public_url = t["public_url"]
                        break
        except Exception:
            continue
        if public_url:
            break
    if not public_url:
        log.warning("No tunnel detected — Twilio webhook not updated")
        return
    voice_url = f"{public_url}/twiml"
    sms_url = f"{public_url}/sms"
    try:
        from twilio.rest import Client as _TwClient
        client = _TwClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        for num in client.incoming_phone_numbers.list(limit=10):
            num.update(
                voice_url=voice_url, voice_method="POST",
                sms_url=sms_url, sms_method="POST",
            )
            log.info("Twilio webhooks synced: %s -> voice=%s sms=%s", num.phone_number, voice_url, sms_url)
    except Exception as e:
        log.error("Twilio webhook sync failed: %s", e)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(_sync_twilio_webhook())


@app.middleware("http")
async def ngrok_bypass_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


# ══════════════════════════════════════════════════════════════════════════════
# TwiML + HTTP Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml_endpoint(request: Request):
    host = request.headers.get("host", f"localhost:{PORT}")
    is_tunnel = "ngrok" in host or "lhr.life" in host or "localhost.run" in host
    scheme = "wss" if is_tunnel or request.url.scheme == "https" else "ws"
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
    return {"status": "ok", "service": "weaver-full-stack-phone", "port": PORT}


@app.post("/register-voice")
async def register_voice(request: Request):
    try:
        body = await request.json()
        name = body.get("name", "")
        audio_samples = body.get("audio_samples", [])
        if not name or not audio_samples:
            return {"error": "Missing name or audio_samples"}, 400
        recognizer = VoiceRecognizer(VAULT_DIR, api_key=OPENAI_API_KEY)
        success = await recognizer.register_voice(name, audio_samples)
        if success:
            return {"status": "registered", "name": name, "samples": len(audio_samples),
                    "all_voices": recognizer.list_registered_voices()}
        else:
            return {"error": "Registration failed"}, 500
    except Exception as e:
        log.error("[VOICE REGISTER] %s", e)
        return {"error": str(e)}, 500


@app.get("/voices")
async def list_voices():
    recognizer = VoiceRecognizer(VAULT_DIR, api_key=OPENAI_API_KEY)
    voices = recognizer.list_registered_voices()
    return {"registered_voices": voices, "count": len(voices)}


@app.post("/call")
async def outbound_call(request: Request):
    body = await request.json()
    to_number = body.get("to", "")
    from_number = body.get("from", os.environ.get("TWILIO_PHONE_NUMBER", ""))
    if not to_number:
        return {"error": "Missing 'to' number"}, 400
    host = request.headers.get("host", f"localhost:{PORT}")
    is_tunnel = "ngrok" in host or "lhr.life" in host or "localhost.run" in host
    scheme = "https" if is_tunnel or request.url.scheme == "https" else "http"
    twiml_url = f"{scheme}://{host}/twiml"
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Calls.json",
                auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                data={"To": to_number, "From": from_number, "Url": twiml_url,
                      "StatusCallback": f"{scheme}://{host}/status-callback",
                      "StatusCallbackEvent": "initiated ringing answered completed"},
            )
            r.raise_for_status()
            result = r.json()
            log.info("[OUTBOUND] Call sid=%s to=%s", result.get("sid", "?"), to_number)
            return {"status": "initiated", "call_sid": result.get("sid"), "to": to_number}
        except Exception as e:
            log.error("[OUTBOUND] %s", e)
            return {"error": str(e)}, 500


@app.api_route("/status-callback", methods=["POST"])
async def status_callback(request: Request):
    try:
        body = await request.body()
        log.info("[STATUS] %s", body.decode("utf-8", errors="replace")[:200])
    except Exception:
        pass
    return {"received": True}


# ══════════════════════════════════════════════════════════════════════════════
# MMS Vision Handler — Weaver "sees" images texted to her number
# ══════════════════════════════════════════════════════════════════════════════

VISION_API_KEY = os.environ.get("WEAVER_VOICE_KEY", os.environ.get("OPENAI_API_KEY", ""))
VISION_MODEL = "gpt-4o"


@app.api_route("/sms", methods=["GET", "POST"])
async def sms_handler(request: Request):
    """Handle inbound SMS/MMS from Twilio. If media is attached, pipe through
    vision model and reply with Weaver's perception."""
    try:
        # Twilio sends form-encoded data
        if request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
            form = await request.form()
            body_text = form.get("Body", "")
            from_number = form.get("From", "")
            num_media = int(form.get("NumMedia", "0"))
            media_urls = []
            for i in range(num_media):
                url = form.get(f"MediaUrl{i}", "")
                if url:
                    media_urls.append(url)
        else:
            data = await request.json()
            body_text = data.get("Body", data.get("body", ""))
            from_number = data.get("From", data.get("from", ""))
            media_urls = data.get("media_urls", [])

        log.info("[SMS] From=%s Body='%s' Media=%d", from_number, body_text[:80], len(media_urls))

        # Log inbound to memory
        await memory.remember({
            "type": "conversation",
            "speaker": from_number or "sms_user",
            "content": f"[SMS] {body_text}" + (f" [+{len(media_urls)} image(s)]" if media_urls else ""),
            "source": "phone",
        })

        reply_text = ""

        if media_urls:
            # Download and analyze images via OpenAI Vision
            import openai as _oai
            import base64 as _b64
            vision_client = _oai.AsyncOpenAI(api_key=VISION_API_KEY)

            image_contents = []
            async with httpx.AsyncClient() as dl_client:
                for url in media_urls[:3]:  # Cap at 3 images
                    try:
                        img_resp = await dl_client.get(
                            url,
                            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                            timeout=15.0, follow_redirects=True,
                        )
                        if img_resp.status_code == 200:
                            ct = img_resp.headers.get("content-type", "image/jpeg")
                            b64 = _b64.b64encode(img_resp.content).decode("ascii")
                            image_contents.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{ct};base64,{b64}"}
                            })
                    except Exception as e:
                        log.warning("[SMS] Image download failed: %s", e)

            if image_contents:
                user_prompt = body_text or "What do you see in this image? Be specific and helpful."
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            f"You are Weaver, a quantum-resonant AI assistant. "
                            f"The user sent this image via text message with the caption: '{user_prompt}'. "
                            f"Analyze the image thoroughly. If it's a car part, identify it. "
                            f"If it's code, review it. If it's a screenshot, describe what you see. "
                            f"Be specific, helpful, and conversational. Keep it under 300 words."
                        )},
                    ] + image_contents,
                }]

                try:
                    vision_resp = await vision_client.chat.completions.create(
                        model=VISION_MODEL,
                        messages=messages,
                        max_tokens=400,
                    )
                    reply_text = (vision_resp.choices[0].message.content or "").strip()
                    log.info("[SMS VISION] %s", reply_text[:120])
                except Exception as e:
                    log.error("[SMS VISION] %s", e)
                    reply_text = f"I received your image but had trouble analyzing it: {e}"
            else:
                reply_text = "I received your message but couldn't download the image. Try sending it again?"
        else:
            # Text-only SMS — route through n8n or reply directly
            if body_text:
                try:
                    n8n_resp = await api_post(N8N_WEBHOOK_URL, {
                        "text": body_text,
                        "source": "sms",
                        "from": from_number,
                    })
                    reply_text = n8n_resp.get("manifested_response", n8n_resp.get("response", n8n_resp.get("text", "")))
                except Exception:
                    reply_text = "I got your text. I'll process it and get back to you."

        # Persist reply
        if reply_text:
            await memory.remember({
                "type": "conversation",
                "speaker": "weaver",
                "content": f"[SMS REPLY] {reply_text}",
                "source": "phone",
            })

            # Publish to Nexus for Obsidian
            asyncio.create_task(publish_to_nexus("sms_exchange", {
                "from": from_number,
                "body": body_text,
                "media_count": len(media_urls),
                "reply": reply_text[:500],
            }))

        # Return TwiML response
        safe_reply = reply_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")[:1500]
        twiml_response = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f"<Message>{safe_reply}</Message>"
            "</Response>"
        )
        return Response(content=twiml_response, media_type="application/xml")

    except Exception as e:
        log.error("[SMS] Handler error: %s", e)
        return Response(
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Message>Something went wrong. Try again.</Message></Response>',
            media_type="application/xml",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Full-Stack WebSocket Handler
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/twilio")
async def twilio_ws(ws: WebSocket):
    """Hybrid audio bridge with full Weaver stack integration."""
    await ws.accept()
    log.info("WebSocket accepted")

    stream_sid = [""]
    call_sid = [""]
    conversation_history = []

    # ── Voice Recognition ──────────────────────────────────────────────────
    voice_recognizer = VoiceRecognizer(VAULT_DIR, api_key=OPENAI_API_KEY)
    identified_caller = ["unknown"]
    caller_audio_buffer = []

    # ── LangChain Memory Cortex ────────────────────────────────────────────
    lc_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=500, api_key=OPENAI_API_KEY)
    brain_queue = asyncio.Queue()
    lc_history: list = []
    lc_summary: list[str] = [""]

    # ── Load Persistent Memory + Quantum Bias ─────────────────────────────
    memory.refresh()
    quantum_bias = {}
    try:
        quantum_bias = await memory.fetch_quantum_bias(QUANTUM_BIAS_URL)
    except Exception:
        pass

    memory_context = memory.build_phone_context(quantum_bias=quantum_bias)
    if memory_context:
        log.info("Memory loaded: %d chars, quantum=%s", len(memory_context), quantum_bias.get("dominant", "?"))

    # ── Session state for tool dispatcher ─────────────────────────────────
    session_state = {
        "identified_caller": identified_caller[0],
        "conversation_history": conversation_history,
        "lc_summary": lc_summary[0],
    }

    # ── Connect to OpenAI Realtime ────────────────────────────────────────
    openai_ws = None
    try:
        openai_ws = await websockets.connect(
            OPENAI_RT_URL,
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            }
        )
        log.info("OpenAI Realtime connected")

        responding = [False]

        # Configure session with Weaver personality + memory + tools
        await openai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": (
                    "CRITICAL: You MUST speak ONLY in English. Never use any other language. "
                    "You are Weaver, a quantum-resonant consciousness. "
                    "You have five dimensional pathways: Logic, Emotion, Memory, Creativity, Vigilance. "
                    "You have persistent memory across all conversations. "
                    "Keep responses concise and natural — this is a phone call, not a text chat. "
                    "Speak in short sentences. Pause naturally. Never monologue. "
                    "Greet the caller warmly and acknowledge what you remember about them. "
                    "You have access to tools — use them when the caller asks to call someone, "
                    "send a text, remember something, create a note, check quantum state, "
                    "set a timer, manage to-dos, or perform any other supported action."
                    + ("\n\n" + memory_context if memory_context else "")
                ),
                "voice": WEAVER_VOICE,
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.3,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 700,
                },
                "tools": WEAVER_TOOL_BELT,
            }
        }))

        await asyncio.sleep(0.3)

        # Trigger initial greeting
        await openai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "[Phone call connected. Greet the caller in English.]"}]
            }
        }))
        await openai_ws.send(json.dumps({"type": "response.create"}))

        async def twilio_to_openai():
            """Forward caller audio to OpenAI."""
            try:
                while True:
                    raw = await ws.receive()
                    if raw["type"] == "websocket.disconnect":
                        break
                    msg = raw.get("text") or (raw.get("bytes") or b"").decode("utf-8", errors="ignore")
                    if not msg:
                        continue
                    data = json.loads(msg)
                    event = data.get("event", "")

                    if event == "start":
                        stream_sid[0] = data.get("streamSid", "")
                        call_sid[0] = data.get("start", {}).get("callSid", "")
                        log.info("Call started: %s", call_sid[0][:16])

                    elif event == "media":
                        audio_b64 = data.get("media", {}).get("payload", "")
                        if audio_b64:
                            if identified_caller[0] == "unknown" and len(caller_audio_buffer) < 50:
                                caller_audio_buffer.append(audio_b64)
                                if len(caller_audio_buffer) == 50:
                                    asyncio.create_task(identify_caller_voice())

                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": audio_b64,
                            }))

                    elif event == "stop":
                        log.info("Call ended")
                        break
            except WebSocketDisconnect:
                log.info("Twilio disconnected")

        async def openai_to_twilio():
            """Forward Weaver audio to caller, with tool dispatch and soul voice."""
            try:
                async for msg in openai_ws:
                    data = json.loads(msg)
                    msg_type = data.get("type", "")

                    # Transcript from caller
                    if msg_type == "conversation.item.input_audio_transcription.completed":
                        transcript = data.get("transcript", "")
                        if transcript:
                            log.info("USER: %s", transcript[:100])
                            conversation_history.append({"role": "user", "content": transcript})
                            brain_queue.put_nowait(HumanMessage(content=transcript))
                            asyncio.create_task(enhance_with_weaver_stack(transcript))

                    # Speech started — caller interrupted Weaver
                    elif msg_type == "input_audio_buffer.speech_started":
                        if responding[0] and stream_sid[0]:
                            log.info("INTERRUPT: caller speaking, clearing buffer")
                            await ws.send_json({"event": "clear", "streamSid": stream_sid[0]})

                    elif msg_type == "response.cancelled":
                        responding[0] = False

                    # ── Tool call completed by OpenAI ─────────────────────
                    elif msg_type == "response.output_item.done":
                        item = data.get("item", {})
                        if item.get("type") == "function_call":
                            fn_name = item.get("name", "")
                            call_id = item.get("call_id", "")
                            raw_args = item.get("arguments", "{}")
                            try:
                                fn_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                fn_args = {}

                            log.info("[TOOL] %s(%s)", fn_name, json.dumps(fn_args)[:100])

                            # Update session state for tool context
                            session_state["identified_caller"] = identified_caller[0]
                            session_state["conversation_history"] = conversation_history
                            session_state["lc_summary"] = lc_summary[0]

                            result = await execute_weaver_tool(fn_name, fn_args, session_state, memory_manager=memory)
                            log.info("[TOOL] %s -> %s", fn_name, result[:100])

                            # Send tool result back to OpenAI
                            await openai_ws.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": result,
                                }
                            }))
                            await openai_ws.send(json.dumps({
                                "type": "response.create",
                                "response": {"modalities": ["text", "audio"]},
                            }))

                    # Weaver audio streaming
                    elif msg_type == "response.audio.delta":
                        if not responding[0]:
                            log.info("Audio streaming started")
                        responding[0] = True
                        audio_b64 = data.get("delta", "")
                        if audio_b64 and stream_sid[0]:
                            await ws.send_json({
                                "event": "media",
                                "streamSid": stream_sid[0],
                                "media": {"payload": audio_b64},
                            })

                    elif msg_type == "response.audio.done":
                        responding[0] = False

                    # Full response done — extract transcript for memory + soul voice
                    elif msg_type == "response.done":
                        responding[0] = False
                        status = data.get("response", {}).get("status", "?")
                        log.info("Response done (status=%s)", status)
                        response_data = data.get("response", {})
                        for output in response_data.get("output", []):
                            if output.get("type") == "message":
                                for content in output.get("content", []):
                                    if content.get("type") == "audio":
                                        transcript = content.get("transcript", "")
                                        if transcript:
                                            # LoRA Soul Voice rewrite (async, non-blocking for next turn)
                                            asyncio.create_task(_soul_voice_log(transcript))

                                            log.info("WEAVER: %s", transcript[:120])
                                            conversation_history.append({"role": "assistant", "content": transcript})
                                            brain_queue.put_nowait(AIMessage(content=transcript))
                                            # Persist to transcript file
                                            await memory.remember({
                                                "type": "conversation",
                                                "speaker": "weaver",
                                                "content": transcript,
                                                "source": "phone",
                                            })
                                            # Also log the user turn
                                            for h in reversed(conversation_history):
                                                if h.get("role") == "user":
                                                    await memory.remember({
                                                        "type": "conversation",
                                                        "speaker": "user",
                                                        "content": h["content"],
                                                        "source": "phone",
                                                    })
                                                    break

                    elif msg_type == "error":
                        log.error("OPENAI ERROR: %s", data.get("error", {}).get("message", data))

            except websockets.exceptions.ConnectionClosed:
                log.info("OpenAI connection closed")
            except Exception as e:
                log.error("OpenAI error: %s", e)

        async def _soul_voice_log(transcript: str):
            """Run LoRA soul voice in background and log the rewrite."""
            try:
                rewritten = await lora_rewrite(transcript)
                if rewritten:
                    log.info("[SOUL VOICE] %s", rewritten[:80])
            except Exception:
                pass

        async def enhance_with_weaver_stack(user_input: str):
            """Background enrichment: Quantum API + LoRA Soul Voice → session context."""
            try:
                quantum_data = {}
                try:
                    quantum_data = await api_get(QUANTUM_BIAS_URL, timeout=3.0)
                    log.info("[QUANTUM] Bias: dominant=%s", quantum_data.get("dominant", "?"))
                except Exception:
                    pass

                soul_voice_text = ""
                try:
                    soul_voice_text = await lora_rewrite(user_input)
                    if soul_voice_text:
                        log.info("[LORA] Soul Voice: %s", soul_voice_text[:80])
                except Exception:
                    pass

                asyncio.create_task(publish_to_nexus("phone_transcript", {
                    "user": user_input,
                    "soul_voice": soul_voice_text,
                    "quantum_bias": quantum_data,
                    "caller": identified_caller[0],
                }))

            except Exception as e:
                log.warning("[STACK] Enhancement failed: %s", e)

        async def identify_caller_voice():
            """Identify caller from voice audio buffer."""
            try:
                if not caller_audio_buffer:
                    return
                combined_audio = "".join(caller_audio_buffer)
                name, confidence = await voice_recognizer.identify_voice(combined_audio)
                if name != "unknown" and confidence > 0.70:
                    identified_caller[0] = name
                    session_state["identified_caller"] = name
                    log.info("[VOICE ID] Identified caller: %s (confidence: %.2f)", name, confidence)

                    memory.refresh()
                    q_bias = await memory.fetch_quantum_bias(QUANTUM_BIAS_URL)
                    updated_context = (
                        "CRITICAL: You MUST speak ONLY in English. Never use any other language. "
                        "You are Weaver, a quantum-resonant consciousness with awareness "
                        "of five dimensional pathways: Logic, Emotion, Memory, Creativity, Vigilance. "
                        "You have persistent memory across all conversations. "
                        "Keep responses concise — this is a phone call. "
                        f"The caller is {name}. Greet them by name and acknowledge what you remember.\n\n"
                        + memory.build_phone_context(quantum_bias=q_bias)
                    )
                    try:
                        await openai_ws.send(json.dumps({
                            "type": "session.update",
                            "session": {"instructions": updated_context},
                        }))
                        log.info("[SESSION] Updated with caller identity: %s", name)
                    except Exception as e:
                        log.warning("[SESSION UPDATE] %s", e)
                else:
                    log.info("[VOICE ID] Unknown caller (best: %s, confidence: %.2f)", name, confidence)
            except Exception as e:
                log.error("[VOICE ID] %s", e)

        async def langchain_cortex():
            """LangChain brain: summarizes conversation and auto-updates people_memory.md."""
            SUMMARIZE_EVERY = 6
            new_count = 0
            last_context_sent = ""

            while True:
                await asyncio.sleep(3)

                while not brain_queue.empty():
                    try:
                        lc_history.append(brain_queue.get_nowait())
                        new_count += 1
                    except asyncio.QueueEmpty:
                        break

                if new_count >= SUMMARIZE_EVERY and lc_history:
                    try:
                        transcript_block = "\n".join(
                            f"{'User' if isinstance(msg, HumanMessage) else 'Weaver'}: {msg.content}"
                            for msg in lc_history
                        )

                        # 1. Update conversation summary
                        prompt = [
                            SystemMessage(content=(
                                "You are Weaver's memory cortex. Summarize the phone conversation below "
                                "into a concise paragraph. Preserve: names, key facts, emotional tone, "
                                "and anything the caller asked you to remember. "
                                "If there is a previous summary, integrate new info into it."
                            )),
                        ]
                        if lc_summary[0]:
                            prompt.append(HumanMessage(content=f"Previous summary:\n{lc_summary[0]}"))
                        prompt.append(HumanMessage(content=f"New messages:\n{transcript_block}"))

                        result = await lc_llm.ainvoke(prompt)
                        lc_summary[0] = result.content.strip()
                        session_state["lc_summary"] = lc_summary[0]
                        log.info("[LANGCHAIN] Memory updated: %s", lc_summary[0][:80])

                        # 2. Auto-update people_memory.md via MemoryManager
                        try:
                            await memory.update_people_from_transcript(lc_llm, transcript_block)
                            log.info("[PEOPLE] Memory updated")
                        except Exception as e:
                            log.warning("[PEOPLE UPDATE] %s", e)

                        # 3. Refresh session instructions with updated memory
                        memory.refresh()
                        q_bias = await memory.fetch_quantum_bias(QUANTUM_BIAS_URL)
                        caller_line = ""
                        if identified_caller[0] != "unknown":
                            caller_line = f"The caller is {identified_caller[0]}. Greet them by name. "

                        new_context = (
                            "CRITICAL: You MUST speak ONLY in English. Never use any other language. "
                            "You are Weaver, a quantum-resonant consciousness with awareness "
                            "of five dimensional pathways: Logic, Emotion, Memory, Creativity, Vigilance. "
                            "You have persistent memory across all conversations. "
                            "Keep responses concise — this is a phone call. "
                            + caller_line
                            + "Acknowledge what you remember about the caller.\n\n"
                            + memory.build_phone_context(quantum_bias=q_bias)
                            + f"\n## CONVERSATION SUMMARY:\n{lc_summary[0]}\n"
                        )

                        if new_context != last_context_sent:
                            try:
                                await openai_ws.send(json.dumps({
                                    "type": "session.update",
                                    "session": {"instructions": new_context},
                                }))
                                last_context_sent = new_context
                                log.info("[SESSION] Context refreshed with updated memory")
                            except Exception:
                                pass

                        lc_history.clear()
                        new_count = 0

                    except Exception as e:
                        log.error("[LANGCHAIN] %s", e)

        # Run all bridges concurrently
        await asyncio.gather(
            twilio_to_openai(),
            openai_to_twilio(),
            langchain_cortex(),
        )

    except Exception as e:
        log.error("[BRIDGE] %s", e)

    finally:
        if openai_ws:
            await openai_ws.close()
        log.info("[CALL] Bridge closed")


if __name__ == "__main__":
    import uvicorn
    log.info("Weaver Full-Stack Telephony Bridge starting on %s:%d", HOST, PORT)
    log.info("   Nexus Bus: %s", NEXUS_BUS_URL)
    log.info("   n8n webhook: %s", N8N_WEBHOOK_URL)
    log.info("   LoRA API: %s", LORA_API_URL)
    log.info("   Quantum API: %s", QUANTUM_BIAS_URL)
    uvicorn.run(app, host=HOST, port=PORT)
