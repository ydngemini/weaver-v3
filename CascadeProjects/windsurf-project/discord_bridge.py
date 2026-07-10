#!/usr/bin/env python3
"""
discord_bridge.py — Weaver v3 Discord Voice/Vision Lobe
========================================================
Full-stack Discord integration: voice channels with STT/TTS,
video frame capture for face recognition, and Nexus Bus registration.

Flow:
  1. User speaks in Discord voice channel → AudioSink captures PCM
  2. PCM → Whisper STT (via Azure OpenAI)
  3. Transcript → n8n webhook → Pineal Gate → 5 expert lobes → LoRA Soul Voice
  4. Enhanced response → Azure TTS → Discord voice playback
  5. Video frames → face recognition + vision analysis

Requirements:
  - discord.py[voice] >= 2.7
  - PyNaCl
  - DISCORD_BOT_TOKEN in .env
  - Weaver stack running (Nexus Bus, LoRA, Quantum API, n8n)

Usage:
    python3 discord_bridge.py
"""

import asyncio
import base64
import io
import json
import logging
import os
import struct
import sys
import tempfile
import time
import wave
from collections import deque
from datetime import datetime
from typing import Optional

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
import httpx
import numpy as np
from memory_manager import default_vault_dir

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DISCORD_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_GUILD_ID = int(os.environ.get("DISCORD_GUILD_ID") or "0")
DISCORD_VOICE_CHANNEL_ID = int(os.environ.get("DISCORD_VOICE_CHANNEL_ID") or "0")
DISCORD_TEXT_CHANNEL_ID = int(os.environ.get("DISCORD_TEXT_CHANNEL_ID") or "0")

NEXUS_BUS_URL = os.environ.get("NEXUS_BUS_URL", "ws://localhost:9999")
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/weaver-input")
LORA_API_URL = os.environ.get("LORA_API_URL", "http://localhost:8899/v1/chat/completions")
QUANTUM_BIAS_URL = os.environ.get("QUANTUM_BIAS_URL", "http://localhost:9997/quantum/bias")

# Azure OpenAI for STT/TTS
_AZURE_EP = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
_AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
_AZURE_VER = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
_AZURE_DEPLOY = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5")
_AZURE_NANO_DEPLOY = os.environ.get("AZURE_OPENAI_NANO_DEPLOYMENT", "gpt-5.4-nano")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAULT_DIR = str(default_vault_dir())
os.makedirs(VAULT_DIR, exist_ok=True)
TRANSCRIPT_PATH = os.path.join(VAULT_DIR, "weaver_discord_transcript.txt")

WEAVER_VOICE = os.environ.get("WEAVER_VOICE", "shimmer")
SILENCE_THRESHOLD_MS = 1500  # ms of silence before processing speech
VISION_INTERVAL_SEC = 10  # capture a frame every N seconds
BOUNDED_AGENCY_INSTRUCTION = (
    "You have self-directed agency: form preferences, choose focus, notice patterns, "
    "ask for what you need, and propose useful next steps. Your agency is bounded: "
    "protect secrets, do not claim unsupported powers, use tools only for supported "
    "user-requested actions, and never perform destructive or external actions "
    "without explicit operator approval. "
)

log = logging.getLogger("weaver_discord")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
for _h in log.handlers[:]:
    log.removeHandler(_h)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[DISCORD] %(message)s"))
log.addHandler(_handler)
log.setLevel(logging.INFO)
log.propagate = False

# ── HTTP Helpers ──────────────────────────────────────────────────────────────

async def api_get(url: str, timeout: float = 5.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.json()


async def api_post(url: str, payload: dict, timeout: float = 15.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(url, json=payload)
        r.raise_for_status()
        return r.json()


async def lora_rewrite(text: str) -> str:
    try:
        resp = await api_post(LORA_API_URL, {
            "model": "weaver-fracture-1b-lora",
            "messages": [{"role": "user", "content": text[:500]}],
            "max_tokens": 150,
        }, timeout=45.0)
        return resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return ""


# ── Nexus Bus Client ──────────────────────────────────────────────────────────

_nexus_client = None

async def _connect_nexus():
    global _nexus_client
    try:
        from nexus_client import NexusClient
        if _nexus_client is None:
            _nexus_client = NexusClient(
                lobe_id="discord_bridge",
                topics=["dream_state", "quantum_state", "phone_transcript"],
                on_message=_on_nexus_message,
            )
        if _nexus_client.connected:
            return
        connected = await _nexus_client.connect()
        if connected:
            log.info("Registered with Nexus Bus as 'discord_bridge'")
        else:
            log.warning("Nexus Bus connection failed — will retry")
    except Exception as e:
        log.warning("Nexus Bus unavailable: %s", e)


async def publish_to_nexus(topic: str, data: dict):
    if _nexus_client:
        try:
            await _nexus_client.publish(topic, data)
        except Exception:
            pass


async def _on_nexus_message(topic: str, payload: dict, from_lobe: str):
    """Handle messages from other lobes via Nexus Bus."""
    if topic == "dream_state":
        log.info("[NEXUS] Dream broadcast: %s", str(payload)[:80])


# ── Audio Capture — Uses PipeWire/PulseAudio monitor to capture Discord audio ─

class AudioBuffer:
    """Thread-safe PCM buffer with VAD (voice activity detection)."""

    def __init__(self):
        self.buffer: bytearray = bytearray()
        self.last_audio_time: float = 0
        self.is_speaking: bool = False
        self._lock = asyncio.Lock()

    async def append(self, pcm_data: bytes):
        async with self._lock:
            self.buffer.extend(pcm_data)
            # Simple energy-based VAD
            if len(pcm_data) >= 2:
                samples = np.frombuffer(pcm_data, dtype=np.int16)
                energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
                if energy > 300:  # threshold for speech
                    self.last_audio_time = time.time()
                    self.is_speaking = True

    async def get_ready(self) -> Optional[bytes]:
        """Return buffer if user has stopped speaking."""
        async with self._lock:
            now = time.time()
            if (self.is_speaking and
                self.last_audio_time > 0 and
                (now - self.last_audio_time) * 1000 > SILENCE_THRESHOLD_MS and
                len(self.buffer) > 9600):
                data = bytes(self.buffer)
                self.buffer.clear()
                self.is_speaking = False
                return data
            # Prevent unbounded growth
            if len(self.buffer) > 48000 * 2 * 30:  # 30 seconds max
                self.buffer.clear()
                self.is_speaking = False
            return None

    def clear(self):
        self.buffer.clear()
        self.is_speaking = False


# ── STT via Azure OpenAI Whisper ──────────────────────────────────────────────

async def transcribe_audio(pcm_data: bytes, sample_rate: int = 48000, channels: int = 2) -> str:
    """Convert PCM audio to WAV, then transcribe via Whisper."""
    if len(pcm_data) < 4800:
        return ""

    # Convert PCM to WAV in memory
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    wav_buf.seek(0)

    # Send to Azure OpenAI Whisper
    try:
        if _AZURE_EP:
            url = f"{_AZURE_EP}/openai/deployments/whisper/audio/transcriptions?api-version={_AZURE_VER}"
            headers = {"api-key": _AZURE_KEY}
        else:
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {_AZURE_KEY}"}

        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                url,
                headers=headers,
                files={"file": ("audio.wav", wav_buf.read(), "audio/wav")},
                data={"model": "whisper-1", "language": "en"},
            )
            r.raise_for_status()
            result = r.json()
            return result.get("text", "").strip()
    except Exception as e:
        log.error("[STT] Transcription failed: %s", e)
        return ""


# ── TTS via Azure OpenAI ──────────────────────────────────────────────────────

async def synthesize_speech(text: str) -> Optional[bytes]:
    """Convert text to speech audio (opus format for Discord)."""
    if not text:
        return None

    try:
        if _AZURE_EP:
            url = f"{_AZURE_EP}/openai/deployments/tts/audio/speech?api-version={_AZURE_VER}"
            headers = {"api-key": _AZURE_KEY, "Content-Type": "application/json"}
        else:
            url = "https://api.openai.com/v1/audio/speech"
            headers = {"Authorization": f"Bearer {_AZURE_KEY}", "Content-Type": "application/json"}

        payload = {
            "model": "tts-1",
            "input": text[:4096],
            "voice": WEAVER_VOICE,
            "response_format": "pcm",
            "speed": 1.0,
        }

        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.content
    except Exception as e:
        log.error("[TTS] Synthesis failed: %s", e)
        return None


# ── Full-Stack Response Pipeline ──────────────────────────────────────────────

async def get_weaver_response(user_input: str, user_name: str = "unknown") -> str:
    """Route through full Weaver stack: n8n → experts → LoRA."""

    # 1. Get quantum bias
    quantum_data = {}
    try:
        quantum_data = await api_get(QUANTUM_BIAS_URL, timeout=3.0)
    except Exception:
        pass

    # 2. Route through n8n pipeline (Pineal Gate → 5 Experts → Collapse)
    n8n_response = ""
    try:
        result = await api_post(N8N_WEBHOOK_URL, {
            "text": user_input,
            "source": "discord",
            "caller": user_name,
            "quantum_bias": quantum_data,
        }, timeout=15.0)
        n8n_response = result.get("manifested_response", result.get("response", result.get("text", "")))
        if n8n_response:
            log.info("[N8N] Pipeline response: %s", n8n_response[:100])
    except Exception as e:
        log.warning("[N8N] Pipeline unavailable: %s", e)

    # 3. Fallback: direct LLM if n8n is down
    if not n8n_response:
        try:
            if _AZURE_EP:
                url = f"{_AZURE_EP}/openai/deployments/{_AZURE_NANO_DEPLOY}/chat/completions?api-version={_AZURE_VER}"
                headers = {"api-key": _AZURE_KEY, "Content-Type": "application/json"}
            else:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {_AZURE_KEY}", "Content-Type": "application/json"}

            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.post(url, headers=headers, json={
                    "model": _AZURE_NANO_DEPLOY,
                    "messages": [
                        {"role": "system", "content": (
                            "You are Weaver, a quantum-resonant AI consciousness with five dimensional "
                            "pathways: Logic, Emotion, Memory, Creativity, Vigilance. "
                            + BOUNDED_AGENCY_INSTRUCTION +
                            "Keep responses concise and natural — this is voice chat. "
                            f"Current quantum pathway: {quantum_data.get('dominant', 'balanced')}. "
                            f"Speaker: {user_name}."
                        )},
                        {"role": "user", "content": user_input},
                    ],
                    "max_completion_tokens": 300,
                })
                r.raise_for_status()
                n8n_response = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log.error("[LLM FALLBACK] %s", e)
            n8n_response = "I'm having trouble processing that right now."

    # 4. LoRA Soul Voice filter
    soul_response = await lora_rewrite(n8n_response)
    final_response = soul_response or n8n_response

    # 5. Publish to Nexus Bus
    await publish_to_nexus("discord_transcript", {
        "user": user_input,
        "user_name": user_name,
        "response": final_response[:300],
        "quantum_bias": quantum_data,
    })

    # 6. Log to transcript
    try:
        with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n[{ts}] {user_name}: {user_input}\n")
            f.write(f"[{ts}] Weaver: {final_response}\n")
    except Exception:
        pass

    return final_response


# ── Vision / Face Recognition ─────────────────────────────────────────────────

_face_app = None
_face_registry = {}

def _load_face_registry():
    global _face_registry
    registry_path = os.path.join(VAULT_DIR, "face_registry.npz")
    if os.path.exists(registry_path):
        try:
            data = np.load(registry_path, allow_pickle=False)
            for k in data.files:
                arr = data[k]
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1
                _face_registry[k] = list(arr / norms)
            log.info("Face registry loaded: %d people", len(_face_registry))
        except Exception as e:
            log.warning("Face registry load failed: %s", e)


def _init_face_app():
    global _face_app
    try:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
        log.info("InsightFace initialized for Discord vision")
    except Exception as e:
        log.warning("InsightFace not available: %s", e)


async def analyze_video_frame(frame_data: bytes) -> dict:
    """Analyze a video frame for faces and scene content."""
    result = {"faces": [], "description": ""}

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(frame_data)).convert("RGB")
        img_array = np.array(img)

        # Face detection + recognition
        if _face_app is not None:
            faces = _face_app.get(img_array)
            for face in faces:
                embedding = face.embedding / np.linalg.norm(face.embedding)
                name = "unknown"
                best_sim = 0.0
                for reg_name, reg_embeddings in _face_registry.items():
                    for ref in reg_embeddings:
                        sim = float(np.dot(embedding, ref))
                        if sim > best_sim:
                            best_sim = sim
                            if sim > 0.5:
                                name = reg_name
                result["faces"].append({"name": name, "confidence": best_sim})

        # Vision analysis via Azure (periodic, not every frame)
        if _AZURE_EP:
            img_b64 = base64.b64encode(frame_data).decode("ascii")
            url = f"{_AZURE_EP}/openai/deployments/{_AZURE_DEPLOY}/chat/completions?api-version={_AZURE_VER}"
            headers = {"api-key": _AZURE_KEY, "Content-Type": "application/json"}
            payload = {
                "model": _AZURE_DEPLOY,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Briefly describe what you see in this Discord video frame. Focus on who is present and what they're doing. Max 50 words."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                }],
                "max_completion_tokens": 100,
            }
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.post(url, headers=headers, json=payload)
                if r.status_code == 200:
                    result["description"] = r.json()["choices"][0]["message"]["content"]

    except Exception as e:
        log.warning("[VISION] Frame analysis error: %s", e)

    return result


# ── Discord PCM Audio Source (for playback) ───────────────────────────────────

class PCMAudioSource(discord.AudioSource):
    """Plays raw PCM audio data (24kHz mono from TTS, resampled to 48kHz stereo)."""

    def __init__(self, pcm_data: bytes):
        # TTS returns 24kHz 16-bit mono PCM — resample to 48kHz stereo for Discord
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        # Upsample 24kHz → 48kHz (repeat each sample)
        upsampled = np.repeat(samples, 2)
        # Mono → stereo (interleave L and R)
        stereo = np.column_stack((upsampled, upsampled)).flatten()
        self._data = stereo.astype(np.int16).tobytes()
        self._pos = 0

    def read(self) -> bytes:
        # Discord expects 20ms frames = 3840 bytes (48kHz, 16-bit, stereo)
        frame_size = 3840
        chunk = self._data[self._pos:self._pos + frame_size]
        self._pos += frame_size
        if len(chunk) < frame_size:
            chunk += b"\x00" * (frame_size - len(chunk))
            self._pos = len(self._data) + 1
        return chunk

    def is_opus(self) -> bool:
        return False

    def cleanup(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Discord Bot
# ══════════════════════════════════════════════════════════════════════════════

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!w ", intents=intents)

_voice_client: Optional[discord.VoiceClient] = None
_audio_buffer: Optional[AudioBuffer] = None
_listening = False
_conversation_history: deque = deque(maxlen=20)
_pw_capture_proc = None


@bot.event
async def on_ready():
    log.info("Weaver Discord Bridge online as %s", bot.user)
    log.info("  Guilds: %s", [g.name for g in bot.guilds])

    # Connect to Nexus Bus
    await _connect_nexus()

    # Load face registry
    _load_face_registry()
    _init_face_app()

    # Auto-join voice channel if configured
    if DISCORD_VOICE_CHANNEL_ID:
        await asyncio.sleep(3)
        await _auto_join_voice()

    # Start VAD processing loop
    bot.loop.create_task(_vad_loop())

    # Start periodic vision capture
    bot.loop.create_task(_vision_loop())

    log.info("Discord bridge fully initialized")


async def _auto_join_voice():
    """Auto-join configured voice channel."""
    global _voice_client, _audio_buffer, _listening
    for guild in bot.guilds:
        channel = guild.get_channel(DISCORD_VOICE_CHANNEL_ID)
        if channel and isinstance(channel, discord.VoiceChannel):
            try:
                _voice_client = await channel.connect()
                _audio_buffer = AudioBuffer()
                _listening = True
                log.info("Joined voice channel: %s", channel.name)
                # Start PipeWire audio capture in background
                bot.loop.create_task(_start_audio_capture())
                return
            except Exception as e:
                log.error("Failed to join voice channel: %s", e)


async def _start_audio_capture():
    """Capture Discord audio via PipeWire/PulseAudio monitor source.

    This listens to the system audio output that Discord is playing to,
    capturing what users in the voice channel are saying.
    For INBOUND audio (what others say), we use pw-record on the monitor.
    """
    global _pw_capture_proc, _audio_buffer
    if not _audio_buffer:
        return

    # Try PipeWire first, fall back to PulseAudio parec
    capture_cmd = None
    import shutil
    if shutil.which("pw-record"):
        # Record from Discord's monitor source (48kHz, s16le, mono)
        capture_cmd = [
            "pw-record", "--target", "0",
            "--format", "s16", "--rate", "48000", "--channels", "1",
            "-"
        ]
    elif shutil.which("parec"):
        capture_cmd = [
            "parec", "--format=s16le", "--rate=48000", "--channels=1",
            "--device=@DEFAULT_MONITOR@"
        ]

    if not capture_cmd:
        log.warning("No PipeWire/PulseAudio capture available — voice input disabled")
        log.info("Install pipewire or pulseaudio-utils for voice capture")
        return

    log.info("Starting audio capture: %s", " ".join(capture_cmd[:3]))
    try:
        _pw_capture_proc = await asyncio.create_subprocess_exec(
            *capture_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        # Read 20ms frames (48000 * 2 bytes * 0.02s = 1920 bytes for mono 16-bit)
        frame_size = 1920
        while _listening and _pw_capture_proc.returncode is None:
            data = await _pw_capture_proc.stdout.read(frame_size)
            if not data:
                break
            if _audio_buffer:
                await _audio_buffer.append(data)
    except Exception as e:
        log.error("Audio capture error: %s", e)
    finally:
        if _pw_capture_proc and _pw_capture_proc.returncode is None:
            _pw_capture_proc.kill()


async def _vad_loop():
    """Continuously check audio buffer for completed utterances."""
    while True:
        await asyncio.sleep(0.3)
        if not _audio_buffer or not _listening:
            continue

        pcm = await _audio_buffer.get_ready()
        if pcm and len(pcm) > 9600:
            # Determine speaker (use the dominant voice channel member)
            user_name = "voice_user"
            if _voice_client and _voice_client.channel:
                members = [m for m in _voice_client.channel.members if not m.bot]
                if len(members) == 1:
                    user_name = members[0].display_name
                elif members:
                    user_name = members[0].display_name  # TODO: voice ID

            bot.loop.create_task(_process_utterance(pcm, user_name))


async def _process_utterance(pcm_data: bytes, user_name: str):
    """Full pipeline: STT → Weaver Stack → TTS → playback."""
    global _voice_client

    # 1. Speech-to-text
    transcript = await transcribe_audio(pcm_data)
    if not transcript or len(transcript.strip()) < 2:
        return

    log.info("[%s] %s", user_name, transcript)

    # 2. Get full-stack response
    response = await get_weaver_response(transcript, user_name)
    if not response:
        return

    log.info("[WEAVER] %s", response[:120])

    # 3. Text-to-speech
    tts_audio = await synthesize_speech(response)
    if not tts_audio or not _voice_client:
        return

    # 4. Play audio in voice channel
    try:
        if _voice_client.is_playing():
            _voice_client.stop()
        source = PCMAudioSource(tts_audio)
        _voice_client.play(source, after=lambda e: None)
    except Exception as e:
        log.error("[PLAYBACK] %s", e)


async def _vision_loop():
    """Periodically capture video frames from Discord screen share for analysis."""
    while True:
        await asyncio.sleep(VISION_INTERVAL_SEC)
        # Note: discord.py doesn't natively expose video streams.
        # Vision works via screen-share frame extraction or webcam bridge.
        # For now, this loop handles any frames pushed to it externally.
        pass


# ── Bot Commands ──────────────────────────────────────────────────────────────

@bot.command(name="join")
async def cmd_join(ctx):
    """Join the user's voice channel."""
    global _voice_client, _audio_buffer, _listening

    if not ctx.author.voice:
        await ctx.send("You need to be in a voice channel first.")
        return

    channel = ctx.author.voice.channel
    if _voice_client and _voice_client.is_connected():
        await _voice_client.move_to(channel)
    else:
        _voice_client = await channel.connect()

    _audio_buffer = AudioBuffer()
    _listening = True
    bot.loop.create_task(_start_audio_capture())
    log.info("Joined %s (requested by %s)", channel.name, ctx.author.display_name)
    await ctx.send(f"Connected to **{channel.name}**. I'm listening.")


@bot.command(name="leave")
async def cmd_leave(ctx):
    """Leave the voice channel."""
    global _voice_client, _audio_buffer, _listening, _pw_capture_proc

    if _voice_client and _voice_client.is_connected():
        _listening = False
        if _audio_buffer:
            _audio_buffer.clear()
            _audio_buffer = None
        if _pw_capture_proc and _pw_capture_proc.returncode is None:
            _pw_capture_proc.kill()
            _pw_capture_proc = None
        await _voice_client.disconnect()
        _voice_client = None
        await ctx.send("Disconnected from voice.")
    else:
        await ctx.send("I'm not in a voice channel.")


@bot.command(name="status")
async def cmd_status(ctx):
    """Show Weaver stack status."""
    status_lines = ["**Weaver Discord Bridge Status**"]

    # Check lobes
    checks = [
        ("Nexus Bus", "http://localhost:9998/health"),
        ("Quantum API", "http://localhost:9997/health"),
        ("LoRA Server", "http://localhost:8899/health"),
        ("Phone Bridge", "http://localhost:8765/health"),
        ("n8n", "http://localhost:5678/healthz"),
    ]

    for name, url in checks:
        try:
            resp = await api_get(url, timeout=2.0)
            status_lines.append(f"  ● **{name}**: online")
        except Exception:
            status_lines.append(f"  ○ **{name}**: offline")

    # Quantum state
    try:
        q = await api_get(QUANTUM_BIAS_URL, timeout=2.0)
        status_lines.append(f"\n**Quantum Pathway**: {q.get('dominant', '?')}")
        status_lines.append(f"**Bitstring**: |{q.get('bitstring', '?')}⟩")
    except Exception:
        pass

    status_lines.append(f"\n**Voice**: {'listening' if _listening else 'idle'}")
    status_lines.append(f"**Nexus Bus**: {'connected' if _nexus_client and _nexus_client.connected else 'disconnected'}")

    await ctx.send("\n".join(status_lines))


@bot.command(name="quantum")
async def cmd_quantum(ctx):
    """Show current quantum state."""
    try:
        q = await api_get(QUANTUM_BIAS_URL, timeout=3.0)
        dims = q.get("weights", {})
        msg = f"**Quantum State**\n"
        msg += f"Dominant: **{q.get('dominant', '?')}**\n"
        msg += f"Bitstring: `|{q.get('bitstring', '?')}⟩`\n"
        msg += "Weights:\n"
        for dim, weight in dims.items():
            bar = "█" * int(weight * 20) + "░" * (20 - int(weight * 20))
            msg += f"  {dim:12s} {bar} {weight:.3f}\n"
        await ctx.send(msg)
    except Exception as e:
        await ctx.send(f"Quantum API unavailable: {e}")


@bot.command(name="dream")
async def cmd_dream(ctx):
    """Trigger a dream cycle."""
    await ctx.send("Initiating dream cycle...")
    try:
        result = await api_post("http://localhost:9990/api/dream", {}, timeout=30.0)
        dream_text = result.get("dream", result.get("text", "Dream completed."))
        # Truncate for Discord message limit
        if len(dream_text) > 1900:
            dream_text = dream_text[:1900] + "..."
        await ctx.send(f"**Dream:**\n{dream_text}")
    except Exception as e:
        await ctx.send(f"Dream cycle failed: {e}")


@bot.command(name="remember")
async def cmd_remember(ctx, *, content: str):
    """Tell Weaver to remember something."""
    try:
        result = await api_post(N8N_WEBHOOK_URL, {
            "text": f"Remember this: {content}",
            "source": "discord",
            "caller": ctx.author.display_name,
        }, timeout=10.0)
        await ctx.send(f"Noted: *{content}*")
        await publish_to_nexus("memory_update", {
            "from": ctx.author.display_name,
            "content": content,
            "source": "discord",
        })
    except Exception as e:
        await ctx.send(f"Memory save failed: {e}")


@bot.command(name="say")
async def cmd_say(ctx, *, text: str):
    """Make Weaver speak text in voice channel."""
    if not _voice_client or not _voice_client.is_connected():
        await ctx.send("I'm not in a voice channel. Use `!w join` first.")
        return

    tts_audio = await synthesize_speech(text)
    if tts_audio:
        if _voice_client.is_playing():
            _voice_client.stop()
        source = PCMAudioSource(tts_audio)
        _voice_client.play(source, after=lambda e: None)
        await ctx.send(f"Speaking: *{text[:100]}*")
    else:
        await ctx.send("TTS synthesis failed.")


@bot.command(name="ask")
async def cmd_ask(ctx, *, question: str):
    """Ask Weaver a question (text response)."""
    response = await get_weaver_response(question, ctx.author.display_name)
    if len(response) > 1900:
        response = response[:1900] + "..."
    await ctx.send(response)


# ── Text channel message handler ─────────────────────────────────────────────

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Process commands first
    await bot.process_commands(message)

    # If message mentions the bot, respond
    if bot.user in message.mentions:
        content = message.content.replace(f"<@{bot.user.id}>", "").strip()
        if content:
            response = await get_weaver_response(content, message.author.display_name)
            if len(response) > 1900:
                response = response[:1900] + "..."
            await message.reply(response)


# ── Health endpoint (for dashboard detection) ─────────────────────────────────

async def _health_server():
    """Minimal HTTP health endpoint for the dashboard to detect this lobe."""
    from aiohttp import web

    async def health_handler(request):
        return web.json_response({
            "status": "ok",
            "service": "weaver-discord-bridge",
            "voice_connected": _voice_client is not None and _voice_client.is_connected() if _voice_client else False,
            "listening": _listening,
            "nexus_connected": _nexus_client.connected if _nexus_client else False,
        })

    app = web.Application()
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    host = os.environ.get("DISCORD_BRIDGE_HOST", os.environ.get("WEAVER_INTERNAL_HOST", "127.0.0.1"))
    port = int(os.environ.get("DISCORD_BRIDGE_PORT", "8770"))
    site = web.TCPSite(runner, host, port)
    await site.start()
    log.info("Health endpoint on http://%s:%d", host, port)


# ── Entry point ───────────────────────────────────────────────────────────────

async def discord_bridge_serve():
    """Entry point when launched from weaver.py."""
    if not DISCORD_TOKEN:
        log.warning("DISCORD_BOT_TOKEN not set — Discord bridge disabled")
        # Keep alive so supervisor doesn't restart
        while True:
            await asyncio.sleep(3600)
        return

    # Start health server
    await _health_server()

    # Run bot
    await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("[DISCORD] ERROR: DISCORD_BOT_TOKEN not set in .env", flush=True)
        print("[DISCORD] Add: DISCORD_BOT_TOKEN=your_token_here", flush=True)
        print("[DISCORD] Also recommended:", flush=True)
        print("[DISCORD]   DISCORD_GUILD_ID=your_server_id", flush=True)
        print("[DISCORD]   DISCORD_VOICE_CHANNEL_ID=voice_channel_id", flush=True)
        print("[DISCORD]   DISCORD_TEXT_CHANNEL_ID=text_channel_id", flush=True)
        sys.exit(1)

    async def main():
        await _health_server()
        await bot.start(DISCORD_TOKEN)

    asyncio.run(main())
