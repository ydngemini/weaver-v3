#!/usr/bin/env python3
"""
weaver_dashboard.py — Weaver v3 Full-Stack Live Dashboard
==========================================================
Comprehensive real-time dashboard aggregating all Weaver subsystems.
Exposes via ngrok for a constant public URL.

Port: 9990 (default)
Public: auto-tunneled via ngrok

Features:
  - Real-time lobe health polling (all HTTP endpoints)
  - Nexus Bus WebSocket live feed
  - Quantum state + pentagon MoE visualization
  - Vault file readers (transcripts, people, dreams)
  - System metrics + uptime tracking
  - SSE stream for zero-refresh browser updates
  - Auto ngrok tunnel with persistent public URL
"""

import asyncio
import json
import os
import re
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from memory_manager import default_vault_dir

PORT = int(os.environ.get("WEAVER_DASHBOARD_PORT", "9990"))
PROJ = os.path.dirname(os.path.abspath(__file__))
VAULT = str(default_vault_dir())
N8N_WEBHOOK_URL = (
    os.environ.get("N8N_WEBHOOK_URL")
    or os.environ.get("WEAVER_N8N_WEBHOOK_URL")
    or "http://127.0.0.1:5678/webhook/weaverv5soulbind/1.%2520input%2520gateway/weaver-input"
)
BRAIN_API_URL = os.environ.get("WEAVER_BRAIN_API_URL", "http://127.0.0.1:8093").rstrip("/")

app = FastAPI(title="Weaver Live Dashboard", version="1.0.0")

WEAVER_LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="Weaver logo">
  <rect width="64" height="64" rx="14" fill="#07080c"/>
  <path d="M32 6 56 24 47 53H17L8 24 32 6Z" fill="#101624" stroke="#e6b84a" stroke-width="3" stroke-linejoin="round"/>
  <path d="M19 22 25 43 32 29 39 43 45 22" fill="none" stroke="#e6b84a" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="32" cy="32" r="4" fill="#34d4ff"/>
</svg>"""

# ── State ────────────────────────────────────────────────────────────────────

_boot_time = time.time()
_poll_count = 0
_ngrok_url: Optional[str] = None
_lobe_states: dict = {}
_quantum_state: dict = {}
_nexus_feed: deque = deque(maxlen=100)
_nexus_stats = {"msg_count": 0, "lobes_seen": set(), "topics_seen": set()}
_sse_subscribers: list[asyncio.Queue] = []

LOBES = [
    ("Nexus Bus",       "http://127.0.0.1:9998/health",  9998, "WebSocket pub/sub broker"),
    ("AWS Brain API",   "http://127.0.0.1:8093/health",  8093, "Bedrock/Nova unified cortex"),
    ("Headless UI",     "http://127.0.0.1:8093/health",  8093, "Headless Nova presence"),
    ("Trained Voice",   "http://127.0.0.1:8092/health",  8092, "OpenVoice cloned voice"),
    ("Codebase API",    "http://127.0.0.1:8091/health",  8091, "Bounded self-inspection API"),
    ("Quantum Soul",    None,                            None, "IBM Quantum 7-qubit loop"),
    ("Quantum API",     "http://127.0.0.1:9997/health",  9997, "Quantum state HTTP server"),
    ("Akashic Hub",     "http://127.0.0.1:9995/health",  9995, "Shared vector state"),
    ("Pineal Gate",     None,                            None, "Pentagon MoE router"),
    ("LoRA Server",     "http://127.0.0.1:8899/health",  8899, "1B Llama Soul Voice"),
    ("Qwen3B Branch",    "http://127.0.0.1:8898/health",  8898, "Local Qwen branch"),
    ("Phone Bridge",    "http://127.0.0.1:8765/health",  8765, "Twilio telephony"),
    ("Health Dashboard","http://127.0.0.1:9996/health",  9996, "Legacy traffic-light"),
    ("ProactivePulse",  None,                            None, "Quantum resonance monitor"),
    ("Dream State",     None,                            None, "Autonomous reflection"),
    ("n8n Workflow",    "http://127.0.0.1:5678/healthz",  5678, "Workflow orchestrator"),
    ("Discord Bridge", "http://127.0.0.1:8770/health",   8770, "Discord voice/vision"),
]


def _latency_class(latency_ms: float | None) -> str:
    if latency_ms is None:
        return "unknown"
    if latency_ms < 220:
        return "fast"
    if latency_ms < 1000:
        return "watch"
    return "slow"


def _fmt_ms(latency_ms: float | None) -> str:
    if latency_ms is None:
        return "n/a"
    if latency_ms >= 1000:
        return f"{latency_ms / 1000:.2f}s"
    return f"{latency_ms:.0f}ms"


def _weaver_key() -> str:
    if os.environ.get("WEAVER_LLM_KEY"):
        return os.environ["WEAVER_LLM_KEY"]
    for path in (os.path.join(PROJ, ".env"), "/etc/default/caddy"):
        try:
            if not os.path.exists(path):
                continue
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("WEAVER_LLM_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            continue
    return ""


async def _fetch_json(url: str, *, headers: dict | None = None, timeout: float = 3.0) -> tuple[dict, float | None, str]:
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(url, headers=headers or {})
            latency_ms = (time.monotonic() - start) * 1000
            if r.status_code != 200:
                return {}, round(latency_ms, 1), f"HTTP {r.status_code}"
            try:
                return r.json(), round(latency_ms, 1), ""
            except Exception:
                return {}, round(latency_ms, 1), "non-json response"
    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        return {}, round(latency_ms, 1), str(exc)[:120]


# ── Health polling ───────────────────────────────────────────────────────────

async def _check_http(name: str, url: str, desc: str) -> dict:
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(url)
            latency_ms = round((time.monotonic() - start) * 1000, 1)
            if r.status_code == 200:
                extra = {}
                try:
                    extra = r.json()
                except Exception:
                    pass
                return {"name": name, "status": "online", "desc": desc, "detail": extra,
                        "latency_ms": latency_ms, "latency_class": _latency_class(latency_ms)}
            return {"name": name, "status": "degraded", "desc": desc,
                    "detail": f"HTTP {r.status_code}", "latency_ms": latency_ms,
                    "latency_class": _latency_class(latency_ms)}
    except httpx.ConnectError:
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        return {"name": name, "status": "offline", "desc": desc, "detail": "Connection refused",
                "latency_ms": latency_ms, "latency_class": _latency_class(None)}
    except Exception as e:
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        return {"name": name, "status": "offline", "desc": desc, "detail": str(e)[:80],
                "latency_ms": latency_ms, "latency_class": _latency_class(None)}


async def _check_process(name: str, desc: str, pattern: str) -> dict:
    try:
        proc = await asyncio.create_subprocess_shell(
            f"pgrep -f '{pattern}'",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0 and stdout.strip():
            return {"name": name, "status": "online", "desc": desc, "detail": "Process running"}
        return {"name": name, "status": "offline", "desc": desc, "detail": "Not found"}
    except Exception as e:
        return {"name": name, "status": "offline", "desc": desc, "detail": str(e)[:80]}


async def _check_quantum_soul() -> dict:
    state_file = os.path.join(VAULT, "quantum_state.txt")
    try:
        if os.path.exists(state_file):
            age = time.time() - os.path.getmtime(state_file)
            if age < 600:
                return {"name": "Quantum Soul", "status": "online",
                        "desc": "IBM Quantum 7-qubit loop",
                        "detail": f"Last measurement {int(age)}s ago"}
            return {"name": "Quantum Soul", "status": "stale",
                    "desc": "IBM Quantum 7-qubit loop",
                    "detail": f"Last measurement {int(age / 60)}m ago"}
        return {"name": "Quantum Soul", "status": "offline",
                "desc": "IBM Quantum 7-qubit loop", "detail": "No state file"}
    except Exception as e:
        return {"name": "Quantum Soul", "status": "offline",
                "desc": "IBM Quantum 7-qubit loop", "detail": str(e)[:80]}


async def _check_dream_state() -> dict:
    dream_file = os.path.join(VAULT, "weaver_dreams.md")
    try:
        if os.path.exists(dream_file):
            age_h = (time.time() - os.path.getmtime(dream_file)) / 3600
            status = "online" if age_h < 6 else "stale"
            return {"name": "Dream State", "status": status,
                    "desc": "Autonomous reflection",
                    "detail": f"Last dream {age_h:.1f}h ago"}
        return {"name": "Dream State", "status": "offline",
                "desc": "Autonomous reflection", "detail": "No dream log"}
    except Exception as e:
        return {"name": "Dream State", "status": "offline",
                "desc": "Autonomous reflection", "detail": str(e)[:80]}


async def poll_all_lobes() -> list[dict]:
    global _poll_count
    _poll_count += 1
    tasks = []
    for name, url, port, desc in LOBES:
        if name == "Quantum Soul":
            tasks.append(_check_quantum_soul())
        elif name == "Dream State":
            tasks.append(_check_dream_state())
        elif name in ("Pineal Gate", "ProactivePulse"):
            tasks.append(_check_process(name, desc, "weaver.py"))
        elif url:
            tasks.append(_check_http(name, url, desc))
    results = await asyncio.gather(*tasks)
    result_list = list(results)
    global _lobe_states
    _lobe_states = {r["name"]: r for r in result_list}
    return result_list


# ── Quantum state reader ────────────────────────────────────────────────────

PATHWAYS = ["Awakening", "Resonance", "Echo", "Prophet", "Fracture", "Weaver", "Void"]
DIMENSION_MAP = {
    "Awakening": "logic", "Resonance": "emotion", "Echo": "memory",
    "Prophet": "creativity", "Fracture": "vigilance", "Weaver": "synthesis", "Void": "entropy",
}


def read_quantum_state() -> dict:
    global _quantum_state
    state_file = os.path.join(VAULT, "quantum_state.txt")
    if not os.path.exists(state_file):
        return _quantum_state

    try:
        text = Path(state_file).read_text(encoding="utf-8").strip()
        if not text:
            return _quantum_state

        result = {
            "raw": text,
            "dominant": "unknown",
            "secondary": None,
            "bitstring": None,
            "timestamp": None,
            "weights": {"logic": 0.5, "emotion": 0.5, "memory": 0.5, "creativity": 0.5, "vigilance": 0.5},
        }

        ts_match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", text)
        if ts_match:
            result["timestamp"] = ts_match.group(1)

        bit_match = re.search(r"\|([01]+)⟩", text)
        if bit_match:
            result["bitstring"] = bit_match.group(1)

        dom_match = re.search(r"reveals? (\w+) as the Dominant Pathway", text)
        if dom_match:
            result["dominant"] = dom_match.group(1)

        sec_match = re.search(r"with (\w+) resonating", text)
        if sec_match:
            result["secondary"] = sec_match.group(1)

        dominant = result["dominant"]
        if dominant in DIMENSION_MAP and DIMENSION_MAP[dominant] in result["weights"]:
            result["weights"][DIMENSION_MAP[dominant]] = 0.95

        secondary = result.get("secondary")
        if secondary and secondary in DIMENSION_MAP and DIMENSION_MAP[secondary] in result["weights"]:
            result["weights"][DIMENSION_MAP[secondary]] = 0.85

        bitstring = result.get("bitstring")
        if bitstring and len(bitstring) >= 5:
            dims = ["logic", "emotion", "memory", "creativity", "vigilance"]
            for i, dim in enumerate(dims):
                if i < len(bitstring):
                    result["weights"][dim] = 0.95 if int(bitstring[i]) == 0 else 0.65

        _quantum_state = result
        return result
    except Exception:
        return _quantum_state


# ── Vault file readers ───────────────────────────────────────────────────────

def read_vault_file(filename: str, tail_lines: int = 30) -> str:
    fpath = os.path.join(VAULT, filename)
    if not os.path.exists(fpath):
        return ""
    try:
        lines = Path(fpath).read_text(encoding="utf-8").strip().splitlines()
        return "\n".join(lines[-tail_lines:])
    except Exception:
        return ""


def read_people_memory() -> list[dict]:
    fpath = os.path.join(VAULT, "people_memory.md")
    if not os.path.exists(fpath):
        return []
    try:
        text = Path(fpath).read_text(encoding="utf-8")
        people = []
        for line in text.strip().splitlines():
            line = line.strip()
            if line.startswith("- **") and "**" in line[4:]:
                end = line.index("**", 4)
                name = line[4:end]
                desc = line[end + 2:].strip().lstrip("—").lstrip(" —").strip()
                people.append({"name": name, "summary": desc[:200]})
        return people
    except Exception:
        return []


def read_dreams(count: int = 3) -> list[dict]:
    fpath = os.path.join(VAULT, "weaver_dreams.md")
    if not os.path.exists(fpath):
        return []
    try:
        text = Path(fpath).read_text(encoding="utf-8")
        blocks = text.split("---")
        dreams = []
        for block in reversed(blocks):
            block = block.strip()
            if not block:
                continue
            ts_match = re.search(r"### Dream — (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", block)
            ts = ts_match.group(1) if ts_match else "unknown"
            content = re.sub(r"### Dream — .*\n?", "", block).strip()
            if content:
                dreams.append({"timestamp": ts, "content": content[:500]})
            if len(dreams) >= count:
                break
        return dreams
    except Exception:
        return []


def _tail_jsonl(path: str, limit: int = 10) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    events: list[dict] = []
    for line in reversed(lines[-max(limit * 4, limit):]):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        content = event.get("content") or event.get("text") or event.get("summary") or ""
        if not content and isinstance(event.get("payload"), dict):
            content = json.dumps(event["payload"], ensure_ascii=False)
        events.append({
            "ts": event.get("ts") or event.get("timestamp") or event.get("time") or "",
            "kind": event.get("kind") or event.get("type") or "event",
            "source": event.get("source") or "",
            "speaker": event.get("speaker") or "",
            "content": str(content)[:360],
        })
        if len(events) >= limit:
            break
    return events


def read_memory_events(limit: int = 10) -> list[dict]:
    sources = [
        os.path.join(VAULT, "weaver_memory_events.jsonl"),
        os.path.join(VAULT, "weaver_browser_memory.jsonl"),
    ]
    events: list[dict] = []
    for source in sources:
        events.extend(_tail_jsonl(source, limit))
    events.sort(key=lambda e: e.get("ts") or "", reverse=True)
    return events[:limit]


async def read_brain_snapshot() -> dict:
    health, health_ms, health_err = await _fetch_json("http://127.0.0.1:8093/health", timeout=2.5)
    key = _weaver_key()
    headers = {"X-Weaver-Key": key} if key else {}
    state: dict = {}
    models: dict = {}
    state_ms = None
    models_ms = None
    state_err = ""
    models_err = ""
    if key:
        state, state_ms, state_err = await _fetch_json("http://127.0.0.1:8093/state", headers=headers, timeout=3.0)
        models, models_ms, models_err = await _fetch_json("http://127.0.0.1:8093/v1/models", headers=headers, timeout=3.0)
    else:
        state_err = "missing WEAVER_LLM_KEY"
        models_err = state_err

    routes = []
    model_data = models.get("data", [])
    if not isinstance(model_data, list):
        model_data = []
    for model in model_data[:14]:
        routes.append({
            "id": model.get("id") or model.get("alias") or "unknown",
            "model_id": model.get("model_id") or model.get("id") or "",
            "region": model.get("region") or "",
            "purpose": model.get("purpose") or "",
            "orchestrated": bool(model.get("orchestrated")),
            "voice_native": bool(model.get("voice_native")),
            "multimodal": bool(model.get("multimodal")),
        })

    status = "online" if health.get("status") in ("ok", "online") else "offline"
    if health_err:
        status = "offline"
    elif state_err or models_err:
        status = "degraded"

    return {
        "status": status,
        "latency_ms": health_ms,
        "state_latency_ms": state_ms,
        "models_latency_ms": models_ms,
        "latency_class": _latency_class(health_ms),
        "error": state_err or models_err or health_err,
        "active": health.get("active"),
        "default_model": health.get("default_model") or state.get("default_model"),
        "headless_model": state.get("headless_model"),
        "headless_thought_model": state.get("headless_thought_model"),
        "headless_dream_model": state.get("headless_dream_model"),
        "thoughts": state.get("thoughts", 0),
        "dreams": state.get("dreams", 0),
        "memory_events": state.get("memory_events", 0),
        "last_error": state.get("last_error", ""),
        "last_thought": str(state.get("last_thought", ""))[:220],
        "last_dream": str(state.get("last_dream", ""))[:300],
        "routes": routes,
    }


async def read_voice_snapshot() -> dict:
    data, latency_ms, err = await _fetch_json("http://127.0.0.1:8092/health", timeout=4.0)
    status = "online" if data.get("status") in ("ok", "online") else "offline"
    if err:
        status = "offline"
    return {
        "status": status,
        "latency_ms": latency_ms,
        "latency_class": _latency_class(latency_ms if not err else None),
        "service": data.get("service") or data.get("name") or "trained voice",
        "detail": data,
        "error": err,
    }


async def read_codebase_snapshot() -> dict:
    data, latency_ms, err = await _fetch_json("http://127.0.0.1:8091/health", timeout=2.5)
    status = "online" if data.get("status") in ("ok", "online") else "offline"
    if err:
        status = "offline"
    return {
        "status": status,
        "latency_ms": latency_ms,
        "latency_class": _latency_class(latency_ms if not err else None),
        "root": data.get("root", ""),
        "service": data.get("service") or "codebase-api",
        "error": err,
    }


# ── Nexus Bus WebSocket listener ─────────────────────────────────────────────

async def _nexus_listener():
    """Background task: connect to Nexus Bus and capture messages."""
    import websockets
    while True:
        try:
            async with websockets.connect("ws://127.0.0.1:9999", ping_interval=20) as ws:
                await ws.send(json.dumps({"action": "register", "lobe_id": "live_dashboard"}))
                await ws.send(json.dumps({"action": "subscribe", "topics": [
                    "vision", "quantum_state", "identity", "manifested_response",
                    "overmind_directive", "routing", "gate", "interference",
                    "dream_state", "proactive_pulse",
                ]}))

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        if msg.get("type") == "broadcast":
                            entry = {
                                "from": msg.get("from", "?"),
                                "topic": msg.get("topic", ""),
                                "payload": json.dumps(msg.get("payload", {}))[:300],
                                "ts": msg.get("ts", datetime.now().isoformat()),
                            }
                            _nexus_feed.append(entry)
                            _nexus_stats["msg_count"] += 1
                            if msg.get("from"):
                                _nexus_stats["lobes_seen"].add(msg["from"])
                            if msg.get("topic"):
                                _nexus_stats["topics_seen"].add(msg["topic"])
                            await _broadcast_sse({"type": "nexus", "data": entry})
                        elif msg.get("type") == "sync":
                            for m in msg.get("messages", []):
                                entry = {
                                    "from": m.get("from", "?"),
                                    "topic": m.get("topic", ""),
                                    "payload": json.dumps(m.get("payload", {}))[:300],
                                    "ts": m.get("ts", ""),
                                }
                                _nexus_feed.append(entry)
                    except Exception:
                        pass
        except Exception:
            await asyncio.sleep(5)


# ── SSE broadcasting ────────────────────────────────────────────────────────

async def _broadcast_sse(event: dict):
    dead = []
    for i, q in enumerate(_sse_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.append(i)
    for i in reversed(dead):
        _sse_subscribers.pop(i)


# ── Background poller ───────────────────────────────────────────────────────

async def _poll_loop():
    while True:
        lobes = await poll_all_lobes()
        qs = read_quantum_state()
        brain, voice, codebase = await asyncio.gather(
            read_brain_snapshot(),
            read_voice_snapshot(),
            read_codebase_snapshot(),
        )
        uptime = time.time() - _boot_time
        online = sum(1 for l in lobes if l["status"] == "online")
        payload = {
            "type": "poll",
            "lobes": lobes,
            "quantum": qs,
            "brain": brain,
            "voice": voice,
            "codebase": codebase,
            "memory_events": read_memory_events(8),
            "online": online,
            "total": len(lobes),
            "uptime": round(uptime),
            "poll_count": _poll_count,
            "nexus_msg_count": _nexus_stats["msg_count"],
            "nexus_lobes": len(_nexus_stats["lobes_seen"]),
            "nexus_topics": len(_nexus_stats["topics_seen"]),
            "ngrok_url": _ngrok_url,
        }
        await _broadcast_sse(payload)
        await asyncio.sleep(5)


# ── Cloudflare tunnel ────────────────────────────────────────────────────────

def _find_cloudflared() -> Optional[str]:
    """Locate cloudflared binary — bundled in project dir or system PATH."""
    local = os.path.join(PROJ, "cloudflared")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    for d in os.environ.get("PATH", "").split(":"):
        p = os.path.join(d, "cloudflared")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


async def _start_tunnel():
    global _ngrok_url
    cf = _find_cloudflared()
    if not cf:
        print("[DASHBOARD] cloudflared not found — no public URL", flush=True)
        return

    try:
        proc = await asyncio.create_subprocess_exec(
            cf, "tunnel", "--url", f"http://127.0.0.1:{PORT}",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        for _ in range(60):
            try:
                line = await asyncio.wait_for(proc.stderr.readline(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            if not line:
                break
            text = line.decode(errors="replace")
            match = re.search(r"(https://[a-zA-Z0-9_-]+\.trycloudflare\.com)", text)
            if match:
                _ngrok_url = match.group(1)
                print(f"\n{'='*60}", flush=True)
                print(f"  WEAVER DASHBOARD PUBLIC: {_ngrok_url}", flush=True)
                print(f"{'='*60}\n", flush=True)
                break
        await proc.wait()
    except Exception as e:
        print(f"[DASHBOARD] tunnel error: {e}", flush=True)


# ── FastAPI startup ──────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    asyncio.create_task(_poll_loop())
    asyncio.create_task(_nexus_listener())
    asyncio.create_task(_start_tunnel())


# ── API endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "weaver-live-dashboard", "port": PORT, "ngrok": _ngrok_url}


@app.get("/api/state")
async def api_state():
    lobes = await poll_all_lobes()
    qs = read_quantum_state()
    brain, voice, codebase = await asyncio.gather(
        read_brain_snapshot(),
        read_voice_snapshot(),
        read_codebase_snapshot(),
    )
    return {
        "lobes": lobes,
        "quantum": qs,
        "brain": brain,
        "voice": voice,
        "codebase": codebase,
        "memory_events": read_memory_events(10),
        "online": sum(1 for l in lobes if l["status"] == "online"),
        "total": len(lobes),
        "uptime": round(time.time() - _boot_time),
        "nexus_feed": list(_nexus_feed)[-20:],
        "nexus_stats": {
            "msg_count": _nexus_stats["msg_count"],
            "lobes_seen": list(_nexus_stats["lobes_seen"]),
            "topics_seen": list(_nexus_stats["topics_seen"]),
        },
        "people": read_people_memory(),
        "dreams": read_dreams(3),
        "transcript": read_vault_file("weaver_transcript.txt", 20),
        "phone_transcript": read_vault_file("weaver_phone_transcript.txt", 15),
        "ngrok_url": _ngrok_url,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/stream")
async def sse_stream(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _sse_subscribers.append(q)

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        finally:
            if q in _sse_subscribers:
                _sse_subscribers.remove(q)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/vault/{filename}")
async def vault_file(filename: str):
    allowed = {"weaver_transcript.txt", "weaver_phone_transcript.txt",
               "quantum_state.txt", "people_memory.md", "weaver_dreams.md",
               "cloud_vision_memory.md"}
    if filename not in allowed:
        return {"error": "not allowed"}
    return {"content": read_vault_file(filename, 50)}


# ── Interactive API endpoints ──────────────────────────────────────────────

@app.post("/api/chat")
async def api_chat(request: Request):
    """Send a message through the full Weaver stack (n8n → experts → LoRA)."""
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return {"error": "empty message"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            resp = await c.post(N8N_WEBHOOK_URL, json={
                "text": text,
                "source": "dashboard",
            })
            if resp.status_code == 200:
                data = resp.json()
                return {"response": (data.get("manifested_response") or data.get("response") or data.get("text") or ""),
                        "metadata": {k: v for k, v in data.items() if k != "manifested_response"}}
            return {
                "error": f"n8n returned {resp.status_code}",
                "detail": resp.text[:200],
                "url": N8N_WEBHOOK_URL,
            }
    except httpx.ConnectError:
        return {"error": "n8n not reachable — is the workflow running?"}
    except Exception as e:
        return {"error": str(e)[:200]}


@app.post("/api/dream")
async def api_trigger_dream():
    """Force a dream cycle now by touching the transcript file to reset idle timer."""
    dream_file = os.path.join(VAULT, "weaver_dreams.md")
    dream_prompt = (
        "[DREAM MODE] Weaver is dreaming on demand. "
        "Reflect on all recent interactions. Find patterns, unresolved threads, "
        "creative connections. Be specific. Reference names and topics. 2-4 paragraphs."
    )
    n8n_error = ""
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            resp = await c.post(N8N_WEBHOOK_URL, json={
                "text": dream_prompt,
                "source": "dream_trigger",
            })
            if resp.status_code == 200:
                data = resp.json()
                dream_text = data.get("manifested_response") or data.get("response") or data.get("text") or ""
                if dream_text:
                    from datetime import datetime as _dt
                    ts = _dt.now().strftime("%Y-%m-%d %H:%M")
                    entry = f"\n\n---\n### Dream — {ts} (triggered)\n{dream_text}\n"
                    with open(dream_file, "a", encoding="utf-8") as f:
                        f.write(entry)
                return {"dream": dream_text, "route": "n8n", "url": N8N_WEBHOOK_URL}
            n8n_error = f"n8n returned {resp.status_code} at {N8N_WEBHOOK_URL}: {resp.text[:200]}"
    except httpx.ConnectError:
        n8n_error = f"n8n not reachable at {N8N_WEBHOOK_URL}"
    except Exception as e:
        n8n_error = str(e)[:200]

    key = _weaver_key()
    headers = {"X-Weaver-Key": key} if key else {}
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            resp = await c.post(
                f"{BRAIN_API_URL}/trigger/dream",
                headers=headers,
                json={"reason": "dashboard-manual"},
            )
            if resp.status_code == 200:
                data = resp.json()
                dream_text = data.get("dream", "")
                if dream_text:
                    from datetime import datetime as _dt
                    ts = _dt.now().strftime("%Y-%m-%d %H:%M")
                    entry = f"\n\n---\n### Dream — {ts} (triggered, brain fallback)\n{dream_text}\n"
                    with open(dream_file, "a", encoding="utf-8") as f:
                        f.write(entry)
                return {"dream": dream_text, "route": "brain-fallback", "n8n_error": n8n_error}
            return {
                "error": f"{n8n_error}; brain fallback returned {resp.status_code}",
                "detail": resp.text[:200],
            }
    except httpx.ConnectError:
        return {"error": f"{n8n_error}; brain fallback not reachable at {BRAIN_API_URL}"}
    except Exception as e:
        return {"error": f"{n8n_error}; brain fallback failed: {str(e)[:160]}"}


@app.post("/api/call")
async def api_trigger_call(request: Request):
    """Trigger an outbound call from Weaver."""
    body = await request.json()
    to_number = body.get("to", "").strip()
    reason = body.get("reason", "Dashboard-triggered call")
    if not to_number:
        return {"error": "missing 'to' number"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            resp = await c.post("http://127.0.0.1:8765/call", json={
                "to": to_number,
                "reason": reason,
            })
            return resp.json()
    except Exception as e:
        return {"error": str(e)[:200]}


@app.post("/api/nexus")
async def api_publish_nexus(request: Request):
    """Publish a message to the Nexus Bus."""
    body = await request.json()
    topic = body.get("topic", "dashboard")
    payload = body.get("payload", {})
    try:
        import websockets as _ws
        async with _ws.connect("ws://127.0.0.1:9999") as conn:
            await conn.send(json.dumps({"topic": topic, "payload": payload, "source": "dashboard"}))
            return {"ok": True, "topic": topic}
    except Exception as e:
        return {"error": str(e)[:200]}


# ── Main HTML dashboard ─────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/favicon.svg", include_in_schema=False)
async def favicon_svg():
    return Response(content=WEAVER_LOGO_SVG, media_type="image/svg+xml")




DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WEAVER — Command Interface</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root {
  --void: #030408;
  --bg: #080a10;
  --surface: #0c0f18;
  --surface2: #111520;
  --surface3: #181d2a;
  --border: #1e2536;
  --border-hot: #2a3448;
  --text: #b8c0d4;
  --dim: #5a6578;
  --muted: #3d4654;
  --accent: #7c5cfc;
  --accent2: #a78bfa;
  --green: #00e87b;
  --red: #ff4444;
  --orange: #ffb830;
  --cyan: #34d4ff;
  --pink: #ec4899;
  --blue: #4488ff;
  --violet: #9b6dff;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font: 400 11.5px/1.5 'JetBrains Mono', monospace;
  min-height: 100vh;
  overflow-x: hidden;
}

/* Grain overlay */
body::after {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 9999;
  opacity: 0.025;
  mix-blend-mode: overlay;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* Header */
.header {
  background: rgba(8,10,16,0.92);
  backdrop-filter: blur(12px) saturate(120%);
  border-bottom: 1px solid var(--border);
  padding: 10px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky; top: 0; z-index: 100;
}
.header h1 {
  font: 700 15px/1 'JetBrains Mono', monospace;
  letter-spacing: 4px;
  text-transform: uppercase;
  background: linear-gradient(135deg, var(--cyan), var(--violet));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.header-right { display: flex; align-items: center; gap: 16px; }
.status-pill {
  display: flex; align-items: center; gap: 6px;
  padding: 4px 10px;
  border: 1px solid var(--border);
  border-radius: 12px;
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.status-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 6px var(--green);
  animation: pulse-glow 2.5s ease-in-out infinite;
}
.status-pill.off .status-dot { background: var(--orange); box-shadow: 0 0 6px var(--orange); }
@keyframes pulse-glow { 0%,100%{opacity:0.6} 50%{opacity:1} }
.public-link { color: var(--dim); font-size: 10px; text-decoration: none; }
.public-link:hover { color: var(--cyan); }
.clock { color: var(--dim); font-size: 10px; font-variant-numeric: tabular-nums; }

/* Tabs */
.tab-bar {
  display: flex;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  padding: 0 20px;
}
.tab-btn {
  padding: 10px 20px;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--dim);
  font: 600 10px/1 'JetBrains Mono', monospace;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  cursor: pointer;
  transition: all 0.2s;
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--cyan); border-bottom-color: var(--cyan); }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* Glass Panel */
.panel {
  background: rgba(12, 15, 24, 0.85);
  backdrop-filter: blur(8px);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px;
  transition: border-color 0.3s;
}
.panel:hover { border-color: var(--border-hot); }
.panel-hdr {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 10px;
}
.panel-title {
  font: 700 9px/1 'JetBrains Mono', monospace;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--dim);
}
.panel-badge {
  font-size: 9px;
  padding: 2px 7px;
  border-radius: 8px;
  background: rgba(52,212,255,0.1);
  color: var(--cyan);
  border: 1px solid rgba(52,212,255,0.2);
}

/* Overview Grid */
.overview-grid {
  display: grid;
  grid-template-columns: 1fr 320px;
  grid-template-rows: auto auto 1fr;
  gap: 12px;
  padding: 16px;
  max-width: 1600px;
  margin: 0 auto;
}
.stats-bar {
  grid-column: 1 / -1;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.stat-card {
  flex: 1;
  min-width: 130px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
  text-align: center;
}
.stat-label { font: 500 8px/1 'JetBrains Mono',monospace; color: var(--dim); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
.stat-val { font: 300 26px/1 'JetBrains Mono',monospace; font-variant-numeric: tabular-nums; }
.stat-val.green { color: var(--green); }
.stat-val.cyan { color: var(--cyan); }
.stat-val.violet { color: var(--violet); }
.stat-val.orange { color: var(--orange); }
.stat-val.pink { color: var(--pink); }

/* Lobe Grid */
.lobe-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
.lobe-card {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 12px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  transition: all 0.25s var(--ease-out-expo);
}
.lobe-card:hover { border-color: var(--border-hot); transform: translateY(-1px); }
.lobe-dot {
  width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
  transition: all 0.4s;
}
.lobe-dot.online { background: var(--green); box-shadow: 0 0 8px var(--green); }
.lobe-dot.offline { background: var(--red); box-shadow: 0 0 8px var(--red); }
.lobe-dot.stale { background: var(--orange); box-shadow: 0 0 8px var(--orange); }
.lobe-dot.degraded { background: var(--orange); }
.lobe-name { font-weight: 600; font-size: 11px; color: var(--text); }
.lobe-desc { font-size: 9px; color: var(--dim); }
.lobe-detail { font-size: 9px; color: var(--muted); margin-top: 2px; }
.lobe-body { flex: 1; min-width: 0; }
.lobe-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.latency-chip {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 1px 6px; border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--dim); font-size: 8px; font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.latency-chip.fast { color: var(--green); border-color: rgba(0,232,123,0.25); background: rgba(0,232,123,0.06); }
.latency-chip.watch { color: var(--orange); border-color: rgba(255,184,48,0.25); background: rgba(255,184,48,0.06); }
.latency-chip.slow { color: var(--red); border-color: rgba(255,68,68,0.25); background: rgba(255,68,68,0.06); }
.ops-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-bottom: 10px; }
.signal-card {
  background: linear-gradient(180deg, rgba(24,29,42,0.72), rgba(12,15,24,0.8));
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 9px 10px;
  min-width: 0;
}
.signal-label {
  color: var(--dim); font-size: 8px; font-weight: 700;
  letter-spacing: 1px; text-transform: uppercase; margin-bottom: 5px;
}
.signal-value {
  color: var(--text); font-size: 12px; font-weight: 700;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.signal-sub { color: var(--muted); font-size: 9px; margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.route-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; max-height: 235px; overflow-y: auto; padding-right: 2px; }
.route-card {
  border: 1px solid rgba(30,37,54,0.85);
  border-left: 2px solid rgba(52,212,255,0.55);
  border-radius: 8px;
  background: rgba(3,4,8,0.42);
  padding: 9px 10px;
  min-width: 0;
}
.route-card.orchestrated { border-left-color: var(--green); }
.route-head { display: flex; align-items: center; justify-content: space-between; gap: 6px; margin-bottom: 4px; }
.route-name { color: #eef3ff; font-weight: 700; font-size: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.route-flags { display: flex; gap: 4px; flex-shrink: 0; }
.route-flag {
  color: var(--dim); border: 1px solid var(--border);
  border-radius: 999px; padding: 1px 5px; font-size: 7px; text-transform: uppercase;
}
.route-flag.on { color: var(--green); border-color: rgba(0,232,123,0.25); }
.route-model { color: var(--cyan); font-size: 9px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.route-purpose { color: var(--dim); font-size: 9px; line-height: 1.45; margin-top: 3px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.memory-feed { max-height: 210px; overflow-y: auto; display: grid; gap: 7px; }
.memory-event {
  border: 1px solid rgba(30,37,54,0.75);
  border-radius: 8px;
  background: rgba(8,10,16,0.6);
  padding: 8px 10px;
}
.memory-meta { display: flex; justify-content: space-between; gap: 8px; color: var(--muted); font-size: 8px; margin-bottom: 4px; }
.memory-kind { color: var(--orange); font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
.memory-content { color: var(--text); font-size: 9.5px; line-height: 1.5; overflow-wrap: anywhere; }
.empty-state { color: var(--muted); font-size: 10px; padding: 8px 2px; }

/* Pentagon */
.pentagon-panel { display: flex; align-items: center; gap: 16px; }
.pentagon-canvas { width: 220px; height: 220px; }
.quantum-info { flex: 1; font-size: 10px; }
.q-dominant { font: 700 18px/1.2 'JetBrains Mono',monospace; color: var(--violet); margin-bottom: 6px; }
.q-bits { font: 300 14px/1 'JetBrains Mono',monospace; color: var(--cyan); margin-bottom: 8px; letter-spacing: 2px; }
.q-meta { color: var(--dim); line-height: 1.8; }
.q-raw { margin-top: 10px; font-size: 9px; color: var(--muted); line-height: 1.6; max-height: 80px; overflow-y: auto; }

/* Feed */
.feed-scroll { max-height: 260px; overflow-y: auto; mask-image: linear-gradient(to bottom, black 85%, transparent 100%); }
.feed-item { padding: 6px 0; border-bottom: 1px solid rgba(30,37,54,0.5); font-size: 10px; }
.feed-from { color: var(--cyan); font-weight: 600; }
.feed-topic { color: var(--violet); }
.feed-ts { color: var(--muted); float: right; font-size: 9px; }

/* Dreams + Transcript */
.scroll-block { max-height: 200px; overflow-y: auto; font-size: 10px; color: var(--dim); line-height: 1.7; white-space: pre-wrap; mask-image: linear-gradient(to bottom, black 85%, transparent 100%); }
.dream-entry { margin-bottom: 12px; padding: 8px; background: rgba(155,109,255,0.04); border-left: 2px solid var(--violet); border-radius: 0 6px 6px 0; }
.dream-ts { font-size: 9px; color: var(--muted); }
.dream-text { font-size: 10px; color: var(--text); line-height: 1.6; margin-top: 4px; }
.people-list { display: flex; flex-wrap: wrap; gap: 6px; }
.person-chip { padding: 4px 10px; background: rgba(52,212,255,0.06); border: 1px solid rgba(52,212,255,0.15); border-radius: 12px; font-size: 10px; color: var(--cyan); }

/* Console Tab */
.console-grid {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 12px;
  padding: 16px;
  max-width: 1600px;
  margin: 0 auto;
  height: calc(100vh - 100px);
}
.chat-panel { display: flex; flex-direction: column; }
.chat-messages {
  flex: 1; overflow-y: auto; padding: 12px;
  background: var(--void);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin: 10px 0;
  mask-image: linear-gradient(to bottom, transparent 0%, black 3%, black 95%, transparent 100%);
}
.chat-msg { margin-bottom: 10px; padding: 8px 12px; border-radius: 8px; font-size: 11px; line-height: 1.6; max-width: 85%; word-wrap: break-word; }
.chat-msg.user { background: rgba(124,92,252,0.12); border: 1px solid rgba(124,92,252,0.25); margin-left: auto; text-align: right; }
.chat-msg.assistant { background: rgba(52,212,255,0.08); border: 1px solid rgba(52,212,255,0.15); }
.chat-msg.system { background: rgba(90,101,120,0.08); border: 1px solid var(--border); color: var(--dim); font-style: italic; text-align: center; max-width: 100%; font-size: 10px; }
.chat-msg.error { background: rgba(255,68,68,0.08); border: 1px solid rgba(255,68,68,0.2); color: var(--red); }
.chat-input-row { display: flex; gap: 8px; }
.chat-input {
  flex: 1; background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
  padding: 10px 14px; color: var(--text); font: 400 11px/1.4 'JetBrains Mono',monospace; outline: none;
  transition: border-color 0.2s;
}
.chat-input:focus { border-color: var(--cyan); box-shadow: 0 0 0 2px rgba(52,212,255,0.08); }
textarea.chat-input { resize: vertical; min-height: 50px; }
.btn {
  padding: 8px 16px; border: 1px solid var(--border); border-radius: 6px;
  background: var(--surface2); color: var(--text); font: 600 10px/1 'JetBrains Mono',monospace;
  letter-spacing: 0.5px; text-transform: uppercase; cursor: pointer;
  transition: all 0.2s var(--ease-out-expo);
}
.btn:hover { border-color: var(--cyan); background: rgba(52,212,255,0.06); color: var(--cyan); }
.btn:active { transform: scale(0.97); }
.btn:disabled { opacity: 0.3; cursor: not-allowed; }
.btn.primary { border-color: var(--cyan); color: var(--cyan); background: rgba(52,212,255,0.08); }
.btn.dream { border-color: var(--violet); color: var(--violet); }
.btn.dream:hover { background: rgba(155,109,255,0.08); }
.btn.danger { border-color: var(--red); color: var(--red); }
.btn.danger:hover { background: rgba(255,68,68,0.08); }
.btn.green { border-color: var(--green); color: var(--green); }
.btn.green:hover { background: rgba(0,232,123,0.08); }
.action-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
.sidebar-section { margin-bottom: 12px; }
.meta-pre { background: var(--void); border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-size: 9px; color: var(--dim); max-height: 180px; overflow: auto; white-space: pre-wrap; word-break: break-all; }
.dream-result { background: var(--void); border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-size: 10px; color: var(--dim); max-height: 200px; overflow-y: auto; line-height: 1.6; }
.typing { color: var(--dim); animation: blink 1.2s infinite; }
@keyframes blink { 0%,100%{opacity:0.3} 50%{opacity:1} }

/* Neural Map */
.neural-wrap { position: relative; width: 100%; height: calc(100vh - 100px); background: radial-gradient(ellipse at center, #0a0c14 0%, #030408 100%); overflow: hidden; cursor: crosshair; }
.neural-wrap canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
.neural-wrap #neuralMain { pointer-events: auto; cursor: crosshair; }
.neural-hud {
  position: absolute; top: 16px; left: 16px;
  background: rgba(8,10,16,0.92); backdrop-filter: blur(12px);
  border: 1px solid rgba(52,212,255,0.15); border-radius: 10px;
  padding: 14px 18px; font-size: 10px; z-index: 10;
  box-shadow: 0 4px 30px rgba(0,0,0,0.5), inset 0 0 20px rgba(52,212,255,0.02);
}
.nhud-title { color: var(--cyan); font-size: 9px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; border-bottom: 1px solid rgba(52,212,255,0.1); padding-bottom: 6px; }
.nhud-row { display: flex; justify-content: space-between; gap: 24px; margin-bottom: 3px; }
.nhud-label { color: var(--dim); font-size: 9px; }
.nhud-val { color: var(--cyan); font-weight: 600; font-variant-numeric: tabular-nums; font-size: 10px; }
.nhud-sep { height: 1px; background: rgba(52,212,255,0.08); margin: 6px 0; }
.neural-tooltip {
  position: absolute; z-index: 20; pointer-events: none;
  background: rgba(8,10,16,0.95); backdrop-filter: blur(12px);
  border: 1px solid rgba(52,212,255,0.3); border-radius: 8px;
  padding: 10px 14px; min-width: 140px; transition: opacity 0.15s;
  box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}
.ntip-name { color: #fff; font-weight: 600; font-size: 11px; margin-bottom: 3px; }
.ntip-status { font-size: 9px; margin-bottom: 2px; }
.ntip-status.online { color: var(--green); }
.ntip-status.offline { color: var(--red); }
.ntip-detail { color: var(--dim); font-size: 9px; line-height: 1.4; }
.neural-info {
  position: absolute; top: 16px; right: 16px; width: 260px; z-index: 15;
  background: rgba(8,10,16,0.94); backdrop-filter: blur(12px);
  border: 1px solid rgba(52,212,255,0.2); border-radius: 10px;
  padding: 16px; box-shadow: 0 8px 40px rgba(0,0,0,0.6);
}
.ninfo-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 1px solid rgba(52,212,255,0.1); padding-bottom: 8px; }
.ninfo-title { color: #fff; font-weight: 600; font-size: 12px; }
.ninfo-close { background: none; border: none; color: var(--dim); font-size: 18px; cursor: pointer; padding: 0 4px; }
.ninfo-close:hover { color: var(--red); }
.ninfo-body { color: var(--dim); font-size: 10px; line-height: 1.6; }
.ninfo-body .ninfo-stat { display: flex; justify-content: space-between; margin-bottom: 3px; }
.ninfo-body .ninfo-stat span:last-child { color: var(--cyan); font-weight: 500; }
.ninfo-body .ninfo-bar { height: 3px; border-radius: 2px; background: rgba(52,212,255,0.1); margin: 4px 0 8px; overflow: hidden; }
.ninfo-body .ninfo-bar-fill { height: 100%; border-radius: 2px; transition: width 0.3s; }
.neural-controls {
  position: absolute; bottom: 16px; right: 16px; display: flex; gap: 6px; z-index: 10;
}
.nctr-btn {
  width: 32px; height: 32px; border-radius: 8px; border: 1px solid rgba(52,212,255,0.2);
  background: rgba(8,10,16,0.8); color: var(--cyan); font-size: 14px;
  cursor: pointer; display: flex; align-items: center; justify-content: center;
  transition: all 0.2s; backdrop-filter: blur(8px);
}
.nctr-btn:hover { background: rgba(52,212,255,0.1); border-color: var(--cyan); transform: scale(1.1); }
.neural-legend { position: absolute; bottom: 16px; left: 16px; display: flex; flex-wrap: wrap; gap: 10px; z-index: 10; max-width: 60%; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 9px; color: var(--dim); cursor: pointer; padding: 2px 6px; border-radius: 4px; transition: all 0.2s; }
.legend-item:hover { background: rgba(255,255,255,0.05); color: #fff; }
.legend-item.active { background: rgba(52,212,255,0.08); color: var(--cyan); }
.legend-dot { width: 8px; height: 8px; border-radius: 50%; box-shadow: 0 0 6px currentColor; }

@media (max-width: 1000px) {
  .overview-grid, .console-grid { grid-template-columns: 1fr; }
  .pentagon-panel { flex-direction: column; }
  .ops-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .route-list { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
  .stats-bar { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .stat-card { min-width: 0; }
  .ops-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<!-- SVG Filters -->
<svg width="0" height="0" style="position:absolute">
<defs>
  <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>
</svg>

<div class="header">
  <h1>WEAVER v3</h1>
  <div class="header-right">
    <a class="public-link" id="publicUrl" href="#" target="_blank" style="display:none"></a>
    <div class="status-pill" id="ssePill">
      <div class="status-dot"></div>
      <span id="sseText">INIT</span>
    </div>
    <span class="clock" id="clock"></span>
  </div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('overview',this)">Overview</button>
  <button class="tab-btn" onclick="switchTab('console',this)">Console</button>
  <button class="tab-btn" onclick="switchTab('neural',this)">Neural Map</button>
</div>

<!-- ══════════ OVERVIEW ══════════ -->
<div class="tab-content active" id="tab-overview">
<div class="overview-grid">
  <div class="stats-bar">
    <div class="stat-card"><div class="stat-label">Online</div><div class="stat-val green" id="sOnline">--</div></div>
    <div class="stat-card"><div class="stat-label">Uptime</div><div class="stat-val cyan" id="sUptime">--</div></div>
    <div class="stat-card"><div class="stat-label">Bus Msgs</div><div class="stat-val violet" id="sMsgs">0</div></div>
    <div class="stat-card"><div class="stat-label">Bus Lobes</div><div class="stat-val orange" id="sBusLobes">0</div></div>
    <div class="stat-card"><div class="stat-label">Topics</div><div class="stat-val pink" id="sTopics">0</div></div>
    <div class="stat-card"><div class="stat-label">Brain</div><div class="stat-val green" id="sBrain">--</div></div>
    <div class="stat-card"><div class="stat-label">Pathway</div><div class="stat-val violet" id="sPathway">--</div></div>
  </div>

  <div class="panel" style="grid-column:1">
    <div class="panel-hdr"><span class="panel-title">System Lobes</span><span class="panel-badge" id="lobeBadge">0/0</span></div>
    <div class="lobe-grid" id="lobeGrid"></div>
  </div>

  <div class="panel" style="grid-column:1">
    <div class="panel-hdr"><span class="panel-title">Cortex Routes</span><span class="panel-badge" id="brainBadge">--</span></div>
    <div class="ops-grid">
      <div class="signal-card"><div class="signal-label">Default</div><div class="signal-value" id="defaultModel">--</div><div class="signal-sub" id="brainLatency">--</div></div>
      <div class="signal-card"><div class="signal-label">Headless</div><div class="signal-value" id="headlessModel">--</div><div class="signal-sub" id="headlessMode">--</div></div>
      <div class="signal-card"><div class="signal-label">Voice</div><div class="signal-value" id="voiceHealth">--</div><div class="signal-sub" id="voiceLatency">--</div></div>
      <div class="signal-card"><div class="signal-label">Codebase</div><div class="signal-value" id="codebaseHealth">--</div><div class="signal-sub" id="codebaseLatency">--</div></div>
    </div>
    <div class="route-list" id="routeList"></div>
  </div>

  <div class="panel" style="grid-row: span 5">
    <div class="panel-hdr"><span class="panel-title">Quantum Pentagon</span></div>
    <div class="pentagon-panel">
      <canvas class="pentagon-canvas" id="pentCanvas" width="220" height="220"></canvas>
      <div class="quantum-info">
        <div class="q-bits" id="qBits">|-------&#10217;</div>
        <div class="q-dominant" id="qDom">--</div>
        <div class="q-meta" id="qMeta"></div>
        <div class="q-raw" id="qRaw"></div>
      </div>
    </div>
    <div style="margin-top:14px">
      <div class="panel-hdr"><span class="panel-title">Dream State</span></div>
      <div id="dreamBox" class="scroll-block" style="max-height:140px">No dreams yet</div>
    </div>
    <div style="margin-top:14px">
      <div class="panel-hdr"><span class="panel-title">People</span></div>
      <div class="people-list" id="peopleBox"></div>
    </div>
  </div>

  <div class="panel" style="grid-column:1">
    <div class="panel-hdr"><span class="panel-title">Memory / Trace Feed</span><span class="panel-badge" id="memoryBadge">0</span></div>
    <div class="memory-feed" id="memoryFeed"><div class="empty-state">Waiting for memory events...</div></div>
  </div>

  <div class="panel" style="grid-column:1">
    <div class="panel-hdr"><span class="panel-title">Nexus Bus Feed</span><span class="panel-badge" id="feedBadge">0</span></div>
    <div class="feed-scroll" id="feedScroll"></div>
  </div>

  <div class="panel" style="grid-column:1">
    <div class="panel-hdr"><span class="panel-title">Transcript</span></div>
    <div class="scroll-block" id="transcriptBox">Loading...</div>
  </div>
</div>
</div>

<!-- ══════════ CONSOLE ══════════ -->
<div class="tab-content" id="tab-console">
<div class="console-grid">
  <div class="chat-panel panel">
    <div class="panel-hdr"><span class="panel-title">Weaver Full-Stack Interface</span><span class="panel-badge" id="chatBadge">ready</span></div>
    <div class="chat-messages" id="chatBox">
      <div class="chat-msg system">Messages route through: 5 Expert Lobes → LoRA Soul Voice → Response</div>
    </div>
    <div class="chat-input-row">
      <input type="text" class="chat-input" id="chatIn" placeholder="Talk to Weaver..." autocomplete="off"/>
      <button class="btn primary" id="chatBtn" onclick="sendChat()">Send</button>
    </div>
    <div style="margin-top:10px">
      <div class="panel-title" style="margin-bottom:6px">RESPONSE METADATA</div>
      <pre class="meta-pre" id="chatMeta">awaiting input...</pre>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;gap:12px">
    <div class="panel">
      <div class="panel-title">Quick Actions</div>
      <div class="action-grid">
        <button class="btn dream" onclick="triggerDream()">Dream Now</button>
        <button class="btn green" onclick="triggerCall()">Call Me</button>
        <button class="btn primary" onclick="publishNexus()">Nexus Ping</button>
        <button class="btn" onclick="fetchState()">Refresh</button>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">Dream Result</div>
      <div class="dream-result" id="dreamOut">Click "Dream Now" to force a full-stack dream cycle.</div>
    </div>
    <div class="panel">
      <div class="panel-title">Nexus Publisher</div>
      <input type="text" class="chat-input" id="nxTopic" placeholder="Topic" value="dashboard" style="margin-bottom:6px"/>
      <textarea class="chat-input" id="nxPayload" placeholder='{"key":"value"}'></textarea>
      <button class="btn primary" style="margin-top:6px;width:100%" onclick="pubCustomNexus()">Publish to Bus</button>
    </div>
  </div>
</div>
</div>

<!-- ══════════ NEURAL MAP ══════════ -->
<div class="tab-content" id="tab-neural">
<div class="neural-wrap" id="neuralWrap">
  <canvas id="neuralBg"></canvas>
  <canvas id="neuralMain"></canvas>
  <canvas id="neuralFx"></canvas>
  <!-- HUD Panel -->
  <div class="neural-hud" id="neuralHud">
    <div class="nhud-title">NEURAL TOPOLOGY</div>
    <div class="nhud-row"><span class="nhud-label">Neurons</span><span class="nhud-val" id="nN">0</span></div>
    <div class="nhud-row"><span class="nhud-label">Firing</span><span class="nhud-val" id="nF">0</span></div>
    <div class="nhud-row"><span class="nhud-label">Synapses</span><span class="nhud-val" id="nS">0</span></div>
    <div class="nhud-row"><span class="nhud-label">Activity</span><span class="nhud-val" id="nA">0%</span></div>
    <div class="nhud-row"><span class="nhud-label">Signals/s</span><span class="nhud-val" id="nSig">0</span></div>
    <div class="nhud-row"><span class="nhud-label">Coherence</span><span class="nhud-val" id="nCoh">0%</span></div>
    <div class="nhud-sep"></div>
    <div class="nhud-row"><span class="nhud-label">Zoom</span><span class="nhud-val" id="nZoom">1.0x</span></div>
    <div class="nhud-row"><span class="nhud-label">Selected</span><span class="nhud-val" id="nSel">—</span></div>
  </div>
  <!-- Tooltip -->
  <div class="neural-tooltip" id="neuralTip" style="display:none">
    <div class="ntip-name" id="tipName"></div>
    <div class="ntip-status" id="tipStatus"></div>
    <div class="ntip-detail" id="tipDetail"></div>
  </div>
  <!-- Region info panel -->
  <div class="neural-info" id="neuralInfo" style="display:none">
    <div class="ninfo-header">
      <span class="ninfo-title" id="infoTitle">Region</span>
      <button class="ninfo-close" onclick="closeNeuralInfo()">×</button>
    </div>
    <div class="ninfo-body" id="infoBody"></div>
  </div>
  <!-- Controls -->
  <div class="neural-controls">
    <button class="nctr-btn" onclick="neuralReset()" title="Reset view">⟲</button>
    <button class="nctr-btn" onclick="neuralBurst()" title="Fire all">⚡</button>
    <button class="nctr-btn" id="nctrPause" onclick="neuralTogglePause()" title="Pause">▮▮</button>
  </div>
  <div class="neural-legend" id="neuralLegend"></div>
</div>
</div>

<script>
// ═══════════════════════════════════════════════════════
// WEAVER DASHBOARD — SOTA COMMAND INTERFACE
// ═══════════════════════════════════════════════════════

function switchTab(name, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  if (btn) btn.classList.add('active');
  if (name === 'neural') { neuralResize(); if (!_neuralRunning) startNeural(); }
}

// ── State ────────────────────────────────────
let _quantumWeights = [0.5, 0.5, 0.5, 0.5, 0.5];
let _neuralRunning = false;

function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function fmtMs(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return 'n/a';
  const n = Number(v);
  return n >= 1000 ? `${(n/1000).toFixed(2)}s` : `${Math.round(n)}ms`;
}

function latencyClass(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return 'unknown';
  if (Number(v) < 220) return 'fast';
  if (Number(v) < 1000) return 'watch';
  return 'slow';
}

function statusLabel(v) {
  return (v || 'offline').toString().toUpperCase();
}

// ── Data fetching ────────────────────────────
async function fetchState() {
  try {
    const r = await fetch('/api/state');
    if (!r.ok) return;
    const d = await r.json();
    updateLobes(d.lobes || []);
    updateQuantum(d.quantum || {});
    updateFeed(d.nexus_feed || []);
    updateTranscript(d.transcript || '');
    updatePeople(d.people || []);
    updateDreams(d.dreams || []);
    updateBrain(d.brain || {}, d.voice || {}, d.codebase || {});
    updateMemoryEvents(d.memory_events || []);
    document.getElementById('sOnline').textContent = `${d.online}/${d.total}`;
    document.getElementById('sUptime').textContent = fmtUp(d.uptime||0);
    document.getElementById('sMsgs').textContent = d.nexus_stats?.msg_count || 0;
    document.getElementById('sBusLobes').textContent = d.nexus_stats?.lobes_seen?.length || 0;
    document.getElementById('sTopics').textContent = d.nexus_stats?.topics_seen?.length || 0;
    document.getElementById('lobeBadge').textContent = `${d.online}/${d.total}`;
    if (d.ngrok_url) {
      const el = document.getElementById('publicUrl');
      el.href = d.ngrok_url; el.textContent = d.ngrok_url.replace('https://',''); el.style.display = 'inline-block';
    }
  } catch(e) { console.warn('fetch:', e.message); }
}

function fmtUp(s) {
  if (s < 60) return s+'s';
  if (s < 3600) return Math.floor(s/60)+'m';
  return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m';
}

function updateLobes(lobes) {
  const g = document.getElementById('lobeGrid');
  g.innerHTML = lobes.map(l => `
    <div class="lobe-card">
      <div class="lobe-dot ${esc(l.status)}"></div>
      <div class="lobe-body">
        <div class="lobe-top">
          <div class="lobe-name">${esc(l.name)}</div>
          <span class="latency-chip ${esc(l.latency_class || latencyClass(l.latency_ms))}">${fmtMs(l.latency_ms)}</span>
        </div>
        <div class="lobe-desc">${esc(l.desc)}</div>
        <div class="lobe-detail">${esc(typeof l.detail === 'object' ? (l.detail.status || l.detail.service || '') : (l.detail||''))}</div>
      </div>
    </div>`).join('');
  // Feed live status into neural map
  if (_neuralRunning) updateNeuralFromLobes(lobes);
}

function updateBrain(brain, voice, codebase) {
  const routes = Array.isArray(brain.routes) ? brain.routes : [];
  setText('sBrain', statusLabel(brain.status));
  setText('brainBadge', `${statusLabel(brain.status)} · ${fmtMs(brain.latency_ms)}`);
  setText('defaultModel', brain.default_model || 'unknown');
  setText('brainLatency', `${routes.length} routes · state ${fmtMs(brain.state_latency_ms)}`);
  setText('headlessModel', brain.headless_model || brain.headless_thought_model || 'unknown');
  setText('headlessMode', `${brain.thoughts || 0} thoughts · ${brain.dreams || 0} dreams`);
  setText('voiceHealth', statusLabel(voice.status));
  setText('voiceLatency', `${voice.service || 'trained voice'} · ${fmtMs(voice.latency_ms)}`);
  setText('codebaseHealth', statusLabel(codebase.status));
  setText('codebaseLatency', `${codebase.service || 'codebase'} · ${fmtMs(codebase.latency_ms)}`);

  const list = document.getElementById('routeList');
  if (!list) return;
  if (!routes.length) {
    list.innerHTML = `<div class="empty-state">${esc(brain.error || 'No cortex route data yet')}</div>`;
    return;
  }
  list.innerHTML = routes.map(r => `
    <div class="route-card ${r.orchestrated ? 'orchestrated' : ''}">
      <div class="route-head">
        <div class="route-name">${esc(r.id)}</div>
        <div class="route-flags">
          <span class="route-flag ${r.orchestrated ? 'on' : ''}">mesh</span>
          <span class="route-flag ${r.multimodal ? 'on' : ''}">vision</span>
          <span class="route-flag ${r.voice_native ? 'on' : ''}">voice</span>
        </div>
      </div>
      <div class="route-model">${esc(r.model_id || r.region || '')}</div>
      <div class="route-purpose">${esc(r.purpose || '')}</div>
    </div>`).join('');
}

function updateMemoryEvents(events) {
  const feed = document.getElementById('memoryFeed');
  const badge = document.getElementById('memoryBadge');
  if (badge) badge.textContent = events.length || 0;
  if (!feed) return;
  if (!events.length) {
    feed.innerHTML = '<div class="empty-state">No memory events reported yet</div>';
    return;
  }
  feed.innerHTML = events.map(e => {
    const time = (e.ts || '').split('T')[1]?.substring(0,8) || (e.ts || '').substring(0,16);
    const source = [e.source, e.speaker].filter(Boolean).join(' / ');
    return `<div class="memory-event">
      <div class="memory-meta"><span><span class="memory-kind">${esc(e.kind || 'event')}</span> ${esc(source)}</span><span>${esc(time)}</span></div>
      <div class="memory-content">${esc(e.content || '')}</div>
    </div>`;
  }).join('');
}

function updateQuantum(q) {
  if (!q || !q.weights) return;
  const dims = ['logic','emotion','memory','creativity','vigilance'];
  _quantumWeights = dims.map(d => q.weights[d] || 0.5);
  document.getElementById('qBits').textContent = q.bitstring ? `|${q.bitstring}⟩` : '|-------⟩';
  document.getElementById('qDom').textContent = q.dominant || '--';
  document.getElementById('sPathway').textContent = q.dominant || '--';
  let meta = '';
  if (q.secondary) meta += `Secondary: ${q.secondary}\n`;
  if (q.timestamp) meta += `Measured: ${q.timestamp}`;
  document.getElementById('qMeta').textContent = meta;
  document.getElementById('qRaw').textContent = q.raw ? q.raw.substring(q.raw.indexOf(']')+2) : '';
  drawPentagon();
}

function updateFeed(feed) {
  const el = document.getElementById('feedScroll');
  document.getElementById('feedBadge').textContent = feed.length;
  el.innerHTML = feed.slice(-15).reverse().map(f => `
    <div class="feed-item">
      <span class="feed-ts">${esc((f.ts||'').split('T')[1]?.substring(0,8)||'')}</span>
      <span class="feed-from">${esc(f.from)}</span>
      <span class="feed-topic">${esc(f.topic)}</span>
    </div>`).join('');
}

function updateTranscript(t) { document.getElementById('transcriptBox').textContent = t || 'No transcript yet'; }

function updatePeople(p) {
  document.getElementById('peopleBox').innerHTML = p.map(x => `<span class="person-chip">${esc(x.name)}</span>`).join('');
}

function updateDreams(dreams) {
  if (!dreams.length) { document.getElementById('dreamBox').innerHTML = '<span style="color:var(--muted)">No dreams yet</span>'; return; }
  document.getElementById('dreamBox').innerHTML = dreams.map(d => `
    <div class="dream-entry">
      <div class="dream-ts">${esc(d.timestamp)}</div>
      <div class="dream-text">${esc((d.content || '').substring(0,300))}</div>
    </div>`).join('');
}

// ── Pentagon Radar ──────────────────────────
function drawPentagon() {
  const c = document.getElementById('pentCanvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  const cx = W/2, cy = H/2, R = 90;
  const N = 5;
  const PHI = (2*Math.PI)/N;
  const labels = ['Logic','Emotion','Memory','Creativity','Vigilance'];
  const colors = ['#34d4ff','#ec4899','#ffb830','#9b6dff','#ff4444'];

  ctx.clearRect(0, 0, W, H);

  // Rings
  for (let ring = 1; ring <= 5; ring++) {
    const r = (R/5)*ring;
    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
      const a = (i%N)*PHI - Math.PI/2;
      const x = cx + r*Math.cos(a), y = cy + r*Math.sin(a);
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }
    ctx.closePath();
    ctx.strokeStyle = ring===5 ? 'rgba(52,212,255,0.3)' : 'rgba(52,212,255,0.07)';
    ctx.lineWidth = ring===5 ? 1.5 : 0.5;
    ctx.stroke();
  }

  // Axes
  for (let i = 0; i < N; i++) {
    const a = i*PHI - Math.PI/2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + R*Math.cos(a), cy + R*Math.sin(a));
    ctx.strokeStyle = 'rgba(52,212,255,0.1)';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }

  // Data polygon
  ctx.beginPath();
  for (let i = 0; i <= N; i++) {
    const idx = i%N;
    const a = idx*PHI - Math.PI/2;
    const r = R * Math.max(0.05, Math.min(1, _quantumWeights[idx]));
    const x = cx + r*Math.cos(a), y = cy + r*Math.sin(a);
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  }
  ctx.closePath();
  ctx.fillStyle = 'rgba(52,212,255,0.08)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(52,212,255,0.7)';
  ctx.lineWidth = 2;
  ctx.shadowColor = '#34d4ff';
  ctx.shadowBlur = 10;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Vertices + labels
  for (let i = 0; i < N; i++) {
    const a = i*PHI - Math.PI/2;
    const r = R * _quantumWeights[i];
    const x = cx + r*Math.cos(a), y = cy + r*Math.sin(a);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI*2);
    ctx.fillStyle = colors[i];
    ctx.shadowColor = colors[i];
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Label
    const lx = cx + (R+18)*Math.cos(a), ly = cy + (R+18)*Math.sin(a);
    ctx.font = '500 9px JetBrains Mono';
    ctx.fillStyle = 'rgba(184,192,212,0.7)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(labels[i], lx, ly);
  }

  // Center dot
  ctx.beginPath();
  ctx.arc(cx, cy, 3, 0, Math.PI*2);
  ctx.fillStyle = 'rgba(155,109,255,0.6)';
  ctx.fill();
}

// ── Console Chat ─────────────────────────────
const chatBox = document.getElementById('chatBox');
const chatIn = document.getElementById('chatIn');
chatIn.addEventListener('keydown', e => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendChat();} });

function addMsg(text, cls) {
  const d = document.createElement('div');
  d.className = 'chat-msg ' + cls;
  d.textContent = text;
  chatBox.appendChild(d);
  chatBox.scrollTop = chatBox.scrollHeight;
  return d;
}

async function sendChat() {
  const text = chatIn.value.trim();
  if (!text) return;
  chatIn.value = '';
  addMsg(text, 'user');
  document.getElementById('chatBtn').disabled = true;
  document.getElementById('chatBadge').textContent = 'thinking...';
  const t = addMsg('processing...', 'system typing');
  try {
    const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    const d = await r.json();
    t.remove();
    if (d.error) addMsg('Error: '+d.error, 'error');
    else {
      addMsg(d.response||'(empty)', 'assistant');
      if (d.metadata) document.getElementById('chatMeta').textContent = JSON.stringify(d.metadata, null, 2);
    }
  } catch(e) { t.remove(); addMsg('Network: '+e.message, 'error'); }
  document.getElementById('chatBtn').disabled = false;
  document.getElementById('chatBadge').textContent = 'ready';
}

async function triggerDream() {
  const el = document.getElementById('dreamOut');
  el.innerHTML = '<span class="typing">dreaming through full stack...</span>';
  try {
    const r = await fetch('/api/dream', {method:'POST'});
    const d = await r.json();
    el.textContent = d.error ? 'Error: '+d.error : (d.dream||'(empty)');
  } catch(e) { el.textContent = 'Error: '+e.message; }
}

async function triggerCall() {
  const to = prompt('Number to call:');
  if (!to) return;
  try {
    const r = await fetch('/api/call', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({to})});
    const d = await r.json();
    addMsg(d.error ? 'Call error: '+d.error : 'Call initiated to '+to, d.error?'error':'system');
  } catch(e) { addMsg('Call failed: '+e.message, 'error'); }
}

function publishNexus() { pubCustomNexus(); }

async function pubCustomNexus() {
  const topic = document.getElementById('nxTopic').value.trim()||'dashboard';
  let payload = {};
  try { const raw = document.getElementById('nxPayload').value.trim(); if(raw) payload = JSON.parse(raw); }
  catch(e) { addMsg('Invalid JSON', 'error'); return; }
  try {
    const r = await fetch('/api/nexus', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({topic,payload})});
    const d = await r.json();
    addMsg(d.ok ? `Published: ${topic}` : 'Error: '+(d.error||''), d.ok?'system':'error');
  } catch(e) { addMsg('Publish failed: '+e.message, 'error'); }
}

// ── Neural Map — Interactive SOTA Visualization ─────────────────
const REGIONS = [
  {id:"nexus_bus",label:"Nexus Bus",color:[52,212,255],cx:0.50,cy:0.10,r:0.055,n:90,desc:"WebSocket pub/sub broker",tier:"core"},
  {id:"quantum",label:"Quantum Soul",color:[236,72,153],cx:0.18,cy:0.20,r:0.048,n:70,desc:"7-qubit IBM Quantum circuit",tier:"core"},
  {id:"akashic",label:"Akashic Hub",color:[167,139,250],cx:0.50,cy:0.28,r:0.055,n:85,desc:"256-d shared vector state",tier:"core"},
  {id:"pineal",label:"Pineal Gate",color:[124,92,252],cx:0.82,cy:0.20,r:0.048,n:70,desc:"Pentagon MoE router",tier:"core"},
  {id:"logic",label:"Logic",color:[52,212,255],cx:0.14,cy:0.50,r:0.038,n:45,desc:"Analytical reasoning lobe",tier:"expert"},
  {id:"emotion",label:"Emotion",color:[236,72,153],cx:0.32,cy:0.50,r:0.038,n:45,desc:"Empathic resonance lobe",tier:"expert"},
  {id:"memory",label:"Memory",color:[245,158,11],cx:0.50,cy:0.50,r:0.038,n:45,desc:"Temporal recall lobe",tier:"expert"},
  {id:"creativity",label:"Creativity",color:[155,109,255],cx:0.68,cy:0.50,r:0.038,n:45,desc:"Generative ideation lobe",tier:"expert"},
  {id:"vigilance",label:"Vigilance",color:[255,68,68],cx:0.86,cy:0.50,r:0.038,n:45,desc:"Threat/priority detection",tier:"expert"},
  {id:"qwen",label:"Qwen 3B",color:[0,232,123],cx:0.32,cy:0.70,r:0.038,n:40,desc:"3B parameter MoE addon",tier:"inference"},
  {id:"lora",label:"LoRA Soul",color:[0,232,123],cx:0.68,cy:0.70,r:0.042,n:50,desc:"1B Llama personality filter",tier:"inference"},
  {id:"phone",label:"Phone",color:[68,136,255],cx:0.12,cy:0.76,r:0.032,n:30,desc:"Twilio telephony bridge",tier:"io"},
  {id:"vtv",label:"VTV",color:[245,158,11],cx:0.88,cy:0.76,r:0.032,n:30,desc:"Vision/audio perception",tier:"io"},
  {id:"dream",label:"Dream",color:[155,109,255],cx:0.25,cy:0.90,r:0.028,n:25,desc:"Autonomous reflection",tier:"io"},
  {id:"obsidian",label:"Obsidian",color:[100,116,139],cx:0.50,cy:0.90,r:0.028,n:20,desc:"Vault bidirectional sync",tier:"io"},
  {id:"pulse",label:"Pulse",color:[255,68,68],cx:0.75,cy:0.90,r:0.028,n:25,desc:"Proactive monitor",tier:"io"},
  {id:"discord",label:"Discord",color:[88,101,242],cx:0.50,cy:0.72,r:0.035,n:35,desc:"Discord voice/vision bridge",tier:"io"},
];
const LINKS = [
  [0,2],[0,3],[0,1],[0,12],[0,11],
  [1,2],[1,3],[2,3],[2,9],[2,10],
  [3,4],[3,5],[3,6],[3,7],[3,8],
  [4,10],[5,10],[6,10],[7,10],[8,10],
  [4,9],[5,9],[6,9],[7,9],[8,9],
  [10,0],[9,0],[11,0],[12,0],
  [13,2],[14,2],[15,3],[11,3],[12,2],
  [16,0],[16,2],[16,10],[16,3],
];
const TIER_COLORS = {core:'rgba(52,212,255,0.06)',expert:'rgba(124,92,252,0.04)',inference:'rgba(0,232,123,0.04)',io:'rgba(100,116,139,0.03)'};

let neurons = [], synapses = [], signals = [], regionCenters = {};
let nBg, nMain, nFx, ctxBg, ctxMain, ctxFx;
let _zoom = 1.0, _panX = 0, _panY = 0, _dragging = false, _dragStart = null;
let _hoveredRegion = null, _selectedRegion = null, _paused = false;
let _signalCount = 0, _signalRate = 0, _lastSignalCheck = 0;
let _lobeStatusMap = {};

function neuralResize() {
  const wrap = document.getElementById('neuralWrap');
  if (!wrap) return;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  if (W < 10 || H < 10) return;
  [nBg, nMain, nFx].forEach(c => { if(c){c.width=W; c.height=H;} });
  buildNeurons(W, H);
  drawBgGrid(W, H);
}

function drawBgGrid(W, H) {
  if (!ctxBg) return;
  ctxBg.clearRect(0, 0, W, H);
  // Radial gradient background
  const grad = ctxBg.createRadialGradient(W/2, H/2, 0, W/2, H/2, W*0.6);
  grad.addColorStop(0, 'rgba(52,212,255,0.015)');
  grad.addColorStop(0.5, 'rgba(124,92,252,0.008)');
  grad.addColorStop(1, 'transparent');
  ctxBg.fillStyle = grad;
  ctxBg.fillRect(0, 0, W, H);
  // Grid lines
  ctxBg.strokeStyle = 'rgba(52,212,255,0.03)';
  ctxBg.lineWidth = 0.5;
  const step = 60;
  for (let x = 0; x < W; x += step) { ctxBg.beginPath(); ctxBg.moveTo(x,0); ctxBg.lineTo(x,H); ctxBg.stroke(); }
  for (let y = 0; y < H; y += step) { ctxBg.beginPath(); ctxBg.moveTo(0,y); ctxBg.lineTo(W,y); ctxBg.stroke(); }
  // Region halos
  REGIONS.forEach(reg => {
    const cx = reg.cx*W, cy = reg.cy*H, radius = reg.r*Math.min(W,H)*2;
    const halo = ctxBg.createRadialGradient(cx, cy, 0, cx, cy, radius);
    halo.addColorStop(0, `rgba(${reg.color[0]},${reg.color[1]},${reg.color[2]},0.04)`);
    halo.addColorStop(1, 'transparent');
    ctxBg.fillStyle = halo;
    ctxBg.fillRect(cx-radius, cy-radius, radius*2, radius*2);
  });
}

function buildNeurons(W, H) {
  neurons = [];
  regionCenters = {};
  REGIONS.forEach(reg => {
    const cx = reg.cx*W, cy = reg.cy*H, spread = reg.r*Math.min(W,H);
    regionCenters[reg.id] = {x:cx, y:cy, r:spread, region:reg};
    for (let i = 0; i < reg.n; i++) {
      const angle = Math.random()*Math.PI*2;
      const dist = Math.pow(Math.random(),0.6)*spread;
      neurons.push({
        x: cx + Math.cos(angle)*dist,
        y: cy + Math.sin(angle)*dist,
        ox: cx, oy: cy,
        r: Math.random()*1.8+0.4,
        color: reg.color,
        fire: 0,
        region: reg.id,
        vx: (Math.random()-0.5)*0.2,
        vy: (Math.random()-0.5)*0.2,
        phase: Math.random()*Math.PI*2,
      });
    }
  });
  synapses = [];
  LINKS.forEach(([a,b]) => {
    const ra = REGIONS[a], rb = REGIONS[b];
    const na = neurons.filter(n => n.region === ra.id);
    const nb = neurons.filter(n => n.region === rb.id);
    const count = Math.min(4, na.length, nb.length);
    for (let i = 0; i < count; i++) {
      synapses.push({a:na[Math.floor(Math.random()*na.length)], b:nb[Math.floor(Math.random()*nb.length)], fromRegion:ra.id, toRegion:rb.id, pulse:0});
    }
  });
}

function startNeural() {
  nBg = document.getElementById('neuralBg');
  nMain = document.getElementById('neuralMain');
  nFx = document.getElementById('neuralFx');
  if (!nBg || !nMain || !nFx) return;
  ctxBg = nBg.getContext('2d');
  ctxMain = nMain.getContext('2d');
  ctxFx = nFx.getContext('2d');
  neuralResize();
  _neuralRunning = true;

  // Legend
  const leg = document.getElementById('neuralLegend');
  leg.innerHTML = REGIONS.map(r => `<div class="legend-item" data-region="${r.id}" onclick="legendClick('${r.id}')"><div class="legend-dot" style="background:rgb(${r.color.join(',')});box-shadow:0 0 6px rgb(${r.color.join(',')})"></div>${r.label}</div>`).join('');

  // Mouse/touch events
  nMain.addEventListener('mousemove', neuralMouseMove);
  nMain.addEventListener('mousedown', neuralMouseDown);
  nMain.addEventListener('mouseup', neuralMouseUp);
  nMain.addEventListener('mouseleave', neuralMouseLeave);
  nMain.addEventListener('wheel', neuralWheel, {passive:false});
  nMain.addEventListener('dblclick', neuralDblClick);

  requestAnimationFrame(neuralFrame);
}

function neuralMouseMove(e) {
  const rect = nMain.getBoundingClientRect();
  const mx = (e.clientX - rect.left - _panX) / _zoom;
  const my = (e.clientY - rect.top - _panY) / _zoom;

  if (_dragging && _dragStart) {
    _panX += e.clientX - _dragStart.x;
    _panY += e.clientY - _dragStart.y;
    _dragStart = {x: e.clientX, y: e.clientY};
    return;
  }

  // Check hover over regions
  let found = null;
  for (const [id, rc] of Object.entries(regionCenters)) {
    const dx = mx - rc.x, dy = my - rc.y;
    if (Math.sqrt(dx*dx+dy*dy) < rc.r*1.5) { found = id; break; }
  }

  if (found !== _hoveredRegion) {
    _hoveredRegion = found;
    const tip = document.getElementById('neuralTip');
    if (found) {
      const reg = REGIONS.find(r=>r.id===found);
      const status = _lobeStatusMap[found] || 'unknown';
      document.getElementById('tipName').textContent = reg.label;
      const statusEl = document.getElementById('tipStatus');
      statusEl.textContent = status === 'online' ? '● Online' : status === 'offline' ? '○ Offline' : '◌ Unknown';
      statusEl.className = 'ntip-status ' + status;
      document.getElementById('tipDetail').textContent = reg.desc;
      tip.style.display = 'block';
      tip.style.left = (e.clientX - rect.left + 15) + 'px';
      tip.style.top = (e.clientY - rect.top - 10) + 'px';
      nMain.style.cursor = 'pointer';
    } else {
      tip.style.display = 'none';
      nMain.style.cursor = 'crosshair';
    }
  } else if (_hoveredRegion) {
    const tip = document.getElementById('neuralTip');
    const rect2 = nMain.getBoundingClientRect();
    tip.style.left = (e.clientX - rect2.left + 15) + 'px';
    tip.style.top = (e.clientY - rect2.top - 10) + 'px';
  }
}

function neuralMouseDown(e) {
  if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
    _dragging = true;
    _dragStart = {x: e.clientX, y: e.clientY};
    nMain.style.cursor = 'grabbing';
    e.preventDefault();
  } else if (e.button === 0 && _hoveredRegion) {
    selectRegion(_hoveredRegion);
  }
}
function neuralMouseUp() { _dragging = false; nMain.style.cursor = _hoveredRegion ? 'pointer' : 'crosshair'; }
function neuralMouseLeave() { _dragging = false; _hoveredRegion = null; document.getElementById('neuralTip').style.display='none'; }

function neuralWheel(e) {
  e.preventDefault();
  const factor = e.deltaY > 0 ? 0.9 : 1.1;
  _zoom = Math.max(0.3, Math.min(4, _zoom * factor));
  document.getElementById('nZoom').textContent = _zoom.toFixed(1)+'x';
}

function neuralDblClick(e) {
  if (_hoveredRegion) {
    // Fire all neurons in hovered region
    neurons.filter(n => n.region === _hoveredRegion).forEach(n => { n.fire = 1; });
    // Send signal burst from this region
    const regionSynapses = synapses.filter(s => s.fromRegion === _hoveredRegion);
    regionSynapses.forEach(s => {
      signals.push({ax:s.a.x, ay:s.a.y, bx:s.b.x, by:s.b.y, t:0, color:s.a.color, size:3.5, trail:true});
      s.a.fire = 1;
    });
  }
}

function selectRegion(id) {
  _selectedRegion = id;
  document.getElementById('nSel').textContent = REGIONS.find(r=>r.id===id)?.label || '—';
  const info = document.getElementById('neuralInfo');
  const reg = REGIONS.find(r=>r.id===id);
  if (!reg) return;
  const status = _lobeStatusMap[id] || 'unknown';
  const neuronCount = neurons.filter(n=>n.region===id).length;
  const firingCount = neurons.filter(n=>n.region===id && n.fire>0).length;
  const connCount = synapses.filter(s=>s.fromRegion===id || s.toRegion===id).length;
  const activity = Math.round(firingCount/Math.max(neuronCount,1)*100);
  document.getElementById('infoTitle').textContent = reg.label;
  document.getElementById('infoBody').innerHTML = `
    <div class="ninfo-stat"><span>Status</span><span style="color:${status==='online'?'var(--green)':status==='offline'?'var(--red)':'var(--dim)'}">${status}</span></div>
    <div class="ninfo-stat"><span>Tier</span><span>${reg.tier}</span></div>
    <div class="ninfo-stat"><span>Neurons</span><span>${neuronCount}</span></div>
    <div class="ninfo-stat"><span>Connections</span><span>${connCount}</span></div>
    <div class="ninfo-stat"><span>Activity</span><span>${activity}%</span></div>
    <div class="ninfo-bar"><div class="ninfo-bar-fill" style="width:${activity}%;background:rgb(${reg.color.join(',')})"></div></div>
    <div style="color:var(--dim);font-size:9px;margin-top:6px">${reg.desc}</div>
    <div style="margin-top:8px"><button class="btn primary" style="font-size:9px;padding:4px 10px" onclick="fireRegion('${id}')">Fire Region</button></div>
  `;
  info.style.display = 'block';
}

function closeNeuralInfo() { document.getElementById('neuralInfo').style.display='none'; _selectedRegion=null; document.getElementById('nSel').textContent='—'; }
function legendClick(id) { selectRegion(id); document.querySelectorAll('.legend-item').forEach(el => el.classList.toggle('active', el.dataset.region===id)); }
function fireRegion(id) { neurons.filter(n=>n.region===id).forEach(n=>{n.fire=1;}); synapses.filter(s=>s.fromRegion===id).forEach(s=>{signals.push({ax:s.a.x,ay:s.a.y,bx:s.b.x,by:s.b.y,t:0,color:s.a.color,size:3,trail:true});}); }
function neuralReset() { _zoom=1; _panX=0; _panY=0; document.getElementById('nZoom').textContent='1.0x'; }
function neuralBurst() { neurons.forEach(n=>{n.fire=1;}); for(let i=0;i<20&&synapses.length;i++){const s=synapses[Math.floor(Math.random()*synapses.length)]; signals.push({ax:s.a.x,ay:s.a.y,bx:s.b.x,by:s.b.y,t:0,color:s.a.color,size:3.5,trail:true});} }
function neuralTogglePause() { _paused=!_paused; document.getElementById('nctrPause').textContent=_paused?'▶':'▮▮'; if(!_paused) requestAnimationFrame(neuralFrame); }

// Update lobe statuses from polling data
function updateNeuralFromLobes(lobes) {
  if (!lobes) return;
  const mapping = {'Nexus Bus':'nexus_bus','Quantum Soul':'quantum','Quantum API':'quantum','Akashic Hub':'akashic','Pineal Gate':'pineal','LoRA Server':'lora','Phone Bridge':'phone','VTV':'vtv','ProactivePulse':'pulse','Dream State':'dream','n8n Workflow':'nexus_bus','Discord Bridge':'discord'};
  lobes.forEach(l => {
    const regionId = mapping[l.name];
    if (regionId) {
      _lobeStatusMap[regionId] = l.status;
      if (l.status === 'online') {
        // Stochastic firing for online lobes
        neurons.filter(n=>n.region===regionId).forEach(n => { if(Math.random()<0.02) n.fire=1; });
      }
    }
  });
}

let _lastNeural = 0;
function neuralFrame(ts) {
  if (!_neuralRunning || _paused) return;
  const dt = Math.min((ts - _lastNeural)/16.67, 3);
  _lastNeural = ts;
  const W = nMain.width, H = nMain.height;
  if (W < 10) { requestAnimationFrame(neuralFrame); return; }

  // Clear with subtle fade
  ctxMain.save();
  ctxMain.setTransform(_zoom, 0, 0, _zoom, _panX, _panY);
  ctxMain.fillStyle = 'rgba(3,4,8,0.15)';
  ctxMain.fillRect(-_panX/_zoom, -_panY/_zoom, W/_zoom, H/_zoom);

  // FX layer clear
  ctxFx.clearRect(0, 0, W, H);
  ctxFx.save();
  ctxFx.setTransform(_zoom, 0, 0, _zoom, _panX, _panY);

  // Draw region boundary rings
  for (const [id, rc] of Object.entries(regionCenters)) {
    const isHovered = id === _hoveredRegion;
    const isSelected = id === _selectedRegion;
    const reg = rc.region;
    const pulsePhase = (ts/2000 + REGIONS.indexOf(reg)*0.5) % (Math.PI*2);
    const pulseAlpha = 0.03 + Math.sin(pulsePhase)*0.015;
    ctxMain.beginPath();
    ctxMain.arc(rc.x, rc.y, rc.r*1.6, 0, Math.PI*2);
    ctxMain.strokeStyle = `rgba(${reg.color[0]},${reg.color[1]},${reg.color[2]},${isHovered?0.2:isSelected?0.15:pulseAlpha})`;
    ctxMain.lineWidth = isHovered ? 1.5 : 0.5;
    ctxMain.setLineDash(isSelected ? [4,4] : []);
    ctxMain.stroke();
    ctxMain.setLineDash([]);

    // Region label
    ctxMain.fillStyle = `rgba(${reg.color[0]},${reg.color[1]},${reg.color[2]},${isHovered?0.7:0.3})`;
    ctxMain.font = `${isHovered?'bold ':''}${isHovered?10:8}px 'JetBrains Mono',monospace`;
    ctxMain.textAlign = 'center';
    ctxMain.fillText(reg.label, rc.x, rc.y - rc.r*1.7);
  }

  // Update and draw neurons
  let firing = 0;
  const time = ts * 0.001;
  neurons.forEach(n => {
    // Organic micro-motion with breathing
    n.x += n.vx*dt + Math.sin(time*0.5 + n.phase)*0.05;
    n.y += n.vy*dt + Math.cos(time*0.4 + n.phase)*0.05;

    // Soft boundary (attract back to origin)
    const dx = n.x - n.ox, dy = n.y - n.oy;
    const dist = Math.sqrt(dx*dx+dy*dy);
    const maxDist = 40;
    if (dist > maxDist) { n.vx -= dx*0.001; n.vy -= dy*0.001; }

    if (n.fire > 0) { n.fire -= 0.015*dt; firing++; }
    if (Math.random() < 0.0008) n.fire = 1;

    // Highlight neurons in hovered region
    const inHovered = n.region === _hoveredRegion;
    const baseAlpha = inHovered ? 0.5 : 0.25;
    const alpha = baseAlpha + n.fire*0.75;
    const size = n.r + n.fire*(inHovered?3:2);

    ctxMain.beginPath();
    ctxMain.arc(n.x, n.y, size, 0, Math.PI*2);
    ctxMain.fillStyle = `rgba(${n.color[0]},${n.color[1]},${n.color[2]},${alpha})`;
    ctxMain.fill();

    // Glow on firing neurons
    if (n.fire > 0.5) {
      ctxFx.beginPath();
      ctxFx.arc(n.x, n.y, size*3, 0, Math.PI*2);
      const g = ctxFx.createRadialGradient(n.x,n.y,0,n.x,n.y,size*3);
      g.addColorStop(0, `rgba(${n.color[0]},${n.color[1]},${n.color[2]},${n.fire*0.3})`);
      g.addColorStop(1, 'transparent');
      ctxFx.fillStyle = g;
      ctxFx.fill();
    }
  });

  // Draw synapses with gradient
  ctxMain.globalCompositeOperation = 'lighter';
  synapses.forEach(s => {
    const alpha = 0.025 + (s.a.fire+s.b.fire)*0.05 + s.pulse*0.1;
    if (alpha < 0.015) return;
    s.pulse *= 0.95;
    const isHighlighted = s.fromRegion===_hoveredRegion || s.toRegion===_hoveredRegion;
    ctxMain.beginPath();
    // Bezier curve for organic feel
    const midX = (s.a.x+s.b.x)/2 + (s.a.y-s.b.y)*0.1;
    const midY = (s.a.y+s.b.y)/2 + (s.b.x-s.a.x)*0.1;
    ctxMain.moveTo(s.a.x, s.a.y);
    ctxMain.quadraticCurveTo(midX, midY, s.b.x, s.b.y);
    ctxMain.strokeStyle = `rgba(${s.a.color[0]},${s.a.color[1]},${s.a.color[2]},${isHighlighted?alpha*3:alpha})`;
    ctxMain.lineWidth = isHighlighted ? 0.8 : 0.35;
    ctxMain.stroke();
  });
  ctxMain.globalCompositeOperation = 'source-over';

  // Signal propagation with trails
  if (Math.random() < 0.04 && synapses.length) {
    const s = synapses[Math.floor(Math.random()*synapses.length)];
    signals.push({ax:s.a.x, ay:s.a.y, bx:s.b.x, by:s.b.y, t:0, color:s.a.color, size:2.5, trail:false});
    s.a.fire = 1;
    s.pulse = 1;
    _signalCount++;
  }

  signals = signals.filter(sig => {
    sig.t += 0.018*dt;
    if (sig.t >= 1) { return false; }
    const t = sig.t;
    // Ease-out interpolation
    const ease = 1 - Math.pow(1-t, 3);
    const x = sig.ax + (sig.bx-sig.ax)*ease;
    const y = sig.ay + (sig.by-sig.ay)*ease;

    // Trail
    if (sig.trail) {
      ctxFx.beginPath();
      ctxFx.moveTo(sig.ax + (sig.bx-sig.ax)*Math.max(0,ease-0.15), sig.ay + (sig.by-sig.ay)*Math.max(0,ease-0.15));
      ctxFx.lineTo(x, y);
      ctxFx.strokeStyle = `rgba(${sig.color[0]},${sig.color[1]},${sig.color[2]},${(1-t)*0.4})`;
      ctxFx.lineWidth = sig.size*0.6;
      ctxFx.stroke();
    }

    // Signal head
    const headSize = (sig.size||2.5) * (1 - t*0.5);
    ctxFx.beginPath();
    ctxFx.arc(x, y, headSize, 0, Math.PI*2);
    ctxFx.fillStyle = `rgba(${sig.color[0]},${sig.color[1]},${sig.color[2]},${(1-t)*0.9})`;
    ctxFx.shadowColor = `rgb(${sig.color[0]},${sig.color[1]},${sig.color[2]})`;
    ctxFx.shadowBlur = 10;
    ctxFx.fill();
    ctxFx.shadowBlur = 0;
    return true;
  });

  ctxMain.restore();
  ctxFx.restore();

  // Signal rate calculation
  if (ts - _lastSignalCheck > 1000) {
    _signalRate = _signalCount;
    _signalCount = 0;
    _lastSignalCheck = ts;
  }

  // Coherence = % of regions with at least one firing neuron
  const activeRegions = new Set(neurons.filter(n=>n.fire>0).map(n=>n.region));
  const coherence = Math.round(activeRegions.size / REGIONS.length * 100);

  // HUD update
  document.getElementById('nN').textContent = neurons.length;
  document.getElementById('nF').textContent = firing;
  document.getElementById('nS').textContent = synapses.length;
  document.getElementById('nA').textContent = Math.round(firing/Math.max(neurons.length,1)*100)+'%';
  document.getElementById('nSig').textContent = _signalRate;
  document.getElementById('nCoh').textContent = coherence+'%';

  requestAnimationFrame(neuralFrame);
}

// ── SSE + Polling ────────────────────────────
let _sseRetries = 0;
function connectSSE() {
  try {
    const es = new EventSource('/api/stream');
    es.onopen = () => { _sseRetries = 0; document.getElementById('sseText').textContent = 'LIVE'; document.getElementById('ssePill').classList.remove('off'); };
    es.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data);
        if (d.type === 'poll') {
          updateLobes(d.lobes||[]);
          updateQuantum(d.quantum||{});
          updateBrain(d.brain||{}, d.voice||{}, d.codebase||{});
          updateMemoryEvents(d.memory_events||[]);
          document.getElementById('sOnline').textContent = `${d.online}/${d.total}`;
          document.getElementById('sUptime').textContent = fmtUp(d.uptime||0);
          document.getElementById('sMsgs').textContent = d.nexus_msg_count||0;
          document.getElementById('sBusLobes').textContent = d.nexus_lobes||0;
          document.getElementById('sTopics').textContent = d.nexus_topics||0;
          document.getElementById('lobeBadge').textContent = `${d.online}/${d.total}`;
        }
        if (d.type === 'nexus' && d.data) {
          const el = document.getElementById('feedScroll');
          el.insertAdjacentHTML('afterbegin', `<div class="feed-item"><span class="feed-from">${esc(d.data.from||'')}</span> <span class="feed-topic">${esc(d.data.topic||'')}</span></div>`);
        }
      } catch(e){}
    };
    es.onerror = () => {
      es.close();
      _sseRetries++;
      document.getElementById('ssePill').classList.add('off');
      if (_sseRetries <= 3) {
        document.getElementById('sseText').textContent = 'RETRY';
        setTimeout(connectSSE, 3000*_sseRetries);
      } else {
        document.getElementById('sseText').textContent = 'POLL';
      }
    };
  } catch(e) {
    document.getElementById('sseText').textContent = 'POLL';
  }
}

// ── Boot ─────────────────────────────────────
setInterval(() => { document.getElementById('clock').textContent = new Date().toLocaleTimeString(); }, 1000);
fetchState();
connectSSE();
setInterval(fetchState, 5000);
drawPentagon();
window.addEventListener('resize', () => { if (_neuralRunning) neuralResize(); });
</script>
</body>
</html>"""


# ── Entrypoint ───────────────────────────────────────────────────────────────

async def weaver_dashboard_serve():
    """Entry point for launching from weaver.py."""
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import uvicorn
    print(f"Weaver Live Dashboard starting on http://127.0.0.1:{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
