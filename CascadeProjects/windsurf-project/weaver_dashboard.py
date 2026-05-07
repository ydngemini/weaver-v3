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
from fastapi.responses import HTMLResponse, StreamingResponse

PORT = int(os.environ.get("WEAVER_DASHBOARD_PORT", "9990"))
PROJ = os.path.dirname(os.path.abspath(__file__))
VAULT = os.path.join(PROJ, "Nexus_Vault")

app = FastAPI(title="Weaver Live Dashboard", version="1.0.0")

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
    ("Nexus Bus",       "http://localhost:9998/health",  9998, "WebSocket pub/sub broker"),
    ("Quantum Soul",    None,                            None, "IBM Quantum 7-qubit loop"),
    ("Quantum API",     "http://localhost:9997/health",  9997, "Quantum state HTTP server"),
    ("Akashic Hub",     "http://localhost:9995/health",  9995, "Shared vector state"),
    ("Pineal Gate",     None,                            None, "Pentagon MoE router"),
    ("LoRA Server",     "http://localhost:8899/health",  8899, "1B Llama Soul Voice"),
    ("Phone Bridge",    "http://localhost:8765/health",  8765, "Twilio telephony"),
    ("Health Dashboard","http://localhost:9996/health",  9996, "Legacy traffic-light"),
    ("ProactivePulse",  None,                            None, "Quantum resonance monitor"),
    ("Dream State",     None,                            None, "Autonomous reflection"),
    ("n8n Workflow",    "http://localhost:5678/healthz",  5678, "Workflow orchestrator"),
]


# ── Health polling ───────────────────────────────────────────────────────────

async def _check_http(name: str, url: str, desc: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(url)
            if r.status_code == 200:
                extra = {}
                try:
                    extra = r.json()
                except Exception:
                    pass
                return {"name": name, "status": "online", "desc": desc, "detail": extra}
            return {"name": name, "status": "degraded", "desc": desc, "detail": f"HTTP {r.status_code}"}
    except httpx.ConnectError:
        return {"name": name, "status": "offline", "desc": desc, "detail": "Connection refused"}
    except Exception as e:
        return {"name": name, "status": "offline", "desc": desc, "detail": str(e)[:80]}


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


# ── Nexus Bus WebSocket listener ─────────────────────────────────────────────

async def _nexus_listener():
    """Background task: connect to Nexus Bus and capture messages."""
    import websockets
    while True:
        try:
            async with websockets.connect("ws://localhost:9999", ping_interval=20) as ws:
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
        uptime = time.time() - _boot_time
        online = sum(1 for l in lobes if l["status"] == "online")
        payload = {
            "type": "poll",
            "lobes": lobes,
            "quantum": qs,
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
            cf, "tunnel", "--url", f"http://localhost:{PORT}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            text = line.decode(errors="replace")
            match = re.search(r"(https://[a-zA-Z0-9_-]+\.trycloudflare\.com)", text)
            if match:
                _ngrok_url = match.group(1)
                print(f"\n{'='*60}", flush=True)
                print(f"  WEAVER LIVE DASHBOARD — PUBLIC URL", flush=True)
                print(f"  {_ngrok_url}", flush=True)
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
    return {
        "lobes": lobes,
        "quantum": qs,
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


# ── Main HTML dashboard ─────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WEAVER v3 — Live Dashboard</title>
<style>
:root {
  --bg: #06060c;
  --surface: #0c0c16;
  --surface2: #10101e;
  --border: #1a1a30;
  --text: #c8c8e0;
  --dim: #555580;
  --accent: #7c5cfc;
  --accent2: #a78bfa;
  --green: #22c55e;
  --red: #ef4444;
  --orange: #f59e0b;
  --cyan: #06b6d4;
  --pink: #ec4899;
  --blue: #3b82f6;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'SF Mono','Fira Code','JetBrains Mono','Cascadia Code',monospace;
  font-size: 12px;
  line-height: 1.5;
  min-height: 100vh;
}

/* ── Header ─────────────────────────────── */
.header {
  background: linear-gradient(135deg, #0d0d1a 0%, #12101f 100%);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
}
.header h1 {
  font-size: 18px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--accent), var(--pink));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: 3px;
}
.header-status {
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 11px;
}
.status-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 20px;
  background: rgba(34,197,94,0.1);
  border: 1px solid rgba(34,197,94,0.3);
}
.status-pill.disconnected {
  background: rgba(239,68,68,0.1);
  border-color: rgba(239,68,68,0.3);
}
.status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--green);
  animation: pulse 2s infinite;
}
.status-pill.disconnected .status-dot {
  background: var(--red);
  animation: none;
}
@keyframes pulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
  50% { opacity: 0.8; box-shadow: 0 0 0 6px rgba(34,197,94,0); }
}
.public-url {
  padding: 4px 10px;
  border-radius: 6px;
  background: rgba(124,92,252,0.1);
  border: 1px solid rgba(124,92,252,0.3);
  color: var(--accent2);
  font-size: 11px;
  cursor: pointer;
  text-decoration: none;
}
.public-url:hover { background: rgba(124,92,252,0.2); }

/* ── Tabs ───────────────────────────────── */
.tab-bar {
  display: flex;
  gap: 0;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 48px;
  z-index: 99;
}
.tab-btn {
  padding: 10px 24px;
  font-family: inherit;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--dim);
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  cursor: pointer;
  transition: all 0.2s;
}
.tab-btn:hover { color: var(--text); background: rgba(124,92,252,0.05); }
.tab-btn.active { color: var(--accent2); border-bottom-color: var(--accent); background: rgba(124,92,252,0.08); }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* ── Layout ─────────────────────────────── */
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  grid-template-rows: auto auto auto auto;
  gap: 1px;
  background: var(--border);
  max-width: 1800px;
  margin: 0 auto;
}
.panel {
  background: var(--surface);
  padding: 16px;
  min-height: 200px;
}
.panel-title {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--dim);
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.panel-badge {
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 10px;
  background: rgba(124,92,252,0.15);
  color: var(--accent2);
}
.stats-row {
  grid-column: 1 / -1;
  display: flex;
  gap: 1px;
  background: var(--border);
}
.stat-cell {
  flex: 1;
  background: var(--surface);
  padding: 14px 18px;
  text-align: center;
}
.stat-label { font-size: 9px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--dim); margin-bottom: 4px; }
.stat-value { font-size: 26px; font-weight: 700; }
.stat-value.green { color: var(--green); }
.stat-value.cyan { color: var(--cyan); }
.stat-value.orange { color: var(--orange); }
.stat-value.pink { color: var(--pink); }
.stat-value.accent { color: var(--accent); }
.stat-value.blue { color: var(--blue); }
.lobe-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.lobe-card {
  padding: 10px 12px; border-radius: 8px; background: var(--surface2);
  border: 1px solid var(--border); display: flex; align-items: center; gap: 10px; transition: all 0.3s;
}
.lobe-card.online { border-color: rgba(34,197,94,0.3); background: rgba(34,197,94,0.04); }
.lobe-card.offline { border-color: rgba(239,68,68,0.2); background: rgba(239,68,68,0.03); }
.lobe-card.stale, .lobe-card.degraded { border-color: rgba(245,158,11,0.3); background: rgba(245,158,11,0.04); }
.lobe-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; transition: all 0.3s; }
.lobe-dot.online { background: var(--green); box-shadow: 0 0 8px var(--green); }
.lobe-dot.offline { background: var(--red); }
.lobe-dot.stale, .lobe-dot.degraded { background: var(--orange); }
.lobe-name { font-size: 11px; font-weight: 600; }
.lobe-desc { font-size: 9px; color: var(--dim); margin-top: 1px; }
.lobe-detail { font-size: 9px; color: var(--dim); margin-left: auto; text-align: right; max-width: 120px; word-break: break-all; }
.pentagon-wrap { display: flex; flex-direction: column; align-items: center; gap: 10px; }
.pentagon-svg { width: 220px; height: 220px; }
.pent-edge { stroke: var(--border); stroke-width: 1; transition: all 0.5s; }
.pent-diag { stroke: var(--border); stroke-width: 0.5; stroke-dasharray: 3,3; opacity: 0.4; }
.pent-label { fill: var(--dim); font-size: 9px; text-anchor: middle; font-family: inherit; }
.pent-node { transition: all 0.5s; }
.pent-weight { fill: var(--text); font-size: 8px; text-anchor: middle; font-family: inherit; }
.quantum-info { text-align: center; padding: 12px; background: var(--surface2); border-radius: 8px; border: 1px solid var(--border); width: 100%; }
.q-bits { font-size: 24px; font-weight: 700; color: var(--pink); letter-spacing: 3px; margin-bottom: 4px; }
.q-pathway { font-size: 11px; color: var(--cyan); }
.q-secondary { font-size: 10px; color: var(--dim); }
.q-timestamp { font-size: 9px; color: var(--dim); margin-top: 4px; }
.q-raw { font-size: 10px; color: var(--dim); margin-top: 8px; padding: 8px; background: var(--bg); border-radius: 4px; line-height: 1.4; max-height: 80px; overflow-y: auto; text-align: left; }
.feed-scroll { max-height: 350px; overflow-y: auto; }
.feed-msg { padding: 6px 10px; margin-bottom: 4px; border-radius: 6px; border-left: 3px solid var(--accent); background: rgba(124,92,252,0.03); font-size: 11px; animation: fade-in 0.3s ease; }
.feed-msg.quantum { border-left-color: var(--pink); background: rgba(236,72,153,0.04); }
.feed-msg .ts { color: var(--dim); font-size: 9px; }
.feed-msg .from { color: var(--cyan); font-weight: 600; }
.feed-msg .topic { color: var(--orange); }
.feed-msg .payload { color: var(--dim); font-size: 10px; margin-top: 2px; word-break: break-all; max-height: 40px; overflow: hidden; }
@keyframes fade-in { from { opacity: 0; transform: translateY(-3px); } to { opacity: 1; } }
.transcript-block { max-height: 300px; overflow-y: auto; font-size: 11px; line-height: 1.6; background: var(--surface2); border-radius: 8px; padding: 12px; border: 1px solid var(--border); }
.transcript-line { margin-bottom: 4px; }
.transcript-line .t-ts { color: var(--dim); font-size: 9px; }
.transcript-line .t-speaker { color: var(--cyan); font-weight: 600; }
.person-card { padding: 8px 12px; margin-bottom: 6px; border-radius: 6px; background: var(--surface2); border: 1px solid var(--border); }
.person-name { color: var(--accent2); font-weight: 600; font-size: 12px; }
.person-desc { color: var(--dim); font-size: 10px; margin-top: 2px; }
.dream-card { padding: 10px 14px; margin-bottom: 8px; border-radius: 8px; background: linear-gradient(135deg, rgba(124,92,252,0.05), rgba(236,72,153,0.05)); border: 1px solid var(--border); }
.dream-ts { color: var(--pink); font-size: 10px; font-weight: 600; margin-bottom: 4px; }
.dream-content { color: var(--text); font-size: 11px; line-height: 1.6; }
.full-width { grid-column: 1 / -1; }
.two-col { grid-column: span 2; }

/* ── Neural Map ─────────────────────────── */
#neuralCanvas {
  width: 100%;
  height: calc(100vh - 110px);
  display: block;
  background: var(--bg);
}
.neural-wrap {
  position: relative;
  background: var(--bg);
}
.neural-legend {
  position: absolute;
  top: 16px;
  left: 16px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  z-index: 10;
  pointer-events: none;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 10px;
  color: var(--dim);
}
.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}
.neural-stats {
  position: absolute;
  top: 16px;
  right: 16px;
  text-align: right;
  z-index: 10;
  pointer-events: none;
}
.neural-stats div {
  font-size: 10px;
  color: var(--dim);
  margin-bottom: 3px;
}
.neural-stats .ns-val {
  color: var(--accent2);
  font-weight: 600;
}
.neural-activity-bar {
  position: absolute;
  bottom: 16px;
  left: 16px;
  right: 16px;
  display: flex;
  gap: 3px;
  height: 40px;
  align-items: flex-end;
  z-index: 10;
  pointer-events: none;
}
.na-bar {
  flex: 1;
  background: var(--accent);
  border-radius: 2px 2px 0 0;
  opacity: 0.6;
  transition: height 0.3s;
  min-height: 2px;
}

/* ── Scrollbar ──────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── Responsive ─────────────────────────── */
@media (max-width: 1200px) {
  .grid { grid-template-columns: 1fr 1fr; }
  .full-width, .two-col { grid-column: 1 / -1; }
}
@media (max-width: 768px) {
  .grid { grid-template-columns: 1fr; }
  .lobe-grid { grid-template-columns: 1fr; }
  .stats-row { flex-wrap: wrap; }
  .stat-cell { min-width: 120px; }
  .header { flex-direction: column; gap: 8px; }
  .tab-btn { padding: 8px 14px; font-size: 10px; }
}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <h1>WEAVER v3 LIVE DASHBOARD</h1>
  <div class="header-status">
    <a class="public-url" id="publicUrl" href="#" target="_blank" style="display:none">Public URL</a>
    <div class="status-pill" id="sseStatus">
      <div class="status-dot"></div>
      <span id="sseText">CONNECTING</span>
    </div>
    <span style="color:var(--dim)" id="clockText"></span>
  </div>
</div>

<!-- Tabs -->
<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('overview')">Overview</button>
  <button class="tab-btn" onclick="switchTab('neural')">Neural Map</button>
</div>

<!-- ═══════════════════ TAB: OVERVIEW ═══════════════════ -->
<div class="tab-content active" id="tab-overview">
<div class="grid">
  <div class="stats-row">
    <div class="stat-cell"><div class="stat-label">Lobes Online</div><div class="stat-value green" id="statOnline">--</div></div>
    <div class="stat-cell"><div class="stat-label">Uptime</div><div class="stat-value accent" id="statUptime">--</div></div>
    <div class="stat-cell"><div class="stat-label">Bus Messages</div><div class="stat-value cyan" id="statMsgs">0</div></div>
    <div class="stat-cell"><div class="stat-label">Active Bus Lobes</div><div class="stat-value orange" id="statBusLobes">0</div></div>
    <div class="stat-cell"><div class="stat-label">Topics Seen</div><div class="stat-value pink" id="statTopics">0</div></div>
    <div class="stat-cell"><div class="stat-label">Dominant Pathway</div><div class="stat-value blue" id="statPathway">--</div></div>
  </div>
  <div class="panel two-col">
    <div class="panel-title">System Lobes <span class="panel-badge" id="lobeBadge">0/0</span></div>
    <div class="lobe-grid" id="lobeGrid"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Quantum Pentagon Gate</div>
    <div class="pentagon-wrap">
      <svg class="pentagon-svg" viewBox="0 0 260 260" id="pentSvg"></svg>
      <div class="quantum-info">
        <div class="q-bits" id="qBits">|-------&#10217;</div>
        <div class="q-pathway" id="qPathway">Pathway: --</div>
        <div class="q-secondary" id="qSecondary"></div>
        <div class="q-timestamp" id="qTimestamp"></div>
        <div class="q-raw" id="qRaw"></div>
      </div>
    </div>
  </div>
  <div class="panel two-col">
    <div class="panel-title">Nexus Bus Live Feed <span class="panel-badge" id="feedBadge">0 msgs</span></div>
    <div class="feed-scroll" id="feedScroll"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Dream State Reflections</div>
    <div id="dreamsContainer"></div>
  </div>
  <div class="panel two-col">
    <div class="panel-title">Recent Conversation <span class="panel-badge">VTV Transcript</span></div>
    <div class="transcript-block" id="transcriptBlock">Loading...</div>
  </div>
  <div class="panel">
    <div class="panel-title">People Memory</div>
    <div id="peopleContainer"></div>
  </div>
</div>
</div>

<!-- ═══════════════════ TAB: NEURAL MAP ═══════════════════ -->
<div class="tab-content" id="tab-neural">
<div class="neural-wrap">
  <canvas id="neuralCanvas"></canvas>
  <div class="neural-legend" id="neuralLegend"></div>
  <div class="neural-stats">
    <div>Neurons: <span class="ns-val" id="nsTotal">0</span></div>
    <div>Firing: <span class="ns-val" id="nsFiring">0</span></div>
    <div>Region: <span class="ns-val" id="nsRegion">--</span></div>
    <div>Activity: <span class="ns-val" id="nsActivity">0%</span></div>
  </div>
  <div class="neural-activity-bar" id="activityBar"></div>
</div>
</div>

<script>
// ══════════════════════════════════════════════════════════════════
// Weaver Live Dashboard + Neural Map
// ══════════════════════════════════════════════════════════════════

// ── Tab switching ────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'neural') resizeCanvas();
}

// ═══════════════════════════════════════════════════════════════
// NEURAL MAP — 1000+ neurons in brain-region clusters
// ═══════════════════════════════════════════════════════════════

const REGIONS = [
  { id: "nexus_bus",     label: "Nexus Bus",         color: "#06b6d4", cx: 0.50, cy: 0.12, rx: 0.18, ry: 0.06, count: 120, desc: "Message Routing" },
  { id: "quantum_soul",  label: "Quantum Soul",      color: "#ec4899", cx: 0.18, cy: 0.28, rx: 0.12, ry: 0.10, count: 100, desc: "7-Qubit Collapse" },
  { id: "akashic_hub",   label: "Akashic Hub",       color: "#a78bfa", cx: 0.50, cy: 0.30, rx: 0.14, ry: 0.10, count: 130, desc: "Vector State Memory" },
  { id: "pineal_gate",   label: "Pineal Gate",       color: "#7c5cfc", cx: 0.82, cy: 0.28, rx: 0.12, ry: 0.10, count: 100, desc: "MoE Pentagon Router" },
  { id: "logic",         label: "Logic Lobe",        color: "#06b6d4", cx: 0.12, cy: 0.50, rx: 0.08, ry: 0.08, count:  80, desc: "Analytical Reasoning" },
  { id: "emotion",       label: "Emotion Lobe",      color: "#ec4899", cx: 0.30, cy: 0.50, rx: 0.08, ry: 0.08, count:  80, desc: "Affective Processing" },
  { id: "memory",        label: "Memory Lobe",       color: "#f59e0b", cx: 0.50, cy: 0.50, rx: 0.08, ry: 0.08, count:  80, desc: "Recall & Association" },
  { id: "creativity",    label: "Creativity Lobe",   color: "#7c5cfc", cx: 0.70, cy: 0.50, rx: 0.08, ry: 0.08, count:  80, desc: "Generative Synthesis" },
  { id: "vigilance",     label: "Vigilance Lobe",    color: "#ef4444", cx: 0.88, cy: 0.50, rx: 0.08, ry: 0.08, count:  80, desc: "Threat Detection" },
  { id: "lora_soul",     label: "LoRA Soul Voice",   color: "#22c55e", cx: 0.50, cy: 0.70, rx: 0.14, ry: 0.08, count: 100, desc: "Personality Filter" },
  { id: "phone_bridge",  label: "Phone Bridge",      color: "#3b82f6", cx: 0.20, cy: 0.72, rx: 0.10, ry: 0.07, count:  60, desc: "Twilio Telephony" },
  { id: "vtv_core",      label: "VTV Perception",    color: "#f59e0b", cx: 0.80, cy: 0.72, rx: 0.10, ry: 0.07, count:  60, desc: "Audio/Video/Face ID" },
  { id: "dream_state",   label: "Dream State",       color: "#8b5cf6", cx: 0.25, cy: 0.88, rx: 0.10, ry: 0.06, count:  50, desc: "Autonomous Reflection" },
  { id: "obsidian",      label: "Obsidian Bridge",   color: "#64748b", cx: 0.50, cy: 0.90, rx: 0.10, ry: 0.05, count:  40, desc: "Knowledge Graph Sync" },
  { id: "proactive",     label: "ProactivePulse",    color: "#ef4444", cx: 0.75, cy: 0.88, rx: 0.10, ry: 0.06, count:  50, desc: "Resonance Monitor" },
];

let neurons = [];
let synapses = [];
let regionActivity = {};
let activityHistory = new Array(60).fill(0);
let canvasReady = false;

function buildNeurons(W, H) {
  neurons = [];
  synapses = [];
  const pad = 40;
  const aW = W - pad*2, aH = H - pad*2;

  for (const reg of REGIONS) {
    const cxP = pad + reg.cx * aW;
    const cyP = pad + reg.cy * aH;
    const rxP = reg.rx * aW;
    const ryP = reg.ry * aH;
    regionActivity[reg.id] = 0;

    for (let i = 0; i < reg.count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const dist = Math.sqrt(Math.random());
      const x = cxP + Math.cos(angle) * dist * rxP * (0.6 + Math.random() * 0.4);
      const y = cyP + Math.sin(angle) * dist * ryP * (0.6 + Math.random() * 0.4);
      neurons.push({
        x, y,
        region: reg.id,
        color: reg.color,
        brightness: 0,
        targetBrightness: 0,
        phase: Math.random() * Math.PI * 2,
        size: 1.2 + Math.random() * 1.8,
      });
    }
  }

  // Build sparse synapses: connect nearby neurons within and across adjacent regions
  const regionIdx = {};
  neurons.forEach((n, i) => {
    if (!regionIdx[n.region]) regionIdx[n.region] = [];
    regionIdx[n.region].push(i);
  });

  // Intra-region synapses (10% of pairs within range)
  for (const reg of REGIONS) {
    const idxs = regionIdx[reg.id] || [];
    for (let a = 0; a < idxs.length; a++) {
      for (let b = a + 1; b < idxs.length; b++) {
        const na = neurons[idxs[a]], nb = neurons[idxs[b]];
        const dx = na.x - nb.x, dy = na.y - nb.y;
        const d = Math.sqrt(dx*dx + dy*dy);
        if (d < 60 && Math.random() < 0.08) {
          synapses.push({ a: idxs[a], b: idxs[b], strength: 0 });
        }
      }
    }
  }

  // Inter-region pathways
  const links = [
    ["nexus_bus","akashic_hub"], ["nexus_bus","pineal_gate"], ["nexus_bus","quantum_soul"],
    ["nexus_bus","vtv_core"], ["nexus_bus","phone_bridge"],
    ["quantum_soul","akashic_hub"], ["quantum_soul","pineal_gate"],
    ["akashic_hub","pineal_gate"], ["akashic_hub","lora_soul"],
    ["pineal_gate","logic"], ["pineal_gate","emotion"], ["pineal_gate","memory"],
    ["pineal_gate","creativity"], ["pineal_gate","vigilance"],
    ["logic","lora_soul"], ["emotion","lora_soul"], ["memory","lora_soul"],
    ["creativity","lora_soul"], ["vigilance","lora_soul"],
    ["lora_soul","phone_bridge"], ["lora_soul","vtv_core"],
    ["vtv_core","nexus_bus"], ["phone_bridge","nexus_bus"],
    ["dream_state","akashic_hub"], ["dream_state","memory"],
    ["obsidian","akashic_hub"], ["obsidian","memory"],
    ["proactive","quantum_soul"], ["proactive","phone_bridge"],
  ];
  for (const [ra, rb] of links) {
    const ia = regionIdx[ra] || [], ib = regionIdx[rb] || [];
    const count = Math.min(6, ia.length, ib.length);
    for (let k = 0; k < count; k++) {
      const a = ia[Math.floor(Math.random() * ia.length)];
      const b = ib[Math.floor(Math.random() * ib.length)];
      synapses.push({ a, b, strength: 0 });
    }
  }
}

function fireRegion(regionId, intensity) {
  if (!intensity) intensity = 0.5 + Math.random() * 0.5;
  regionActivity[regionId] = Math.min(1, (regionActivity[regionId] || 0) + intensity);
  for (const n of neurons) {
    if (n.region === regionId) {
      n.targetBrightness = Math.min(1, n.targetBrightness + intensity * (0.3 + Math.random() * 0.7));
    }
  }
  for (const s of synapses) {
    if (neurons[s.a].region === regionId || neurons[s.b].region === regionId) {
      s.strength = Math.min(1, s.strength + intensity * 0.6);
    }
  }
}

function drawNeural() {
  const canvas = document.getElementById('neuralCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Draw synapses
  for (const s of synapses) {
    if (s.strength < 0.02) continue;
    const na = neurons[s.a], nb = neurons[s.b];
    ctx.beginPath();
    ctx.moveTo(na.x, na.y);
    ctx.lineTo(nb.x, nb.y);
    const alpha = s.strength * 0.4;
    const mix = na.brightness > nb.brightness ? na : nb;
    ctx.strokeStyle = mix.color.replace(')', `,${alpha})`).replace('rgb','rgba').replace('#','');
    // Hex to rgba
    const hex = mix.color;
    const rr = parseInt(hex.slice(1,3),16), gg = parseInt(hex.slice(3,5),16), bb = parseInt(hex.slice(5,7),16);
    ctx.strokeStyle = `rgba(${rr},${gg},${bb},${alpha})`;
    ctx.lineWidth = 0.5 + s.strength * 1.5;
    ctx.stroke();
    s.strength *= 0.96;
  }

  // Draw neurons
  let firingCount = 0;
  for (const n of neurons) {
    // Animate brightness toward target
    n.brightness += (n.targetBrightness - n.brightness) * 0.15;
    n.targetBrightness *= 0.97;

    // Idle shimmer
    const shimmer = 0.03 + 0.02 * Math.sin(Date.now() * 0.002 + n.phase);
    const b = Math.max(shimmer, n.brightness);
    if (b > 0.15) firingCount++;

    const hex = n.color;
    const rr = parseInt(hex.slice(1,3),16), gg = parseInt(hex.slice(3,5),16), bb = parseInt(hex.slice(5,7),16);

    // Glow for active neurons
    if (b > 0.3) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.size * 3 + b * 8, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${rr},${gg},${bb},${b * 0.15})`;
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(n.x, n.y, n.size + b * 2, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${rr},${gg},${bb},${0.15 + b * 0.85})`;
    ctx.fill();
  }

  // Draw region labels
  const pad = 40;
  const aW = W - pad*2, aH = H - pad*2;
  ctx.font = '11px "SF Mono","Fira Code",monospace';
  ctx.textAlign = 'center';
  for (const reg of REGIONS) {
    const x = pad + reg.cx * aW;
    const y = pad + reg.cy * aH - reg.ry * aH - 14;
    const act = regionActivity[reg.id] || 0;
    const alpha = 0.3 + act * 0.7;
    ctx.fillStyle = `rgba(200,200,224,${alpha})`;
    ctx.fillText(reg.label, x, y);
    if (act > 0.1) {
      ctx.font = '9px "SF Mono","Fira Code",monospace';
      ctx.fillStyle = `rgba(200,200,224,${alpha * 0.6})`;
      ctx.fillText(reg.desc, x, y + 13);
      ctx.font = '11px "SF Mono","Fira Code",monospace';
    }
  }

  // Decay region activity
  for (const id of Object.keys(regionActivity)) {
    regionActivity[id] *= 0.95;
  }

  // Update stats
  document.getElementById('nsTotal').textContent = neurons.length;
  document.getElementById('nsFiring').textContent = firingCount;
  const totalAct = Object.values(regionActivity).reduce((a,b)=>a+b,0) / REGIONS.length;
  document.getElementById('nsActivity').textContent = Math.round(totalAct * 100) + '%';

  // Find hottest region
  let hotRegion = '--';
  let hotVal = 0;
  for (const [id, v] of Object.entries(regionActivity)) {
    if (v > hotVal) { hotVal = v; hotRegion = id; }
  }
  if (hotVal > 0.05) {
    const reg = REGIONS.find(r => r.id === hotRegion);
    document.getElementById('nsRegion').textContent = reg ? reg.label : hotRegion;
  } else {
    document.getElementById('nsRegion').textContent = 'idle';
  }

  // Activity history bar
  activityHistory.push(Math.round(totalAct * 100));
  if (activityHistory.length > 60) activityHistory.shift();
  const barContainer = document.getElementById('activityBar');
  if (barContainer && barContainer.children.length === 0) {
    for (let i = 0; i < 60; i++) {
      const bar = document.createElement('div');
      bar.className = 'na-bar';
      barContainer.appendChild(bar);
    }
  }
  if (barContainer) {
    const bars = barContainer.children;
    for (let i = 0; i < 60 && i < bars.length; i++) {
      bars[i].style.height = Math.max(2, activityHistory[i] * 0.4) + 'px';
    }
  }

  requestAnimationFrame(drawNeural);
}

function resizeCanvas() {
  const canvas = document.getElementById('neuralCanvas');
  if (!canvas) return;
  const wrap = canvas.parentElement;
  canvas.width = wrap.clientWidth;
  canvas.height = window.innerHeight - 110;
  if (!canvasReady || neurons.length === 0) {
    buildNeurons(canvas.width, canvas.height);
    canvasReady = true;
    // Build legend
    const leg = document.getElementById('neuralLegend');
    leg.innerHTML = REGIONS.map(r =>
      `<div class="legend-item"><div class="legend-dot" style="background:${r.color}"></div>${r.label}</div>`
    ).join('');
  }
}

window.addEventListener('resize', () => {
  if (document.getElementById('tab-neural').classList.contains('active')) resizeCanvas();
});

// Map Nexus Bus events to neural region firing
function neuralProcessEvent(entry) {
  const from = (entry.from || '').toLowerCase();
  const topic = (entry.topic || '').toLowerCase();
  const text = from + ' ' + topic;

  // Nexus bus always lights on any message
  fireRegion('nexus_bus', 0.3);

  if (text.includes('quantum') || text.includes('qsoul'))     fireRegion('quantum_soul', 0.7);
  if (text.includes('akashic') || text.includes('hub'))        fireRegion('akashic_hub', 0.5);
  if (text.includes('pineal') || text.includes('gate') || text.includes('routing')) fireRegion('pineal_gate', 0.7);
  if (text.includes('logic'))                                  fireRegion('logic', 0.8);
  if (text.includes('emotion'))                                fireRegion('emotion', 0.8);
  if (text.includes('memory'))                                 fireRegion('memory', 0.8);
  if (text.includes('creativ'))                                fireRegion('creativity', 0.8);
  if (text.includes('vigilan'))                                fireRegion('vigilance', 0.8);
  if (text.includes('lora') || text.includes('soul'))          fireRegion('lora_soul', 0.6);
  if (text.includes('phone') || text.includes('twilio') || text.includes('call')) fireRegion('phone_bridge', 0.6);
  if (text.includes('vtv') || text.includes('vision') || text.includes('face') || text.includes('audio')) fireRegion('vtv_core', 0.7);
  if (text.includes('dream'))                                  fireRegion('dream_state', 0.6);
  if (text.includes('obsidian'))                               fireRegion('obsidian', 0.5);
  if (text.includes('pulse') || text.includes('proactive') || text.includes('interference')) fireRegion('proactive', 0.7);

  // Speaking = light up perception + soul voice + phone/vtv
  if (text.includes('manifested') || text.includes('response') || text.includes('tts') || text.includes('speak')) {
    fireRegion('lora_soul', 0.9);
    fireRegion('vtv_core', 0.5);
    fireRegion('phone_bridge', 0.5);
  }

  // Thinking = light up experts + pineal + akashic
  if (text.includes('expert') || text.includes('collapse') || text.includes('fracture') || text.includes('moe')) {
    fireRegion('pineal_gate', 0.8);
    fireRegion('akashic_hub', 0.6);
    fireRegion('logic', 0.4);
    fireRegion('emotion', 0.4);
    fireRegion('memory', 0.4);
    fireRegion('creativity', 0.4);
    fireRegion('vigilance', 0.4);
  }
}

// Map poll-based lobe statuses to background neural glow
function neuralProcessLobes(lobes) {
  for (const l of lobes) {
    if (l.status !== 'online') continue;
    const name = l.name.toLowerCase();
    if (name.includes('nexus'))    fireRegion('nexus_bus', 0.08);
    if (name.includes('quantum') && !name.includes('api')) fireRegion('quantum_soul', 0.08);
    if (name.includes('akashic'))  fireRegion('akashic_hub', 0.08);
    if (name.includes('pineal'))   fireRegion('pineal_gate', 0.08);
    if (name.includes('lora'))     fireRegion('lora_soul', 0.08);
    if (name.includes('phone'))    fireRegion('phone_bridge', 0.08);
    if (name.includes('dream'))    fireRegion('dream_state', 0.08);
    if (name.includes('pulse'))    fireRegion('proactive', 0.08);
  }
}


// ═══════════════════════════════════════════════════════════════
// OVERVIEW TAB — existing functionality
// ═══════════════════════════════════════════════════════════════

const DIMS = ["logic","emotion","memory","creativity","vigilance"];
const DIM_COLORS = {logic:"#06b6d4",emotion:"#ec4899",memory:"#f59e0b",creativity:"#7c5cfc",vigilance:"#ef4444"};
const cx=130, cy=125, r=80;
const pentPts = DIMS.map((_,i) => {
  const a = -Math.PI/2 + (2*Math.PI*i/5);
  return {x: cx+r*Math.cos(a), y: cy+r*Math.sin(a)};
});
let weights = {logic:0.5,emotion:0.5,memory:0.5,creativity:0.5,vigilance:0.5};
let feedCount = 0;

function initPentagon() {
  const svg = document.getElementById("pentSvg");
  let h = '';
  for (let i=0;i<5;i++) for (let j=i+2;j<5;j++) {
    if (j-i===4) continue;
    h += `<line class="pent-diag" x1="${pentPts[i].x}" y1="${pentPts[i].y}" x2="${pentPts[j].x}" y2="${pentPts[j].y}"/>`;
  }
  for (let i=0;i<5;i++) {
    const j=(i+1)%5;
    h += `<line class="pent-edge" id="edge${i}" x1="${pentPts[i].x}" y1="${pentPts[i].y}" x2="${pentPts[j].x}" y2="${pentPts[j].y}"/>`;
  }
  h += `<circle cx="${cx}" cy="${cy}" r="4" fill="#555580" opacity="0.5"/>`;
  h += `<text x="${cx}" y="${cy+16}" fill="#555580" font-size="8" text-anchor="middle" font-family="inherit">Weaver</text>`;
  for (let i=0;i<5;i++) {
    const p=pentPts[i], d=DIMS[i];
    h += `<circle class="pent-node" id="pn-${d}" cx="${p.x}" cy="${p.y}" r="10" fill="${DIM_COLORS[d]}" opacity="0.3"/>`;
    const ly = p.y > cy ? p.y+22 : p.y-16;
    h += `<text class="pent-label" x="${p.x}" y="${ly}">${d}</text>`;
    const wy = p.y > cy ? p.y+32 : p.y-26;
    h += `<text class="pent-weight" id="pw-${d}" x="${p.x}" y="${wy}">0.50</text>`;
  }
  h += `<polygon id="pentInner" fill="rgba(124,92,252,0.1)" stroke="var(--accent)" stroke-width="1.5" points=""/>`;
  svg.innerHTML = h;
}

function updatePentagon(w) {
  weights = w || weights;
  for (const d of DIMS) {
    const node = document.getElementById(`pn-${d}`);
    const wt = document.getElementById(`pw-${d}`);
    if (!node) continue;
    const v = weights[d] || 0.5;
    node.setAttribute("r", 6 + v * 16);
    node.setAttribute("opacity", 0.15 + v * 0.85);
    if (wt) wt.textContent = v.toFixed(2);
  }
  const inner = document.getElementById("pentInner");
  if (inner) {
    const pts = DIMS.map((d,i) => {
      const v = (weights[d]||0.5);
      const dist = r * 0.15 + r * 0.85 * v;
      const a = -Math.PI/2 + (2*Math.PI*i/5);
      return `${cx+dist*Math.cos(a)},${cy+dist*Math.sin(a)}`;
    });
    inner.setAttribute("points", pts.join(" "));
  }
}

function updateLobes(lobes) {
  const grid = document.getElementById("lobeGrid");
  const online = lobes.filter(l=>l.status==="online").length;
  document.getElementById("lobeBadge").textContent = `${online}/${lobes.length}`;
  document.getElementById("statOnline").textContent = `${online}/${lobes.length}`;
  grid.innerHTML = lobes.map(l => {
    const detail = typeof l.detail === "object" ? "" : (l.detail || "");
    return `<div class="lobe-card ${l.status}"><div class="lobe-dot ${l.status}"></div><div><div class="lobe-name">${l.name}</div><div class="lobe-desc">${l.desc}</div></div><div class="lobe-detail">${detail}</div></div>`;
  }).join("");
  neuralProcessLobes(lobes);
}

function updateQuantum(q) {
  if (!q) return;
  const bits = q.bitstring ? `|${q.bitstring}&#10217;` : "|-------&#10217;";
  document.getElementById("qBits").innerHTML = bits;
  document.getElementById("qPathway").textContent = `Dominant: ${q.dominant || "--"}`;
  document.getElementById("qSecondary").textContent = q.secondary ? `Secondary: ${q.secondary}` : "";
  document.getElementById("qTimestamp").textContent = q.timestamp ? `Measured: ${q.timestamp}` : "";
  document.getElementById("qRaw").textContent = q.raw || "";
  document.getElementById("statPathway").textContent = q.dominant || "--";
  if (q.weights) updatePentagon(q.weights);
}

function addFeedMsg(entry) {
  const scroll = document.getElementById("feedScroll");
  const div = document.createElement("div");
  const isQ = (entry.topic||"").includes("quantum") || (entry.from||"").includes("quantum");
  div.className = "feed-msg" + (isQ ? " quantum" : "");
  const ts = entry.ts ? new Date(entry.ts).toLocaleTimeString() : "";
  let payload = entry.payload || "";
  if (payload.length > 200) payload = payload.slice(0,200) + "...";
  div.innerHTML = `<span class="ts">${ts}</span> <span class="from">${entry.from||"?"}</span> <span class="topic">#${entry.topic||""}</span><div class="payload">${payload}</div>`;
  scroll.appendChild(div);
  while (scroll.children.length > 100) scroll.removeChild(scroll.firstChild);
  scroll.scrollTop = scroll.scrollHeight;
  feedCount++;
  document.getElementById("feedBadge").textContent = `${feedCount} msgs`;
  neuralProcessEvent(entry);
}

function updateTranscript(text) {
  const block = document.getElementById("transcriptBlock");
  if (!text) { block.textContent = "No transcript yet"; return; }
  block.innerHTML = text.split("\n").map(line => {
    const m = line.match(/^\[([^\]]+)\]\s*(\w+):\s*(.*)/);
    if (m) return `<div class="transcript-line"><span class="t-ts">[${m[1]}]</span> <span class="t-speaker">${m[2]}:</span> ${m[3]}</div>`;
    return `<div class="transcript-line">${line}</div>`;
  }).join("");
  block.scrollTop = block.scrollHeight;
}

function updatePeople(people) {
  const c = document.getElementById("peopleContainer");
  if (!people || !people.length) { c.innerHTML = '<div style="color:var(--dim)">No people memory</div>'; return; }
  c.innerHTML = people.map(p => `<div class="person-card"><div class="person-name">${p.name}</div><div class="person-desc">${p.summary}</div></div>`).join("");
}

function updateDreams(dreams) {
  const c = document.getElementById("dreamsContainer");
  if (!dreams || !dreams.length) { c.innerHTML = '<div style="color:var(--dim)">No dreams yet</div>'; return; }
  c.innerHTML = dreams.map(d => `<div class="dream-card"><div class="dream-ts">Dream - ${d.timestamp}</div><div class="dream-content">${d.content}</div></div>`).join("");
}

function fmtUptime(s) {
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

setInterval(() => {
  document.getElementById("clockText").textContent = new Date().toLocaleTimeString();
}, 1000);

async function initialLoad() {
  try {
    const r = await fetch("/api/state");
    const d = await r.json();
    updateLobes(d.lobes || []);
    updateQuantum(d.quantum || {});
    updateTranscript(d.transcript || "");
    updatePeople(d.people || []);
    updateDreams(d.dreams || []);
    if (d.nexus_feed) d.nexus_feed.forEach(addFeedMsg);
    document.getElementById("statMsgs").textContent = d.nexus_stats?.msg_count || 0;
    document.getElementById("statBusLobes").textContent = d.nexus_stats?.lobes_seen?.length || 0;
    document.getElementById("statTopics").textContent = d.nexus_stats?.topics_seen?.length || 0;
    document.getElementById("statUptime").textContent = fmtUptime(d.uptime || 0);
    if (d.ngrok_url) {
      const el = document.getElementById("publicUrl");
      el.href = d.ngrok_url;
      el.textContent = d.ngrok_url.replace("https://","");
      el.style.display = "inline-block";
    }
  } catch(e) { console.error("Initial load:", e); }
}

function connectSSE() {
  const pill = document.getElementById("sseStatus");
  const text = document.getElementById("sseText");
  const es = new EventSource("/api/stream");
  es.onopen = () => { pill.classList.remove("disconnected"); text.textContent = "LIVE"; };
  es.onmessage = (ev) => {
    try {
      const d = JSON.parse(ev.data);
      if (d.type === "poll") {
        updateLobes(d.lobes || []);
        updateQuantum(d.quantum || {});
        document.getElementById("statUptime").textContent = fmtUptime(d.uptime || 0);
        document.getElementById("statMsgs").textContent = d.nexus_msg_count || 0;
        document.getElementById("statBusLobes").textContent = d.nexus_lobes || 0;
        document.getElementById("statTopics").textContent = d.nexus_topics || 0;
        if (d.ngrok_url) {
          const el = document.getElementById("publicUrl");
          el.href = d.ngrok_url;
          el.textContent = d.ngrok_url.replace("https://","");
          el.style.display = "inline-block";
        }
      }
      if (d.type === "nexus") addFeedMsg(d.data);
    } catch(e) {}
  };
  es.onerror = () => {
    pill.classList.add("disconnected");
    text.textContent = "RECONNECTING";
    es.close();
    setTimeout(connectSSE, 3000);
  };
}

async function refreshVault() {
  try {
    const r = await fetch("/api/state");
    const d = await r.json();
    updateTranscript(d.transcript || "");
    updatePeople(d.people || []);
    updateDreams(d.dreams || []);
  } catch(e) {}
}
setInterval(refreshVault, 30000);

// ── Boot ─────────────────────────────────
initPentagon();
updatePentagon(weights);
initialLoad();
connectSSE();
requestAnimationFrame(drawNeural);
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
    print(f"Weaver Live Dashboard starting on http://localhost:{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
