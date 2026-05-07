#!/usr/bin/env python3
"""
health_dashboard.py — Weaver v3 Health Monitoring Dashboard
============================================================
Traffic-light status for all lobes. Auto-refreshes every 5 seconds.

Port: 9996 (default)
"""

import asyncio
import os
import time
from datetime import datetime

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

PORT = int(os.environ.get("HEALTH_DASHBOARD_PORT", "9996"))

app = FastAPI(title="Weaver Health Dashboard", version="2.0.0")

_BOOT_TIME = time.time()
_check_count = 0
_lobe_history: dict[str, list[str]] = {}

LOBES = [
    ("Nexus Bus",        "http://localhost:9998/health",  "WebSocket pub/sub broker"),
    ("Quantum Soul",     None,                            "IBM Quantum 7-qubit loop (checks state file)"),
    ("Quantum API",      "http://localhost:9997/health",  "Quantum state HTTP server"),
    ("Akashic Hub API",  "http://localhost:9995/health",  "Live vector-state quantum bias"),
    ("Pineal Gate",      None,                            "MoE router (embedded in weaver.py)"),
    ("LoRA Server",      "http://localhost:8899/health",  "1B Llama LoRA personality filter"),
    ("Phone Bridge",     "http://localhost:8765/health",  "Twilio telephony + SMS/MMS"),
    ("ProactivePulse",   None,                            "Quantum resonance monitor (embedded in weaver.py)"),
    ("Dream State",      None,                            "Autonomous reflection (checks dream log)"),
    ("n8n Workflow",     "http://localhost:5678/healthz",  "Workflow orchestrator"),
]

VAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Nexus_Vault")
QUANTUM_STATE_FILE = os.path.join(VAULT_DIR, "quantum_state.txt")


async def check_http_lobe(name: str, url: str, desc: str) -> dict:
    """Check an HTTP-based lobe."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return {"name": name, "status": "online", "icon": "🟢", "desc": desc, "detail": ""}
            else:
                return {"name": name, "status": "degraded", "icon": "🟡", "desc": desc, "detail": f"HTTP {r.status_code}"}
    except httpx.ConnectError:
        return {"name": name, "status": "offline", "icon": "🔴", "desc": desc, "detail": "Connection refused"}
    except Exception as e:
        return {"name": name, "status": "offline", "icon": "🔴", "desc": desc, "detail": str(e)[:60]}


async def check_quantum_soul() -> dict:
    """Check Quantum Soul by state file freshness."""
    name = "Quantum Soul"
    desc = "IBM Quantum 7-qubit loop (checks state file)"
    try:
        if os.path.exists(QUANTUM_STATE_FILE):
            mtime = os.path.getmtime(QUANTUM_STATE_FILE)
            age = time.time() - mtime
            if age < 600:  # Updated within 10 minutes (loop is 5 min)
                return {"name": name, "status": "online", "icon": "🟢", "desc": desc, "detail": f"Last measurement {int(age)}s ago"}
            else:
                return {"name": name, "status": "stale", "icon": "🟡", "desc": desc, "detail": f"Last measurement {int(age/60)}m ago"}
        return {"name": name, "status": "offline", "icon": "🔴", "desc": desc, "detail": "No state file"}
    except Exception as e:
        return {"name": name, "status": "error", "icon": "🔴", "desc": desc, "detail": str(e)[:60]}


async def check_process(name: str, desc: str, pattern: str) -> dict:
    """Check if a process matching the pattern is running."""
    try:
        result = await asyncio.create_subprocess_shell(
            f"pgrep -f '{pattern}'",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await result.communicate()
        if result.returncode == 0 and stdout.strip():
            return {"name": name, "status": "online", "icon": "🟢", "desc": desc, "detail": f"{pattern} running"}
        return {"name": name, "status": "offline", "icon": "🔴", "desc": desc, "detail": f"{pattern} not found"}
    except Exception as e:
        return {"name": name, "status": "error", "icon": "🔴", "desc": desc, "detail": str(e)[:60]}


DREAM_LOG = os.path.join(VAULT_DIR, "weaver_dreams.md")


async def check_dream_state() -> dict:
    """Check Dream State by dream log freshness."""
    name = "Dream State"
    desc = "Autonomous reflection (checks dream log)"
    try:
        if os.path.exists(DREAM_LOG):
            mtime = os.path.getmtime(DREAM_LOG)
            age_h = (time.time() - mtime) / 3600
            if age_h < 6:
                return {"name": name, "status": "online", "icon": "🟢", "desc": desc, "detail": f"Last dream {age_h:.1f}h ago"}
            return {"name": name, "status": "stale", "icon": "🟡", "desc": desc, "detail": f"Last dream {age_h:.1f}h ago"}
        return {"name": name, "status": "offline", "icon": "🔴", "desc": desc, "detail": "No dream log yet"}
    except Exception as e:
        return {"name": name, "status": "error", "icon": "🔴", "desc": desc, "detail": str(e)[:60]}


async def gather_all_status() -> list:
    """Check all lobes concurrently."""
    global _check_count
    _check_count += 1
    tasks = []
    for name, url, desc in LOBES:
        if name == "Quantum Soul":
            tasks.append(check_quantum_soul())
        elif name == "Dream State":
            tasks.append(check_dream_state())
        elif name in ("Pineal Gate", "ProactivePulse"):
            tasks.append(check_process(name, desc, "weaver.py"))
        elif url:
            tasks.append(check_http_lobe(name, url, desc))
    results = await asyncio.gather(*tasks)
    for r in results:
        hist = _lobe_history.setdefault(r["name"], [])
        hist.append(r["status"])
        if len(hist) > 60:
            hist.pop(0)
    return results


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Render HTML dashboard with auto-refresh."""
    results = await gather_all_status()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    online = sum(1 for r in results if r["status"] == "online")
    total = len(results)

    # Read quantum state snippet
    quantum_snippet = ""
    try:
        if os.path.exists(QUANTUM_STATE_FILE):
            with open(QUANTUM_STATE_FILE, "r") as f:
                quantum_snippet = f.read().strip()[:200]
    except Exception:
        pass

    rows = ""
    for r in results:
        bg = "#0a2a0a" if r["status"] == "online" else "#2a0a0a" if r["status"] == "offline" else "#2a2a0a"
        rows += f"""
        <tr style="background: {bg};">
            <td style="padding: 8px; font-size: 24px;">{r['icon']}</td>
            <td style="padding: 8px;"><strong>{r['name']}</strong><br><small style="color: #888;">{r['desc']}</small></td>
            <td style="padding: 8px; color: {'#0f0' if r['status'] == 'online' else '#f00' if r['status'] == 'offline' else '#ff0'};">{r['status'].upper()}</td>
            <td style="padding: 8px; color: #888;">{r.get('detail', '')}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Weaver v3 Health</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{ font-family: 'Courier New', monospace; background: #0a0a0a; color: #0f0; margin: 20px; }}
        h1 {{ color: #0ff; }}
        table {{ border-collapse: collapse; width: 100%; }}
        tr {{ border-bottom: 1px solid #222; }}
        .summary {{ font-size: 18px; margin: 10px 0 20px 0; }}
        .quantum {{ background: #111; padding: 10px; border: 1px solid #333; margin-top: 20px; font-size: 12px; color: #aaa; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <h1>🌀 Weaver v3 — System Health</h1>
    <div class="summary">
        {online}/{total} lobes online &nbsp; | &nbsp; Last check: {now}
    </div>
    <table>
        <tr style="border-bottom: 2px solid #444;">
            <th style="padding: 8px;"></th>
            <th style="padding: 8px; text-align: left;">Lobe</th>
            <th style="padding: 8px; text-align: left;">Status</th>
            <th style="padding: 8px; text-align: left;">Detail</th>
        </tr>
        {rows}
    </table>
    <div class="quantum">
⚛️  Latest Quantum State:
{quantum_snippet if quantum_snippet else '(no measurement yet)'}
    </div>
</body>
</html>"""
    return html


@app.get("/health")
async def health():
    return {"status": "ok", "service": "weaver-health-dashboard", "port": PORT}


@app.get("/api/status")
async def api_status():
    """JSON version of the dashboard for programmatic access."""
    results = await gather_all_status()
    return {
        "lobes": results,
        "online": sum(1 for r in results if r["status"] == "online"),
        "total": len(results),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/metrics")
async def api_metrics():
    """Prometheus-style metrics: uptime, check counts, per-lobe availability."""
    now = time.time()
    uptime = now - _BOOT_TIME
    lobe_avail = {}
    for name, hist in _lobe_history.items():
        if hist:
            online_pct = round(sum(1 for s in hist if s == "online") / len(hist) * 100, 1)
            lobe_avail[name] = {
                "availability_pct": online_pct,
                "current": hist[-1],
                "samples": len(hist),
            }
    return {
        "uptime_seconds": round(uptime, 1),
        "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m",
        "total_checks": _check_count,
        "lobe_availability": lobe_avail,
        "timestamp": datetime.now().isoformat(),
    }


async def health_dashboard_serve():
    """Entry point for launching from weaver.py."""
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import uvicorn
    print(f"🌀 Weaver Health Dashboard starting on http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
