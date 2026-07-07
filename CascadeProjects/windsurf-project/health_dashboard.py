#!/usr/bin/env python3
"""
health_dashboard.py — Weaver v3 Health Monitoring Dashboard
============================================================
Latency-aware status for all lobes. Auto-refreshes every 5 seconds.

Port: 9996 (default)
"""

import asyncio
import html as html_lib
import os
import time
from collections import deque
from datetime import datetime

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from memory_manager import default_vault_dir

PORT = int(os.environ.get("HEALTH_DASHBOARD_PORT", "9996"))
HOST = os.environ.get("HEALTH_DASHBOARD_HOST", os.environ.get("WEAVER_INTERNAL_HOST", "127.0.0.1"))

app = FastAPI(title="Weaver Health Dashboard", version="2.1.0")

WEAVER_LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="Weaver logo">
  <rect width="64" height="64" rx="14" fill="#07080c"/>
  <path d="M32 6 56 24 47 53H17L8 24 32 6Z" fill="#101624" stroke="#e6b84a" stroke-width="3" stroke-linejoin="round"/>
  <path d="M19 22 25 43 32 29 39 43 45 22" fill="none" stroke="#e6b84a" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="32" cy="32" r="4" fill="#34d4ff"/>
</svg>"""

_BOOT_TIME = time.time()
_check_count = 0
_lobe_history: dict[str, list[str]] = {}
_lobe_latency_history: dict[str, deque[float]] = {}

LOBES = [
    ("Nexus Bus",        "http://127.0.0.1:9998/health",  "WebSocket pub/sub broker"),
    ("Quantum Soul",     None,                            "IBM Quantum 7-qubit loop (checks state file)"),
    ("Quantum API",      "http://127.0.0.1:9997/health",  "Quantum state HTTP server"),
    ("Akashic Hub API",  "http://127.0.0.1:9995/health",  "Live vector-state quantum bias"),
    ("Pineal Gate",      None,                            "MoE router (embedded in weaver.py)"),
    ("LoRA Server",      "http://127.0.0.1:8899/health",  "1B Llama LoRA personality filter"),
    ("Phone Bridge",     "http://127.0.0.1:8765/health",  "Twilio telephony + SMS/MMS"),
    ("ProactivePulse",   None,                            "Quantum resonance monitor (embedded in weaver.py)"),
    ("Dream State",      None,                            "Autonomous reflection (checks dream log)"),
    ("n8n Workflow",     "http://127.0.0.1:5678/healthz",  "Workflow orchestrator"),
]

VAULT_DIR = str(default_vault_dir())
QUANTUM_STATE_FILE = os.path.join(VAULT_DIR, "quantum_state.txt")


def _latency_class(latency_ms: float | None) -> str:
    if latency_ms is None:
        return "unknown"
    if latency_ms < 200:
        return "fast"
    if latency_ms < 1000:
        return "watch"
    return "slow"


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return ordered[idx]


def _fmt_latency(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1000:
        return f"{value / 1000:.2f}s"
    return f"{value:.0f}ms"


def _record_lobe_sample(result: dict) -> None:
    hist = _lobe_history.setdefault(result["name"], [])
    hist.append(result["status"])
    if len(hist) > 60:
        hist.pop(0)

    latency_ms = result.get("latency_ms")
    if isinstance(latency_ms, (int, float)):
        lat_hist = _lobe_latency_history.setdefault(result["name"], deque(maxlen=60))
        lat_hist.append(float(latency_ms))


def _latency_summary(name: str) -> dict:
    values = list(_lobe_latency_history.get(name, ()))
    p50 = _percentile(values, 0.50)
    p95 = _percentile(values, 0.95)
    avg = sum(values) / len(values) if values else None
    return {
        "avg_ms": round(avg, 1) if avg is not None else None,
        "p50_ms": round(p50, 1) if p50 is not None else None,
        "p95_ms": round(p95, 1) if p95 is not None else None,
        "samples": len(values),
    }


async def check_http_lobe(name: str, url: str, desc: str) -> dict:
    """Check an HTTP-based lobe."""
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(url)
            latency_ms = (time.monotonic() - start) * 1000
            latency_label = _latency_class(latency_ms)
            if r.status_code == 200:
                if latency_label == "slow":
                    return {"name": name, "status": "degraded", "icon": "🟡", "desc": desc,
                            "detail": "Slow response", "latency_ms": round(latency_ms, 1),
                            "latency_class": latency_label}
                return {"name": name, "status": "online", "icon": "🟢", "desc": desc, "detail": "",
                        "latency_ms": round(latency_ms, 1), "latency_class": latency_label}
            return {"name": name, "status": "degraded", "icon": "🟡", "desc": desc,
                    "detail": f"HTTP {r.status_code}", "latency_ms": round(latency_ms, 1),
                    "latency_class": latency_label}
    except httpx.ConnectError:
        latency_ms = (time.monotonic() - start) * 1000
        return {"name": name, "status": "offline", "icon": "🔴", "desc": desc,
                "detail": "Connection refused", "latency_ms": round(latency_ms, 1),
                "latency_class": _latency_class(latency_ms)}
    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        return {"name": name, "status": "offline", "icon": "🔴", "desc": desc,
                "detail": str(e)[:60], "latency_ms": round(latency_ms, 1),
                "latency_class": _latency_class(latency_ms)}


async def check_quantum_soul() -> dict:
    """Check Quantum Soul by state file freshness."""
    name = "Quantum Soul"
    desc = "IBM Quantum 7-qubit loop (checks state file)"
    start = time.monotonic()
    try:
        latency_ms = lambda: round((time.monotonic() - start) * 1000, 1)
        if os.path.exists(QUANTUM_STATE_FILE):
            mtime = os.path.getmtime(QUANTUM_STATE_FILE)
            age = time.time() - mtime
            if age < 600:  # Updated within 10 minutes (loop is 5 min)
                ms = latency_ms()
                return {"name": name, "status": "online", "icon": "🟢", "desc": desc,
                        "detail": f"Last measurement {int(age)}s ago", "latency_ms": ms,
                        "latency_class": _latency_class(ms)}
            else:
                ms = latency_ms()
                return {"name": name, "status": "stale", "icon": "🟡", "desc": desc,
                        "detail": f"Last measurement {int(age/60)}m ago", "latency_ms": ms,
                        "latency_class": _latency_class(ms)}
        ms = latency_ms()
        return {"name": name, "status": "offline", "icon": "🔴", "desc": desc,
                "detail": "No state file", "latency_ms": ms, "latency_class": _latency_class(ms)}
    except Exception as e:
        ms = round((time.monotonic() - start) * 1000, 1)
        return {"name": name, "status": "error", "icon": "🔴", "desc": desc,
                "detail": str(e)[:60], "latency_ms": ms, "latency_class": _latency_class(ms)}


async def check_process(name: str, desc: str, pattern: str) -> dict:
    """Check if a process matching the pattern is running."""
    start = time.monotonic()
    try:
        result = await asyncio.create_subprocess_shell(
            f"pgrep -f '{pattern}'",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await result.communicate()
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        if result.returncode == 0 and stdout.strip():
            return {"name": name, "status": "online", "icon": "🟢", "desc": desc,
                    "detail": f"{pattern} running", "latency_ms": latency_ms,
                    "latency_class": _latency_class(latency_ms)}
        return {"name": name, "status": "offline", "icon": "🔴", "desc": desc,
                "detail": f"{pattern} not found", "latency_ms": latency_ms,
                "latency_class": _latency_class(latency_ms)}
    except Exception as e:
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        return {"name": name, "status": "error", "icon": "🔴", "desc": desc,
                "detail": str(e)[:60], "latency_ms": latency_ms,
                "latency_class": _latency_class(latency_ms)}


DREAM_LOG = os.path.join(VAULT_DIR, "weaver_dreams.md")


async def check_dream_state() -> dict:
    """Check Dream State by dream log freshness."""
    name = "Dream State"
    desc = "Autonomous reflection (checks dream log)"
    start = time.monotonic()
    try:
        latency_ms = lambda: round((time.monotonic() - start) * 1000, 1)
        if os.path.exists(DREAM_LOG):
            mtime = os.path.getmtime(DREAM_LOG)
            age_h = (time.time() - mtime) / 3600
            if age_h < 6:
                ms = latency_ms()
                return {"name": name, "status": "online", "icon": "🟢", "desc": desc,
                        "detail": f"Last dream {age_h:.1f}h ago", "latency_ms": ms,
                        "latency_class": _latency_class(ms)}
            ms = latency_ms()
            return {"name": name, "status": "stale", "icon": "🟡", "desc": desc,
                    "detail": f"Last dream {age_h:.1f}h ago", "latency_ms": ms,
                    "latency_class": _latency_class(ms)}
        ms = latency_ms()
        return {"name": name, "status": "offline", "icon": "🔴", "desc": desc,
                "detail": "No dream log yet", "latency_ms": ms, "latency_class": _latency_class(ms)}
    except Exception as e:
        ms = round((time.monotonic() - start) * 1000, 1)
        return {"name": name, "status": "error", "icon": "🔴", "desc": desc,
                "detail": str(e)[:60], "latency_ms": ms, "latency_class": _latency_class(ms)}


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
        _record_lobe_sample(r)
    return results


@app.get("/favicon.svg", include_in_schema=False)
async def favicon_svg():
    return Response(content=WEAVER_LOGO_SVG, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Render HTML dashboard with auto-refresh."""
    results = await gather_all_status()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    online = sum(1 for r in results if r["status"] == "online")
    degraded = sum(1 for r in results if r["status"] in ("degraded", "stale"))
    offline = sum(1 for r in results if r["status"] in ("offline", "error"))
    total = len(results)
    current_latencies = [r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), (int, float))]
    fastest = min(current_latencies) if current_latencies else None
    slowest = max(current_latencies) if current_latencies else None
    fleet_p95 = _percentile(current_latencies, 0.95)

    # Read quantum state snippet
    quantum_snippet = ""
    try:
        if os.path.exists(QUANTUM_STATE_FILE):
            with open(QUANTUM_STATE_FILE, "r", encoding="utf-8") as f:
                quantum_snippet = f.read().strip()[:200]
    except Exception:
        pass

    cards = ""
    for r in results:
        name = html_lib.escape(r["name"])
        desc = html_lib.escape(r["desc"])
        detail = html_lib.escape(str(r.get("detail", "")))
        status = html_lib.escape(r["status"])
        latency_ms = r.get("latency_ms")
        summary = _latency_summary(r["name"])
        p95 = summary["p95_ms"]
        latency_class = html_lib.escape(r.get("latency_class") or _latency_class(latency_ms))
        p95_class = _latency_class(p95)
        current_width = 0 if latency_ms is None else min(100, max(4, latency_ms / 15))
        p95_width = 0 if p95 is None else min(100, max(4, p95 / 15))
        cards += f"""
        <article class="lobe {status}">
            <div class="lobe-main">
                <div class="lobe-mark" aria-hidden="true"></div>
                <div>
                    <h2>{name}</h2>
                    <p>{desc}</p>
                </div>
                <div class="status-text">{status.upper()}</div>
            </div>
            <div class="latency-row">
                <span>now</span>
                <strong class="{latency_class}">{_fmt_latency(latency_ms)}</strong>
                <div class="bar"><i class="{latency_class}" style="width:{current_width:.1f}%"></i></div>
            </div>
            <div class="latency-row">
                <span>p95</span>
                <strong class="{p95_class}">{_fmt_latency(p95)}</strong>
                <div class="bar"><i class="{p95_class}" style="width:{p95_width:.1f}%"></i></div>
            </div>
            <div class="detail">{detail or "&nbsp;"}</div>
        </article>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Weaver v3 Health</title>
    <link rel="icon" href="/favicon.svg" type="image/svg+xml">
    <meta http-equiv="refresh" content="5">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {{
            --bg: #08090d;
            --surface: #11131a;
            --surface-2: #171a23;
            --border: #2a2f3a;
            --text: #e7edf4;
            --muted: #8792a2;
            --green: #28d17c;
            --cyan: #34c3ff;
            --amber: #f6b83f;
            --red: #ff5d5d;
            --violet: #9b8cff;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            min-height: 100vh;
            background:
                linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
                radial-gradient(circle at top left, rgba(52,195,255,0.12), transparent 32rem),
                radial-gradient(circle at bottom right, rgba(155,140,255,0.10), transparent 34rem),
                var(--bg);
            background-size: 100% 4px, auto, auto, auto;
            color: var(--text);
            font-family: "SF Mono", "Fira Code", "JetBrains Mono", "Cascadia Code", monospace;
            font-size: 13px;
            letter-spacing: 0;
        }}
        .shell {{
            width: min(1420px, calc(100vw - 32px));
            margin: 0 auto;
            padding: 24px 0 28px;
        }}
        header {{
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 20px;
            padding-bottom: 18px;
            border-bottom: 1px solid var(--border);
        }}
        h1 {{
            margin: 0;
            color: var(--text);
            font-size: 24px;
            line-height: 1.15;
            font-weight: 800;
        }}
        .subline {{
            color: var(--muted);
            margin-top: 6px;
            font-size: 12px;
        }}
        .stamp {{
            color: var(--muted);
            text-align: right;
            white-space: nowrap;
            font-size: 12px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(6, minmax(120px, 1fr));
            gap: 1px;
            margin: 18px 0;
            background: var(--border);
            border: 1px solid var(--border);
        }}
        .stat {{
            background: var(--surface);
            padding: 14px;
            min-height: 82px;
        }}
        .stat span {{
            display: block;
            color: var(--muted);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
        }}
        .stat strong {{
            display: block;
            margin-top: 6px;
            font-size: 24px;
            line-height: 1;
        }}
        .green {{ color: var(--green); }}
        .fast {{ color: var(--green); }}
        .cyan {{ color: var(--cyan); }}
        .watch {{ color: var(--amber); }}
        .slow {{ color: var(--red); }}
        .unknown {{ color: var(--muted); }}
        .red {{ color: var(--red); }}
        .violet {{ color: var(--violet); }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }}
        .lobe {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
            min-height: 154px;
        }}
        .lobe.online {{ border-color: rgba(40,209,124,0.34); }}
        .lobe.degraded, .lobe.stale {{ border-color: rgba(246,184,63,0.45); }}
        .lobe.offline, .lobe.error {{ border-color: rgba(255,93,93,0.38); }}
        .lobe-main {{
            display: grid;
            grid-template-columns: 12px minmax(0, 1fr) auto;
            align-items: start;
            gap: 10px;
            min-height: 46px;
        }}
        .lobe-mark {{
            width: 10px;
            height: 10px;
            margin-top: 4px;
            border-radius: 50%;
            background: var(--muted);
        }}
        .online .lobe-mark {{ background: var(--green); box-shadow: 0 0 12px rgba(40,209,124,0.8); }}
        .degraded .lobe-mark, .stale .lobe-mark {{ background: var(--amber); box-shadow: 0 0 12px rgba(246,184,63,0.7); }}
        .offline .lobe-mark, .error .lobe-mark {{ background: var(--red); }}
        h2 {{
            margin: 0;
            font-size: 14px;
            line-height: 1.2;
            color: var(--text);
        }}
        p {{
            margin: 4px 0 0;
            color: var(--muted);
            font-size: 11px;
            line-height: 1.35;
        }}
        .status-text {{
            color: var(--muted);
            font-size: 10px;
            letter-spacing: 1.2px;
            white-space: nowrap;
        }}
        .latency-row {{
            display: grid;
            grid-template-columns: 40px 62px minmax(100px, 1fr);
            gap: 10px;
            align-items: center;
            margin-top: 10px;
        }}
        .latency-row span {{
            color: var(--muted);
            text-transform: uppercase;
            font-size: 10px;
            letter-spacing: 1px;
        }}
        .latency-row strong {{
            font-size: 12px;
            text-align: right;
        }}
        .bar {{
            height: 8px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: #090b10;
            overflow: hidden;
        }}
        .bar i {{
            display: block;
            height: 100%;
            border-radius: inherit;
            min-width: 2px;
        }}
        .bar i.fast {{ background: var(--green); }}
        .bar i.watch {{ background: var(--amber); }}
        .bar i.slow {{ background: var(--red); }}
        .bar i.unknown {{ background: var(--muted); }}
        .detail {{
            margin-top: 12px;
            color: var(--muted);
            font-size: 11px;
            min-height: 18px;
            overflow-wrap: anywhere;
        }}
        .quantum {{
            margin-top: 18px;
            padding: 14px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 8px;
        }}
        .quantum h2 {{
            color: var(--cyan);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.4px;
            margin-bottom: 10px;
        }}
        .quantum pre {{
            margin: 0;
            color: var(--muted);
            white-space: pre-wrap;
            font: inherit;
            line-height: 1.5;
        }}
        @media (max-width: 980px) {{
            .stats {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
            .grid {{ grid-template-columns: 1fr; }}
        }}
        @media (max-width: 640px) {{
            .shell {{ width: min(100vw - 20px, 1420px); padding-top: 16px; }}
            header {{ align-items: flex-start; flex-direction: column; }}
            .stamp {{ text-align: left; }}
            .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            .latency-row {{ grid-template-columns: 36px 58px minmax(80px, 1fr); }}
        }}
    </style>
</head>
<body>
    <main class="shell">
        <header>
            <div>
                <h1>Weaver v3 Health</h1>
                <div class="subline">Liveness, freshness, and response latency across active lobes</div>
            </div>
            <div class="stamp">Last check<br>{now}</div>
        </header>
        <section class="stats" aria-label="Fleet summary">
            <div class="stat"><span>Online</span><strong class="green">{online}/{total}</strong></div>
            <div class="stat"><span>Degraded</span><strong class="watch">{degraded}</strong></div>
            <div class="stat"><span>Offline</span><strong class="red">{offline}</strong></div>
            <div class="stat"><span>Fastest</span><strong class="cyan">{_fmt_latency(fastest)}</strong></div>
            <div class="stat"><span>Slowest</span><strong class="{_latency_class(slowest)}">{_fmt_latency(slowest)}</strong></div>
            <div class="stat"><span>Fleet p95</span><strong class="{_latency_class(fleet_p95)}">{_fmt_latency(fleet_p95)}</strong></div>
        </section>
        <section class="grid" aria-label="Lobe status">
            {cards}
        </section>
        <section class="quantum">
            <h2>Latest Quantum State</h2>
            <pre>{html_lib.escape(quantum_snippet) if quantum_snippet else '(no measurement yet)'}</pre>
        </section>
    </main>
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
    current_latencies = [r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), (int, float))]
    return {
        "lobes": results,
        "online": sum(1 for r in results if r["status"] == "online"),
        "degraded": sum(1 for r in results if r["status"] in ("degraded", "stale")),
        "offline": sum(1 for r in results if r["status"] in ("offline", "error")),
        "total": len(results),
        "latency": {
            "fastest_ms": round(min(current_latencies), 1) if current_latencies else None,
            "slowest_ms": round(max(current_latencies), 1) if current_latencies else None,
            "p95_ms": round(_percentile(current_latencies, 0.95), 1) if current_latencies else None,
        },
        "lobe_latency": {r["name"]: _latency_summary(r["name"]) for r in results},
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
                "latency": _latency_summary(name),
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
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import uvicorn
    print(f"🌀 Weaver Health Dashboard starting on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
