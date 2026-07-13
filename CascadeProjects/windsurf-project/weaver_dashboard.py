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
  - Quantum state + 156-qubit Kingston manifold visualization
  - Vault file readers (transcripts, people, dreams)
  - System metrics + uptime tracking
  - SSE stream for zero-refresh browser updates
  - Auto ngrok tunnel with persistent public URL
"""

import asyncio
import ast
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
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from memory_manager import default_vault_dir

PORT = int(os.environ.get("WEAVER_DASHBOARD_PORT", "9990"))
HOST = os.environ.get("WEAVER_DASHBOARD_HOST", os.environ.get("WEAVER_INTERNAL_HOST", "127.0.0.1"))
PROJ = os.path.dirname(os.path.abspath(__file__))
VAULT = str(default_vault_dir())
N8N_WEBHOOK_URL = (
    os.environ.get("N8N_WEBHOOK_URL")
    or os.environ.get("WEAVER_N8N_WEBHOOK_URL")
    or "http://127.0.0.1:5678/webhook/weaver-input"
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
_nexus_publisher = None
_sse_subscribers: list[asyncio.Queue] = []

LOBES = [
    ("Nexus Bus",       "http://127.0.0.1:9998/health",  9998, "WebSocket pub/sub broker"),
    ("AWS Brain API",   "http://127.0.0.1:8093/health",  8093, "Bedrock/Nova unified cortex"),
    ("Headless UI",     "http://127.0.0.1:8093/health",  8093, "Headless Nova presence"),
    ("Trained Voice",   "http://127.0.0.1:8092/health",  8092, "OpenVoice cloned voice"),
    ("Codebase API",    "http://127.0.0.1:8091/health",  8091, "Bounded self-inspection API"),
    ("Quantum Soul",    None,                            None, "156-qubit Kingston manifold"),
    ("Quantum API",     "http://127.0.0.1:9997/health",  9997, "Quantum state HTTP server"),
    ("Akashic Hub",     "http://127.0.0.1:9995/health",  9995, "Shared vector state"),
    ("Pineal Gate",     None,                            None, "Kingston entropy router"),
    ("LoRA Server",     "http://127.0.0.1:8899/health",  8899, "1B Llama Soul Voice"),
    ("Qwen3B Branch",    "http://127.0.0.1:8898/health",  8898, "Local Qwen branch"),
    ("Phone Bridge",    "http://127.0.0.1:8765/health",  8765, "Twilio telephony"),
    ("Health Dashboard","http://127.0.0.1:9996/health",  9996, "Legacy traffic-light"),
    ("ProactivePulse",  None,                            None, "Quantum resonance monitor"),
    ("Dream State",     None,                            None, "Autonomous reflection"),
    ("n8n Workflow",    "http://127.0.0.1:5678/healthz",  5678, "Workflow orchestrator"),
    ("Discord Bridge", "http://127.0.0.1:8770/health",   8770, "Discord voice/vision"),
]


def _discover_codebase_root() -> Path:
    start = Path(PROJ).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "avatar").exists() and (candidate / "CascadeProjects" / "windsurf-project").exists():
            return candidate
    return start


CODEBASE_ROOT = Path(os.environ.get("WEAVER_DASHBOARD_CODEBASE_ROOT") or _discover_codebase_root()).resolve()
WIND = "CascadeProjects/windsurf-project"
_SKIP_PATH_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", "dist", "build", ".terraform",
}
_THREAD_CACHE: dict[str, dict] = {}
_THREAD_DETAIL_CACHE: dict[str, dict] = {}

THREAD_DEFS = [
    {
        "id": "runtime",
        "label": "Runtime Spine",
        "ring": "core",
        "desc": "Supervisor, launcher, and process lifecycle wiring",
        "status_lobes": ["Nexus Bus", "AWS Brain API", "Codebase API"],
        "keywords": ["weaver", "supervisor", "startup", "health"],
        "patterns": [
            f"{WIND}/weaver.py",
            f"{WIND}/start_weaver.sh",
            f"{WIND}/setup_weaver.sh",
            f"{WIND}/Makefile",
        ],
    },
    {
        "id": "cortex",
        "label": "Cortex Routes",
        "ring": "cortex",
        "desc": "Bedrock/Nova routing, specialist selection, and unified model surface",
        "status_lobes": ["AWS Brain API", "Headless UI"],
        "keywords": ["brain", "cortex", "route", "model", "bedrock", "headless"],
        "patterns": [
            f"{WIND}/bedrock_brain_api.py",
            f"{WIND}/slm_experts.py",
            f"{WIND}/lora_server.py",
            f"{WIND}/qwen3b_server.py",
            f"{WIND}/n8n_weaver_*.json",
        ],
    },
    {
        "id": "voice",
        "label": "Voice Stack",
        "ring": "interface",
        "desc": "Realtime voice, trained TTS, phone speech, and local audio loops",
        "status_lobes": ["Trained Voice", "Phone Bridge", "Discord Bridge"],
        "keywords": ["voice", "audio", "tts", "speech", "phone", "discord"],
        "patterns": [
            f"{WIND}/deploy/tts/**/*.py",
            f"{WIND}/deploy/tts/**/*.sh",
            f"{WIND}/vtv_basic.py",
            f"{WIND}/voice_recognition.py",
            f"{WIND}/twilio_weaver_bridge.py",
            f"{WIND}/discord_bridge.py",
        ],
    },
    {
        "id": "memory",
        "label": "Memory Plane",
        "ring": "core",
        "desc": "Persistent vault memory, Akashic vector state, and Obsidian sync",
        "status_lobes": ["Akashic Hub"],
        "keywords": ["memory", "akashic", "vault", "obsidian", "recall"],
        "patterns": [
            f"{WIND}/memory_manager.py",
            f"{WIND}/akashic_hub.py",
            f"{WIND}/obsidian_bridge.py",
            f"{WIND}/nexus_client.py",
            "CascadeProjects/SYPHER_VAULT/**/*.md",
        ],
    },
    {
        "id": "bus",
        "label": "Nexus Bus",
        "ring": "core",
        "desc": "WebSocket pub/sub fabric and live event fanout",
        "status_lobes": ["Nexus Bus"],
        "keywords": ["nexus", "broadcast", "pubsub", "topic"],
        "patterns": [
            f"{WIND}/nexus_bus.py",
            f"{WIND}/nexus_client.py",
        ],
    },
    {
        "id": "quantum",
        "label": "Quantum Mesh",
        "ring": "cortex",
        "desc": "Kingston manifold, quantum API, and bias projection",
        "status_lobes": ["Quantum Soul", "Quantum API"],
        "keywords": ["quantum", "kingston", "qubit", "interference"],
        "patterns": [
            f"{WIND}/quantum_*.py",
            f"{WIND}/weaver_core/quantum*.py",
            f"{WIND}/weaver_core/pineal_gate.py",
        ],
    },
    {
        "id": "routing",
        "label": "Routing Gate",
        "ring": "cortex",
        "desc": "Liquid fracture, Pineal entropy routing, and expert collapse",
        "status_lobes": ["Pineal Gate", "ProactivePulse"],
        "keywords": ["pineal", "gate", "fracture", "routing", "expert"],
        "patterns": [
            f"{WIND}/pineal_gate.py",
            f"{WIND}/liquid_fracture.py",
            f"{WIND}/slm_experts.py",
            f"{WIND}/weaver_core/pineal_gate.py",
            f"{WIND}/weaver_core/quantum_governor.py",
        ],
    },
    {
        "id": "browser",
        "label": "Browser Bodies",
        "ring": "interface",
        "desc": "Embodied avatar, headless presence, hotkeys, and visual assets",
        "status_lobes": ["Headless UI"],
        "keywords": ["browser", "avatar", "headless", "vision", "ui"],
        "patterns": [
            "avatar/**/*.html",
            "avatar/**/*.py",
            "avatar/**/*.svg",
            "tools/local-linux-hotkeys/**/*",
        ],
    },
    {
        "id": "codebase",
        "label": "Self Inspection",
        "ring": "interface",
        "desc": "Read-only codebase and bounded public-web context APIs",
        "status_lobes": ["Codebase API"],
        "keywords": ["codebase", "search", "context", "inspect"],
        "patterns": [
            f"{WIND}/codebase_api.py",
            f"{WIND}/README.md",
            "README.md",
        ],
    },
    {
        "id": "bridges",
        "label": "External Bridges",
        "ring": "interface",
        "desc": "Twilio, Discord, Obsidian, and operator bridge modules",
        "status_lobes": ["Phone Bridge", "Discord Bridge"],
        "keywords": ["bridge", "twilio", "discord", "obsidian", "sms"],
        "patterns": [
            f"{WIND}/*bridge*.py",
            f"{WIND}/weaver_tools.py",
        ],
    },
    {
        "id": "dashboards",
        "label": "Dashboards",
        "ring": "support",
        "desc": "Operator control plane, health dashboard, and status projection",
        "status_lobes": ["Health Dashboard"],
        "keywords": ["dashboard", "status", "health", "telemetry"],
        "patterns": [
            f"{WIND}/weaver_dashboard.py",
            f"{WIND}/health_dashboard.py",
            f"{WIND}/deploy/Caddyfile",
        ],
    },
    {
        "id": "deployment",
        "label": "Deployment",
        "ring": "support",
        "desc": "Caddy, systemd, Docker, Terraform, and cloud setup scripts",
        "status_lobes": ["n8n Workflow"],
        "keywords": ["deploy", "caddy", "systemd", "docker", "terraform"],
        "patterns": [
            f"{WIND}/deploy/**/*",
            f"{WIND}/Dockerfile*",
            f"{WIND}/docker-compose.yml",
        ],
    },
    {
        "id": "training",
        "label": "Training Forge",
        "ring": "support",
        "desc": "Datasets, distillation, MoE pretraining, and forge scripts",
        "status_lobes": ["LoRA Server", "Qwen3B Branch"],
        "keywords": ["train", "forge", "lora", "dataset", "distill"],
        "patterns": [
            f"{WIND}/forge*.py",
            f"{WIND}/pretrain_moe/**/*.py",
            f"{WIND}/pretrain_moe/**/*.md",
        ],
    },
    {
        "id": "validation",
        "label": "Validation",
        "ring": "support",
        "desc": "Integration, stress, audio, and workflow test coverage",
        "status_lobes": ["Health Dashboard"],
        "keywords": ["test", "validation", "stress", "debug"],
        "patterns": [
            f"{WIND}/tests/**/*.py",
            f"{WIND}/weaver_preflight.py",
            f"{WIND}/whole_codebase_tests.py",
        ],
    },
]

THREAD_EDGES = [
    ("runtime", "bus"), ("runtime", "cortex"), ("runtime", "quantum"),
    ("runtime", "dashboards"), ("bus", "memory"), ("bus", "voice"),
    ("bus", "browser"), ("bus", "bridges"), ("cortex", "routing"),
    ("cortex", "voice"), ("cortex", "memory"), ("routing", "quantum"),
    ("routing", "training"), ("quantum", "memory"), ("browser", "voice"),
    ("browser", "codebase"), ("dashboards", "codebase"), ("dashboards", "bus"),
    ("deployment", "runtime"), ("validation", "runtime"), ("bridges", "voice"),
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


def _rel_code_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(CODEBASE_ROOT).as_posix()
    except Exception:
        return path.name


def _allowed_code_path(path: Path) -> bool:
    rel_parts = set(path.parts)
    if rel_parts & _SKIP_PATH_PARTS:
        return False
    name = path.name.lower()
    if name.endswith((".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".wav", ".mp3", ".pth", ".gguf")):
        return False
    if name in {"tokenizer.json", "agentic_traces.json", "agentic_traces.jsonl"}:
        return False
    try:
        if path.stat().st_size > 2_000_000:
            return False
    except OSError:
        return False
    return True


def _thread_paths(patterns: list[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in patterns:
        try:
            matches = CODEBASE_ROOT.glob(pattern)
            for path in matches:
                if path.is_file() and _allowed_code_path(path):
                    paths[_rel_code_path(path)] = path
        except Exception:
            continue
    return [paths[key] for key in sorted(paths)]


def _count_lines(path: Path) -> int:
    try:
        total = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 64), b""):
                total += chunk.count(b"\n")
        return total
    except Exception:
        return 0


def _file_metric(path: Path) -> dict:
    try:
        stat = path.stat()
        rel = _rel_code_path(path)
        cached = _THREAD_CACHE.get(rel)
        if cached and cached.get("mtime") == stat.st_mtime and cached.get("size") == stat.st_size:
            return cached
        item = {
            "path": rel,
            "name": path.name,
            "lines": _count_lines(path),
            "size_kb": round(stat.st_size / 1024, 1),
            "mtime": stat.st_mtime,
        }
        _THREAD_CACHE[rel] = item
        return item
    except Exception:
        return {"path": _rel_code_path(path), "name": path.name, "lines": 0, "size_kb": 0, "mtime": 0}


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        parent = _decorator_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _source_detail(path: Path) -> dict:
    rel = _rel_code_path(path)
    try:
        stat = path.stat()
    except OSError:
        return {"symbols": [], "imports": [], "endpoints": [], "selectors": [], "updated": ""}
    cache_key = f"{rel}:{stat.st_mtime}:{stat.st_size}"
    cached = _THREAD_DETAIL_CACHE.get(cache_key)
    if cached:
        return cached

    detail = {
        "symbols": [],
        "imports": [],
        "endpoints": [],
        "selectors": [],
        "updated": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
    }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:180_000]
    except Exception:
        _THREAD_DETAIL_CACHE[cache_key] = detail
        return detail

    suffix = path.suffix.lower()
    if suffix == ".py":
        try:
            tree = ast.parse(text)
            symbols: list[str] = []
            imports: set[str] = set()
            endpoints: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "fn"
                    symbols.append(f"{kind} {node.name}:{getattr(node, 'lineno', 0)}")
                    for dec in getattr(node, "decorator_list", []):
                        name = _decorator_name(dec)
                        if name.startswith("app.") or ".route" in name:
                            route = ""
                            if isinstance(dec, ast.Call) and dec.args and isinstance(dec.args[0], ast.Constant):
                                route = str(dec.args[0].value)
                            endpoints.append(f"{name} {route}".strip())
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
            detail["symbols"] = symbols[:14]
            detail["imports"] = sorted(imports)[:14]
            detail["endpoints"] = endpoints[:10]
        except SyntaxError:
            pass
    elif suffix in {".html", ".js", ".mjs"}:
        functions = re.findall(r"\b(?:async\s+)?function\s+([A-Za-z0-9_$]+)\s*\(", text)
        const_fns = re.findall(r"\b(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?\(", text)
        ids = re.findall(r"\bid=['\"]([A-Za-z0-9_:-]+)['\"]", text)
        detail["symbols"] = [f"fn {name}" for name in (functions + const_fns)[:14]]
        detail["selectors"] = [f"#{name}" for name in ids[:14]]
    elif suffix in {".json", ".service", ".sh", ".md", ".yml", ".yaml", ".tf"}:
        headings = re.findall(r"(?m)^(?:#{1,3}\s+|\[Unit\]|\[Service\]|[A-Za-z0-9_.-]+\s*=)\s*(.{2,90})", text)
        commands = re.findall(r"(?m)^\s*(?:ExecStart|command|CMD|RUN|uvicorn|python3?|node|docker|systemctl)\b[^\n]{0,120}", text)
        detail["symbols"] = [s.strip()[:90] for s in headings[:8]]
        detail["endpoints"] = [s.strip()[:120] for s in commands[:8]]

    # Keep the cache bounded. This runs every dashboard poll.
    if len(_THREAD_DETAIL_CACHE) > 500:
        _THREAD_DETAIL_CACHE.clear()
    _THREAD_DETAIL_CACHE[cache_key] = detail
    return detail


def _combined_status(statuses: list[str]) -> str:
    clean = [s for s in statuses if s]
    if not clean:
        return "unknown"
    if all(s == "online" for s in clean):
        return "online"
    if any(s == "online" for s in clean):
        return "degraded"
    if any(s in {"degraded", "stale"} for s in clean):
        return "degraded"
    if all(s in {"offline", "error"} for s in clean):
        return "offline"
    return "unknown"


def _current_nexus_lobes(lobes: list[dict]) -> list[str]:
    for lobe in lobes:
        if lobe.get("name") != "Nexus Bus":
            continue
        detail = lobe.get("detail")
        if isinstance(detail, dict) and isinstance(detail.get("lobe_ids"), list):
            return [str(item) for item in detail["lobe_ids"]]
    return []


def _status_activity(status: str) -> float:
    return {
        "online": 0.78,
        "degraded": 0.54,
        "stale": 0.42,
        "offline": 0.12,
        "error": 0.12,
        "unknown": 0.28,
    }.get(status, 0.28)


def _recent_topic_hits(keywords: list[str], feed: list[dict]) -> int:
    if not keywords:
        return 0
    words = [w.lower() for w in keywords]
    hits = 0
    for item in feed[-25:]:
        text = " ".join(str(item.get(k, "")) for k in ("from", "topic", "payload")).lower()
        if any(word in text for word in words):
            hits += 1
    return hits


def _recent_topic_events(keywords: list[str], feed: list[dict], limit: int = 5) -> list[dict]:
    if not keywords:
        return []
    words = [w.lower() for w in keywords]
    events: list[dict] = []
    for item in reversed(feed[-50:]):
        text = " ".join(str(item.get(k, "")) for k in ("from", "topic", "payload")).lower()
        if not any(word in text for word in words):
            continue
        events.append({
            "from": item.get("from", ""),
            "topic": item.get("topic", ""),
            "ts": item.get("ts", ""),
            "payload": str(item.get("payload", ""))[:220],
        })
        if len(events) >= limit:
            break
    return events


def read_codebase_threads(
    lobes: list[dict],
    brain: dict,
    voice: dict,
    codebase: dict,
    nexus_feed: list[dict],
) -> dict:
    lobe_by_name = {str(l.get("name", "")): l for l in lobes}
    now = time.time()
    threads: list[dict] = []
    unique_files: dict[str, int] = {}
    active_threads = 0

    for item in THREAD_DEFS:
        paths = _thread_paths(item["patterns"])
        metrics = [_file_metric(path) for path in paths]
        files = len(metrics)
        lines = sum(int(m.get("lines", 0) or 0) for m in metrics)
        newest = max((float(m.get("mtime", 0) or 0) for m in metrics), default=0)
        lobe_statuses = [str(lobe_by_name.get(name, {}).get("status", "")) for name in item.get("status_lobes", [])]
        status = _combined_status(lobe_statuses)

        if item["id"] == "cortex" and brain.get("status"):
            status = _combined_status([status, str(brain.get("status"))])
        elif item["id"] == "voice" and voice.get("status"):
            status = _combined_status([status, str(voice.get("status"))])
        elif item["id"] == "codebase" and codebase.get("status"):
            status = _combined_status([status, str(codebase.get("status"))])

        topic_hits = _recent_topic_hits(item.get("keywords", []), nexus_feed)
        fresh_bonus = 0.10 if newest and now - newest < 3600 else 0.04 if newest and now - newest < 86400 else 0.0
        density = min(0.12, files / 180)
        activation = min(1.0, _status_activity(status) + min(0.18, topic_hits * 0.045) + fresh_bonus + density)
        if activation >= 0.55:
            active_threads += 1
        for metric in metrics:
            unique_files[str(metric.get("path", ""))] = int(metric.get("lines", 0) or 0)

        hot_files = sorted(metrics, key=lambda m: (m.get("mtime", 0), m.get("lines", 0)), reverse=True)[:14]
        enriched_files = []
        for metric in hot_files:
            file_path = CODEBASE_ROOT / str(metric.get("path", ""))
            enriched_files.append({**metric, **_source_detail(file_path)})
        lobe_details = []
        for lobe_name in item.get("status_lobes", []):
            lobe = lobe_by_name.get(lobe_name)
            if not lobe:
                lobe_details.append({"name": lobe_name, "status": "unknown", "latency_ms": None, "detail": ""})
                continue
            detail = lobe.get("detail", "")
            if isinstance(detail, dict):
                detail = detail.get("service") or detail.get("status") or json.dumps(detail, ensure_ascii=False)[:160]
            lobe_details.append({
                "name": lobe_name,
                "status": lobe.get("status", "unknown"),
                "latency_ms": lobe.get("latency_ms"),
                "detail": str(detail)[:180],
            })
        newest_file = enriched_files[0] if enriched_files else {}
        symbols = []
        endpoints = []
        imports = []
        selectors = []
        for file_info in enriched_files:
            symbols.extend(file_info.get("symbols", []))
            endpoints.extend(file_info.get("endpoints", []))
            imports.extend(file_info.get("imports", []))
            selectors.extend(file_info.get("selectors", []))
        threads.append({
            "id": item["id"],
            "label": item["label"],
            "ring": item["ring"],
            "desc": item["desc"],
            "status": status,
            "activation": round(activation, 3),
            "topic_hits": topic_hits,
            "files": files,
            "lines": lines,
            "newest_age_s": round(now - newest) if newest else None,
            "newest_file": newest_file.get("path", ""),
            "updated": newest_file.get("updated", ""),
            "status_lobes": item.get("status_lobes", []),
            "lobe_details": lobe_details,
            "recent_events": _recent_topic_events(item.get("keywords", []), nexus_feed),
            "symbols": symbols[:18],
            "endpoints": endpoints[:14],
            "imports": sorted(set(imports))[:16],
            "selectors": selectors[:14],
            "hot_files": enriched_files,
        })

    return {
        "root": str(CODEBASE_ROOT),
        "generated_at": datetime.now().isoformat(),
        "stats": {
            "threads": len(threads),
            "active": active_threads,
            "files": len(unique_files),
            "lines": sum(unique_files.values()),
            "edges": len(THREAD_EDGES),
        },
        "threads": threads,
        "edges": [{"from": a, "to": b} for a, b in THREAD_EDGES],
    }


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
                        "desc": "156-qubit Kingston manifold",
                        "detail": f"Last measurement {int(age)}s ago"}
            return {"name": "Quantum Soul", "status": "stale",
                    "desc": "156-qubit Kingston manifold",
                    "detail": f"Last measurement {int(age / 60)}m ago"}
        return {"name": "Quantum Soul", "status": "offline",
                "desc": "156-qubit Kingston manifold", "detail": "No state file"}
    except Exception as e:
        return {"name": "Quantum Soul", "status": "offline",
                "desc": "156-qubit Kingston manifold", "detail": str(e)[:80]}


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

DIMENSION_MAP = {
    "Logic": "logic", "Emotion": "emotion", "Intuition": "creativity",
    "Memory": "memory", "Sovereignty": "vigilance", "Attention": "emotion",
    "Reflection": "memory", "Language": "logic", "Planning": "creativity",
    "Novelty": "creativity", "Stability": "vigilance", "Meta-Reasoning": "logic",
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

        dom_match = re.search(r"reveals? ([\w-]+) as the Dominant Pathway", text)
        if dom_match:
            result["dominant"] = dom_match.group(1)

        sec_match = re.search(r"with ([\w-]+) resonating", text)
        if sec_match:
            result["secondary"] = sec_match.group(1)

        bitstring = result.get("bitstring")
        if bitstring:
            from quantum_networks import DIMENSION_QUBITS

            qubits = bitstring.zfill(12)[::-1]
            for dim, indices in DIMENSION_QUBITS.items():
                active = sum(qubits[index] == "1" for index in indices if index < len(qubits))
                result["weights"][dim] = round(active / max(len(indices), 1), 3)

        dominant = result["dominant"]
        if dominant in DIMENSION_MAP and DIMENSION_MAP[dominant] in result["weights"]:
            result["weights"][DIMENSION_MAP[dominant]] = max(
                result["weights"][DIMENSION_MAP[dominant]], 0.95
            )

        secondary = result.get("secondary")
        if secondary and secondary in DIMENSION_MAP and DIMENSION_MAP[secondary] in result["weights"]:
            result["weights"][DIMENSION_MAP[secondary]] = max(
                result["weights"][DIMENSION_MAP[secondary]], 0.85
            )

        _quantum_state = result
        return result
    except Exception:
        return _quantum_state


def read_quantum_architecture() -> dict:
    try:
        from dataclasses import asdict
        from quantum_networks import (
            ARCHITECTURE_MODULES,
            CORE_QUBITS,
            SYSTEM_SUMMARY,
            TOPOLOGICAL_LAYERS,
            EntanglementTopology,
            architecture_graph_stats,
        )
        return {
            "summary": dict(SYSTEM_SUMMARY),
            "core_qubits": [asdict(q) for q in CORE_QUBITS],
            "modules": [asdict(m) for m in ARCHITECTURE_MODULES],
            "topological_layers": [asdict(layer) for layer in TOPOLOGICAL_LAYERS],
            "graph_stats": architecture_graph_stats(),
            "topology_edges": {
                "dodecahedron": EntanglementTopology.dodecahedron(),
                "state_encoding": EntanglementTopology.state_encoding(),
                "open_system": EntanglementTopology.open_system(),
                "entropy_routing": EntanglementTopology.entropy_routing(),
                "measurement_readout": EntanglementTopology.measurement_readout(),
            },
        }
    except Exception as exc:
        return {"error": str(exc)[:160]}


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

    voice_runtime = state.get("voice_realtime") if isinstance(state.get("voice_realtime"), dict) else {}
    raw_slo = voice_runtime.get("slo") if isinstance(voice_runtime.get("slo"), dict) else {}
    slo_keys = {
        "status", "window", "samples", "success_rate", "error_budget_remaining_pct",
        "reaction_target_ms", "queue_target_ms", "semantic_target_ms",
        "reaction_p50_ms", "reaction_p95_ms", "queue_p50_ms", "queue_p95_ms",
        "cortex_p50_ms", "cortex_p95_ms", "semantic_p50_ms", "semantic_p95_ms",
    }
    voice_slo = {
        key: value for key, value in raw_slo.items()
        if key in slo_keys and (isinstance(value, (int, float)) or key == "status")
    }
    raw_prewarm = voice_runtime.get("prewarm") if isinstance(voice_runtime.get("prewarm"), dict) else {}
    voice_prewarm = {
        "enabled": bool(raw_prewarm.get("enabled")),
        "status": str(raw_prewarm.get("status") or "pending")[:24],
        "latency_ms": raw_prewarm.get("latency_ms") if isinstance(raw_prewarm.get("latency_ms"), (int, float)) else None,
    }

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
        "voice_slo": voice_slo,
        "voice_prewarm": voice_prewarm,
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
                    "overmind_directive", "routing", "gate_decision", "interference",
                    "dream_state", "proactive_pulse", "lobe_status", "sms_exchange",
                    "state_reconciliation", "phone_transcript", "discord_transcript",
                    "memory_update", "timer_fired", "reminder_fired", "play_sound",
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
        feed_snapshot = list(_nexus_feed)
        codebase_threads = await asyncio.to_thread(
            read_codebase_threads, lobes, brain, voice, codebase, feed_snapshot
        )
        uptime = time.time() - _boot_time
        online = sum(1 for l in lobes if l["status"] == "online")
        nexus_lobes = _current_nexus_lobes(lobes)
        payload = {
            "type": "poll",
            "lobes": lobes,
            "quantum": qs,
            "quantum_architecture": read_quantum_architecture(),
            "brain": brain,
            "voice": voice,
            "codebase": codebase,
            "codebase_threads": codebase_threads,
            "memory_events": read_memory_events(8),
            "online": online,
            "total": len(lobes),
            "uptime": round(uptime),
            "poll_count": _poll_count,
            "nexus_msg_count": _nexus_stats["msg_count"],
            "nexus_lobes": len(nexus_lobes),
            "nexus_lobe_ids": nexus_lobes,
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
    if os.environ.get("WEAVER_DASHBOARD_TUNNEL", "").lower() not in {"1", "true", "yes", "on"}:
        print("[DASHBOARD] public quick tunnel disabled (set WEAVER_DASHBOARD_TUNNEL=1 to enable)", flush=True)
        return
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
    feed_snapshot = list(_nexus_feed)
    codebase_threads = await asyncio.to_thread(
        read_codebase_threads, lobes, brain, voice, codebase, feed_snapshot
    )
    nexus_lobes = _current_nexus_lobes(lobes)
    return {
        "lobes": lobes,
        "quantum": qs,
        "quantum_architecture": read_quantum_architecture(),
        "brain": brain,
        "voice": voice,
        "codebase": codebase,
        "codebase_threads": codebase_threads,
        "memory_events": read_memory_events(10),
        "online": sum(1 for l in lobes if l["status"] == "online"),
        "total": len(lobes),
        "uptime": round(time.time() - _boot_time),
        "nexus_feed": list(_nexus_feed)[-20:],
        "nexus_stats": {
            "msg_count": _nexus_stats["msg_count"],
            "lobes_seen": list(_nexus_stats["lobes_seen"]),
            "current_lobe_ids": nexus_lobes,
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
    global _nexus_publisher
    try:
        body = await request.json()
    except Exception:
        return {"error": "invalid JSON"}
    if not isinstance(body, dict):
        return {"error": "body must be a JSON object"}
    topic = str(body.get("topic", "dashboard")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", topic):
        return {"error": "invalid topic"}
    payload = body.get("payload", {})
    if len(json.dumps(payload, ensure_ascii=False)) > 65_536:
        return {"error": "payload too large"}
    try:
        from nexus_client import NexusClient
        if _nexus_publisher is None:
            _nexus_publisher = NexusClient("dashboard_control")
        if await _nexus_publisher.publish(topic, payload):
            return {"ok": True, "topic": topic}
        return {"error": "Nexus Bus unavailable"}
    except Exception as e:
        return {"error": str(e)[:200]}


# ── Main HTML dashboard ─────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/favicon.svg", include_in_schema=False)
async def favicon_svg():
    return Response(content=WEAVER_LOGO_SVG, media_type="image/svg+xml")


@app.get("/vendor/three.module.js", include_in_schema=False)
async def three_module_js():
    path = CODEBASE_ROOT / "avatar" / "vendor" / "three.module.js"
    if not path.exists():
        return Response("three.module.js not found", status_code=404)
    return FileResponse(path, media_type="application/javascript")




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
.tab-btn:focus-visible { outline: 2px solid var(--cyan); outline-offset: -3px; }
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
.ops-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin-bottom: 10px; }
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
.signal-value.nominal { color: var(--green); }
.signal-value.watch { color: var(--orange); }
.signal-value.breached { color: var(--red); }
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

/* Kingston Manifold */
.manifold-panel { display: grid; gap: 10px; }
.manifold-frame {
  position: relative;
  width: 100%;
  aspect-ratio: 1.12 / 1;
  min-height: 330px;
  overflow: hidden;
  border: 1px solid rgba(52,212,255,0.14);
  border-radius: 8px;
  background:
    radial-gradient(circle at 50% 42%, rgba(52,212,255,0.08), transparent 55%),
    linear-gradient(180deg, rgba(11,14,22,0.92), rgba(3,4,8,0.96));
}
.manifold-canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.manifold-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; }
.q-stat {
  min-width: 0;
  border: 1px solid rgba(30,37,54,0.9);
  border-radius: 7px;
  background: rgba(3,4,8,0.42);
  padding: 7px 8px;
}
.q-stat-label { color: var(--dim); font-size: 7px; font-weight: 700; letter-spacing: 0.7px; text-transform: uppercase; }
.q-stat-val { color: var(--text); font-size: 12px; font-weight: 700; font-variant-numeric: tabular-nums; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.q-process-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
.q-process-card {
  min-width: 0;
  border: 1px solid rgba(30,37,54,0.72);
  border-left: 2px solid var(--cyan);
  border-radius: 7px;
  background: rgba(8,10,16,0.58);
  padding: 7px 8px;
}
.q-process-title { color: #eef3ff; font-size: 8px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.q-process-detail { color: var(--dim); font-size: 8px; line-height: 1.35; margin-top: 3px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.quantum-info { display: grid; grid-template-columns: 1fr; gap: 6px; font-size: 10px; }
.q-dominant { font: 700 17px/1.2 'JetBrains Mono',monospace; color: var(--violet); }
.q-bits { font: 300 12px/1.2 'JetBrains Mono',monospace; color: var(--cyan); letter-spacing: 1px; overflow-wrap: anywhere; }
.q-meta { color: var(--dim); line-height: 1.6; white-space: pre-line; }
.q-raw { font-size: 9px; color: var(--muted); line-height: 1.5; max-height: 76px; overflow-y: auto; }

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
.neural-wrap.stale canvas { opacity: 0.42; filter: grayscale(0.7) saturate(0.35); }
.neural-live-overlay {
  position: absolute; left: 50%; top: 16px; transform: translateX(-50%);
  display: none; z-index: 18; pointer-events: none;
  border: 1px solid rgba(255,184,48,0.35); border-radius: 8px;
  background: rgba(8,10,16,0.88); color: var(--orange);
  padding: 8px 12px; font-size: 9px; font-weight: 700;
  letter-spacing: 1.2px; text-transform: uppercase;
  box-shadow: 0 8px 32px rgba(0,0,0,0.45);
}
.neural-wrap.stale .neural-live-overlay { display: block; }
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
.nhud-val.live { color: var(--green); }
.nhud-val.stale { color: var(--orange); }
.nhud-val.offline { color: var(--red); }
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

/* Thread Matrix */
.thread-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 12px;
  padding: 16px;
  max-width: 1720px;
  margin: 0 auto;
  min-height: calc(100vh - 100px);
}
.thread-stage {
  position: relative;
  min-height: 620px;
  overflow: hidden;
  border: 1px solid rgba(52,212,255,0.16);
  border-radius: 8px;
  background:
    linear-gradient(rgba(52,212,255,0.035) 1px, transparent 1px) 0 0 / 42px 42px,
    linear-gradient(90deg, rgba(52,212,255,0.035) 1px, transparent 1px) 0 0 / 42px 42px,
    radial-gradient(circle at 50% 48%, rgba(52,212,255,0.09), transparent 58%),
    #030408;
}
.thread-stage canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
}
.thread-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  outline: none;
}
.thread-canvas canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
}
.thread-hud {
  position: absolute;
  left: 14px;
  top: 14px;
  z-index: 3;
  display: grid;
  grid-template-columns: repeat(4, minmax(72px, 1fr));
  gap: 6px;
  width: min(520px, calc(100% - 28px));
}
.thread-metric {
  border: 1px solid rgba(52,212,255,0.14);
  border-radius: 7px;
  background: rgba(8,10,16,0.78);
  padding: 7px 8px;
}
.thread-metric span:first-child {
  display: block;
  color: var(--dim);
  font-size: 7px;
  font-weight: 700;
  letter-spacing: 0.8px;
  text-transform: uppercase;
}
.thread-metric span:last-child {
  color: var(--cyan);
  font-size: 13px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.thread-side {
  display: grid;
  grid-template-rows: auto minmax(220px, 1fr);
  gap: 12px;
  min-width: 0;
}
.thread-list {
  display: grid;
  gap: 7px;
  max-height: 420px;
  overflow-y: auto;
  padding-right: 2px;
}
.thread-row {
  display: grid;
  grid-template-columns: 10px minmax(0, 1fr) 46px;
  align-items: center;
  gap: 8px;
  padding: 8px 9px;
  border: 1px solid rgba(30,37,54,0.86);
  border-radius: 7px;
  background: rgba(8,10,16,0.58);
  cursor: pointer;
}
.thread-row:hover { border-color: rgba(52,212,255,0.30); background: rgba(52,212,255,0.045); }
.thread-row.active { border-color: rgba(52,212,255,0.48); background: rgba(52,212,255,0.075); }
.thread-row-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--dim);
}
.thread-row-dot.online { background: var(--green); box-shadow: 0 0 8px rgba(0,232,123,0.75); }
.thread-row-dot.degraded, .thread-row-dot.stale { background: var(--orange); box-shadow: 0 0 8px rgba(255,184,48,0.65); }
.thread-row-dot.offline, .thread-row-dot.error { background: var(--red); }
.thread-row-main { min-width: 0; }
.thread-row-title {
  color: #eef3ff;
  font-size: 10px;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.thread-row-sub {
  color: var(--dim);
  font-size: 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.thread-row-activation {
  color: var(--cyan);
  font-size: 10px;
  font-weight: 700;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.thread-detail {
  border: 1px solid rgba(30,37,54,0.86);
  border-radius: 8px;
  background: rgba(3,4,8,0.50);
  padding: 10px;
  min-height: 220px;
}
.thread-detail-title {
  color: #eef3ff;
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 4px;
}
.thread-detail-desc {
  color: var(--dim);
  font-size: 9px;
  line-height: 1.5;
  margin-bottom: 9px;
}
.thread-detail-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
  margin-bottom: 10px;
}
.thread-detail-stat {
  border: 1px solid rgba(30,37,54,0.8);
  border-radius: 6px;
  padding: 6px 7px;
  background: rgba(12,15,24,0.58);
}
.thread-detail-stat span:first-child {
  display: block;
  color: var(--dim);
  font-size: 7px;
  text-transform: uppercase;
  letter-spacing: 0.7px;
}
.thread-detail-stat span:last-child {
  display: block;
  color: var(--text);
  font-size: 11px;
  font-weight: 700;
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.thread-files {
  max-height: 190px;
  overflow-y: auto;
  display: grid;
  gap: 5px;
}
.thread-file {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
  color: var(--muted);
  font-size: 8px;
  border: 1px solid transparent;
  border-radius: 5px;
  padding: 3px 4px;
  cursor: pointer;
}
.thread-file:hover { color: var(--text); border-color: rgba(52,212,255,0.20); background: rgba(52,212,255,0.04); }
.thread-file.active { color: var(--cyan); border-color: rgba(52,212,255,0.36); background: rgba(52,212,255,0.08); }
.thread-file span:first-child {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.thread-file span:last-child { color: var(--dim); font-variant-numeric: tabular-nums; }
.thread-mini-list {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-bottom: 9px;
}
.thread-chip {
  max-width: 100%;
  color: var(--text);
  border: 1px solid rgba(30,37,54,0.86);
  border-radius: 999px;
  background: rgba(12,15,24,0.64);
  padding: 3px 7px;
  font-size: 8px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.thread-chip.good { color: var(--green); border-color: rgba(0,232,123,0.24); }
.thread-chip.warn { color: var(--orange); border-color: rgba(255,184,48,0.24); }
.thread-chip.hot { color: var(--cyan); border-color: rgba(52,212,255,0.24); }
.thread-section-title {
  color: var(--dim);
  font-size: 7px;
  font-weight: 700;
  letter-spacing: 0.9px;
  text-transform: uppercase;
  margin: 9px 0 5px;
}

@media (max-width: 1000px) {
  .overview-grid, .console-grid, .thread-grid { grid-template-columns: 1fr; }
  .manifold-frame { min-height: 300px; }
  .ops-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .route-list { grid-template-columns: 1fr; }
  .thread-stage { min-height: 520px; }
}
@media (max-width: 560px) {
  .tab-bar { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); padding: 0; }
  .tab-btn { min-width: 0; padding: 10px 6px; letter-spacing: .7px; }
  .stats-bar { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .stat-card { min-width: 0; }
  .ops-grid { grid-template-columns: 1fr; }
  .manifold-stats, .q-process-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .manifold-frame { min-height: 260px; }
  .thread-hud { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .thread-stage { min-height: 440px; }
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

<div class="tab-bar" role="tablist" aria-label="Dashboard views">
  <button id="tab-button-overview" type="button" role="tab" aria-selected="true" aria-controls="tab-overview" class="tab-btn active" onclick="switchTab('overview',this)">Overview</button>
  <button id="tab-button-console" type="button" role="tab" aria-selected="false" aria-controls="tab-console" class="tab-btn" onclick="switchTab('console',this)">Console</button>
  <button id="tab-button-threads" type="button" role="tab" aria-selected="false" aria-controls="tab-threads" class="tab-btn" onclick="switchTab('threads',this)">Thread Matrix</button>
  <button id="tab-button-neural" type="button" role="tab" aria-selected="false" aria-controls="tab-neural" class="tab-btn" onclick="switchTab('neural',this)">Neural Map</button>
</div>

<!-- ══════════ OVERVIEW ══════════ -->
<div class="tab-content active" id="tab-overview" role="tabpanel" aria-labelledby="tab-button-overview">
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
      <div class="signal-card"><div class="signal-label">Live Voice SLO</div><div class="signal-value" id="liveVoiceSlo" aria-live="polite">NO DATA</div><div class="signal-sub" id="liveVoiceLatency">p95 -- · budget --</div></div>
      <div class="signal-card"><div class="signal-label">Codebase</div><div class="signal-value" id="codebaseHealth">--</div><div class="signal-sub" id="codebaseLatency">--</div></div>
    </div>
    <div class="route-list" id="routeList"></div>
  </div>

  <div class="panel" style="grid-row: span 5">
    <div class="panel-hdr"><span class="panel-title">156-Qubit Kingston Matrix</span><span class="panel-badge" id="qTopology">kingston</span></div>
    <div class="manifold-panel">
      <div class="manifold-frame">
        <canvas class="manifold-canvas" id="kingstonCanvas" width="540" height="480" aria-label="156-qubit Kingston dodecahedron architecture">156-qubit Kingston dodecahedron architecture</canvas>
      </div>
      <div class="manifold-stats">
        <div class="q-stat"><div class="q-stat-label">Core</div><div class="q-stat-val" id="qCore">12</div></div>
        <div class="q-stat"><div class="q-stat-label">Reservoir</div><div class="q-stat-val" id="qReservoir">144</div></div>
        <div class="q-stat"><div class="q-stat-label">Local Coupling</div><div class="q-stat-val" id="qCouplings">30</div></div>
        <div class="q-stat"><div class="q-stat-label">H_CR</div><div class="q-stat-val" id="qHcr">144</div></div>
      </div>
      <div class="q-process-grid" id="qProcessGrid"></div>
      <div class="quantum-info">
        <div class="q-bits" id="qBits">|------------&#10217;</div>
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
<div class="tab-content" id="tab-console" role="tabpanel" aria-labelledby="tab-button-console">
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

<!-- ══════════ THREAD MATRIX ══════════ -->
<div class="tab-content" id="tab-threads" role="tabpanel" aria-labelledby="tab-button-threads">
<div class="thread-grid">
  <div class="thread-stage" id="threadStage">
    <div id="threadCanvas" class="thread-canvas" role="application" tabindex="0" aria-label="Interactive 3D codebase thread activation matrix"></div>
    <div class="thread-hud">
      <div class="thread-metric"><span>Threads</span><span id="tmThreads">--</span></div>
      <div class="thread-metric"><span>Active</span><span id="tmActive">--</span></div>
      <div class="thread-metric"><span>Files</span><span id="tmFiles">--</span></div>
      <div class="thread-metric"><span>Lines</span><span id="tmLines">--</span></div>
    </div>
  </div>
  <aside class="thread-side">
    <div class="panel">
      <div class="panel-hdr"><span class="panel-title">Activation Threads</span><span class="panel-badge" id="tmBadge">waiting</span></div>
      <div class="thread-list" id="threadList"><div class="empty-state">Waiting for codebase telemetry...</div></div>
    </div>
    <div class="panel">
      <div class="panel-hdr"><span class="panel-title">Selected Thread</span><span class="panel-badge" id="threadSelectedBadge">none</span></div>
      <div class="thread-detail" id="threadDetail"><div class="empty-state">Select a node or row to inspect its source files.</div></div>
    </div>
  </aside>
</div>
</div>

<!-- ══════════ NEURAL MAP ══════════ -->
<div class="tab-content" id="tab-neural" role="tabpanel" aria-labelledby="tab-button-neural">
<div class="neural-wrap" id="neuralWrap">
  <canvas id="neuralBg"></canvas>
  <canvas id="neuralMain"></canvas>
  <canvas id="neuralFx"></canvas>
  <div class="neural-live-overlay" id="neuralLiveOverlay">Waiting for live telemetry</div>
  <!-- HUD Panel -->
  <div class="neural-hud" id="neuralHud">
    <div class="nhud-title">NEURAL TOPOLOGY</div>
    <div class="nhud-row"><span class="nhud-label">Link</span><span class="nhud-val stale" id="nLive">BOOT</span></div>
    <div class="nhud-row"><span class="nhud-label">Last Data</span><span class="nhud-val" id="nAge">--</span></div>
    <div class="nhud-sep"></div>
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
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.remove('active');
    b.setAttribute('aria-selected', 'false');
  });
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  if (btn) {
    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');
  }
  if (name === 'threads') startThreadMatrix();
  if (name === 'neural') { neuralResize(); if (!_neuralRunning) startNeural(); }
}
document.querySelector('.tab-bar')?.addEventListener('keydown', event => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  const tabs = [...document.querySelectorAll('.tab-btn')];
  const current = Math.max(0, tabs.indexOf(document.activeElement));
  const next = event.key === 'Home' ? 0
    : event.key === 'End' ? tabs.length - 1
    : (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
  event.preventDefault();
  tabs[next].focus();
  switchTab(tabs[next].id.replace('tab-button-', ''), tabs[next]);
});

// ── State ────────────────────────────────────
let _quantumWeights = [0.5, 0.5, 0.5, 0.5, 0.5];
let _quantumDimWeights = {logic:0.5, emotion:0.5, memory:0.5, creativity:0.5, vigilance:0.5};
let _quantumArchitecture = null;
let _quantumLast = {};
let _matrixRAF = null;
let _neuralRunning = false;
let _threadMatrixRunning = false;
let _threadMatrix = {
  threads: [],
  edges: [],
  stats: {},
  nodes: {},
  hover: null,
  selected: null,
  selectedFile: null,
  raf: 0,
};
let _liveState = {
  sse: false,
  browserOnline: navigator.onLine !== false,
  lastFetchAt: 0,
  lastPollAt: 0,
  lastNexusAt: 0,
  lastKind: 'boot',
  lastError: '',
  staleAfterMs: 12000,
};

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

function liveLastAt() {
  return Math.max(_liveState.lastFetchAt, _liveState.lastPollAt, _liveState.lastNexusAt);
}

function liveAgeMs() {
  const last = liveLastAt();
  return last ? Date.now() - last : Infinity;
}

function isNeuralFresh() {
  return _liveState.browserOnline && liveAgeMs() <= _liveState.staleAfterMs;
}

function liveAgeLabel() {
  const age = liveAgeMs();
  if (!Number.isFinite(age)) return '--';
  if (age < 1000) return 'now';
  if (age < 60000) return `${Math.floor(age / 1000)}s`;
  return `${Math.floor(age / 60000)}m`;
}

function updateLiveHud() {
  const fresh = isNeuralFresh();
  const wrap = document.getElementById('neuralWrap');
  if (wrap) wrap.classList.toggle('stale', !fresh);
  const live = document.getElementById('nLive');
  if (live) {
    live.className = 'nhud-val ' + (fresh ? 'live' : (_liveState.browserOnline ? 'stale' : 'offline'));
    live.textContent = fresh ? 'LIVE' : (_liveState.browserOnline ? 'STALE' : 'OFFLINE');
  }
  setText('nAge', liveAgeLabel());
  const overlay = document.getElementById('neuralLiveOverlay');
  if (overlay) {
    const age = liveAgeLabel();
    overlay.textContent = _liveState.browserOnline
      ? (age === '--' ? 'Waiting for live telemetry' : `Waiting for live telemetry - last data ${age} ago`)
      : 'Browser offline - neural map frozen';
  }
}

function markLive(kind) {
  const now = Date.now();
  _liveState.lastKind = kind;
  _liveState.lastError = '';
  _liveState.browserOnline = navigator.onLine !== false;
  if (kind === 'poll') _liveState.lastPollAt = now;
  else if (kind === 'nexus') _liveState.lastNexusAt = now;
  else _liveState.lastFetchAt = now;
  updateLiveHud();
}

function markOffline(reason) {
  _liveState.lastError = reason || 'offline';
  _liveState.browserOnline = reason === 'browser offline' ? false : navigator.onLine !== false;
  updateLiveHud();
}

// ── Data fetching ────────────────────────────
async function fetchState() {
  try {
    const r = await fetch('/api/state', {cache: 'no-store'});
    if (!r.ok) { markOffline(`state ${r.status}`); return; }
    const d = await r.json();
    markLive('fetch');
    updateLobes(d.lobes || []);
    updateQuantum(d.quantum || {}, d.quantum_architecture || null);
    updateFeed(d.nexus_feed || []);
    updateTranscript(d.transcript || '');
    updatePeople(d.people || []);
    updateDreams(d.dreams || []);
    updateBrain(d.brain || {}, d.voice || {}, d.codebase || {});
    updateThreads(d.codebase_threads || {});
    updateMemoryEvents(d.memory_events || []);
    document.getElementById('sOnline').textContent = `${d.online}/${d.total}`;
    document.getElementById('sUptime').textContent = fmtUp(d.uptime||0);
    document.getElementById('sMsgs').textContent = d.nexus_stats?.msg_count || 0;
    document.getElementById('sBusLobes').textContent = d.nexus_stats?.current_lobe_ids?.length || 0;
    document.getElementById('sTopics').textContent = d.nexus_stats?.topics_seen?.length || 0;
    document.getElementById('lobeBadge').textContent = `${d.online}/${d.total}`;
    if (d.ngrok_url) {
      const el = document.getElementById('publicUrl');
      el.href = d.ngrok_url; el.textContent = d.ngrok_url.replace('https://',''); el.style.display = 'inline-block';
    }
  } catch(e) { markOffline(e.message); console.warn('fetch:', e.message); }
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
  const slo = brain.voice_slo && typeof brain.voice_slo === 'object' ? brain.voice_slo : {};
  const prewarm = brain.voice_prewarm && typeof brain.voice_prewarm === 'object' ? brain.voice_prewarm : {};
  setText('sBrain', statusLabel(brain.status));
  setText('brainBadge', `${statusLabel(brain.status)} · ${fmtMs(brain.latency_ms)}`);
  setText('defaultModel', brain.default_model || 'unknown');
  setText('brainLatency', `${routes.length} routes · state ${fmtMs(brain.state_latency_ms)}`);
  setText('headlessModel', brain.headless_model || brain.headless_thought_model || 'unknown');
  setText('headlessMode', `${brain.thoughts || 0} thoughts · ${brain.dreams || 0} dreams`);
  setText('voiceHealth', statusLabel(voice.status));
  setText('voiceLatency', `${voice.service || 'trained voice'} · ${fmtMs(voice.latency_ms)}`);
  setText('liveVoiceSlo', String(slo.status || 'no data').toUpperCase());
  const sloElement = document.getElementById('liveVoiceSlo');
  if (sloElement) sloElement.className = `signal-value ${['nominal','watch','breached'].includes(slo.status) ? slo.status : ''}`;
  const success = Number.isFinite(Number(slo.success_rate)) ? `${Math.round(Number(slo.success_rate) * 100)}% good` : '-- good';
  const budget = Number.isFinite(Number(slo.error_budget_remaining_pct)) ? `${Math.round(Number(slo.error_budget_remaining_pct))}% budget` : '-- budget';
  const semanticP95 = fmtMs(slo.semantic_p95_ms);
  const warm = `${prewarm.status || 'pending'}${Number.isFinite(Number(prewarm.latency_ms)) ? ` ${fmtMs(prewarm.latency_ms)}` : ''}`;
  setText('liveVoiceLatency', `p95 ${semanticP95} · ${success} · ${budget} · warm ${warm}`);
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

function updateQuantum(q, architecture) {
  if (architecture && !architecture.error) _quantumArchitecture = architecture;
  if (!q || !q.weights) return;
  _quantumLast = q;
  const dims = ['logic','emotion','memory','creativity','vigilance'];
  _quantumWeights = dims.map(d => q.weights[d] || 0.5);
  _quantumDimWeights = Object.fromEntries(dims.map((d, i) => [d, _quantumWeights[i]]));
  const arch = _quantumArchitecture || {};
  const summary = arch.summary || {};
  const stats = arch.graph_stats || {};
  setText('qTopology', summary.subtitle || 'kingston');
  setText('qCore', summary.core_qubits ?? 12);
  setText('qReservoir', summary.reservoir_qubits ?? 144);
  setText('qCouplings', stats.core_local_couplings ?? summary.core_local_couplings ?? 30);
  setText('qHcr', stats.core_reservoir_couplings ?? 144);
  updateQuantumProcesses(arch.modules || []);
  document.getElementById('qBits').textContent = q.bitstring ? `|${q.bitstring}⟩` : '|------------⟩';
  document.getElementById('qDom').textContent = q.dominant || '--';
  document.getElementById('sPathway').textContent = q.dominant || '--';
  let meta = '';
  if (q.secondary) meta += `Secondary: ${q.secondary}\n`;
  if (q.timestamp) meta += `Measured: ${q.timestamp}\n`;
  meta += `Topology: ${stats.connectivity || summary.connectivity || 'sparse small-world'} · ${stats.dynamics || summary.dynamics || 'Lindblad open system'}`;
  document.getElementById('qMeta').textContent = meta;
  document.getElementById('qRaw').textContent = q.raw ? q.raw.substring(q.raw.indexOf(']')+2) : '';
  startKingstonMatrix();
}

function updateQuantumProcesses(modules) {
  const grid = document.getElementById('qProcessGrid');
  if (!grid) return;
  const fallback = [
    {label:'State Encoding', detail:'Second-order ZZ feature map', color:'#3daeff'},
    {label:'Open System Dynamics', detail:'Lindblad drive and damping', color:'#78d88b'},
    {label:'Entropy Routing', detail:'Von Neumann entropy MoE routing', color:'#ffcf4f'},
    {label:'Measurement / Readout', detail:'Observable projection and response', color:'#eef1f6'},
  ];
  const rows = (modules && modules.length ? modules : fallback).slice(0, 4);
  grid.innerHTML = rows.map(m => `
    <div class="q-process-card" style="border-left-color:${esc(m.color || '#34d4ff')}">
      <div class="q-process-title">${esc(m.label || m.key || 'module')}</div>
      <div class="q-process-detail">${esc(m.detail || '')}</div>
    </div>`).join('');
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

// ── 156-Qubit Kingston Matrix ──────────────────────────
const CORE_LAYOUT_3D = [
  [-1,1.618,0],[1,1.618,0],[-1,-1.618,0],[1,-1.618,0],
  [0,-1,1.618],[0,1,1.618],[0,-1,-1.618],[0,1,-1.618],
  [1.618,0,-1],[1.618,0,1],[-1.618,0,-1],[-1.618,0,1],
].map(p => {
  const n = Math.hypot(p[0], p[1], p[2]) || 1;
  return [p[0]/n, p[1]/n, p[2]/n];
});
const DIM_COLORS = {
  logic: '#34d4ff',
  emotion: '#ec4899',
  memory: '#ffb830',
  creativity: '#9b6dff',
  vigilance: '#ff4444',
  synthesis: '#00e87b',
  entropy: '#94a3b8',
};
const CORE_FALLBACK = [
  ['Logic','logic','#9b5cff'], ['Emotion','emotion','#7c5cff'], ['Intuition','creativity','#4f72ff'],
  ['Memory','memory','#3daeff'], ['Sovereignty','vigilance','#56d7ff'], ['Attention','emotion','#78d88b'],
  ['Reflection','memory','#d8d65f'], ['Language','logic','#ffcf4f'], ['Planning','creativity','#ffa24a'],
  ['Novelty','creativity','#ffd76a'], ['Stability','vigilance','#ff914d'], ['Meta-Reasoning','logic','#d83cff'],
].map((v, i) => ({index:i, role:v[0], dimension:v[1], color:v[2]}));

function startKingstonMatrix() {
  if (!_matrixRAF) _matrixRAF = requestAnimationFrame(kingstonFrame);
}

function kingstonFrame(ts) {
  drawKingstonMatrix(ts || performance.now());
  _matrixRAF = requestAnimationFrame(kingstonFrame);
}

function fitCanvas(c) {
  const rect = c.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.floor(rect.width * ratio));
  const h = Math.max(1, Math.floor(rect.height * ratio));
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
  }
  const ctx = c.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {ctx, W: rect.width, H: rect.height};
}

function coreActivity(core) {
  const dim = core.dimension || 'logic';
  const base = Number(_quantumDimWeights[dim] ?? 0.45);
  const bits = (_quantumLast.bitstring || '').padStart(12, '0');
  const bit = bits.length >= 12 ? bits[bits.length - 1 - core.index] : '0';
  return Math.max(0.18, Math.min(1, base + (bit === '1' ? 0.22 : 0)));
}

function corePoint(index, W, H, time) {
  const p = CORE_LAYOUT_3D[index % CORE_LAYOUT_3D.length];
  const ry = time * 0.12;
  const rx = -0.34;
  const x1 = p[0] * Math.cos(ry) + p[2] * Math.sin(ry);
  const z1 = -p[0] * Math.sin(ry) + p[2] * Math.cos(ry);
  const y1 = p[1] * Math.cos(rx) - z1 * Math.sin(rx);
  const z2 = p[1] * Math.sin(rx) + z1 * Math.cos(rx);
  const scale = Math.min(W, H) * 0.23;
  return {
    x: W * 0.50 + x1 * scale,
    y: H * 0.43 + y1 * scale,
    z: z2,
  };
}

function reservoirPoint(index, W, H, time) {
  const n = 144;
  const a = index * 2.399963229728653;
  const r = Math.sqrt((index + 0.5) / n);
  const shell = Math.min(W, H) * (0.30 + r * 0.20);
  const breathe = 1 + 0.015 * Math.sin(time * 1.1 + index * 0.17);
  return {
    x: W * 0.50 + Math.cos(a + time * 0.025) * shell * 1.10 * breathe,
    y: H * 0.43 + Math.sin(a + time * 0.018) * shell * 0.82 * breathe,
  };
}

function drawKingstonMatrix(ts) {
  const c = document.getElementById('kingstonCanvas');
  if (!c) return;
  const {ctx, W, H} = fitCanvas(c);
  const time = ts * 0.001;
  const arch = _quantumArchitecture || {};
  const core = (arch.core_qubits && arch.core_qubits.length ? arch.core_qubits : CORE_FALLBACK).slice(0, 12);
  const edges = arch.topology_edges || {};
  const localEdges = edges.dodecahedron || [[0,1],[0,5],[0,7],[0,10],[0,11],[1,5],[1,7],[1,8],[1,9],[2,3],[2,4],[2,6],[2,10],[2,11],[3,4],[3,6],[3,8],[3,9],[4,5],[4,9],[4,11],[5,9],[5,11],[6,7],[6,8],[6,10],[7,8],[7,10],[8,9],[10,11]];
  const stateEdges = edges.state_encoding || [];
  const entropyEdges = edges.entropy_routing || [];
  const readoutEdges = edges.measurement_readout || [];
  const corePts = core.map(q => ({...q, ...corePoint(q.index, W, H, time)}));
  const reservoirPts = Array.from({length:144}, (_, i) => reservoirPoint(i, W, H, time));

  ctx.clearRect(0, 0, W, H);
  const bg = ctx.createRadialGradient(W * 0.50, H * 0.43, 0, W * 0.50, H * 0.43, Math.min(W, H) * 0.58);
  bg.addColorStop(0, 'rgba(52,212,255,0.10)');
  bg.addColorStop(0.55, 'rgba(155,109,255,0.045)');
  bg.addColorStop(1, 'rgba(3,4,8,0)');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (let i = 0; i < reservoirPts.length; i++) {
    const p = reservoirPts[i];
    const q = core[i % core.length];
    const activity = coreActivity(q);
    const pulse = Math.max(0, Math.sin(time * 2.2 + i * 0.29)) * activity;
    const color = i % 3 === 0 ? [52,212,255] : i % 3 === 1 ? [124,92,252] : [0,232,123];
    if (i % 2 === 0) {
      const next = reservoirPts[(i + 1) % reservoirPts.length];
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(next.x, next.y);
      ctx.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},0.035)`;
      ctx.lineWidth = 0.45;
      ctx.stroke();
    }
    if (i % 12 === 0 || pulse > 0.55) {
      const cp = corePts[i % corePts.length];
      ctx.beginPath();
      ctx.moveTo(cp.x, cp.y);
      ctx.lineTo(p.x, p.y);
      ctx.strokeStyle = `rgba(52,212,255,${0.025 + pulse * 0.12})`;
      ctx.lineWidth = 0.5 + pulse * 0.8;
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(p.x, p.y, 1.1 + pulse * 1.5, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${0.28 + pulse * 0.58})`;
    ctx.fill();
  }

  localEdges.forEach(([a, b]) => {
    const pa = corePts[a], pb = corePts[b];
    if (!pa || !pb) return;
    const act = (coreActivity(pa) + coreActivity(pb)) / 2;
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.strokeStyle = `rgba(238,241,246,${0.15 + act * 0.20})`;
    ctx.lineWidth = 0.65 + act * 0.9;
    ctx.stroke();
  });

  stateEdges.forEach(([a, b]) => {
    const pa = corePts[a], pb = corePts[b];
    if (!pa || !pb) return;
    const act = (coreActivity(pa) + coreActivity(pb)) / 2;
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.strokeStyle = `rgba(61,174,255,${0.16 + act * 0.22})`;
    ctx.lineWidth = 1 + act * 0.8;
    ctx.stroke();
  });

  [...entropyEdges, ...readoutEdges].forEach(([a, b], i) => {
    const pa = corePts[a], pb = corePts[b];
    if (!pa || !pb) return;
    const act = (coreActivity(pa) + coreActivity(pb)) / 2;
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.setLineDash(i % 2 ? [4, 5] : [2, 5]);
    ctx.strokeStyle = `rgba(255,207,79,${0.10 + act * 0.18})`;
    ctx.lineWidth = 0.7 + act * 0.7;
    ctx.stroke();
    ctx.setLineDash([]);
  });

  corePts
    .slice()
    .sort((a, b) => a.z - b.z)
    .forEach(q => {
      const activity = coreActivity(q);
      const color = q.color || DIM_COLORS[q.dimension] || '#34d4ff';
      const radius = 7 + activity * 7;
      const halo = ctx.createRadialGradient(q.x, q.y, 0, q.x, q.y, radius * 3.2);
      halo.addColorStop(0, hexToRgba(color, 0.38 * activity));
      halo.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(q.x, q.y, radius * 3.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(q.x, q.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = hexToRgba(color, 0.88);
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = `rgba(238,241,246,${0.36 + activity * 0.34})`;
      ctx.stroke();
      ctx.fillStyle = '#030408';
      ctx.font = '700 9px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`Q${q.index}`, q.x, q.y);
      const labelY = q.y + radius + 12;
      ctx.fillStyle = `rgba(238,243,255,${0.56 + activity * 0.28})`;
      ctx.font = '600 8px JetBrains Mono';
      ctx.fillText(String(q.role || '').slice(0, 13), q.x, labelY);
    });
  ctx.restore();

  ctx.fillStyle = 'rgba(238,243,255,0.86)';
  ctx.font = '700 11px JetBrains Mono';
  ctx.textAlign = 'left';
  ctx.fillText('WEAVER V3 - 156-QUBIT KINGSTON MANIFOLD', 14, 22);
  ctx.fillStyle = 'rgba(52,212,255,0.72)';
  ctx.font = '600 9px JetBrains Mono';
  ctx.fillText('12-core dodecahedron dual embedded in Q12-Q155 reservoir', 14, 38);
  ctx.textAlign = 'right';
  ctx.fillStyle = 'rgba(184,192,212,0.60)';
  ctx.fillText('local coupling | long-range entanglement | H_CR', W - 14, H - 18);
}

function hexToRgba(hex, alpha) {
  const m = String(hex || '').replace('#', '');
  if (m.length !== 6) return `rgba(52,212,255,${alpha})`;
  const r = parseInt(m.slice(0,2), 16);
  const g = parseInt(m.slice(2,4), 16);
  const b = parseInt(m.slice(4,6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Codebase Thread Matrix ──────────────────────────
const THREAD_COLORS = {
  core: '#34d4ff',
  cortex: '#9b6dff',
  interface: '#00e87b',
  support: '#ffb830',
};
let _threadScenePromise = null;
let _thread3d = {
  THREE: null,
  host: null,
  scene: null,
  camera: null,
  renderer: null,
  root: null,
  ringGroup: null,
  edgeGroup: null,
  nodeGroup: null,
  fileGroup: null,
  labelGroup: null,
  raycaster: null,
  pointer: null,
  ready: false,
  interactive: [],
  nodeMeshes: new Map(),
  fileMeshes: new Map(),
  dragging: false,
  dragMoved: false,
  dragLast: null,
  rotX: -0.54,
  rotY: 0.35,
  zoom: 7.8,
};
window.__weaverThread3d = _thread3d;

function fmtCount(value) {
  const n = Number(value || 0);
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'm';
  if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k';
  return String(Math.round(n));
}

function fmtAge(seconds) {
  const n = Number(seconds);
  if (!Number.isFinite(n) || n < 0) return 'unknown';
  if (n < 60) return `${Math.round(n)}s`;
  if (n < 3600) return `${Math.round(n / 60)}m`;
  if (n < 86400) return `${(n / 3600).toFixed(n < 7200 ? 1 : 0)}h`;
  return `${Math.round(n / 86400)}d`;
}

function truncateText(value, limit = 58) {
  const text = String(value || '');
  return text.length > limit ? text.slice(0, Math.max(0, limit - 1)) + '...' : text;
}

function threadColorHex(threadOrRing) {
  const ring = typeof threadOrRing === 'string' ? threadOrRing : threadOrRing?.ring;
  return THREAD_COLORS[ring || 'core'] || THREAD_COLORS.core;
}

function threadColorNumber(threadOrRing) {
  return parseInt(threadColorHex(threadOrRing).slice(1), 16);
}

function updateThreads(data) {
  const threads = Array.isArray(data.threads) ? data.threads : [];
  _threadMatrix.threads = threads;
  _threadMatrix.edges = Array.isArray(data.edges) ? data.edges : [];
  _threadMatrix.stats = data.stats || {};
  setText('tmThreads', fmtCount(_threadMatrix.stats.threads || threads.length));
  setText('tmActive', fmtCount(_threadMatrix.stats.active || threads.filter(t => Number(t.activation) >= 0.55).length));
  setText('tmFiles', fmtCount(_threadMatrix.stats.files || threads.reduce((sum, t) => sum + Number(t.files || 0), 0)));
  setText('tmLines', fmtCount(_threadMatrix.stats.lines || threads.reduce((sum, t) => sum + Number(t.lines || 0), 0)));
  setText('tmBadge', data.generated_at ? data.generated_at.split('T')[1]?.slice(0, 8) || 'live' : 'live');
  if (!threadById(_threadMatrix.selected) && threads.length) {
    _threadMatrix.selected = threads.slice().sort((a, b) => Number(b.activation || 0) - Number(a.activation || 0))[0].id;
  }
  const selectedThread = threadById(_threadMatrix.selected);
  if (!selectedThread || !threadFileByPath(selectedThread, _threadMatrix.selectedFile)) {
    _threadMatrix.selectedFile = null;
  }
  renderThreadList();
  updateThreadDetail();
  if (_thread3d.ready) syncThreadScene();
}

function threadById(id) {
  return _threadMatrix.threads.find(t => t.id === id) || null;
}

function threadFileByPath(thread, path) {
  if (!thread || !path) return null;
  return (Array.isArray(thread.hot_files) ? thread.hot_files : []).find(f => f.path === path) || null;
}

function renderThreadList() {
  const list = document.getElementById('threadList');
  if (!list) return;
  const rows = _threadMatrix.threads.slice().sort((a, b) => Number(b.activation || 0) - Number(a.activation || 0));
  if (!rows.length) {
    list.innerHTML = '<div class="empty-state">Waiting for codebase telemetry...</div>';
    return;
  }
  list.innerHTML = rows.map(t => `
    <div class="thread-row ${_threadMatrix.selected === t.id ? 'active' : ''}" data-thread="${esc(t.id)}">
      <span class="thread-row-dot ${esc(t.status || 'unknown')}"></span>
      <span class="thread-row-main">
        <span class="thread-row-title">${esc(t.label || t.id)}</span>
        <span class="thread-row-sub">${esc(t.ring || 'thread')} &middot; ${fmtCount(t.files)} files &middot; ${fmtCount(t.lines)} lines &middot; ${fmtCount((t.endpoints || []).length)} routes</span>
      </span>
      <span class="thread-row-activation">${Math.round(Number(t.activation || 0) * 100)}%</span>
    </div>
  `).join('');
  list.querySelectorAll('[data-thread]').forEach(row => {
    row.addEventListener('click', () => selectThread(row.dataset.thread, null));
  });
}

function selectThread(id, filePath = null) {
  if (!threadById(id)) return;
  _threadMatrix.selected = id;
  _threadMatrix.selectedFile = filePath || null;
  renderThreadList();
  updateThreadDetail();
  updateThreadHighlights();
}

function updateThreadDetail() {
  const detail = document.getElementById('threadDetail');
  const badge = document.getElementById('threadSelectedBadge');
  if (!detail) return;
  const thread = threadById(_threadMatrix.selected);
  if (!thread) {
    if (badge) badge.textContent = 'none';
    detail.innerHTML = '<div class="empty-state">Select a node or row to inspect its source files.</div>';
    return;
  }
  if (badge) badge.textContent = `${Math.round(Number(thread.activation || 0) * 100)}%`;
  const files = Array.isArray(thread.hot_files) ? thread.hot_files : [];
  const selectedFile = threadFileByPath(thread, _threadMatrix.selectedFile) || null;
  const lobeDetails = Array.isArray(thread.lobe_details) ? thread.lobe_details : [];
  const symbols = Array.isArray(thread.symbols) ? thread.symbols : [];
  const endpoints = Array.isArray(thread.endpoints) ? thread.endpoints : [];
  const imports = Array.isArray(thread.imports) ? thread.imports : [];
  const selectors = Array.isArray(thread.selectors) ? thread.selectors : [];
  const events = Array.isArray(thread.recent_events) ? thread.recent_events : [];
  detail.innerHTML = `
    <div class="thread-detail-title">${esc(thread.label || thread.id)}</div>
    <div class="thread-detail-desc">${esc(thread.desc || '')}</div>
    <div class="thread-detail-grid">
      <div class="thread-detail-stat"><span>Status</span><span>${esc(thread.status || 'unknown')}</span></div>
      <div class="thread-detail-stat"><span>Ring</span><span>${esc(thread.ring || 'thread')}</span></div>
      <div class="thread-detail-stat"><span>Files</span><span>${fmtCount(thread.files)}</span></div>
      <div class="thread-detail-stat"><span>Lines</span><span>${fmtCount(thread.lines)}</span></div>
      <div class="thread-detail-stat"><span>Bus Hits</span><span>${fmtCount(thread.topic_hits)}</span></div>
      <div class="thread-detail-stat"><span>Newest</span><span>${esc(thread.updated || fmtAge(thread.newest_age_s))}</span></div>
    </div>
    ${lobeDetails.length ? `
      <div class="thread-section-title">Live Lobes</div>
      <div class="thread-mini-list">
        ${lobeDetails.map(l => `<span class="thread-chip ${l.status === 'online' ? 'good' : l.status === 'offline' ? 'warn' : ''}" title="${esc(l.detail || '')}">${esc(l.name)} ${esc(l.status || 'unknown')}${l.latency_ms == null ? '' : ` ${esc(l.latency_ms)}ms`}</span>`).join('')}
      </div>
    ` : ''}
    ${selectedFile ? `
      <div class="thread-section-title">Selected File</div>
      <div class="thread-detail-grid">
        <div class="thread-detail-stat"><span>Path</span><span title="${esc(selectedFile.path)}">${esc(selectedFile.path)}</span></div>
        <div class="thread-detail-stat"><span>Updated</span><span>${esc(selectedFile.updated || 'unknown')}</span></div>
        <div class="thread-detail-stat"><span>Lines</span><span>${fmtCount(selectedFile.lines)}</span></div>
        <div class="thread-detail-stat"><span>Size</span><span>${selectedFile.size_kb == null ? `${fmtCount(selectedFile.size)}b` : `${esc(selectedFile.size_kb)} KB`}</span></div>
      </div>
      ${(selectedFile.symbols || []).length ? `<div class="thread-section-title">File Symbols</div><div class="thread-mini-list">${selectedFile.symbols.slice(0, 10).map(s => `<span class="thread-chip hot">${esc(typeof s === 'string' ? s : `${s.kind || 'symbol'} ${s.name || ''}${s.line ? `:${s.line}` : ''}`)}</span>`).join('')}</div>` : ''}
      ${(selectedFile.endpoints || []).length ? `<div class="thread-section-title">File Routes</div><div class="thread-mini-list">${selectedFile.endpoints.slice(0, 8).map(e => `<span class="thread-chip good">${esc(typeof e === 'string' ? e : `${e.method || 'route'} ${e.path || e.name || ''}`)}</span>`).join('')}</div>` : ''}
      ${(selectedFile.imports || []).length || (selectedFile.selectors || []).length ? `<div class="thread-section-title">File Dependencies</div><div class="thread-mini-list">${[...(selectedFile.imports || []), ...(selectedFile.selectors || [])].slice(0, 12).map(v => `<span class="thread-chip">${esc(v)}</span>`).join('')}</div>` : ''}
    ` : `
      <div class="thread-section-title">Newest File</div>
      <div class="thread-mini-list"><span class="thread-chip hot" title="${esc(thread.newest_file || '')}">${esc(thread.newest_file || 'unknown')}</span></div>
    `}
    ${endpoints.length ? `<div class="thread-section-title">Thread Routes</div><div class="thread-mini-list">${endpoints.slice(0, 10).map(e => `<span class="thread-chip good">${esc(typeof e === 'string' ? e : `${e.method || 'route'} ${e.path || e.name || ''}`)}</span>`).join('')}</div>` : ''}
    ${symbols.length ? `<div class="thread-section-title">Top Symbols</div><div class="thread-mini-list">${symbols.slice(0, 12).map(s => `<span class="thread-chip hot">${esc(typeof s === 'string' ? s : `${s.kind || 'symbol'} ${s.name || ''}`)}</span>`).join('')}</div>` : ''}
    ${(imports.length || selectors.length) ? `<div class="thread-section-title">Imports / Selectors</div><div class="thread-mini-list">${[...imports, ...selectors].slice(0, 14).map(v => `<span class="thread-chip">${esc(v)}</span>`).join('')}</div>` : ''}
    ${events.length ? `<div class="thread-section-title">Recent Topic Energy</div><div class="thread-mini-list">${events.map(e => `<span class="thread-chip warn" title="${esc(e.payload || '')}">${esc(e.from || 'bus')}:${esc(e.topic || '')}</span>`).join('')}</div>` : ''}
    <div class="thread-section-title">Clickable Source Dots</div>
    <div class="thread-files">
      ${files.length ? files.map(f => `<div class="thread-file ${_threadMatrix.selectedFile === f.path ? 'active' : ''}" data-thread="${esc(thread.id)}" data-path="${esc(f.path)}"><span title="${esc(f.path)}">${esc(f.path)}</span><span>${fmtCount(f.lines)}l</span></div>`).join('') : '<div class="empty-state">No files mapped yet</div>'}
    </div>
  `;
  detail.querySelectorAll('.thread-file[data-thread][data-path]').forEach(row => {
    row.addEventListener('click', () => selectThread(row.dataset.thread, row.dataset.path));
  });
}

function startThreadMatrix() {
  _threadMatrixRunning = true;
  if (!_threadScenePromise) _threadScenePromise = ensureThreadScene();
  _threadScenePromise.then(ok => {
    if (!ok) return;
    threadResize();
    syncThreadScene();
    if (!_threadMatrix.raf) _threadMatrix.raf = requestAnimationFrame(threadFrame);
  });
}

async function ensureThreadScene() {
  const host = document.getElementById('threadCanvas');
  if (!host) return false;
  try {
    if (!_thread3d.THREE) _thread3d.THREE = await import('/vendor/three.module.js');
  } catch (err) {
    host.innerHTML = '<div class="empty-state" style="padding:18px">Three.js module unavailable. Check /vendor/three.module.js.</div>';
    return false;
  }
  if (_thread3d.ready && _thread3d.host === host) return true;

  const THREE = _thread3d.THREE;
  host.innerHTML = '';
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true, preserveDrawingBuffer: true, powerPreference: 'high-performance'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x030408, 0);
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x030408, 0.045);
  const camera = new THREE.PerspectiveCamera(46, 1, 0.1, 80);
  camera.position.set(0, 0, _thread3d.zoom);

  const root = new THREE.Group();
  const ringGroup = new THREE.Group();
  const edgeGroup = new THREE.Group();
  const nodeGroup = new THREE.Group();
  const fileGroup = new THREE.Group();
  const labelGroup = new THREE.Group();
  root.add(ringGroup, edgeGroup, nodeGroup, fileGroup, labelGroup);
  scene.add(root);
  scene.add(new THREE.AmbientLight(0x9fb6ff, 0.42));
  const key = new THREE.DirectionalLight(0xffffff, 0.95);
  key.position.set(3.2, 4.6, 6.5);
  scene.add(key);
  const rim = new THREE.PointLight(0x34d4ff, 1.2, 14, 1.6);
  rim.position.set(-3.5, -2.4, 3.0);
  scene.add(rim);
  const violet = new THREE.PointLight(0x9b6dff, 0.85, 14, 2.0);
  violet.position.set(3.0, 2.8, -2.0);
  scene.add(violet);

  Object.assign(_thread3d, {
    host, scene, camera, renderer, root, ringGroup, edgeGroup, nodeGroup, fileGroup, labelGroup,
    raycaster: new THREE.Raycaster(),
    pointer: new THREE.Vector2(),
    ready: true,
  });
  bindThreadSceneEvents();
  return true;
}

function bindThreadSceneEvents() {
  const canvas = _thread3d.renderer?.domElement;
  if (!canvas || canvas.dataset.bound === '1') return;
  canvas.dataset.bound = '1';
  canvas.addEventListener('pointerdown', event => {
    _thread3d.dragging = true;
    _thread3d.dragMoved = false;
    _thread3d.dragLast = {x: event.clientX, y: event.clientY};
    canvas.setPointerCapture?.(event.pointerId);
    canvas.style.cursor = 'grabbing';
  });
  canvas.addEventListener('pointermove', event => {
    if (_thread3d.dragging && _thread3d.dragLast) {
      const dx = event.clientX - _thread3d.dragLast.x;
      const dy = event.clientY - _thread3d.dragLast.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) _thread3d.dragMoved = true;
      _thread3d.rotY += dx * 0.006;
      _thread3d.rotX = Math.max(-1.18, Math.min(0.72, _thread3d.rotX + dy * 0.005));
      _thread3d.dragLast = {x: event.clientX, y: event.clientY};
      return;
    }
    const hit = threadPick(event);
    const nextHover = hit?.userData?.threadId || null;
    if (_threadMatrix.hover !== nextHover) {
      _threadMatrix.hover = nextHover;
      updateThreadHighlights();
    }
    canvas.style.cursor = hit ? 'pointer' : 'grab';
  });
  canvas.addEventListener('pointerup', event => {
    _thread3d.dragging = false;
    _thread3d.dragLast = null;
    canvas.releasePointerCapture?.(event.pointerId);
    canvas.style.cursor = 'grab';
  });
  canvas.addEventListener('mouseleave', () => {
    _thread3d.dragging = false;
    _thread3d.dragLast = null;
    _threadMatrix.hover = null;
    updateThreadHighlights();
  });
  canvas.addEventListener('click', event => {
    if (_thread3d.dragMoved) {
      _thread3d.dragMoved = false;
      return;
    }
    const hit = threadPick(event);
    if (!hit) return;
    if (hit.userData.kind === 'file') selectThread(hit.userData.threadId, hit.userData.path);
    else selectThread(hit.userData.threadId, null);
  });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    _thread3d.zoom = Math.max(4.4, Math.min(12.5, _thread3d.zoom + event.deltaY * 0.006));
  }, {passive: false});
}

function threadPick(event) {
  if (!_thread3d.ready || !_thread3d.camera || !_thread3d.raycaster) return null;
  const rect = _thread3d.renderer.domElement.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  _thread3d.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  _thread3d.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  _thread3d.raycaster.setFromCamera(_thread3d.pointer, _thread3d.camera);
  const hits = _thread3d.raycaster.intersectObjects(_thread3d.interactive, false);
  const fileHit = hits.find(hit => hit.object?.userData?.kind === 'file');
  return hits.length ? (fileHit || hits[0]).object : null;
}

function threadResize() {
  if (!_thread3d.ready || !_thread3d.host || !_thread3d.renderer || !_thread3d.camera) return;
  const rect = _thread3d.host.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  _thread3d.renderer.setSize(width, height, false);
  _thread3d.camera.aspect = width / height;
  _thread3d.camera.updateProjectionMatrix();
}

function layoutThreadPositions() {
  const groups = {core: [], cortex: [], interface: [], support: []};
  _threadMatrix.threads.forEach(t => (groups[t.ring] || groups.support).push(t));
  const radii = {core: 1.05, cortex: 2.05, interface: 2.95, support: 3.75};
  const heights = {core: 0.78, cortex: 0.26, interface: -0.30, support: -0.82};
  const starts = {core: -Math.PI / 2, cortex: -Math.PI * 0.72, interface: -Math.PI * 0.62, support: -Math.PI * 0.55};
  const positions = {};
  Object.entries(groups).forEach(([ring, items]) => {
    const count = Math.max(1, items.length);
    items.forEach((thread, index) => {
      const a = starts[ring] + (Math.PI * 2 * index / count);
      const radius = radii[ring] || radii.support;
      positions[thread.id] = {
        angle: a,
        ring,
        x: Math.cos(a) * radius,
        y: Math.sin(a) * radius * 0.68,
        z: (heights[ring] || 0) + Math.sin(a * 2.0) * 0.26,
      };
    });
  });
  return positions;
}

function disposeObject(obj) {
  if (!obj) return;
  if (obj.geometry) obj.geometry.dispose();
  if (obj.material) {
    const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
    materials.forEach(mat => {
      if (mat.map) mat.map.dispose();
      mat.dispose?.();
    });
  }
}

function clearGroup(group) {
  if (!group) return;
  while (group.children.length) {
    const child = group.children.pop();
    if (child.children?.length) child.children.forEach(disposeObject);
    disposeObject(child);
  }
}

function syncThreadScene() {
  if (!_thread3d.ready || !_thread3d.THREE) return;
  const THREE = _thread3d.THREE;
  clearGroup(_thread3d.ringGroup);
  clearGroup(_thread3d.edgeGroup);
  clearGroup(_thread3d.nodeGroup);
  clearGroup(_thread3d.fileGroup);
  clearGroup(_thread3d.labelGroup);
  _thread3d.interactive = [];
  _thread3d.nodeMeshes.clear();
  _thread3d.fileMeshes.clear();

  const positions = layoutThreadPositions();
  _threadMatrix.nodes = positions;
  const ringDefs = [
    ['core', 1.05, 5, 0.78],
    ['cortex', 2.05, 7, 0.26],
    ['interface', 2.95, 9, -0.30],
    ['support', 3.75, 12, -0.82],
  ];
  ringDefs.forEach(([ring, radius, sides, z], idx) => {
    const torus = new THREE.Mesh(
      new THREE.TorusGeometry(radius, 0.008 + idx * 0.002, 8, sides),
      new THREE.MeshBasicMaterial({color: threadColorNumber(ring), transparent: true, opacity: 0.16, depthWrite: false})
    );
    torus.scale.y = 0.68;
    torus.position.z = z;
    torus.rotation.z = idx * 0.09;
    _thread3d.ringGroup.add(torus);
  });

  _threadMatrix.edges.forEach(edge => {
    const a = positions[edge.from];
    const b = positions[edge.to];
    if (!a || !b) return;
    const ta = threadById(edge.from);
    const tb = threadById(edge.to);
    const activation = (Number(ta?.activation || 0) + Number(tb?.activation || 0)) / 2;
    const start = new THREE.Vector3(a.x, a.y, a.z);
    const end = new THREE.Vector3(b.x, b.y, b.z);
    const mid = start.clone().add(end).multiplyScalar(0.5);
    mid.z += 0.18 + activation * 0.35;
    const curve = new THREE.CatmullRomCurve3([start, mid, end]);
    const geom = new THREE.BufferGeometry().setFromPoints(curve.getPoints(18));
    const line = new THREE.Line(
      geom,
      new THREE.LineBasicMaterial({color: threadColorNumber(ta || tb || 'core'), transparent: true, opacity: 0.12 + activation * 0.34})
    );
    _thread3d.edgeGroup.add(line);
  });

  _threadMatrix.threads.forEach(thread => {
    const p = positions[thread.id];
    if (!p) return;
    const activation = Number(thread.activation || 0);
    const color = threadColorNumber(thread);
    const radius = Math.max(0.18, Math.min(0.48, 0.18 + Math.sqrt(Number(thread.files || 0)) * 0.032 + activation * 0.12));
    const body = new THREE.Mesh(
      new THREE.IcosahedronGeometry(radius, 3),
      new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.18 + activation * 0.32,
        metalness: 0.42,
        roughness: 0.28,
        transparent: true,
        opacity: 0.92,
      })
    );
    body.position.set(p.x, p.y, p.z);
    body.userData = {kind: 'thread', threadId: thread.id, baseScale: 1, baseRadius: radius};
    _thread3d.nodeGroup.add(body);
    _thread3d.interactive.push(body);
    _thread3d.nodeMeshes.set(thread.id, body);

    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(radius * 1.92, 24, 16),
      new THREE.MeshBasicMaterial({color, transparent: true, opacity: 0.055 + activation * 0.08, depthWrite: false})
    );
    halo.position.copy(body.position);
    halo.userData = {kind: 'halo', threadId: thread.id, baseScale: 1};
    _thread3d.nodeGroup.add(halo);
    body.userData.halo = halo;

    const label = makeThreadLabel(`${thread.label || thread.id}  ${Math.round(activation * 100)}%`, threadColorHex(thread));
    label.position.set(p.x, p.y - radius - 0.18, p.z + 0.04);
    label.userData = {threadId: thread.id};
    _thread3d.labelGroup.add(label);
    body.userData.label = label;

    const files = (Array.isArray(thread.hot_files) ? thread.hot_files : []).slice(0, 14);
    files.forEach((file, index) => {
      const a = index * 2.3999632297 + p.angle * 0.5;
      const orbit = radius + 0.46 + (index % 4) * 0.095;
      const vertical = ((index % 5) - 2) * 0.065;
      const dotRadius = Math.max(0.075, Math.min(0.16, 0.068 + Math.sqrt(Number(file.lines || 0)) * 0.0017));
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(dotRadius, 18, 12),
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.25 + activation * 0.25,
          metalness: 0.24,
          roughness: 0.36,
          transparent: true,
          opacity: 0.82,
        })
      );
      dot.position.set(
        p.x + Math.cos(a) * orbit,
        p.y + Math.sin(a) * orbit * 0.72,
        p.z + vertical + Math.cos(a * 0.7) * 0.16
      );
      dot.userData = {
        kind: 'file',
        threadId: thread.id,
        path: file.path,
        baseScale: 1,
        activation,
      };
      _thread3d.fileGroup.add(dot);
      _thread3d.interactive.push(dot);
      _thread3d.fileMeshes.set(`${thread.id}::${file.path}`, dot);

      const fiber = new THREE.BufferGeometry().setFromPoints([body.position, dot.position]);
      const fiberLine = new THREE.Line(
        fiber,
        new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.08 + activation * 0.12})
      );
      _thread3d.edgeGroup.add(fiberLine);
    });
  });
  updateThreadHighlights();
  threadResize();
}

function makeThreadLabel(text, color) {
  const THREE = _thread3d.THREE;
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  ctx.font = '700 34px JetBrains Mono, monospace';
  const metrics = ctx.measureText(text);
  canvas.width = Math.min(1024, Math.max(256, Math.ceil(metrics.width + 44)));
  canvas.height = 72;
  ctx.font = '700 34px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(3,4,8,0.66)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.36;
  ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.fillText(truncateText(text, 32), canvas.width / 2, canvas.height / 2 + 1);
  const texture = new THREE.CanvasTexture(canvas);
  if (THREE.SRGBColorSpace) texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: texture, transparent: true, depthWrite: false}));
  sprite.scale.set(canvas.width / 360, canvas.height / 360, 1);
  return sprite;
}

function updateThreadHighlights() {
  if (!_thread3d.ready) return;
  _thread3d.nodeMeshes.forEach((mesh, id) => {
    const selected = _threadMatrix.selected === id;
    const hovered = _threadMatrix.hover === id;
    const thread = threadById(id);
    const activation = Number(thread?.activation || 0);
    mesh.userData.targetScale = selected ? 1.22 : hovered ? 1.12 : 1;
    mesh.material.emissiveIntensity = (selected ? 0.65 : hovered ? 0.48 : 0.18) + activation * 0.28;
    if (mesh.userData.halo) mesh.userData.halo.material.opacity = selected ? 0.18 : hovered ? 0.13 : 0.055 + activation * 0.08;
    if (mesh.userData.label) mesh.userData.label.material.opacity = selected || hovered ? 1 : 0.72;
  });
  _thread3d.fileMeshes.forEach(mesh => {
    const selectedThread = _threadMatrix.selected === mesh.userData.threadId;
    const selectedFile = selectedThread && _threadMatrix.selectedFile === mesh.userData.path;
    mesh.userData.targetScale = selectedFile ? 1.95 : selectedThread ? 1.32 : 1;
    mesh.material.opacity = selectedFile ? 1 : selectedThread ? 0.94 : 0.46;
    mesh.material.emissiveIntensity = selectedFile ? 0.82 : selectedThread ? 0.46 : 0.20 + Number(mesh.userData.activation || 0) * 0.18;
  });
}

function threadFrame(ts) {
  _threadMatrix.raf = 0;
  if (!_threadMatrixRunning || !_thread3d.ready) return;
  const time = (ts || performance.now()) * 0.001;
  const auto = _thread3d.dragging ? 0 : Math.sin(time * 0.16) * 0.055;
  _thread3d.root.rotation.x = _thread3d.rotX;
  _thread3d.root.rotation.y = _thread3d.rotY + auto;
  _thread3d.root.rotation.z = Math.sin(time * 0.11) * 0.025;
  _thread3d.camera.position.set(0, 0, _thread3d.zoom);
  _thread3d.camera.lookAt(0, 0, 0);

  _thread3d.nodeMeshes.forEach((mesh, id) => {
    const thread = threadById(id);
    const activation = Number(thread?.activation || 0);
    const pulse = (0.5 + 0.5 * Math.sin(time * 3.1 + mesh.position.x * 1.7)) * activation * 0.06;
    const target = Number(mesh.userData.targetScale || 1) + pulse;
    const current = mesh.scale.x || 1;
    const next = current + (target - current) * 0.16;
    mesh.scale.setScalar(next);
    if (mesh.userData.halo) {
      const haloScale = (Number(mesh.userData.targetScale || 1) + pulse * 1.8);
      mesh.userData.halo.scale.setScalar(haloScale);
    }
  });
  let fileIndex = 0;
  _thread3d.fileMeshes.forEach(mesh => {
    const pulse = 1 + Math.sin(time * 4.2 + fileIndex * 0.37) * 0.045 * (0.4 + Number(mesh.userData.activation || 0));
    fileIndex += 1;
    const target = Number(mesh.userData.targetScale || 1) * pulse;
    const current = mesh.scale.x || 1;
    const next = current + (target - current) * 0.20;
    mesh.scale.setScalar(next);
  });
  _thread3d.renderer.render(_thread3d.scene, _thread3d.camera);
  _threadMatrix.raf = requestAnimationFrame(threadFrame);
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
  {id:"quantum",label:"Quantum Soul",color:[236,72,153],cx:0.18,cy:0.20,r:0.048,n:70,desc:"156-qubit Kingston manifold",tier:"core"},
  {id:"akashic",label:"Akashic Hub",color:[167,139,250],cx:0.50,cy:0.28,r:0.055,n:85,desc:"256-d shared vector state",tier:"core"},
  {id:"pineal",label:"Pineal Gate",color:[124,92,252],cx:0.82,cy:0.20,r:0.048,n:70,desc:"Kingston entropy router",tier:"core"},
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
    queueNeuralPulse(_hoveredRegion, 1.5);
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
function queueNeuralPulse(regionId, intensity=1) {
  if (!isNeuralFresh() || !neurons.length) return;
  const regionNeurons = neurons.filter(n => n.region === regionId);
  if (!regionNeurons.length) return;
  const count = Math.max(2, Math.min(10, Math.round(regionNeurons.length * 0.08 * intensity)));
  for (let i = 0; i < count; i++) {
    const n = regionNeurons[Math.floor(Math.random() * regionNeurons.length)];
    n.fire = Math.max(n.fire, 0.72 + Math.min(intensity, 1.5) * 0.18);
  }
  const outbound = synapses.filter(s => s.fromRegion === regionId || s.toRegion === regionId);
  const signalCount = Math.min(4, outbound.length);
  for (let i = 0; i < signalCount; i++) {
    const s = outbound[Math.floor(Math.random() * outbound.length)];
    signals.push({ax:s.a.x, ay:s.a.y, bx:s.b.x, by:s.b.y, t:0, color:s.a.color, size:2.5 + intensity, trail:true});
    s.a.fire = Math.max(s.a.fire, 0.8);
    s.b.fire = Math.max(s.b.fire, 0.45);
    s.pulse = 1;
    _signalCount++;
  }
}
function fireRegion(id) { queueNeuralPulse(id, 1.4); }
function neuralReset() { _zoom=1; _panX=0; _panY=0; document.getElementById('nZoom').textContent='1.0x'; }
function neuralBurst() {
  if (!isNeuralFresh()) { updateLiveHud(); return; }
  Object.keys(regionCenters).forEach(id => queueNeuralPulse(id, 0.9));
}
function neuralTogglePause() { _paused=!_paused; document.getElementById('nctrPause').textContent=_paused?'▶':'▮▮'; if(!_paused) requestAnimationFrame(neuralFrame); }

function pulseFromNexus(entry) {
  if (!entry) return;
  const text = `${entry.from || ''} ${entry.topic || ''} ${entry.event || ''}`.toLowerCase();
  const routes = [
    ['quantum', ['quantum', 'qubit', 'kingston', 'matrix']],
    ['pineal', ['pineal', 'route', 'routing', 'gate', 'moe']],
    ['akashic', ['akashic', 'memory', 'recall']],
    ['lora', ['lora', 'soul', 'forge', 'fracture', 'dataset']],
    ['qwen', ['qwen', 'local', 'slm']],
    ['phone', ['phone', 'twilio', 'call']],
    ['vtv', ['voice', 'audio', 'vision', 'vtv', 'tts']],
    ['dream', ['dream', 'sleep']],
    ['discord', ['discord']],
    ['pulse', ['pulse', 'proactive']],
  ];
  const route = routes.find(([, words]) => words.some(word => text.includes(word)));
  queueNeuralPulse(route ? route[0] : 'nexus_bus', 1.15);
}

// Update lobe statuses from polling data
function updateNeuralFromLobes(lobes) {
  if (!lobes) return;
  const mapping = {'Nexus Bus':'nexus_bus','Quantum Soul':'quantum','Quantum API':'quantum','Akashic Hub':'akashic','Pineal Gate':'pineal','LoRA Server':'lora','Phone Bridge':'phone','VTV':'vtv','ProactivePulse':'pulse','Dream State':'dream','n8n Workflow':'nexus_bus','Discord Bridge':'discord'};
  lobes.forEach(l => {
    const regionId = mapping[l.name];
    if (regionId) {
      _lobeStatusMap[regionId] = l.status;
      if (l.status === 'online') queueNeuralPulse(regionId, l.latency_class === 'slow' ? 0.5 : 0.9);
    }
  });
}

let _lastNeural = 0;
function neuralFrame(ts) {
  if (!_neuralRunning || _paused) return;
  const dt = Math.min((ts - _lastNeural)/16.67, 3);
  _lastNeural = ts;
  const live = isNeuralFresh();
  const motionDt = live ? dt : 0;
  const W = nMain.width, H = nMain.height;
  if (W < 10) { requestAnimationFrame(neuralFrame); return; }
  updateLiveHud();

  // Clear with subtle fade
  ctxMain.save();
  ctxMain.setTransform(_zoom, 0, 0, _zoom, _panX, _panY);
  ctxMain.fillStyle = live ? 'rgba(3,4,8,0.15)' : 'rgba(3,4,8,0.04)';
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
    const pulseAlpha = live ? 0.03 + Math.sin(pulsePhase)*0.015 : 0.012;
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
    // Motion is telemetry-gated; stale/offline freezes the map.
    n.x += n.vx*motionDt + (live ? Math.sin(time*0.5 + n.phase)*0.05 : 0);
    n.y += n.vy*motionDt + (live ? Math.cos(time*0.4 + n.phase)*0.05 : 0);

    // Soft boundary (attract back to origin)
    const dx = n.x - n.ox, dy = n.y - n.oy;
    const dist = Math.sqrt(dx*dx+dy*dy);
    const maxDist = 40;
    if (dist > maxDist) { n.vx -= dx*0.001; n.vy -= dy*0.001; }

    if (n.fire > 0) {
      if (live) n.fire = Math.max(0, n.fire - 0.015*dt);
      firing++;
    }

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
    if (live) s.pulse *= 0.95;
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

  signals = signals.filter(sig => {
    sig.t += live ? 0.018*dt : 0;
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
    _signalRate = live ? _signalCount : 0;
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
let _eventSource = null;
function connectSSE() {
  try {
    if (_eventSource) _eventSource.close();
    const es = new EventSource('/api/stream');
    _eventSource = es;
    es.onopen = () => {
      _liveState.sse = true;
      _sseRetries = 0;
      document.getElementById('sseText').textContent = 'LIVE';
      document.getElementById('ssePill').classList.remove('off');
      updateLiveHud();
    };
    es.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data);
        if (d.type === 'poll') {
          markLive('poll');
          updateLobes(d.lobes||[]);
          updateQuantum(d.quantum||{}, d.quantum_architecture || null);
          updateBrain(d.brain||{}, d.voice||{}, d.codebase||{});
          updateThreads(d.codebase_threads||{});
          updateMemoryEvents(d.memory_events||[]);
          document.getElementById('sOnline').textContent = `${d.online}/${d.total}`;
          document.getElementById('sUptime').textContent = fmtUp(d.uptime||0);
          document.getElementById('sMsgs').textContent = d.nexus_msg_count||0;
          document.getElementById('sBusLobes').textContent = d.nexus_lobes||0;
          document.getElementById('sTopics').textContent = d.nexus_topics||0;
          document.getElementById('lobeBadge').textContent = `${d.online}/${d.total}`;
        }
        if (d.type === 'nexus' && d.data) {
          markLive('nexus');
          pulseFromNexus(d.data);
          const el = document.getElementById('feedScroll');
          el.insertAdjacentHTML('afterbegin', `<div class="feed-item"><span class="feed-from">${esc(d.data.from||'')}</span> <span class="feed-topic">${esc(d.data.topic||'')}</span></div>`);
        }
      } catch(e){}
    };
    es.onerror = () => {
      _liveState.sse = false;
      markOffline('sse error');
      es.close();
      if (_eventSource === es) _eventSource = null;
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
    _liveState.sse = false;
    markOffline(e.message || 'sse unavailable');
    document.getElementById('sseText').textContent = 'POLL';
  }
}

// ── Boot ─────────────────────────────────────
setInterval(() => { document.getElementById('clock').textContent = new Date().toLocaleTimeString(); }, 1000);
window.addEventListener('offline', () => {
  _liveState.browserOnline = false;
  _liveState.sse = false;
  if (_eventSource) { _eventSource.close(); _eventSource = null; }
  markOffline('browser offline');
});
window.addEventListener('online', () => {
  _liveState.browserOnline = true;
  updateLiveHud();
  fetchState();
  connectSSE();
});
updateLiveHud();
fetchState();
connectSSE();
setInterval(fetchState, 5000);
setInterval(updateLiveHud, 1000);
startKingstonMatrix();
window.addEventListener('resize', () => {
  if (_neuralRunning) neuralResize();
  if (_threadMatrixRunning) threadResize();
});
</script>
</body>
</html>"""


# ── Entrypoint ───────────────────────────────────────────────────────────────

async def weaver_dashboard_serve():
    """Entry point for launching from weaver.py."""
    import uvicorn
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import uvicorn
    print(f"Weaver Live Dashboard starting on http://{HOST}:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT)
