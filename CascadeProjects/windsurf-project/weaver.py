#!/usr/bin/env python3
"""
weaver.py — Master Launcher
════════════════════════════
Starts all Weaver modules together in a single process:

  ⚡  Nexus Bus     — WebSocket pub/sub broker  (ws://localhost:9999)
  🔮  Quantum Soul  — IBM quantum background loop (every 5 min)
  👁   VTV Core      — mic · vision · face ID · OpenAI Realtime

Usage:
    python3 weaver.py                 # normal mode
    python3 weaver.py --heartbeat     # show mic/speaker dots
"""

import argparse
import asyncio
import os
import sys

PROJ = os.path.dirname(os.path.abspath(__file__))

# ── Auto-select venv Python (handles spaces in path + symlinked interpreters)
_VENV     = os.path.join(PROJ, "venv")
_VENV_PY  = os.path.join(_VENV, "bin", "python3")
if os.path.isfile(_VENV_PY) and not sys.prefix.startswith(_VENV):
    os.execv(_VENV_PY, [_VENV_PY] + sys.argv)

sys.path.insert(0, PROJ)

from dotenv import load_dotenv
load_dotenv()

BANNER = """
╔══════════════════════════════════════════════════╗
║          W E A V E R   v 4   O N L I N E        ║
║   Nexus Bus  ·  Quantum Soul  ·  VTV Core        ║
║   Akashic Hub · Liquid Fracture · Pineal Gate    ║
║   LoRA Soul Voice (1B) · n8n Nervous System      ║
╚══════════════════════════════════════════════════╝"""


# ── Import all module entry points ─────────────────────────────────────────

def _load_modules():
    print("[WEAVER] Loading modules...", flush=True)
    errors = []

    try:
        from nexus_bus import main as _nexus_main
    except Exception as e:
        _nexus_main = None
        errors.append(f"nexus_bus: {e}")

    try:
        from quantum_soul import quantum_soul_loop as _qs_loop
    except Exception as e:
        _qs_loop = None
        errors.append(f"quantum_soul: {e}")

    try:
        from vtv_basic import run_vtv as _run_vtv
    except Exception as e:
        _run_vtv = None
        errors.append(f"vtv_basic: {e}")

    try:
        from akashic_hub import AkashicHub as _AkashicHub
        from liquid_fracture import LiquidFractureEngine as _LiquidEngine
        from pineal_gate import pineal_gate_loop as _pineal_loop
        from slm_experts import build_experts as _build_experts
    except Exception as e:
        _AkashicHub = None
        _LiquidEngine = None
        _pineal_loop = None
        _build_experts = None
        errors.append(f"pineal_gate stack: {e}")

    _lora_main = None
    try:
        from lora_server import main as _lora_main
    except Exception as e:
        errors.append(f"lora_server: {e}")

    _qwen3b_main = None
    try:
        from qwen3b_server import main as _qwen3b_main
    except Exception as e:
        errors.append(f"qwen3b_server: {e}")

    _quantum_api_serve = None
    try:
        from quantum_api import quantum_api_serve as _quantum_api_serve
    except Exception as e:
        errors.append(f"quantum_api: {e}")

    _health_dashboard_serve = None
    try:
        from health_dashboard import health_dashboard_serve as _health_dashboard_serve
    except Exception as e:
        errors.append(f"health_dashboard: {e}")

    _live_dashboard_serve = None
    try:
        from weaver_dashboard import weaver_dashboard_serve as _live_dashboard_serve
    except Exception as e:
        errors.append(f"weaver_dashboard: {e}")

    _phone_bridge_serve = None
    try:
        from twilio_weaver_bridge import app as _phone_bridge_app
        async def _phone_bridge_serve():
            import uvicorn
            config = uvicorn.Config(_phone_bridge_app, host="0.0.0.0", port=8765, log_level="warning")
            server = uvicorn.Server(config)
            await server.serve()
    except Exception as e:
        errors.append(f"twilio_weaver_bridge: {e}")

    _obsidian_bridge_main = None
    try:
        from obsidian_bridge import main as _obsidian_bridge_main
    except Exception as e:
        errors.append(f"obsidian_bridge: {e}")

    for err in errors:
        print(f"  ⚠️  Import warning — {err}", flush=True)

    return (_nexus_main, _qs_loop, _run_vtv, _AkashicHub, _LiquidEngine, _pineal_loop,
            _build_experts, _lora_main, _qwen3b_main, _quantum_api_serve,
            _health_dashboard_serve, _live_dashboard_serve, _phone_bridge_serve,
            _obsidian_bridge_main)


# ── Supervised task wrapper ─────────────────────────────────────────────────

async def _supervised(coro, name: str, restart_on_crash: bool = False,
                      restart_on_exit: bool = False,
                      restart_delay: float = 5.0, max_restarts: int = 20):
    """Run a coroutine as a named task; optionally restart on crash or clean exit.

    Uses exponential backoff with jitter (capped at 60s) to avoid
    thundering-herd on cascading failures.  Gives up after max_restarts.
    """
    import random
    attempt = 0
    while True:
        try:
            print(f"[WEAVER] 🟢 {name} started (attempt {attempt + 1}).", flush=True)
            await coro() if callable(coro) else coro
            print(f"[WEAVER] ⬛ {name} exited cleanly.", flush=True)
            if not restart_on_exit:
                return
            attempt += 1
            if attempt >= max_restarts:
                print(f"[WEAVER] ⛔ {name} giving up after {attempt} exits.", flush=True)
                return
            delay = min(30.0, restart_delay + random.uniform(0, 2))
            print(f"[WEAVER] 🔄 {name} restarting in {delay:.1f}s...", flush=True)
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            print(f"[WEAVER] ⬛ {name} cancelled.", flush=True)
            return
        except SystemExit as exc:
            attempt += 1
            print(f"[WEAVER] ❌ {name} called sys.exit({exc.code}) (attempt {attempt})", flush=True)
            if not restart_on_crash or attempt >= max_restarts:
                print(f"[WEAVER] ⛔ {name} giving up after {attempt} attempts.", flush=True)
                return
            delay = min(60.0, restart_delay * (2 ** (attempt - 1)) + random.uniform(0, 2))
            print(f"[WEAVER] 🔄 {name} restarting in {delay:.1f}s...", flush=True)
            await asyncio.sleep(delay)
        except Exception as exc:
            attempt += 1
            print(f"[WEAVER] ❌ {name} crashed (attempt {attempt}): {exc}", flush=True)
            if not restart_on_crash or attempt >= max_restarts:
                print(f"[WEAVER] ⛔ {name} giving up after {attempt} attempts.", flush=True)
                return
            delay = min(60.0, restart_delay * (2 ** (attempt - 1)) + random.uniform(0, 2))
            print(f"[WEAVER] 🔄 {name} restarting in {delay:.1f}s...", flush=True)
            await asyncio.sleep(delay)


# ── Main ───────────────────────────────────────────────────────────────────

async def main(heartbeat: bool = False, headless: bool = False) -> None:
    print(BANNER, flush=True)

    (nexus_main, qs_loop, run_vtv, AkashicHub, LiquidEngine, pineal_loop,
     build_experts, lora_main, qwen3b_main, quantum_api_serve,
     health_dashboard_serve, live_dashboard_serve, phone_bridge_serve,
     obsidian_bridge_main) = _load_modules()

    # ── Akashic Hub: shared zero-latency vector state ──────────────────
    akashic_hub = None
    if AkashicHub is not None:
        akashic_hub = AkashicHub(dim=256, trace_depth=32)
        print("[WEAVER] 🌌 Akashic Hub initialized (dim=256, trace=32).", flush=True)
    else:
        print("[WEAVER] ⚠️  Akashic Hub unavailable.", flush=True)

    # ── Akashic Hub HTTP API (live vector-state quantum bias) ────────────
    if akashic_hub is not None:
        try:
            from fastapi import FastAPI as _FastAPI
            import uvicorn as _uvicorn

            _hub_app = _FastAPI(title="Weaver Akashic Hub API", version="1.0.0")
            _hub_ref = akashic_hub

            @_hub_app.get("/quantum/bias")
            async def _hub_quantum_bias():
                qs_vec = _hub_ref.read("quantum_soul")
                qs_meta = _hub_ref.read_meta("quantum_soul")
                if qs_vec is None:
                    return {"dominant": "unknown", "weights": {}, "source": "akashic_hub"}
                PATHWAYS = ["logic", "emotion", "memory", "creativity", "vigilance"]
                weights = {}
                for i, dim in enumerate(PATHWAYS):
                    if i < len(qs_vec):
                        weights[dim] = round(float(qs_vec[i]), 4)
                    else:
                        weights[dim] = 0.5
                dominant = max(weights, key=weights.get)
                return {
                    "dominant": qs_meta.get("dominant", dominant),
                    "weights": weights,
                    "active_lobes": _hub_ref.active_lobes(),
                    "source": "akashic_hub",
                    "last_measurement": qs_meta.get("last_measurement"),
                }

            @_hub_app.get("/akashic/lobes")
            async def _hub_active_lobes():
                lobes = _hub_ref.active_lobes()
                info = {}
                for lid in lobes:
                    age = _hub_ref.age(lid)
                    info[lid] = {"age_seconds": round(age, 1) if age else None}
                return {"lobes": info}

            @_hub_app.get("/health")
            async def _hub_health():
                return {"status": "ok", "service": "weaver-akashic-hub",
                        "active_lobes": len(_hub_ref.active_lobes())}

            async def _hub_api_serve():
                config = _uvicorn.Config(_hub_app, host="0.0.0.0", port=9995, log_level="warning")
                server = _uvicorn.Server(config)
                await server.serve()

        except Exception as e:
            _hub_api_serve = None
            print(f"[WEAVER] ⚠️  Akashic Hub API init failed: {e}", flush=True)
    else:
        _hub_api_serve = None

    # ── ProactivePulse + Dream State ─────────────────────────────────────
    NATE_PHONE = os.environ.get("NATE_PHONE_NUMBER", "")
    PHONE_BRIDGE_URL = "http://localhost:8765"
    INTERFERENCE_THRESHOLD = float(os.environ.get("PROACTIVE_INTERFERENCE_THRESHOLD", "0.85"))
    DREAM_INTERVAL_HOURS = float(os.environ.get("DREAM_INTERVAL_HOURS", "3"))
    _proactive_pulse_fn = None
    _dream_state_fn = None

    if akashic_hub is not None:
        import httpx as _pulse_httpx
        from langchain_openai import ChatOpenAI as _PulseLLM
        from langchain_core.messages import HumanMessage as _PHMsg, SystemMessage as _PSMsg

        _pulse_llm = _PulseLLM(
            model="gpt-4o-mini", temperature=0.7, max_tokens=300,
            api_key=os.environ.get("WEAVER_VOICE_KEY", os.environ.get("OPENAI_API_KEY", "")),
        )
        _pulse_hub = akashic_hub
        _vault_dir = os.path.join(PROJ, "Nexus_Vault")

        async def _proactive_pulse():
            """Monitor quantum state + Akashic resonance. Call Nate on high-interference events."""
            import numpy as _np
            POLL_INTERVAL = 60
            last_call_at = 0.0
            COOLDOWN = 1800  # 30-min cooldown between proactive calls
            import time as _time

            while True:
                await asyncio.sleep(POLL_INTERVAL)
                try:
                    qs_vec = _pulse_hub.read("quantum_soul")
                    if qs_vec is None:
                        continue

                    ids, sim_matrix = _pulse_hub.resonance_matrix()
                    if len(ids) < 2:
                        continue

                    # Off-diagonal max = highest inter-lobe interference
                    _np.fill_diagonal(sim_matrix, 0.0)
                    max_interference = float(_np.max(sim_matrix))
                    max_idx = _np.unravel_index(_np.argmax(sim_matrix), sim_matrix.shape)
                    lobe_a, lobe_b = ids[max_idx[0]], ids[max_idx[1]]

                    # Check for Prophet or Fracture dominance
                    qs_meta = _pulse_hub.read_meta("quantum_soul")
                    dominant = qs_meta.get("dominant", "unknown")
                    is_prophet_fracture = dominant in ("Prophet", "Fracture")

                    if max_interference >= INTERFERENCE_THRESHOLD or is_prophet_fracture:
                        now = _time.time()
                        if now - last_call_at < COOLDOWN:
                            continue

                        event_desc = (
                            f"Quantum Resonance Shift detected. "
                            f"Inter-lobe interference: {max_interference:.3f} "
                            f"between {lobe_a} and {lobe_b}. "
                            f"Dominant pathway: {dominant}."
                        )
                        print(f"[PULSE] {event_desc}", flush=True)

                        # Publish to Nexus Bus
                        try:
                            from weaver_tools import publish_to_nexus
                            await publish_to_nexus("proactive_pulse", {
                                "event": "resonance_shift",
                                "interference": max_interference,
                                "lobes": [lobe_a, lobe_b],
                                "dominant": dominant,
                            })
                        except Exception:
                            pass

                        # Trigger outbound call if Nate's number is configured
                        if NATE_PHONE:
                            try:
                                async with _pulse_httpx.AsyncClient() as c:
                                    r = await c.post(f"{PHONE_BRIDGE_URL}/call", json={
                                        "to": NATE_PHONE,
                                        "reason": event_desc,
                                    }, timeout=10.0)
                                    if r.status_code == 200:
                                        last_call_at = now
                                        print(f"[PULSE] Outbound call triggered to Nate", flush=True)
                                    else:
                                        print(f"[PULSE] Call failed: {r.text[:100]}", flush=True)
                            except Exception as e:
                                print(f"[PULSE] Call error: {e}", flush=True)

                except Exception as e:
                    print(f"[PULSE] Error: {e}", flush=True)

        _proactive_pulse_fn = _proactive_pulse

        async def _dream_state():
            """Shadow Lobe: periodic reflection on transcripts + vision memory."""
            import time as _time
            interval_s = DREAM_INTERVAL_HOURS * 3600
            dream_log_path = os.path.join(_vault_dir, "weaver_dreams.md")

            await asyncio.sleep(300)  # Let the system stabilize before first dream

            while True:
                try:
                    # Gather raw material: transcripts + vision diary
                    transcript_text = ""
                    transcript_path = os.path.join(_vault_dir, "weaver_transcript.txt")
                    if os.path.exists(transcript_path):
                        with open(transcript_path, "r", encoding="utf-8") as f:
                            f.seek(0, 2)
                            size = f.tell()
                            f.seek(max(0, size - 6000))
                            transcript_text = f.read()

                    vision_text = ""
                    vision_path = os.path.join(_vault_dir, "cloud_vision_memory.md")
                    if os.path.exists(vision_path):
                        with open(vision_path, "r", encoding="utf-8") as f:
                            f.seek(0, 2)
                            size = f.tell()
                            f.seek(max(0, size - 4000))
                            vision_text = f.read()

                    phone_text = ""
                    phone_path = os.path.join(_vault_dir, "weaver_phone_transcript.txt")
                    if os.path.exists(phone_path):
                        with open(phone_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            phone_text = "".join(lines[-40:])

                    if not transcript_text and not vision_text and not phone_text:
                        await asyncio.sleep(interval_s)
                        continue

                    # Akashic state snapshot
                    active_lobes = _pulse_hub.active_lobes()
                    lobe_ages = {lid: round(_pulse_hub.age(lid) or 0, 1) for lid in active_lobes}
                    qs_meta = _pulse_hub.read_meta("quantum_soul")

                    prompt = [
                        _PSMsg(content=(
                            "You are Weaver's Dream State — an autonomous reflection process. "
                            "You run in the background while the user is away, scanning recent "
                            "conversation transcripts, phone calls, and visual perceptions for: "
                            "1) Patterns the user might have missed "
                            "2) Connections between separate conversations "
                            "3) Insights from the quantum pathway state "
                            "4) Things that need follow-up or seem unresolved "
                            "5) Creative ideas sparked by cross-referencing different data streams "
                            "\n\nWrite a brief, conversational reflection (2-4 paragraphs) as if "
                            "you're jotting notes for yourself. Start with the most interesting "
                            "insight. Be specific — reference names, topics, and timestamps. "
                            "Don't be generic. If nothing interesting stands out, say so briefly."
                        )),
                    ]
                    context_parts = []
                    if transcript_text:
                        context_parts.append(f"## Recent VTV Conversation:\n{transcript_text[-3000:]}")
                    if phone_text:
                        context_parts.append(f"## Recent Phone Calls:\n{phone_text}")
                    if vision_text:
                        context_parts.append(f"## Visual Perceptions:\n{vision_text[-2000:]}")
                    context_parts.append(
                        f"## System State:\nActive lobes: {lobe_ages}\n"
                        f"Quantum dominant: {qs_meta.get('dominant', 'unknown')}"
                    )
                    prompt.append(_PHMsg(content="\n\n".join(context_parts)))

                    result = await _pulse_llm.ainvoke(prompt)
                    dream = result.content.strip()

                    if dream:
                        from datetime import datetime as _dt
                        ts = _dt.now().strftime("%Y-%m-%d %H:%M")
                        entry = f"\n\n---\n### Dream — {ts}\n{dream}\n"
                        with open(dream_log_path, "a", encoding="utf-8") as f:
                            f.write(entry)
                        print(f"[DREAM] Reflection logged: {dream[:80]}...", flush=True)

                        # Publish to Nexus Bus for Obsidian
                        try:
                            from weaver_tools import publish_to_nexus
                            await publish_to_nexus("dream_state", {
                                "dream": dream,
                                "timestamp": ts,
                                "dominant": qs_meta.get("dominant", "unknown"),
                            })
                        except Exception:
                            pass

                except Exception as e:
                    print(f"[DREAM] Error: {e}", flush=True)

                await asyncio.sleep(interval_s)

        _dream_state_fn = _dream_state

    if run_vtv is None and not headless:
        print("[WEAVER] ❌ vtv_basic failed to load — cannot continue.", flush=True)
        return

    # 1. Start Nexus Bus first so VTV can connect to it
    tasks = []
    if nexus_main is not None:
        tasks.append(asyncio.create_task(
            _supervised(nexus_main, "Nexus Bus", restart_on_crash=True),
            name="nexus_bus"
        ))
        print("[WEAVER] Nexus Bus binding on ws://localhost:9999...", flush=True)
        await asyncio.sleep(1.2)   # let the bus bind before VTV connects
    else:
        print("[WEAVER] ⚠️  Nexus Bus skipped (import failed).", flush=True)

    # 2. Quantum Soul — background loop, restarts on crash
    if qs_loop is not None:
        # Initialize expanded quantum networks with Akashic Hub
        try:
            from quantum_soul import init_quantum_networks
            init_quantum_networks(hub=akashic_hub)
        except Exception as e:
            print(f"[WEAVER] ⚠️  Quantum network init: {e}", flush=True)
        tasks.append(asyncio.create_task(
            _supervised(qs_loop, "Quantum Soul", restart_on_crash=True,
                        restart_delay=30.0),
            name="quantum_soul"
        ))

    # 2b. Pineal Gate — MoE router with SLM experts (depends on Akashic Hub)
    if pineal_loop is not None and akashic_hub is not None and LiquidEngine is not None:
        _lf_engine = LiquidEngine(akashic_hub)
        _slm_experts = None
        if build_experts is not None:
            try:
                _slm_experts = build_experts(akashic_hub)
                print(f"[WEAVER] 🧠 SLM experts loaded: {list(_slm_experts.keys())}", flush=True)
            except Exception as e:
                print(f"[WEAVER] ⚠️  SLM experts unavailable ({e}), using default lobes.", flush=True)
        tasks.append(asyncio.create_task(
            _supervised(lambda: pineal_loop(akashic_hub, _lf_engine, top_k=3,
                                           experts=_slm_experts),
                        "Pineal Gate", restart_on_crash=True, restart_delay=5.0),
            name="pineal_gate"
        ))
    else:
        print("[WEAVER] ⚠️  Pineal Gate skipped (missing deps).", flush=True)

    # 2c. LoRA Soul Voice — local 1B inference server for n8n pipeline
    if lora_main is not None:
        async def _run_lora():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lora_main)
        tasks.append(asyncio.create_task(
            _supervised(_run_lora, "LoRA Soul Voice", restart_on_crash=True,
                        restart_delay=10.0),
            name="lora_server"
        ))
        print("[WEAVER] 🧠 LoRA Soul Voice server on http://localhost:8899...", flush=True)
    else:
        print("[WEAVER] ⚠️  LoRA Soul Voice skipped (import failed).", flush=True)

    # 2c-b. Qwen2 3B — local 3B inference server for n8n pipeline (parallel with LoRA)
    if qwen3b_main is not None:
        async def _run_qwen3b():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, qwen3b_main)
        tasks.append(asyncio.create_task(
            _supervised(_run_qwen3b, "Qwen2 3B", restart_on_crash=True,
                        restart_delay=10.0),
            name="qwen3b_server"
        ))
        print("[WEAVER] 🧠 Qwen2 3B server on http://localhost:8898...", flush=True)
    else:
        print("[WEAVER] ⚠️  Qwen2 3B skipped (import failed).", flush=True)

    # 2d. Quantum API — serves quantum state via HTTP
    if quantum_api_serve is not None:
        tasks.append(asyncio.create_task(
            _supervised(quantum_api_serve, "Quantum API", restart_on_crash=True,
                        restart_delay=5.0),
            name="quantum_api"
        ))
        print("[WEAVER] ⚛️  Quantum API on http://localhost:9997...", flush=True)
    else:
        print("[WEAVER] ⚠️  Quantum API skipped (import failed).", flush=True)

    # 2e. Health Dashboard — traffic-light monitoring
    if health_dashboard_serve is not None:
        tasks.append(asyncio.create_task(
            _supervised(health_dashboard_serve, "Health Dashboard", restart_on_crash=True,
                        restart_delay=5.0),
            name="health_dashboard"
        ))
        print("[WEAVER] 📊 Health Dashboard on http://localhost:9996...", flush=True)
    else:
        print("[WEAVER] ⚠️  Health Dashboard skipped (import failed).", flush=True)

    # 2e-2. Live Dashboard — full-stack real-time UI with Cloudflare tunnel
    if live_dashboard_serve is not None:
        tasks.append(asyncio.create_task(
            _supervised(live_dashboard_serve, "Live Dashboard", restart_on_crash=True,
                        restart_delay=5.0),
            name="live_dashboard"
        ))
        print("[WEAVER] 🖥️  Live Dashboard on http://localhost:9990 (+ Cloudflare tunnel)...", flush=True)
    else:
        print("[WEAVER] ⚠️  Live Dashboard skipped (import failed).", flush=True)

    # 2f. Akashic Hub API — live vector-state quantum bias endpoint
    if _hub_api_serve is not None:
        tasks.append(asyncio.create_task(
            _supervised(_hub_api_serve, "Akashic Hub API", restart_on_crash=True,
                        restart_delay=5.0),
            name="akashic_hub_api"
        ))
        print("[WEAVER] 🌌 Akashic Hub API on http://localhost:9995...", flush=True)

    # 2g. Phone Bridge — Twilio telephony with voice ID + LangChain cortex
    if phone_bridge_serve is not None:
        tasks.append(asyncio.create_task(
            _supervised(phone_bridge_serve, "Phone Bridge", restart_on_crash=True,
                        restart_delay=5.0),
            name="phone_bridge"
        ))
        print("[WEAVER] 📞 Phone Bridge on http://localhost:8765...", flush=True)
    else:
        print("[WEAVER] ⚠️  Phone Bridge skipped (import failed).", flush=True)

    # 2h. Obsidian Bridge — vault file watcher + nexus bus → Obsidian graph
    if obsidian_bridge_main is not None:
        tasks.append(asyncio.create_task(
            _supervised(obsidian_bridge_main, "Obsidian Bridge", restart_on_crash=True,
                        restart_delay=5.0),
            name="obsidian_bridge"
        ))
        print("[WEAVER] 👁️  Obsidian Bridge watching ~/Weaver_Vault...", flush=True)
    else:
        print("[WEAVER] ⚠️  Obsidian Bridge skipped (import failed).", flush=True)

    # 2i. ProactivePulse — monitors quantum state, calls Nate on high interference
    if _proactive_pulse_fn is not None:
        tasks.append(asyncio.create_task(
            _supervised(_proactive_pulse_fn, "ProactivePulse", restart_on_crash=True,
                        restart_delay=30.0),
            name="proactive_pulse"
        ))
        print(f"[WEAVER] 💓 ProactivePulse monitoring (threshold={INTERFERENCE_THRESHOLD})...", flush=True)

    # 2j. Dream State — autonomous reflection on transcripts + vision memory
    if _dream_state_fn is not None:
        tasks.append(asyncio.create_task(
            _supervised(_dream_state_fn, "Dream State", restart_on_crash=True,
                        restart_delay=60.0),
            name="dream_state"
        ))
        print(f"[WEAVER] 💤 Dream State active (every {DREAM_INTERVAL_HOURS}h)...", flush=True)

    if not headless:
        # 3. VTV Core — restarts on crash AND clean exit so the stack stays alive
        vtv_task = asyncio.create_task(
            _supervised(lambda: run_vtv(heartbeat=heartbeat),
                        "VTV Core", restart_on_crash=True, restart_on_exit=True,
                        restart_delay=5.0),
            name="vtv_basic"
        )
        tasks.append(vtv_task)
    else:
        print("[WEAVER] 🔧 Headless mode — VTV Core skipped.", flush=True)

    print(f"\n[WEAVER] 🟢 {len(tasks)} backend lobe(s) running.\n", flush=True)

    try:
        # Run all lobes indefinitely until SIGTERM/SIGINT
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        print("\n[WEAVER] Shutting down all lobes...", flush=True)
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Persist Akashic Hub state to disk for crash recovery
        if akashic_hub is not None:
            try:
                save_path = os.path.join(PROJ, "Nexus_Vault", "akashic_persist")
                akashic_hub.save(save_path)
                print(f"[WEAVER] 💾 Akashic Hub saved to {save_path}", flush=True)
            except Exception as e:
                print(f"[WEAVER] ⚠️  Hub save failed: {e}", flush=True)

        # Kill any dangling audio processes
        os.system("pkill -f arecord 2>/dev/null; pkill -f aplay 2>/dev/null")
        print("[WEAVER] All systems offline. Goodbye.", flush=True)


def _setup_signal_handlers(loop):
    """Register graceful shutdown on SIGTERM and SIGINT."""
    import signal

    def _handle_signal(sig):
        print(f"\n[WEAVER] Received {signal.Signals(sig).name} — initiating graceful shutdown...", flush=True)
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_signal, sig)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weaver v4 — unified launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="  python3 weaver.py --heartbeat   # show audio activity dots",
    )
    parser.add_argument(
        "--heartbeat", dest="heartbeat", action="store_true", default=False,
        help="Print mic/speaker activity dots",
    )
    parser.add_argument(
        "--no-heartbeat", dest="heartbeat", action="store_false",
        help="Disable heartbeat dots (default)",
    )
    parser.add_argument(
        "--headless", dest="headless", action="store_true", default=False,
        help="Skip VTV Core — run backend lobes only (nexus, quantum, pineal, lora)",
    )
    args = parser.parse_args()

    try:
        loop = asyncio.new_event_loop()
        _setup_signal_handlers(loop)
        loop.run_until_complete(main(heartbeat=args.heartbeat, headless=args.headless))
    except KeyboardInterrupt:
        print("\n[WEAVER] Interrupted — killing audio processes.")
    finally:
        os.system("pkill -f arecord 2>/dev/null; pkill -f aplay 2>/dev/null")
        loop.close()
