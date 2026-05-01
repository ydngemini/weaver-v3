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
            _build_experts, _lora_main, _quantum_api_serve, _health_dashboard_serve,
            _phone_bridge_serve, _obsidian_bridge_main)


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
     build_experts, lora_main, quantum_api_serve, health_dashboard_serve,
     phone_bridge_serve, obsidian_bridge_main) = _load_modules()

    # ── Akashic Hub: shared zero-latency vector state ──────────────────
    akashic_hub = None
    if AkashicHub is not None:
        akashic_hub = AkashicHub(dim=256, trace_depth=32)
        print("[WEAVER] 🌌 Akashic Hub initialized (dim=256, trace=32).", flush=True)
    else:
        print("[WEAVER] ⚠️  Akashic Hub unavailable.", flush=True)

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

    # 2f. Phone Bridge — Twilio telephony with voice ID + LangChain cortex
    if phone_bridge_serve is not None:
        tasks.append(asyncio.create_task(
            _supervised(phone_bridge_serve, "Phone Bridge", restart_on_crash=True,
                        restart_delay=5.0),
            name="phone_bridge"
        ))
        print("[WEAVER] 📞 Phone Bridge on http://localhost:8765...", flush=True)
    else:
        print("[WEAVER] ⚠️  Phone Bridge skipped (import failed).", flush=True)

    # 2g. Obsidian Bridge — vault file watcher + nexus bus → Obsidian graph
    if obsidian_bridge_main is not None:
        tasks.append(asyncio.create_task(
            _supervised(obsidian_bridge_main, "Obsidian Bridge", restart_on_crash=True,
                        restart_delay=5.0),
            name="obsidian_bridge"
        ))
        print("[WEAVER] 👁️  Obsidian Bridge watching ~/Weaver_Vault...", flush=True)
    else:
        print("[WEAVER] ⚠️  Obsidian Bridge skipped (import failed).", flush=True)

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
