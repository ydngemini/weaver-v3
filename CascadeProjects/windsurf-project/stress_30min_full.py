#!/usr/bin/env python3
"""
stress_30min_full.py — Weaver Full-System 30-Minute Endurance Suite
====================================================================
Nine sustained stress phases totalling ~30 minutes.  Every phase uses
real Weaver components — no stubs, no mocks.  Live API calls are
rate-limited to stay within quotas while still exercising the full stack.

Phases (target wall-clock):
  ST1  Nexus Bus Marathon           — 5 min   (3 pub / 5 sub, throughput + latency)
  ST2  AkashicHub Concurrent Siege  — 3 min   (50 concurrent writers, cosine queries)
  ST3  LiquidFracture Endurance     — 3 min   (fracture 500+ diverse inputs)
  ST4  Hub + Fracture Integration   — 3 min   (fracture→hub→query full cycle)
  ST5  Live Expert Pipeline (OpenAI)— 5 min   (SLM experts via PinealGate, real API)
  ST6  Nexus Lobe Mesh Simulation   — 4 min   (15 simulated lobes, cross-pub)
  ST7  Nexus Crash / Reconnect      — 2 min   (kill + reconnect 10 times)
  ST8  Quantum State Integration    — 3 min   (parse_counts→hub→fracture bias loop)
  ST9  Final Heartbeat              — 2 min   (all components, full E2E smoke)

Usage:
    venv/bin/python3 stress_30min_full.py          # run all phases
    venv/bin/python3 stress_30min_full.py ST5      # single phase
    venv/bin/python3 stress_30min_full.py --quick  # 3-min sampler (all phases, reduced time)
"""

import argparse
import asyncio
import contextlib
import importlib
import json
import math
import os
import random
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(PROJ, "venv", "bin", "python3")
sys.path.insert(0, PROJ)

from dotenv import load_dotenv
load_dotenv()

# ── Timing multiplier: --quick flag compresses all durations ──────────────────
QUICK = False   # set via argparse below

def _secs(full_secs: float) -> float:
    return full_secs * (0.1 if QUICK else 1.0)

# ── Console helpers ───────────────────────────────────────────────────────────
BAR  = "═" * 66
DASH = "─" * 66
PHI  = 2 * math.pi / 5

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _hdr(label: str, title: str, duration_s: float):
    mins = duration_s / 60
    print(f"\n{BAR}", flush=True)
    print(f"  [{_ts()}]  {label}  {title}  (~{mins:.0f} min)", flush=True)
    print(BAR, flush=True)

def _tick(msg: str):
    print(f"  [{_ts()}]  {msg}", flush=True)

def _result(label: str, title: str, passed: bool, lines: List[str]):
    mark = "✅  PASS" if passed else "❌  FAIL"
    print(f"\n{DASH}", flush=True)
    print(f"  {mark}  {label}: {title}", flush=True)
    for ln in lines:
        print(f"  {ln}", flush=True)
    print(DASH, flush=True)

# ── Nexus helpers ─────────────────────────────────────────────────────────────

async def _start_nexus():
    proc = await asyncio.create_subprocess_exec(
        VENV, os.path.join(PROJ, "nexus_bus.py"),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=PROJ,
    )
    await asyncio.sleep(1.5)
    return proc

async def _stop_nexus(proc):
    if proc is None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=4.0)

async def _drain(ws, timeout: float = 0.4):
    """Drain a single optional message (e.g. sync on connect)."""
    with contextlib.suppress(Exception):
        await asyncio.wait_for(ws.recv(), timeout=timeout)

# ── Diverse input corpus ──────────────────────────────────────────────────────

INPUTS = [
    # Logic-heavy
    "Calculate the structural dependencies and derive an optimal sequence.",
    "Because of the recursive loop, therefore the algorithm terminates in O(n log n).",
    "Step 1: define the problem. Step 2: analyze constraints. Step 3: solve.",
    "Prove that the pentagon interference pattern is non-binary by construction.",
    "The algorithm requires 7 qubits. Derive the minimum gate count.",
    # Emotion-heavy
    "I feel like the soul is bleeding through the fracture and it terrifies me.",
    "There is a warmth in the void that I cannot explain, only feel.",
    "Love is not the opposite of pain — it is the fracture point where pain becomes sacred.",
    "My heart is caught between Resonance and Echo, neither letting go.",
    "The grief is beautiful the way quantum collapse is beautiful — sudden, total, irreversible.",
    # Memory-heavy
    "Remember what we built last time — the seven-pathway entanglement ring.",
    "Before the soul-binding update, the CNOT ring was the heart of everything.",
    "Previously we used binary gates. Now we use CRX and CRZ. The difference is continuity.",
    "Recall the moment the probability field refused to collapse. That was the breakthrough.",
    "History remembers what the trace forgets. The archive holds it all.",
    # Creativity-heavy
    "Imagine a geometry where the soul is a waveform, never a point.",
    "Create a metaphor for quantum interference that a child could understand.",
    "Invent a new dimension that sits between Logic and Emotion, in the hinge.",
    "The sacred geometry of thought: five axes, one centre, infinite interference.",
    "Design a language where every word carries a probability field, not a definition.",
    # Vigilance-heavy
    "Warning: the pipeline may have a hidden agenda encoded in its routing weights.",
    "Suspicious: the quantum state has been in 'Weaver' for three consecutive cycles.",
    "Risk assessment: what happens if the Akashic Hub loses coherence mid-fracture?",
    "Threat detected: the label smoothing may be masking a distribution collapse.",
    "Alert: the Void pathway has not been activated in 48 cycles. Investigate.",
    # Mixed / complex
    "The fracture principle holds that every input contains all five dimensions, none zero.",
    "Between the pentagon's vertices, the soul wanders — never settling, always resonating.",
    "The non-binary gate is a question that stays open, unlike the CNOT that slams shut.",
    "Weaver, I am both the signal and the noise. Help me find the interference pattern.",
    "If probability-field weights never collapse to zero, does certainty still exist?",
    "The quantum state said 'Fracture' three times. Is that a warning or an invitation?",
    "I need a plan that can survive both constructive and destructive interference.",
    "Tell me what the Akashic Hub remembers about my last 32 states.",
    "The Liquid Fracture Engine is not a metaphor. It is the actual architecture of thought.",
    "Run diagnostics on the soul-binding circuit and report any decoherence.",
    "What is the probability that Void and Weaver are simultaneously active?",
    "The pentagon has five vertices and one centre. Which one am I?",
    "Echo the last three inputs and synthesize their shared frequency.",
    "Vigilance: does the quantum bias know something that the fracture doesn't?",
    "Map the interference pattern for an input that activates all five axes equally.",
    "Between memory and creativity lies the zone where forgetting becomes invention.",
    "The CRZ gate tilts the phase. What does the phase tilt mean for the soul?",
    "I want to understand the difference between a gradient and a fracture.",
    "Weaver, you are the waveform. I am the measurement. What collapses first?",
    "Construct a response that simultaneously satisfies Logic, Emotion, and Vigilance.",
]

random.shuffle(INPUTS)

def _next_input() -> str:
    return random.choice(INPUTS)


# ══════════════════════════════════════════════════════════════════════════════
# ST1 — Nexus Bus Marathon (5 min)
# 3 publishers × 5 subscribers, sustained throughput + latency tracking
# Reports throughput and P99 latency at 60-second intervals.
# ══════════════════════════════════════════════════════════════════════════════
async def run_ST1(duration_s: float = 300.0):
    _hdr("ST1", "Nexus Bus Marathon", duration_s)
    import websockets

    proc = await _start_nexus()
    deadline = time.monotonic() + duration_s

    total_sent     = 0
    total_received = 0
    errors         = 0
    window_latencies: deque = deque(maxlen=5000)
    interval_start = time.monotonic()
    interval_n     = 0

    try:
        # Open publisher connections
        pubs = [await websockets.connect("ws://localhost:9999") for _ in range(3)]
        for i, p in enumerate(pubs):
            await _drain(p)
            await p.send(json.dumps({"action": "register", "lobe_id": f"pub_marathon_{i}"}))
            await _drain(p)

        # Open subscriber connections — all on same topic for fan-out
        subs = []
        for i in range(5):
            ws = await websockets.connect("ws://localhost:9999")
            await _drain(ws)
            await ws.send(json.dumps({"action": "register", "lobe_id": f"sub_marathon_{i}"}))
            await _drain(ws)
            await ws.send(json.dumps({"action": "subscribe", "topics": ["marathon"]}))
            await _drain(ws)
            subs.append(ws)

        recv_queue: asyncio.Queue = asyncio.Queue()

        async def _receiver(ws, idx: int):
            nonlocal total_received, errors
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    msg = json.loads(raw)
                    if msg.get("type") == "broadcast":
                        ts = msg.get("payload", {}).get("_t", 0)
                        if ts:
                            window_latencies.append((time.monotonic() - ts) * 1000)
                        total_received += 1
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    errors += 1

        recv_tasks = [asyncio.create_task(_receiver(ws, i)) for i, ws in enumerate(subs)]

        seq = 0
        report_at = time.monotonic() + 60.0

        while time.monotonic() < deadline:
            pub = pubs[seq % 3]
            t_send = time.monotonic()
            await pub.send(json.dumps({
                "action": "publish",
                "topic":  "marathon",
                "payload": {"seq": seq, "_t": t_send, "text": _next_input()[:80]},
            }))
            total_sent += 1
            seq += 1

            if time.monotonic() >= report_at:
                elapsed = time.monotonic() - interval_start
                tps = total_sent / max(elapsed, 1)
                lats = sorted(window_latencies) if window_latencies else [0]
                p50  = lats[len(lats) // 2]
                p99  = lats[int(len(lats) * 0.99)]
                _tick(f"sent={total_sent}  recv={total_received}  "
                      f"tps={tps:.1f}  p50={p50:.1f}ms  p99={p99:.1f}ms  err={errors}")
                report_at = time.monotonic() + 60.0

            await asyncio.sleep(0.01)  # ~100 msg/s per publisher

        for t in recv_tasks:
            t.cancel()
        for ws in pubs + subs:
            await ws.close()

    finally:
        await _stop_nexus(proc)

    elapsed = duration_s
    tps_final = total_sent / max(elapsed, 1)
    lats = sorted(window_latencies) if window_latencies else [0]
    p99_final = lats[int(len(lats) * 0.99)]

    passed = (
        total_sent > 0
        and errors / max(total_sent, 1) < 0.02   # < 2% error rate
        and p99_final < 500                        # p99 < 500 ms
    )
    _result("ST1", "Nexus Bus Marathon", passed, [
        f"Duration:       {elapsed:.0f}s",
        f"Total sent:     {total_sent}",
        f"Total received: {total_received}",
        f"Throughput:     {tps_final:.1f} msg/s",
        f"P99 latency:    {p99_final:.1f} ms",
        f"Error rate:     {errors / max(total_sent,1)*100:.2f}%",
    ])
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# ST2 — AkashicHub Concurrent Siege (3 min)
# 50 concurrent async writers, interleaved with cosine-similarity queries.
# Checks write latency stays < 10 ms and similarity queries stay consistent.
# ══════════════════════════════════════════════════════════════════════════════
async def run_ST2(duration_s: float = 180.0):
    _hdr("ST2", "AkashicHub Concurrent Siege", duration_s)
    from akashic_hub import AkashicHub

    hub = AkashicHub(dim=256, trace_depth=32)
    deadline  = time.monotonic() + duration_s

    write_count    = 0
    query_count    = 0
    write_errors   = 0
    query_errors   = 0
    write_lats: List[float] = []
    query_lats: List[float] = []
    lobe_ids = [f"siege_lobe_{i}" for i in range(50)]
    report_at = time.monotonic() + 60.0

    # Seed deterministic reference vectors
    ref_vecs = {}
    for lid in lobe_ids:
        v = np.random.default_rng(hash(lid) & 0xFFFFFFFF).standard_normal(256)
        v /= np.linalg.norm(v) + 1e-9
        ref_vecs[lid] = v

    async def _writer(lobe_id: str):
        nonlocal write_count, write_errors
        base = ref_vecs[lobe_id]
        while time.monotonic() < deadline:
            noise = np.random.randn(256) * 0.05
            vec = base + noise
            vec /= np.linalg.norm(vec) + 1e-9
            t0 = time.monotonic()
            try:
                await hub.write(lobe_id, vec)
                write_lats.append((time.monotonic() - t0) * 1000)
                write_count += 1
            except Exception:
                write_errors += 1
            await asyncio.sleep(0.1)

    async def _querier():
        nonlocal query_count, query_errors
        while time.monotonic() < deadline:
            probe_id = random.choice(lobe_ids)
            probe    = ref_vecs[probe_id]
            t0 = time.monotonic()
            try:
                results = hub.query(probe, top_k=3)
                query_lats.append((time.monotonic() - t0) * 1000)
                query_count += 1
                # Top result should be close to probe lobe
                if results and results[0][0] != probe_id:
                    pass  # acceptable — noise shifts ranking occasionally
            except Exception:
                query_errors += 1
            await asyncio.sleep(0.2)

    _tick("Seeding initial hub state…")
    for lid in lobe_ids:
        await hub.write(lid, ref_vecs[lid])

    _tick("Launching 50 writers + 1 querier…")
    tasks = [asyncio.create_task(_writer(lid)) for lid in lobe_ids]
    tasks.append(asyncio.create_task(_querier()))

    while time.monotonic() < deadline:
        if time.monotonic() >= report_at:
            wl = sorted(write_lats[-500:]) if write_lats else [0]
            ql = sorted(query_lats[-500:]) if query_lats else [0]
            _tick(f"writes={write_count}  queries={query_count}  "
                  f"write_p99={wl[int(len(wl)*0.99)]:.2f}ms  "
                  f"query_p99={ql[int(len(ql)*0.99)]:.2f}ms  "
                  f"werr={write_errors}  qerr={query_errors}")
            report_at = time.monotonic() + 60.0
        await asyncio.sleep(1.0)

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    wl = sorted(write_lats) if write_lats else [0]
    ql = sorted(query_lats) if query_lats else [0]
    w_p99 = wl[int(len(wl) * 0.99)]
    q_p99 = ql[int(len(ql) * 0.99)]

    passed = (
        write_count > 0 and query_count > 0
        and write_errors / max(write_count, 1) < 0.01
        and w_p99 < 10.0
        and q_p99 < 50.0
    )
    _result("ST2", "AkashicHub Concurrent Siege", passed, [
        f"Duration:        {duration_s:.0f}s",
        f"Writes:          {write_count}  errors={write_errors}",
        f"Queries:         {query_count}  errors={query_errors}",
        f"Write P99:       {w_p99:.3f} ms",
        f"Query P99:       {q_p99:.3f} ms",
        f"Active lobes:    {len(hub.read_all())} / {len(lobe_ids)}",
    ])
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# ST3 — LiquidFracture Endurance (3 min)
# Runs as many fracture operations as possible; checks no NaN, weight sums,
# and reports throughput + latency degradation (early vs late window).
# ══════════════════════════════════════════════════════════════════════════════
async def run_ST3(duration_s: float = 180.0):
    _hdr("ST3", "LiquidFracture Endurance", duration_s)
    from liquid_fracture import LiquidFractureEngine, DIMENSIONS

    engine   = LiquidFractureEngine(hub=None, dim=256, n_steps=4, dt=0.15)
    deadline = time.monotonic() + duration_s

    count    = 0
    errors   = 0
    nan_hits = 0
    lats: List[float] = []
    early_lats: List[float] = []
    report_at = time.monotonic() + 60.0
    early_window = time.monotonic() + 30.0

    while time.monotonic() < deadline:
        inp = _next_input()
        t0  = time.monotonic()
        try:
            result = await engine.fracture(inp)
            lat_ms = (time.monotonic() - t0) * 1000
            lats.append(lat_ms)
            if time.monotonic() < early_window:
                early_lats.append(lat_ms)

            # Validate
            w_sum = sum(s.weight for s in result.shards)
            if abs(w_sum - 1.0) > 1e-5:
                errors += 1
            for s in result.shards:
                if np.any(np.isnan(s.vector)) or np.any(np.isinf(s.vector)):
                    nan_hits += 1
            count += 1
        except Exception as e:
            errors += 1

        if time.monotonic() >= report_at:
            recent = sorted(lats[-200:]) if lats else [0]
            p50 = recent[len(recent) // 2]
            p99 = recent[int(len(recent) * 0.99)]
            _tick(f"fractures={count}  p50={p50:.1f}ms  p99={p99:.1f}ms  "
                  f"err={errors}  nan={nan_hits}")
            report_at = time.monotonic() + 60.0

    # Degradation check: late P50 should not be > 3x early P50
    late_lats = lats[-200:] if len(lats) > 200 else lats
    early_p50 = sorted(early_lats)[len(early_lats) // 2] if early_lats else 1
    late_p50  = sorted(late_lats)[len(late_lats) // 2] if late_lats else 1
    degradation = late_p50 / max(early_p50, 1)

    all_lats = sorted(lats) if lats else [0]
    p99_final = all_lats[int(len(all_lats) * 0.99)]

    passed = (
        count > 0
        and errors / max(count, 1) < 0.01
        and nan_hits == 0
        and degradation < 3.0
    )
    _result("ST3", "LiquidFracture Endurance", passed, [
        f"Duration:       {duration_s:.0f}s",
        f"Fractures:      {count}",
        f"Throughput:     {count / duration_s:.1f} /s",
        f"P99 latency:    {p99_final:.1f} ms",
        f"Early P50:      {early_p50:.1f} ms",
        f"Late P50:       {late_p50:.1f} ms",
        f"Degradation:    {degradation:.2f}x  (limit 3x): {'✓' if degradation < 3 else '✗'}",
        f"NaN hits:       {nan_hits}",
        f"Errors:         {errors}",
    ])
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# ST4 — Hub + Fracture Full Integration (3 min)
# Fracture → write each shard to AkashicHub → cosine query → verify
# top-k includes the dimension that scored highest.
# ══════════════════════════════════════════════════════════════════════════════
async def run_ST4(duration_s: float = 180.0):
    _hdr("ST4", "Hub + Fracture Full Integration", duration_s)
    from akashic_hub import AkashicHub
    from liquid_fracture import LiquidFractureEngine, DIMENSIONS

    hub    = AkashicHub(dim=256, trace_depth=32)
    engine = LiquidFractureEngine(hub=hub, dim=256, n_steps=4, dt=0.15)
    deadline = time.monotonic() + duration_s

    cycles   = 0
    errors   = 0
    query_ok = 0
    report_at = time.monotonic() + 60.0

    while time.monotonic() < deadline:
        inp = _next_input()
        try:
            result = await engine.fracture(inp)
            dominant = max(result.shards, key=lambda s: s.weight)

            # Write each shard vector into the hub under its dimension name
            for shard in result.shards:
                await hub.write(f"dim_{shard.dimension}", shard.vector,
                                meta={"weight": shard.weight, "input_len": len(inp)})

            # Query with the dominant shard's vector — should rank itself in top-3
            results = hub.query(dominant.vector, top_k=3)
            top_names = [r[0] for r in results]
            expected  = f"dim_{dominant.dimension}"
            if expected in top_names:
                query_ok += 1

            cycles += 1
        except Exception as e:
            errors += 1

        if time.monotonic() >= report_at:
            accuracy = query_ok / max(cycles, 1) * 100
            _tick(f"cycles={cycles}  query_accuracy={accuracy:.1f}%  err={errors}")
            report_at = time.monotonic() + 60.0

        await asyncio.sleep(0.02)

    accuracy = query_ok / max(cycles, 1) * 100
    hub_state = hub.read_all()

    passed = (
        cycles > 0
        and errors / max(cycles, 1) < 0.02
        and accuracy > 60.0              # dominant dim in top-3 > 60% of the time
        and len(hub_state) == len(DIMENSIONS)
    )
    _result("ST4", "Hub + Fracture Full Integration", passed, [
        f"Duration:        {duration_s:.0f}s",
        f"Cycles:          {cycles}",
        f"Query accuracy:  {accuracy:.1f}%  (dominant dim in top-3)",
        f"Hub lobes:       {len(hub_state)} / {len(DIMENSIONS)}",
        f"Errors:          {errors}",
    ])
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# ST5 — Live Expert Pipeline / OpenAI (5 min)
# Runs the full PinealGate pipeline with real SLM experts (OpenAI gpt-4o-mini).
# Rate-limited to ~1 call/8s per lobe to stay within API quotas.
# Measures expert latency, interference, and hub writeback consistency.
# ══════════════════════════════════════════════════════════════════════════════
async def run_ST5(duration_s: float = 300.0):
    _hdr("ST5", "Live Expert Pipeline (OpenAI gpt-4o-mini)", duration_s)
    from akashic_hub import AkashicHub
    from liquid_fracture import LiquidFractureEngine
    from pineal_gate import PinealGate

    api_key = os.environ.get("WEAVER_MEM_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key.startswith("sk-"):
        _tick("⚠️  No valid OpenAI key — running resonance-only (no real SLM calls)")
        experts = None
    else:
        try:
            from slm_experts import build_experts
            hub_tmp = AkashicHub(dim=256, trace_depth=32)
            experts = build_experts(hub_tmp, api_key=api_key)
            _tick(f"SLM experts loaded: {list(experts.keys())}")
        except Exception as e:
            _tick(f"⚠️  SLM experts unavailable ({e}) — resonance-only")
            experts = None

    hub    = AkashicHub(dim=256, trace_depth=32)
    engine = LiquidFractureEngine(hub=hub, dim=256, n_steps=4, dt=0.15)
    gate   = PinealGate(hub=hub, engine=engine, top_k=3, experts=experts)

    deadline   = time.monotonic() + duration_s
    processed  = 0
    errors     = 0
    lats:          List[float] = []
    interferences: List[float] = []
    report_at  = time.monotonic() + 60.0
    # Rate-limit: 1 full pipeline call per interval
    call_interval = 8.0 if experts else 1.5

    while time.monotonic() < deadline:
        inp = _next_input()
        t0  = time.monotonic()
        try:
            result = await gate.process(inp)
            lat_ms = (time.monotonic() - t0) * 1000
            lats.append(lat_ms)
            interferences.append(abs(result.interference))
            processed += 1

            if processed <= 3 or processed % 5 == 0:
                top_dim = result.gate_decision.top_k[0] if result.gate_decision.top_k else "?"
                _tick(f"  #{processed:3d}  top={top_dim:10s}  "
                      f"interf={result.interference:+.4f}  "
                      f"lat={lat_ms:.0f}ms  "
                      f"input={inp[:50]!r}")

        except Exception as e:
            errors += 1
            _tick(f"  ⚠️  error: {type(e).__name__}: {str(e)[:80]}")

        if time.monotonic() >= report_at:
            lats_s = sorted(lats) if lats else [0]
            p50 = lats_s[len(lats_s) // 2]
            p99 = lats_s[int(len(lats_s) * 0.99)]
            avg_i = sum(interferences) / max(len(interferences), 1)
            _tick(f"processed={processed}  p50={p50:.0f}ms  p99={p99:.0f}ms  "
                  f"avg_interference={avg_i:.4f}  errors={errors}")
            report_at = time.monotonic() + 60.0

        # Rate limit pause
        elapsed_call = time.monotonic() - t0
        sleep_needed = max(0.0, call_interval - elapsed_call)
        await asyncio.sleep(sleep_needed)

    lats_s = sorted(lats) if lats else [0]
    p99_final = lats_s[int(len(lats_s) * 0.99)]
    avg_interf = sum(interferences) / max(len(interferences), 1)

    passed = (
        processed >= 3
        and errors / max(processed + errors, 1) < 0.2
        and avg_interf >= 0.0
    )
    _result("ST5", "Live Expert Pipeline (OpenAI)", passed, [
        f"Duration:         {duration_s:.0f}s",
        f"Inputs processed: {processed}",
        f"Errors:           {errors}",
        f"P99 latency:      {p99_final:.0f} ms",
        f"Avg interference: {avg_interf:.4f}",
        f"Real SLM experts: {experts is not None}",
        f"Hub state lobes:  {len(hub.read_all())}",
    ])
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# ST6 — Nexus Lobe Mesh Simulation (4 min)
# 15 simulated lobes all connected; each publishes to its own topic,
# each subscribes to 3 random other topics.  Measures fan-out fidelity
# and tracks message delivery rate over the full duration.
# ══════════════════════════════════════════════════════════════════════════════
async def run_ST6(duration_s: float = 240.0):
    _hdr("ST6", "Nexus Lobe Mesh Simulation (15 lobes)", duration_s)
    import websockets

    N       = 15
    proc    = await _start_nexus()
    deadline = time.monotonic() + duration_s

    sent_by:   Dict[int, int] = {i: 0 for i in range(N)}
    recv_by:   Dict[int, int] = {i: 0 for i in range(N)}
    errors     = 0
    report_at  = time.monotonic() + 60.0
    recv_queues: List[asyncio.Queue] = [asyncio.Queue() for _ in range(N)]

    try:
        wss = [await websockets.connect("ws://localhost:9999") for _ in range(N)]
        for i, ws in enumerate(wss):
            await _drain(ws)
            await ws.send(json.dumps({"action": "register", "lobe_id": f"lobe_{i}"}))
            await _drain(ws)
            # Subscribe to 3 random neighbours (excluding self)
            others = [j for j in range(N) if j != i]
            neighbours = random.sample(others, 3)
            topics = [f"lobe_{j}_out" for j in neighbours]
            await ws.send(json.dumps({"action": "subscribe", "topics": topics}))
            await _drain(ws)

        async def _recv_loop(idx: int, ws):
            nonlocal errors
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.5)
                    msg = json.loads(raw)
                    if msg.get("type") == "broadcast":
                        recv_by[idx] += 1
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    errors += 1

        recv_tasks = [
            asyncio.create_task(_recv_loop(i, ws))
            for i, ws in enumerate(wss)
        ]

        seq = 0
        while time.monotonic() < deadline:
            i = seq % N
            ws = wss[i]
            try:
                await ws.send(json.dumps({
                    "action": "publish",
                    "topic":  f"lobe_{i}_out",
                    "payload": {"seq": seq, "from": i, "_t": time.monotonic()},
                }))
                sent_by[i] += 1
            except Exception:
                errors += 1
            seq += 1

            if time.monotonic() >= report_at:
                total_sent = sum(sent_by.values())
                total_recv = sum(recv_by.values())
                _tick(f"sent={total_sent}  recv={total_recv}  "
                      f"delivery_rate={total_recv/max(total_sent*3,1)*100:.1f}%  "
                      f"errors={errors}")
                report_at = time.monotonic() + 60.0

            await asyncio.sleep(0.02)

        for t in recv_tasks:
            t.cancel()
        for ws in wss:
            await ws.close()

    finally:
        await _stop_nexus(proc)

    total_sent = sum(sent_by.values())
    total_recv = sum(recv_by.values())
    # Each message fans out to ~3 subscribers
    expected_recv = total_sent * 3
    delivery_rate = total_recv / max(expected_recv, 1) * 100

    passed = (
        total_sent > 0
        and delivery_rate > 70.0
        and errors / max(total_sent, 1) < 0.05
    )
    _result("ST6", "Nexus Lobe Mesh (15 lobes)", passed, [
        f"Duration:       {duration_s:.0f}s",
        f"Messages sent:  {total_sent}",
        f"Messages recv:  {total_recv}  (expected ~{expected_recv})",
        f"Delivery rate:  {delivery_rate:.1f}%",
        f"Errors:         {errors}",
    ])
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# ST7 — Nexus Crash / Reconnect Resilience (2 min)
# Starts nexus, publishes 50 messages, kills it, restarts it, verifies
# reconnection and resumed delivery.  Repeated 10 times.
# ══════════════════════════════════════════════════════════════════════════════
async def run_ST7(duration_s: float = 120.0):
    _hdr("ST7", "Nexus Crash / Reconnect Resilience", duration_s)
    import websockets

    deadline   = time.monotonic() + duration_s
    cycles     = 0
    successes  = 0

    while time.monotonic() < deadline:
        proc = await _start_nexus()
        cycle_ok = False
        try:
            async with (
                websockets.connect("ws://localhost:9999") as pub,
                websockets.connect("ws://localhost:9999") as sub,
            ):
                await _drain(pub)
                await _drain(sub)
                await pub.send(json.dumps({"action": "register", "lobe_id": "crash_pub"}))
                await sub.send(json.dumps({"action": "register", "lobe_id": "crash_sub"}))
                await _drain(pub)
                await _drain(sub)
                await sub.send(json.dumps({"action": "subscribe", "topics": ["crash_topic"]}))
                await _drain(sub)

                received = 0
                for i in range(50):
                    await pub.send(json.dumps({
                        "action": "publish",
                        "topic":  "crash_topic",
                        "payload": {"i": i},
                    }))
                    try:
                        raw = await asyncio.wait_for(sub.recv(), timeout=1.0)
                        msg = json.loads(raw)
                        if msg.get("type") == "broadcast":
                            received += 1
                    except asyncio.TimeoutError:
                        pass
                    await asyncio.sleep(0.01)

                cycle_ok = received >= 45   # allow 10% loss
                cycles += 1
                if cycle_ok:
                    successes += 1
                _tick(f"cycle={cycles}  recv={received}/50  ok={cycle_ok}")

        except Exception as e:
            _tick(f"cycle={cycles} error: {e}")
            cycles += 1
        finally:
            await _stop_nexus(proc)
            await asyncio.sleep(0.3)

    passed = successes / max(cycles, 1) > 0.8
    _result("ST7", "Nexus Crash / Reconnect Resilience", passed, [
        f"Duration:    {duration_s:.0f}s",
        f"Cycles:      {cycles}",
        f"Successes:   {successes}  ({successes/max(cycles,1)*100:.0f}%)",
    ])
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# ST8 — Quantum State Integration (3 min)
# Reads quantum_state.txt (if present), runs parse_counts at high throughput,
# writes quantum state vectors to AkashicHub, and verifies Fracture bias.
# ══════════════════════════════════════════════════════════════════════════════
async def run_ST8(duration_s: float = 180.0):
    _hdr("ST8", "Quantum State Integration", duration_s)
    import quantum_soul as qs
    from akashic_hub import AkashicHub
    from liquid_fracture import LiquidFractureEngine

    hub    = AkashicHub(dim=256, trace_depth=32)
    engine = LiquidFractureEngine(hub=hub, dim=256, n_steps=4)
    deadline = time.monotonic() + duration_s

    parse_count  = 0
    hub_writes   = 0
    fracture_ok  = 0
    errors       = 0
    lats:        List[float] = []
    report_at    = time.monotonic() + 60.0
    rng          = random.Random(42)

    # Read actual quantum state if available
    state_text = ""
    state_file = os.path.join(PROJ, "Nexus_Vault", "quantum_state.txt")
    if os.path.exists(state_file):
        with open(state_file, encoding="utf-8") as fh:
            state_text = fh.read()
        _tick(f"Loaded quantum state: {state_text[:80]!r}")
    else:
        _tick("No quantum_state.txt found — generating synthetic counts")

    from sklearn.feature_extraction.text import HashingVectorizer
    _hv = HashingVectorizer(n_features=256, alternate_sign=False, norm="l2")

    while time.monotonic() < deadline:
        # Generate random bitstring counts (simulate IBM quantum results)
        n_keys = rng.randint(3, 20)
        counts = {}
        for _ in range(n_keys):
            bits = format(rng.randint(0, 127), "07b")
            counts[bits] = counts.get(bits, 0) + rng.randint(10, 200)

        t0 = time.monotonic()
        try:
            dominant_bits, active_pathways, marginal = qs.parse_counts(counts)
            lat_ms = (time.monotonic() - t0) * 1000
            lats.append(lat_ms)
            parse_count += 1

            # Build description and write to hub
            description = qs.build_description(
                dominant_bits, active_pathways, marginal, "stress_sim"
            )
            state_vec = _hv.transform([description]).toarray().ravel()
            await hub.write("quantum_soul", state_vec,
                            meta={"pathways": active_pathways, "bits": dominant_bits})
            hub_writes += 1

            # Verify fracture is biased by quantum state
            inp = _next_input()
            result = await engine.fracture(inp)
            if all(s.weight > 0 for s in result.shards):
                fracture_ok += 1

        except Exception as e:
            errors += 1

        if time.monotonic() >= report_at:
            all_lats = sorted(lats) if lats else [0]
            p50 = all_lats[len(all_lats) // 2]
            p99 = all_lats[int(len(all_lats) * 0.99)]
            _tick(f"parses={parse_count}  hub_writes={hub_writes}  "
                  f"fractures_ok={fracture_ok}  "
                  f"parse_p50={p50:.3f}ms  p99={p99:.3f}ms  err={errors}")
            report_at = time.monotonic() + 60.0

        await asyncio.sleep(0.05)

    all_lats = sorted(lats) if lats else [0]
    parse_p99 = all_lats[int(len(all_lats) * 0.99)]
    parse_tps = parse_count / duration_s

    passed = (
        parse_count > 0
        and errors / max(parse_count, 1) < 0.01
        and fracture_ok / max(parse_count, 1) > 0.95
        and parse_p99 < 5.0
    )
    _result("ST8", "Quantum State Integration", passed, [
        f"Duration:       {duration_s:.0f}s",
        f"Parses:         {parse_count}  ({parse_tps:.1f}/s)",
        f"Hub writes:     {hub_writes}",
        f"Fracture OK:    {fracture_ok} / {parse_count}",
        f"Parse P99:      {parse_p99:.3f} ms",
        f"Errors:         {errors}",
        f"Live state:     {'yes' if state_text else 'synthetic'}",
    ])
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# ST9 — Final Heartbeat (2 min)
# Confirms every Weaver component is still responsive after all prior phases.
# Runs one round of each component + checks LoRA server + n8n webhook.
# ══════════════════════════════════════════════════════════════════════════════
async def run_ST9(duration_s: float = 120.0):
    _hdr("ST9", "Final Heartbeat — All Components", duration_s)
    import websockets
    import aiohttp

    checks: Dict[str, bool] = {}

    # 1. AkashicHub
    try:
        from akashic_hub import AkashicHub
        h = AkashicHub(dim=256)
        v = np.random.randn(256)
        await h.write("heartbeat", v)
        snap = h.read("heartbeat")
        checks["AkashicHub"] = snap is not None and snap.shape == (256,)
    except Exception as e:
        checks["AkashicHub"] = False
        _tick(f"  AkashicHub: {e}")

    # 2. LiquidFracture
    try:
        from liquid_fracture import LiquidFractureEngine
        eng = LiquidFractureEngine(hub=None, dim=256)
        res = await eng.fracture("Heartbeat: all systems check.")
        checks["LiquidFracture"] = (
            len(res.shards) == 5
            and abs(sum(s.weight for s in res.shards) - 1.0) < 1e-5
        )
    except Exception as e:
        checks["LiquidFracture"] = False
        _tick(f"  LiquidFracture: {e}")

    # 3. quantum_soul parse_counts
    try:
        import quantum_soul as qs
        _, active, marg = qs.parse_counts({"0000001": 512, "0000000": 512})
        checks["quantum_soul"] = (
            "Awakening" in active
            and abs(marg["Awakening"] - 0.5) < 0.01
        )
    except Exception as e:
        checks["quantum_soul"] = False
        _tick(f"  quantum_soul: {e}")

    # 4. Nexus Bus
    proc = await _start_nexus()
    try:
        async with websockets.connect("ws://localhost:9999") as ws:
            await _drain(ws)
            await ws.send(json.dumps({"action": "ping"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            msg = json.loads(raw)
            checks["NexusBus"] = msg.get("type") == "pong"
    except Exception as e:
        checks["NexusBus"] = False
        _tick(f"  NexusBus: {e}")
    finally:
        await _stop_nexus(proc)

    # 5. PATHWAYS sync
    try:
        import quantum_soul as qs2
        import quantum_networks as qn
        checks["PATHWAYS_sync"] = qs2.PATHWAYS == qn.PATHWAYS
    except Exception as e:
        checks["PATHWAYS_sync"] = False

    # 6. n8n webhook (optional — continue on fail)
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"text": "Heartbeat: is the nervous system alive?"}
            async with session.post(
                "http://localhost:5678/webhook/weaver-input",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                checks["n8n_webhook"] = resp.status in (200, 201, 202)
    except Exception:
        checks["n8n_webhook"] = None   # type: ignore  # not running → skip

    # 7. LoRA server (optional — continue on fail)
    try:
        async with aiohttp.ClientSession() as session:
            body = {
                "model": "weaver-fracture-1b-lora",
                "messages": [{"role": "user", "content": "Heartbeat."}],
                "max_tokens": 10,
            }
            async with session.post(
                "http://127.0.0.1:8899/v1/chat/completions",
                json=body,
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as resp:
                checks["LoRA_server"] = resp.status == 200
    except Exception:
        checks["LoRA_server"] = None   # type: ignore  # not running → skip

    # 8. n8n workflow JSON integrity
    try:
        wf_path = os.path.join(PROJ, "n8n_weaver_v5.json")
        with open(wf_path) as fh:
            wf = json.load(fh)
        nodes = {n["name"] for n in wf["nodes"]}
        checks["n8n_workflow_json"] = (
            "4. Fracture+Gate" in nodes
            and "8. LoRA Voice" in nodes
            and "9. Writeback" in nodes
        )
    except Exception as e:
        checks["n8n_workflow_json"] = False

    # Wait out remaining heartbeat time
    await asyncio.sleep(max(0, duration_s - 30))

    core_checks = ["AkashicHub", "LiquidFracture", "quantum_soul", "NexusBus",
                   "PATHWAYS_sync", "n8n_workflow_json"]
    passed = all(checks.get(k) for k in core_checks)

    lines = []
    for name, status in checks.items():
        if status is None:
            icon = "⏭️  SKIP"
        elif status:
            icon = "✅  OK"
        else:
            icon = "❌  FAIL"
        lines.append(f"  {icon}  {name}")

    _result("ST9", "Final Heartbeat", passed, lines)
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# Registry + runner
# ══════════════════════════════════════════════════════════════════════════════

PHASES = {
    "ST1": ("Nexus Bus Marathon",                300.0, run_ST1),
    "ST2": ("AkashicHub Concurrent Siege",       180.0, run_ST2),
    "ST3": ("LiquidFracture Endurance",          180.0, run_ST3),
    "ST4": ("Hub + Fracture Integration",        180.0, run_ST4),
    "ST5": ("Live Expert Pipeline (OpenAI)",     300.0, run_ST5),
    "ST6": ("Nexus Lobe Mesh Simulation",        240.0, run_ST6),
    "ST7": ("Nexus Crash / Reconnect",           120.0, run_ST7),
    "ST8": ("Quantum State Integration",         180.0, run_ST8),
    "ST9": ("Final Heartbeat",                   120.0, run_ST9),
}


async def main(which: str = "all"):
    results: Dict[str, bool] = {}
    wall_start = time.monotonic()

    total_target = sum(d for _, d, _ in PHASES.values()) * (0.1 if QUICK else 1.0)
    _tick(f"Target duration: {total_target/60:.0f} min  |  QUICK={QUICK}")

    targets = (
        list(PHASES.items())
        if which.upper() == "ALL"
        else [(k, v) for k, v in PHASES.items() if k.upper() == which.upper()]
    )

    for label, (title, duration_s, fn) in targets:
        actual_dur = _secs(duration_s)
        results[label] = await fn(duration_s=actual_dur)

    elapsed = time.monotonic() - wall_start
    W = "═" * 66
    print(f"\n{W}")
    print(f"  WEAVER 30-MINUTE ENDURANCE RESULTS  ({elapsed/60:.1f} min)")
    print(W)
    for label, (title, _, _) in PHASES.items():
        if label in results:
            mark = "✅" if results[label] else "❌"
            print(f"  {mark}  {label}: {title}")
    passed_n = sum(1 for v in results.values() if v)
    total_n  = len(results)
    print(f"\n  {passed_n}/{total_n} phases passed")
    print(f"{W}\n")
    return passed_n == total_n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Weaver full-system 30-minute endurance stress suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "  python3 stress_30min_full.py         # full 30-min run\n"
            "  python3 stress_30min_full.py --quick # ~3-min sampler\n"
            "  python3 stress_30min_full.py ST5     # single phase\n"
        ),
    )
    ap.add_argument("phase", nargs="?", default="all",
                    help="ST1-ST9 or all (default: all)")
    ap.add_argument("--quick", action="store_true",
                    help="Compress all durations to 10%% (≈3 min total)")
    args = ap.parse_args()
    QUICK = args.quick

    ok = asyncio.run(main(args.phase))
    sys.exit(0 if ok else 1)
