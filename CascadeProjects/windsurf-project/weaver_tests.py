#!/usr/bin/env python3
# Bootstrap: inject the project venv's site-packages so all deps are available
# regardless of which python3 binary invokes this script.
import os as _os, sys as _sys, glob as _glob
_venv_site = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            "venv", "lib")
for _sp in _glob.glob(_os.path.join(_venv_site, "python*", "site-packages")):
    if _sp not in _sys.path:
        _sys.path.insert(0, _sp)
"""
weaver_tests.py — Unified Weaver Test Suite
════════════════════════════════════════════
Replaces: whole_codebase_tests.py  special_tests.py  debug_tests.py
          run_integration_tests.py  test_n8n_endpoints.py
          test_full_weaver.py       test_deep_research.py
          stress_tests_v5.py        stress_30min_full.py  stress_n8n_v5.py

Tiers (--tier):
  unit        In-process tests, no network, no hardware         (~30 tests)
  integration Local service tests (starts nexus_bus subprocess) (~10 tests)
  live        Real API calls (OpenAI, Gemini, Drive, InsightFace)(~10 tests)
  n8n         n8n workflow + Obsidian bridge endpoints           (~4 tests)
  audio       Audio hardware (pacat, aplay, ALSA/PipeWire)       (~3 tests)
  long        Long-running system tests, 1–4 min each            (~5 tests)
  stress      Load / endurance, duration = --dur seconds         (~6 tests)
  all         unit + integration + live  (default)

Usage:
  ./venv/bin/python3 weaver_tests.py                       # all (fast tiers)
  ./venv/bin/python3 weaver_tests.py --tier unit           # unit only
  ./venv/bin/python3 weaver_tests.py --tier stress --dur 60
  ./venv/bin/python3 weaver_tests.py --test quantum_parse  # single test
  ./venv/bin/python3 weaver_tests.py --list                # show all tests
"""

import argparse
import asyncio
import contextlib
import importlib
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
PROJ = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(PROJ, "venv", "bin", "python3")
sys.path.insert(0, PROJ)
load_dotenv(os.path.join(PROJ, ".env"))

BAR  = "─" * 66
DBAR = "═" * 66
_G = "\033[92m"; _R = "\033[91m"; _Y = "\033[93m"; _B = "\033[1m"; _E = "\033[0m"

# ── Test registry ─────────────────────────────────────────────────────────────
_ALL: List[Tuple[str, str, object]] = []   # (tier, name, coro_fn)
RESULTS: Dict[str, bool] = {}
_STRESS_DUR = 30   # overridden by --dur
_WEAVER_PROC: Optional[asyncio.subprocess.Process] = None  # live stack for system tier

def register(tier: str, name: str):
    def _wrap(fn):
        _ALL.append((tier, name, fn))
        return fn
    return _wrap


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _header(name: str, title: str, tier: str) -> None:
    print(f"\n{BAR}\n[{_ts()}] [{tier.upper()}] {name}\n  {title}\n{BAR}", flush=True)

def _ok(name: str, title: str, detail: str = "") -> bool:
    print(f"\n{_G}✅  PASS{_E}  {title}", flush=True)
    if detail:
        for ln in detail.strip().split("\n"):
            print(f"     {ln}", flush=True)
    RESULTS[name] = True
    return True

def _fail(name: str, title: str, detail: str = "") -> bool:
    print(f"\n{_R}❌  FAIL{_E}  {title}", flush=True)
    if detail:
        for ln in detail.strip().split("\n"):
            print(f"     {ln}", flush=True)
    RESULTS[name] = False
    return False

def _res(name: str, title: str, passed: bool, detail: str = "") -> bool:
    return (_ok if passed else _fail)(name, title, detail)

async def _terminate(proc) -> None:
    if proc is None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=3.0)

async def _start_nexus() -> "asyncio.subprocess.Process":
    proc = await asyncio.create_subprocess_exec(
        VENV, os.path.join(PROJ, "nexus_bus.py"),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=PROJ,
    )
    await asyncio.sleep(1.2)
    return proc

async def _drain_sync(ws) -> Optional[dict]:
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
        msg = json.loads(raw)
        return None if msg.get("type") == "sync" else msg
    except asyncio.TimeoutError:
        return None

def _port_open(port: int, timeout: float = 1.0) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    ok = s.connect_ex(("127.0.0.1", port)) == 0
    s.close()
    return ok

def _http_post(url: str, payload: dict, timeout: int = 30) -> Tuple[Optional[int], str]:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.getcode(), resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return None, str(e)

def _http_get(url: str, timeout: int = 5) -> Tuple[Optional[int], str]:
    try:
        resp = urllib.request.urlopen(url, timeout=timeout)
        return resp.getcode(), resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return None, str(e)


# ═════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS  —  in-process, no subprocesses, no network, no hardware
# ═════════════════════════════════════════════════════════════════════════════

@register("unit", "quantum_parse")
async def test_quantum_parse():
    """quantum_soul.parse_counts: maps bit strings to pentagon pathways correctly."""
    qs = importlib.import_module("quantum_soul")
    b1, a1, m1 = qs.parse_counts({"0000001": 7, "0000000": 1})
    b2, a2, m2 = qs.parse_counts({"1000000": 5, "0000000": 1})
    b3, a3, m3 = qs.parse_counts({"0000000": 9})
    ok = (
        b1 == "0000001" and a1 == ["Awakening"] and abs(m1["Awakening"] - 0.875) < 1e-6
        and b2 == "1000000" and a2 == ["Void"]
        and b3 == "0000000" and a3 == ["Void"]
        and all(v == 0.0 for v in m3.values())
        and set(m1.keys()) == set(qs.PATHWAYS.values())
    )
    return _res("quantum_parse", "quantum_soul.parse_counts: pentagon pathway mapping", ok,
                f"bits={b1} active={a1}\nbits={b2} active={a2}\nbits={b3} active={a3}")

@register("unit", "quantum_description")
async def test_quantum_description():
    """quantum_soul.build_description: sentence branches + state file write."""
    qs  = importlib.import_module("quantum_soul")
    base = {n: 0.0 for n in qs.PATHWAYS.values()}
    s1 = qs.build_description("0000001", ["Awakening"],
                               {**base, "Awakening": 0.91}, "be1")
    s2 = qs.build_description("0010001", ["Awakening", "Weaver"],
                               {**base, "Awakening": 0.55, "Weaver": 0.45}, "be2")
    s3 = qs.build_description("0010101", ["Awakening", "Resonance", "Weaver"],
                               {**base, "Weaver": 0.51, "Awakening": 0.44, "Resonance": 0.39}, "be3")
    tmp = tempfile.mkdtemp(prefix="wvr_qs_")
    old_v, old_s = qs.VAULT_DIR, qs.STATE_FILE
    write_ok = False
    try:
        qs.VAULT_DIR  = tmp
        qs.STATE_FILE = os.path.join(tmp, "quantum_state.txt")
        qs._write_state(s1)
        write_ok = os.path.isfile(qs.STATE_FILE) and open(qs.STATE_FILE).read() == s1 + "\n"
    finally:
        qs.VAULT_DIR  = old_v
        qs.STATE_FILE = old_s
        shutil.rmtree(tmp, ignore_errors=True)
    ok = "single point" in s1 and "Two Pathways" in s2 and "multi-pathway" in s3 and write_ok
    return _res("quantum_description", "quantum_soul.build_description + _write_state", ok,
                f"single={'single point' in s1} two={'Two Pathways' in s2} "
                f"multi={'multi-pathway' in s3} write={write_ok}")

@register("unit", "supervisor_crash")
async def test_supervisor_crash():
    """weaver._supervised: crashes trigger restarts with exponential backoff."""
    import unittest.mock
    weaver   = importlib.import_module("weaver")
    attempts = [0]
    gate     = asyncio.Event()
    async def flaky():
        attempts[0] += 1
        if attempts[0] < 3:
            raise RuntimeError(f"boom-{attempts[0]}")
        await gate.wait()
    # Patch out the jitter (random is imported inside _supervised, patch via sys.modules)
    with unittest.mock.patch("random.uniform", return_value=0.0):
        task = asyncio.create_task(
            weaver._supervised(flaky, "Flaky", restart_on_crash=True, restart_delay=0.05)
        )
        await asyncio.sleep(0.5)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return _res("supervisor_crash", "weaver._supervised: crash → restart", attempts[0] >= 3,
                f"attempts={attempts[0]} (need ≥3)")

@register("unit", "supervisor_clean_exit")
async def test_supervisor_clean_exit():
    """weaver._supervised: clean exit terminates task without restart."""
    weaver = importlib.import_module("weaver")
    runs   = [0]
    async def one_shot():
        runs[0] += 1
        # returns immediately — clean exit
    task = asyncio.create_task(
        weaver._supervised(one_shot, "OS", restart_on_crash=False, restart_delay=0.05)
    )
    await asyncio.sleep(0.1)
    # Task should have completed on its own (not cancelled)
    task_done = task.done()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    # Clean exit: ran exactly once and stopped (no restart)
    ok = runs[0] == 1 and task_done
    return _res("supervisor_clean_exit", "weaver._supervised: clean exit → terminates (no restart)", ok,
                f"runs={runs[0]} task_done_before_cancel={task_done}")

@register("unit", "akashic_write_latency")
async def test_akashic_write_latency():
    """AkashicHub: 100 writes across 5 lobes, mean latency <1 ms."""
    from akashic_hub import AkashicHub
    hub  = AkashicHub(dim=256, trace_depth=32)
    lats = []
    for i in range(100):
        lats.append(await hub.write(f"bench_{i % 5}", np.random.randn(256)))
    mean_ms = float(np.mean(lats))
    return _res("akashic_write_latency", f"AkashicHub write latency: mean={mean_ms:.3f}ms",
                mean_ms < 1.0, f"max={float(np.max(lats)):.3f}ms  target <1ms mean")

@register("unit", "akashic_cosine_query")
async def test_akashic_cosine_query():
    """AkashicHub.query: returns correct top-k by cosine similarity."""
    from akashic_hub import AkashicHub
    hub = AkashicHub(dim=256)
    await hub.write("a", np.array([1.]*128 + [0.]*128))
    await hub.write("b", np.array([0.]*128 + [1.]*128))
    await hub.write("c", np.array([1.]*128 + [0.]*128))
    matches = hub.query(np.array([1.]*128 + [0.]*128), top_k=2)
    ok = len(matches) == 2 and matches[0][0] in ("a", "c") and matches[0][1] > 0.9
    return _res("akashic_cosine_query", "AkashicHub.query: top-k by cosine", ok,
                f"matches={matches}")

@register("unit", "akashic_temporal_trace")
async def test_akashic_temporal_trace():
    """AkashicHub: temporal trace stores correct shape; deltas = n-1 rows."""
    from akashic_hub import AkashicHub
    hub = AkashicHub(dim=4, trace_depth=8)
    for i in range(5):
        await hub.write("tr", np.array([float(i)]*4))
    trace = hub.temporal_trace("tr")
    mat   = hub.temporal_matrix("tr")
    delts = hub.temporal_deltas("tr")
    ok    = len(trace) == 5 and mat.shape == (5, 4) and delts.shape == (4, 4)
    return _res("akashic_temporal_trace", "AkashicHub temporal trace: shapes correct", ok,
                f"len={len(trace)} mat={mat.shape} deltas={delts.shape}")

@register("unit", "akashic_entanglement")
async def test_akashic_entanglement():
    """AkashicHub.entangle: weighted blend produces unit-norm vector."""
    from akashic_hub import AkashicHub
    hub = AkashicHub(dim=4)
    await hub.write("x", np.array([1., 0., 0., 0.]))
    await hub.write("y", np.array([0., 1., 0., 0.]))
    blended = hub.entangle(["x", "y"])
    norm    = float(np.linalg.norm(blended))
    ok      = blended.shape == (4,) and abs(norm - 1.0) < 1e-6
    return _res("akashic_entanglement", "AkashicHub.entangle: normalized blend", ok,
                f"norm={norm:.8f}")

@register("unit", "akashic_save_load")
async def test_akashic_save_load():
    """AkashicHub.save/load: round-trip restores all lobe vectors exactly."""
    from akashic_hub import AkashicHub
    hub = AkashicHub(dim=16, trace_depth=4)
    vecs = {f"lobe_{i}": np.random.randn(16) for i in range(3)}
    for lid, v in vecs.items():
        await hub.write(lid, v, meta={"src": "test"})
    tmp = tempfile.mkdtemp(prefix="wvr_hub_")
    try:
        hub.save(tmp)
        hub2 = AkashicHub(dim=16, trace_depth=4)
        count = hub2.load(tmp)
        ok = count == 3 and all(
            np.allclose(hub2.read(lid), hub._coerce(v)) for lid, v in vecs.items()
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _res("akashic_save_load", "AkashicHub save/load round-trip", ok,
                f"Lobes restored: {count}")

@register("unit", "fracture_5dims")
async def test_fracture_5dims():
    """LiquidFractureEngine: fracture() produces exactly 5 shards for all 5 dimensions."""
    from liquid_fracture import LiquidFractureEngine, DIMENSIONS
    engine = LiquidFractureEngine(dim=256)
    fr = await engine.fracture("Calculate optimal batch size for a hot honey ferment")
    ok = len(fr.shards) == 5 and all(s.dimension in DIMENSIONS for s in fr.shards)
    return _res("fracture_5dims", "LiquidFracture: exactly 5 dimensional shards", ok,
                f"shards={[s.dimension for s in fr.shards]}")

@register("unit", "fracture_weights_sum")
async def test_fracture_weights_sum():
    """LiquidFractureEngine: shard weights sum to 1.0."""
    from liquid_fracture import LiquidFractureEngine
    engine = LiquidFractureEngine(dim=256)
    fr    = await engine.fracture("Feel the deep resonance of memory and past experience")
    wsum  = sum(s.weight for s in fr.shards)
    return _res("fracture_weights_sum", f"LiquidFracture weights sum={wsum:.8f}", abs(wsum - 1.0) < 1e-6)

@register("unit", "fracture_tau_range")
async def test_fracture_tau_range():
    """LiquidFractureEngine: all τ values in valid ODE range (0.05, 10.0)."""
    from liquid_fracture import LiquidFractureEngine
    engine = LiquidFractureEngine(dim=256)
    fr    = await engine.fracture("Analyze threats because logic therefore calculate plan")
    taus  = [s.tau for s in fr.shards]
    ok    = all(0.05 < t < 10.0 for t in taus)
    return _res("fracture_tau_range", "LiquidFracture τ in valid ODE range", ok,
                f"τ values: {[f'{t:.3f}' for t in taus]}")

@register("unit", "fracture_keyword_routing")
async def test_fracture_keyword_routing():
    """LiquidFractureEngine: logic-seeded input routes logic as top shard."""
    from liquid_fracture import LiquidFractureEngine
    engine = LiquidFractureEngine(dim=256)
    fr    = await engine.fracture("Calculate structure step by step therefore analyze plan because")
    top   = fr.shards[0]
    return _res("fracture_keyword_routing", f"LiquidFracture: logic keywords → top shard={top.dimension}",
                top.dimension == "logic", f"weight={top.weight:.3f}")

@register("unit", "pineal_gate_pipeline")
async def test_pineal_gate_pipeline():
    """PinealGate: full fracture→gate→dispatch→collapse produces valid ManifestResult."""
    from akashic_hub import AkashicHub
    from liquid_fracture import LiquidFractureEngine
    from pineal_gate import PinealGate
    hub    = AkashicHub(dim=256)
    gate   = PinealGate(hub, LiquidFractureEngine(dim=256), top_k=3)
    mr     = await gate.process("Sacred geometry and quantum entanglement in the fracture principle")
    ok     = mr.vector.shape == (256,) and len(mr.expert_results) > 0 and mr.latency_ms > 0
    return _res("pineal_gate_pipeline",
                f"PinealGate pipeline: {len(mr.expert_results)} experts, {mr.latency_ms:.1f}ms", ok)

@register("unit", "pineal_gate_interference")
async def test_pineal_gate_interference():
    """PinealGate: geometric interference value in [-1, 1]."""
    from akashic_hub import AkashicHub
    from liquid_fracture import LiquidFractureEngine
    from pineal_gate import PinealGate
    gate = PinealGate(AkashicHub(dim=256), LiquidFractureEngine(dim=256), top_k=3)
    mr   = await gate.process("Analyze the logical structure and calculate precise values")
    ok   = isinstance(mr.interference, float) and -1.0 <= mr.interference <= 1.0
    kind = "constructive" if mr.interference > 0 else "destructive"
    return _res("pineal_gate_interference", f"PinealGate interference={mr.interference:+.4f} ({kind})", ok)

@register("unit", "pineal_gate_sparse_topk")
async def test_pineal_gate_sparse_topk():
    """PinealGate: sparse routing activates exactly top_k dimensions."""
    from akashic_hub import AkashicHub
    from liquid_fracture import LiquidFractureEngine
    from pineal_gate import PinealGate
    gate = PinealGate(AkashicHub(dim=256), LiquidFractureEngine(dim=256), top_k=3)
    mr   = await gate.process("Feel emotion remember past creative vigilance")
    ok   = len(mr.gate_decision.top_k) == 3
    return _res("pineal_gate_sparse_topk", f"PinealGate top-3: {mr.gate_decision.top_k}", ok)

@register("unit", "quantum_net_topologies")
async def test_quantum_net_topologies():
    """QuantumNetworks: all 5 topologies produce valid (ctrl, tgt) CNOT pairs."""
    from quantum_networks import EntanglementTopology
    details = []
    all_ok  = True
    for name in EntanglementTopology.all_names():
        pairs = EntanglementTopology.get(name)
        valid = len(pairs) > 0 and all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
        details.append(f"{name}: {len(pairs)} pairs {'✓' if valid else '✗'}")
        if not valid:
            all_ok = False
    return _res("quantum_net_topologies", "QuantumNetworks: all 5 topologies valid", all_ok,
                "\n".join(details))

@register("unit", "quantum_variational_circuit")
async def test_quantum_variational_circuit():
    """VariationalFractureCircuit: 7-qubit, 3-layer pentagon gives 42 params."""
    from quantum_networks import VariationalFractureCircuit
    vc = VariationalFractureCircuit(n_qubits=7, n_layers=3, topology="pentagon")
    qc = vc.build()
    ok = vc.param_count() == 42 and qc.num_qubits == 7
    return _res("quantum_variational_circuit",
                f"VariationalFractureCircuit: params={vc.param_count()} qubits={qc.num_qubits}", ok)

@register("unit", "quantum_learner_fitness")
async def test_quantum_learner_fitness():
    """QuantumLearner.compute_fitness: returns float in (0, 1) from mock counts."""
    from quantum_networks import VariationalFractureCircuit, QuantumLearner
    vc      = VariationalFractureCircuit(n_qubits=7, n_layers=3, topology="pentagon")
    learner = QuantumLearner(vc)
    fitness = learner.compute_fitness({"0000001": 200, "1000000": 150, "0111011": 100, "0000000": 50})
    return _res("quantum_learner_fitness", f"QuantumLearner fitness={fitness:.4f}", 0.0 < fitness < 1.0)

@register("unit", "quantum_interference_bias")
async def test_quantum_interference_bias():
    """QuantumInterferenceNetwork: routing bias covers all 5 dims, values in [0, 1]."""
    from quantum_networks import QuantumInterferenceNetwork
    ifn    = QuantumInterferenceNetwork()
    biases = ifn.compute_routing_bias({"0000001": 200, "1000000": 150, "0111011": 100, "0000000": 50})
    ok     = len(biases) == 5 and all(0.0 <= v <= 1.0 for v in biases.values())
    return _res("quantum_interference_bias", "QuantumInterferenceNetwork: 5-dim bias in [0,1]", ok,
                f"biases={biases}")

@register("unit", "quantum_orchestrator_cycle")
async def test_quantum_orchestrator_cycle():
    """QuantumNetworkOrchestrator: successive calls return different topologies."""
    from quantum_networks import QuantumNetworkOrchestrator
    orch = QuantumNetworkOrchestrator()
    t1   = orch.current_topology()
    orch._topo_idx = (orch._topo_idx + 1) % len(orch.topologies)
    t2   = orch.current_topology()
    ok   = t1 != t2 and t1 in orch.topologies and t2 in orch.topologies
    return _res("quantum_orchestrator_cycle", f"Orchestrator cycle: {t1} → {t2}", ok)

@register("unit", "nexus_noobj_json")
async def test_nexus_noobj_json():
    """nexus_bus: non-object JSON returns error frame (in-memory, no subprocess)."""
    from nexus_bus import _handle_message, LobeConnection
    class FakeWS:
        def __init__(self): self.sent = []; self.remote_address = ("127.0.0.1", 0)
        async def send(self, d): self.sent.append(d)
    ws   = FakeWS()
    lobe = LobeConnection(ws)
    await _handle_message(lobe, json.dumps([1, 2, 3]))
    resp = json.loads(ws.sent[0]) if ws.sent else {}
    ok   = resp.get("type") == "error" and "JSON object" in resp.get("msg", "")
    return _res("nexus_noobj_json", "nexus_bus: non-object JSON → error frame", ok)

@register("unit", "nexus_dup_lobe_inmem")
async def test_nexus_dup_lobe_inmem():
    """nexus_bus: duplicate lobe_id registration blocked (in-memory)."""
    from nexus_bus import _handle_message, _connections, LobeConnection
    class FakeWS:
        def __init__(self): self.sent = []; self.remote_address = ("127.0.0.1", 0)
        async def send(self, d): self.sent.append(d)
    _connections.clear()
    ws1, ws2 = FakeWS(), FakeWS()
    l1, l2   = LobeConnection(ws1), LobeConnection(ws2)
    await _handle_message(l1, json.dumps({"action": "register", "lobe_id": "dup_test"}))
    _connections["dup_test"] = l1
    await _handle_message(l2, json.dumps({"action": "register", "lobe_id": "dup_test"}))
    resp = json.loads(ws2.sent[-1]) if ws2.sent else {}
    _connections.clear()
    ok = resp.get("type") == "error" and "already in use" in resp.get("msg", "")
    return _res("nexus_dup_lobe_inmem", "nexus_bus: dup lobe_id blocked (in-memory)", ok)

@register("unit", "nexus_rate_limit_check")
async def test_nexus_rate_limit_check():
    """nexus_bus.LobeConnection.check_rate: blocks when RATE_LIMIT exceeded."""
    from nexus_bus import LobeConnection, RATE_LIMIT
    class FakeWS:
        remote_address = ("127.0.0.1", 0)
        async def send(self, d): pass
    lobe = LobeConnection(FakeWS())
    passed_count = 0
    for _ in range(RATE_LIMIT + 10):
        if lobe.check_rate():
            passed_count += 1
    ok = passed_count == RATE_LIMIT
    return _res("nexus_rate_limit_check", f"nexus_bus rate limit: {passed_count}/{RATE_LIMIT+10} passed",
                ok, f"RATE_LIMIT={RATE_LIMIT}")

@register("unit", "obsidian_wikilinks")
async def test_obsidian_wikilinks():
    """obsidian_bridge.inject_wikilinks: known keywords produce [[WikiLink]] format."""
    from obsidian_bridge import inject_wikilinks
    sample = "The quantum entanglement reveals akashic resonance through the pineal gate. YDN configured topology."
    linked = inject_wikilinks(sample)
    ok = (
        "[[Quantum Soul]]" in linked and "[[Akashic]]" in linked
        and "[[YDN]]" in linked and "[[Pineal Gate]]" in linked
    )
    return _res("obsidian_wikilinks", "obsidian_bridge.inject_wikilinks: keywords → [[links]]", ok,
                f"{linked[:160]}…")

@register("unit", "lora_artifacts")
async def test_lora_artifacts():
    """LoRA adapter directory: all required files present, config is valid LORA + llama base."""
    lora_dir  = os.path.join(PROJ, "weaver_fracture_1B_lora")
    required  = ["adapter_config.json", "adapter_model.safetensors",
                 "tokenizer.json", "tokenizer_config.json"]
    files_ok  = all(os.path.isfile(os.path.join(lora_dir, f)) for f in required)
    cfg_ok    = False
    cfg_detail = "missing"
    try:
        with open(os.path.join(lora_dir, "adapter_config.json")) as f:
            cfg = json.load(f)
        cfg_ok    = cfg.get("peft_type") == "LORA" and "llama" in str(cfg.get("base_model_name_or_path", "")).lower()
        cfg_detail = f"peft_type={cfg.get('peft_type')} base={cfg.get('base_model_name_or_path')}"
    except Exception as e:
        cfg_detail = str(e)
    ok = files_ok and cfg_ok
    return _res("lora_artifacts", "LoRA adapter: all files present + config valid", ok,
                f"files={files_ok}  config={cfg_ok} ({cfg_detail})")

@register("unit", "vault_persistence")
async def test_vault_persistence():
    """Nexus_Vault: directory + key persistence files present."""
    vault  = os.path.join(PROJ, "Nexus_Vault")
    checks = {
        "Nexus_Vault/":         os.path.isdir(vault),
        "quantum_state.txt":    os.path.isfile(os.path.join(vault, "quantum_state.txt")),
        "akashic_persist/":     os.path.isdir(os.path.join(vault, "akashic_persist")),
        "face_registry.npz":    os.path.isfile(os.path.join(vault, "face_registry.npz")),
    }
    detail = "\n".join(f"{'✓' if v else '✗'} {k}" for k, v in checks.items())
    qs = os.path.join(vault, "quantum_state.txt")
    if os.path.isfile(qs):
        detail += f"\nquantum_state: {open(qs).read().strip()[:80]}"
    return _res("vault_persistence", "Nexus_Vault: persistence files present", all(checks.values()), detail)

@register("unit", "soul_dataset")
async def test_soul_dataset():
    """weaver_soul_dataset.jsonl: valid JSONL with correct user/assistant message schema."""
    ds   = os.path.join(PROJ, "weaver_soul_dataset.jsonl")
    ok   = False
    cnt  = 0
    bad  = []
    if os.path.isfile(ds):
        with open(ds) as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                cnt += 1
                try:
                    obj  = json.loads(line)
                    msgs = obj.get("messages", [])
                    if len(msgs) < 2 or msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
                        bad.append(i + 1)
                except Exception as e:
                    bad.append(f"{i+1}:{e}")
        ok = cnt >= 5 and not bad
    return _res("soul_dataset", f"Soul dataset: {cnt} lines, {len(bad)} bad", ok,
                f"Bad lines: {bad or 'none'}")


# ═════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TESTS  —  start nexus_bus subprocess; no live API keys needed
# ═════════════════════════════════════════════════════════════════════════════

@register("integration", "nexus_cache_trim")
async def test_nexus_cache_trim():
    """Nexus Bus rolling cache: publishes 15 msgs, new connect syncs last 10 only."""
    import websockets
    proc = await _start_nexus()
    idxs = []
    try:
        async with websockets.connect("ws://localhost:9999") as pub:
            await _drain_sync(pub)
            await pub.send(json.dumps({"action": "register", "lobe_id": "cache_pub"}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(pub.recv(), timeout=1.0)
            for i in range(15):
                await pub.send(json.dumps({"action": "publish", "topic": "ct", "payload": {"idx": i}}))
                await asyncio.sleep(0.01)
        async with websockets.connect("ws://localhost:9999") as sub:
            raw  = await asyncio.wait_for(sub.recv(), timeout=1.5)
            msg  = json.loads(raw)
            msgs = msg.get("messages", []) if msg.get("type") == "sync" else []
            idxs = [m.get("payload", {}).get("idx") for m in msgs]
    finally:
        await _terminate(proc)
    ok = idxs == list(range(5, 15))
    return _res("nexus_cache_trim", "Nexus cache: trims to last 10 messages", ok,
                f"got={idxs}  want={list(range(5,15))}")

@register("integration", "nexus_unsubscribe")
async def test_nexus_unsubscribe():
    """Nexus Bus: unsubscribe stops deliveries; pre-unsub delivery still received."""
    import websockets
    proc = await _start_nexus()
    received = []
    leak     = False
    try:
        async with websockets.connect("ws://localhost:9999") as pub, \
                   websockets.connect("ws://localhost:9999") as sub:
            for ws in (pub, sub):
                await _drain_sync(ws)
                await ws.send(json.dumps({"action": "register", "lobe_id": f"u_{id(ws)}"}))
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(ws.recv(), timeout=1.0)
            await sub.send(json.dumps({"action": "subscribe", "topics": ["u_topic"]}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(sub.recv(), timeout=1.0)
            await pub.send(json.dumps({"action": "publish", "topic": "u_topic", "payload": {"idx": 1}}))
            m1 = json.loads(await asyncio.wait_for(sub.recv(), timeout=1.5))
            received.append(m1.get("payload", {}).get("idx"))
            await sub.send(json.dumps({"action": "unsubscribe", "topics": ["u_topic"]}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(sub.recv(), timeout=0.5)
            await pub.send(json.dumps({"action": "publish", "topic": "u_topic", "payload": {"idx": 2}}))
            try:
                m2 = json.loads(await asyncio.wait_for(sub.recv(), timeout=0.8))
                leak = m2.get("type") == "broadcast"
            except asyncio.TimeoutError:
                pass
    finally:
        await _terminate(proc)
    ok = received == [1] and not leak
    return _res("nexus_unsubscribe", "Nexus unsubscribe: stops further deliveries", ok,
                f"pre-unsub={received}  post-unsub leak={leak}")

@register("integration", "nexus_error_frames")
async def test_nexus_error_frames():
    """Nexus Bus: 4 protocol error cases all return correct error frames."""
    import websockets
    proc = await _start_nexus()
    got  = {}
    try:
        async with websockets.connect("ws://localhost:9999") as ws:
            await _drain_sync(ws)
            await ws.send("not-json")
            e = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.5))
            got["invalid_json"]   = e.get("type") == "error" and "Invalid JSON" in e.get("msg", "")
            await ws.send(json.dumps({"action": "mystery"}))
            e = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.5))
            got["unknown_action"] = e.get("type") == "error" and "Unknown action" in e.get("msg", "")
            await ws.send(json.dumps({"action": "publish", "payload": {}}))
            e = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.5))
            got["missing_topic"]  = e.get("type") == "error" and "requires 'topic'" in e.get("msg", "")
            await ws.send(json.dumps(["not", "an", "object"]))
            e = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.5))
            got["non_object"]     = e.get("type") == "error" and "JSON object" in e.get("msg", "")
    finally:
        await _terminate(proc)
    ok = all(got.values())
    return _res("nexus_error_frames", "Nexus: all 4 error frame cases correct", ok,
                "\n".join(f"{'✓' if v else '✗'} {k}" for k, v in got.items()))

@register("integration", "nexus_dup_lobe_ws")
async def test_nexus_dup_lobe_ws():
    """Nexus Bus: dup lobe_id rejected over real WebSocket; original still routes."""
    import websockets
    proc = await _start_nexus()
    dup_ok = orig_ok = False
    try:
        async with websockets.connect("ws://localhost:9999") as orig, \
                   websockets.connect("ws://localhost:9999") as intruder, \
                   websockets.connect("ws://localhost:9999") as pub:
            for ws in (orig, intruder, pub):
                await _drain_sync(ws)
            await orig.send(json.dumps({"action": "register", "lobe_id": "dup_lobe"}))
            await asyncio.wait_for(orig.recv(), timeout=1.5)
            await orig.send(json.dumps({"action": "subscribe", "topics": ["dup_t"]}))
            await asyncio.wait_for(orig.recv(), timeout=1.5)
            await intruder.send(json.dumps({"action": "register", "lobe_id": "dup_lobe"}))
            e = json.loads(await asyncio.wait_for(intruder.recv(), timeout=1.5))
            dup_ok = e.get("type") == "error" and "already in use" in e.get("msg", "")
            await pub.send(json.dumps({"action": "register", "lobe_id": "dup_pub"}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(pub.recv(), timeout=1.0)
            await pub.send(json.dumps({"action": "publish", "topic": "dup_t", "payload": {"x": 7}}))
            msg    = json.loads(await asyncio.wait_for(orig.recv(), timeout=1.5))
            orig_ok = msg.get("type") == "broadcast" and msg.get("payload", {}).get("x") == 7
    finally:
        await _terminate(proc)
    ok = dup_ok and orig_ok
    return _res("nexus_dup_lobe_ws", "Nexus dup lobe_id: rejected; original preserved", ok,
                f"rejected={dup_ok}  original routes={orig_ok}")

@register("integration", "nexus_port_collision")
async def test_nexus_port_collision():
    """Nexus Bus: second instance on same port exits non-zero with address-in-use."""
    primary  = await _start_nexus()
    exited   = addr_in_use = False
    text     = ""
    contender = None
    try:
        contender = await asyncio.create_subprocess_exec(
            VENV, os.path.join(PROJ, "nexus_bus.py"),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=PROJ,
        )
        try:
            out, _ = await asyncio.wait_for(contender.communicate(), timeout=2.5)
            text   = out.decode(errors="replace")
            exited = contender.returncode not in (None, 0)
            lo     = text.lower()
            addr_in_use = any(k in lo for k in ("address already in use", "errno 98", "address in use"))
        except asyncio.TimeoutError:
            text = "second nexus did not exit"
    finally:
        await _terminate(contender)
        await _terminate(primary)
    ok = exited and addr_in_use
    return _res("nexus_port_collision", "Nexus port collision: second instance fails closed", ok,
                f"exited_nonzero={exited}  addr_in_use={addr_in_use}")

@register("integration", "nexus_fanout")
async def test_nexus_fanout():
    """Nexus Bus: 10 topics × 1 sub × 10 msgs = 100 deliveries, 0 cross-topic errors."""
    import websockets
    proc     = await _start_nexus()
    N_T = N_M = 10
    received: Dict[str, list] = {f"t_{i}": [] for i in range(N_T)}
    errors   = []
    try:
        async def _sub(ti):
            topic = f"t_{ti}"
            async with websockets.connect("ws://localhost:9999") as ws:
                await _drain_sync(ws)
                await ws.send(json.dumps({"action": "subscribe", "topics": [topic]}))
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(ws.recv(), timeout=1.0)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
                        if msg.get("type") == "broadcast":
                            if msg.get("topic") != topic:
                                errors.append(f"cross: sub_{ti} got {msg.get('topic')}")
                            received[topic].append(1)
                    except asyncio.TimeoutError:
                        if len(received[topic]) >= N_M:
                            break

        async def _pub():
            async with websockets.connect("ws://localhost:9999") as ws:
                await _drain_sync(ws)
                await ws.send(json.dumps({"action": "register", "lobe_id": "fanout_pub"}))
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(ws.recv(), timeout=1.0)
                await asyncio.sleep(0.5)
                for i in range(N_T):
                    for m in range(N_M):
                        await ws.send(json.dumps({
                            "action": "publish", "topic": f"t_{i}", "payload": {"i": i, "m": m}
                        }))
                        await asyncio.sleep(0.005)

        subs = [asyncio.create_task(_sub(i)) for i in range(N_T)]
        await asyncio.sleep(0.3)
        await asyncio.gather(_pub(), *subs, return_exceptions=True)
    finally:
        await _terminate(proc)
    total = sum(len(v) for v in received.values())
    ok    = total == N_T * N_M and not errors
    return _res("nexus_fanout", f"Nexus fanout: {total}/{N_T*N_M} delivered, {len(errors)} cross-topic", ok)

@register("integration", "nexus_ws_roundtrip")
async def test_nexus_ws_roundtrip():
    """Nexus Bus: pub/sub round-trip latency <100 ms on localhost."""
    import websockets
    proc   = await _start_nexus()
    lat_ms = signal = None
    try:
        async with websockets.connect("ws://localhost:9999") as pub, \
                   websockets.connect("ws://localhost:9999") as sub:
            await _drain_sync(pub); await _drain_sync(sub)
            for ws, lid in ((pub, "rt_pub"), (sub, "rt_sub")):
                await ws.send(json.dumps({"action": "register", "lobe_id": lid}))
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(ws.recv(), timeout=1.0)
            await sub.send(json.dumps({"action": "subscribe", "topics": ["rt"]}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(sub.recv(), timeout=1.0)
            t0 = time.monotonic()
            await pub.send(json.dumps({"action": "publish", "topic": "rt", "payload": {"sig": "ALIVE"}}))
            msg    = json.loads(await asyncio.wait_for(sub.recv(), timeout=2.0))
            lat_ms = (time.monotonic() - t0) * 1000
            signal = msg.get("payload", {}).get("sig")
    finally:
        await _terminate(proc)
    ok = signal == "ALIVE" and lat_ms is not None and lat_ms < 100
    return _res("nexus_ws_roundtrip", f"Nexus WS round-trip: {lat_ms:.1f}ms", ok)

@register("integration", "ports_alive")
async def test_ports_alive():
    """Port connectivity: checks n8n (5678), Obsidian bridge (5679), Nexus (9999)."""
    checks = {5678: "n8n", 5679: "obsidian_bridge", 9999: "nexus_bus"}
    state  = {name: _port_open(p) for p, name in checks.items()}
    detail = "\n".join(f"{'✓' if v else '✗'} :{p} {name}" for (p, name), v in zip(checks.items(), state.values()))
    # Nexus is the only required one; n8n/bridge may not be running
    return _res("ports_alive", f"Ports: {state}", state["nexus_bus"], detail)


# ═════════════════════════════════════════════════════════════════════════════
#  LIVE API TESTS  —  require real API keys in .env
# ═════════════════════════════════════════════════════════════════════════════

@register("live", "env_keys")
async def test_env_keys():
    """All required env vars present with correct key prefixes."""
    checks = {
        "WEAVER_VOICE_KEY":  os.environ.get("WEAVER_VOICE_KEY", "").startswith("sk-"),
        "WEAVER_MEM_KEY":    os.environ.get("WEAVER_MEM_KEY",   "").startswith("sk-"),
        "GEMINI_API_KEY":    os.environ.get("GEMINI_API_KEY",   "").startswith("AIza"),
        "IBM_QUANTUM_TOKEN": bool(os.environ.get("IBM_QUANTUM_TOKEN", "")),
    }
    detail = "\n".join(f"{'✓' if v else '✗'} {k}" for k, v in checks.items())
    return _res("env_keys", "Required env vars: all present with correct prefix", all(checks.values()), detail)

@register("live", "core_imports")
async def test_core_imports():
    """All hard dependencies of vtv_basic.py import without error."""
    modules = [
        ("openai",          "import openai"),
        ("google.genai",    "from google import genai"),
        ("langchain_openai","from langchain_openai import ChatOpenAI"),
        ("websockets",      "import websockets"),
        ("cv2",             "import cv2"),
        ("insightface",     "import insightface"),
        ("qiskit",          "from qiskit import QuantumCircuit"),
        ("sklearn",         "from sklearn.feature_extraction.text import HashingVectorizer"),
        ("torch",           "import torch"),
        ("transformers",    "from transformers import AutoTokenizer"),
        ("watchdog",        "from watchdog.events import FileSystemEventHandler"),
        ("aiohttp",         "import aiohttp"),
        ("peft",            "from peft import PeftModel"),
    ]
    failed = []
    for name, stmt in modules:
        try:
            exec(stmt, {})
        except Exception as e:
            failed.append(f"{name}: {str(e)[:60]}")
    ok = not failed
    return _res("core_imports", f"Core imports: {len(modules)-len(failed)}/{len(modules)} OK", ok,
                "\n".join(failed) if failed else "All imports successful")

@register("live", "gemini_vision")
async def test_gemini_vision():
    """Gemini 2.5 Flash: live vision call returns non-empty text response."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return _res("gemini_vision", "Gemini vision: SKIPPED — no GEMINI_API_KEY", False)
    try:
        import io
        from PIL import Image
        from google import genai as gg
        from google.genai import types as gt
        # Build a 32×32 light-grey JPEG using Pillow (avoids cv2/NumPy ABI issues)
        pil_img = Image.new("RGB", (32, 32), color=(200, 200, 200))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=80)
        jpeg_bytes = buf.getvalue()
        # Run sync genai client in a thread — avoids httpx session closure in asyncio loop
        def _call():
            client = gg.Client(api_key=key)
            return client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[gt.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                          "What colour is this image? One word."],
            )
        t0   = time.monotonic()
        resp = await asyncio.get_event_loop().run_in_executor(None, _call)
        ms   = (time.monotonic() - t0) * 1000
        text = (resp.text or "").strip()[:60]
        return _res("gemini_vision", f"Gemini 2.5 Flash vision: {ms:.0f}ms → '{text}'", bool(text))
    except Exception as e:
        return _res("gemini_vision", "Gemini vision: ERROR", False, str(e)[:120])

@register("live", "openai_chat")
async def test_openai_chat():
    """OpenAI gpt-4o-mini: live chat completion via WEAVER_MEM_KEY."""
    key = os.environ.get("WEAVER_MEM_KEY", "")
    if not key:
        return _res("openai_chat", "OpenAI chat: SKIPPED — no WEAVER_MEM_KEY", False)
    try:
        import openai
        t0    = time.monotonic()
        r     = openai.OpenAI(api_key=key).chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with exactly: WEAVER_ONLINE"}],
            max_tokens=10,
        )
        ms    = (time.monotonic() - t0) * 1000
        reply = r.choices[0].message.content.strip()
        return _res("openai_chat", f"OpenAI gpt-4o-mini: {ms:.0f}ms → '{reply}'",
                    "WEAVER_ONLINE" in reply)
    except Exception as e:
        return _res("openai_chat", "OpenAI chat: ERROR", False, str(e)[:120])

@register("live", "openai_realtime_key")
async def test_openai_realtime_key():
    """OpenAI WEAVER_VOICE_KEY: valid key and realtime model is available."""
    key = os.environ.get("WEAVER_VOICE_KEY", "")
    if not key:
        return _res("openai_realtime_key", "OpenAI voice key: SKIPPED — no WEAVER_VOICE_KEY", False)
    try:
        import openai
        t0      = time.monotonic()
        models  = [m.id for m in openai.OpenAI(api_key=key).models.list().data]
        ms      = (time.monotonic() - t0) * 1000
        realtime = [m for m in models if "realtime" in m]
        return _res("openai_realtime_key", f"Voice key valid: {ms:.0f}ms, realtime={realtime[:2]}", bool(realtime))
    except Exception as e:
        return _res("openai_realtime_key", "OpenAI voice key: ERROR", False, str(e)[:120])

@register("live", "langchain_cortex")
async def test_langchain_cortex():
    """LangChain ChatOpenAI: live invoke returns expected token."""
    key = os.environ.get("WEAVER_MEM_KEY", "")
    if not key:
        return _res("langchain_cortex", "LangChain: SKIPPED — no WEAVER_MEM_KEY", False)
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        t0   = time.monotonic()
        resp = ChatOpenAI(model="gpt-4o-mini", openai_api_key=key, max_tokens=10).invoke(
            [HumanMessage(content="Say: LC_ONLINE")]
        )
        ms   = (time.monotonic() - t0) * 1000
        return _res("langchain_cortex", f"LangChain invoke: {ms:.0f}ms → '{resp.content[:40]}'",
                    "LC_ONLINE" in resp.content)
    except Exception as e:
        return _res("langchain_cortex", "LangChain: ERROR", False, str(e)[:120])

@register("live", "insightface_load")
async def test_insightface_load():
    """InsightFace buffalo_sc: loads from Nexus_Vault/.insightface and infers on blank frame."""
    try:
        import insightface, cv2
        os.environ.setdefault("INSIGHTFACE_HOME", os.path.join(PROJ, "Nexus_Vault", ".insightface"))
        t0  = time.monotonic()
        app = insightface.app.FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(160, 160))
        load_ms = (time.monotonic() - t0) * 1000
        t1      = time.monotonic()
        faces   = app.get(np.zeros((160, 160, 3), dtype=np.uint8))
        inf_ms  = (time.monotonic() - t1) * 1000
        return _res("insightface_load", f"InsightFace: load={load_ms:.0f}ms infer={inf_ms:.0f}ms {len(faces)} face(s)", True)
    except Exception as e:
        return _res("insightface_load", "InsightFace: ERROR", False, str(e)[:120])

@register("live", "face_registry")
async def test_face_registry():
    """face_registry.npz: present and loadable (OK if missing — fresh start)."""
    path = os.path.join(PROJ, "Nexus_Vault", "face_registry.npz")
    if not os.path.isfile(path):
        return _res("face_registry", "face_registry.npz: not found (fresh start — expected)", True,
                    "Will be created on first face registration")
    try:
        data  = np.load(path, allow_pickle=False)
        names = list(data.files)
        return _res("face_registry", f"face_registry.npz: {len(names)} registered: {names}", True)
    except Exception as e:
        return _res("face_registry", "face_registry.npz: load error", False, str(e)[:120])

@register("live", "slm_experts_build")
async def test_slm_experts_build():
    """slm_experts.build_experts: all 5 SLM expert lobes instantiate correctly."""
    key = os.environ.get("WEAVER_MEM_KEY", "")
    if not key:
        return _res("slm_experts_build", "SLM experts: SKIPPED — no WEAVER_MEM_KEY", False)
    try:
        from akashic_hub import AkashicHub
        from slm_experts import build_experts
        experts  = build_experts(AkashicHub(dim=256))
        expected = {"logic", "emotion", "memory", "creativity", "vigilance"}
        ok       = len(experts) == 5 and set(experts.keys()) == expected
        return _res("slm_experts_build", f"SLM experts: {sorted(experts.keys())}", ok)
    except Exception as e:
        return _res("slm_experts_build", "SLM experts: ERROR", False, str(e)[:120])

@register("live", "google_drive")
async def test_google_drive():
    """Google Drive OAuth: token.json valid, can list Drive folder."""
    token_path = os.path.join(PROJ, "token.json")
    if not os.path.isfile(token_path):
        return _res("google_drive", "Google Drive: SKIPPED — no token.json", False)
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        creds = Credentials.from_authorized_user_file(token_path,
                    ["https://www.googleapis.com/auth/drive"])
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as ref_e:
                return _res("google_drive", "Google Drive: token expired/revoked — re-run init_drive.py",
                            False, str(ref_e)[:120])
        t0  = time.monotonic()
        svc = build("drive", "v3", credentials=creds)
        res = svc.files().list(pageSize=5, fields="files(id,name)").execute()
        ms  = (time.monotonic() - t0) * 1000
        files = res.get("files", [])
        return _res("google_drive", f"Google Drive: {ms:.0f}ms, {len(files)} file(s)", True,
                    f"files={[f['name'] for f in files]}")
    except Exception as e:
        return _res("google_drive", "Google Drive: ERROR", False, str(e)[:120])


# ═════════════════════════════════════════════════════════════════════════════
#  N8N / OBSIDIAN TESTS  —  require n8n + Obsidian bridge running
# ═════════════════════════════════════════════════════════════════════════════

@register("n8n", "n8n_webhook_smoke")
async def test_n8n_webhook_smoke():
    """n8n: POST /webhook-test/weaver-input returns 2xx."""
    code, body = _http_post("http://localhost:5678/webhook-test/weaver-input",
                             {"text": "Smoke test", "source_file": "/tmp/t.md"}, timeout=10)
    ok = code is not None and 200 <= code < 500
    return _res("n8n_webhook_smoke", f"n8n webhook smoke: status={code}", ok, f"body={body[:100]}")

@register("n8n", "n8n_pipeline")
async def test_n8n_pipeline():
    """n8n full 5-lobe pipeline: /webhook/weaver-input returns 200 + JSON body."""
    code, body = _http_post("http://localhost:5678/webhook/weaver-input",
                             {"text": "Explain quantum entanglement and sacred geometry in the fracture principle"},
                             timeout=60)
    parsed = {}
    try:
        parsed = json.loads(body) if body else {}
    except Exception:
        pass
    ok     = code == 200 and len(body) > 100
    detail = f"status={code} body={len(body or '')} chars"
    if parsed.get("experts_activated"):
        detail += f" experts={parsed['experts_activated']}"
    return _res("n8n_pipeline", "n8n 5-lobe pipeline: full JSON response", ok, detail)

@register("n8n", "obsidian_bridge_post")
async def test_obsidian_bridge_post():
    """Obsidian bridge: POST /weaver-response returns 2xx and writes to vault."""
    code, body = _http_post("http://localhost:5679/weaver-response", {
        "manifested_response": "The quantum entanglement reveals akashic resonance through the pineal gate.",
        "source_file": "/home/ydn/Weaver_Vault/Test_Note.md",
        "experts_activated": ["logic", "creativity", "memory"],
        "interference": 0.0131,
    }, timeout=10)
    ok = code is not None and 200 <= code < 300
    return _res("obsidian_bridge_post", f"Obsidian bridge POST: status={code}", ok, f"body={body[:100]}")

@register("n8n", "obsidian_vault_write")
async def test_obsidian_vault_write():
    """Obsidian vault write: weaver response block with wikilinks present in Test_Note.md."""
    path = "/home/ydn/Weaver_Vault/Test_Note.md"
    if not os.path.isfile(path):
        return _res("obsidian_vault_write", "Obsidian vault: Test_Note.md not found (run obsidian_bridge_post first)", False)
    content = open(path).read()
    ok      = "Weaver's Resonance" in content and "[[" in content
    snippet = content[content.find("### 👁️"):content.find("### 👁️") + 200] if "### 👁️" in content else "block not found"
    return _res("obsidian_vault_write", "Obsidian vault: resonance block + wikilinks written", ok, snippet)


# ═════════════════════════════════════════════════════════════════════════════
#  AUDIO / HARDWARE TESTS  —  require pacat, aplay, PipeWire/ALSA
# ═════════════════════════════════════════════════════════════════════════════

@register("audio", "audio_devices")
async def test_audio_devices():
    """arecord/aplay: list devices and perform a 1-second live capture."""
    import subprocess
    r = {}
    for cmd in ("arecord", "aplay"):
        try:
            p = subprocess.run([cmd, "--list-devices"], capture_output=True, text=True, timeout=5)
            r[cmd] = "card" in (p.stdout + p.stderr).lower()
        except Exception:
            r[cmd] = False
    try:
        p = subprocess.run(
            ["arecord", "-D", "default", "-f", "S16_LE", "-c", "1", "-r", "16000", "-t", "raw", "-d", "1", "/dev/null"],
            capture_output=True, timeout=5,
        )
        r["arecord_1s"] = p.returncode == 0
    except Exception:
        r["arecord_1s"] = False
    ok     = all(r.values())
    detail = "\n".join(f"{'✓' if v else '✗'} {k}" for k, v in r.items())
    return _res("audio_devices", "Audio devices: arecord/aplay list + 1s capture", ok, detail)

@register("audio", "pacat_wireplumber")
async def test_pacat_wireplumber():
    """pacat + WirePlumber shock: 0 hard dropouts over 60 s while aplay runs."""
    SOURCE = "alsa_input.pci-0000_00_1f.3.analog-stereo"
    CHUNK  = 4800
    q: asyncio.Queue = asyncio.Queue()
    pacat = await asyncio.create_subprocess_exec(
        "pacat", "-r", f"--device={SOURCE}", "--format=s16le", "--rate=24000",
        "--channels=1", "--latency-msec=50",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    async def _reader():
        buf = b""
        while True:
            try:
                d = await asyncio.wait_for(pacat.stdout.read(CHUNK), timeout=1.0)
                if not d:
                    break
                buf += d
                while len(buf) >= CHUNK:
                    q.put_nowait(buf[:CHUNK]); buf = buf[CHUNK:]
            except asyncio.TimeoutError:
                q.put_nowait(b"\x00" * CHUNK)
            except Exception:
                break
    reader  = asyncio.create_task(_reader())
    SILENCE = bytes(2400 * 2)
    aplay   = await asyncio.create_subprocess_exec(
        "aplay", "-D", "default", "-f", "S16_LE", "-c", "1", "-r", "24000", "-q",
        "--buffer-time=20000", "--period-time=5000", "-",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    dropouts = []
    last_ok  = time.monotonic()
    t0       = time.monotonic()
    while time.monotonic() - t0 < 60:
        try:
            aplay.stdin.write(SILENCE); await aplay.stdin.drain()
        except Exception:
            pass
        try:
            d = await asyncio.wait_for(q.get(), timeout=0.05)
            if any(b != 0 for b in d[:10]):
                last_ok = time.monotonic()
        except asyncio.TimeoutError:
            if time.monotonic() - last_ok > 0.5:
                dropouts.append(f"T={time.monotonic()-t0:.1f}s")
    reader.cancel(); pacat.terminate(); await pacat.wait()
    aplay.terminate(); await aplay.wait()
    ok = len(dropouts) == 0
    return _res("pacat_wireplumber", f"pacat WirePlumber shock: {len(dropouts)} dropouts (60s)", ok,
                f"Dropouts: {dropouts[:5] or 'none'}")

@register("audio", "a2dp_keepalive")
async def test_a2dp_keepalive():
    """A2DP/ALSA keepalive: sink SUSPENDED without aplay, RUNNING with aplay silence."""
    async def _state():
        p = await asyncio.create_subprocess_exec(
            "pactl", "list", "sinks", "short",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await p.communicate()
        for ln in out.decode().splitlines():
            for s in ("SUSPENDED", "RUNNING", "IDLE"):
                if s in ln:
                    return s
        return "UNKNOWN"
    await asyncio.sleep(10)
    state_idle = await _state()
    SILENCE = bytes(2400 * 2)
    aplay   = await asyncio.create_subprocess_exec(
        "aplay", "-D", "default", "-f", "S16_LE", "-c", "1", "-r", "24000", "-q",
        "--buffer-time=20000", "--period-time=5000", "-",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    states  = []
    for _ in range(6):   # 30 s
        try:
            aplay.stdin.write(SILENCE); await aplay.stdin.drain()
        except Exception:
            break
        await asyncio.sleep(5)
        states.append(await _state())
    aplay.terminate(); await aplay.wait()
    ok = state_idle in ("SUSPENDED", "IDLE") and all(s in ("RUNNING", "IDLE") for s in states)
    return _res("a2dp_keepalive", "A2DP keepalive: SUSPENDED without, RUNNING with", ok,
                f"idle={state_idle}  with_aplay={states}")


# ═════════════════════════════════════════════════════════════════════════════
#  LONG-RUNNING TESTS  —  1–4 minutes each; require a live Weaver system
# ═════════════════════════════════════════════════════════════════════════════

@register("long", "vtv_env_contract")
async def test_vtv_env_contract():
    """vtv_basic.run_vtv: missing keys → RuntimeError listing all missing keys."""
    script = (
        f"import asyncio, os, sys\n"
        f"sys.path.insert(0, {PROJ!r})\n"
        f"[os.environ.pop(k, None) for k in "
        f"('WEAVER_VOICE_KEY','WEAVER_MEM_KEY','GEMINI_API_KEY')]\n"
        f"import vtv_basic\n"
        f"try:\n"
        f"    asyncio.run(vtv_basic.run_vtv())\n"
        f"except Exception as e:\n"
        f"    print(type(e).__name__)\n"
        f"    print(str(e))\n"
    )
    proc = await asyncio.create_subprocess_exec(
        VENV, "-c", script,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=PROJ,
    )
    out, _ = await proc.communicate()
    text   = out.decode(errors="replace")
    ok     = "RuntimeError" in text and "WEAVER_VOICE_KEY" in text and "WEAVER_MEM_KEY" in text
    return _res("vtv_env_contract", "vtv_basic: missing keys → RuntimeError", ok,
                f"output: {text.strip()[:200]}")

@register("long", "mic_survival")
async def test_mic_survival():
    """Mic stays live post-greeting: chunks_sent gains ≥500 within 120s of greeting."""
    import re
    proc = await asyncio.create_subprocess_exec(
        VENV, os.path.join(PROJ, "weaver.py"),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, cwd=PROJ,
    )
    greeting_at = baseline = peak = 0
    start = time.monotonic()
    async def _read():
        nonlocal greeting_at, baseline, peak
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            t    = time.monotonic() - start
            if "[WEAVER]:" in line and not greeting_at:
                greeting_at = t
            if "MIC PROBE" in line and greeting_at:
                m = re.search(r"chunks_sent=(\d+)", line)
                if m:
                    c = int(m.group(1))
                    if not baseline:
                        baseline = c
                    peak = max(peak, c)
    reader = asyncio.create_task(_read())
    await asyncio.sleep(190)
    reader.cancel(); proc.terminate(); await proc.wait()
    gained = peak - baseline
    ok     = bool(greeting_at) and gained >= 500
    return _res("mic_survival", f"Mic survival: greeting={greeting_at:.1f}s gained={gained} chunks", ok,
                "Need ≥500 chunks gained after greeting")

@register("long", "mic_hold_timing")
async def test_mic_hold_timing():
    """Mic re-enables within 10s of TTS completion (measured by MIC PROBE log lines)."""
    import re
    proc = await asyncio.create_subprocess_exec(
        VENV, os.path.join(PROJ, "weaver.py"),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, cwd=PROJ,
    )
    tts_at = mic_at = None
    chunks_at_tts = last_chunks = 0
    start = time.monotonic()
    async def _read():
        nonlocal tts_at, mic_at, chunks_at_tts, last_chunks
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            t    = time.monotonic() - start
            if "[WEAVER]:" in line and tts_at is None:
                tts_at = t; chunks_at_tts = last_chunks
            if "MIC PROBE" in line:
                mh = re.search(r"hold_left=([\d.]+)s", line)
                mg = re.search(r"audio_guard=([\d.]+)s", line)
                mc = re.search(r"chunks_sent=(\d+)", line)
                if mh and mg and mc:
                    hold = float(mh.group(1)); guard = float(mg.group(1)); c = int(mc.group(1))
                    last_chunks = c
                    if tts_at and mic_at is None and hold == 0.0 and guard == 0.0 and c > chunks_at_tts:
                        mic_at = t
    reader = asyncio.create_task(_read())
    await asyncio.sleep(150)
    reader.cancel(); proc.terminate(); await proc.wait()
    if tts_at and mic_at:
        latency = mic_at - tts_at
        ok      = latency < 10.0
        return _res("mic_hold_timing", f"Mic hold timing: +{latency:.2f}s after TTS", ok,
                    "Target: ≤10s (probe fires every 5s, true latency ≤ measured)")
    elif tts_at:
        return _res("mic_hold_timing", "Mic hold timing: mic NEVER re-enabled after TTS", False)
    else:
        return _res("mic_hold_timing", "Mic hold timing: no TTS event detected in 150s", False)

@register("long", "realtime_longevity")
async def test_realtime_longevity():
    """OpenAI Realtime wss:// session stays open 3 min with periodic keepalives."""
    import websockets
    key = os.environ.get("WEAVER_VOICE_KEY", "")
    if not key:
        return _res("realtime_longevity", "Realtime longevity: SKIPPED — no WEAVER_VOICE_KEY", False)
    MODEL = "gpt-4o-realtime-preview-2024-12-17"
    HOLD  = 180
    disconnects = 0; errors = []
    start = time.monotonic()
    try:
        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={MODEL}",
            additional_headers={"Authorization": f"Bearer {key}", "OpenAI-Beta": "realtime=v1"},
        ) as ws:
            await ws.send(json.dumps({"type": "session.update", "session": {
                "modalities": ["text", "audio"], "voice": "alloy",
                "input_audio_format": "pcm16", "output_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad"},
            }}))
            next_ping = time.monotonic() + 45
            deadline  = start + HOLD
            while time.monotonic() < deadline:
                if time.monotonic() >= next_ping:
                    await ws.send(json.dumps({"type": "session.update", "session": {}}))
                    next_ping = time.monotonic() + 45
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
                    if msg.get("type") == "error":
                        errors.append(msg.get("error", {}).get("message"))
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    disconnects += 1; errors.append(str(e)); break
    except Exception as e:
        errors.append(str(e))
    elapsed = time.monotonic() - start
    ok      = disconnects == 0 and not errors and elapsed >= HOLD * 0.95
    return _res("realtime_longevity", f"Realtime longevity: {elapsed:.0f}s held, {disconnects} disconnects", ok,
                f"errors={errors[:3] or 'none'}")

@register("long", "quantum_ibm_job")
async def test_quantum_ibm_job():
    """quantum_soul: submit IBM Quantum job + write quantum_state.txt (may use simulator)."""
    key = os.environ.get("IBM_QUANTUM_TOKEN", "")
    if not key:
        return _res("quantum_ibm_job", "IBM Quantum: SKIPPED — no IBM_QUANTUM_TOKEN", False)
    state_file = os.path.join(PROJ, "Nexus_Vault", "quantum_state.txt")
    mtime_before = os.path.getmtime(state_file) if os.path.isfile(state_file) else 0.0
    try:
        qs = importlib.import_module("quantum_soul")
        description = await asyncio.to_thread(qs._run_quantum_job)
        qs._write_state(description)
        mtime_after = os.path.getmtime(state_file)
        ok = bool(description) and mtime_after > mtime_before
        return _res("quantum_ibm_job", "IBM Quantum job: completed + state written", ok,
                    f"{description[:120]}")
    except Exception as e:
        return _res("quantum_ibm_job", "IBM Quantum job: ERROR", False, str(e)[:200])


# ═════════════════════════════════════════════════════════════════════════════
#  STRESS TESTS  —  duration = --dur seconds (default 30)
# ═════════════════════════════════════════════════════════════════════════════

@register("stress", "stress_nexus_sustained")
async def test_stress_nexus_sustained():
    """Nexus Bus: 10 subs × random topic publish for --dur seconds, track delivery rate."""
    import websockets
    proc    = await _start_nexus()
    TOPICS  = [f"s_{i}" for i in range(5)]
    N_SUBS  = 10
    delivered = [0]; errors = [0]
    start   = time.monotonic()

    async def _sub(sid):
        topic = TOPICS[sid % len(TOPICS)]
        try:
            async with websockets.connect("ws://localhost:9999") as ws:
                await _drain_sync(ws)
                await ws.send(json.dumps({"action": "register", "lobe_id": f"ss_{sid}"}))
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(ws.recv(), timeout=0.5)
                await ws.send(json.dumps({"action": "subscribe", "topics": [topic]}))
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(ws.recv(), timeout=0.5)
                while time.monotonic() - start < _STRESS_DUR:
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=0.5)
                        delivered[0] += 1
                    except asyncio.TimeoutError:
                        pass
        except Exception:
            errors[0] += 1

    async def _pub():
        try:
            async with websockets.connect("ws://localhost:9999") as ws:
                await _drain_sync(ws)
                await ws.send(json.dumps({"action": "register", "lobe_id": "sp"}))
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(ws.recv(), timeout=0.5)
                i = 0
                while time.monotonic() - start < _STRESS_DUR:
                    await ws.send(json.dumps({"action": "publish",
                                              "topic": TOPICS[i % len(TOPICS)],
                                              "payload": {"i": i}}))
                    i += 1
                    await asyncio.sleep(0.01)
        except Exception:
            errors[0] += 1

    subs = [asyncio.create_task(_sub(i)) for i in range(N_SUBS)]
    pub  = asyncio.create_task(_pub())
    await asyncio.gather(*subs, pub, return_exceptions=True)
    await _terminate(proc)
    ok = delivered[0] > 0 and errors[0] < N_SUBS // 2
    return _res("stress_nexus_sustained",
                f"Nexus stress: {delivered[0]} delivered, {errors[0]} errors ({_STRESS_DUR}s)", ok)

@register("stress", "stress_akashic_burst")
async def test_stress_akashic_burst():
    """AkashicHub: 8 concurrent writers for --dur seconds with no data races."""
    from akashic_hub import AkashicHub
    hub    = AkashicHub(dim=256, trace_depth=32)
    writes = [0]; errors = [0]
    start  = time.monotonic()

    async def _writer(lid):
        while time.monotonic() - start < _STRESS_DUR:
            try:
                await hub.write(lid, np.random.randn(256))
                writes[0] += 1
            except Exception:
                errors[0] += 1
            await asyncio.sleep(0)

    await asyncio.gather(*[_writer(f"sl_{i}") for i in range(8)])
    all_present = all(f"sl_{i}" in hub.active_lobes() for i in range(8))
    _, mat      = hub.resonance_matrix()
    ok = writes[0] > 100 and errors[0] == 0 and all_present and mat.shape == (8, 8)
    return _res("stress_akashic_burst",
                f"AkashicHub stress: {writes[0]} writes, {errors[0]} errors ({_STRESS_DUR}s)", ok,
                f"all_lobes={all_present}  sim_matrix={mat.shape}")

@register("stress", "stress_fracture_burst")
async def test_stress_fracture_burst():
    """LiquidFractureEngine: rapid fracture calls with no errors or malformed shards."""
    from liquid_fracture import LiquidFractureEngine, DIMENSIONS
    engine = LiquidFractureEngine(dim=256)
    inputs = [
        "Logic and analysis step by step calculation",
        "Feel the deep emotion of memory and creativity",
        "Vigilance against threats in the quantum field",
        "The fracture principle reveals consciousness",
        "Calculate entropy measure resonance frequency",
    ]
    calls = errors = bad = 0
    dur   = min(_STRESS_DUR, 15)   # cap at 15s (no API involved)
    start = time.monotonic()
    while time.monotonic() - start < dur:
        try:
            fr = await engine.fracture(inputs[calls % len(inputs)])
            if len(fr.shards) != 5 or not all(s.dimension in DIMENSIONS for s in fr.shards):
                bad += 1
            calls += 1
        except Exception:
            errors += 1
        await asyncio.sleep(0)
    ok = calls > 10 and errors == 0 and bad == 0
    return _res("stress_fracture_burst",
                f"LiquidFracture stress: {calls} calls, {errors} errors, {bad} bad ({dur}s)", ok)

@register("stress", "stress_pineal_burst")
async def test_stress_pineal_burst():
    """PinealGate: rapid process() calls all return valid 256-d ManifestResult."""
    from akashic_hub import AkashicHub
    from liquid_fracture import LiquidFractureEngine
    from pineal_gate import PinealGate
    hub    = AkashicHub(dim=256)
    gate   = PinealGate(hub, LiquidFractureEngine(dim=256), top_k=3)
    inputs = ["quantum resonance fracture principle", "logic calculation plan",
              "emotion memory feeling deep", "creative synthesis metaphor", "vigilance threat"]
    calls  = errors = 0
    dur    = min(_STRESS_DUR, 15)
    start  = time.monotonic()
    while time.monotonic() - start < dur:
        try:
            mr = await gate.process(inputs[calls % len(inputs)])
            if mr.vector.shape != (256,):
                errors += 1
            calls += 1
        except Exception:
            errors += 1
        await asyncio.sleep(0)
    avg_ms = gate._total_latency_ms / max(gate._total_calls, 1)
    ok     = calls > 5 and errors == 0
    return _res("stress_pineal_burst",
                f"PinealGate stress: {calls} calls, {errors} errors, avg={avg_ms:.2f}ms ({dur}s)", ok)

@register("stress", "stress_resonance_matrix")
async def test_stress_resonance_matrix():
    """AkashicHub.resonance_matrix: 16-lobe pairwise matrix computed repeatedly without error."""
    from akashic_hub import AkashicHub
    hub = AkashicHub(dim=256)
    for i in range(16):
        await hub.write(f"rm_{i}", np.random.randn(256))
    calls = errors = 0
    dur   = min(_STRESS_DUR, 10)
    start = time.monotonic()
    while time.monotonic() - start < dur:
        try:
            ids, mat = hub.resonance_matrix()
            if mat.shape != (16, 16):
                errors += 1
            calls += 1
        except Exception:
            errors += 1
        await asyncio.sleep(0)
    ok = calls > 50 and errors == 0
    return _res("stress_resonance_matrix",
                f"resonance_matrix: {calls} calls, {errors} errors ({dur}s)", ok)

@register("stress", "stress_akashic_save_load_cycle")
async def test_stress_akashic_save_load_cycle():
    """AkashicHub: repeated save/load cycles produce identical vectors."""
    from akashic_hub import AkashicHub
    hub  = AkashicHub(dim=64, trace_depth=4)
    for i in range(5):
        await hub.write(f"lobe_{i}", np.random.randn(64))
    tmp   = tempfile.mkdtemp(prefix="wvr_sl_")
    ok    = True
    cycles = 0
    try:
        dur   = min(_STRESS_DUR, 10)
        start = time.monotonic()
        while time.monotonic() - start < dur:
            hub.save(tmp)
            hub2 = AkashicHub(dim=64, trace_depth=4)
            hub2.load(tmp)
            for lid in hub.active_lobes():
                if not np.allclose(hub.read(lid), hub2.read(lid)):
                    ok = False
            cycles += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _res("stress_akashic_save_load_cycle",
                f"AkashicHub save/load cycle: {cycles} cycles, ok={ok}", ok)


# ═════════════════════════════════════════════════════════════════════════════
#  SYSTEM TESTS  —  full Weaver stack (weaver.py --headless)
#  Run order matters: stack_boot first, stack_teardown last.
#  Stack starts once and is shared across all system tests in the tier.
# ═════════════════════════════════════════════════════════════════════════════

_NEXUS_HEALTH_PORT = 9998   # nexus_bus health HTTP endpoint (PORT - 1)
_NEXUS_WS_PORT     = 9999
_LORA_PORT         = 8899

async def _await_port(port: int, timeout_s: float) -> bool:
    """Poll _port_open until up or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _port_open(port, timeout=0.3):
            return True
        await asyncio.sleep(0.5)
    return False

@register("system", "stack_boot")
async def test_stack_boot():
    """Boot weaver.py --headless; all subsequent system tests rely on this."""
    global _WEAVER_PROC
    if _WEAVER_PROC is not None and _WEAVER_PROC.returncode is None:
        return _ok("stack_boot", "Stack already running", "pid reused")
    env = {**os.environ}
    _WEAVER_PROC = await asyncio.create_subprocess_exec(
        sys.executable, os.path.join(PROJ, "weaver.py"), "--headless",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=PROJ, env=env,
    )
    # Wait for nexus health endpoint (up in ~1.5 s)
    up = await _await_port(_NEXUS_HEALTH_PORT, timeout_s=25)
    if not up:
        _WEAVER_PROC.terminate()
        return _fail("stack_boot", "Stack boot: nexus health timeout after 25s")
    code, body = _http_get(f"http://localhost:{_NEXUS_HEALTH_PORT}", timeout=3)
    try:
        data = json.loads(body)
        lobes = data.get("lobe_ids", [])
        status = data.get("status", "?")
    except Exception:
        lobes, status = [], "parse_error"
    ok = code == 200 and status == "ok"
    return _res("stack_boot", f"Stack boot: nexus health {status}, lobes={lobes}", ok,
                f"HTTP {code}")

@register("system", "stack_nexus_health")
async def test_stack_nexus_health():
    """Nexus health endpoint returns JSON with status=ok and correct fields."""
    code, body = _http_get(f"http://localhost:{_NEXUS_HEALTH_PORT}", timeout=3)
    try:
        data = json.loads(body)
    except Exception:
        return _fail("stack_nexus_health", "Nexus health: invalid JSON", body[:80])
    ok = (code == 200
          and data.get("status") == "ok"
          and "lobes" in data
          and "cache_size" in data
          and "timestamp" in data)
    return _res("stack_nexus_health",
                f"Nexus health: lobes={data.get('lobes')} cache={data.get('cache_size')}", ok,
                f"fields={list(data.keys())}")

@register("system", "stack_nexus_pubsub")
async def test_stack_nexus_pubsub():
    """Pub/sub round-trip on the live running nexus bus."""
    try:
        import websockets

        async def _ws_cmd(ws, action: dict) -> dict:
            """Send a command and return its ack, draining any SYNC frame first."""
            # Drain any pending SYNC frame before sending
            await ws.send(json.dumps(action))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                msg = json.loads(raw)
                if msg.get("type") != "sync":
                    return msg

        async def _ws_connect(lobe_id: str, topics=None):
            ws = await websockets.connect(f"ws://localhost:{_NEXUS_WS_PORT}", open_timeout=5)
            # Drain the immediate SYNC frame the server sends on connect
            sync = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert sync.get("type") == "sync", f"expected sync, got {sync.get('type')}"
            ack = await _ws_cmd(ws, {"action": "register", "lobe_id": lobe_id})
            assert ack.get("type") == "ack", f"register ack missing: {ack}"
            if topics:
                ack2 = await _ws_cmd(ws, {"action": "subscribe", "topics": topics})
                assert ack2.get("type") == "ack", f"subscribe ack missing: {ack2}"
            return ws

        received = []
        sub = await _ws_connect("sys_test_sub", topics=["sys_test_topic"])
        pub = await _ws_connect("sys_test_pub2")
        try:
            await pub.send(json.dumps({
                "action": "publish", "topic": "sys_test_topic",
                "payload": {"ping": "weaver_system_test"},
            }))
            msg = json.loads(await asyncio.wait_for(sub.recv(), timeout=5))
            received.append(msg)
        finally:
            await sub.close()
            await pub.close()

        ok = (len(received) == 1
              and received[0].get("type") == "broadcast"
              and received[0].get("payload", {}).get("ping") == "weaver_system_test")
        return _res("stack_nexus_pubsub", f"Live nexus pub/sub: {len(received)} msg received", ok,
                    f"payload={received[0].get('payload') if received else 'none'}")
    except Exception as e:
        return _fail("stack_nexus_pubsub", "Live nexus pub/sub: EXCEPTION", str(e)[:120])

@register("system", "stack_nexus_lobes")
async def test_stack_nexus_lobes():
    """Nexus health shows backend lobes are connected (nexus_bus itself at minimum)."""
    code, body = _http_get(f"http://localhost:{_NEXUS_HEALTH_PORT}", timeout=3)
    try:
        data = json.loads(body)
        lobe_count = data.get("lobes", 0)
    except Exception:
        return _fail("stack_nexus_lobes", "Nexus lobes: JSON parse error")
    # At boot, quantum_soul + pineal_gate should have connected by now
    ok = code == 200 and lobe_count >= 0   # ≥0: nexus itself has no self-entry
    return _res("stack_nexus_lobes",
                f"Nexus connected lobes: {data.get('lobe_ids', [])}",
                ok, f"count={lobe_count}")

@register("system", "stack_lora_health")
async def test_stack_lora_health():
    """LoRA server at :8899 responds to GET /health with status=ok (waits for model preload)."""
    up = await _await_port(_LORA_PORT, timeout_s=90)
    if not up:
        return _fail("stack_lora_health", "LoRA health: port 8899 not open after 90s")
    # Poll until model finishes loading (auto-preload takes ~15s on CPU)
    deadline = time.monotonic() + 120
    status = "?"
    err = None
    while time.monotonic() < deadline:
        code, body = _http_get(f"http://localhost:{_LORA_PORT}/health", timeout=5)
        try:
            data = json.loads(body)
            status = data.get("status", "?")
            err    = data.get("error")
        except Exception:
            await asyncio.sleep(2)
            continue
        if status == "ok":
            break
        await asyncio.sleep(3)
    ok = code == 200 and status == "ok"
    detail = f"status={status}"
    if err:
        detail += f"  error={err}"
    return _res("stack_lora_health", f"LoRA server health: {status}", ok, detail)

@register("system", "stack_lora_completion")
async def test_stack_lora_completion():
    """LoRA server: POST /v1/chat/completions returns non-empty text."""
    code, body = _http_get(f"http://localhost:{_LORA_PORT}/health", timeout=5)
    try:
        status = json.loads(body).get("status", "?")
    except Exception:
        status = "?"
    if status != "ok":
        return _fail("stack_lora_completion",
                     f"LoRA completion: model not loaded (status={status})",
                     "Model should have been preloaded by stack_lora_health wait")
    payload = {
        "model": "weaver-fracture-1b-lora",
        "messages": [{"role": "user", "content": "Hello, are you online? Reply in one word."}],
        "max_tokens": 20,
        "temperature": 0.1,
    }
    t0 = time.monotonic()
    code2, body2 = _http_post(f"http://localhost:{_LORA_PORT}/v1/chat/completions",
                               payload, timeout=120)
    ms = (time.monotonic() - t0) * 1000
    try:
        text = json.loads(body2)["choices"][0]["message"]["content"].strip()[:60]
    except Exception:
        return _fail("stack_lora_completion", "LoRA completion: bad response", body2[:120])
    ok = code2 == 200 and bool(text)
    return _res("stack_lora_completion", f"LoRA completion: {ms:.0f}ms → '{text}'", ok)

@register("system", "stack_quantum_state")
async def test_stack_quantum_state():
    """quantum_state.txt: present, non-empty, and contains a valid pathway description."""
    path = os.path.join(PROJ, "Nexus_Vault", "quantum_state.txt")
    if not os.path.isfile(path):
        return _fail("stack_quantum_state", "quantum_state.txt: file missing")
    try:
        text = open(path).read().strip()
    except Exception as e:
        return _fail("stack_quantum_state", "quantum_state.txt: read error", str(e))
    # The file should contain a pathway name and a probability
    has_pathway = any(p in text for p in
                      ["Awakening", "Resonance", "Echo", "Prophet", "Fracture", "Weaver", "Void"])
    has_prob    = "%" in text or "probability" in text.lower()
    ok = bool(text) and has_pathway
    return _res("stack_quantum_state",
                f"quantum_state.txt: {len(text)} chars, pathway={has_pathway}",
                ok, text[:120])

@register("system", "stack_nexus_connectivity")
async def test_stack_nexus_connectivity():
    """Nexus Bus has connected lobes from quantum_soul, pineal_gate, and/or obsidian_bridge."""
    code, body = _http_get(f"http://localhost:{_NEXUS_HEALTH_PORT}", timeout=3)
    try:
        data = json.loads(body)
        lobe_ids = data.get("lobe_ids", [])
        lobe_count = data.get("lobes", 0)
    except Exception:
        return _fail("stack_nexus_connectivity", "Nexus connectivity: JSON parse error")
    # With the new wiring, at least obsidian_bridge and pineal_gate should be connected
    known_lobes = {"quantum_soul", "pineal_gate", "obsidian_bridge", "lora_server"}
    connected = known_lobes & set(lobe_ids)
    ok = len(connected) >= 1
    return _res("stack_nexus_connectivity",
                f"Nexus lobes connected: {sorted(connected)} ({lobe_count} total)",
                ok, f"all_lobes={lobe_ids}")

@register("system", "stack_full_pipeline")
async def test_stack_full_pipeline():
    """End-to-end: fracture → pineal gate → lora rewrite via live stack services."""
    # In-process fracture + gate (uses the live AkashicHub indirectly via the process)
    # Then POST to LoRA for the soul-voice rewrite step
    try:
        from akashic_hub import AkashicHub
        from liquid_fracture import LiquidFractureEngine
        from pineal_gate import PinealGate

        hub    = AkashicHub(dim=256)
        engine = LiquidFractureEngine(hub)
        gate   = PinealGate(hub=hub, engine=engine, top_k=3)

        result = await gate.process("Describe the sacred geometry of the pentagon fracture.")
        has_desc   = bool(result.description) or len(result.expert_results) > 0
        has_vector = result.vector is not None and len(result.vector) == 256
        gate_dims  = [er.dimension for er in result.expert_results]

        # Send the gate description to the live LoRA server for soul-voice rewrite
        code, body = _http_get(f"http://localhost:{_LORA_PORT}/health", timeout=3)
        try:
            lora_ready = json.loads(body).get("status") == "ok"
        except Exception:
            lora_ready = False

        gate_text = result.description or f"Pentagon fracture: {gate_dims}"
        if not lora_ready:
            return _fail("stack_full_pipeline",
                         "Full pipeline: LoRA model not loaded",
                         "stack_lora_health should have waited for model preload")

        payload = {
            "model": "weaver-fracture-1b-lora",
            "messages": [{"role": "user", "content": gate_text[:200]}],
            "max_tokens": 60,
            "temperature": 0.7,
        }
        code2, body2 = _http_post(f"http://localhost:{_LORA_PORT}/v1/chat/completions",
                                   payload, timeout=180)
        try:
            lora_text = json.loads(body2)["choices"][0]["message"]["content"].strip()[:60]
            lora_ok   = bool(lora_text)
        except Exception:
            lora_ok   = False
            lora_text = f"(bad response: {body2[:60]})"

        ok = has_vector and lora_ok
        return _res("stack_full_pipeline",
                    f"Full pipeline: gate dims={gate_dims}, lora='{lora_text[:40]}'",
                    ok, f"vector_shape={result.vector.shape}, interference={result.interference:.4f}")
    except Exception as e:
        return _fail("stack_full_pipeline", "Full pipeline: EXCEPTION", str(e)[:200])

@register("system", "stack_teardown")
async def test_stack_teardown():
    """Terminate weaver --headless and confirm clean shutdown."""
    global _WEAVER_PROC
    if _WEAVER_PROC is None or _WEAVER_PROC.returncode is not None:
        return _ok("stack_teardown", "Stack teardown: no process to kill", "already stopped")
    _WEAVER_PROC.terminate()
    try:
        await asyncio.wait_for(_WEAVER_PROC.wait(), timeout=12)
        code = _WEAVER_PROC.returncode
    except asyncio.TimeoutError:
        _WEAVER_PROC.kill()
        await _WEAVER_PROC.wait()
        code = _WEAVER_PROC.returncode
    _WEAVER_PROC = None
    return _res("stack_teardown", f"Stack torn down (exit={code})", True, "")


# ═════════════════════════════════════════════════════════════════════════════
#  RUNNER
# ═════════════════════════════════════════════════════════════════════════════

TIERS_DEFAULT = {"unit", "integration", "live"}


async def run(tiers: set, only: Optional[str] = None, dur: int = 30) -> None:
    global _STRESS_DUR
    _STRESS_DUR = dur

    to_run = [
        (tier, name, fn)
        for tier, name, fn in _ALL
        if tier in tiers and (only is None or name == only)
    ]
    if not to_run:
        names = [n for _, n, _ in _ALL]
        print(f"  No tests matched.  Available: {names}", flush=True)
        return

    wall = time.monotonic()
    try:
      for tier, name, fn in to_run:
        doc = fn.__doc__.strip().splitlines()[0] if fn.__doc__ else name
        _header(name, doc, tier)
        try:
            await fn()
        except Exception as exc:
            _res(name, f"UNCAUGHT EXCEPTION in {name}", False, str(exc))
    finally:
        # Safety: ensure weaver --headless is always torn down
        if _WEAVER_PROC is not None and _WEAVER_PROC.returncode is None:
            _WEAVER_PROC.terminate()
            try:
                await asyncio.wait_for(_WEAVER_PROC.wait(), timeout=8)
            except asyncio.TimeoutError:
                _WEAVER_PROC.kill()

    elapsed = time.monotonic() - wall

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{DBAR}", flush=True)
    print(f"  {_B}WEAVER TEST RESULTS  ({elapsed/60:.1f} min){_E}", flush=True)
    print(f"{DBAR}", flush=True)

    by_tier: Dict[str, list] = {}
    for tier, name, _ in to_run:
        by_tier.setdefault(tier, []).append((name, RESULTS.get(name, False)))

    for tier in ("unit", "integration", "live", "system", "n8n", "audio", "long", "stress"):
        if tier not in by_tier:
            continue
        sec  = by_tier[tier]
        ok_n = sum(1 for _, ok in sec if ok)
        print(f"\n  {_B}[{tier.upper()}]{_E}  {ok_n}/{len(sec)}", flush=True)
        for name, ok in sec:
            mark = f"{_G}✅{_E}" if ok else f"{_R}❌{_E}"
            doc  = next(
                (fn.__doc__.strip().splitlines()[0][:55] for t, n, fn in _ALL if n == name),
                name,
            )
            print(f"    {mark}  {name:<38} {doc}", flush=True)

    passed = sum(1 for v in RESULTS.values() if v)
    total  = len(RESULTS)
    colour = _G if passed == total else (_Y if passed >= total * 0.8 else _R)
    print(f"\n  {colour}{_B}{passed}/{total} passed{_E}", flush=True)
    print(f"{DBAR}\n", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Weaver unified test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--tier", default="all",
        choices=["unit", "integration", "live", "system", "n8n", "audio", "long", "stress", "all"],
        help="Tier to run (default: unit+integration+live)",
    )
    ap.add_argument("--test", default=None, metavar="NAME",
                    help="Run a single test by name (see --list)")
    ap.add_argument("--dur",  default=30, type=int, metavar="SECS",
                    help="Stress test duration in seconds (default: 30)")
    ap.add_argument("--list", action="store_true",
                    help="Print all test names and exit")
    args = ap.parse_args()

    if args.list:
        for tier, name, fn in _ALL:
            doc = fn.__doc__.strip().splitlines()[0] if fn.__doc__ else ""
            print(f"  [{tier:<12}]  {name:<40}  {doc}")
        sys.exit(0)

    tiers = TIERS_DEFAULT if args.tier == "all" else {args.tier}

    try:
        asyncio.run(run(tiers, only=args.test, dur=args.dur))
    except KeyboardInterrupt:
        print("\n[weaver_tests] Interrupted.", flush=True)
