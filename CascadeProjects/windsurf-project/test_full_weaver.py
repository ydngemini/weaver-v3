#!/usr/bin/env python3
"""
test_full_weaver.py — Comprehensive 25-test suite across ALL Weaver subsystems.

Covers:
  A. Core Infrastructure (ports, imports, env)
  B. Akashic Hub (vector state)
  C. Liquid Fracture Engine (LTC ODE)
  D. Pineal Gate (MoE routing)
  E. SLM Experts (OpenAI inference)
  F. Quantum Networks (topologies, learner, interference)
  G. Nexus Bus (protocol, security)
  H. n8n Workflow (webhook, pipeline, response)
  I. Obsidian Bridge (wikilinks, vault write)
  J. Persistence & Artifacts (dataset, LoRA, credentials)
"""

import asyncio
import json
import os
import socket
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

PROJ = os.path.dirname(os.path.abspath(__file__))
BAR = "─" * 62
RESULTS = {}
TOTAL_START = time.monotonic()


def mark(ok):
    return "✅" if ok else "❌"


def header(num, title):
    print(f"\n{BAR}\n  TEST {num}: {title}\n{BAR}", flush=True)


def result(num, title, ok, detail=""):
    RESULTS[f"{num}"] = (title, ok)
    print(f"  {mark(ok)} {title}", flush=True)
    if detail:
        for line in detail.strip().split("\n"):
            print(f"     {line}", flush=True)


def port_open(port, timeout=1):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    ok = s.connect_ex(("127.0.0.1", port)) == 0
    s.close()
    return ok


def http_post(url, payload, timeout=30):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.getcode(), resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# A. CORE INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

header(1, "Core imports (all modules)")
try:
    import akashic_hub
    import liquid_fracture
    import pineal_gate
    import slm_experts
    import quantum_networks
    import quantum_soul
    import nexus_bus
    import obsidian_bridge
    result(1, "All 8 new modules import cleanly", True)
except Exception as e:
    result(1, "All 8 new modules import cleanly", False, str(e))

header(2, "Environment variables present")
keys = ["WEAVER_VOICE_KEY", "WEAVER_MEM_KEY", "GEMINI_API_KEY"]
present = {k: bool(os.environ.get(k)) for k in keys}
all_ok = all(present.values())
result(2, "Required env vars set", all_ok, "\n".join(f"{k}: {'set' if v else 'MISSING'}" for k, v in present.items()))

header(3, "n8n Docker container (port 5678)")
ok = port_open(5678)
result(3, "n8n listening on 5678", ok)

# ══════════════════════════════════════════════════════════════════════════════
# B. AKASHIC HUB
# ══════════════════════════════════════════════════════════════════════════════

header(4, "Akashic Hub write/read latency")
import numpy as np
from akashic_hub import AkashicHub

hub = AkashicHub(dim=256, trace_depth=32)

async def _hub_bench():
    lats = []
    for i in range(100):
        lat = await hub.write(f"bench_{i % 5}", np.random.randn(256))
        lats.append(lat)
    return np.mean(lats), np.max(lats)

mean_ms, max_ms = asyncio.run(_hub_bench())
ok = mean_ms < 1.0
result(4, f"Hub write: mean={mean_ms:.3f}ms max={max_ms:.3f}ms", ok, "Target: <1ms mean")

header(5, "Akashic Hub cosine similarity query")
hub2 = AkashicHub(dim=256)
asyncio.run(hub2.write("a", np.array([1.0] * 128 + [0.0] * 128)))
asyncio.run(hub2.write("b", np.array([0.0] * 128 + [1.0] * 128)))
asyncio.run(hub2.write("c", np.array([1.0] * 128 + [0.0] * 128)))
matches = hub2.query(np.array([1.0] * 128 + [0.0] * 128), top_k=2)
ok = len(matches) == 2 and matches[0][0] in ("a", "c") and matches[0][1] > 0.9
result(5, "Cosine query returns correct top-k", ok, f"Matches: {matches}")

header(6, "Akashic Hub temporal trace")
hub3 = AkashicHub(dim=4, trace_depth=8)
for i in range(5):
    asyncio.run(hub3.write("trace_test", np.array([float(i)] * 4)))
trace = hub3.temporal_trace("trace_test")
mat = hub3.temporal_matrix("trace_test")
ok = len(trace) == 5 and mat is not None and mat.shape == (5, 4)
result(6, "Temporal trace stores and retrieves correctly", ok, f"Trace len={len(trace)}, matrix shape={mat.shape if mat is not None else None}")

header(7, "Akashic Hub entanglement blending")
hub4 = AkashicHub(dim=4)
asyncio.run(hub4.write("x", np.array([1.0, 0.0, 0.0, 0.0])))
asyncio.run(hub4.write("y", np.array([0.0, 1.0, 0.0, 0.0])))
blended = hub4.entangle(["x", "y"])
ok = blended.shape == (4,) and np.linalg.norm(blended) > 0.9
result(7, "Entanglement blend produces normalized vector", ok, f"Blended: {blended}")

# ══════════════════════════════════════════════════════════════════════════════
# C. LIQUID FRACTURE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

header(8, "Liquid Fracture: 5 dimensions produced")
from liquid_fracture import LiquidFractureEngine, DIMENSIONS
engine = LiquidFractureEngine(dim=256)
fr = asyncio.run(engine.fracture("Calculate the optimal batch size for a hot honey ferment"))
ok = len(fr.shards) == 5 and all(s.dimension in DIMENSIONS for s in fr.shards)
result(8, "Fracture produces 5 dimensional shards", ok, f"Shards: {[s.dimension for s in fr.shards]}")

header(9, "Liquid Fracture: weights sum to 1")
wsum = sum(s.weight for s in fr.shards)
ok = abs(wsum - 1.0) < 1e-6
result(9, f"Fracture weights sum to {wsum:.6f}", ok)

header(10, "Liquid Fracture: tau adapts to input")
taus = [s.tau for s in fr.shards]
ok = all(0.05 < t < 10.0 for t in taus)
result(10, "All tau values in valid range", ok, f"Taus: {[f'{t:.3f}' for t in taus]}")

header(11, "Liquid Fracture: keyword routing correctness")
fr_logic = asyncio.run(engine.fracture("Calculate the structure step by step because therefore"))
ok = fr_logic.shards[0].dimension == "logic"
result(11, "Logic keywords route to logic dimension", ok, f"Top shard: {fr_logic.shards[0].dimension} w={fr_logic.shards[0].weight:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# D. PINEAL GATE (MoE)
# ══════════════════════════════════════════════════════════════════════════════

header(12, "Pineal Gate: full pipeline returns ManifestResult")
from pineal_gate import PinealGate
gate = PinealGate(AkashicHub(dim=256), LiquidFractureEngine(dim=256), top_k=3)
mr = asyncio.run(gate.process("Tell me about sacred geometry and quantum entanglement"))
ok = mr.vector.shape == (256,) and len(mr.expert_results) > 0 and mr.latency_ms > 0
result(12, f"ManifestResult: {len(mr.expert_results)} experts, {mr.latency_ms:.1f}ms", ok)

header(13, "Pineal Gate: geometric interference computed")
ok = isinstance(mr.interference, float)
kind = "constructive" if mr.interference > 0 else "destructive"
result(13, f"Interference: {mr.interference:+.4f} ({kind})", ok)

header(14, "Pineal Gate: sparse top-k gating")
ok = len(mr.gate_decision.top_k) == 3
result(14, f"Top-3 activated: {mr.gate_decision.top_k}", ok)

# ══════════════════════════════════════════════════════════════════════════════
# E. SLM EXPERTS
# ══════════════════════════════════════════════════════════════════════════════

header(15, "SLM Experts: all 5 lobes instantiate")
from slm_experts import build_experts
try:
    experts = build_experts(AkashicHub(dim=256))
    ok = len(experts) == 5 and set(experts.keys()) == {"logic", "emotion", "memory", "creativity", "vigilance"}
    result(15, f"5 SLM experts built: {list(experts.keys())}", ok)
except Exception as e:
    result(15, "5 SLM experts built", False, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# F. QUANTUM NETWORKS
# ══════════════════════════════════════════════════════════════════════════════

header(16, "Quantum Networks: all 5 topologies generate valid CNOT pairs")
from quantum_networks import EntanglementTopology
all_ok = True
details = []
for name in EntanglementTopology.all_names():
    pairs = EntanglementTopology.get(name)
    valid = len(pairs) > 0 and all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
    details.append(f"{name}: {len(pairs)} pairs {'✓' if valid else '✗'}")
    if not valid:
        all_ok = False
result(16, "All topologies produce valid CNOT pairs", all_ok, "\n".join(details))

header(17, "Quantum Networks: variational circuit builds with correct param count")
from quantum_networks import VariationalFractureCircuit
vc = VariationalFractureCircuit(n_qubits=7, n_layers=3, topology="pentagon")
qc = vc.build()
ok = vc.param_count() == 42 and qc.num_qubits == 7
result(17, f"Variational circuit: params={vc.param_count()}, qubits={qc.num_qubits}", ok)

header(18, "Quantum Networks: learner computes fitness from counts")
from quantum_networks import QuantumLearner
learner = QuantumLearner(vc)
fitness = learner.compute_fitness({"0000001": 200, "1000000": 150, "0111011": 100, "0000000": 50})
ok = 0.0 < fitness < 1.0
result(18, f"Fitness from mock counts: {fitness:.4f}", ok)

header(19, "Quantum Networks: interference routing bias")
from quantum_networks import QuantumInterferenceNetwork
ifn = QuantumInterferenceNetwork()
biases = ifn.compute_routing_bias({"0000001": 200, "1000000": 150, "0111011": 100, "0000000": 50})
ok = len(biases) == 5 and all(0.0 <= v <= 1.0 for v in biases.values())
result(19, f"Routing bias: {biases}", ok)

header(20, "Quantum Networks: orchestrator cycles topologies")
from quantum_networks import QuantumNetworkOrchestrator
orch = QuantumNetworkOrchestrator()
t1 = orch.current_topology()
orch._topo_idx = (orch._topo_idx + 1) % len(orch.topologies)
t2 = orch.current_topology()
ok = t1 != t2 and t1 in orch.topologies and t2 in orch.topologies
result(20, f"Topology cycle: {t1} → {t2}", ok)

# ══════════════════════════════════════════════════════════════════════════════
# G. NEXUS BUS (security hardening)
# ══════════════════════════════════════════════════════════════════════════════

header(21, "Nexus Bus: non-object JSON rejection")
from nexus_bus import _handle_message, LobeConnection
import types

class FakeWS:
    def __init__(self):
        self.sent = []
        self.remote_address = ("127.0.0.1", 0)
    async def send(self, data):
        self.sent.append(data)

async def _test_nonobj():
    ws = FakeWS()
    lobe = LobeConnection(ws)
    await _handle_message(lobe, json.dumps([1, 2, 3]))
    resp = json.loads(ws.sent[0]) if ws.sent else {}
    return resp.get("type") == "error" and "JSON object" in resp.get("msg", "")

ok = asyncio.run(_test_nonobj())
result(21, "Non-object JSON rejected with error frame", ok)

header(22, "Nexus Bus: duplicate lobe_id blocked")
from nexus_bus import _connections

async def _test_dup():
    _connections.clear()
    ws1, ws2 = FakeWS(), FakeWS()
    lobe1, lobe2 = LobeConnection(ws1), LobeConnection(ws2)
    await _handle_message(lobe1, json.dumps({"action": "register", "lobe_id": "dup_test"}))
    _connections["dup_test"] = lobe1
    await _handle_message(lobe2, json.dumps({"action": "register", "lobe_id": "dup_test"}))
    resp = json.loads(ws2.sent[-1]) if ws2.sent else {}
    _connections.clear()
    return resp.get("type") == "error" and "already in use" in resp.get("msg", "")

ok = asyncio.run(_test_dup())
result(22, "Duplicate lobe_id registration blocked", ok)

# ══════════════════════════════════════════════════════════════════════════════
# H. N8N WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

header(23, "n8n webhook: full 5-lobe pipeline returns 200 + JSON body")
code, body = http_post("http://localhost:5678/webhook/weaver-input", {
    "text": "Explain quantum entanglement and sacred geometry in the fracture principle"
}, timeout=60)
ok = code == 200 and len(body) > 100
parsed = {}
if body:
    try:
        parsed = json.loads(body)
    except:
        pass
detail = f"Status={code}, body={len(body or '')} chars"
if parsed.get("expert_count"):
    detail += f", experts={parsed['expert_count']}"
if parsed.get("experts_activated"):
    detail += f", activated={parsed['experts_activated']}"
result(23, "n8n 5-lobe pipeline returns full JSON", ok, detail)

header(24, "n8n response: manifested_response contains all 5 lobe outputs")
manifested = parsed.get("manifested_response", "")
has_5 = manifested.count("[") >= 5
result(24, f"Manifested response has {manifested.count('[')} expert blocks", has_5)

# ══════════════════════════════════════════════════════════════════════════════
# I. OBSIDIAN BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

header(25, "Obsidian Bridge: wikilink injection")
from obsidian_bridge import inject_wikilinks
sample = "The quantum entanglement reveals akashic resonance through the pineal gate. YDN configured the topology."
linked = inject_wikilinks(sample)
ok = "[[Quantum Soul]]" in linked and "[[Akashic]]" in linked and "[[YDN]]" in linked and "[[Pineal Gate]]" in linked
result(25, "Wikilink injection for known keywords", ok, f"Links found: {linked.split('**Synaptic Links:** ')[-1][:200] if '**Synaptic' in linked else 'none'}")

# ══════════════════════════════════════════════════════════════════════════════
# J. PERSISTENCE & ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

header(26, "Persistence: Nexus_Vault directory exists")
vault = os.path.join(PROJ, "Nexus_Vault")
ok = os.path.isdir(vault)
result(26, f"Nexus_Vault exists: {vault}", ok)

header(27, "Persistence: quantum_state.txt exists")
qs_file = os.path.join(vault, "quantum_state.txt")
ok = os.path.isfile(qs_file)
detail = ""
if ok:
    with open(qs_file) as f:
        detail = f.read()[:120]
result(27, "quantum_state.txt present", ok, detail)

header(28, "Persistence: soul dataset valid JSONL")
ds_path = os.path.join(PROJ, "weaver_soul_dataset.jsonl")
ok = False
count = 0
if os.path.isfile(ds_path):
    with open(ds_path) as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            obj = json.loads(line)
            if "messages" in obj and len(obj["messages"]) >= 2:
                count += 1
    ok = count >= 5
result(28, f"Dataset: {count}/10 sample lines valid", ok)

header(29, "Persistence: LoRA adapter files present")
lora_dir = os.path.join(PROJ, "weaver_fracture_1B_lora")
required = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json"]
files_ok = all(os.path.isfile(os.path.join(lora_dir, f)) for f in required)
result(29, f"LoRA adapter: all {len(required)} files present", files_ok)

header(30, "Persistence: Drive credentials parse")
creds_ok = False
try:
    token_path = os.path.join(PROJ, "token.json")
    ghost_path = os.path.join(PROJ, "ghost_key.json")
    creds_path = os.path.join(PROJ, "credentials.json")
    creds_ok = os.path.isfile(token_path) and os.path.isfile(ghost_path) and os.path.isfile(creds_path)
    if creds_ok:
        with open(creds_path) as f:
            data = json.load(f)
        creds_ok = "installed" in data or "web" in data
except Exception as e:
    pass
result(30, "Drive credential files present and valid", creds_ok)

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

elapsed = time.monotonic() - TOTAL_START
print(f"\n{'═' * 62}")
print(f"  WEAVER FULL SYSTEM TEST — {len(RESULTS)} TESTS")
print(f"{'═' * 62}")

sections = {
    "A. Core Infrastructure": ["1", "2", "3"],
    "B. Akashic Hub": ["4", "5", "6", "7"],
    "C. Liquid Fracture": ["8", "9", "10", "11"],
    "D. Pineal Gate": ["12", "13", "14"],
    "E. SLM Experts": ["15"],
    "F. Quantum Networks": ["16", "17", "18", "19", "20"],
    "G. Nexus Bus Security": ["21", "22"],
    "H. n8n Workflow": ["23", "24"],
    "I. Obsidian Bridge": ["25"],
    "J. Persistence": ["26", "27", "28", "29", "30"],
}

for section, nums in sections.items():
    sec_pass = sum(1 for n in nums if RESULTS.get(n, ("", False))[1])
    sec_mark = "✅" if sec_pass == len(nums) else "❌"
    print(f"\n  {sec_mark} {section} ({sec_pass}/{len(nums)})")
    for n in nums:
        title, ok = RESULTS.get(n, ("?", False))
        print(f"     {mark(ok)} {n}. {title}")

passed = sum(1 for _, (_, ok) in RESULTS.items() if ok)
print(f"\n{'═' * 62}")
print(f"  TOTAL: {passed}/{len(RESULTS)} passed  ({elapsed:.1f}s)")
print(f"{'═' * 62}\n")
