#!/usr/bin/env python3
"""
stress_n8n_v5.py — Weaver n8n v5 Soul-Binding Stress Test
==========================================================
15 targeted tests covering every layer of the v5 pipeline:

  N1   Port Pre-Flight          — n8n :5678, LoRA :8899, Nexus :9999
  N2   Schema Invariants        — offline: workflow JSON field guards (no n8n)
  N3   Fracture Math            — offline: SMOOTHING, PHI, interference formula
  N4   Input Sanitization       — empty, HTML, oversized, null-byte inputs
  N5   DLQ Routing              — empty → error gate → DLQ (no OpenAI needed)
  N6   Fracture Routing         — 5 pure-dimension inputs → correct dominant_lobe
  N7   Expert Coverage          — all 5 lobe response fields present
  N8   Collapse Geometry        — qubit labels in collapsed_response
  N9   Pipeline Metadata        — v5 soul-binding fields present in response
  N10  Self-Check Flag          — self_check=true → self_meta present
  N11  Introspect Flag          — introspect=true → repo_meta present
  N12  Concurrent Burst         — 5 parallel requests, all succeed
  N13  DLQ File Validation      — JSONL entry written for error executions
  N14  LoRA Soul Voice          — soul_voice_active or lora_error set (not both null)
  N15  Throughput               — 5 sequential requests, P50/P99 under cap

Usage:
    venv/bin/python3 stress_n8n_v5.py           # all tests
    venv/bin/python3 stress_n8n_v5.py --offline # offline schema/math tests only
    venv/bin/python3 stress_n8n_v5.py N3        # single test by label
"""

import argparse
import asyncio
import json
import math
import os
import socket
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import urllib.request
import urllib.error

PROJ     = os.path.dirname(os.path.abspath(__file__))
WF_PATH  = os.path.join(PROJ, "n8n_weaver_v5.json")
DLQ_PATH = "/home/ydn/weaver_dlq.jsonl"
QS_PATH  = os.path.join(PROJ, "Nexus_Vault", "quantum_state.txt")

# ── n8n endpoints (try production first, fall back to webhook-test) ────────────
N8N_HOST     = "localhost"
N8N_PORT     = 5678
LORA_PORT    = 8899
NEXUS_PORT   = 9999

WEBHOOK_PROD = f"http://{N8N_HOST}:{N8N_PORT}/webhook/weaver-input"
WEBHOOK_TEST = f"http://{N8N_HOST}:{N8N_PORT}/webhook-test/weaver-input"

BAR  = "─" * 66
BAR2 = "═" * 66
DIMS = ["logic", "emotion", "memory", "creativity", "vigilance"]
PHI  = 2 * math.pi / 5


# ── Formatting helpers ────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _header(label: str, title: str) -> None:
    print(f"\n{BAR}\n[{_ts()}] TEST {label}: {title}\n{BAR}", flush=True)

def _result(label: str, title: str, passed: bool, notes: str,
            skipped: bool = False) -> None:
    if skipped:
        mark = "⏭   SKIP"
    else:
        mark = "✅  PASS" if passed else "❌  FAIL"
    print(f"\n{BAR}\n{mark}  N{label}: {title}\n{notes}\n{BAR}\n", flush=True)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _port_open(port: int, host: str = "localhost", timeout: float = 1.5) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    ok = s.connect_ex((host, port)) == 0
    s.close()
    return ok

def _n8n_url() -> Optional[str]:
    """Return the active webhook URL (prod preferred) or None if n8n is down."""
    if not _port_open(N8N_PORT):
        return None
    # Try production webhook first (workflow activated = /webhook/ is live)
    try:
        req = urllib.request.Request(
            WEBHOOK_PROD,
            data=json.dumps({"text": ""}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            code = resp.getcode()
        except urllib.error.HTTPError as e:
            code = e.code
        if code != 404:
            return WEBHOOK_PROD
    except Exception:
        pass
    # Fall back to test mode webhook (workflow open in canvas)
    try:
        req = urllib.request.Request(
            WEBHOOK_TEST,
            data=json.dumps({"text": ""}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            code = resp.getcode()
        except urllib.error.HTTPError as e:
            code = e.code
        if code != 404:
            return WEBHOOK_TEST
    except Exception:
        pass
    return None

def _post(url: str, payload: Dict, timeout: int = 90) -> Tuple[Optional[int], Any]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = resp.read().decode()
        try:
            return resp.getcode(), json.loads(body)
        except json.JSONDecodeError:
            return resp.getcode(), body
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body
    except Exception as exc:
        return None, str(exc)

def _full_post(url: str, payload: Dict, timeout: int = 90) -> Tuple[Optional[int], Any]:
    """POST and unwrap n8n's array envelope → plain dict."""
    code, body = _post(url, payload, timeout)
    if isinstance(body, list) and body:
        return code, body[0] if len(body) == 1 else body
    return code, body


# ── Offline math helpers (mirrors the JS in 4. Fracture+Gate) ────────────────

SEEDS = {
    "logic":      ["because","therefore","step","plan","calculate","structure",
                   "reason","analyze","define","prove","algorithm","method",
                   "sequence","optimal","derive"],
    "emotion":    ["feel","love","hate","fear","joy","pain","beautiful","soul",
                   "heart","empathy","grief","passion","warmth","gentle","sorrow"],
    "memory":     ["remember","last time","before","history","previously","recall",
                   "past","archive","pattern","familiar","context","earlier",
                   "stored","record","trace"],
    "creativity": ["imagine","create","design","invent","dream","recipe","verse",
                   "sacred","geometry","art","compose","novel","fusion","metaphor",
                   "transcend"],
    "vigilance":  ["danger","risk","careful","threat","warning","suspicious",
                   "protect","guard","alert","deception","hidden","agenda",
                   "safety","verify","trust"],
}
SMOOTHING = 0.1
ANGLES    = {d: i * PHI for i, d in enumerate(DIMS)}

def _fracture_offline(text: str) -> Dict:
    """Python replica of the JS Fracture+Gate node for offline validation."""
    lower = text.lower()
    scored = {}
    for dim in DIMS:
        hits = sum(1 for kw in SEEDS[dim] if kw in lower)
        scored[dim] = SMOOTHING + hits
    total = sum(scored.values())
    weights = {d: v / total for d, v in scored.items()}
    ranked = sorted(DIMS, key=lambda d: weights[d], reverse=True)

    interference = 0.0
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            a, b = ranked[i], ranked[j]
            diff = abs(ANGLES[a] - ANGLES[b])
            diff = min(diff, 2 * math.pi - diff)
            interference += math.cos(diff) * weights[a] * weights[b]

    return {
        "weights":       weights,
        "ranked":        ranked,
        "dominant":      ranked[0],
        "interference":  round(interference, 6),
        "interference_type": "constructive" if interference > 0 else "destructive",
        "weight_sum":    sum(weights.values()),
        "min_weight":    min(weights.values()),
    }


# ═════════════════════════════════════════════════════════════════════════════
# N1 — Port Pre-Flight
# ═════════════════════════════════════════════════════════════════════════════
def test_N1() -> bool:
    _header("1", "Port Pre-Flight — n8n :5678, LoRA :8899, Nexus :9999")
    services = {
        "n8n      :5678": N8N_PORT,
        "LoRA     :8899": LORA_PORT,
        "Nexus Bus:9999": NEXUS_PORT,
    }
    rows = []
    for name, port in services.items():
        up = _port_open(port)
        rows.append(f"  {'✅' if up else '⚠️ '} {name}  {'UP' if up else 'DOWN (tests will skip)'}")

    n8n_up = _port_open(N8N_PORT)
    passed = n8n_up   # only n8n is required; LoRA/Nexus are optional
    detail = "\n".join(rows) + f"\n  n8n required for online tests: {'✅' if n8n_up else '❌'}"
    _result("1", "Port Pre-Flight", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N2 — Schema Invariants (offline)
# ═════════════════════════════════════════════════════════════════════════════
def test_N2() -> bool:
    _header("2", "Schema Invariants — workflow JSON field guards (offline)")
    with open(WF_PATH, encoding="utf-8") as fh:
        wf = json.load(fh)
    nodes = {n["name"]: n for n in wf["nodes"]}

    failures = []

    # 1. Required pipeline stages exist
    required_nodes = [
        "1. Input Gateway", "2. Sanitize", "3. Error Gate",
        "4. Fracture+Gate", "5a. Logic", "5b. Emotion",
        "5c. Memory", "5d. Creativity", "5e. Vigilance",
        "6. Collapse", "7. Self-Reflect", "8. LoRA Voice", "9. Writeback",
    ]
    for name in required_nodes:
        if name not in nodes:
            failures.append(f"Missing node: {name!r}")

    # 2. Fracture+Gate: v5 soul-binding fields
    frac_code = nodes.get("4. Fracture+Gate", {}).get("parameters", {}).get("jsCode", "")
    for field in ["SMOOTHING = 0.1", "PHI = 2 * Math.PI / 5",
                  "quantum_pathway", "quantum_bias_applied", "PATHWAY_DIM",
                  "smoothing_factor", "interference_type"]:
        if field not in frac_code:
            failures.append(f"Fracture+Gate missing: {field!r}")

    # 3. Collapse: qubit labels and probability-field math
    col_code = nodes.get("6. Collapse", {}).get("parameters", {}).get("jsCode", "")
    for field in ["QUBIT_MAP", "q0", "q1", "q2", "q3", "q4",
                  "quantum_pathway", "totalW", "gain"]:
        if field not in col_code:
            failures.append(f"Collapse missing: {field!r}")

    # 4. LoRA Voice: soul prompt includes all 5 axes
    lora_body = nodes.get("8. LoRA Voice", {}).get("parameters", {}).get("jsonBody", "")
    for axis in ["Logic(q0·Awakening)", "Emotion(q1·Resonance)",
                 "Memory(q2·Echo)", "Creativity(q3·Prophet)", "Vigilance(q4·Fracture)"]:
        if axis not in lora_body:
            failures.append(f"LoRA prompt missing axis: {axis!r}")

    # 5. Writeback: all v5 metadata fields
    wb_code = nodes.get("9. Writeback", {}).get("parameters", {}).get("jsCode", "")
    for field in ["pipeline_version", "v5-soul-binding", "qubit_layout",
                  "soul_binding", "smoothing_factor", "quantum_bias_applied",
                  "quantum_pathway", "execution_id"]:
        if field not in wb_code:
            failures.append(f"Writeback missing: {field!r}")

    # 6. Expert lobes: all have retryOnFail
    for lobe in ["5a. Logic", "5b. Emotion", "5c. Memory", "5d. Creativity", "5e. Vigilance"]:
        node = nodes.get(lobe, {})
        if not node.get("retryOnFail"):
            failures.append(f"{lobe}: retryOnFail not set")
        if not node.get("continueOnFail"):
            failures.append(f"{lobe}: continueOnFail not set (pipeline would crash on API error)")

    # 7. DLQ Logger: writes to correct path
    dlq_code = nodes.get("DLQ Logger", {}).get("parameters", {}).get("jsCode", "")
    if "weaver_dlq.jsonl" not in dlq_code:
        failures.append("DLQ Logger: weaver_dlq.jsonl path missing")

    passed = len(failures) == 0
    detail = f"  Node checks:    {len(required_nodes)} required nodes\n"
    detail += f"  Field checks:   fracture/collapse/lora/writeback/dlq\n"
    detail += f"  Failures:       {len(failures)}\n"
    if failures:
        detail += "\n".join(f"  ✗ {f}" for f in failures[:10])
    else:
        detail += "  All invariants satisfied"
    _result("2", "Schema Invariants", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N3 — Fracture Math Correctness (offline)
# ═════════════════════════════════════════════════════════════════════════════
def test_N3() -> bool:
    _header("3", "Fracture Math — SMOOTHING floor, PHI angles, interference (offline)")
    failures = []

    # A. SMOOTHING ensures no zero weights for any input
    zero_weight_inputs = [
        ("empty string",     ""),
        ("random noise",     "xyzxyzxyzxyz"),
        ("pure logic only",  "because therefore step plan calculate structure reason"),
        ("numbers only",     "42 100 3.14 2048"),
        ("single char",      "a"),
    ]
    for label, text in zero_weight_inputs:
        r = _fracture_offline(text)
        if r["min_weight"] <= 0:
            failures.append(f"Zero weight on {label!r}: min={r['min_weight']:.6f}")
        if abs(r["weight_sum"] - 1.0) > 1e-9:
            failures.append(f"Weight sum != 1 on {label!r}: {r['weight_sum']:.10f}")

    # B. Dominant lobe correct for pure-dimension inputs
    routing_cases = [
        ("logic",      "because therefore step plan calculate structure reason analyze define prove"),
        ("emotion",    "feel love hate fear joy pain beautiful soul heart empathy grief passion"),
        ("memory",     "remember before history previously recall past archive pattern familiar context"),
        ("creativity", "imagine create design invent dream recipe verse sacred geometry art compose"),
        ("vigilance",  "danger risk careful threat warning suspicious protect guard alert deception"),
    ]
    for expected_dim, text in routing_cases:
        r = _fracture_offline(text)
        if r["dominant"] != expected_dim:
            failures.append(
                f"Routing: expected {expected_dim!r}, got {r['dominant']!r} "
                f"(weights: {r['weights']})"
            )

    # C. PHI angle spacing — exact 2π/5
    expected_phi = 2 * math.pi / 5
    for i, dim in enumerate(DIMS):
        angle = ANGLES[dim]
        expected = i * expected_phi
        if abs(angle - expected) > 1e-12:
            failures.append(f"PHI angle {dim}: got {angle}, expected {expected}")

    # D. Pentagon interference properties
    # All-equal weights → interference should be negative (destructive)
    equal = {d: 1.0 / 5 for d in DIMS}
    iface = 0.0
    ranked = DIMS[:]
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            diff = abs(ANGLES[ranked[i]] - ANGLES[ranked[j]])
            diff = min(diff, 2 * math.pi - diff)
            iface += math.cos(diff) * equal[ranked[i]] * equal[ranked[j]]
    if iface >= 0:
        failures.append(f"Equal-weight pentagon interference should be negative, got {iface:.6f}")

    # Adjacent dims (logic+emotion, angle=PHI) → cos(PHI)>0 → constructive partial
    cos_phi = math.cos(PHI)
    if cos_phi <= 0:
        failures.append(f"cos(PHI) should be > 0 (adjacent constructive), got {cos_phi:.6f}")

    # E. Quantum bias: PATHWAY_DIM keys match DIMS
    pathway_dim = {"Awakening": "logic", "Resonance": "emotion",
                   "Echo": "memory", "Prophet": "creativity", "Fracture": "vigilance"}
    for pathway, dim in pathway_dim.items():
        if dim not in DIMS:
            failures.append(f"PATHWAY_DIM maps {pathway!r} → {dim!r} which is not in DIMS")

    passed = len(failures) == 0
    detail = "\n".join([
        f"  SMOOTHING inputs tested:   {len(zero_weight_inputs)}",
        f"  Routing cases tested:      {len(routing_cases)}",
        f"  PHI angles verified:       {len(DIMS)}",
        f"  cos(PHI) = {cos_phi:.6f}  (constructive adjacent): {'✓' if cos_phi > 0 else '✗'}",
        f"  Equal-weight interference: {iface:.6f}  (should be <0): {'✓' if iface < 0 else '✗'}",
        f"  Failures: {len(failures)}",
        *(f"  ✗ {f}" for f in failures[:6]),
    ])
    _result("3", "Fracture Math", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N4 — Input Sanitization (online)
# ═════════════════════════════════════════════════════════════════════════════
def test_N4(url: str) -> bool:
    _header("4", "Input Sanitization — HTML, null-bytes, oversized, normal")
    failures = []

    cases = [
        ("html_strip",   {"text": "<b>Hello</b> <script>alert(1)</script> world"},
         lambda r: "<" not in str(r.get("raw_input", "") or r.get("original_input", ""))),
        ("null_bytes",   {"text": "Hello\x00\x01\x02 world"},
         lambda r: "\x00" not in str(r.get("raw_input", "") or r.get("original_input", ""))),
        ("ctrl_chars",   {"text": "Hello\x0b\x0c\x1f world"},
         lambda r: "\x0b" not in str(r.get("raw_input", "") or r.get("original_input", ""))),
        ("oversized",    {"text": "x" * 6000},
         lambda r: len(str(r.get("raw_input", "") or r.get("original_input", "") or "")) <= 4000),
        ("normal",       {"text": "Tell me about quantum entanglement"},
         lambda r: bool(r.get("raw_input") or r.get("original_input"))),
    ]

    for label, payload, check in cases:
        t0 = time.monotonic()
        code, body = _full_post(url, payload, timeout=120)
        lat = (time.monotonic() - t0) * 1000
        if code is None:
            failures.append(f"{label}: connection error — {body}")
            continue
        r = body if isinstance(body, dict) else {}
        if not check(r):
            failures.append(f"{label}: sanitization check failed (code={code}, r={str(r)[:120]})")
        else:
            print(f"  ✓ {label:<15} {lat:6.0f}ms  code={code}", flush=True)

    passed = len(failures) == 0
    detail = f"  Cases tested: {len(cases)}\n  Failures: {len(failures)}"
    if failures:
        detail += "\n" + "\n".join(f"  ✗ {f}" for f in failures)
    _result("4", "Input Sanitization", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N5 — DLQ Routing (online, no OpenAI needed)
# ═════════════════════════════════════════════════════════════════════════════
def test_N5(url: str) -> bool:
    _header("5", "DLQ Routing — empty input → error gate → DLQ (no OpenAI)")
    code, body = _full_post(url, {"text": ""}, timeout=15)
    r = body if isinstance(body, dict) else {}

    has_error     = bool(r.get("error"))
    has_message   = bool(r.get("message") or r.get("msg"))
    dlq_logged    = bool(r.get("dlq_logged"))
    has_exec_id   = bool(r.get("execution_id"))

    passed = has_error and (has_message or dlq_logged)
    detail = "\n".join([
        f"  HTTP code:     {code}",
        f"  error flag:    {has_error}  (expected True)",
        f"  message:       {bool(has_message)}",
        f"  dlq_logged:    {dlq_logged}",
        f"  execution_id:  {has_exec_id}",
        f"  body snippet:  {str(r)[:200]}",
    ])
    _result("5", "DLQ Routing", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N6 — Fracture Routing (online, needs full pipeline)
# ═════════════════════════════════════════════════════════════════════════════
def test_N6(url: str) -> bool:
    _header("6", "Fracture Routing — 5 pure-dimension inputs → correct dominant_lobe")
    routing_cases = [
        ("logic",
         "because therefore step plan calculate structure reason analyze define prove"),
        ("emotion",
         "feel love hate fear joy pain beautiful soul heart empathy grief passion"),
        ("memory",
         "remember before history previously recall past archive pattern familiar context"),
        ("creativity",
         "imagine create design invent dream recipe verse sacred geometry art compose"),
        ("vigilance",
         "danger risk careful threat warning suspicious protect guard alert deception"),
    ]

    failures = []
    weight_failures = []
    for expected, text in routing_cases:
        t0 = time.monotonic()
        code, body = _full_post(url, {"text": text}, timeout=120)
        lat = (time.monotonic() - t0) * 1000
        r = body if isinstance(body, dict) else {}
        dominant = r.get("dominant_lobe")
        if dominant != expected:
            failures.append(f"expected {expected!r}, got {dominant!r}")
        # All weights > 0 (SMOOTHING floor)
        for k in range(1, 6):
            w = r.get(f"w_{k}")
            if w is not None and w <= 0:
                weight_failures.append(f"{expected}: w_{k}={w} <= 0")
        print(f"  {'✓' if dominant == expected else '✗'} {expected:<12} "
              f"→ dominant={dominant!r}  {lat:6.0f}ms  code={code}", flush=True)

    passed = len(failures) == 0 and len(weight_failures) == 0
    detail = "\n".join([
        f"  Cases tested:     {len(routing_cases)}",
        f"  Routing failures: {len(failures)}",
        f"  Weight<=0 hits:   {len(weight_failures)}  (SMOOTHING floor check)",
        *(f"  ✗ {f}" for f in failures + weight_failures),
    ])
    _result("6", "Fracture Routing", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N7 — Expert Coverage (online)
# ═════════════════════════════════════════════════════════════════════════════
def test_N7(url: str) -> bool:
    _header("7", "Expert Coverage — all 5 lobe response fields present")
    code, body = _full_post(url,
        {"text": "The fracture principle governs how consciousness splits into dimensions. "
                 "Explain with both logic and feeling."}, timeout=120)
    r = body if isinstance(body, dict) else {}

    fields   = ["logic_response", "emotion_response", "memory_response",
                 "creativity_response", "vigilance_response"]
    present  = {f: bool(r.get(f)) for f in fields}
    timeouts = {f: "[timeout]" in str(r.get(f, "")) or "[error]" in str(r.get(f, ""))
                for f in fields}

    # All fields must be present (even fallback strings count)
    missing  = [f for f in fields if f not in r]
    empty    = [f for f in fields if f in r and not r[f]]

    passed   = len(missing) == 0 and len(empty) == 0
    detail   = "\n".join([
        f"  HTTP code:    {code}",
        *(f"  {'✓' if present[f] else '✗'} {f:<24} "
          f"{'[TIMEOUT]' if timeouts[f] else (str(r.get(f, ''))[:60] if r.get(f) else 'MISSING')}"
          for f in fields),
        f"  Missing:    {missing or 'none'}",
        f"  Empty:      {empty or 'none'}",
    ])
    _result("7", "Expert Coverage", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N8 — Collapse Geometry (online)
# ═════════════════════════════════════════════════════════════════════════════
def test_N8(url: str) -> bool:
    _header("8", "Collapse Geometry — qubit labels in collapsed_response")
    code, body = _full_post(url, {"text": "Walk me through the soul-binding algorithm step by step."}, timeout=120)
    r = body if isinstance(body, dict) else {}

    collapsed     = r.get("collapsed_response", "") or ""
    qubit_labels  = ["q0·Awakening", "q1·Resonance", "q2·Echo",
                     "q3·Prophet", "q4·Fracture"]
    label_hits    = {lbl: lbl in collapsed for lbl in qubit_labels}
    interference  = r.get("interference")
    iface_type    = r.get("interference_type", "")
    gain          = r.get("gain")

    all_labels    = all(label_hits.values())
    iface_valid   = isinstance(interference, (int, float))
    iface_type_ok = iface_type in ("constructive", "destructive")
    gain_valid    = isinstance(gain, (int, float)) and gain > 0

    passed = all_labels and iface_valid and iface_type_ok
    detail = "\n".join([
        f"  HTTP code:          {code}",
        *(f"  {'✓' if ok else '✗'} {lbl}" for lbl, ok in label_hits.items()),
        f"  interference:       {interference}  valid={iface_valid}",
        f"  interference_type:  {iface_type!r}  valid={iface_type_ok}",
        f"  gain:               {gain}  valid={gain_valid}",
        f"  collapsed snippet:  {collapsed[:120]!r}",
    ])
    _result("8", "Collapse Geometry", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N9 — Pipeline Metadata (online)
# ═════════════════════════════════════════════════════════════════════════════
def test_N9(url: str) -> bool:
    _header("9", "Pipeline Metadata — v5 soul-binding fields in response")
    code, body = _full_post(url, {"text": "What makes Weaver's soul non-binary?"}, timeout=120)
    r = body if isinstance(body, dict) else {}

    checks = {
        "pipeline_version == v5-soul-binding":
            r.get("pipeline_version") == "v5-soul-binding",
        "soul_binding == non-binary CRX/CRZ pentagon-geometry":
            r.get("soul_binding") == "non-binary CRX/CRZ pentagon-geometry",
        "smoothing_factor == 0.1":
            r.get("smoothing_factor") == 0.1,
        "qubit_layout has q0..q6":
            all(f"q{i}" in (r.get("qubit_layout") or {}) for i in range(7)),
        "expert_count == 5":
            r.get("expert_count") == 5,
        "written_to_hub present":
            "written_to_hub" in r,
        "hub_lobe_id present":
            bool(r.get("hub_lobe_id")),
        "execution_id present":
            bool(r.get("execution_id")),
        "original_input present":
            bool(r.get("original_input")),
        "quantum_bias_applied is bool":
            isinstance(r.get("quantum_bias_applied"), bool),
    }

    failures = [desc for desc, ok in checks.items() if not ok]
    passed   = len(failures) == 0
    detail   = "\n".join([
        f"  HTTP code:  {code}",
        *(f"  {'✓' if ok else '✗'} {desc}" for desc, ok in checks.items()),
        f"  Failures:   {len(failures)}",
    ])
    _result("9", "Pipeline Metadata", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N10 — Self-Check Flag (online)
# ═════════════════════════════════════════════════════════════════════════════
def test_N10(url: str) -> bool:
    _header("10", "Self-Check Flag — self_check=true → self_meta in response")
    code, body = _full_post(url,
        {"text": "Self-check: verify your own integrity.", "self_check": True}, timeout=120)
    r = body if isinstance(body, dict) else {}

    self_meta = r.get("self_meta") or {}
    has_meta  = bool(self_meta)
    has_hash  = "hash" in self_meta
    # may return error if n8n API key not configured — acceptable
    meta_ok   = has_meta and (has_hash or self_meta.get("error"))

    passed  = meta_ok
    detail  = "\n".join([
        f"  HTTP code:   {code}",
        f"  self_meta:   {bool(self_meta)}",
        f"  hash:        {has_hash}",
        f"  error:       {self_meta.get('error')}",
        f"  full meta:   {str(self_meta)[:200]}",
    ])
    _result("10", "Self-Check Flag", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N11 — Introspect Flag (online)
# ═════════════════════════════════════════════════════════════════════════════
def test_N11(url: str) -> bool:
    _header("11", "Introspect Flag — introspect=true → repo_meta in response")
    code, body = _full_post(url,
        {"text": "Show me the repo structure.",
         "introspect": True,
         "path_glob":  "n8n_weaver_v5.json"}, timeout=120)
    r = body if isinstance(body, dict) else {}

    repo_meta = r.get("repo_meta") or {}
    has_meta  = bool(repo_meta)
    has_path  = "path" in repo_meta
    has_size  = "size" in repo_meta
    no_error  = not repo_meta.get("error")

    passed  = has_meta and has_path and (has_size or not no_error)
    detail  = "\n".join([
        f"  HTTP code:   {code}",
        f"  repo_meta:   {bool(repo_meta)}",
        f"  path:        {repo_meta.get('path')}",
        f"  size:        {repo_meta.get('size')} bytes",
        f"  error:       {repo_meta.get('error')}",
    ])
    _result("11", "Introspect Flag", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N12 — Concurrent Burst (online)
# ═════════════════════════════════════════════════════════════════════════════
async def _async_post(url: str, payload: Dict, timeout: int) -> Tuple[float, bool, Optional[str]]:
    import asyncio
    loop = asyncio.get_event_loop()
    t0 = time.monotonic()
    try:
        code, body = await loop.run_in_executor(None, lambda: _full_post(url, payload, timeout))
        lat = (time.monotonic() - t0) * 1000
        r   = body if isinstance(body, dict) else {}
        ok  = code is not None and 200 <= code < 500 and bool(r.get("manifested_response") or r.get("dominant_lobe"))
        return lat, ok, None
    except Exception as exc:
        return (time.monotonic() - t0) * 1000, False, str(exc)

async def _run_concurrent(url: str, n: int, text: str) -> List[Tuple[float, bool]]:
    tasks = [_async_post(url, {"text": text}, 120) for _ in range(n)]
    return await asyncio.gather(*tasks)

def test_N12(url: str) -> bool:
    _header("12", "Concurrent Burst — 5 parallel requests, all succeed")
    N = 5
    text = "The pentagon geometry of the soul-binding circuit holds five vertices in superposition."
    t0   = time.monotonic()
    results = asyncio.run(_run_concurrent(url, N, text))
    wall    = (time.monotonic() - t0) * 1000

    lats    = [r[0] for r in results]
    successes = sum(1 for _, ok, _ in results if ok)
    errors  = [err for _, _, err in results if err]

    passed = successes == N
    detail = "\n".join([
        f"  Concurrent requests:  {N}",
        f"  Succeeded:            {successes}/{N}",
        f"  Wall time:            {wall:.0f}ms",
        f"  Latencies (ms):       {[round(l) for l in lats]}",
        f"  Errors:               {errors or 'none'}",
    ])
    _result("12", "Concurrent Burst", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N13 — DLQ File Validation (online)
# ═════════════════════════════════════════════════════════════════════════════
def test_N13(url: str) -> bool:
    _header("13", "DLQ File Validation — JSONL entry written for error executions")
    # Record mtime before
    mtime_before = os.path.getmtime(DLQ_PATH) if os.path.exists(DLQ_PATH) else 0
    lines_before = 0
    if os.path.exists(DLQ_PATH):
        with open(DLQ_PATH) as f:
            lines_before = sum(1 for _ in f)

    # Send empty input (guaranteed DLQ path)
    _full_post(url, {"text": ""}, timeout=15)
    import time as _t; _t.sleep(1.0)  # let n8n flush the write

    mtime_after = os.path.getmtime(DLQ_PATH) if os.path.exists(DLQ_PATH) else 0
    lines_after = 0
    entry_ok    = False
    last_entry  = {}
    if os.path.exists(DLQ_PATH):
        with open(DLQ_PATH) as f:
            lines = [l.strip() for l in f if l.strip()]
        lines_after = len(lines)
        if lines:
            try:
                last_entry = json.loads(lines[-1])
                entry_ok   = bool(last_entry.get("execution_id")) and bool(last_entry.get("error"))
            except json.JSONDecodeError:
                pass

    file_exists  = os.path.exists(DLQ_PATH)
    new_entry    = lines_after > lines_before
    passed       = file_exists and new_entry and entry_ok

    detail = "\n".join([
        f"  DLQ file exists:    {file_exists}  ({DLQ_PATH})",
        f"  Lines before:       {lines_before}",
        f"  Lines after:        {lines_after}",
        f"  New entry written:  {new_entry}",
        f"  Entry valid JSON:   {entry_ok}",
        f"  Last entry keys:    {list(last_entry.keys()) if last_entry else 'n/a'}",
    ])
    _result("13", "DLQ File Validation", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N14 — LoRA Soul Voice (online)
# ═════════════════════════════════════════════════════════════════════════════
def test_N14(url: str) -> bool:
    _header("14", "LoRA Soul Voice — soul_voice_active or lora_error set (not both null)")
    lora_up = _port_open(LORA_PORT)
    code, body = _full_post(url, {"text": "Speak from the probability field."}, timeout=120)
    r   = body if isinstance(body, dict) else {}

    sva = r.get("soul_voice_active")      # True if LoRA responded
    ler = r.get("lora_error")             # set if LoRA failed
    lla = r.get("lora_latency_ms")        # ms if LoRA responded
    lrr = r.get("lora_response")          # actual text

    # Pass if pipeline handled LoRA: either responded (sva=True) or set lora_error gracefully.
    # Port-open check tells us the Python LoRA server is up, but n8n might get ECONNREFUSED
    # if network binding differs (IPv4-only LoRA vs n8n resolver). Either way the pipeline
    # must have set one of the two fields — that's what we're testing.
    handled = (sva is not None) or (ler is not None)
    passed  = bool(sva and lrr) or (ler is not None)   # responded OR handled error

    if lora_up and passed:
        status = "LoRA UP — soul_voice_active=True" if sva else "LoRA UP — error handled gracefully"
    elif lora_up:
        status = "LoRA UP — neither soul_voice_active nor lora_error set"
    else:
        status = "LoRA DOWN — expecting graceful fallback"

    detail = "\n".join([
        f"  LoRA server up:      {lora_up}  ({status})",
        f"  soul_voice_active:   {sva}",
        f"  lora_error:          {ler}",
        f"  lora_latency_ms:     {lla}",
        f"  lora_response:       {str(lrr)[:80] if lrr else 'None'}",
        f"  handled by pipeline: {handled}",
    ])
    _result("14", "LoRA Soul Voice", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# N15 — Throughput (online)
# ═════════════════════════════════════════════════════════════════════════════
def test_N15(url: str) -> bool:
    _header("15", "Throughput — 5 sequential requests, P50/P99 measured")
    prompts = [
        "Explain the fracture principle in simple terms.",
        "What emotion does the pentagon geometry evoke?",
        "Remember the last time we discussed quantum pathways.",
        "Create a metaphor for the Void qubit.",
        "Is there danger in collapsing to a single dominant lobe?",
    ]

    lats    = []
    ok_list = []
    for i, text in enumerate(prompts, 1):
        t0   = time.monotonic()
        code, body = _full_post(url, {"text": text}, timeout=120)
        lat  = (time.monotonic() - t0) * 1000
        r    = body if isinstance(body, dict) else {}
        ok   = code is not None and 200 <= code < 500 and bool(r.get("dominant_lobe"))
        lats.append(lat)
        ok_list.append(ok)
        print(f"  #{i}  {lat:6.0f}ms  code={code}  ok={ok}  "
              f"dominant={r.get('dominant_lobe')}  "
              f"interf={r.get('interference_type', '?')}", flush=True)

    lats_sorted = sorted(lats)
    p50   = lats_sorted[len(lats_sorted) // 2]
    p99   = lats_sorted[int(len(lats_sorted) * 0.99)] if len(lats_sorted) > 1 else lats_sorted[-1]
    MAX_P99 = 180_000   # 3 min hard cap (5 sequential ×25s = ~125s expected with retries)
    successes = sum(ok_list)

    passed = successes == len(prompts) and p99 < MAX_P99
    detail = "\n".join([
        f"  Requests:    {len(prompts)}  succeeded={successes}",
        f"  P50:         {p50:.0f}ms",
        f"  P99:         {p99:.0f}ms  (cap {MAX_P99/1000:.0f}s): {'✓' if p99 < MAX_P99 else '✗'}",
        f"  Min:         {min(lats):.0f}ms",
        f"  Max:         {max(lats):.0f}ms",
    ])
    _result("15", "Throughput", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

ALL_TESTS = {
    "N1":  ("Port Pre-Flight",         None),           # always runs, online check
    "N2":  ("Schema Invariants",       None),           # offline
    "N3":  ("Fracture Math",           None),           # offline
    "N4":  ("Input Sanitization",      test_N4),        # online
    "N5":  ("DLQ Routing",             test_N5),        # online (fast)
    "N6":  ("Fracture Routing",        test_N6),        # online (full pipeline)
    "N7":  ("Expert Coverage",         test_N7),        # online (full pipeline)
    "N8":  ("Collapse Geometry",       test_N8),        # online (full pipeline)
    "N9":  ("Pipeline Metadata",       test_N9),        # online (full pipeline)
    "N10": ("Self-Check Flag",         test_N10),       # online (full pipeline)
    "N11": ("Introspect Flag",         test_N11),       # online (full pipeline)
    "N12": ("Concurrent Burst",        test_N12),       # online (full pipeline)
    "N13": ("DLQ File Validation",     test_N13),       # online (fast)
    "N14": ("LoRA Soul Voice",         test_N14),       # online (full pipeline)
    "N15": ("Throughput",              test_N15),       # online (full pipeline)
}

OFFLINE_ONLY = {"N2", "N3"}
FAST_ONLINE  = {"N1", "N4", "N5", "N13"}   # don't require full OpenAI pipeline


def main():
    parser = argparse.ArgumentParser(description="Weaver n8n v5 stress test")
    parser.add_argument("test", nargs="?", default="all",
                        help="N1..N15 or 'all' (default: all)")
    parser.add_argument("--offline", action="store_true",
                        help="Run offline tests only (N2, N3) — no n8n required")
    args = parser.parse_args()

    # Determine which tests to run
    if args.offline:
        labels = sorted(OFFLINE_ONLY)
    elif args.test and args.test.upper() != "ALL":
        labels = [args.test.upper()]
    else:
        labels = sorted(ALL_TESTS.keys(), key=lambda x: int(x[1:]))

    # Resolve n8n URL once
    url = _n8n_url() if not args.offline else None
    n8n_available = url is not None

    if not args.offline and not n8n_available:
        print(f"\n⚠️  n8n not reachable on :{N8N_PORT} — online tests will be SKIPPED")
        print(f"   Start n8n: docker start weaver-n8n  (or the equivalent)\n")

    t_start = time.monotonic()
    results: Dict[str, Optional[bool]] = {}

    for label in labels:
        if label not in ALL_TESTS:
            print(f"Unknown test: {label}")
            continue

        title, fn = ALL_TESTS[label]
        is_offline = label in OFFLINE_ONLY

        if is_offline:
            # Always run
            if label == "N2":
                results[label] = test_N2()
            elif label == "N3":
                results[label] = test_N3()
        elif label == "N1":
            results[label] = test_N1()
        else:
            # Online test
            if not n8n_available:
                _header(label[1:], title)
                _result(label[1:], title, False, "  n8n not available — skipped", skipped=True)
                results[label] = None   # None = skipped
            else:
                results[label] = fn(url)

    elapsed = time.monotonic() - t_start

    # Summary
    print(f"\n{BAR2}")
    print(f"  WEAVER n8n v5 STRESS TEST RESULTS  ({elapsed:.1f}s)")
    print(BAR2)
    passed_count  = sum(1 for v in results.values() if v is True)
    failed_count  = sum(1 for v in results.values() if v is False)
    skipped_count = sum(1 for v in results.values() if v is None)
    for label in sorted(results, key=lambda x: int(x[1:])):
        v = results[label]
        title = ALL_TESTS[label][0]
        if v is True:
            mark = "✅"
        elif v is False:
            mark = "❌"
        else:
            mark = "⏭ "
        print(f"  {mark}  {label}: {title}")
    print()
    print(f"  {passed_count}/{len(results)} passed  "
          f"{failed_count} failed  {skipped_count} skipped")
    print(BAR2)

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
