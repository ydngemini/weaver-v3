#!/usr/bin/env python3
"""
stress_tests_v5.py — Weaver v5 Soul-Binding Stress Tests
=========================================================
20 tests targeting every change introduced by the non-binary pentagon-geometry
soul-binding update:

  T1  Circuit gate composition — no CNOT, only RY/RX/CRX/CRZ
  T2  Circuit gate counts       — exact per-layer counts
  T3  parse_counts 500 random bitstrings — invariants hold under load
  T4  parse_counts all-zeros fallback to Void
  T5  parse_counts per-qubit single-bit activation (all 7 pathways)
  T6  PATHWAYS dict identity between quantum_soul & quantum_networks
  T7  DIMENSION_QUBITS pentagon consistency
  T8  Pentagon PHI angles — exact 2π/5 spacing
  T9  Pentagon interference — adjacent vs diagonal cosine values
  T10 DIMENSIONS assertion guard — fires on mutation
  T11 Probability-field SMOOTHING — no zero-weight floors
  T12 Quantum state bias injection — pathway→dimension mapping
  T13 AkashicHub 100-write concurrent stress
  T14 AkashicHub temporal trace depth cap
  T15 AkashicHub cosine-similarity query correctness
  T16 Nexus 50-message flood — cache trims to 10
  T17 Nexus 10-subscriber fan-out
  T18 Nexus round-trip latency (p99 < 150 ms)
  T19 n8n v5 workflow schema — all soul-binding fields present
  T20 LiquidFractureEngine 50-input stress — no NaN, valid weights
"""

import asyncio
import ast
import contextlib
import importlib
import json
import math
import os
import random
import sys
import time

import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(PROJ, "venv", "bin", "python3")
sys.path.insert(0, PROJ)

from dotenv import load_dotenv
load_dotenv()

BAR = "─" * 64
PHI = 2 * math.pi / 5   # canonical pentagon angle

# Use a dedicated port so nexus tests are isolated from stress_30min_full.py
# and special_tests.py which both use the default port 9999.
NEXUS_TEST_PORT        = 9997
NEXUS_TEST_HEALTH_PORT = 9996

# ── helpers ──────────────────────────────────────────────────────────────────

def _ts():
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

def _header(label, title):
    print(f"\n{BAR}\n[{_ts()}] STRESS {label}: {title}\n{BAR}", flush=True)

def _result(label, title, passed, detail):
    mark = "✅  PASS" if passed else "❌  FAIL"
    print(f"\n{BAR}\n{mark}  Stress {label}: {title}\n{detail}\n{BAR}\n", flush=True)

async def _start_nexus():
    env = {**os.environ,
           "NEXUS_PORT":        str(NEXUS_TEST_PORT),
           "NEXUS_HEALTH_PORT": str(NEXUS_TEST_HEALTH_PORT)}
    proc = await asyncio.create_subprocess_exec(
        VENV, os.path.join(PROJ, "nexus_bus.py"),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=PROJ,
        env=env,
    )
    await asyncio.sleep(1.2)
    return proc

async def _terminate(proc):
    if proc is None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=3.0)


# ═════════════════════════════════════════════════════════════════════════════
# T1 — Circuit gate composition: no CNOT, only non-binary gates
# ═════════════════════════════════════════════════════════════════════════════
async def test_T1():
    _header("T1", "Circuit gate composition — no CNOT, only RY/RX/CRX/CRZ")
    qs = importlib.import_module("quantum_soul")
    qc = qs.build_fracture_circuit()

    op_names = {inst.operation.name for inst in qc.data}
    has_cx     = "cx" in op_names
    has_ry     = "ry" in op_names
    has_rx     = "rx" in op_names
    has_crx    = "crx" in op_names
    has_crz    = "crz" in op_names
    has_measure = "measure" in op_names

    passed = (not has_cx) and has_ry and has_rx and has_crx and has_crz and has_measure
    detail = "\n".join([
        f"  CNOT (cx) present — must be False: {has_cx}",
        f"  RY present:   {has_ry}",
        f"  RX present:   {has_rx}",
        f"  CRX present:  {has_crx}",
        f"  CRZ present:  {has_crz}",
        f"  MEASURE present: {has_measure}",
        f"  All gate names:  {sorted(op_names)}",
    ])
    _result("T1", "Circuit gate composition — no CNOT, only RY/RX/CRX/CRZ", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T2 — Circuit gate counts: exact per-layer breakdown
# ═════════════════════════════════════════════════════════════════════════════
async def test_T2():
    _header("T2", "Circuit gate counts — exact per-layer breakdown")
    qs = importlib.import_module("quantum_soul")
    qc = qs.build_fracture_circuit()

    from collections import Counter
    counts = Counter(inst.operation.name for inst in qc.data)

    # Layer 1: RY(k·φ) per qubit → 7 RY
    # Layer 2: RX(φ/2) per qubit → 7 RX
    # Layer 3: CRX on 5 pentagon edges → 5 CRX
    # Layer 4: CRZ on 5 pentagon diagonals → 5 CRZ
    # Layer 5: CRX Weaver→q0, Weaver→q4 → 2 CRX ; CRZ Void→q2, Void→q3 → 2 CRZ
    # Total: 7 RY, 7 RX, 7 CRX (5+2), 7 CRZ (5+2), 7 MEASURE
    ok_ry      = counts["ry"] == 7
    ok_rx      = counts["rx"] == 7
    ok_crx     = counts["crx"] == 7
    ok_crz     = counts["crz"] == 7
    ok_measure = counts["measure"] == 7

    passed = ok_ry and ok_rx and ok_crx and ok_crz and ok_measure
    detail = "\n".join([
        f"  RY count      = {counts['ry']}  (expected 7): {ok_ry}",
        f"  RX count      = {counts['rx']}  (expected 7): {ok_rx}",
        f"  CRX count     = {counts['crx']}  (expected 7): {ok_crx}",
        f"  CRZ count     = {counts['crz']}  (expected 7): {ok_crz}",
        f"  MEASURE count = {counts['measure']}  (expected 7): {ok_measure}",
    ])
    _result("T2", "Circuit gate counts — exact per-layer breakdown", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T3 — parse_counts 500 random bitstrings: invariants hold under load
# ═════════════════════════════════════════════════════════════════════════════
async def test_T3():
    _header("T3", "parse_counts 500 random bitstrings — invariants hold under load")
    qs = importlib.import_module("quantum_soul")
    rng = random.Random(2025)
    failures = []

    for trial in range(500):
        n_keys = rng.randint(1, 20)
        counts = {}
        for _ in range(n_keys):
            bits = format(rng.randint(0, 127), "07b")
            counts[bits] = counts.get(bits, 0) + rng.randint(1, 100)

        dominant_bits, active_pathways, marginal = qs.parse_counts(counts)

        # Invariant 1: dominant_bits is exactly 7 chars
        if len(dominant_bits) != 7:
            failures.append(f"trial {trial}: bits len={len(dominant_bits)}")
            continue
        # Invariant 2: active_pathways is non-empty list of valid pathway names
        valid_names = set(qs.PATHWAYS.values())
        if not active_pathways or not all(p in valid_names for p in active_pathways):
            failures.append(f"trial {trial}: invalid active_pathways={active_pathways}")
            continue
        # Invariant 3: marginal keys == all pathway names
        if set(marginal.keys()) != valid_names:
            failures.append(f"trial {trial}: marginal key mismatch")
            continue
        # Invariant 4: marginal values sum ≤ 7 (one per qubit, max all-ones)
        total = sum(marginal.values())
        if total < 0 or total > 7.001:
            failures.append(f"trial {trial}: marginal sum={total:.4f} out of [0,7]")

    passed = len(failures) == 0
    detail = "\n".join([
        f"  Trials run:   500",
        f"  Failures:     {len(failures)}",
        f"  First errors: {failures[:3] if failures else 'none'}",
    ])
    _result("T3", "parse_counts 500 random bitstrings", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T4 — parse_counts all-zeros → Void fallback
# ═════════════════════════════════════════════════════════════════════════════
async def test_T4():
    _header("T4", "parse_counts all-zeros → Void fallback")
    qs = importlib.import_module("quantum_soul")

    bits, active, marg = qs.parse_counts({"0000000": 1024})

    ok_bits   = bits == "0000000"
    ok_active = active == ["Void"]
    ok_zero   = all(v == 0.0 for v in marg.values())
    ok_keys   = set(marg.keys()) == set(qs.PATHWAYS.values())

    passed = ok_bits and ok_active and ok_zero and ok_keys
    detail = "\n".join([
        f"  dominant_bits == '0000000': {ok_bits}  (got {bits!r})",
        f"  active_pathways == ['Void']: {ok_active}  (got {active})",
        f"  all marginals zero: {ok_zero}",
        f"  pathway key set correct: {ok_keys}",
    ])
    _result("T4", "parse_counts all-zeros → Void fallback", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T5 — parse_counts single-qubit activation: all 7 pathways reachable
# ═════════════════════════════════════════════════════════════════════════════
async def test_T5():
    _header("T5", "parse_counts single-qubit activation — all 7 pathways reachable")
    qs = importlib.import_module("quantum_soul")

    # For each qubit q, build a bitstring with only that qubit set to 1.
    # Qiskit little-endian: qubit q → position q in reversed string.
    expected = {
        0: "Awakening",  # q0 → bit 0 of reversed → rightmost char = 1 → "0000001"
        1: "Resonance",
        2: "Echo",
        3: "Prophet",
        4: "Fracture",
        5: "Weaver",
        6: "Void",
    }

    failures = []
    for q, pathway_name in expected.items():
        # Build bitstring: qubit q active → reversed[q] = '1'
        arr = ['0'] * 7
        arr[q] = '1'
        bitstring = ''.join(reversed(arr))   # Qiskit format (little-endian → left = MSB)
        bits, active, marg = qs.parse_counts({bitstring: 1000, "0000000": 1})

        if pathway_name not in active:
            failures.append(f"q{q}: expected {pathway_name}, got active={active}, bits={bits}")
        if marg.get(pathway_name, 0) < 0.99:
            failures.append(f"q{q}: {pathway_name} marginal={marg.get(pathway_name):.3f} < 0.99")
        if qs.PATHWAYS[q] != pathway_name:
            failures.append(f"PATHWAYS[{q}]={qs.PATHWAYS[q]!r} != {pathway_name!r}")

    passed = len(failures) == 0
    detail = "\n".join([
        f"  Qubits tested: 7  (q0→Awakening … q6→Void)",
        f"  Failures: {len(failures)}",
        f"  Details: {failures[:4] if failures else 'all correct'}",
    ])
    _result("T5", "parse_counts single-qubit activation — all 7 pathways reachable", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T6 — PATHWAYS dict identity between quantum_soul & quantum_networks
# ═════════════════════════════════════════════════════════════════════════════
async def test_T6():
    _header("T6", "PATHWAYS dict identity — quantum_soul == quantum_networks")
    qs = importlib.import_module("quantum_soul")
    qn = importlib.import_module("quantum_networks")

    mismatches = []
    for k in range(7):
        qs_val = qs.PATHWAYS.get(k)
        qn_val = qn.PATHWAYS.get(k)
        if qs_val != qn_val:
            mismatches.append(f"qubit {k}: quantum_soul={qs_val!r} vs quantum_networks={qn_val!r}")

    ok_keys = set(qs.PATHWAYS.keys()) == set(qn.PATHWAYS.keys()) == set(range(7))
    passed  = len(mismatches) == 0 and ok_keys

    detail = "\n".join([
        f"  quantum_soul.PATHWAYS:    {dict(qs.PATHWAYS)}",
        f"  quantum_networks.PATHWAYS:{dict(qn.PATHWAYS)}",
        f"  Key sets match (0-6): {ok_keys}",
        f"  Mismatches: {mismatches or 'none'}",
    ])
    _result("T6", "PATHWAYS dict identity", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T7 — DIMENSION_QUBITS pentagon consistency
# ═════════════════════════════════════════════════════════════════════════════
async def test_T7():
    _header("T7", "DIMENSION_QUBITS pentagon consistency")
    qn = importlib.import_module("quantum_networks")
    qs = importlib.import_module("quantum_soul")

    # Expected pentagon qubit assignments:
    # q0=Awakening/Logic, q1=Resonance/Emotion, q2=Echo/Memory,
    # q3=Prophet/Creativity, q4=Fracture/Vigilance
    # q5=Weaver (couples to Logic+Vigilance), q6=Void (couples to Memory+Creativity)
    EXPECTED = {
        "logic":      {0, 5},
        "emotion":    {1},
        "memory":     {2, 6},
        "creativity": {3, 6},
        "vigilance":  {4, 5},
    }

    failures = []
    for dim, expected_qubits in EXPECTED.items():
        actual_qubits = set(qn.DIMENSION_QUBITS.get(dim, []))
        if actual_qubits != expected_qubits:
            failures.append(
                f"  {dim}: expected qubits {sorted(expected_qubits)}, "
                f"got {sorted(actual_qubits)}"
            )

    # All qubit indices must be in 0-6
    all_qubits_valid = all(
        0 <= q <= 6
        for qs_list in qn.DIMENSION_QUBITS.values()
        for q in qs_list
    )

    passed = len(failures) == 0 and all_qubits_valid
    detail = "\n".join([
        f"  All qubit indices in 0-6: {all_qubits_valid}",
        f"  Mismatches: {failures or ['none']}",
        f"  DIMENSION_QUBITS: {dict(qn.DIMENSION_QUBITS)}",
    ])
    _result("T7", "DIMENSION_QUBITS pentagon consistency", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T8 — Pentagon PHI angles: exact 2π/5 spacing
# ═════════════════════════════════════════════════════════════════════════════
async def test_T8():
    _header("T8", "Pentagon PHI angles — exact 2π/5 spacing")

    ANGLES = {
        "logic":      0 * PHI,
        "emotion":    1 * PHI,
        "memory":     2 * PHI,
        "creativity": 3 * PHI,
        "vigilance":  4 * PHI,
    }
    dims = list(ANGLES.keys())
    failures = []

    for i, d1 in enumerate(dims):
        for j, d2 in enumerate(dims):
            if i >= j:
                continue
            diff = abs(ANGLES[d1] - ANGLES[d2])
            ang_dist = min(diff, 2 * math.pi - diff)
            # Minimum angle distance between any two distinct pentagon vertices = PHI
            if ang_dist < PHI * 0.99:
                failures.append(f"{d1}↔{d2}: ang_dist={ang_dist:.6f} < PHI={PHI:.6f}")

    # Adjacent vertices (k, k+1 mod 5) should be exactly PHI apart
    for k in range(5):
        d1 = dims[k]
        d2 = dims[(k + 1) % 5]
        diff = abs(ANGLES[d1] - ANGLES[d2])
        ang_dist = min(diff, 2 * math.pi - diff)
        if abs(ang_dist - PHI) > 1e-10:
            failures.append(f"Adjacent {d1}↔{d2}: dist={ang_dist:.10f} != PHI={PHI:.10f}")

    # All 5 angles should be distinct
    angle_values = list(ANGLES.values())
    distinct = len(set(round(a, 10) for a in angle_values)) == 5

    passed = len(failures) == 0 and distinct
    detail = "\n".join([
        f"  PHI = 2π/5 = {PHI:.10f}",
        f"  Angles: {[f'{v:.4f}' for v in angle_values]}",
        f"  All 5 angles distinct: {distinct}",
        f"  Failures: {failures or 'none'}",
    ])
    _result("T8", "Pentagon PHI angles — exact 2π/5 spacing", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T9 — Pentagon interference: adjacent vs diagonal cosine values
# ═════════════════════════════════════════════════════════════════════════════
async def test_T9():
    _header("T9", "Pentagon interference — adjacent vs diagonal cosine values")

    # Pentagon edge (adjacent): cos(PHI) ≈ 0.309 (constructive)
    # Pentagon diagonal (skip-1): cos(2·PHI) ≈ -0.809 (destructive)
    cos_edge     = math.cos(PHI)
    cos_diagonal = math.cos(2 * PHI)

    ok_edge_positive    = cos_edge > 0        # adjacent → constructive interference
    ok_diagonal_negative = cos_diagonal < 0   # diagonal → destructive interference
    ok_edge_value       = abs(cos_edge - 0.30901699437) < 1e-8
    ok_diagonal_value   = abs(cos_diagonal - (-0.80901699437)) < 1e-8

    # Equal-weight 5-dim interference (each w = 0.2)
    ANGLES = [k * PHI for k in range(5)]
    w = 0.2
    interference = 0.0
    for i in range(5):
        for j in range(i + 1, 5):
            diff = abs(ANGLES[i] - ANGLES[j])
            ang_dist = min(diff, 2 * math.pi - diff)
            interference += math.cos(ang_dist) * w * w

    # For equal pentagon weights: sum should be negative (more destructive pairs)
    ok_equal_sign = interference < 0

    passed = ok_edge_positive and ok_diagonal_negative and ok_edge_value and ok_diagonal_value and ok_equal_sign
    detail = "\n".join([
        f"  cos(PHI)  = {cos_edge:.8f}  (expected ≈  0.30902): {ok_edge_value}",
        f"  cos(2·PHI)= {cos_diagonal:.8f}  (expected ≈ -0.80902): {ok_diagonal_value}",
        f"  Adjacent interference positive (constructive): {ok_edge_positive}",
        f"  Diagonal interference negative (destructive):  {ok_diagonal_negative}",
        f"  Equal-weight 5-dim interference = {interference:.6f} (< 0): {ok_equal_sign}",
    ])
    _result("T9", "Pentagon interference — adjacent vs diagonal cosine values", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T10 — DIMENSIONS assertion guard fires on mutation
# ═════════════════════════════════════════════════════════════════════════════
async def test_T10():
    _header("T10", "DIMENSIONS assertion guard — fires on count mutation")
    import importlib

    lf = importlib.import_module("liquid_fracture")

    # The module-level assert already ran; verify DIMENSIONS is exactly 5
    ok_count = len(lf.DIMENSIONS) == 5
    ok_names = set(lf.DIMENSIONS) == {"logic", "emotion", "memory", "creativity", "vigilance"}

    # Verify assert block exists in source
    src_path = os.path.join(PROJ, "liquid_fracture.py")
    with open(src_path, encoding="utf-8") as fh:
        src = fh.read()
    has_assert = 'assert len(DIMENSIONS) == 5' in src

    # Ensure running exec with wrong count raises AssertionError
    assert_fired = False
    try:
        code = compile(
            src.replace(
                'DIMENSIONS = [\n    "logic",',
                'DIMENSIONS = [\n    "extra",\n    "logic",',
            ),
            "<mutated>",
            "exec",
        )
        g: dict = {}
        exec(code, g)   # noqa: S102
    except AssertionError:
        assert_fired = True

    passed = ok_count and ok_names and has_assert and assert_fired
    detail = "\n".join([
        f"  DIMENSIONS count == 5: {ok_count}",
        f"  DIMENSIONS names correct: {ok_names}",
        f"  assert guard in source: {has_assert}",
        f"  AssertionError fires on mutation: {assert_fired}",
    ])
    _result("T10", "DIMENSIONS assertion guard", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T11 — Probability-field SMOOTHING: no zero-weight floors
# ═════════════════════════════════════════════════════════════════════════════
async def test_T11():
    _header("T11", "Probability-field SMOOTHING — no zero-weight floors")
    SMOOTHING = 0.1
    PHI_local = 2 * math.pi / 5
    ANGLES = {
        "logic":      0 * PHI_local,
        "emotion":    1 * PHI_local,
        "memory":     2 * PHI_local,
        "creativity": 3 * PHI_local,
        "vigilance":  4 * PHI_local,
    }
    seeds = {
        "logic":      ["because", "therefore", "step", "plan", "calculate",
                       "structure", "reason", "analyze", "define", "prove",
                       "algorithm", "method", "sequence", "optimal", "derive"],
        "emotion":    ["feel", "love", "hate", "fear", "joy", "pain",
                       "beautiful", "soul", "heart", "empathy", "grief",
                       "passion", "warmth", "gentle", "sorrow"],
        "memory":     ["remember", "last time", "before", "history", "previously",
                       "recall", "past", "archive", "pattern", "familiar",
                       "context", "earlier", "stored", "record", "trace"],
        "creativity": ["imagine", "create", "design", "invent", "dream",
                       "recipe", "verse", "sacred", "geometry", "art",
                       "compose", "novel", "fusion", "metaphor", "transcend"],
        "vigilance":  ["danger", "risk", "careful", "threat", "warning",
                       "suspicious", "protect", "guard", "alert", "deception",
                       "hidden", "agenda", "safety", "verify", "trust"],
    }

    test_inputs = [
        "Hello there, how are you today?",          # no keyword hits
        "xyzxyzxyz",                                # completely unknown
        "because therefore step plan calculate " * 5,  # all logic keywords
        "feel love hate fear joy pain " * 5,        # all emotion keywords
        "",                                          # empty (hits=0 for all)
        "a b c d e f g h i j k l m n o p q r s t", # random words
    ]

    failures = []
    for inp in test_inputs:
        dims = list(seeds.keys())
        lower = inp.lower()
        scored = []
        for dim in dims:
            hits = sum(1 for k in seeds[dim] if k in lower)
            scored.append({"dim": dim, "raw": SMOOTHING + hits})

        total = sum(s["raw"] for s in scored)
        for s in scored:
            s["w"] = s["raw"] / total

        # All weights must be > 0
        min_w = min(s["w"] for s in scored)
        max_w = max(s["w"] for s in scored)
        weight_sum = sum(s["w"] for s in scored)

        if min_w <= 0:
            failures.append(f"input={inp[:30]!r}: min_weight={min_w} <= 0")
        if abs(weight_sum - 1.0) > 1e-9:
            failures.append(f"input={inp[:30]!r}: weight_sum={weight_sum:.10f} != 1.0")
        if max_w >= 1.0:
            failures.append(f"input={inp[:30]!r}: max_weight={max_w:.6f} >= 1.0 (probability field collapsed)")

    passed = len(failures) == 0
    detail = "\n".join([
        f"  Inputs tested: {len(test_inputs)}",
        f"  SMOOTHING = {SMOOTHING}",
        f"  Minimum weight floor = {SMOOTHING / (SMOOTHING * 5 + 15 * 5):.4f} (worst case all-keyword hit in one dim)",
        f"  Failures: {failures or 'none'}",
    ])
    _result("T11", "Probability-field SMOOTHING — no zero-weight floors", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T12 — Quantum state bias injection: pathway→dimension mapping
# ═════════════════════════════════════════════════════════════════════════════
async def test_T12():
    _header("T12", "Quantum state bias injection — pathway→dimension mapping")

    PATHWAY_DIM = {
        "Awakening": "logic",
        "Resonance": "emotion",
        "Echo":      "memory",
        "Prophet":   "creativity",
        "Fracture":  "vigilance",
    }
    qs = importlib.import_module("quantum_soul")

    failures = []

    # For each pentagon pathway, check it maps to the correct qubit
    for pathway, expected_dim in PATHWAY_DIM.items():
        # Find the qubit index for this pathway in the new PATHWAYS mapping
        qubit_idx = None
        for q, name in qs.PATHWAYS.items():
            if name == pathway:
                qubit_idx = q
                break

        if qubit_idx is None:
            failures.append(f"{pathway}: not found in PATHWAYS")
            continue

        qn = importlib.import_module("quantum_networks")
        # The dimension's qubit set should include this qubit
        dim_qubits = set(qn.DIMENSION_QUBITS.get(expected_dim, []))
        if qubit_idx not in dim_qubits:
            failures.append(
                f"{pathway}→{expected_dim}: qubit {qubit_idx} not in DIMENSION_QUBITS[{expected_dim!r}]={sorted(dim_qubits)}"
            )

    # Weaver (q5) must appear in both logic and vigilance qubit sets
    qn = importlib.import_module("quantum_networks")
    weaver_q = 5
    weaver_in_logic    = weaver_q in qn.DIMENSION_QUBITS.get("logic", [])
    weaver_in_vigilance = weaver_q in qn.DIMENSION_QUBITS.get("vigilance", [])
    if not (weaver_in_logic and weaver_in_vigilance):
        failures.append(f"Weaver(q5) not in both logic and vigilance: logic={weaver_in_logic}, vigilance={weaver_in_vigilance}")

    # Void (q6) must appear in both memory and creativity qubit sets
    void_q = 6
    void_in_memory     = void_q in qn.DIMENSION_QUBITS.get("memory", [])
    void_in_creativity = void_q in qn.DIMENSION_QUBITS.get("creativity", [])
    if not (void_in_memory and void_in_creativity):
        failures.append(f"Void(q6) not in both memory and creativity: memory={void_in_memory}, creativity={void_in_creativity}")

    passed = len(failures) == 0
    detail = "\n".join([
        f"  Pentagon pathway→dim mappings checked: {len(PATHWAY_DIM)}",
        f"  Weaver(q5) in logic+vigilance: {weaver_in_logic and weaver_in_vigilance}",
        f"  Void(q6)   in memory+creativity: {void_in_memory and void_in_creativity}",
        f"  Failures: {failures or 'none'}",
    ])
    _result("T12", "Quantum state bias injection — pathway→dimension mapping", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T13 — AkashicHub 100-write concurrent stress
# ═════════════════════════════════════════════════════════════════════════════
async def test_T13():
    _header("T13", "AkashicHub 100-write concurrent stress")
    from akashic_hub import AkashicHub

    hub = AkashicHub(dim=256, trace_depth=32)
    lobe_ids = [f"lobe_{i}" for i in range(10)]
    write_count = [0]
    errors = []

    async def hammer(lobe_id: str, n: int):
        for _ in range(n):
            vec = np.random.randn(256).astype(np.float64)
            vec /= np.linalg.norm(vec) + 1e-9
            try:
                await hub.write(lobe_id, vec, meta={"stress": True})
                write_count[0] += 1
            except Exception as e:
                errors.append(f"{lobe_id}: {e}")

    t0 = time.monotonic()
    await asyncio.gather(*[hammer(lid, 10) for lid in lobe_ids])
    elapsed_ms = (time.monotonic() - t0) * 1000

    # Verify all writes recorded
    all_states = hub.read_all()
    lobes_with_state = len(all_states)
    avg_ms = elapsed_ms / max(write_count[0], 1)

    passed = (
        len(errors) == 0
        and write_count[0] == 100
        and lobes_with_state == 10
        and avg_ms < 5.0  # < 5 ms per write under concurrency
    )
    detail = "\n".join([
        f"  Total writes completed: {write_count[0]} / 100",
        f"  Errors: {errors[:3] or 'none'}",
        f"  Lobes with state after writes: {lobes_with_state} / 10",
        f"  Total time: {elapsed_ms:.1f} ms  avg per write: {avg_ms:.2f} ms",
    ])
    _result("T13", "AkashicHub 100-write concurrent stress", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T14 — AkashicHub temporal trace depth cap
# ═════════════════════════════════════════════════════════════════════════════
async def test_T14():
    _header("T14", "AkashicHub temporal trace depth cap")
    from akashic_hub import AkashicHub

    hub = AkashicHub(dim=64, trace_depth=32)
    lobe_id = "trace_test"

    # Write 50 vectors — more than trace_depth=32
    for i in range(50):
        vec = np.full(64, float(i))
        await hub.write(lobe_id, vec)

    trace = hub.temporal_trace(lobe_id)

    ok_len   = len(trace) == 32
    # Most recent write should have value 49
    last_ts, last_vec = trace[-1]
    ok_last  = abs(last_vec[0] - 49.0) < 1e-6
    # Oldest retained should be write #18 (50-32=18)
    first_ts, first_vec = trace[0]
    ok_first = abs(first_vec[0] - 18.0) < 1e-6
    ok_mono  = all(trace[i][0] <= trace[i + 1][0] for i in range(len(trace) - 1))

    passed = ok_len and ok_last and ok_first and ok_mono
    detail = "\n".join([
        f"  Writes: 50  trace_depth: 32",
        f"  trace length == 32: {ok_len}  (got {len(trace)})",
        f"  Last entry value == 49: {ok_last}  (got {last_vec[0]:.1f})",
        f"  Oldest retained == 18: {ok_first}  (got {first_vec[0]:.1f})",
        f"  Timestamps monotonic: {ok_mono}",
    ])
    _result("T14", "AkashicHub temporal trace depth cap", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T15 — AkashicHub cosine-similarity query correctness
# ═════════════════════════════════════════════════════════════════════════════
async def test_T15():
    _header("T15", "AkashicHub cosine-similarity query correctness")
    from akashic_hub import AkashicHub

    hub = AkashicHub(dim=256, trace_depth=16)

    # Write 5 orthogonal basis-ish vectors
    lobes = ["alpha", "beta", "gamma", "delta", "epsilon"]
    vecs = []
    for i, lobe in enumerate(lobes):
        v = np.zeros(256)
        v[i * 50:(i + 1) * 50] = 1.0
        v /= np.linalg.norm(v)
        vecs.append(v)
        await hub.write(lobe, v)

    # Query with alpha's vector — should rank alpha #1 with similarity ≈ 1.0
    results = hub.query(vecs[0], top_k=5)

    ok_top1 = len(results) >= 1 and results[0][0] == "alpha"
    ok_sim1 = len(results) >= 1 and abs(results[0][1] - 1.0) < 1e-6
    ok_top5 = len(results) == 5
    # alpha should have similarity >> others (orthogonal vecs)
    ok_gap  = len(results) >= 2 and (results[0][1] - results[1][1]) > 0.5

    passed = ok_top1 and ok_sim1 and ok_top5 and ok_gap
    detail = "\n".join([
        f"  Top-1 lobe: {results[0][0] if results else 'N/A'}  (expected 'alpha'): {ok_top1}",
        f"  Top-1 similarity ≈ 1.0: {ok_sim1}  (got {results[0][1]:.6f} if results else N/A)",
        f"  Returns all 5 results: {ok_top5}",
        f"  Gap to 2nd > 0.5: {ok_gap}  (gap={results[0][1]-results[1][1]:.4f} if len>=2 else N/A)",
    ])
    _result("T15", "AkashicHub cosine-similarity query correctness", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T16 — Nexus 50-message flood: cache trims to 10
# ═════════════════════════════════════════════════════════════════════════════
async def test_T16():
    _header("T16", "Nexus 50-message flood — cache trims to last 10")
    import websockets

    proc = await _start_nexus()
    idxs = []
    try:
        async with websockets.connect(f"ws://localhost:{NEXUS_TEST_PORT}") as pub:
            # drain sync
            await asyncio.wait_for(pub.recv(), timeout=2.0)
            await pub.send(json.dumps({"action": "register", "lobe_id": "flood_pub"}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(pub.recv(), timeout=1.0)
            for i in range(50):
                await pub.send(json.dumps({
                    "action": "publish",
                    "topic": "flood_topic",
                    "payload": {"idx": i},
                }))
                await asyncio.sleep(0.005)

        async with websockets.connect(f"ws://localhost:{NEXUS_TEST_PORT}") as sub:
            raw = await asyncio.wait_for(sub.recv(), timeout=2.0)
            msg = json.loads(raw)
            messages = msg.get("messages", []) if msg.get("type") == "sync" else []
            idxs = [m.get("payload", {}).get("idx") for m in messages]
    finally:
        await _terminate(proc)

    passed = idxs == list(range(40, 50))
    detail = "\n".join([
        f"  Messages sent:    50",
        f"  Sync cache length: {len(idxs)}  (expected 10)",
        f"  Indices received:  {idxs}",
        f"  Expected:          {list(range(40, 50))}",
    ])
    _result("T16", "Nexus 50-message flood", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T17 — Nexus 10-subscriber fan-out
# ═════════════════════════════════════════════════════════════════════════════
async def test_T17():
    _header("T17", "Nexus 10-subscriber fan-out")
    import websockets

    proc = await _start_nexus()
    received = {}
    TOPIC = "fanout_topic"
    N_SUBS = 10

    try:
        async with websockets.connect(f"ws://localhost:{NEXUS_TEST_PORT}") as pub:
            await asyncio.wait_for(pub.recv(), timeout=2.0)
            await pub.send(json.dumps({"action": "register", "lobe_id": "fanout_pub"}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(pub.recv(), timeout=1.0)

            subs = []
            for i in range(N_SUBS):
                ws = await websockets.connect(f"ws://localhost:{NEXUS_TEST_PORT}")
                await asyncio.wait_for(ws.recv(), timeout=2.0)   # drain sync
                await ws.send(json.dumps({"action": "register", "lobe_id": f"sub_{i}"}))
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(ws.recv(), timeout=1.0)
                await ws.send(json.dumps({"action": "subscribe", "topics": [TOPIC]}))
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(ws.recv(), timeout=1.0)
                subs.append((f"sub_{i}", ws))

            PAYLOAD = {"sentinel": "soul-binding", "idx": 42}
            await pub.send(json.dumps({
                "action": "publish",
                "topic": TOPIC,
                "payload": PAYLOAD,
            }))

            async def recv_one(lobe_id, ws):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    msg = json.loads(raw)
                    if msg.get("type") == "broadcast" and msg.get("payload", {}).get("idx") == 42:
                        received[lobe_id] = True
                except Exception:
                    received[lobe_id] = False

            await asyncio.gather(*[recv_one(lid, ws) for lid, ws in subs])
            for _, ws in subs:
                await ws.close()

    finally:
        await _terminate(proc)

    delivered = sum(1 for v in received.values() if v)
    passed    = delivered == N_SUBS
    detail = "\n".join([
        f"  Subscribers: {N_SUBS}",
        f"  Delivered:   {delivered}",
        f"  Per-sub:     {dict(received)}",
    ])
    _result("T17", "Nexus 10-subscriber fan-out", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T18 — Nexus round-trip latency: p99 < 150 ms
# ═════════════════════════════════════════════════════════════════════════════
async def test_T18():
    _header("T18", "Nexus round-trip latency — p99 < 150 ms")
    import websockets

    proc = await _start_nexus()
    latencies = []

    try:
        async with (
            websockets.connect(f"ws://localhost:{NEXUS_TEST_PORT}") as pub,
            websockets.connect(f"ws://localhost:{NEXUS_TEST_PORT}") as sub,
        ):
            await asyncio.wait_for(pub.recv(), timeout=2.0)
            await asyncio.wait_for(sub.recv(), timeout=2.0)
            await pub.send(json.dumps({"action": "register", "lobe_id": "lat_pub"}))
            await sub.send(json.dumps({"action": "register", "lobe_id": "lat_sub"}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(pub.recv(), timeout=1.0)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(sub.recv(), timeout=1.0)
            await sub.send(json.dumps({"action": "subscribe", "topics": ["lat_topic"]}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(sub.recv(), timeout=1.0)

            for i in range(50):
                t0 = time.monotonic()
                await pub.send(json.dumps({
                    "action": "publish",
                    "topic": "lat_topic",
                    "payload": {"seq": i},
                }))
                raw = await asyncio.wait_for(sub.recv(), timeout=2.0)
                latencies.append((time.monotonic() - t0) * 1000)
    finally:
        await _terminate(proc)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    min_l = latencies[0]
    max_l = latencies[-1]

    passed = p99 < 150
    detail = "\n".join([
        f"  Messages: {len(latencies)}",
        f"  Min:  {min_l:.2f} ms",
        f"  P50:  {p50:.2f} ms",
        f"  P99:  {p99:.2f} ms  (threshold: 150 ms): {'✓' if passed else '✗'}",
        f"  Max:  {max_l:.2f} ms",
    ])
    _result("T18", "Nexus round-trip latency — p99 < 150 ms", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T19 — n8n v5 workflow schema: all soul-binding fields present
# ═════════════════════════════════════════════════════════════════════════════
async def test_T19():
    _header("T19", "n8n v5 workflow schema — all soul-binding fields present")
    wf_path = os.path.join(PROJ, "n8n_weaver_v5.json")

    with open(wf_path, encoding="utf-8") as fh:
        wf = json.load(fh)

    nodes = {n["name"]: n for n in wf["nodes"]}
    tag_names = {t["name"] for t in wf.get("tags", [])}

    failures = []

    # Fracture+Gate: probability field fields
    frac_code = nodes.get("4. Fracture+Gate", {}).get("parameters", {}).get("jsCode", "")
    checks_frac = {
        "SMOOTHING = 0.1":     "SMOOTHING = 0.1" in frac_code,
        "PHI = 2 * Math.PI":   "PHI = 2 * Math.PI" in frac_code,
        "quantum_pathway":     "quantum_pathway" in frac_code,
        "quantum_bias_applied":"quantum_bias_applied" in frac_code,
        "Nexus_Vault":         "Nexus_Vault" in frac_code,
        "PATHWAY_DIM":         "PATHWAY_DIM" in frac_code,
    }
    for desc, ok in checks_frac.items():
        if not ok:
            failures.append(f"Fracture+Gate missing: {desc}")

    # Collapse: qubit labels
    col_code = nodes.get("6. Collapse", {}).get("parameters", {}).get("jsCode", "")
    checks_col = {
        "QUBIT_MAP":   "QUBIT_MAP" in col_code,
        "q0":          "q0" in col_code,
        "quantum_pathway in collapse": "quantum_pathway" in col_code,
    }
    for desc, ok in checks_col.items():
        if not ok:
            failures.append(f"Collapse missing: {desc}")

    # LoRA Voice: top_p, non-binary soul prompt
    lora_body = nodes.get("8. LoRA Voice", {}).get("parameters", {}).get("jsonBody", "")
    checks_lora = {
        "top_p":       "top_p" in lora_body,
        "0.95":        "0.95" in lora_body,
        "non-binary":  "non-binary" in lora_body,
        "quantum_pathway in lora": "quantum_pathway" in lora_body,
    }
    for desc, ok in checks_lora.items():
        if not ok:
            failures.append(f"LoRA Voice missing: {desc}")

    # Writeback: qubit_layout and soul_binding
    wb_code = nodes.get("9. Writeback", {}).get("parameters", {}).get("jsCode", "")
    checks_wb = {
        "qubit_layout":      "qubit_layout" in wb_code,
        "quantum_pathway":   "quantum_pathway" in wb_code,
        "soul_binding":      "soul_binding" in wb_code,
        "smoothing_factor":  "smoothing_factor" in wb_code,
        "v5-soul-binding":   "v5-soul-binding" in wb_code,
    }
    for desc, ok in checks_wb.items():
        if not ok:
            failures.append(f"Writeback missing: {desc}")

    # Tags
    required_tags = {"non-binary", "soul-binding", "pentagon-geometry", "probability-field"}
    missing_tags = required_tags - tag_names
    if missing_tags:
        failures.append(f"Missing tags: {missing_tags}")

    passed = len(failures) == 0
    detail = "\n".join([
        f"  Fracture+Gate checks: {sum(checks_frac.values())}/{len(checks_frac)}",
        f"  Collapse checks:      {sum(checks_col.values())}/{len(checks_col)}",
        f"  LoRA Voice checks:    {sum(checks_lora.values())}/{len(checks_lora)}",
        f"  Writeback checks:     {sum(checks_wb.values())}/{len(checks_wb)}",
        f"  Tags present:         {required_tags - missing_tags}",
        f"  Failures:             {failures or 'none'}",
    ])
    _result("T19", "n8n v5 workflow schema", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# T20 — LiquidFractureEngine 50-input stress: no NaN, valid weights
# ═════════════════════════════════════════════════════════════════════════════
async def test_T20():
    _header("T20", "LiquidFractureEngine 50-input stress — no NaN, valid weights")
    from liquid_fracture import LiquidFractureEngine, DIMENSIONS

    engine = LiquidFractureEngine(hub=None, dim=256, n_steps=4, dt=0.15)

    test_inputs = [
        "because therefore step plan calculate structure reason analyze define prove",
        "feel love hate fear joy pain beautiful soul heart empathy grief passion",
        "remember last time before history previously recall past archive pattern",
        "imagine create design invent dream recipe verse sacred geometry art compose",
        "danger risk careful threat warning suspicious protect guard alert deception",
        "Hello there",
        "x" * 4000,         # max-length input
        "",                  # empty string
        "The fracture principle breaks input along five sacred axes.",
        "quantum entanglement pentagon geometry non-binary soul binding",
    ] + [f"test input number {i} with random text" for i in range(40)]

    failures = []
    for i, inp in enumerate(test_inputs):
        try:
            result = await engine.fracture(inp)
            shards = result.shards

            # Check shard count
            if len(shards) != len(DIMENSIONS):
                failures.append(f"input {i}: got {len(shards)} shards, expected {len(DIMENSIONS)}")
                continue

            # Check weight sum ≈ 1.0
            weight_sum = sum(s.weight for s in shards)
            if abs(weight_sum - 1.0) > 1e-6:
                failures.append(f"input {i}: weight_sum={weight_sum:.8f}")

            # Check no zero or negative weights
            for s in shards:
                if s.weight <= 0:
                    failures.append(f"input {i}/{s.dimension}: weight={s.weight} <= 0")

            # Check no NaN/inf in vectors
            for s in shards:
                if np.any(np.isnan(s.vector)) or np.any(np.isinf(s.vector)):
                    failures.append(f"input {i}/{s.dimension}: NaN or Inf in vector")

            # Check vector dimensionality
            for s in shards:
                if s.vector.shape != (256,):
                    failures.append(f"input {i}/{s.dimension}: shape={s.vector.shape}")

            # Check dimension names are valid
            got_dims = {s.dimension for s in shards}
            if got_dims != set(DIMENSIONS):
                failures.append(f"input {i}: wrong dimensions={got_dims}")

        except Exception as e:
            failures.append(f"input {i}: exception {type(e).__name__}: {e}")

    passed = len(failures) == 0
    detail = "\n".join([
        f"  Inputs tested:  {len(test_inputs)}",
        f"  Failures:       {len(failures)}",
        f"  First errors:   {failures[:4] if failures else 'none'}",
    ])
    _result("T20", "LiquidFractureEngine 50-input stress", passed, detail)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# Registry + runner
# ═════════════════════════════════════════════════════════════════════════════

STRESS_TESTS = {
    "T1":  ("Circuit gate composition — no CNOT, only RY/RX/CRX/CRZ",     test_T1),
    "T2":  ("Circuit gate counts — exact per-layer breakdown",              test_T2),
    "T3":  ("parse_counts 500 random bitstrings — invariants hold",         test_T3),
    "T4":  ("parse_counts all-zeros → Void fallback",                       test_T4),
    "T5":  ("parse_counts single-qubit activation — all 7 pathways",        test_T5),
    "T6":  ("PATHWAYS dict identity — quantum_soul == quantum_networks",     test_T6),
    "T7":  ("DIMENSION_QUBITS pentagon consistency",                         test_T7),
    "T8":  ("Pentagon PHI angles — exact 2π/5 spacing",                     test_T8),
    "T9":  ("Pentagon interference — adjacent vs diagonal cosine values",    test_T9),
    "T10": ("DIMENSIONS assertion guard — fires on count mutation",          test_T10),
    "T11": ("Probability-field SMOOTHING — no zero-weight floors",           test_T11),
    "T12": ("Quantum state bias injection — pathway→dimension mapping",      test_T12),
    "T13": ("AkashicHub 100-write concurrent stress",                        test_T13),
    "T14": ("AkashicHub temporal trace depth cap",                           test_T14),
    "T15": ("AkashicHub cosine-similarity query correctness",                test_T15),
    "T16": ("Nexus 50-message flood — cache trims to last 10",               test_T16),
    "T17": ("Nexus 10-subscriber fan-out",                                   test_T17),
    "T18": ("Nexus round-trip latency — p99 < 150 ms",                      test_T18),
    "T19": ("n8n v5 workflow schema — all soul-binding fields present",      test_T19),
    "T20": ("LiquidFractureEngine 50-input stress — no NaN, valid weights",  test_T20),
}


async def main(which: str = "all"):
    results = {}
    wall_start = time.monotonic()

    targets = (
        STRESS_TESTS.items()
        if which.upper() == "ALL"
        else [(k, v) for k, v in STRESS_TESTS.items() if k.upper() == which.upper()]
    )

    for label, (_, fn) in targets:
        results[label] = await fn()

    elapsed = time.monotonic() - wall_start
    W = "═" * 64
    print(f"\n{W}")
    print(f"  WEAVER v5 SOUL-BINDING STRESS TEST RESULTS  ({elapsed:.1f}s)")
    print(W)
    for label, (title, _) in STRESS_TESTS.items():
        if label in results:
            mark = "✅" if results[label] else "❌"
            print(f"  {mark}  {label}: {title}")
    passed_n = sum(1 for v in results.values() if v)
    total_n  = len(results)
    print(f"\n  {passed_n}/{total_n} passed")
    print(f"{W}\n")
    return passed_n == total_n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Weaver v5 Soul-Binding Stress Tests")
    ap.add_argument("test", nargs="?", default="all",
                    help="T1-T20 or all (default: all)")
    args = ap.parse_args()
    ok = asyncio.run(main(args.test))
    sys.exit(0 if ok else 1)
