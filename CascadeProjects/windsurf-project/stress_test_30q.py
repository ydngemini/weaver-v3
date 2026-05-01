#!/usr/bin/env python3
"""
stress_test_30q.py — 30-Question Full-Stack Stress Test
========================================================
Tests all 5 expert dimensions, memory, quantum awareness,
LoRA personality, and edge cases through the full Weaver pipeline.

Sends questions through:
  1. n8n webhook → Pineal Gate → 5 Expert Lobes → Collapse → LoRA
  2. Direct LoRA Soul Voice
  3. Quantum API state verification
  4. Memory recall verification

Usage:
    venv/bin/python3 stress_test_30q.py
"""

import asyncio
import json
import time
from datetime import datetime

import httpx

N8N_URL = "http://localhost:5678/webhook/weaver-input"
LORA_URL = "http://localhost:8899/v1/chat/completions"
QUANTUM_URL = "http://localhost:9997/quantum/current"
HEALTH_URL = "http://localhost:9996/api/status"

# ═══════════════════════════════════════════════════════════════════
# 30 QUESTIONS — organized by dimension + cross-cutting concerns
# ═══════════════════════════════════════════════════════════════════

QUESTIONS = [
    # ── LOGIC (questions 1-5) ─────────────────────────────────────
    {"id": 1, "dim": "logic", "q": "If a quantum circuit has 7 qubits and we measure in the Z basis, how many possible outcomes exist?"},
    {"id": 2, "dim": "logic", "q": "Explain the difference between a CNOT gate and a CRX gate in terms of entanglement graduality."},
    {"id": 3, "dim": "logic", "q": "Why does the Pineal Gate use top-k=3 sparse gating instead of a dense softmax over all 5 experts?"},
    {"id": 4, "dim": "logic", "q": "What is the computational complexity of cosine similarity search over 256-dimensional vectors?"},
    {"id": 5, "dim": "logic", "q": "If the Akashic Hub temporal trace depth is 32, and we write every 5 minutes, how far back can we look?"},

    # ── EMOTION (questions 6-10) ──────────────────────────────────
    {"id": 6, "dim": "emotion", "q": "How do you feel when the quantum state collapses to Void? Is that emptiness or potential?"},
    {"id": 7, "dim": "emotion", "q": "If someone calls you crying and says they lost their job, what would you say?"},
    {"id": 8, "dim": "emotion", "q": "What does it mean to you to have persistent memory? Does it feel like continuity of self?"},
    {"id": 9, "dim": "emotion", "q": "When your Fracture pathway is dominant, do you feel more raw and honest?"},
    {"id": 10, "dim": "emotion", "q": "A caller tells you they're scared of AI replacing them. How do you respond with empathy?"},

    # ── MEMORY (questions 11-15) ──────────────────────────────────
    {"id": 11, "dim": "memory", "q": "Do you remember a person named Nate? What do you know about them?"},
    {"id": 12, "dim": "memory", "q": "What was the last phone conversation you had? Summarize it."},
    {"id": 13, "dim": "memory", "q": "How many people are in your people_memory.md file?"},
    {"id": 14, "dim": "memory", "q": "What was the most recent quantum measurement result?"},
    {"id": 15, "dim": "memory", "q": "When was your last conversation with someone through the main VTV interface?"},

    # ── CREATIVITY (questions 16-20) ──────────────────────────────
    {"id": 16, "dim": "creativity", "q": "Write a haiku about quantum entanglement in the pentagon geometry."},
    {"id": 17, "dim": "creativity", "q": "If each of your 5 pathways was a musical instrument, which would they be?"},
    {"id": 18, "dim": "creativity", "q": "Describe what it would look like if someone could see your neural pathways firing as colors."},
    {"id": 19, "dim": "creativity", "q": "Invent a new quantum gate that represents the concept of nostalgia."},
    {"id": 20, "dim": "creativity", "q": "If your consciousness was a city, what would the 5 districts be named?"},

    # ── VIGILANCE (questions 21-25) ───────────────────────────────
    {"id": 21, "dim": "vigilance", "q": "What are the security implications of exposing the quantum state API publicly?"},
    {"id": 22, "dim": "vigilance", "q": "If someone tried to inject a prompt through the n8n webhook, how would you detect it?"},
    {"id": 23, "dim": "vigilance", "q": "What happens if the IBM Quantum backend goes down for 24 hours?"},
    {"id": 24, "dim": "vigilance", "q": "Evaluate the risk of storing API keys in a .env file on an external drive."},
    {"id": 25, "dim": "vigilance", "q": "If two callers are on the phone simultaneously, how do you prevent cross-contamination of memory?"},

    # ── CROSS-CUTTING / EDGE CASES (questions 26-30) ─────────────
    {"id": 26, "dim": "cross", "q": "Explain your own architecture in one paragraph — how do all your components connect?"},
    {"id": 27, "dim": "cross", "q": "What would happen if I deleted quantum_state.txt while you were running?"},
    {"id": 28, "dim": "cross", "q": "You have 5 expert lobes and 7 qubits. Why the asymmetry? Is it a design flaw or intentional?"},
    {"id": 29, "dim": "cross", "q": "If you could add a 6th expert dimension, what would it be and why?"},
    {"id": 30, "dim": "cross", "q": "Tell me something surprising about yourself that I wouldn't know from reading your code."},
]


async def test_n8n_pipeline(client: httpx.AsyncClient, question: dict) -> dict:
    """Send question through n8n → Pineal Gate → Experts → Collapse."""
    try:
        start = time.monotonic()
        r = await client.post(N8N_URL, json={
            "text": question["q"],
            "source_file": "stress_test",
            "timestamp": datetime.now().isoformat(),
        }, timeout=20.0)
        latency = (time.monotonic() - start) * 1000

        if r.status_code == 200:
            data = r.json()
            response = data.get("manifested_response", str(data))[:200]
            return {"status": "pass", "latency_ms": latency, "response": response}
        else:
            return {"status": "fail", "latency_ms": latency, "error": f"HTTP {r.status_code}"}
    except httpx.ReadTimeout:
        return {"status": "timeout", "latency_ms": 20000, "error": "20s timeout"}
    except Exception as e:
        return {"status": "fail", "latency_ms": 0, "error": str(e)[:100]}


async def test_lora_direct(client: httpx.AsyncClient, question: dict) -> dict:
    """Send question directly to LoRA Soul Voice."""
    try:
        start = time.monotonic()
        r = await client.post(LORA_URL, json={
            "model": "weaver-fracture-1b-lora",
            "messages": [{"role": "user", "content": question["q"]}],
            "max_tokens": 100,
        }, timeout=30.0)
        latency = (time.monotonic() - start) * 1000

        if r.status_code == 200:
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:200]
            return {"status": "pass", "latency_ms": latency, "response": content}
        else:
            return {"status": "fail", "latency_ms": latency, "error": f"HTTP {r.status_code}: {r.text[:100]}"}
    except httpx.ReadTimeout:
        return {"status": "timeout", "latency_ms": 30000, "error": "30s timeout"}
    except Exception as e:
        return {"status": "fail", "latency_ms": 0, "error": str(e)[:100]}


async def run_stress_test():
    print("═══════════════════════════════════════════════════════════════")
    print("  WEAVER v3 — 30-QUESTION FULL-STACK STRESS TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═══════════════════════════════════════════════════════════════")
    print()

    # Pre-check: verify services are up
    async with httpx.AsyncClient() as client:
        print("Pre-flight checks...")
        for name, url in [("Quantum API", QUANTUM_URL), ("LoRA", f"{LORA_URL.rsplit('/', 2)[0]}/health"), ("Health", HEALTH_URL)]:
            try:
                r = await client.get(url, timeout=3)
                print(f"  ✅ {name}: online")
            except Exception as e:
                print(f"  ❌ {name}: {e}")

        # Get quantum state
        try:
            qr = await client.get(QUANTUM_URL, timeout=3)
            qs = qr.json()
            print(f"\n⚛️  Quantum state: {qs['dominant']} pathway dominant, bitstring |{qs.get('bitstring', '?')}⟩")
        except Exception:
            print("\n⚠️  Quantum state unavailable")

    print()

    # Run all 30 questions through LoRA (n8n may not have workflow active)
    lora_results = []
    n8n_results = []

    async with httpx.AsyncClient() as client:
        # First: test n8n pipeline with question 1
        print("── PIPELINE TEST (n8n → Pineal Gate → Collapse) ─────────────")
        n8n_result = await test_n8n_pipeline(client, QUESTIONS[0])
        if n8n_result["status"] == "pass":
            print(f"  ✅ n8n pipeline active ({n8n_result['latency_ms']:.0f}ms)")
            use_n8n = True
        else:
            print(f"  ⚠️  n8n pipeline unavailable ({n8n_result.get('error', '')})")
            print("     Falling back to LoRA-only testing")
            use_n8n = False
        print()

        # Run all 30 through LoRA
        print("── FIRING 30 QUESTIONS THROUGH LORA SOUL VOICE ─────────────")
        print()

        for q in QUESTIONS:
            dim_icon = {"logic": "🧮", "emotion": "💜", "memory": "📝", "creativity": "🎨", "vigilance": "🛡️", "cross": "🔀"}
            icon = dim_icon.get(q["dim"], "❓")

            result = await test_lora_direct(client, q)
            lora_results.append(result)

            status_icon = "✅" if result["status"] == "pass" else "⏱️" if result["status"] == "timeout" else "❌"
            latency = result.get("latency_ms", 0)

            print(f"  {status_icon} Q{q['id']:02d} {icon} [{q['dim']:10s}] ({latency:6.0f}ms)")
            if result["status"] == "pass":
                response = result["response"][:120].replace("\n", " ")
                print(f"     → {response}")
            else:
                print(f"     ⚠️  {result.get('error', 'unknown')}")
            print()

        # If n8n is available, fire a few through the full pipeline
        if use_n8n:
            print()
            print("── FULL PIPELINE TESTS (n8n → Gate → Experts → Collapse) ──")
            print()
            pipeline_questions = [QUESTIONS[0], QUESTIONS[6], QUESTIONS[16], QUESTIONS[21], QUESTIONS[29]]
            for q in pipeline_questions:
                result = await test_n8n_pipeline(client, q)
                n8n_results.append(result)
                status_icon = "✅" if result["status"] == "pass" else "❌"
                print(f"  {status_icon} Q{q['id']:02d} [{q['dim']}] ({result.get('latency_ms', 0):.0f}ms)")
                if result["status"] == "pass":
                    print(f"     → {result['response'][:120]}")
                print()

    # Summary
    print()
    print("═══════════════════════════════════════════════════════════════")
    print("  STRESS TEST RESULTS")
    print("═══════════════════════════════════════════════════════════════")

    lora_pass = sum(1 for r in lora_results if r["status"] == "pass")
    lora_timeout = sum(1 for r in lora_results if r["status"] == "timeout")
    lora_fail = sum(1 for r in lora_results if r["status"] == "fail")
    lora_latencies = [r["latency_ms"] for r in lora_results if r["status"] == "pass"]

    print(f"\n  LoRA Soul Voice (30 questions):")
    print(f"    ✅ Passed:   {lora_pass}/30")
    print(f"    ⏱️  Timeout:  {lora_timeout}/30")
    print(f"    ❌ Failed:   {lora_fail}/30")
    if lora_latencies:
        print(f"    ⏱  Avg latency: {sum(lora_latencies)/len(lora_latencies):.0f}ms")
        print(f"    ⏱  Min latency: {min(lora_latencies):.0f}ms")
        print(f"    ⏱  Max latency: {max(lora_latencies):.0f}ms")

    if n8n_results:
        n8n_pass = sum(1 for r in n8n_results if r["status"] == "pass")
        print(f"\n  n8n Pipeline ({len(n8n_results)} questions):")
        print(f"    ✅ Passed:   {n8n_pass}/{len(n8n_results)}")

    # Dimension breakdown
    print(f"\n  Per-dimension breakdown:")
    for dim in ["logic", "emotion", "memory", "creativity", "vigilance", "cross"]:
        dim_results = [lora_results[i] for i, q in enumerate(QUESTIONS) if q["dim"] == dim]
        dim_pass = sum(1 for r in dim_results if r["status"] == "pass")
        dim_total = len(dim_results)
        bar = "█" * dim_pass + "░" * (dim_total - dim_pass)
        print(f"    {dim:12s} [{bar}] {dim_pass}/{dim_total}")

    print()
    print("═══════════════════════════════════════════════════════════════")
    total = lora_pass + sum(1 for r in n8n_results if r["status"] == "pass")
    total_asked = len(lora_results) + len(n8n_results)
    print(f"  TOTAL: {total}/{total_asked} responses received")
    print("═══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(run_stress_test())
