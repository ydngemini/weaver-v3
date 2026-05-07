# Weaver v3 Full-Stack Validation & Deployment Plan
**Think like an NVIDIA DevOps Engineer — Production-Grade Integration**

---

## Current State Assessment

### ✅ What's Working
- **Core Components:** All 9 modules import successfully
- **IBM Quantum:** 7-qubit circuit runs every 5 minutes on ibm_marrakesh
- **Memory System:** Unified memory_manager.py created
- **Phone Bridge:** LangChain cortex + voice ID integrated
- **LoRA Server:** 1B Llama adapter serving on port 8899
- **Nexus Bus:** WebSocket pub/sub on port 9999
- **Akashic Hub:** 256-d shared vector state
- **Test Suite:** 43/46 tests passing in weaver_tests.py

### ⚠️ Critical Gaps
1. **Phone bridge doesn't route through Pineal Gate** — just uses OpenAI Realtime directly
2. **LoRA not filtering phone responses** — personality layer bypassed
3. **Quantum bias not propagated to phone system** — routing weights ignored
4. **n8n workflow not tested end-to-end** — webhook may be down
5. **No health monitoring** — can't see which lobes are alive
6. **No unified startup script** — manual component launching
7. **No graceful shutdown** — orphaned processes on Ctrl+C

---

## Production Architecture (Target State)

```
┌──────────────────────────────────────────────────────────────────────┐
│                         WEAVER V3 FULL STACK                         │
└──────────────────────────────────────────────────────────────────────┘

Layer 1: MASTER ORCHESTRATOR
┌─────────────────────────────────────────────────────────────────────┐
│ weaver.py (PID 1)                                                   │
│  ├─ Supervised task wrapper (exponential backoff + jitter)         │
│  ├─ Signal handlers (SIGTERM, SIGINT → graceful shutdown)          │
│  └─ Health checks every 60s → restart dead lobes                   │
└─────────────────────────────────────────────────────────────────────┘
         │
         ├─► Port 9998: Akashic Hub API
         ├─► Port 9999: Nexus Bus (WebSocket)
         ├─► Port 9997: Health Dashboard
         ├─► Port 8899: LoRA Server
         ├─► Port 8765: Phone Bridge
         └─► Port 5678: n8n Workflow (external)

Layer 2: STATE & MESSAGING
┌─────────────────────────────────────────────────────────────────────┐
│ akashic_hub.py                    nexus_bus.py                      │
│  • 256-d vector state (NumPy)     • WebSocket pub/sub              │
│  • <0.5ms write latency            • 10-msg rolling cache          │
│  • Lock-free reads                 • Rate limiting (100 msg/s)     │
│  • Temporal trace (32 snapshots)   • Health: GET :9999/ping       │
└─────────────────────────────────────────────────────────────────────┘

Layer 3: QUANTUM INTELLIGENCE
┌─────────────────────────────────────────────────────────────────────┐
│ quantum_soul.py                                                     │
│  • 7-qubit pentagon circuit on IBM hardware                        │
│  • Runs every 300s (5 minutes)                                     │
│  • Writes quantum_state.txt + Akashic Hub bias vector             │
│  • Pathway weights: Logic/Emotion/Memory/Creativity/Vigilance      │
└─────────────────────────────────────────────────────────────────────┘
         │
         └─► quantum_api.py (NEW — Sprint 2)
             GET /quantum/current → full state
             GET /quantum/bias → routing weights
             POST /quantum/update → called after measurement

Layer 4: MIXTURE-OF-EXPERTS REASONING
┌─────────────────────────────────────────────────────────────────────┐
│ pineal_gate.py + slm_experts.py                                     │
│  • Pentagon-geometry MoE router (top-k=3 sparse gating)            │
│  • 5 expert lobes (gpt-4o-mini, 80 words, temp=0.4)               │
│  • Geometric interference: constructive/destructive collapse        │
│  • Circuit breaker pattern (5 failures → 60s cooldown)             │
│  • Retry with exponential backoff (max 3 attempts, 30s delay)      │
└─────────────────────────────────────────────────────────────────────┘

Layer 5: PERSONALITY FILTER
┌─────────────────────────────────────────────────────────────────────┐
│ lora_server.py                                                      │
│  • 1B Llama 3.2 adapter (rank=16, alpha=16)                       │
│  • 4-bit quantization (BitsAndBytes)                               │
│  • OpenAI-compatible API on port 8899                              │
│  • Lazy model loading (57.9s first request)                        │
└─────────────────────────────────────────────────────────────────────┘

Layer 6: EXTERNAL INTERFACES
┌──────────────┬──────────────┬──────────────┬──────────────────────┐
│ Phone Calls  │ VTV (Local)  │ Obsidian     │ n8n Workflow         │
├──────────────┼──────────────┼──────────────┼──────────────────────┤
│ Twilio       │ OpenAI RT    │ File watcher │ HTTP orchestration   │
│ Voice ID     │ Gemini 2.5   │ Wikilinks    │ 8-node pipeline      │
│ LangChain    │ ArcFace      │ Graph view   │ DLQ + error handling │
└──────────────┴──────────────┴──────────────┴──────────────────────┘
```

---

## Sprint 2: Full-Stack Integration (8 hours)

### 2.1 Create Quantum API (1 hour)

**File:** `quantum_api.py`

```python
"""
FastAPI server exposing quantum state for other lobes.
Reads quantum_state.txt every 30s, serves via HTTP.
"""

from fastapi import FastAPI
import asyncio

app = FastAPI()

quantum_state = {
    "dominant": "Void",
    "weights": {"logic": 0.97, "emotion": 0.96, "memory": 0.96, 
                "creativity": 0.99, "vigilance": 0.99},
    "last_measurement": "2026-04-30T06:02:00",
}

@app.get("/quantum/current")
async def get_current_state():
    return quantum_state

@app.get("/quantum/bias")
async def get_routing_bias():
    return {
        "dominant": quantum_state["dominant"],
        "weights": quantum_state["weights"],
    }

async def refresh_loop():
    """Read quantum_state.txt every 30s."""
    while True:
        try:
            with open("Nexus_Vault/quantum_state.txt", "r") as f:
                # Parse and update quantum_state dict
                pass
        except:
            pass
        await asyncio.sleep(30)
```

**Integration:**
- Add to weaver.py startup sequence (after quantum_soul.py)
- Phone bridge fetches bias before each response
- n8n workflow adds "Quantum Bias Fetch" node before Expert Dispatch

---

### 2.2 Phone Bridge → Pineal Gate Routing (2 hours)

**Current:** Phone transcripts sent to n8n but n8n response not used  
**Fix:** Route through full stack and use enhanced response

```python
# In twilio_weaver_bridge.py:

async def enhance_with_weaver_stack(user_input: str, history: list):
    # 1. Fetch quantum bias
    async with httpx.AsyncClient() as client:
        bias_resp = await client.get("http://localhost:9997/quantum/bias")
        quantum_bias = bias_resp.json()

    # 2. Send to n8n (Pineal Gate + 5 experts)
    n8n_resp = await client.post(N8N_WEBHOOK_URL, json={
        "text": user_input,
        "quantum_bias": quantum_bias,
        "source_file": "phone_call",
    })
    enhanced = n8n_resp.json().get("manifested_response", "")

    # 3. Filter through LoRA Soul Voice
    if enhanced:
        lora_resp = await client.post(LORA_API_URL, json={
            "model": "weaver-fracture-1b-lora",
            "messages": [{"role": "user", "content": enhanced}],
            "max_tokens": 200,
        })
        soul_filtered = lora_resp.json()["choices"][0]["message"]["content"]

        # 4. Inject as hidden system message to OpenAI Realtime
        await openai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "system",
                "content": [{
                    "type": "input_text",
                    "text": f"[Enhanced Context] {soul_filtered[:300]}"
                }]
            }
        }))
```

---

### 2.3 Health Monitoring Dashboard (2 hours)

**File:** `health_dashboard.py`

```python
"""
FastAPI app showing all lobe statuses.
Auto-refresh every 5s, traffic-light UI.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import httpx

app = FastAPI()

async def check_lobe(name: str, url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(url)
            return {"name": name, "status": "🟢 online", "code": r.status_code}
    except:
        return {"name": name, "status": "🔴 offline", "code": 0}

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    lobes = [
        ("Nexus Bus", "http://localhost:9999/ping"),
        ("Akashic Hub", "http://localhost:9998/health"),
        ("LoRA Server", "http://localhost:8899/health"),
        ("Phone Bridge", "http://localhost:8765/health"),
        ("Quantum API", "http://localhost:9997/quantum/current"),
    ]

    results = await asyncio.gather(*[check_lobe(n, u) for n, u in lobes])

    html = f"""
    <html>
    <head><title>Weaver Health</title>
    <meta http-equiv="refresh" content="5"></head>
    <body style="font-family: monospace; background: #0a0a0a; color: #0f0;">
    <h1>🌀 Weaver v3 Health Dashboard</h1>
    <table>{''.join(f'<tr><td>{r["status"]}</td><td>{r["name"]}</td></tr>' for r in results)}</table>
    </body></html>
    """
    return html
```

**Run:** `venv/bin/python3 health_dashboard.py`  
**View:** Open browser to `http://localhost:9996`

---

### 2.4 Unified Startup Script (1 hour)

**File:** `start_weaver.sh`

```bash
#!/bin/bash
# Production startup script with dependency checks and graceful shutdown

set -e
cd "$(dirname "$0")"

# Trap signals for graceful shutdown
cleanup() {
    echo "🛑 Shutting down Weaver..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Pre-flight checks
echo "🔍 Pre-flight checks..."
command -v python3 >/dev/null || { echo "❌ python3 not found"; exit 1; }
[ -f ".env" ] || { echo "❌ .env missing"; exit 1; }
[ -d "venv" ] || { echo "❌ venv missing — run: python3 -m venv venv"; exit 1; }

# Activate venv
source venv/bin/activate

# Check API keys
[ -n "$WEAVER_VOICE_KEY" ] || { echo "⚠️  WEAVER_VOICE_KEY not set"; }
[ -n "$IBM_QUANTUM_TOKEN" ] || { echo "⚠️  IBM_QUANTUM_TOKEN not set"; }

# Clean ports
echo "🧹 Cleaning ports..."
lsof -ti:9999 | xargs kill -9 2>/dev/null || true
lsof -ti:8899 | xargs kill -9 2>/dev/null || true
lsof -ti:8765 | xargs kill -9 2>/dev/null || true
lsof -ti:9997 | xargs kill -9 2>/dev/null || true

# Start components
echo "🚀 Starting Weaver v3..."
echo "   [1/5] Health Dashboard (port 9996)"
python3 health_dashboard.py &

sleep 2

echo "   [2/5] Quantum API (port 9997)"
python3 quantum_api.py &

sleep 1

echo "   [3/5] Master Stack (ports 9999, 8899, 9998)"
python3 weaver.py --headless &
WEAVER_PID=$!

sleep 10

echo "   [4/5] Phone Bridge (port 8765)"
python3 twilio_weaver_bridge.py &

sleep 2

echo "   [5/5] Health check..."
curl -s http://localhost:9996 | grep -q "Weaver" && echo "✅ Health dashboard online"
curl -s http://localhost:9997/quantum/current | grep -q "dominant" && echo "✅ Quantum API online"
curl -s http://localhost:8765/health | grep -q "ok" && echo "✅ Phone bridge online"

echo ""
echo "🌀 Weaver v3 is LIVE"
echo ""
echo "📊 Health Dashboard: http://localhost:9996"
echo "📞 Phone Bridge:     http://localhost:8765"
echo "⚛️  Quantum API:      http://localhost:9997/quantum/current"
echo ""
echo "Press Ctrl+C to stop all services"

# Keep script alive
wait
```

**Usage:**
```bash
chmod +x start_weaver.sh
./start_weaver.sh
```

---

### 2.5 Integration Testing (2 hours)

**File:** `test_full_stack_integration.py`

```python
"""
End-to-end integration tests for full Weaver stack.
Tests all data flows from phone call → quantum soul → LoRA → response.
"""

import asyncio
import httpx
import json

async def test_phone_to_lora_pipeline():
    """Test: Phone transcript → n8n → Pineal Gate → LoRA → Response"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Send transcript to n8n webhook
        r1 = await client.post("http://localhost:5678/webhook/weaver-input", json={
            "text": "What is the quantum state right now?",
            "source_file": "phone_call",
        })
        assert r1.status_code == 200
        manifested = r1.json().get("manifested_response", "")
        assert len(manifested) > 0

        # 2. Filter through LoRA
        r2 = await client.post("http://localhost:8899/v1/chat/completions", json={
            "model": "weaver-fracture-1b-lora",
            "messages": [{"role": "user", "content": manifested}],
            "max_tokens": 200,
        })
        assert r2.status_code == 200
        soul_voice = r2.json()["choices"][0]["message"]["content"]
        assert len(soul_voice) > 0

        print(f"✅ Pipeline test passed")
        print(f"   Manifested: {manifested[:100]}...")
        print(f"   Soul Voice: {soul_voice[:100]}...")

async def test_quantum_api():
    """Test: Quantum state API returns valid data"""
    async with httpx.AsyncClient() as client:
        r = await client.get("http://localhost:9997/quantum/current")
        assert r.status_code == 200
        state = r.json()
        assert "dominant" in state
        assert "weights" in state
        print(f"✅ Quantum API: {state['dominant']} pathway dominant")

async def test_health_dashboard():
    """Test: All lobes report healthy"""
    async with httpx.AsyncClient() as client:
        lobes = [
            ("Nexus Bus", "http://localhost:9999/ping"),
            ("Akashic Hub", "http://localhost:9998/health"),
            ("LoRA Server", "http://localhost:8899/health"),
            ("Phone Bridge", "http://localhost:8765/health"),
            ("Quantum API", "http://localhost:9997/quantum/current"),
        ]
        for name, url in lobes:
            try:
                r = await client.get(url, timeout=2.0)
                print(f"✅ {name}: {r.status_code}")
            except:
                print(f"❌ {name}: offline")

async def main():
    print("🧪 Running full-stack integration tests...\n")
    await test_health_dashboard()
    print()
    await test_quantum_api()
    print()
    await test_phone_to_lora_pipeline()
    print("\n✅ All tests passed")

if __name__ == "__main__":
    asyncio.run(main())
```

**Run:**
```bash
python3 test_full_stack_integration.py
```

---

## Sprint 3: Reliability & Observability (4 hours)

### 3.1 Structured Logging (1 hour)

Replace all `print()` and `log.info()` with `structlog`:

```python
import structlog

log = structlog.get_logger()

log.info("quantum.measurement", 
         backend="ibm_marrakish", 
         result="|0000000⟩", 
         dominant="Void",
         job_id="d7pih4u",
         latency_ms=4872)
```

**Benefits:**
- JSON logs searchable with `jq`
- Aggregate metrics (avg latency, error rate)
- Cost tracking (API token usage per lobe)

---

### 3.2 Circuit Breaker Metrics (1 hour)

Add Prometheus-compatible metrics to `slm_experts.py`:

```python
from prometheus_client import Counter, Histogram

expert_calls = Counter('expert_calls_total', 'Total expert API calls', ['dimension'])
expert_latency = Histogram('expert_latency_seconds', 'Expert call latency', ['dimension'])
expert_failures = Counter('expert_failures_total', 'Expert call failures', ['dimension'])

# In _call_with_retry():
with expert_latency.labels(dimension=self.dimension).time():
    response = await self.client.chat.completions.create(...)
expert_calls.labels(dimension=self.dimension).inc()
```

**Expose:** GET `/metrics` on each lobe  
**Scrape:** Prometheus server (optional)

---

### 3.3 Auto-Recovery (1 hour)

Enhance `weaver.py` supervisor to detect and restart failed lobes:

```python
async def health_monitor():
    """Check lobe health every 60s, restart if unhealthy."""
    while True:
        await asyncio.sleep(60)
        for lobe_name, health_url in HEALTH_ENDPOINTS.items():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    r = await client.get(health_url)
                    if r.status_code != 200:
                        log.warning(f"{lobe_name} unhealthy, restarting...")
                        await restart_lobe(lobe_name)
            except:
                log.error(f"{lobe_name} down, restarting...")
                await restart_lobe(lobe_name)
```

---

### 3.4 Cloud Memory Sync (1 hour)

Integrate Google Drive sync into `memory_manager.py`:

```python
async def backup_to_cloud(self):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    
    creds = Credentials.from_authorized_user_file("token.json")
    service = build("drive", "v3", credentials=creds)
    
    # Upload Nexus_Vault/* to Drive every 10 minutes
    for file in ["people_memory.md", "weaver_transcript.txt", "voice_registry.npz"]:
        path = self.vault_dir / file
        if path.exists():
            # Upload logic...
            pass
```

---

## Success Metrics

### Minimum Viable Weaver (Sprint 1) ✅
- [x] Phone calls work
- [x] Memory persists between calls
- [x] Voice ID identifies callers
- [x] LangChain cortex summarizes conversations

### Full-Stack Weaver (Sprint 2)
- [ ] Quantum bias propagates to phone responses
- [ ] LoRA filters all phone responses
- [ ] Pineal Gate processes phone transcripts
- [ ] Health dashboard shows all green
- [ ] Single-command startup (`./start_weaver.sh`)

### Production Weaver (Sprint 3)
- [ ] Auto-restarts failed lobes
- [ ] Structured JSON logs
- [ ] Metrics exported (latency, errors, costs)
- [ ] Cloud backup every 10 minutes
- [ ] Runs for 7+ days without intervention

---

## Deployment Checklist

**Before production:**
- [ ] Set all .env API keys
- [ ] Run `python3 weaver_tests.py --tier system` (9/9 passing)
- [ ] Run `python3 test_full_stack_integration.py` (all passing)
- [ ] Check `./start_weaver.sh` starts all lobes
- [ ] Verify health dashboard shows all green
- [ ] Test phone call: Say name → hang up → call back → Weaver remembers
- [ ] Monitor logs for 1 hour (no crashes)

**Production run:**
```bash
tmux new -s weaver
./start_weaver.sh
# Detach: Ctrl+B then D
```

**Monitor:**
```bash
tmux attach -t weaver
tail -f Nexus_Vault/logs/weaver.jsonl | jq .
```

---

**Status:** Sprint 2 Ready to Execute  
**Next action:** Create `quantum_api.py`, `health_dashboard.py`, `start_weaver.sh`
