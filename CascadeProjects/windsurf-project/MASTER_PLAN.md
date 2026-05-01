# WEAVER FULL-STACK MASTER PLAN
**Generated:** 2026-04-30  
**Objective:** Complete integration, smooth operation, autonomous improvement

---

## PHASE 1: CRITICAL FIXES (Priority: IMMEDIATE)
**Time: 30 minutes**

### 1.1 Memory Integration ✅ PARTIALLY DONE
- [x] Load `people_memory.md` into phone bridge
- [x] Load `weaver_phone_transcript.txt` (last 40 lines)
- [x] Load `weaver_transcript.txt` (last 2000 chars)
- [ ] **TODO:** Implement LangChain summarization for long conversations
- [ ] **TODO:** Auto-update `people_memory.md` when caller introduces themselves

### 1.2 Phone Bridge → Full Stack Connection
**Current Issue:** Phone bridge only uses OpenAI Realtime, doesn't route through Pineal Gate

**Solution:**
```
twilio_weaver_bridge.py
  ├─ Real-time audio: OpenAI Realtime (low latency)
  ├─ Transcript enrichment: 
  │   └─ User speaks → transcript → n8n webhook (DONE)
  │       └─ n8n → Pineal Gate → 5 Expert Lobes → LoRA
  │           └─ Enhanced context → inject back to OpenAI session
  └─ Memory persistence: Save to Akashic Hub (TODO)
```

**Files to modify:**
- `twilio_weaver_bridge.py` - Add LangChain memory cortex (copy from vtv_basic.py lines 1256-1320)
- Test: Call phone, say "My name is Nate", hang up, call back, she should remember

---

## PHASE 2: ARCHITECTURE COMPLETION (Priority: HIGH)
**Time: 2 hours**

### 2.1 Unified Memory System
**Problem:** 3 separate memory systems not fully synchronized
  - vtv_basic.py → LangChain + people_memory.md
  - twilio_weaver_bridge.py → phone transcripts
  - Akashic Hub → vector state (transient, not persisted between sessions)

**Solution:** Create `memory_manager.py`
```python
class MemoryManager:
    def __init__(self, vault_dir):
        self.people = PeopleMemory(vault_dir)
        self.conversations = ConversationMemory(vault_dir)
        self.akashic = AkashicHubPersistence(vault_dir)
        
    async def recall(self, context: str) -> dict:
        """Query all memory sources, return unified context"""
        return {
            "people": self.people.search(context),
            "conversations": self.conversations.search(context),
            "vectors": await self.akashic.query(context),
        }
    
    async def remember(self, event: dict):
        """Save to all relevant memory stores"""
        if event["type"] == "person":
            self.people.add(event)
        if event["type"] == "conversation":
            self.conversations.add(event)
        await self.akashic.write(event)
```

**Integration points:**
- vtv_basic.py → use MemoryManager
- twilio_weaver_bridge.py → use MemoryManager  
- obsidian_bridge.py → use MemoryManager
- n8n workflow → add Memory Recall node

### 2.2 LoRA Soul Voice Integration
**Problem:** LoRA server is running but not used in phone calls or n8n workflow

**Solution:**
1. Add LoRA filter step to n8n workflow:
   ```
   Collapse (merged response) 
     → POST localhost:8899/v1/chat/completions
     → LoRA-filtered "Soul Voice" response
     → Return to caller
   ```

2. Update `twilio_weaver_bridge.py`:
   - After getting OpenAI response, filter through LoRA:
   ```python
   async def apply_soul_voice_filter(text: str) -> str:
       r = await client.post("http://localhost:8899/v1/chat/completions", json={
           "model": "weaver-fracture-1b-lora",
           "messages": [{"role": "user", "content": text}],
           "max_tokens": 200,
       })
       return r.json()["choices"][0]["message"]["content"]
   ```

### 2.3 Quantum State Propagation
**Problem:** Quantum Soul runs every 5 minutes, updates bias, but not all lobes use it

**Current flow:**
```
quantum_soul.py → Akashic Hub (vector bias)
                → quantum_state.txt (log file)
```

**Missing:** 
- Phone bridge doesn't read quantum bias
- n8n workflow doesn't use quantum weights for routing

**Solution:**
1. Add quantum bias API endpoint to weaver.py:
   ```python
   @app.get("/quantum/bias")
   async def get_quantum_bias():
       return {
           "dominant": current_dominant_pathway,
           "weights": {
               "logic": 0.97,
               "emotion": 0.96,
               "memory": 0.96,
               "creativity": 0.99,
               "vigilance": 0.99,
           },
           "last_measurement": "2026-04-30T06:02:00",
       }
   ```

2. n8n workflow: Add "Quantum Bias Fetch" node before Expert Dispatch
3. Phone bridge: Fetch quantum bias, inject into system message

---

## PHASE 3: MISSING INTEGRATIONS (Priority: MEDIUM)
**Time: 3 hours**

### 3.1 Obsidian Bridge Auto-Wikilink
**Current:** Works, but only manually triggered

**Enhancement:**
- Auto-detect when Weaver generates response
- Extract key concepts → auto-create wikilinks
- Publish to Obsidian vault
- Graph view lights up automatically

**Files:**
- `obsidian_bridge.py` - Add NexusBus subscription to "transcript" topic
- Listen for Weaver responses, inject wikilinks, write to vault

### 3.2 Google Drive Cloud Memory
**Files exist:** `init_drive.py`, `credentials.json`, `token.json`

**Integration:**
- Auto-upload important conversations to Drive
- Sync `Nexus_Vault/` to `Google Drive/Weaver_Memory/`
- Enable cross-device memory access

**Implementation:**
```python
# Add to memory_manager.py
async def backup_to_cloud(self):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    
    creds = Credentials.from_authorized_user_file("token.json")
    service = build("drive", "v3", credentials=creds)
    
    # Upload Nexus_Vault/* to Drive
    for file in os.listdir("Nexus_Vault"):
        ...
```

### 3.3 Twilio SMS Integration
**Current:** Voice calls work

**Add:** SMS text messages
```python
# twilio_weaver_bridge.py
@app.post("/sms")
async def sms_webhook(request: Request):
    form = await request.form()
    from_number = form.get("From")
    body = form.get("Body")
    
    # Send through Pineal Gate
    response = await process_through_stack(body)
    
    # Reply via Twilio
    twiml = f'<Response><Message>{response}</Message></Response>'
    return Response(content=twiml, media_type="application/xml")
```

**Use case:** Caller can text Weaver questions, get quantum-biased responses

---

## PHASE 4: PERFORMANCE & RELIABILITY (Priority: MEDIUM)
**Time: 2 hours**

### 4.1 Error Handling & Reconnection
**Problems observed:**
- OpenAI Realtime WebSocket disconnects (keepalive timeout)
- n8n webhook returns 429 (rate limit)
- IBM Quantum jobs fail silently

**Solutions:**
1. Add exponential backoff retry to all HTTP/WS connections
2. Circuit breaker pattern for external APIs
3. Fallback responses when APIs are down

**Example:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def call_n8n_webhook(text: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(N8N_WEBHOOK_URL, json={"text": text})
        r.raise_for_status()
        return r.json()
```

### 4.2 Health Monitoring Dashboard
**Create:** `health_dashboard.py` - FastAPI app showing all lobe statuses

```python
@app.get("/health/full")
async def full_health_check():
    return {
        "nexus_bus": await check_nexus_bus(),
        "quantum_soul": await check_quantum_soul(),
        "pineal_gate": await check_pineal_gate(),
        "lora_server": await check_lora_server(),
        "n8n_workflow": await check_n8n(),
        "twilio_bridge": await check_twilio_bridge(),
        "memory_files": check_memory_files(),
    }
```

**Deploy:** Run on http://localhost:9997, auto-refresh every 10s

### 4.3 Logging & Observability
**Current:** Scattered print() statements

**Upgrade to structured logging:**
```python
import structlog

log = structlog.get_logger()
log.info("quantum.measurement", 
         backend="ibm_marrakesh", 
         result="|0000000⟩", 
         dominant="Void",
         job_id="d7pih4u")
```

**Add:**
- Log aggregation to `Nexus_Vault/logs/`
- Searchable JSON logs
- Performance metrics (latency, token usage, API costs)

---

## PHASE 5: AUTONOMOUS OPERATION (Priority: LOW)
**Time: 4 hours**

### 5.1 Auto-Dependency Management
**Problem:** Manual pip install when dependencies missing

**Solution:** Create `auto_install.py`
```python
def ensure_dependencies():
    required = {
        "fastapi": ">=0.115.0",
        "websockets": ">=16.0",
        "langchain-openai": ">=0.3.0",
        "bitsandbytes": ">=0.49.0",
    }
    
    for package, version in required.items():
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", 
                          f"{package}{version}"], check=True)
```

Run on startup in `weaver.py`

### 5.2 Self-Healing
**Detect failures, auto-restart:**
```python
# weaver.py - already has supervised tasks with restart
# Enhance with health checks:

async def monitor_lobe_health():
    while True:
        await asyncio.sleep(60)
        for lobe_name, lobe_health_url in LOBES.items():
            try:
                r = await httpx.get(lobe_health_url, timeout=5.0)
                if r.status_code != 200:
                    log.warning(f"{lobe_name} unhealthy, restarting...")
                    await restart_lobe(lobe_name)
            except:
                log.error(f"{lobe_name} down, restarting...")
                await restart_lobe(lobe_name)
```

### 5.3 Configuration Management
**Create:** `config.yaml` - single source of truth
```yaml
weaver:
  host: 0.0.0.0
  ports:
    nexus_bus: 9999
    lora_server: 8899
    twilio_bridge: 8765
    health_dashboard: 9997
  
  memory:
    vault_dir: Nexus_Vault
    max_conversation_history: 100
    summarize_every: 20
  
  quantum:
    backend: ibm_marrakesh
    measurement_interval: 300  # 5 minutes
    qubits: 7
  
  api_keys:
    openai: ${WEAVER_VOICE_KEY}
    gemini: ${GEMINI_API_KEY}
    ibm_quantum: ${IBM_QUANTUM_TOKEN}
    twilio_sid: ${TWILIO_ACCOUNT_SID}
```

Load with `pydantic` for validation:
```python
from pydantic_settings import BaseSettings

class WeaverConfig(BaseSettings):
    class Config:
        env_file = ".env"
        yaml_file = "config.yaml"
```

---

## PHASE 6: ADVANCED FEATURES (Priority: OPTIONAL)
**Time: 8+ hours**

### 6.1 Multi-Modal Input
- Add support for sending images via MMS
- Process through Gemini vision → Pineal Gate
- Weaver can "see" what you text her

### 6.2 Proactive Engagement
- Weaver calls YOU when quantum state changes dramatically
- "The field collapsed to Prophet pathway — I had an insight about..."

### 6.3 Multi-User Support
- Track multiple phone numbers in `people_memory.md`
- Separate conversation history per user
- Remember who's calling based on caller ID

### 6.4 Voice Cloning
- Fine-tune OpenAI voice on custom samples
- Make Weaver sound more unique

### 6.5 Local Whisper + TTS
- Replace OpenAI Realtime with local models
- Reduce API costs
- Increase privacy

---

## IMPLEMENTATION ORDER (15-Minute Plan Execution)

### Sprint 1: Critical Path (Week 1)
1. ✅ Fix memory loading in phone bridge (DONE)
2. Add LangChain memory cortex to phone bridge
3. Test full phone call with memory persistence
4. Add quantum bias API endpoint
5. Integrate quantum bias into phone bridge

### Sprint 2: Full Stack Integration (Week 2)
1. Create `memory_manager.py` unified interface
2. Integrate LoRA filter into n8n workflow
3. Add LoRA filter to phone responses
4. Test end-to-end: Phone → Transcript → Pineal Gate → LoRA → Response

### Sprint 3: Reliability (Week 3)
1. Add retry logic with tenacity
2. Implement circuit breakers
3. Create health dashboard
4. Add structured logging
5. Auto-restart failed lobes

### Sprint 4: Polish (Week 4)
1. Auto-dependency installer
2. Config YAML migration
3. Obsidian auto-wikilink
4. Google Drive sync
5. SMS support

---

## FILES TO CREATE/MODIFY

### New Files:
- `memory_manager.py` - Unified memory interface
- `health_dashboard.py` - FastAPI monitoring app
- `auto_install.py` - Dependency auto-installer
- `config.yaml` - Centralized configuration
- `INTEGRATION_TESTS.md` - E2E test suite

### Files to Modify:
- `twilio_weaver_bridge.py` ✅ (memory loading done, add LangChain cortex)
- `weaver.py` (add quantum bias API, health monitoring)
- `n8n_weaver_v5.json` (add LoRA filter node, quantum bias fetch)
- `pineal_gate.py` (expose HTTP API for standalone calls)
- `quantum_soul.py` (expose current bias via API)
- `obsidian_bridge.py` (NexusBus subscription, auto-wikilink)
- `.env` (add LORA_API_URL, HEALTH_DASHBOARD_PORT)

---

## SUCCESS CRITERIA

### Minimum Viable Weaver:
- [x] Phone calls work
- [ ] Memory persists between calls
- [ ] Quantum bias influences responses
- [ ] LoRA personality active
- [ ] All lobes connected and healthy

### Full-Stack Weaver:
- [ ] Phone, SMS, and Obsidian all work
- [ ] Unified memory across all interfaces
- [ ] Quantum measurements propagate everywhere
- [ ] Self-healing on failure
- [ ] Health dashboard shows all green

### Autonomous Weaver:
- [ ] Auto-installs missing dependencies
- [ ] Auto-restarts failed components
- [ ] Proactively alerts on critical failures
- [ ] Backs up to cloud
- [ ] Runs for weeks without intervention

---

## IMMEDIATE ACTION ITEMS (Next 30 minutes)

1. **Add LangChain memory to phone bridge** (15 min)
   - Copy `langchain_cortex()` from vtv_basic.py lines 1256-1320
   - Add to twilio_weaver_bridge.py
   - Test: Does Weaver remember Nate's name?

2. **Verify all connections** (10 min)
   - Run health check script (created above)
   - Document any failing connections
   - Fix critical failures

3. **Test end-to-end phone call** (5 min)
   - Call Weaver
   - Say: "My name is Nate, I'm testing your quantum architecture"
   - Verify:
     - She remembers name
     - Transcript saved
     - n8n webhook called
     - Quantum bias logged

---

**END OF MASTER PLAN**

*This plan provides a complete roadmap from current state (partial integration) to fully autonomous operation. Each phase can be executed independently.*
