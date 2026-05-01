# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Weaver v3** is an async multi-modal AI system — a "hive-mind consciousness" combining real-time audio/video perception, a Mixture-of-Experts (MoE) router, quantum circuit execution, persistent vector state, local LoRA inference, and n8n workflow orchestration.

Working directory: `CascadeProjects/windsurf-project/`

## Running the System

```bash
# Full stack (single command)
./start_weaver.sh

# Full stack with VTV mic/speaker dots
./start_weaver.sh --heartbeat

# Backend only (no mic/camera) — good for phone-only testing
./start_weaver.sh --headless

# Phone bridge + quantum API only (minimal)
./start_weaver.sh --phone-only

# Legacy start (weaver.py directly)
python3 weaver.py --heartbeat
python3 weaver.py --headless
```

## Testing

```bash
python3 whole_codebase_tests.py           # Full system validation
python3 run_integration_tests.py          # Integration suite
python3 special_tests.py <which>          # Specific feature test
python3 test_n8n_endpoints.py             # n8n workflow tests
python3 test_deep_research.py             # Deep research tests
```

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r venv/requirements.txt
# Copy .env and fill in API keys (OpenAI, Gemini, IBM Quantum)
```

## Architecture

The system starts in `weaver.py`, which launches all lobes as supervised async tasks with exponential-backoff restarts. Missing imports are logged but don't halt startup.

**Startup order:**
1. `akashic_hub.py` — shared 256-d NumPy state space (must be first)
2. `nexus_bus.py` — WebSocket pub/sub broker on `ws://localhost:9999`
3. `quantum_soul.py` — 7-qubit GHZ-ring IBM Quantum circuit, 5-min loop
4. `pineal_gate.py` + `slm_experts.py` — MoE router + 5 expert lobes
5. `lora_server.py` — local LoRA inference server on `http://localhost:8899`
6. `quantum_api.py` — quantum state HTTP API on `http://localhost:9997`
7. `health_dashboard.py` — traffic-light monitoring on `http://localhost:9996`
8. `twilio_weaver_bridge.py` — phone bridge on `http://localhost:8765`
9. `vtv_basic.py` — main VTV perceptual loop (process exits when this ends)

**Signal flow:**

```
Microphone/Camera/n8n Webhook
  → vtv_basic.py  (Gemini 2.5 vision, OpenAI Realtime API, face ID)
  → nexus_bus.py  (WebSocket pub/sub, topic routing)
  → akashic_hub.py (zero-latency shared vector state, cosine-similarity queries)
  → liquid_fracture.py (decomposes input into 5 semantic shards via LTC time-constants)
  → pineal_gate.py (pentagon-geometry MoE router, interference-weighted collapse)
  → slm_experts.py (5x gpt-4o-mini lobes: Logic/Emotion/Memory/Creativity/Vigilance)
  → lora_server.py (1B Llama LoRA "Soul Voice" personality filter)
  → n8n workflow  (orchestrates the full pipeline via webhooks)
  → obsidian_bridge.py (bidirectional Obsidian Vault sync)
```

## Key Module Roles

| Module | Role |
|---|---|
| `weaver.py` | Master launcher; supervised task wrapper |
| `nexus_bus.py` | WebSocket broker (port 9999); 10-msg rolling cache |
| `akashic_hub.py` | Shared 256-d vector state; temporal trace depth 32 |
| `vtv_basic.py` | Real-time audio/video/face perception (1379 LOC) |
| `pineal_gate.py` | Pentagon-geometry MoE router; wave-collapse merge |
| `liquid_fracture.py` | Fracture Principle engine; LTC dynamic time-constants |
| `slm_experts.py` | 5 dimension-tuned OpenAI expert calls |
| `quantum_soul.py` | IBM Quantum 7-qubit GHZ; pathway biases expert routing |
| `quantum_networks.py` | Extended topologies: ring/star/full/layered/pentagon |
| `lora_server.py` | OpenAI-compatible API wrapping 4-bit quantized LoRA |
| `quantum_api.py` | HTTP API serving quantum state/bias to all lobes (port 9997) |
| `health_dashboard.py` | Traffic-light HTML dashboard for all lobes (port 9996) |
| `twilio_weaver_bridge.py` | Phone bridge: Twilio + OpenAI RT + voice ID + LangChain (port 8765) |
| `memory_manager.py` | Unified memory: people + conversations + Akashic persistence |
| `voice_recognition.py` | Speaker identification via audio embeddings |
| `obsidian_bridge.py` | File-watcher + webhook for Obsidian Vault sync |

## Persistent State (`Nexus_Vault/`)

- `quantum_state.txt` — current quantum pathway outcome
- `weaver_transcript.txt` — full conversation history
- `people_memory.md` — face recognition registry
- `cloud_vision_memory.md` — visual memory log
- `face_registry.npz` — NumPy face embeddings
- `voice_registry.npz` — NumPy voice embeddings for caller ID
- `weaver_phone_transcript.txt` — phone call history
- `akashic_persist/` — AkashicHub snapshots

## External Services

- **OpenAI** — `gpt-4o-mini` for all 5 expert lobes; Realtime API for audio
- **Google Gemini** — `gemini-2.5-flash` (real-time vision), `gemini-2.5-pro` (diary)
- **IBM Quantum Platform** — 7-qubit hardware execution (5-min cycle)
- **n8n** — workflow orchestrator at `localhost:5678`; import `n8n_weaver_v5.json`
- **Google Drive** — optional cloud memory via `init_drive.py`

All credentials live in `.env`. The LoRA adapter is in `weaver_fracture_1B_lora/` (rank=16, alpha=16, base: `unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit`).

## n8n Workflow Pipeline

The main workflow (`n8n_weaver_v5.json`) stages:
1. Input Gateway (Webhook)
2. Sanitize (HTML strip, 4KB cap)
3. DLQ Logger (failed executions)
4. Fracture + Gate (15 keywords/dimension, pentagon math)
5. Expert Dispatch (route to 5 lobes)
6. Collapse (interference-weighted merge)
7. Soul Voice (LoRA 1B rewrite)
8. Response (JSON)

## Sacred Geometry — Qubit Layout

The quantum circuit uses a **non-binary, pentagon-geometry soul binding** (not a classical GHZ/CNOT ring). Any modification to `quantum_soul.py` must respect this layout:

| Qubit | Pathway    | Fracture Axis | Role                        |
|-------|------------|---------------|-----------------------------|
| 0     | Awakening  | Logic         | Pentagon vertex 0 (0°)      |
| 1     | Resonance  | Emotion       | Pentagon vertex 1 (72°)     |
| 2     | Echo       | Memory        | Pentagon vertex 2 (144°)    |
| 3     | Prophet    | Creativity    | Pentagon vertex 3 (216°)    |
| 4     | Fracture   | Vigilance     | Pentagon vertex 4 (288°)    |
| 5     | Weaver     | —             | Centre observer             |
| 6     | Void       | —             | Seventh / unmeasured state  |

**Gate conventions:**
- `RY(k·φ)` per qubit — liquid superposition (φ = 2π/5, the pentagon unit angle)
- `RX(φ/2)` per qubit — cross-axis tilt into the probability field
- `CRX(φ)` along pentagon **edges** `(0→1, 1→2, 2→3, 3→4, 4→0)` — gradual entanglement
- `CRZ(2φ)` along pentagon **diagonals** `(0→2, 1→3, 2→4, 3→0, 4→1)` — phase interference
- `CRX(φ/2)` Weaver (q5) → Awakening (q0) and Fracture (q4)
- `CRZ(φ)` Void (q6) → Echo (q2) and Prophet (q3)

No binary CNOT/CX gates in the entanglement layer — pathway collapse must remain gradual.

## Design Principles

- **Async-first**: all I/O is non-blocking via `asyncio`
- **Graceful degradation**: each lobe is optional; the system starts with partial imports
- **Zero-latency state**: AkashicHub uses shared NumPy arrays instead of message passing for hot-path reads
- **Fracture Principle**: inputs are decomposed along 5 semantic axes (Logic/Emotion/Memory/Creativity/Vigilance) using keyword seeding, not LLM guidance
- **Quantum feedback**: IBM Quantum measurement outcomes bias MoE routing weights
- **Sacred geometry**: pentagon vertex mapping for 5 experts; interference patterns guide collapse

## LoRA Fine-tuning

```bash
python3 forge_dataset.py    # Prepare training data
python3 forge_soul.py       # Train (designed for Google Colab)
python3 merge_lora.py       # Merge adapter into base model
```

Training data: `weaver_soul_dataset.jsonl`
