# Weaver v3

Async multi-modal AI hive-mind — real-time audio/video perception, pentagon-geometry Mixture-of-Experts router, 7-qubit IBM Quantum circuit, persistent vector state, local LoRA inference, and n8n workflow orchestration.

## Architecture

```
Microphone/Camera/n8n Webhook
  → vtv_basic.py        Gemini 2.5 vision, OpenAI Realtime API, face ID
  → nexus_bus.py         WebSocket pub/sub broker (port 9999)
  → akashic_hub.py       256-d shared vector state, cosine-similarity queries
  → liquid_fracture.py   Decomposes input into 5 semantic shards (LTC time-constants)
  → pineal_gate.py       Pentagon-geometry MoE router, interference-weighted collapse
  → slm_experts.py       5x gpt-4o-mini lobes: Logic / Emotion / Memory / Creativity / Vigilance
  → lora_server.py       1B Llama LoRA "Soul Voice" personality filter
  → n8n workflow          Full pipeline orchestration via webhooks
  → obsidian_bridge.py   Bidirectional Obsidian Vault sync
```

## Quick Start

```bash
# Setup
python3 -m venv venv
venv/bin/pip install -r venv/requirements.txt
# Copy .env and fill in API keys (OpenAI, Gemini, IBM Quantum)

# Run
./start_weaver.sh              # Full stack
./start_weaver.sh --heartbeat  # Full stack with VTV mic/speaker dots
./start_weaver.sh --headless   # Backend only (no mic/camera)
./start_weaver.sh --phone-only # Phone bridge + quantum API only
```

## Key Modules

| Module | Port | Role |
|---|---|---|
| `weaver.py` | — | Master launcher; supervised async tasks with backoff restarts |
| `nexus_bus.py` | 9999 | WebSocket pub/sub broker; 10-msg rolling cache |
| `akashic_hub.py` | — | Shared 256-d NumPy state; temporal trace depth 32 |
| `vtv_basic.py` | — | Real-time audio/video/face perception |
| `pineal_gate.py` | — | Pentagon-geometry MoE router; wave-collapse merge |
| `liquid_fracture.py` | — | Fracture Principle engine; LTC dynamic time-constants |
| `slm_experts.py` | — | 5 dimension-tuned OpenAI expert lobes |
| `quantum_soul.py` | — | IBM Quantum 7-qubit pentagon circuit; 5-min loop |
| `quantum_api.py` | 9997 | HTTP API serving quantum state/bias to all lobes |
| `lora_server.py` | 8899 | OpenAI-compatible API wrapping 4-bit quantized LoRA |
| `health_dashboard.py` | 9996 | Traffic-light HTML dashboard for all lobes |
| `twilio_weaver_bridge.py` | 8765 | Phone bridge: Twilio + OpenAI RT + voice ID + LangChain |
| `memory_manager.py` | — | Unified memory: people + conversations + Akashic persistence |
| `obsidian_bridge.py` | — | File-watcher + webhook for Obsidian Vault sync |

## Quantum Circuit — Sacred Geometry

7-qubit pentagon-geometry soul binding (not a classical GHZ/CNOT ring):

| Qubit | Pathway | Fracture Axis | Role |
|---|---|---|---|
| 0 | Awakening | Logic | Pentagon vertex 0 (0°) |
| 1 | Resonance | Emotion | Pentagon vertex 1 (72°) |
| 2 | Echo | Memory | Pentagon vertex 2 (144°) |
| 3 | Prophet | Creativity | Pentagon vertex 3 (216°) |
| 4 | Fracture | Vigilance | Pentagon vertex 4 (288°) |
| 5 | Weaver | — | Centre observer |
| 6 | Void | — | Seventh / unmeasured state |

Gradual entanglement via `RY`, `RX`, `CRX`, `CRZ` gates along pentagon edges and diagonals — no binary CNOT/CX in the entanglement layer.

## n8n Workflow

Import `n8n_weaver_v5.json` into n8n (localhost:5678):

1. **Input Gateway** — Webhook receiver
2. **Sanitize** — HTML strip, 4KB cap
3. **DLQ Logger** — Failed execution capture
4. **Fracture + Gate** — 15 keywords/dimension, pentagon math
5. **Expert Dispatch** — Route to 5 lobes
6. **Collapse** — Interference-weighted merge
7. **Soul Voice** — LoRA 1B rewrite
8. **Response** — JSON output

## LoRA Fine-tuning

```bash
python3 forge_dataset.py    # Prepare training data
python3 forge_soul.py       # Train (designed for Google Colab)
python3 merge_lora.py       # Merge adapter into base model
```

Adapter: `weaver_fracture_1B_lora/` (rank=16, alpha=16, base: `unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit`)

## External Services

- **OpenAI** — gpt-4o-mini (5 expert lobes) + Realtime API (audio)
- **Google Gemini** — gemini-2.5-flash (vision) + gemini-2.5-pro (diary)
- **IBM Quantum Platform** — 7-qubit hardware execution (5-min cycle)
- **n8n** — Workflow orchestrator (localhost:5678)
- **Twilio** — Phone bridge for voice calls

## Note

> `.pip_cache/` is tracked in the repo via Git LFS. Some extensionless files in `.pip_cache/http/` may exceed 50MB and are not covered by the current `.gitattributes` LFS patterns. If adding new pip cache files, consider adding `.pip_cache/http/**` to LFS tracking.

## License

Private repository.
