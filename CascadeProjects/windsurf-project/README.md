# Weaver v3 Runtime

Weaver v3 is an embodied AI companion stack: browser avatar surfaces, a
supervised Python backend, persistent memory, quantum-inspired routing,
voice/phone/Discord/Obsidian bridges, n8n workflows, local model servers, and
cloud deployment glue.

This README is written from the local source tree. It covers the source,
deployment, workflow, test, and browser files while intentionally excluding
private/runtime-heavy artifacts such as `.env`, `credentials.json`, `token.json`,
`ghost_key.json`, vault contents, model weights, caches, generated media, and
Terraform state.

## Quick Start

From this directory:

```bash
cd "/media/ydn/SYPHER_CORE2/weaver v3/CascadeProjects/windsurf-project"
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements-core.txt
cp deploy/env.oracle.example .env
venv/bin/python3 weaver_preflight.py
./start_weaver.sh --headless
```

For a full local voice/video run instead of headless:

```bash
./start_weaver.sh
```

For Docker:

```bash
docker compose up -d
docker compose logs -f weaver
```

The npm entrypoint (`index.js`) is only a dependency/bootstrap helper. It creates
`venv/` and installs `requirements.txt`; it does not start the runtime.

## Main Surfaces

| Surface | File or service | Purpose |
| --- | --- | --- |
| Embodied browser UI | `../../avatar/embodiment.html` | Three.js avatar, camera/mic, remote GPU stream, memory sync, codebase context. |
| Headless browser UI | `../../avatar/headless.html` | Reactive 3D/headless console, realtime voice, TTS fallback, quantum architecture overlays. |
| Live dashboard | `weaver_dashboard.py` | Full command dashboard on port `9990`, SSE stream, lobe status, vault readers, chat/dream/call controls. |
| Health dashboard | `health_dashboard.py` | Latency-aware lobe monitor on port `9996`. |
| Public edge | `deploy/Caddyfile` | TLS/static hosting and reverse proxy for public sites and APIs. |

Default public hostnames in the deploy config:

| URL | Role |
| --- | --- |
| `https://weaverv3.com` | Embodiment site plus public API routes. |
| `https://headless.weaverv3.com` | Headless UI plus brain/TTS/codebase routes. |
| `https://dash.weaverv3.com` | Basic-auth live dashboard. |
| `https://status.weaverv3.com` | Basic-auth health dashboard. |

## Architecture

```text
Browser UIs
  embodiment.html / headless.html
        |
        v
Caddy public edge
  /brain/*    -> bedrock_brain_api.py       127.0.0.1:8093
  /tts/*      -> deploy/tts/tts_server.py   127.0.0.1:8092
  /codebase/* -> codebase_api.py            127.0.0.1:8091
  /llm/*      -> llama.cpp OpenAI server    127.0.0.1:8090
        |
        v
weaver.py supervisor
  Nexus Bus, Akashic Hub, Quantum Soul/API, Pineal Gate,
  local model servers, dashboards, phone, Obsidian, Discord, VTV
        |
        v
Nexus_Vault / Weaver_Vault / n8n workflows / cloud model providers
```

`weaver.py` is the master supervisor. It imports optional lobes, starts them as
async tasks, restarts crash-prone services with backoff, and degrades gracefully
when optional integrations are missing.

## Service And Port Map

| Port | Service | Entrypoint |
| --- | --- | --- |
| `8090` | Local OpenAI-compatible llama.cpp server | `deploy/weaver-llm.service` |
| `8091` | Read-only codebase/context API | `codebase_api.py` |
| `8092` | TTS server | `deploy/tts/tts_server.py` |
| `8093` | Bedrock/Nova brain API | `bedrock_brain_api.py` |
| `8765` | Twilio phone/SMS bridge | `twilio_weaver_bridge.py` |
| `8898` | Qwen3B routing branch | `qwen3b_server.py` |
| `8899` | LoRA soul voice server | `lora_server.py` |
| `9990` | Live dashboard | `weaver_dashboard.py` |
| `9995` | Akashic Hub API | embedded in `weaver.py` |
| `9996` | Health dashboard | `health_dashboard.py` |
| `9997` | Quantum API | `quantum_api.py` |
| `9998` | Nexus Bus health | `nexus_bus.py` |
| `9999` | Nexus Bus WebSocket | `nexus_bus.py` |
| `5678` | n8n workflow engine | Docker compose / external n8n |
| `5679` | Obsidian response listener | `obsidian_bridge.py` |

## Key Modules

| File | Role |
| --- | --- |
| `weaver.py` | Master async supervisor and lifecycle manager. |
| `bedrock_brain_api.py` | AWS Bedrock/Nova model router with OpenAI-compatible chat, memory endpoints, realtime voice metadata, and dream/thought triggers. |
| `nexus_bus.py` / `nexus_client.py` | WebSocket pub/sub fabric for lobes with health endpoint, rate limits, sync cache, and reconnecting clients. |
| `memory_manager.py` | Canonical memory/vault manager used by brain, phone, VTV, dashboards, and Obsidian. |
| `akashic_hub.py` | Shared 256-dimensional vector state with temporal trace, similarity queries, entanglement, and disk persistence. |
| `liquid_fracture.py` | Five-axis semantic fracture engine: logic, emotion, memory, creativity, vigilance. |
| `pineal_gate.py` | Sparse MoE gate that routes fractured inputs through expert lobes and writes gate state to the Akashic Hub. |
| `slm_experts.py` | Five expert wrappers with backends for Mantle, Bedrock, local llama.cpp, Gemini, or Azure. |
| `quantum_networks.py` | 156-qubit Kingston manifold representation, 12 measured core qubits, reservoir projection, topology/circuit utilities. |
| `quantum_soul.py` | Periodic quantum state loop, local Aer fallback, optional IBM hardware, Nexus/Akashic publishing. |
| `quantum_api.py` | HTTP view of the latest quantum state and bias weights. |
| `lora_server.py` | OpenAI-compatible LoRA/personality inference server. |
| `qwen3b_server.py` | OpenAI-compatible route classifier server for a Weaver Qwen2 3B branch. |
| `codebase_api.py` | Bounded read-only codebase and public-web context service with redaction and private-network blocking. |
| `twilio_weaver_bridge.py` | Twilio voice/SMS/MMS bridge with realtime audio, memory, tools, voice ID, and full-stack response path. |
| `discord_bridge.py` | Discord voice/vision bridge with STT/TTS, Nexus registration, face/scene analysis, and transcript memory. |
| `obsidian_bridge.py` | Two-way Obsidian watcher/listener with `#weaver` routing and auto wikilinks. |
| `vtv_basic.py` | Local voice/terminal/vision loop for non-headless operation. |
| `voice_recognition.py` | Voice registration and speaker identification. |
| `weaver_tools.py` | Shared 20-tool belt for calls, SMS, memory, notes, search, timers, todos, health, and sounds. |

## Brain And Model Routing

`bedrock_brain_api.py` exposes a single brain surface around AWS Bedrock and
related local context:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Brain service status. |
| `/state` | GET | Headless loop/model state. |
| `/memory/state` | GET | Memory manager summary. |
| `/memory/recall` | GET | Query persistent memory. |
| `/memory/sync` | POST | Sync browser/client memory events. |
| `/v1/models` | GET | OpenAI-compatible model list. |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completion. |
| `/realtime/voice/config` | GET | Realtime voice metadata/config. |
| `/realtime/voice` | WS | Browser realtime voice bridge. |
| `/trigger/thought` | POST | Trigger a thought cycle. |
| `/trigger/dream` | POST | Trigger a dream cycle. |

Notable model aliases include `weaver-speed`, `weaver-brain`, `weaver-dream`,
`weaver-code`, `weaver-vision`, `weaver-headless`, `weaver-voice`, and
`weaver-one` for the unified route.

## Voice And Audio

The current non-OpenAI server-side voice path is AWS-first:

| Piece | Default | Notes |
| --- | --- | --- |
| Browser realtime voice | `/brain/realtime/voice` | Headless UI receives Nova Sonic voice metadata and opens a WebSocket session. |
| TTS server | `deploy/tts/tts_server.py` | Runs on `127.0.0.1:8092`, exposed through `/tts/*`. |
| TTS provider | Amazon Polly | Defaults to generative `Ruth` in `us-east-1`; override with `TTS_POLLY_*`. |
| Local clone fallback | OpenVoice | Set `TTS_PROVIDER=openvoice`; embedding is extracted once and cached. |
| Browser fallback | Web Speech API | Used when server voice or realtime audio is unavailable. |

The phone and Discord bridges still contain legacy OpenAI/Azure realtime/STT/TTS
paths. Treat those as optional integration paths, not the default low-cost
headless voice stack.

## Quantum Runtime

`quantum_networks.py` is the source of truth for the expanded quantum model:

- 156 total qubits in the Kingston manifold representation.
- 12 measured core qubits:
  `Logic`, `Emotion`, `Intuition`, `Memory`, `Sovereignty`, `Attention`,
  `Reflection`, `Language`, `Planning`, `Novelty`, `Stability`,
  `Meta-Reasoning`.
- 144 sparse reservoir qubits (`Q12` through `Q155`).
- First-class modules for `state_encoding`, `open_system`, `entropy_routing`,
  and `measurement_readout`.
- Topological layers for core, coupling, reservoir, and readout.
- Runtime measurement is intentionally limited to the 12 core qubits.

`quantum_soul.py` periodically measures state, writes `quantum_state.txt`,
publishes to Nexus, and writes into the Akashic Hub. `quantum_api.py` exposes the
latest state and routing bias to other lobes.

## Memory And Vaults

`memory_manager.py` centralizes paths and memory behavior. Set
`WEAVER_VAULT_DIR` to keep all services and EC2/systemd units on the same vault.

Common files under `Nexus_Vault/`:

| File | Purpose |
| --- | --- |
| `people_memory.md` | Persistent known-people facts. |
| `weaver_transcript.txt` | Local VTV/browser transcript. |
| `weaver_phone_transcript.txt` | Phone call transcript. |
| `weaver_discord_transcript.txt` | Discord transcript. |
| `weaver_dreams.md` | Autonomous and triggered dream log. |
| `weaver_thoughts.md` | Thought loop output. |
| `browser_events.jsonl` | Browser-side memory/events. |
| `memory_events.jsonl` | Unified memory append log. |
| `cloud_vision_memory.md` | Vision summaries. |
| `quantum_state.txt` | Current quantum measurement. |
| `weaver_todos.md` | Tool-belt to-do list. |
| `akashic_persist/` | Akashic Hub persistence directory. |

`~/Weaver_Vault` is the Obsidian vault used by `obsidian_bridge.py`.

## n8n Workflows

Two workflow exports are included:

| File | Nodes | Role |
| --- | ---: | --- |
| `n8n_weaver_v5.json` | 31 | Current soul-binding workflow: input gateway, sanitize/error gate, self-inspect, repo/code search, five lobes, internet context, collapse, reflection, LoRA/Qwen merge, writeback. |
| `n8n_weaver_final.json` | 13 | Smaller legacy nervous-system workflow. |

The primary webhook defaults vary by environment. Check `.env`,
`deploy/env.oracle.example`, `docker-compose.yml`, and `weaver_dashboard.py` for
the active `N8N_WEBHOOK_URL`.

## Deployment

| Target | Files | Notes |
| --- | --- | --- |
| Local dev | `start_weaver.sh`, `setup_weaver.sh`, `Makefile` | Uses `venv/`, `.env`, local ports, optional VTV. |
| Docker compose | `docker-compose.yml`, `Dockerfile` | Runs Weaver core, n8n, and optional ngrok profile. |
| AWS EC2 | `deploy/README_AWS.md`, `deploy/aws-terraform/`, systemd units | t4g-style deployment with Caddy, Bedrock brain, Polly TTS, llama.cpp local experts. |
| Oracle ARM | `deploy/README_ORACLE.md`, `deploy/setup_oracle.sh`, `deploy/setup_oracle_extras.sh` | Always-free-style headless ARM path with local experts and optional public Caddy edge. |
| TTS GPU pod | `deploy/tts/pod-gpu/` | RunPod-style GPU TTS helper and restore scripts. |

Systemd units:

| Unit | Purpose |
| --- | --- |
| `deploy/weaver.service` | Headless full stack via `weaver.py --headless`. |
| `deploy/weaver-brain.service` | Bedrock brain API on `127.0.0.1:8093`. |
| `deploy/weaver-llm.service` | llama.cpp OpenAI-compatible local expert server on `127.0.0.1:8090`. |
| `deploy/tts/weaver-tts.service` | Polly/OpenVoice TTS server on `127.0.0.1:8092`. |
| `deploy/oracle-backend.service` | Companion Oracle backend service. |

## Environment

Use `deploy/env.oracle.example` for the current cloud/headless shape. The older
`.env.example` is still useful as a legacy reference.

Important variables:

| Variable | Purpose |
| --- | --- |
| `WEAVER_VAULT_DIR` | Shared persistent memory root. |
| `WEAVER_CODEBASE_ROOT` | Root exposed to the read-only codebase API. |
| `WEAVER_LLM_KEY` | Shared Caddy/API key for `/brain`, `/tts`, `/codebase`, and `/llm`. |
| `WEAVER_LLM_BACKEND` | Expert backend: `mantle`, `bedrock`, `local`, `gemini`, or `azure`. |
| `WEAVER_LOCAL_LLM_URL` | Local llama.cpp/OpenAI-compatible fallback. |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | Bedrock and Polly region. |
| `TTS_PROVIDER` | `polly` or `openvoice`. |
| `TTS_POLLY_VOICE` | Polly voice, default `Ruth`. |
| `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` | Phone/SMS/MMS bridge. |
| `DISCORD_BOT_TOKEN` | Discord bridge. |
| `IBM_QUANTUM_TOKEN` | Optional IBM hardware backend. |
| `N8N_WEBHOOK_URL` | n8n input gateway. |

## Security Notes

- Do not commit `.env`, `credentials.json`, `token.json`, `ghost_key.json`,
  vault directories, model weights, cache files, Terraform state, or `*.tfvars`.
- `codebase_api.py` is read-only, redacts secret-looking values, blocks binary
  and oversized files, and blocks private/loopback/reserved hosts for public-web
  fetches.
- Caddy gates `/brain/*`, `/tts/*`, `/codebase/*`, and `/llm/*` with
  `X-Weaver-Key` when `WEAVER_LLM_KEY` is set.
- `dash.weaverv3.com` and `status.weaverv3.com` are protected with Caddy
  `basic_auth` via `WEAVER_DASH_HASH`.
- Some legacy tests and scripts print key prefixes or call paid APIs. Run them
  only in a trusted terminal.
- Remove any live hardcoded credentials from source before publishing or sharing
  this repository.

## Tests

Unified test runner:

```bash
venv/bin/python3 tests/weaver_tests.py --list
venv/bin/python3 tests/weaver_tests.py --tier unit
venv/bin/python3 tests/weaver_tests.py --tier integration
venv/bin/python3 tests/weaver_tests.py --tier stress --dur 30
```

Longer stress and legacy suites:

```bash
venv/bin/python3 tests/stress_30min_full.py --quick
venv/bin/python3 tests/stress_30min_full.py
venv/bin/python3 whole_codebase_tests.py
```

Be careful with live tiers: they may require valid cloud credentials, n8n, local
ports, or paid APIs.

## Training And Model Forge

| Area | Files | Purpose |
| --- | --- | --- |
| Soul dataset | `forge_dataset.py`, `soul_parser.py` | Build/validate SFT data from the Fracture Principle material. |
| Hybrid routing data | `forge_fracture_dataset.py` | Generate entropy/QPU/local-routing synthetic examples. |
| Omega/reversal data | `forge_omega_fuel.py`, dataset JSONL files | Manifold reversal and QML trace examples. |
| LoRA training | `forge_soul.py`, `forge_nemotron.py` | Colab/GPU LoRA fine-tuning scripts. |
| Merge/export | `merge_lora.py`, `deploy/merge_for_gguf.py`, `deploy/build_soul_gguf.sh` | Merge adapters and produce GGUF-compatible artifacts. |
| From-scratch MoE | `pretrain_moe/` | Random-init MoE GPT pretrain kit with resumable Colab/Kaggle workflows. |
| Tool-call SFT | `pretrain_moe/distill/` | Agentic trace generation, SFT, and evaluation. |
| Bedrock distill | `ignite_bedrock_forge.py` | AWS Bedrock teacher-student distillation job scaffold. |

## Current Consistency Notes

These are codebase drift items worth fixing before a clean production release:

- `Dockerfile.twilio` references `twilio_bridge.py`, but the actual bridge file
  is `twilio_weaver_bridge.py`.
- Some comments/docs still say v4 or 7-qubit quantum. The current quantum source
  is the 156-qubit/12-core Kingston manifold in `quantum_networks.py`.
- `deploy/tts/requirements.txt` is referenced nowhere as a real file; the TTS
  service uses the project environment plus `deploy/tts/setup_tts.sh` and its
  systemd unit.
- Some legacy integration tests still assume OpenAI/Azure keys, while the current
  headless voice direction is AWS Bedrock/Nova plus Polly/OpenVoice fallback.

## Useful Commands

```bash
# Validate local config
venv/bin/python3 weaver_preflight.py

# Start headless stack
./start_weaver.sh --headless

# Start only phone bridge path
./start_weaver.sh --phone-only

# Run brain API directly
venv/bin/python3 -m uvicorn bedrock_brain_api:app --host 127.0.0.1 --port 8093

# Run TTS directly
venv/bin/python3 deploy/tts/tts_server.py

# Run dashboards directly
venv/bin/python3 health_dashboard.py
venv/bin/python3 weaver_dashboard.py

# Run read-only codebase API directly
venv/bin/python3 codebase_api.py
```
