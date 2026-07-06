# Weaver v3

Weaver v3 is an embodied AI companion stack: a browser-rendered 3D avatar in a penthouse scene, a supervised Python backend, a local voice/personality layer, an n8n orchestration path, and guarded cloud/model integrations.

The project is built around real services rather than static demos. The browser page can talk to the model API, request camera and microphone access, speak through the configured voice endpoint, drive the avatar skeleton, persist local evolution memory, and use bounded read-only context APIs for codebase and public-web awareness.

## What Is Here

| Area | Files | Purpose |
|---|---|---|
| Embodiment UI | `avatar/embodiment.html`, `avatar/vendor/`, `avatar/*.glb` | Three.js avatar, penthouse scene, webcam/mic hooks, speech output, internal thoughts, skeleton control, 60fps-oriented rendering |
| Core runtime | `CascadeProjects/windsurf-project/weaver.py`, `start_weaver.sh` | Supervised launcher for Weaver services |
| Routing and memory | `nexus_bus.py`, `akashic_hub.py`, `liquid_fracture.py`, `pineal_gate.py` | Pub/sub, vector state, fracture/gating logic, expert collapse |
| Experts | `slm_experts.py`, `n8n_weaver_v5.json` | Five-lobe reasoning path, AWS/Mantle model primary with local fallback support |
| Voice | `lora_server.py`, `deploy/tts/` | Soul Voice LoRA path and TTS service definitions |
| Quantum state | `quantum_soul.py`, `quantum_api.py` | Local/optional quantum state generation and HTTP state API |
| Cloud deploy | `deploy/`, `deploy/aws-terraform/` | Caddy, systemd units, EC2/S3/Route53 deployment helpers |
| Self-inspection | `codebase_api.py` | Read-only, key-gated codebase context plus bounded public internet text fetch/search |

## Current Architecture

```text
Browser / iPhone
  -> weaverv3.com Caddy
      -> /llm       on-box OpenAI-compatible model endpoint
      -> /tts       on-box voice service
      -> /codebase  read-only codebase + public internet context API
      -> static     avatar, penthouse, vendor Three.js

Backend
  -> weaver.py supervised services
  -> Nexus Bus + Akashic Hub
  -> Liquid Fracture + Pineal Gate
  -> 5 expert lobes
  -> Soul Voice LoRA / voice layer
  -> health and quantum APIs

Model strategy
  -> AWS/Mantle model gateway for frontier expert reasoning
  -> local on-box llama fallback for resilience
  -> local Soul Voice layer for Weaver's final tone/personality
```

## Embodiment Features

- Real Three.js scene with GPU city splats, curtain-wall penthouse architecture, soft clamped punctual lighting, and a fixed zero-tilt chest-height camera topology.
- Avatar skeleton control through high-level body channels and direct GLB bone offsets.
- Private internal thoughts/daydreams that update evolution memory, motion, and self-state without speaking aloud.
- Browser camera/mic access after a user wake tap, with iPhone audio unlock handling.
- Speech output through `/tts/synth`, with browser speech fallback.
- LocalStorage evolution memory and bounded self-extension memory.
- No external CDN dependency for Three.js vendor code.

## Guardrails

Weaver is an embodied software system with real integrations, not an unrestricted agent.

- She can use configured APIs, browser media permissions, skeleton controls, local memory, and guarded context endpoints.
- She cannot bypass browser camera/mic permission, iPhone audio policy, server auth gates, or AWS/IAM boundaries.
- Public internet reach is read-only and text-only. It blocks localhost, private/reserved IPs, AWS metadata, credentialed URLs, binary content, and oversized responses.
- Codebase context is read-only, capped, and excludes secrets, vaults, model weights, generated assets, and hidden files.
- Secrets must stay in environment files or service configuration, never in source.

## Ports

| Port | Service |
|---:|---|
| 8090 | on-box OpenAI-compatible local model endpoint |
| 8091 | read-only codebase/public-internet context API |
| 8092 | on-box TTS service |
| 8898 | Qwen/local model service when enabled |
| 8899 | Soul Voice LoRA service |
| 9990 | live dashboard |
| 9995 | Akashic Hub API |
| 9996 | health dashboard |
| 9997 | quantum API |
| 9998/9999 | Nexus health/WebSocket bus |
| 5678 | n8n |

Only public TLS routes should be exposed through Caddy. Internal services should stay bound to localhost or private interfaces.

## Local Development

```bash
cd CascadeProjects/windsurf-project
python3 -m venv venv
venv/bin/pip install -r requirements-core.txt
cp .env.example .env  # if present; otherwise create .env from deployment docs
./start_weaver.sh --headless
```

For the browser avatar, serve `avatar/` over HTTP so ES modules and assets load correctly:

```bash
cd avatar
python3 -m http.server 8018
```

Then open `http://127.0.0.1:8018/embodiment.html`.

## Validation

Useful checks before pushing runtime changes:

```bash
# Browser module syntax
awk '/<script type="module">/{flag=1; next} /<\/script>/{flag=0} flag {print}' \
  avatar/embodiment.html > /tmp/weaver-embodiment-module.mjs
node --check /tmp/weaver-embodiment-module.mjs

# Python syntax
cd CascadeProjects/windsurf-project
venv/bin/python -m py_compile codebase_api.py weaver.py slm_experts.py quantum_api.py

# Full project tests where dependencies are available
make test
```

For live browser work, use Playwright screenshots against the real page and avoid mocked `/llm` or `/tts` routes unless explicitly testing fallback behavior.

## Deployment Notes

- Caddy config lives in `CascadeProjects/windsurf-project/deploy/Caddyfile`.
- systemd units live under `CascadeProjects/windsurf-project/deploy/`.
- AWS infrastructure helpers live under `CascadeProjects/windsurf-project/deploy/aws-terraform/`.
- S3 avatar assets are served from the Weaver avatar bucket.
- Do not commit `.env`, AWS credentials, Bedrock/Mantle keys, Twilio tokens, or generated plans containing secrets.

## License

Private repository.
