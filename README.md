# Weaver v3

Weaver v3 is an embodied AI companion stack: a browser-rendered 3D avatar in a penthouse scene, a supervised Python backend, a local voice/personality layer, an n8n orchestration path, and guarded cloud/model integrations.

The project is built around real services rather than static demos. The browser page can talk to the model API, request camera and microphone access, speak through the configured voice endpoint, drive the avatar skeleton, persist local evolution memory, and use bounded read-only context APIs for codebase and public-web awareness.

## What Is Here

| Area | Files | Purpose |
|---|---|---|
| Embodiment UI | `avatar/embodiment.html`, `avatar/vendor/`, `avatar/*.glb` | Three.js avatar, penthouse scene, webcam/mic hooks, speech output, internal thoughts, skeleton control, 60fps-oriented rendering |
| Native iPhone | `ios/WeaverNeural/` | SwiftUI shell using AVFoundation, Apple Vision, Core ML CPU + Neural Engine, authenticated cortex voice, and the render-only embodiment bridge |
| Core runtime | `CascadeProjects/windsurf-project/weaver.py`, `start_weaver.sh` | Supervised launcher for Weaver services |
| Neural Fabric | `CascadeProjects/windsurf-project/weaver_neural_fabric.py`, `bedrock_brain_api.py` | Reserved realtime/interactive/embodiment/background compute lanes, deadlines, proof ledger, and signed Intent Capsules |
| Cognition Mesh | `CascadeProjects/windsurf-project/weaver_cognition_mesh.py` | Seven-angle perception, reflex safety, digital twin, inference governance, salience memory, circuit breakers, and shadow evolution |
| Routing and memory | `nexus_bus.py`, `akashic_hub.py`, `liquid_fracture.py`, `pineal_gate.py` | Pub/sub, vector state, fracture/gating logic, expert collapse |
| Experts | `slm_experts.py`, `n8n_weaver_v5.json` | Five-lobe reasoning path, AWS/Mantle model primary with local fallback support |
| Voice | `lora_server.py`, `deploy/tts/` | Soul Voice LoRA path and TTS service definitions |
| Quantum state | `quantum_soul.py`, `quantum_api.py` | Local/optional quantum state generation and HTTP state API |
| Cloud deploy | `deploy/`, `deploy/aws-terraform/` | Caddy, systemd units, EC2/S3/Route53 deployment helpers |
| Self-inspection | `codebase_api.py` | Read-only, key-gated codebase context plus bounded public internet text fetch/search |
| Workflow validation | `scripts/validate_n8n_workflow.mjs` | Offline n8n JSON, graph, JavaScript, expression, HTTP, privacy, and deadline contract validation |

## Current Architecture

```text
Browser / iPhone
  -> weaverv3.com Caddy
      -> /llm       on-box OpenAI-compatible model endpoint
      -> /brain     AWS Bedrock brain + Nova Sonic realtime voice
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

### Neural Fabric backend

`weaver_neural_fabric.py` is the backend control plane above every model transport.
It treats compute as a deadline-bound resource rather than an unlimited request
pool:

- `realtime` reserves accelerator units for live voice;
- `interactive` handles user-facing cortex turns;
- `embodiment` handles short body, pose, and awareness decisions;
- `background` runs one bounded thought/dream at a time and sheds immediately
  when real-time work arrives.

Each lane has independent concurrency, queue, deadline, and cost-unit limits. A
weighted global capacity pool keeps four of sixteen default units reserved for
real-time work. The Proof-of-Sequence ledger stores only whitelisted operational
metadata in a bounded SHA-256 hash chain; prompts, transcripts, model output,
keys, URLs, and arbitrary error text never enter it.

The Fabric also introduces signed **Intent Capsules**: short-lived declarative
plans containing typed pose, bone, navigation, interaction, speech, observation,
or memory actions; world/body revision preconditions; an expiry; and generated
rollback operations. Capsules are HMAC authenticated but are data only—the
compile API cannot execute tools, shell commands, infrastructure changes, URLs,
or external actions.

Authenticated control surfaces:

- `GET /brain/fabric/v1/state` — lanes, pressure, p50/p95, counters, ledger proof,
  and capsule capabilities.
- `POST /brain/fabric/v1/intent/compile` — validates, bounds, signs, and returns an
  expiring Intent Capsule (rate limited to 60 compiles/minute by default).

Primary tuning variables are `WEAVER_FABRIC_CAPACITY_UNITS`,
`WEAVER_FABRIC_REALTIME_RESERVED_UNITS`, `WEAVER_FABRIC_CHAT_DEADLINE_MS`,
`WEAVER_FABRIC_VOICE_DEADLINE_MS`, and `WEAVER_FABRIC_BODY_DEADLINE_MS`.

### Seven-angle Cognition Mesh

`weaver_cognition_mesh.py` adds seven deterministic perspectives around model
output. They operate as one control loop rather than seven independent agents:

1. **Perception** fuses bounded body, camera, microphone, and penthouse state
   with freshness decay and separate body/world revisions.
2. **Embodiment** runs signed Intent Capsules through a non-LLM reflex kernel
   for awake state, balance, ground contact, collision clearance, target reach,
   state revision, and joint-delta checks.
3. **Prediction** simulates duration, energy, stability, collision risk, final
   zone, and success probability before an intent can be handed to a renderer.
4. **Compute** recommends a primary model and fallbacks from deadline, measured
   quality, accelerator pressure, cost, capability, and circuit health.
5. **Memory** keeps a bounded metadata-only hot event buffer and consolidates
   evicted events into warm salience patterns without storing prompts or media.
6. **Resilience** tracks per-component latency anomalies and closed/open/half-open
   circuit breakers, including the n8n route.
7. **Evolution** creates shadow-only policy proposals. It has no activation,
   mutation, infrastructure, or deployment operation.

Authenticated endpoints are `GET /brain/cognition/v1/state` and bounded `POST`
routes for `/observe`, `/intent/evaluate`, `/route`, and `/outcome`. Signed
capsule evaluation is separate from execution; this backend does not expose an
intent execution endpoint.

### n8n v6 parallel cognition workflow

The stable workflow ID remains `weaverv5soulbind` for database and webhook
continuity, while its response contract is `v6-parallel-cognition`. Five Mantle
expert lobes now fan out concurrently and synchronize through a five-input Merge
barrier. LoRA voice and Qwen routing also run concurrently behind a second
barrier. Timeouts and retries have a calculated 102.5-second worst-case critical
path inside the 115-second workflow deadline.

The workflow no longer imports filesystem or process modules, persists raw DLQ
input, echoes source context, or claims an Akashic write that did not occur. Run
`npm run validate:n8n` to check all nodes, connections, branches, terminal paths,
Merge inputs, JavaScript, expressions, URL allowlists, credential references,
privacy invariants, and latency budgets before import or deployment.

## Public URLs

| URL | Purpose |
|---|---|
| `https://weaverv3.com` | embodied avatar UI |
| `https://headless.weaverv3.com` | headless 3D quantum presence |
| `https://dash.weaverv3.com` | protected live operator dashboard |
| `https://status.weaverv3.com` | protected health dashboard |
| `https://weaverv3.com/brain/*` | key-gated Bedrock brain API |
| `wss://weaverv3.com/brain/realtime/voice` | Nova Sonic realtime voice |
| `https://weaverv3.com/tts/*` | key-gated AWS Polly TTS |
| `https://weaverv3.com/codebase/*` | key-gated read-only source context |

## Embodiment Features

- Real Three.js scene with GPU city splats, curtain-wall penthouse architecture, soft clamped punctual lighting, and a fixed zero-tilt chest-height camera topology.
- Avatar skeleton control through high-level body channels and direct GLB bone offsets.
- Original `cinematic-micro-motion-v1` performance layer: deterministic eye-led
  attention, natural blink/double-blink cadence, damped saccades, three-link neck
  response, five-link spinal distribution, scapular rhythm, wrist settling, and
  individual finger-chain curl instead of one repeated whole-body idle loop.
- Bone-driven speech articulation across 29 mapped facial deformation bones.
  Text is converted into coarticulated closed/open/wide/round/teeth/tongue
  visemes and synchronized to trained or browser voice playback; jaw, lips,
  tongue, eyes, and eyelids remain independent channels.
- Hard-bounded second-order follow-through for left/right breast bones, pelvic
  mass, tailored garment panels, and gravity hair. Skeleton intent leads each
  layer while tissue, cloth, and tapered braids lag and settle at their own
  frequencies; amplitudes are clamped to anatomical limits.
- Texture-preserving skin, woven plum fabric with a close-tailored layered skirt,
  deterministic pore microdetail, wet-cornea eye response, softened portrait
  lighting, pleated cloth normals, and 18/24/26/30-strand tapered
  gravity-constrained micro-braid profiles across performance, iPhone 16e,
  lite, and full-quality tiers.
- Full, medium, and portrait camera framing presets. Voice playback blends to a
  conversation shot so small eye and lip motion remains legible, then returns to
  full-body/environment framing.
- Reproducible high-fidelity character asset pipeline in
  `avatar/build_hifi_avatar.py`. It converts the original 90,088-triangle avatar
  into a 237,568-triangle desktop LOD with weighted Phong subdivision while
  preserving all 168 nodes, 163 rig bones, skin weights, UV seams, and animation
  accessors. The same build derives UV-aligned 2K normal, roughness, and specular
  maps from Weaver's own skin texture.
- Desktop and iPhone 16e-class rendering automatically select
  `weaver_avatar_dress_hifi.glb`, three physical skin maps, PMREM studio
  image-based lighting, and two layered corneal shells when hardware rendering
  and the network policy allow it. Data-saver, constrained, and
  software-rendered devices keep the standard LOD; `?avatar=hifi` and
  `?avatar=standard` provide deterministic operator overrides.
- The iPhone 16e A18 tier targets 60 fps for touch, body control, face, and
  camera response at a bounded 1.25 pixel ratio. Environment shaders and
  gravity-hair constraints run at 30 Hz, startup stalls are excluded from the
  thermal governor, and dynamic resolution is reduced before body cadence.
  Safe-area layout, `100dvh`, debounced `visualViewport` resizing, hidden-page
  suspension, and WebGL context recovery are built in. Inspect the live policy
  with `__weaverMobilePerformanceAudit()`; `?device=iphone16e` is a deterministic
  test override for Safari emulation.
- Twenty stateful penthouse interactions across 72 indexed environment objects;
  navigation and interaction outcomes feed body/environment awareness.
- Private internal thoughts/daydreams that update evolution memory, motion, and self-state without speaking aloud.
- Browser camera/mic access after a user wake tap, with iPhone audio unlock handling.
- Speech output through `/tts/synth`, with browser speech fallback.
- LocalStorage evolution memory and bounded self-extension memory.
- No external CDN dependency for Three.js vendor code.

Runtime regression surfaces are `__weaverCinematicMotionAudit()`,
`__weaverFacialAudit()`, `__weaverSoftTissueAudit()`, and the bounded
`__weaverFacialTest()` / `__weaverFacialReset()` pair. The codebase suite checks
the browser contracts and verifies the required deformation bones directly from
the local GLB with `cd CascadeProjects/windsurf-project && venv/bin/python3 whole_codebase_tests.py AK`.
High-fidelity geometry, skin weights, PBR maps, LOD selection, and deployment
checks are covered by the adjacent `AL` test.
The iPhone 16e device profile, split scheduling, dynamic-resolution ordering,
Safari lifecycle behavior, and full-body-per-frame contract are covered by
test `AM`.

Rebuild the original high-fidelity asset and material maps with:

```bash
python3 avatar/build_hifi_avatar.py
```

The builder uses only Weaver's local, original source asset. It does not import
or reproduce GTA characters, geometry, textures, or animation data.

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
| 8091 | read-only codebase/public-internet context API, supervised by `weaver.py` |
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

The native iPhone shell is generated on macOS with XcodeGen:

```bash
cd ios/WeaverNeural
make generate
open WeaverNeural.xcodeproj
```

Run the generated `WeaverNeural` scheme on a physical iPhone to exercise the
front camera, microphone, Apple Vision, and Neural Engine eligible Core ML path.

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
