# Weaver Headless v2 Architecture Contract

Status: accepted implementation contract.

This contract evolves headless mode without creating a second brain, command
system, embodiment authority, or mobile sensor stack.

## Non-negotiable invariants

1. Weaver Brain is the only public conversational speaker.
2. Private expert, coder, thought, dream, prompt, and chain-of-thought content
   never crosses a public state or progress channel.
3. Intent Capsules remain the only declarative mutation authority.
4. A transport may relay a signed capsule or its evaluation receipt; it may not
   accept an arbitrary command, shell string, JavaScript body, or action name.
5. `IntentCompiler.verify()` must pass before Cognition Mesh evaluation.
6. Cognition Mesh and its reflex kernel evaluate plans; they do not become a
   hidden execution engine.
7. The native iOS shell owns camera, microphone, Apple Vision, Core ML, Keychain,
   and native realtime voice lifecycles.
8. The native WKWebView remains render-only, nonpersistent, and credential-free.
9. The mobile website is a browser fallback, not a replacement sensor or voice
   authority inside the native shell.
10. The 200 ms target covers an audible or visual reaction acknowledgement. It
    is not represented as full semantic completion.
11. n8n semantic work may legitimately remain active for its full validated
    115-second budget without making the state transport stale.
12. Every release preserves the current polling, HTTP chat, and voice paths
    until its replacement passes production smoke tests.

## Ownership boundaries

| Concern | Authority |
|---|---|
| Public conversation | Unified cortex with Weaver Brain public speaker |
| Intent compilation and signature | `IntentCompiler` in Neural Fabric |
| Intent integrity verification | `IntentCompiler.verify()` |
| Safety and precondition decision | Cognition Mesh `ReflexKernel` |
| Counterfactual plan scoring | Cognition Mesh digital twin |
| Body action application | Existing embodied/native consumer only |
| Headless state scheduling | Headless service on Neural Fabric lanes |
| Realtime state delivery | Headless v2 transport service |
| Native sensors and microphone | SwiftUI/AVFoundation/Vision/Core ML shell |
| Browser rendering and browser mic | Browser headless fallback |
| Expert fan-out and synthesis | Validated n8n v6 workflow |

## Target backend modules

The migration extracts focused modules while leaving the existing public
routes as compatibility adapters:

- `headless_schemas.py`: Pydantic contracts and public error envelopes.
- `headless_state.py`: atomic revisioned snapshots and bounded delta history.
- `headless_scheduler.py`: private thought/dream scheduling and QoS admission.
- `headless_transport.py`: authenticated WebSocket lifecycle and subscriptions.
- `headless_privacy.py`: safe summaries and public-payload redaction.
- `awareness_fusion.py`: deterministic body/world/control-plane freshness fusion.
- `runtime_resilience.py`: bounded caches, request coalescing, ETags, and
  dependency circuit breakers.

`bedrock_brain_api.py` remains the composition root until the cutover is proven.

## Versioned state contract

Every complete state message contains:

- `schema_version`: integer protocol version.
- `revision`: monotonically increasing integer.
- `generated_at`: UTC timestamp.
- `freshness`: bounded source ages and stale flags.
- `system`: public health, readiness, and degraded reasons.
- `awareness`: one bounded confidence verdict across body, world, Cognition
  Mesh, Neural Fabric, and dependency freshness, with stable degraded reasons.
- `voice`: lifecycle and SLO metadata without transcript content by default.
- `cognition`: safe counts/status only; never raw private reasoning.
- `fabric`: lane pressure and ledger validity without private payloads.

A delta names its `base_revision` and `revision`. A client that cannot apply a
delta requests a fresh snapshot instead of guessing.

The fusion layer treats body and world freshness as safety-critical, carries
camera and microphone staleness as explicit limitations, and fails closed on
an invalid Fabric proof ledger or a stale required dependency. Optional route
loss (for example n8n while the direct cortex fallback is healthy) yields a
`limited` state instead of falsely declaring Weaver unavailable. A dependency
in `busy` state stays fresh for the duration of a bounded long turn, so the
115-second n8n budget cannot be mistaken for a dead service.

## Realtime voice reliability

Native voice protocol v2 keeps PCM efficient by prefixing each binary frame
with `WVR2`, a monotonic sequence number, and capture time. The server uses a
24-frame/120-ms bounded reorder window, acknowledges the highest contiguous
sequence, counts duplicates and loss, and never grows the buffer to conceal a
bad link. Legacy unframed PCM remains available as a rollback path.

Every server event has a monotonic sequence and may be acknowledged by the
client. Short-lived, single-use resume tickets preserve only sequence cursors;
the authenticated brain key is still required. Sessions announce renewal 30
seconds before the upstream streaming limit and publish a 250-ms-to-8-second
exponential reconnect policy with 20% jitter. An interruption advances a
generation, cancels in-flight realtime Fabric work, and prevents a stale answer
from speaking after a newer user turn.

Device telemetry is restricted to bounded link/audio/thermal categories. It
accepts no device identifier, transcript, raw sensor data, route name, or
credential. Native AVFoundation remains the microphone authority; the web
surface is only a fallback client of the same protocol.

## Cancellable public chat stream

Chat does not share the state WebSocket and cannot carry body commands. An
authenticated, CSRF-protected `POST /headless/v2/chat/stream` starts one of at
most four interactive turns and returns bounded server-sent events. The first
events are an immediate acceptance cue and generic `queued`/`thinking`
progress, which keep a 115-second n8n turn visibly alive without exposing
expert names, drafts, prompts, routes, or chain of thought.

Internal specialists complete behind the existing public-speaker checks. Only
the final answer that passes the Weaver boundary is divided into public deltas;
streaming an unvalidated model draft is prohibited. Disconnecting the stream
or calling `DELETE /headless/v2/chat/{turn_id}` cancels the associated
interactive Fabric task. Grounding is preserved because the stream invokes the
same unified cortex and server-owned codebase context path as typed chat.

## Runtime resilience and read efficiency

SDK client prewarming initializes cached clients only; it never sends a model
inference request. One bounded worker serializes credential-provider access and
the complete prewarm has a 12-second default deadline. An unavailable prewarm
degrades readiness metadata but cannot block server startup.

Bedrock, Mantle, n8n, and the on-box cortex each have explicit I/O deadlines and
independent closed/open/half-open circuit breakers. Breaker snapshots contain
only stable counters and retry timing, never raw exceptions or endpoints.
Caller cancellation is neutral: it releases the recovery probe without
misclassifying a user interruption as dependency failure. n8n circuit loss
continues through the established direct-cortex fallback rather than creating a
new response path.

Identical read-only code-grounding work is coalesced and cached for ten seconds
in a 32-entry copy-isolated cache. State initialization is coalesced only while
the revision store is empty; later mutations are never combined, so the newest
awareness revision cannot be lost. State responses have deterministic
revision-prefixed ETags and support an empty-body `304` when an authenticated
client explicitly revalidates. They retain `no-store` because privacy takes
precedence over implicit browser caching.

The read-only source index treats an explicitly named filename as a hard
retrieval constraint. File admission is bounded at 384 KiB, search reads remain
capped at 256 KiB, and returned context remains capped at 12 KiB.

## Liveness, readiness, and deep health

`GET /health/live` is an unauthenticated, process-only event-loop check. It has
no dependency reads, model calls, state refreshes, or filesystem writes and is
the only endpoint suitable for restart liveness.

`GET /health/ready` evaluates current in-process control-plane state. It returns
`503` only when a required component cannot safely accept work, such as an
invalid Fabric ledger, an unavailable cortex fallback set, or an enabled state
store that has not initialized. Load pressure and active work remain available;
they do not invite a restart. The existing `/health` response remains unchanged
for compatibility.

`GET /health/deep` requires the operator key, is rate-limited and no-store, and
adds bounded status-only probes for configured local dependencies. It never
reads a response body, invokes a model, or exposes an endpoint, raw exception,
prompt, transcript, route, or model ID. An active n8n request is reported as
`busy` and is never probed, preserving the full 115-second semantic envelope.
Deep reports always remain reachable with HTTP 200 because they are operator
diagnostics, not restart signals.

## Correlation, telemetry, and SLOs

One correlation ID follows an HTTP or WebSocket connection through async child
tasks, the state transport, typed chat, live voice, and the n8n request payload.
Client-supplied IDs are accepted only when they match the 64-character safe
identifier grammar. New v2 and diagnostic responses echo the ID; legacy
responses preserve their existing header shape unless a caller explicitly
supplies one.

The observability collector never reads a request or response body. Its
structured JSON events and in-memory traces contain only allowlisted categorical
fields such as normalized route, method, phase, lane, protocol, revision, and
result code. Prompt, message, content, transcript, model, endpoint, exception,
credential, and arbitrary diagnostic fields have no storage path. Trace history
is capped at 256, each operation retains at most 256 samples, and operation
cardinality is capped at 64 with overflow aggregated into `runtime.other`.

Golden signals cover traffic, server/client errors, latency percentiles, current
in-flight work, and maximum observed saturation. Error budgets distinguish the
200-ms reaction cue from semantic completion: chat reaction targets 200 ms,
chat semantic completion follows the 115-second n8n envelope, voice reaction
targets 200 ms, and voice semantic latency retains its 3-second soft target.
Caller cancellation is counted but does not burn availability budget.

`GET /health/observability` is operator-key authenticated, rate-limited, and
no-store. It exposes only strict aggregate schemas, up to 32 recent bounded
traces, the headless error-budget view, and the existing bounded voice SLO
summary.

## Versioned n8n speaker boundary

The canonical workflow accepts only `weaver-headless-n8n-v1`. The request has
an exact key set, a safe correlation ID, a fixed 115,000-ms deadline, bounded
text/source/search fields, and a closed scalar Cognition Mesh context. Missing,
additional, aliased, oversized, or mistyped fields take the metadata-only
rejection path. Correlation is echoed exactly so the brain can reject a validly
shaped response from the wrong turn.

The workflow may retain expert drafts, collapse geometry, routes, source
evidence, and local-model errors as private transient state while it reasons.
The terminal node has only two public shapes: an exact Weaver success or an
exact rejection without speech. Success is possible only after the Qwen Weaver
speaker boundary and reflection flags pass. The public envelope contains the
reviewed manifestation plus bounded aggregate expert counts; it cannot contain
prompts, drafts, lobe identity, raw errors, source, LoRA text, routing, or
geometry.

FastAPI validates the same discriminated union again, requires the response
correlation ID to match the request, and exposes only an allowlisted route
summary. Any schema drift, added private field, rejection, or correlation
mismatch is discarded and the direct Weaver cortex remains the fallback. The
offline validator executes adversarial request and writeback cases in addition
to checking all 35 nodes, 42 edges, sandbox syntax, privacy rules, and the
102,500-ms critical path inside the 115,000-ms workflow deadline.

## Public edge and runtime containment

Caddy routes `/brain/headless/v2/*` before the legacy shared-key matcher. The
dedicated route strips only `/brain` and delegates browser-session, CSRF, and
stable public-error decisions to FastAPI; it is not an authentication bypass.
Legacy brain, codebase, TTS, and local-LLM routes retain the edge key gate.
Realtime and long-turn connections have independent bounded transports: the
state/voice path has a 30-second handshake limit and 15-minute stream lifetime,
while semantic HTTP responses allow 130 seconds so the validated 115-second n8n
turn cannot be cut off by the proxy.

All public origins emit HSTS, content-type, frame, referrer, opener, permissions,
and CSP headers. HTML and authenticated/API responses are `no-store`;
versionless static assets receive a bounded one-day cache with one week of stale
revalidation. The active headless entry contains no inline script, style, or
event handler, so its CSP allows only same-origin scripts and styles without
`unsafe-inline`. The prior 2,390-line page remains a tracked rollback artifact;
the deployment transaction restores its matching prior Caddyfile and web root
rather than trying to run that legacy artifact under the strict current CSP.

## Modular browser shell

The active `headless.html` is a sub-16-KiB semantic entry point. Design tokens
and layout rules live in separate CSS resources. Twelve native ES modules separate
shared state/UI helpers, short-lived browser sessions, voice transport support,
voice orchestration, visual constants, semantic visual runtime, visualization,
cortex access, realtime state recovery, install/viewport lifecycle,
accessibility/operator controls, and composition.
Imports are local and dependency-light; Three.js remains the
only dynamic visual dependency and is served from the pinned local vendor tree.
There is no package manager, bundler, hydration runtime, or third-party CDN on
the critical path.

Deployment requires every module and stylesheet, copies them as one directory,
and compares each deployed SHA-256 through the local TLS edge. The entry HTML
and old public-page checks remain, while whole-codebase Test BG enforces the
module graph, independent syntax, no-inline entry, design-token split, strict
CSP, checksum coverage, and rollback artifact. A 390×844 browser smoke also
proves all twelve external resources load. The iPhone 16e tier preserves a 60-FPS
cadence target with a 1.25 DPR ceiling and adaptive resolution; reduced motion
uses the deterministic 2D fallback. Controls retain their mobile target sizes.

## Headless workspace hierarchy

The reactive field remains an ambient representation, not the information
architecture. A compact status header reports presence, brain, voice, privacy,
and diagnostics. Conversation is the primary `role=log` workspace and labels
the verified Weaver-only speaker boundary. A separate read-only awareness rail
contains fused confidence, bounded activity counts, safe cognition metadata,
and the privacy explanation. The persistent composer owns wake/live/mic/text
controls without moving operator diagnostics into the normal conversation.

The diagnostics drawer is inert and `aria-hidden` while closed, takes focus only
after it becomes visible, closes with Escape or its scrim, and returns focus to
the opener. Its copy explicitly limits it to bounded operational metadata. The
desktop grid preserves a clear visual center between conversation and awareness;
at 390×844 the workspace becomes a vertical scroll region while the composer
stays inside the safe viewport. Browser geometry tests prove no horizontal
overflow, non-overlapping desktop columns, visible primary regions, and correct
drawer focus behavior at both viewports.

Even the legacy polling compatibility path no longer renders raw thought or
dream text. It maps only counts, availability, and allowlisted safe topics into
fixed privacy language. A dream trigger likewise reports completion without
placing the returned private content in the DOM. Test BH guards this hierarchy,
ID wiring, privacy projection, drawer behavior, and responsive contract.

## Authenticated conversation lifecycle

The active browser sends the long-lived Weaver key only to the v2 session
bootstrap, removes any prior local/session-storage copy, and uses the returned
HttpOnly SameSite cookie plus rotating CSRF token for subsequent state, chat,
cancellation, and renewal requests. Session metadata is memory-only and the
deployed diagnostics expose status and expiry—not credentials.

Text turns render an optimistic operator message and a pending Weaver message,
then consume bounded SSE frames from `/headless/v2/chat/stream`. Only ordered
`speaker=weaver` deltas can enter the DOM, and all insertion uses `textContent`.
The transcript keeps a bounded in-memory history and offers stop, retry, and
copy controls plus Enter/Shift+Enter, Command/Control-K, and Escape keyboard
behavior. Generic queued/thinking/synthesizing progress remains active for the
full server-authoritative turn: the client has no semantic timeout, so the
five-second progress cadence and 130-second edge allowance safely cover the
115-second n8n deadline. Test BI holds this session, speaker, stream, budget,
interaction, and deployment contract against drift.

## Explicit voice session

The browser voice tray presents permission guidance before capture, a live level
meter, local device class, bounded captions, the measured reaction acknowledgment,
and Replay, Interrupt, Hide, Live/Stop, and one-shot Mic controls. Permission
denial and missing recognition/audio support retain typed chat and explain the
recovery action. Browser captions accept a public agent response only when its
speaker is `weaver`; specialist text is rejected before display.

Live browser capture uses the same HttpOnly session as state/chat and proves the
rotating CSRF token in the WebSocket subprotocol. Audio uses the native-efficient
WVR2 sequence/timestamp envelope, input and output acknowledgments, privacy-safe
device telemetry, 15-second pings, resumable renewal, explicit barge-in, and an
eight-attempt jittered exponential reconnect bound. The server revalidates the
still-live session every 30 seconds. SwiftUI clients retain their key-subprotocol
bridge and remain the sole owners of AVFoundation capture, Vision, Core ML, and
device sensors; `nativeShell=1` therefore makes the web layer render-only.

Trained browser speech now goes through `/headless/v2/voice/synth`, a CSRF-checked,
rate-limited FastAPI proxy that permits only the fixed loopback TTS service and
bounds request text, time, response bytes, and audio content types. The service
key never enters an ordinary browser synthesis request. Browser speech remains a
local fallback, and responses wait for a Wake/Replay gesture when audio has not
been unlocked. Test BJ plus the iPhone Chrome voice audit prove session auth,
WVR2 framing, permission/caption/interruption/reconnect behavior, native ownership,
speaker containment, and zero console/page errors.

## Semantic field and mobile lifecycle

The ambient field consumes only the exact v2 public snapshot. Its deterministic
signal mapper projects current cognition phase, awareness status/confidence,
Neural Fabric pressure and active/queued lanes, ledger validity, source
freshness, voice status, and monotonic revision. It does not infer active work
from cumulative thought/dream counts, consume private cognition, or execute an
Intent Capsule. Seeded geometry and revision/time-derived noise replace runtime
randomness, so a repeated verified state produces the same bounded visual
semantics while animation remains continuous.

The render governor records frame interval and bounded work estimates. On the
iPhone 16e class it targets 60 FPS, caps physical DPR at 1.25, and reduces a
0.65–1.0 render scale under sustained pressure before changing cadence. Recovery
is deliberately slower than shedding. Reduced motion, thermal pressure, low
power mode, and Save-Data select bounded efficiency profiles; reduced motion or
missing WebGL uses the same semantic mapping in the 2D field. The visual audit
exposes profile, effective DPR, scale, frame metrics, viewport, and safe semantic
metadata without prompts or transcripts.

`visualViewport`, screen orientation, safe-area insets, and keyboard inset feed
one app-height contract for portrait and landscape. The install lifecycle is
browser-only: `nativeShell=1` disables service-worker registration because the
SwiftUI shell owns persistence, credentials, microphone, camera, Vision, and
Core ML. The browser service worker precaches only same-origin public GET shell
assets. It never handles or caches `/brain/`, `/tts/`, `/llm/`, `/codebase/`, or
`/gpu-render/`; offline session bootstrap stops before making a private request.
The worker and manifest are edge `no-store` entries, while their versioned public
dependencies retain bounded static caching. Test BK and the iPhone portrait,
landscape, offline, and reduced-motion audits enforce these boundaries.

## Realtime recovery and inclusive operator controls

After one authenticated HTTP snapshot, the browser opens
`/headless/v2/stream` with its HttpOnly session and in-memory CSRF WebSocket
proof. The client accepts structurally exact public messages, applies only
monotonic snapshots or contiguous top-level public deltas, and sends only a
resume cursor. It cannot submit a capsule, compile an intent, call an execution
adapter, or turn a receipt into an action. On a gap it requests resume; on a
resync error it refreshes through authenticated HTTP. A heartbeat watchdog and
eight-attempt bounded exponential reconnect run independently of semantic turn
duration, so n8n may remain in `thinking` for its full 115-second budget while
10-second state heartbeats keep the transport healthy. Polling remains active
as a 60-second safety read when connected and becomes the primary bounded
fallback during recovery.

Visible recovery state is deduplicated for assistive technology. Offline mode
pauses the socket before any private request, leaves the public shell usable,
and reconnects state when the browser reports the network restored. The manual
Reconnect action performs the same session/state/stream sequence. Diagnostics
report connection status, bounded stream age, poll failures, session lifetime,
shell status, render profile/DPR, voice reconnect count, revision/freshness,
reaction target, semantic budget, speaker boundary, and reflex authority. They
never read transcript, caption, topic, prompt, draft, or credential fields.

The diagnostics surface is an `aria-modal` dialog. Opening it makes the app
shell inert, moves focus inside, traps forward/backward Tab, closes with Escape
or the scrim, and restores the actual prior control. Sensory preferences store
only three booleans: user-reduced motion, higher contrast, and ambient-field
visibility. System reduced-motion and contrast requests always remain
authoritative; forced-colors hides the decorative canvas while preserving text.
On coarse pointers every visible button is at least 44 pixels high. Test BL and
the 390×844 browser accessibility-tree audit enforce focus, labels, unique IDs,
live regions, recovery, redaction, target size, contrast, and zero console/page
errors.

The production systemd units run as unprivileged service users where the
existing runtime permits it, drop ambient and bounding capabilities, isolate
devices and temporary directories, restrict kernel and namespace surfaces, and
apply process, file-descriptor, CPU, memory, restart, and shutdown bounds. ML
services intentionally retain executable-memory compatibility where native
inference/JIT libraries require it. n8n runs from one immutable image digest
with a read-only root, no capabilities, no-new-privileges, no-exec temporary
filesystems, a non-root image user, bounded logs/resources, loopback listeners,
disabled public/editor/community-package surfaces, blocked runner environment
access, and no saved execution bodies.

n8n's execution deadline is 115 seconds, its task runner is allowed 120 seconds,
graceful application shutdown is 125 seconds, Docker stop is 130 seconds, and
systemd stop is 150 seconds. This ordering preserves a legitimate long turn
without permitting an unbounded drain. The optional ngrok tunnel is digest
pinned and its inspector is loopback-only.

Deployment installs and validates the Caddyfile and all five service units
before restart, imports only the validator-approved canonical workflow, sends
an exact versioned n8n smoke request, and requires the exact Weaver-only public
response. It then proves the browser session lifecycle and security/cache
headers through the local TLS edge. Failure restores the prior tracked tree,
units, Caddyfile, web roots, and pre-import n8n database. Whole-codebase Test BF
holds these edge, container, unit, and transaction invariants against drift.

## Realtime state protocol

Production uses `wss://`. The state channel is independent from the Nova Sonic
audio socket so state backpressure cannot corrupt audio cadence.

Server messages:

- `hello`: protocol version, heartbeat interval, current revision.
- `snapshot`: complete public headless state.
- `delta`: bounded changes from a named base revision.
- `progress`: public turn phase only.
- `capsule_receipt`: capsule ID plus verification/reflex/evaluation status.
- `heartbeat`: transport liveness independent of inference.
- `error`: stable public error code, retryability, and correlation ID.

Client messages:

- `subscribe`: allowlisted public channels only.
- `resume`: last applied revision.
- `ping`: client liveness probe.
- `capsule_submit`: an already signed Intent Capsule only.

`capsule_submit` is rejected unless the payload is a structurally exact capsule,
its signature and expiry verify, its ID has not been replayed, and its action
types remain within the compiler allowlist. Acceptance routes it through the
existing Cognition Mesh evaluation/reflex path. The socket never applies the
actions itself.

## Long-turn lifecycle

The public lifecycle is deliberately generic:

1. `accepted`
2. `queued`
3. `thinking`
4. `synthesizing`
5. `completed`, `cancelled`, or `failed`

No message reveals expert names, lobe names, prompts, drafts, or internal model
selection. Heartbeats continue throughout all phases. The client treats a
missing heartbeat as a transport problem, but does not treat a long `thinking`
phase as a transport timeout. Server semantic deadlines remain authoritative.

## Browser and native mobile relationship

The standalone mobile website may use browser media APIs only after explicit
permission and only for its own session. It receives adaptive rendering,
safe-area, orientation, reconnect, and offline-shell improvements.

The native SwiftUI app continues to use:

- AVFoundation for camera and microphone capture.
- Apple Vision and Core ML for bounded attention/face state.
- URLSessionWebSocketTask for authenticated cortex voice.
- Keychain for the long-lived key.
- A nonpersistent WKWebView for render-only body and room output.

No web optimization may move credentials or raw native sensor frames into the
WKWebView. The existing bounded native render bridge remains the only interface.

## Authentication target

The compatibility key header remains until a short-lived session path is live.
The target browser flow exchanges the long-lived key once for a host-only,
Secure, HttpOnly, SameSite=Strict session cookie. State-changing HTTP requests
also require a bounded CSRF value held in memory. WebSocket authentication occurs
during the handshake and is periodically revalidated. No key enters a URL,
log, local storage, state payload, or error response.

## Compatibility and rollback

The v2 transport runs in shadow mode before UI adoption. Polling remains the
fallback. The old single-file UI remains deployable until the modular shell,
CSP, asset installation, checksums, and public smoke tests pass as one release.
