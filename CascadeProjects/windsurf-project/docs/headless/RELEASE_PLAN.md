# Weaver Headless v2 Release Plan

Each tranche is independently deployable and reversible. A tranche advances
only when the full existing suite and its new contract tests pass.

## Feature flags

| Flag | Default during migration | Purpose |
|---|---|---|
| `WEAVER_HEADLESS_V2_STATE` | `0` | Build revisioned snapshots in shadow mode |
| `WEAVER_HEADLESS_V2_STREAM` | `0` | Enable the read-mostly state WebSocket |
| `WEAVER_HEADLESS_V2_SESSION` | `0` | Enable short-lived browser sessions |
| `WEAVER_HEADLESS_V2_SUMMARIES` | `0` | Replace raw thought/dream fields with safe summaries |
| `WEAVER_HEADLESS_V2_UI` | `0` | Serve the modular headless shell |
| `WEAVER_HEADLESS_V2_PROGRESS` | `0` | Emit generic long-turn progress states |

Flags are environment configuration, never browser-controlled trust decisions.
Disabling a flag restores the preceding compatibility path without changing
stored memory or Intent Capsule signatures.

The checked-in production brain unit prepares all six flags at `1`, but that
unit is not installed independently. The deployment transaction enables it only
after the modular UI, compatibility fallbacks, full validation matrix, and
rollback archive are present in the same reviewed commit.

## Atomic tranches

### R0 — Baseline and contracts

- Commit measurements, budgets, architecture ownership, and rollback rules.
- Add structural tests for the invariants.
- No production behavior change.

### R1 — Shadow state core

- Add schemas and revisioned state service.
- Produce v2 snapshots in memory while `/state` remains authoritative.
- Compare legacy and v2 public fields in tests and telemetry.

### R2 — Read-mostly transport

- Add authenticated state WebSocket in shadow mode.
- Validate heartbeat, resume, backpressure, and malformed message behavior.
- Relay verified capsule receipts only; never execute socket commands.

### R3 — Secure browser session

- Add short-lived HttpOnly session bootstrap and CSRF protection.
- Preserve the header path for native and rollback compatibility.
- Prove logout, expiry, replay, and WebSocket revalidation.

### R4 — Privacy-safe state cutover

- Enable safe thought/dream summaries.
- Remove raw private cognition from the public v2 schema.
- Keep legacy data server-side for bounded internal memory only.

### R5 — Modular web shell

- Deploy external CSS and native JavaScript modules with strict CSP.
- Install all assets and checksums atomically with the HTML entry point.
- Keep the legacy HTML artifact available to the rollback handler.

Implemented in the target tree: the active semantic entry is under 16 KiB, CSS is split
into tokens and shell rules, twelve local ES modules own the runtime, strict CSP
has no inline exceptions, deployment verifies all module hashes, and the
original 2,390-line page remains tracked for rollback. Production remains
unchanged until the later interface, voice, accessibility, and final gates pass.

The interface hierarchy is also implemented in the target tree: status header,
primary conversation log, separate read-only awareness rail, privacy-safe
cognition cards, persistent composer, and inert operator diagnostics pass
desktop and 390×844 geometry/focus checks. The authenticated transcript now
uses v2 Weaver-only SSE, bounded in-memory history, optimistic output,
stop/retry/copy controls, keyboard commands, rotating CSRF, and indefinite
generic progress within the server's bounded 115-second turn. Explicit voice
session UX is also implemented: permission and device guidance, level and
caption feedback, replay/interruption, session-authenticated WVR2 transport,
ACK/telemetry/renew/reconnect handling, session-proxied trained TTS, browser
fallback, and render-only native-shell behavior pass Test BJ and an iPhone
Chrome microphone/WebSocket simulation.

The reactive field is now a deterministic projection of the verified v2
snapshot: cognition phase, fused awareness, Neural Fabric pressure/lanes,
ledger validity, freshness, and voice state drive its bounded energy. Cumulative
private activity counts never imply current cognition, and decorative
`Math.random()` paths are gone. The iPhone 16e profile keeps a 60-FPS cadence
target while an observed-frame governor sheds DPR before cadence; reduced motion
uses the semantic 2D fallback. A same-origin install lifecycle handles visual
viewport, keyboard, orientation, and safe areas. Its service worker caches only
the public GET shell, explicitly bypasses all brain/TTS/LLM/codebase/render
traffic, and is disabled in the native render-only shell. Test BK plus real
390×844, 844×390, offline, and reduced-motion Chromium audits pass without
horizontal overflow, private cache entries, or console errors.

The browser now uses the authenticated read-mostly state WebSocket as its
primary state path. It accepts exact hello/snapshot/delta/heartbeat/progress/
receipt/error envelopes, applies only contiguous public revisions, resumes from
the last verified cursor, watches independent 10-second heartbeats, and keeps a
60-second polling safety read while connected. Its only outbound message is a
resume cursor; it has no capsule submission or execution surface. Bounded
reconnect falls back to authenticated polling and never converts a 115-second
semantic phase into a transport timeout.

Accessibility and recovery are also implemented: the diagnostics dialog makes
the background inert, traps and restores focus, supports Escape, and reports
only redacted session/network/stream/render/voice/privacy metadata. Persistent
non-sensitive controls can reduce motion, increase contrast, or hide the
ambient field while leaving equivalent text state available. Forced-colors,
system contrast/motion, coarse-pointer 44-pixel targets, deduplicated live
regions, offline/online recovery, and manual reconnect are covered by Test BL
and a real iPhone-sized browser/AX-tree audit. Tested contrast ratios are at
least 6.5:1 and the audit has no unlabeled buttons, duplicate IDs, private
diagnostic content, or page/console errors.

### R6 — Long-turn and voice UX

- Enable generic progress states and cancellable streaming output.
- Keep heartbeats independent across 115-second n8n turns.
- Add voice sequencing, interruption, renewal, and reconnect tests.

Implemented in the target tree. R6 remains undeployed until the complete
release gates are finished.

### R7 — Runtime hardening and cutover

- Prewarm SDK clients without inference under a bounded startup deadline.
- Enable per-dependency deadlines and circuit breakers, safe read coalescing,
  bounded TTL caches, and authenticated ETag revalidation.
- Point restart checks only at process liveness; use typed readiness for traffic
  admission and authenticated deep health for operator diagnostics.
- Gate cutover on redacted correlation traces, golden-signal saturation, the
  200-ms reaction budgets, and the 115-second semantic-turn budget.
- Require the exact `weaver-headless-n8n-v1` request/rejection/success union,
  correlation echo, Weaver-only speaker attestation, and validator v2 privacy
  execution checks before importing the canonical workflow.
- Enable target security/cache headers and resource constraints.
- Run load, chaos, accessibility, performance, native protocol, and n8n gates.
- Cut over only after production hash and semantic speaker-boundary checks.

The implementation portion of R7 is complete in the target tree. Immutable
n8n/ngrok images, bounded n8n execution and drain timing, hardened systemd
services, Caddy security/cache policy, a FastAPI-owned v2 session route,
exact-contract deploy smokes, and atomic Caddy/unit rollback are guarded by
Test BF. Test BM adds concurrent state, single-flight, saturation shedding,
task-cancellation, voice-first preemption, and packet-chaos coverage. The
permanent Playwright matrix is guarded by Test BN and executes at 320 px,
390×844, 768 px, 1024 px, and 1440 px. Its local release run has zero overflow,
page errors, or console errors; the iPhone 16e profile is 60 FPS at a 1.25 DPR
ceiling and recovers after an offline transition. Test BO pins and deploys the
audited Python security floor, verifies package compatibility, and constrains
the two documented advisory exceptions with exact versions and service
isolation. Production cutover and live hash/semantic verification remain the
final gates; checked-in configuration by itself does not imply deployment.

## Rollback gates

Every deployment must restore the preceding release if any of these fail:

- Weaver Brain is not the sole public speaker.
- A private draft, prompt, thought, dream, or expert label reaches a client.
- An unsigned, expired, replayed, or reflex-blocked capsule is accepted.
- The native WKWebView receives authentication material.
- Heartbeats stop during a healthy long-running semantic turn.
- The legacy polling/chat/voice fallback is unavailable.
- New static assets, CSP, or public page hashes do not match the tracked commit.
- Any whole-codebase, n8n, accessibility, or security contract fails.
