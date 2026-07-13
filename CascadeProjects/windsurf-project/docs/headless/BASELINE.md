# Weaver Headless Baseline

Measured on 2026-07-13 against the production headless origin before the v2
headless work began. These measurements are the comparison point for the
budgets in `performance-budgets.json`; they are not all acceptance targets.

No authentication material, prompt text, private thought, private dream, or
model response is stored in this document.

## Web measurement method

- Playwright Chromium, clean context per viewport.
- Production URL: `https://headless.weaverv3.com/`.
- The page was inspected only after `networkidle`, document completion, and
  visual boot readiness.
- FPS is a one-second `requestAnimationFrame` sample in headless Chromium.
- The absolute FPS values are not a substitute for physical iPhone/Safari
  measurement; they are a reproducible software-rendered regression signal.

## Web results

| Viewport | TTFB | FCP/LCP | CLS | Long tasks | FPS |
|---|---:|---:|---:|---:|---:|
| 320×568 | 174.1 ms | 360 ms | 0.00124 | 24 | 20.7 |
| iPhone 16e CSS viewport, 390×844 | 135.2 ms | 368 ms | 0.00046 | 24 | 13.9 |
| 768×1024 | 178.7 ms | 372 ms | 0.00020 | 14 | 7.0 |
| 1024×768 | 119.6 ms | 336 ms | 0.00016 | 14 | 6.9 |
| 1440×900 | 230.9 ms | 452 ms | 0.00007 | 13 | 4.8 |

Stable behavior already present:

- HTTP 200 and visual boot completion at every viewport.
- Zero console errors and zero page errors.
- Zero horizontal overflow.
- All visible controls have names and meet the 44-pixel touch target.
- Cumulative layout shift is already small.
- Locked mode does not ask for the key until the user selects Wake.

Measured deficiencies:

- The page has no semantic `header`, `main`, or `aside` landmarks.
- The WebGL field is severely fill-rate/overdraw bound as viewport size grows.
- The page produced 13–24 long tasks during the measurement window.
- The HTML is 88,040 decoded bytes and 26,037 transferred bytes.
- The single Three.js resource path brought total decoded resource bytes to
  1,272,972 and transferred resource bytes to 281,966.
- Production returned compression headers but none of the target CSP,
  content-type, frame, referrer, permissions, or explicit cache headers.

## Backend results

Seven local production samples per read endpoint:

| Endpoint | Median | p95 | Maximum |
|---|---:|---:|---:|
| `/state` | 2.87 ms | 3.10 ms | 3.10 ms |
| `/fabric/v1/state` | 2.45 ms | 2.50 ms | 2.50 ms |
| `/cognition/v1/state` | 2.40 ms | 2.46 ms | 2.46 ms |
| `/realtime/voice/config` | 0.89 ms | 1.05 ms | 1.05 ms |

Additional observations:

- Health was nominal; the first measured health request took 36.44 ms.
- Two bounded full-cortex conversational turns took 8.78 s and 6.34 s.
- Both turns kept `weaver-brain` as the selected specialist and public speaker,
  with the speaker boundary applied and private draft hidden.
- The voice SLO window contained no post-restart samples, so it did not prove
  either the 200 ms reaction target or the 3 s semantic target.
- State has legacy timestamps and tick counters but no monotonic revision.
- Awareness confidence was `0.0` without a fresh native/body observation.
- Intent Capsules are HMAC-SHA256 signed, expire within 60 seconds, contain at
  most eight actions, and allow exactly seven action types.

## Native iOS result

`ios/WeaverNeural/Tools/validate_repository.py` passed all 44 repository files.
Core ML, asset, privacy, security, voice, and credential-free render-bridge
contracts were valid. This is a structural Linux-side baseline; physical A18,
AVFoundation, Vision, Neural Engine, and VoiceOver behavior still require an
iPhone/Xcode validation pass.
