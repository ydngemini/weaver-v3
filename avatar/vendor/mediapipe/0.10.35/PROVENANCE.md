# MediaPipe face detector assets

Version-pinned, same-origin browser assets used by `avatar/embodiment.html` for
on-device face tracking.

- Package: `@mediapipe/tasks-vision@0.10.35` (Apache-2.0)
- Package source: <https://www.npmjs.com/package/@mediapipe/tasks-vision>
- Upstream project: <https://github.com/google-ai-edge/mediapipe>
- Model: MediaPipe BlazeFace short range, float16
- Model source: <https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite>

The standard and no-SIMD WASM loaders are vendored. The page calls
`FilesetResolver.forVisionTasks(..., false)`, requests the MediaPipe `GPU`
delegate on iPhone, and retries with `CPU` if the browser cannot initialize the
GPU delegate. No runtime CDN or server-side GPU is required.

## SHA-256

```text
b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f  blaze_face_short_range.tflite
55d7ab624fbb70dcc5adc4ae6d7ea9cfcb569139d3dbfbf2b1deafcb966bc0fe  vision_bundle.mjs
e7fd9858e8e8f221d9b96eddc11f8e077f263e0b7bbd79d3cbe882b134274f8c  wasm/vision_wasm_internal.js
6a5c64584c2ab61c763b6e204afbdbc7ce1caf7f5216187322bca8df94f646bc  wasm/vision_wasm_internal.wasm
438d1fe8ff7f4d946025bc211c291543c037d8a3785ed4eee60f1f521b236296  wasm/vision_wasm_nosimd_internal.js
8a3092d34c79d3f57e6ba8592105e8a90f6b07c27891ffecd14cca428bfd3e31  wasm/vision_wasm_nosimd_internal.wasm
```
