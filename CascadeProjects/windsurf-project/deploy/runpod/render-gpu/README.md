# Weaver RunPod GPU Splat Renderer

Runs a real CUDA/Torch Gaussian-style splat renderer on the RunPod GPU and
streams MJPEG frames on port `8888`.

Endpoints:

- `GET /health` or `/metrics` - renderer/GPU health JSON
- `GET /frame.jpg?w=1280&h=720` - one rendered frame
- `GET /stream.mjpg?fps=60&w=1280&h=720` - live MJPEG stream

The browser page composites this under the local Three.js avatar through the
same-origin Caddy route `/gpu-render/*`.
