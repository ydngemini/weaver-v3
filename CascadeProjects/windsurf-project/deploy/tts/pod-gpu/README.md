# Weaver GPU voice — XTTS-v2 on RunPod

Her production voice runs here: **XTTS-v2** (Coqui) on a **RunPod RTX 4090**,
cloning from `weaver_voice_ref.wav`. Fresh synth ≈ **0.9s on the GPU** (~1.8s
end-to-end through weaverv3.com incl. the AWS↔RunPod hop) vs 12–28s on the CPU
box. Higher fidelity than the CPU OpenVoice clone, same wire contract so
`embodiment.html` needs no change.

## Where it lives
- Pod: `f6t7m2t1errly1` (RTX 4090, Ubuntu 24.04, torch 2.8+cu128, Europe).
- Server: `tts_server_gpu.py` on the pod at `/root/tts/`, bound `0.0.0.0:8888`
  (8888 is the pod's exposed RunPod HTTP-proxy port; Jupyter was moved off it).
- Public endpoint: `https://f6t7m2t1errly1-8888.proxy.runpod.net`
  (`/health` open, `/synth` requires `X-Weaver-Key`).

## Wiring
`weaverv3.com/tts/*` → Caddy on the AWS box → `reverse_proxy` to the RunPod
proxy URL (Host header overridden). The key is checked at **both** hops (Caddy
`{$WEAVER_LLM_KEY}` and the pod's `WEAVER_TTS_KEY`) since the proxy URL is public.
See `deploy/Caddyfile` `handle_path /tts/*`.

## Operating it
- **(Re)start after a pod stop/restart:** `ssh` into the pod and run
  `/root/tts/start.sh` (respawn-wrapped: frees 8888, restarts XTTS, auto-restarts
  on crash). XTTS takes ~40–60s to load + clone the voice on boot.
- **This is a manually-provisioned pod, not IaC** — a fresh pod needs: copy
  `weaver_voice_ref.wav` + `tts_server_gpu.py` to `/root/tts/`, `pip install`
  the Coqui `TTS` stack (transformers 4.57.x works), then `start.sh`.
- **Fallback:** the AWS box still runs CPU OpenVoice on `:8092`. If the pod is
  down, `/tts` 502s and the page falls back to the browser voice on its own; or
  point Caddy's `/tts` `reverse_proxy` back to `127.0.0.1:8092` for the CPU clone.
- **Cost:** the 4090 pod bills while running — stop it when not in use.

## Durability — her voice lives on AWS S3, not RunPod

RunPod's `/workspace` is a network volume in *their* Europe datacenter and its
container disk (`/root`) is wiped on **terminate**. So the durable copy of her
voice is kept on **AWS S3** (`s3://weaver-avatar-404870839825/voice/`) — survives
RunPod entirely and is consistent with the rest of the stack being on AWS.

What's in S3: `weaver_voice_ref.wav` (her irreplaceable voice sample),
`tts_server_gpu.py`, `start.sh`, and the pre-synthesized `cache/`. **Not** in S3:
the `WEAVER_TTS_KEY` secret (only in the pod's untracked `/workspace/tts/.env`),
and the 1.8 GB XTTS model (a public Coqui download — re-fetched on first run;
`restore_from_s3.sh` pip-installs the stack which triggers it).

**Restore onto a fresh/terminated pod** (run from a trusted machine with the
swarm-admin profile + pod SSH key — no AWS creds ever touch the rented pod):
```bash
WEAVER_TTS_KEY='<key>' ./restore_from_s3.sh root@<pod-ip> <ssh-port>
```
It syncs S3 → pod `/workspace/tts`, installs the XTTS stack, seeds `.env`, and
starts the server. To back up new cache lines, re-run the small
`aws s3 sync` from a trusted machine (the pod can't reach S3 by design).
