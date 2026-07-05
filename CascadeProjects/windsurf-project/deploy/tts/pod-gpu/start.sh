#!/bin/bash
# Weaver GPU voice launcher (RunPod RTX 4090). Re-run after a pod restart.
cd /root/tts
# WEAVER_TTS_KEY gates /synth. Provide it via the environment or an UNTRACKED
# /root/tts/.env — never hardcode it here (this file is in a public repo).
[ -f /root/tts/.env ] && set -a && . /root/tts/.env && set +a
: "${WEAVER_TTS_KEY:?set WEAVER_TTS_KEY (env or /root/tts/.env) — it gates /synth}"
export TTS_HOST=0.0.0.0 TTS_PORT=8888
pkill -f jupyter-lab 2>/dev/null || true          # free the exposed 8888
pkill -f tts_server_gpu 2>/dev/null || true; sleep 2
# respawn loop: if XTTS crashes, restart it
nohup bash -c 'while true; do python3 -u tts_server_gpu.py >> server.log 2>&1; echo "[start] exited, respawning $(date)" >> server.log; sleep 3; done' >/dev/null 2>&1 &
echo "weaver-tts(gpu) supervised pid $!"
