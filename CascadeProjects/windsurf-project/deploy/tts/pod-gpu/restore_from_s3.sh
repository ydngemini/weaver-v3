#!/bin/bash
# Restore her voice onto a FRESH RunPod pod from AWS S3 (the durable store).
#
# Design: the durable copy of her voice lives in S3 (s3://.../voice/) — it
# survives RunPod pod TERMINATION and RunPod's storage entirely. This script
# runs from a TRUSTED machine (the one holding the swarm-admin AWS profile +
# the pod SSH key) and pushes S3 -> pod. **No AWS credentials ever touch the
# rented pod, and the X-Weaver-Key secret is never stored in S3.**
#
#   usage:  restore_from_s3.sh <pod-ssh-host> <pod-ssh-port>
#   e.g.    restore_from_s3.sh root@213.173.110.228 11683
#   env:    SSH_KEY (default ~/.ssh/id_ed25519), WEAVER_TTS_KEY (to seed the pod .env)
set -euo pipefail
POD_HOST="${1:?pod ssh host, e.g. root@1.2.3.4}"
POD_PORT="${2:?pod ssh port, e.g. 11683}"
KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
BUCKET="s3://weaver-avatar-404870839825/voice"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

echo "[1/4] pull her voice from S3 ($BUCKET)…"
aws s3 sync "$BUCKET" "$TMP" --profile swarm-admin --region us-east-1

echo "[2/4] install the XTTS stack on the pod (model auto-downloads on first run)…"
ssh -i "$KEY" -p "$POD_PORT" "$POD_HOST" \
  'pip install -q coqui-tts "transformers==4.57.*" soundfile fastapi uvicorn 2>&1 | tail -1 || true'

echo "[3/4] copy voice data -> pod /workspace/tts…"
ssh -i "$KEY" -p "$POD_PORT" "$POD_HOST" 'mkdir -p /workspace/tts/cache'
scp -i "$KEY" -P "$POD_PORT" -r "$TMP/." "$POD_HOST:/workspace/tts/"
# seed the key on the pod's UNTRACKED .env (never in S3 or git)
ssh -i "$KEY" -p "$POD_PORT" "$POD_HOST" \
  "printf 'WEAVER_TTS_KEY=%s\n' '${WEAVER_TTS_KEY:-CHANGE_ME}' > /workspace/tts/.env && chmod 600 /workspace/tts/.env"

echo "[4/4] start…"
ssh -i "$KEY" -p "$POD_PORT" "$POD_HOST" 'chmod +x /workspace/tts/start.sh && /workspace/tts/start.sh'
echo "done. If you didn't pass WEAVER_TTS_KEY, set it in /workspace/tts/.env on the pod and re-run start.sh."
