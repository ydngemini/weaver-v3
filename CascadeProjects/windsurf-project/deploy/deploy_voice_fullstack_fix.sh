#!/usr/bin/env bash
# Deploy the 2026-07-09 voice + full-stack fix to the weaverv3.com box.
#
# What it does, in order:
#   1. Copies the fixed embodiment.html / headless.html to the Caddy roots.
#   2. Copies bedrock_brain_api.py (n8n MoE routing + Bedrock fallback) and
#      restarts weaver-brain.
#   3. Installs the updated weaver-tts.service (trained OpenVoice clone
#      primary, Polly Ruth fallback) and restarts weaver-tts.
#   4. Ensures the n8n Mantle credential exists, imports + publishes
#      n8n_weaver_v5.json, repairs the production webhook row, restarts n8n,
#      and smoke-tests POST /webhook/weaver-input.
#   5. Verifies: brain /health, a weaver-one chat turn, last_cortex_route,
#      and a TTS synth (expect audio/wav = OpenVoice; first synth is slow).
#
# Usage (from the repo, any directory):
#   bash "CascadeProjects/windsurf-project/deploy/deploy_voice_fullstack_fix.sh"
# Env overrides: BOX=ubuntu@34.200.158.166 SSH_KEY=~/.ssh/id_ed25519
set -euo pipefail

BOX="${BOX:-ubuntu@34.200.158.166}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # .../deploy
PROJ="$(dirname "$HERE")"                                 # .../windsurf-project
AVATAR="$(cd "$PROJ/../.." && pwd)/avatar"                # repo-root/avatar
SSH=(ssh -i "$SSH_KEY" -o ConnectTimeout=10 "$BOX")
SCP=(scp -i "$SSH_KEY" -o ConnectTimeout=10)

echo "── staging files to $BOX:/tmp ──"
"${SCP[@]}" \
  "$AVATAR/embodiment.html" \
  "$AVATAR/headless.html" \
  "$PROJ/bedrock_brain_api.py" \
  "$PROJ/n8n_weaver_v5.json" \
  "$HERE/tts/weaver-tts.service" \
  "$HERE/repair_n8n_weaver_webhook.py" \
  "$BOX:/tmp/"

"${SSH[@]}" bash -s <<'REMOTE'
set -euo pipefail

echo "── 1. static pages ──"
for pair in "embodiment.html:/var/www/weaver" "headless.html:/var/www/weaver-headless"; do
  src="/tmp/${pair%%:*}"; root="${pair##*:}"
  # Caddy uses plain file_server: the page is index.html unless a named copy exists.
  if sudo test -f "$root/${pair%%:*}"; then dest="$root/${pair%%:*}"; else dest="$root/index.html"; fi
  sudo cp "$src" "$dest"
  echo "  $src -> $dest"
done

echo "── 2. brain api ──"
cp /tmp/bedrock_brain_api.py /home/ubuntu/weaver/CascadeProjects/windsurf-project/bedrock_brain_api.py
sudo systemctl restart weaver-brain
sleep 4
curl -fsS http://127.0.0.1:8093/health >/dev/null && echo "  weaver-brain healthy"

echo "── 3. trained voice (openvoice primary, polly fallback) ──"
sudo cp /tmp/weaver-tts.service /etc/systemd/system/weaver-tts.service
sudo systemctl daemon-reload
sudo systemctl restart weaver-tts
echo "  weaver-tts restarted (model load takes ~30-60s on first synth)"

echo "── 4. n8n MoE pipeline ──"
# 4a. Mantle credential (id azure-openai-header, referenced by all lobe nodes).
DB=/var/lib/docker/volumes/n8n_data/_data/database.sqlite
HAVE_CRED=$(sudo python3 - <<'PY'
import sqlite3
db = sqlite3.connect("/var/lib/docker/volumes/n8n_data/_data/database.sqlite")
try:
    n = db.execute("select count(*) from credentials_entity where id='azure-openai-header'").fetchone()[0]
except Exception:
    n = 0
print(n)
PY
)
if [ "$HAVE_CRED" = "0" ]; then
  echo "  credential missing — importing from box .env MANTLE_API_KEY"
  MANTLE_KEY=$(grep -oP '^MANTLE_API_KEY=\K.*' /home/ubuntu/weaver/CascadeProjects/windsurf-project/.env | tr -d '"' | tr -d "'")
  python3 - "$MANTLE_KEY" <<'PY'
import json, sys
cred = [{
    "id": "azure-openai-header",
    "name": "Azure OpenAI Header Auth",
    "type": "httpHeaderAuth",
    "data": {"name": "Authorization", "value": f"Bearer {sys.argv[1]}"},
}]
open("/tmp/n8n_cred.json", "w").write(json.dumps(cred))
PY
  docker cp /tmp/n8n_cred.json n8n:/tmp/n8n_cred.json
  docker exec -u node n8n n8n import:credentials --input=/tmp/n8n_cred.json
  rm -f /tmp/n8n_cred.json
else
  echo "  credential azure-openai-header present"
fi
# 4b. import + publish the v5 workflow, then repair the production webhook row.
docker cp /tmp/n8n_weaver_v5.json n8n:/tmp/wf.json
docker exec -u node n8n n8n import:workflow --input=/tmp/wf.json
docker exec -u node n8n n8n publish:workflow --id=weaverv5soulbind || \
  docker exec -u node n8n n8n update:workflow --id=weaverv5soulbind --active=true
sudo python3 /tmp/repair_n8n_weaver_webhook.py --no-container-restart
sudo systemctl restart n8n
echo "  waiting for n8n to come back"
for i in $(seq 1 30); do
  sleep 2
  if curl -fsS http://127.0.0.1:5678/healthz >/dev/null 2>&1; then break; fi
done
echo "── 4c. webhook smoke test ──"
curl -fsS -X POST http://127.0.0.1:5678/webhook/weaver-input \
  -H 'Content-Type: application/json' \
  -d '{"text":"Deploy smoke test: say one short sentence."}' \
  | head -c 600
echo

echo "── 5. end-to-end verification ──"
KEY=$(sudo grep -oP '^WEAVER_LLM_KEY=\K.*' /etc/default/caddy)
echo "  5a. weaver-one chat turn (full stack expected):"
curl -fsS -m 60 -X POST http://127.0.0.1:8093/v1/chat/completions \
  -H "X-Weaver-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"weaver-one","max_tokens":120,"messages":[{"role":"user","content":"One sentence: how are you routed right now?"}]}' \
  | head -c 800
echo
echo "  5b. cortex route used:"
curl -fsS "http://127.0.0.1:8093/state" -H "X-Weaver-Key: $KEY" | python3 -c 'import json,sys; s=json.load(sys.stdin); print("   ", s.get("last_cortex_route"), "| n8n err:", s.get("last_n8n_error",""))'
echo "  5c. TTS synth (expect audio/wav from OpenVoice; first call loads the model):"
curl -fsS -m 180 -X POST http://127.0.0.1:8092/synth \
  -H 'Content-Type: application/json' -d '{"text":"Voice check, this is my real voice."}' \
  -o /tmp/synthcheck.bin -w '    content-type: %{content_type}  time: %{time_total}s\n' || echo "    (direct :8092 synth failed — check journalctl -u weaver-tts)"
file /tmp/synthcheck.bin 2>/dev/null | sed 's/^/    /' || true
echo "── deploy complete ──"
REMOTE

echo "── public md5 check (local vs live) ──"
for pair in "embodiment.html:https://weaverv3.com/" "headless.html:https://headless.weaverv3.com/"; do
  f="${pair%%:*}"; url="${pair#*:}"
  local_md5=$(md5sum "$AVATAR/$f" | cut -d' ' -f1)
  live_md5=$(curl -fsS "$url" | md5sum | cut -d' ' -f1)
  [ "$local_md5" = "$live_md5" ] && echo "  $f == live ($local_md5)" || echo "  MISMATCH $f local=$local_md5 live=$live_md5"
done
echo "All done."
