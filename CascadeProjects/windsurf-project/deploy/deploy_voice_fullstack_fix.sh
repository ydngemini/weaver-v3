#!/usr/bin/env bash
# Deploy and verify Weaver's full runtime, n8n MoE route, and trained voice.
set -Eeuo pipefail

BOX="${BOX:-ubuntu@34.200.158.166}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
ROOT="$(cd "$PROJ/../.." && pwd)"
DEPLOY_SHA="$(git -C "$ROOT" rev-parse HEAD)"
STAGE="/tmp/weaver-fullstack-deploy-${DEPLOY_SHA}-$$"
ARCHIVE="$(mktemp "/tmp/weaver-${DEPLOY_SHA}.XXXXXX.tar.gz")"
EIC_PUSH="${EIC_PUSH:-1}"
EIC_INSTANCE_ID="${EIC_INSTANCE_ID:-i-01e15a540b9efb7a0}"
EIC_PROFILE="${EIC_PROFILE:-swarm-admin}"
EIC_REGION="${EIC_REGION:-us-east-1}"
EIC_OS_USER="${EIC_OS_USER:-ubuntu}"
SSH=(ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$BOX")
SCP=(scp -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
REMOTE_STAGE_CREATED=0

push_eic_key() {
  [ "$EIC_PUSH" = "1" ] || return 0
  aws ec2-instance-connect send-ssh-public-key \
    --instance-id "$EIC_INSTANCE_ID" \
    --instance-os-user "$EIC_OS_USER" \
    --ssh-public-key "file://$SSH_KEY.pub" \
    --profile "$EIC_PROFILE" \
    --region "$EIC_REGION" \
    --output text --query Success
}

cleanup_local() {
  rc=$?
  trap - EXIT
  rm -f "$ARCHIVE"
  if (( rc != 0 && REMOTE_STAGE_CREATED )); then
    "${SSH[@]}" "rm -rf '$STAGE'" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap cleanup_local EXIT

if ! git -C "$ROOT" diff --quiet || ! git -C "$ROOT" diff --cached --quiet; then
  echo "Refusing to deploy uncommitted tracked changes; commit the reviewed tree first." >&2
  exit 1
fi

node "$PROJ/scripts/validate_n8n_workflow.mjs" "$PROJ/n8n_weaver_v5.json"
PYTHONPYCACHEPREFIX=/tmp/weaver-pycache "$PROJ/venv/bin/python3" -m py_compile \
  "$PROJ/weaver_neural_fabric.py" "$PROJ/weaver_cognition_mesh.py" "$PROJ/bedrock_brain_api.py"

echo "── stage deploy $DEPLOY_SHA on $BOX ──"
push_eic_key
git -C "$ROOT" archive --format=tar.gz --output "$ARCHIVE" "$DEPLOY_SHA"
remote_available=$("${SSH[@]}" "df --output=avail -B1 /tmp | tail -1")
(( remote_available >= 268435456 )) || { echo "remote /tmp has insufficient free space" >&2; exit 1; }
"${SSH[@]}" "mkdir -m 700 '$STAGE'"
REMOTE_STAGE_CREATED=1
"${SCP[@]}" "$ARCHIVE" "$BOX:$STAGE/repo.tar.gz"
# EIC keys live for 60 seconds; refresh after the potentially slow archive upload.
push_eic_key

"${SSH[@]}" "DEPLOY_SHA='$DEPLOY_SHA' STAGE='$STAGE' bash -s" <<'REMOTE'
set -Eeuo pipefail

APP=/home/ubuntu/weaver/CascadeProjects/windsurf-project
DEPLOY_ROOT=/home/ubuntu/weaver
DB=/var/lib/docker/volumes/n8n_data/_data/database.sqlite
N8N_STOPPED=0
DB_ROLLBACK=""
BACKUP_READY=0
DEGRADED=0
RELEASE=$(mktemp -d /tmp/weaver-release.XXXXXX)
BACKUP=$(mktemp -d /tmp/weaver-rollback.XXXXXX)

exec 9>/tmp/weaver-fullstack-deploy.lock
flock -n 9 || { rm -rf "$STAGE" "$RELEASE" "$BACKUP"; echo "another Weaver deployment is running" >&2; exit 1; }

cleanup() {
  rc=$?
  trap - EXIT
  set +e
  rm -f /tmp/n8n_cred.json /tmp/webhook.json /tmp/brain.json /tmp/synthcheck.bin /tmp/tts.headers \
    /tmp/public-synthcheck.bin /tmp/public-tts.headers /tmp/cognition-capsule.json \
    /tmp/cognition-evaluate.json
  sudo rm -f /tmp/existing-creds.json
  sudo docker exec -u root n8n rm -f /tmp/n8n_cred.json /tmp/existing-creds.json /tmp/wf.json >/dev/null 2>&1 || true
  if (( rc != 0 )) && [ -n "$DB_ROLLBACK" ] && sudo test -f "$DB_ROLLBACK"; then
    echo "── restoring pre-import n8n database backup ──"
    if (( ! N8N_STOPPED )); then
      if sudo systemctl stop n8n; then
        N8N_STOPPED=1
      else
        echo "n8n stop failed; preserving backup for manual recovery: $DB_ROLLBACK" >&2
        rc=1
      fi
    fi
    container_running=$(sudo docker inspect -f '{{.State.Running}}' n8n 2>/dev/null || echo false)
    if (( N8N_STOPPED )) && ! systemctl is-active --quiet n8n && [ "$container_running" != "true" ]; then
      sudo rm -f "$DB-wal" "$DB-shm"
      sudo cp -a "$DB_ROLLBACK" "$DB" || rc=1
      sudo python3 -c 'import sqlite3; db=sqlite3.connect("file:/var/lib/docker/volumes/n8n_data/_data/database.sqlite?mode=ro", uri=True); assert db.execute("pragma integrity_check").fetchone()[0] == "ok"' || rc=1
    else
      echo "refusing to overwrite live n8n database; backup preserved: $DB_ROLLBACK" >&2
      rc=1
    fi
  fi
  if (( rc != 0 && BACKUP_READY )); then
    echo "── rolling back tracked tree, units, and web roots ──"
    cp -a "$BACKUP/tree/." "$DEPLOY_ROOT/" || true
    if [ -f "$BACKUP/new-files" ]; then
      while IFS= read -r -d '' rel; do
        rm -f "$DEPLOY_ROOT/$rel"
      done < "$BACKUP/new-files"
    fi
    sudo tar -xzf "$BACKUP/units.tgz" -C /etc/systemd/system || true
    sudo install -m 0755 "$APP/deploy/repair_n8n_weaver_webhook.py" /usr/local/sbin/repair_n8n_weaver_webhook.py || true
    sudo rm -rf /var/www/weaver /var/www/weaver-headless
    sudo tar -xzf "$BACKUP/web.tgz" -C /var/www || true
    sudo systemctl daemon-reload || true
  fi
  if (( rc != 0 && BACKUP_READY )); then
    sudo systemctl restart n8n weaver-brain weaver-tts weaver || true
    N8N_STOPPED=0
  elif (( N8N_STOPPED )); then
    echo "cleanup: restarting n8n"
    sudo systemctl start n8n || rc=1
    N8N_STOPPED=0
  fi
  rm -rf "$STAGE" "$RELEASE"
  sudo rm -rf "$BACKUP"
  if (( rc != 0 )); then
    echo "── failure diagnostics ──"
    systemctl is-active weaver-brain weaver-tts weaver-llm weaver n8n caddy || true
    sudo journalctl -u n8n -u weaver -u weaver-brain -n 60 --no-pager || true
  fi
  exit "$rc"
}
trap cleanup EXIT

wait_http() {
  name=$1
  url=$2
  attempts=${3:-60}
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      echo "  $name healthy"
      return 0
    fi
    sleep 2
  done
  echo "$name failed health check: $url" >&2
  return 1
}

wait_n8n_workflow_ready() {
  attempts=${1:-60}
  marker='Activated workflow "Weaver Nervous System v6 (parallel cognition mesh)" (ID: weaverv5soulbind)'
  for _ in $(seq 1 "$attempts"); do
    if sudo docker logs n8n 2>&1 | grep -Fq "$marker"; then
      echo "  n8n workflow activation complete"
      return 0
    fi
    sleep 2
  done
  echo "n8n became healthy but did not activate workflow weaverv5soulbind" >&2
  sudo docker logs --tail 80 n8n >&2 || true
  return 1
}

assert_n8n_offline() {
  if systemctl is-active --quiet n8n; then
    echo "refusing offline database access while n8n.service is active" >&2
    return 1
  fi
  sudo docker info >/dev/null
  state=$(sudo docker inspect -f '{{.State.Running}}' n8n 2>/dev/null || printf absent)
  [ "$state" != "true" ] || {
    echo "refusing offline database access while n8n container is running" >&2
    return 1
  }
}

echo "── deploy metadata ──"
echo "  sha: $DEPLOY_SHA"
echo "  host: $(hostname)"
echo "  time: $(date -Is)"
df -h / "$APP"
sudo test -f "$DB" || { echo "expected n8n database is missing: $DB" >&2; exit 1; }
available=$(sudo df --output=avail -B1 "$(dirname "$DB")" | tail -1)
db_size=$(sudo stat -c %s "$DB")
required=$((db_size * 2 + 536870912))
(( available >= required )) || {
  echo "insufficient disk for verified n8n backup: available=$available required=$required" >&2
  exit 1
}
sudo python3 - <<'PY'
import sqlite3
db=sqlite3.connect("file:/var/lib/docker/volumes/n8n_data/_data/database.sqlite?mode=ro", uri=True)
assert db.execute("pragma integrity_check").fetchone()[0] == "ok"
tables={row[0] for row in db.execute("select name from sqlite_master where type='table'")}
required={"workflow_entity","credentials_entity","webhook_entity"}
assert required <= tables, f"missing n8n tables: {sorted(required-tables)}"
print("  n8n database integrity and schema: ok")
PY

echo "── 1. install and verify complete tracked commit ──"
tar -xzf "$STAGE/repo.tar.gz" -C "$RELEASE"
mkdir -p "$BACKUP/tree"
while IFS= read -r -d '' source; do
  rel=${source#"$RELEASE/"}
  target="$DEPLOY_ROOT/$rel"
  if [ -e "$target" ] || [ -L "$target" ]; then
    mkdir -p "$BACKUP/tree/$(dirname "$rel")"
    cp -a "$target" "$BACKUP/tree/$rel"
  else
    printf '%s\0' "$rel" >> "$BACKUP/new-files"
  fi
done < <(find "$RELEASE" \( -type f -o -type l \) -print0)
sudo tar -czf "$BACKUP/units.tgz" -C /etc/systemd/system \
  n8n.service weaver.service weaver-brain.service weaver-tts.service
sudo tar -czf "$BACKUP/web.tgz" -C /var/www weaver weaver-headless
BACKUP_READY=1

cp -a "$RELEASE/." "$DEPLOY_ROOT/"
while IFS= read -r -d '' source; do
  rel=${source#"$RELEASE/"}
  cmp -s "$source" "$DEPLOY_ROOT/$rel" || { echo "tracked-file mismatch: $rel" >&2; exit 1; }
done < <(find "$RELEASE" -type f -print0)
echo "  verified $(find "$RELEASE" -type f | wc -l) tracked files from $DEPLOY_SHA"

node "$APP/scripts/validate_n8n_workflow.mjs" "$APP/n8n_weaver_v5.json"
N8N_IMAGE="docker.n8n.io/n8nio/n8n:2.25.7@sha256:761374d4eb841b0a22771d6bd68f0e8d827b4979ae4e490045517b13fc1259dd"
sudo docker pull --quiet "$N8N_IMAGE" >/dev/null
# Pulling tag@digest guarantees content identity but does not necessarily create
# a mutable tag alias. Inspect the same immutable reference that was pulled.
sudo docker image inspect "$N8N_IMAGE" \
  --format '  verified n8n image: {{index .RepoDigests 0}}'

echo "  ensuring supervised bridge dependencies"
if ! "$APP/venv/bin/python3" -c 'import langchain_openai, discord, twilio' >/dev/null 2>&1; then
  "$APP/venv/bin/python3" -m pip install --disable-pip-version-check \
    --no-cache-dir \
    --requirement "$APP/requirements-bridges.txt"
fi
"$APP/venv/bin/python3" -c 'import langchain_openai, discord, twilio; print("  bridge imports: ok")'

sudo install -m 0644 "$APP/deploy/n8n.service" /etc/systemd/system/n8n.service
sudo install -m 0644 "$APP/deploy/weaver.service" /etc/systemd/system/weaver.service
sudo install -m 0644 "$APP/deploy/weaver-brain.service" /etc/systemd/system/weaver-brain.service
sudo install -m 0644 "$APP/deploy/tts/weaver-tts.service" /etc/systemd/system/weaver-tts.service
sudo install -m 0755 "$APP/deploy/repair_n8n_weaver_webhook.py" /usr/local/sbin/repair_n8n_weaver_webhook.py
sudo install -m 0644 "$DEPLOY_ROOT/avatar/embodiment.html" /var/www/weaver/index.html
sudo install -m 0644 "$DEPLOY_ROOT/avatar/embodiment.html" /var/www/weaver/embodiment.html
sudo install -m 0644 "$DEPLOY_ROOT/avatar/headless.html" /var/www/weaver-headless/index.html
sudo install -m 0644 "$DEPLOY_ROOT/avatar/headless.html" /var/www/weaver-headless/headless.html
for asset in \
  weaver_avatar_dress.glb \
  weaver_apartment.glb \
  weaver_avatar_dress_hifi.glb \
  textures/skin_normal_hifi.png \
  textures/skin_roughness_hifi.png \
  textures/skin_specular_hifi.png; do
  test -s "$DEPLOY_ROOT/avatar/$asset" || { echo "missing required visual asset: $asset" >&2; exit 1; }
done
sudo install -m 0644 "$DEPLOY_ROOT/avatar/weaver_avatar_dress.glb" /var/www/weaver/weaver_avatar_dress.glb
sudo install -m 0644 "$DEPLOY_ROOT/avatar/weaver_apartment.glb" /var/www/weaver/weaver_apartment.glb
sudo install -m 0644 "$DEPLOY_ROOT/avatar/weaver_avatar_dress_hifi.glb" /var/www/weaver/weaver_avatar_dress_hifi.glb
sudo install -d -m 0755 /var/www/weaver/textures
for map in skin_normal_hifi.png skin_roughness_hifi.png skin_specular_hifi.png; do
  sudo install -m 0644 "$DEPLOY_ROOT/avatar/textures/$map" "/var/www/weaver/textures/$map"
done
for root in /var/www/weaver /var/www/weaver-headless; do
  sudo rm -rf "$root/vendor"
  sudo cp -a "$DEPLOY_ROOT/avatar/vendor" "$root/vendor"
  sudo install -m 0644 "$DEPLOY_ROOT/avatar/weaver-logo.svg" "$root/weaver-logo.svg"
done
sudo systemctl daemon-reload

echo "── 2. verified pre-migration n8n backup ──"
sudo systemctl stop n8n
N8N_STOPPED=1
assert_n8n_offline
backup_output=$(sudo python3 /usr/local/sbin/repair_n8n_weaver_webhook.py --offline --backup-only)
echo "$backup_output"
DB_ROLLBACK=$(printf '%s\n' "$backup_output" | sed -n 's/^backup=//p' | tail -1)
test -n "$DB_ROLLBACK" && sudo test -f "$DB_ROLLBACK"

echo "── 3. restart runtime services ──"
sudo systemctl restart weaver-brain
sudo systemctl restart weaver-tts
sudo systemctl restart weaver
sudo systemctl start n8n
N8N_STOPPED=0
wait_http "brain" http://127.0.0.1:8093/health 45
wait_http "Nexus Bus" http://127.0.0.1:9998/health 60
wait_http "live dashboard" http://127.0.0.1:9990/health 60
wait_http "codebase API" http://127.0.0.1:8091/health 60
wait_http "quantum API" http://127.0.0.1:9997/health 60
wait_http "Akashic Hub" http://127.0.0.1:9995/health 60
wait_http "health dashboard" http://127.0.0.1:9996/health 60
wait_http "phone bridge" http://127.0.0.1:8765/health 60
wait_http "Obsidian bridge" http://127.0.0.1:5679/health 60
wait_http "n8n" http://127.0.0.1:5678/healthz 60
sudo docker inspect n8n | python3 -c '
import json,sys
data=json.load(sys.stdin)[0]
host=data.get("HostConfig", {})
env=set(data.get("Config", {}).get("Env", []))
assert host.get("ReadonlyRootfs") is True, host
assert "ALL" in (host.get("CapDrop") or []), host.get("CapDrop")
assert "no-new-privileges:true" in (host.get("SecurityOpt") or []), host.get("SecurityOpt")
assert "N8N_BLOCK_ENV_ACCESS_IN_NODE=true" in env, env
assert "N8N_RUNNERS_ENABLED=true" in env, env
assert data.get("Config", {}).get("Image") == "docker.n8n.io/n8nio/n8n:2.25.7@sha256:761374d4eb841b0a22771d6bd68f0e8d827b4979ae4e490045517b13fc1259dd", data.get("Config", {}).get("Image")
print("  n8n container: pinned, read-only, capability-dropped, sandboxed")
'
wait_http "local llama API" http://127.0.0.1:8090/v1/models 60
curl -fsS -m 90 -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"weaver-local","max_tokens":8,"messages":[{"role":"user","content":"Reply: connected"}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("choices", [{}])[0].get("message", {}).get("content", "").strip(), d; print("  local llama inference: connected")'

curl -fsS http://127.0.0.1:9995/runtime/tasks | python3 -c '
import json,sys
data=json.load(sys.stdin)
tasks=data.get("tasks", {})
required={"nexus_bus","quantum_soul","pineal_gate","lora_server","qwen3b_server","quantum_api","health_dashboard","live_dashboard","codebase_api","akashic_hub_api","phone_bridge","obsidian_bridge","proactive_pulse","dream_state"}
bad={name:tasks.get(name) for name in required if not tasks.get(name, {}).get("running")}
assert not bad, f"missing/stopped supervised tasks: {bad}; live={sorted(tasks)}"
print("  supervised runtime tasks:", len(required), "required running")
'
curl -fsS http://127.0.0.1:8765/health | python3 -c '
import json,sys
data=json.load(sys.stdin)
assert data.get("status") == "ok", data
assert data.get("n8n_route") == "http://127.0.0.1:5678/webhook/weaver-input", data
print("  phone bridge: online; Twilio configured:", bool(data.get("twilio_configured")))
'
DISCORD_CONFIGURED=$("$APP/venv/bin/python3" -c 'from dotenv import dotenv_values; print("1" if (dotenv_values("/home/ubuntu/weaver/CascadeProjects/windsurf-project/.env").get("DISCORD_BOT_TOKEN") or "").strip() else "0")')
if [ "$DISCORD_CONFIGURED" = "1" ]; then
  wait_http "Discord bridge" http://127.0.0.1:8770/health 60
  DISCORD_VOICE_CONFIGURED=$("$APP/venv/bin/python3" -c 'from dotenv import dotenv_values; print("1" if (dotenv_values("/home/ubuntu/weaver/CascadeProjects/windsurf-project/.env").get("DISCORD_VOICE_CHANNEL_ID") or "").strip() else "0")')
  curl -fsS http://127.0.0.1:8770/health | DISCORD_VOICE_CONFIGURED="$DISCORD_VOICE_CONFIGURED" python3 -c '
import json,sys
import os
data=json.load(sys.stdin)
assert data.get("bot_ready") and data.get("nexus_connected"), data
if os.environ.get("DISCORD_VOICE_CONFIGURED") == "1":
    assert data.get("voice_connected"), data
print("  Discord bridge: bot ready and Nexus connected")
'
fi

echo "── 4. refresh credential, import, and activate workflow ──"
umask 077
"$APP/venv/bin/python3" - <<'PY'
import json
from dotenv import dotenv_values

values = dotenv_values("/home/ubuntu/weaver/CascadeProjects/windsurf-project/.env")
key = (values.get("MANTLE_API_KEY") or "").strip()
if not key:
    raise SystemExit(0)
payload = [{
    "id": "azure-openai-header",
    "name": "Azure OpenAI Header Auth",
    "type": "httpHeaderAuth",
    "data": {"name": "Authorization", "value": f"Bearer {key}"},
}]
with open("/tmp/n8n_cred.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
PY
if [ -f /tmp/n8n_cred.json ]; then
  sudo docker exec -i -u node n8n sh -c 'umask 077; cat > /tmp/n8n_cred.json' < /tmp/n8n_cred.json
  sudo docker exec -u node n8n n8n import:credentials --input=/tmp/n8n_cred.json
  sudo docker exec -u node n8n rm -f /tmp/n8n_cred.json
  rm -f /tmp/n8n_cred.json
  echo "  Mantle credential refreshed from deployment environment"
else
  sudo python3 - <<'PY'
import sqlite3
db=sqlite3.connect("file:/var/lib/docker/volumes/n8n_data/_data/database.sqlite?mode=ro", uri=True)
row=db.execute('SELECT id, type FROM credentials_entity WHERE id=?', ("azure-openai-header",)).fetchone()
assert row == ("azure-openai-header", "httpHeaderAuth"), f"existing Mantle credential metadata is missing or wrong: {row}"
print("  existing encrypted Mantle credential preserved")
PY
fi
sudo docker exec -i -u node n8n sh -c 'umask 077; cat > /tmp/wf.json' < "$APP/n8n_weaver_v5.json"
sudo docker exec -u node n8n n8n import:workflow --input=/tmp/wf.json
sudo docker exec -u node n8n rm -f /tmp/wf.json
sudo docker exec -u node n8n n8n publish:workflow --id=weaverv5soulbind || \
  sudo docker exec -u node n8n n8n update:workflow --id=weaverv5soulbind --active=true
sudo systemctl restart n8n
wait_http "n8n after workflow import" http://127.0.0.1:5678/healthz 60

echo "── 5. wait for local expert models ──"
wait_http "LoRA Soul Voice" http://127.0.0.1:8899/health 120
wait_http "Qwen3B router" http://127.0.0.1:8898/health 120
sudo docker exec -i n8n node - <<'NODE'
const targets = [
  ['codebase', 'http://host.docker.internal:8091/health'],
  ['lora', 'http://host.docker.internal:8899/health'],
  ['qwen3b', 'http://host.docker.internal:8898/health'],
];
(async () => {
  for (const [name, url] of targets) {
    const response = await fetch(url, {signal: AbortSignal.timeout(10000)});
    const body = await response.json();
    if (!response.ok || body.status !== 'ok') {
      throw new Error(`${name} unavailable: HTTP ${response.status} ${JSON.stringify(body)}`);
    }
    console.log(`  container -> ${name}: HTTP ${response.status} ${body.status}`);
  }
})().catch(error => { console.error(error); process.exit(1); });
NODE

echo "── 6. webhook attempt and conditional self-heal ──"
wait_n8n_workflow_ready 60
WEBHOOK_CODE=$(curl -sS -o /tmp/webhook.json -w '%{http_code}' -m 240 \
  -X POST http://127.0.0.1:5678/webhook/weaver-input \
  -H 'Content-Type: application/json' \
  -d '{"text":"Deploy connectivity test. Reply with one short sentence."}') || WEBHOOK_CODE=000
echo "  webhook attempt 1: HTTP $WEBHOOK_CODE"
if [ "$WEBHOOK_CODE" != "200" ]; then
  [ "$WEBHOOK_CODE" = "404" ] || { head -c 500 /tmp/webhook.json 2>/dev/null || true; exit 1; }
  echo "  canonical route missing; applying owner-safe offline repair"
  sudo systemctl stop n8n
  N8N_STOPPED=1
  assert_n8n_offline
  sudo python3 /usr/local/sbin/repair_n8n_weaver_webhook.py --offline --no-backup
  sudo systemctl start n8n
  wait_http "n8n after repair" http://127.0.0.1:5678/healthz 60
  wait_n8n_workflow_ready 60
  N8N_STOPPED=0
  WEBHOOK_CODE=$(curl -sS -o /tmp/webhook.json -w '%{http_code}' -m 240 \
    -X POST http://127.0.0.1:5678/webhook/weaver-input \
    -H 'Content-Type: application/json' \
    -d '{"text":"Deploy connectivity retest. Reply with one short sentence."}') || WEBHOOK_CODE=000
  echo "  webhook attempt 2: HTTP $WEBHOOK_CODE"
fi
[ "$WEBHOOK_CODE" = "200" ]
python3 - <<'PY'
import json
data=json.load(open("/tmp/webhook.json", encoding="utf-8"))
assert isinstance(data.get("manifested_response"), str) and data["manifested_response"].strip(), data
assert data.get("pipeline_version") == "v6-parallel-cognition", data
assert data.get("pipeline_architecture") == "parallel-fanout-barrier", data
assert data.get("expert_count") == 5, data
assert data.get("experts_completed") == 5, data
assert data.get("expert_parallel") is True, data
assert data.get("qwen3b_active") is True, data
assert not data.get("lora_error"), data.get("lora_error")
assert not data.get("qwen3b_error"), data.get("qwen3b_error")
assert isinstance(data.get("lora_latency_ms"), (int, float)) and data["lora_latency_ms"] > 0, data
assert isinstance(data.get("qwen3b_latency_ms"), (int, float)) and data["qwen3b_latency_ms"] > 0, data
assert data.get("cognition_mesh_active") is True, data
assert data.get("written_to_hub") is False, data
assert "original_input" not in data and "collapsed_response" not in data, data
print("  webhook: v6 parallel cognition; both local models executed; response=", data["manifested_response"][:180])
PY
sudo systemctl restart weaver-brain
wait_http "brain after n8n recovery" http://127.0.0.1:8093/health 45

echo "── 7. Nexus dashboard publish round trip ──"
"$APP/venv/bin/python3" - <<'PY'
import asyncio
import json
import urllib.request
import websockets

async def main():
    topic = "deploy.connectivity"
    async with websockets.connect("ws://127.0.0.1:9999") as sub:
        sync = json.loads(await sub.recv())
        assert sync["type"] == "sync", sync
        await sub.send(json.dumps({"action": "register", "lobe_id": "deploy_probe_sub"}))
        assert json.loads(await sub.recv())["type"] == "ack"
        await sub.send(json.dumps({"action": "subscribe", "topics": [topic]}))
        assert json.loads(await sub.recv())["type"] == "ack"

        def publish():
            data = json.dumps({"topic": topic, "payload": {"sha": "connectivity-ok"}}).encode()
            request = urllib.request.Request(
                "http://127.0.0.1:9990/api/nexus",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.load(response)

        result = await asyncio.to_thread(publish)
        assert result.get("ok") is True, result
        message = json.loads(await asyncio.wait_for(sub.recv(), timeout=10))
        assert message.get("type") == "broadcast", message
        assert message.get("topic") == topic, message
        assert message.get("from") == "dashboard_control", message
        print("  publish round trip:", message["from"], "->", message["topic"])

asyncio.run(main())
PY
curl -fsS http://127.0.0.1:9998/health | "$APP/venv/bin/python3" -c '
import json,sys
data=json.load(sys.stdin)
ids=set(data.get("lobe_ids", []))
required={"live_dashboard","pineal_gate","obsidian_bridge","dashboard_control"}
missing=required-ids
assert not missing, f"missing persistent Nexus lobes: {sorted(missing)}; live={sorted(ids)}"
print("  persistent Nexus lobes:", ", ".join(sorted(ids)))
'

echo "── 8. brain and public-route verification ──"
KEY=$(sudo sed -n 's/^WEAVER_LLM_KEY=//p' /etc/default/caddy | head -1)
KEY=${KEY%\"}; KEY=${KEY#\"}; KEY=${KEY%\'}; KEY=${KEY#\'}
test -n "$KEY"
curl -fsS -m 240 -X POST http://127.0.0.1:8093/v1/chat/completions \
  -H "X-Weaver-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"weaver-one","max_tokens":120,"messages":[{"role":"user","content":"One sentence: how are you routed right now?"}]}' \
  -o /tmp/brain.json
python3 - <<'PY'
import json
data=json.load(open("/tmp/brain.json", encoding="utf-8"))
text=data.get("choices", [{}])[0].get("message", {}).get("content", "")
assert text.strip(), data
print("  brain response:", text[:180])
PY
curl -fsS http://127.0.0.1:8093/state -H "X-Weaver-Key: $KEY" | python3 -c '
import json,sys
state=json.load(sys.stdin)
assert state.get("last_cortex_route") == "n8n-moe", state
assert not state.get("last_n8n_error"), state
print("  cortex route:", state["last_cortex_route"], "| n8n error: none")
'
curl -fsS http://127.0.0.1:8093/fabric/v1/state -H "X-Weaver-Key: $KEY" | python3 -c '
import json,sys
state=json.load(sys.stdin)
assert state.get("technology") == "weaver-neural-fabric", state
assert state.get("ledger", {}).get("valid") is True, state
accelerator=state.get("accelerator", {})
assert accelerator.get("realtime_reserved_units", 0) > 0, accelerator
print("  neural fabric:", state.get("status"), "| pressure:", accelerator.get("pressure"), "| ledger: valid")
'
curl -fsS -X POST http://127.0.0.1:8093/fabric/v1/intent/compile \
  -H "X-Weaver-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"goal":"Verify bounded lounge navigation","priority":"embodiment","ttl_ms":10000,"actions":[{"type":"navigate","zone":"lounge"},{"type":"interact","interaction":"reading_book"}]}' \
  | python3 -c '
import json,sys
data=json.load(sys.stdin)
capsule=data.get("capsule", {})
assert data.get("verified") is True, data
assert capsule.get("integrity", {}).get("algorithm") == "hmac-sha256", capsule
assert capsule.get("rollback") == ["cancel_interaction", "stop_locomotion"], capsule
print("  intent capsule:", capsule.get("capsule_id"), "| signed and bounded")
'

curl -fsS http://127.0.0.1:8093/cognition/v1/state -H "X-Weaver-Key: $KEY" | python3 -c '
import json,sys
data=json.load(sys.stdin)
assert data.get("technology") == "weaver-cognition-mesh", data
assert data.get("angles") == ["perception", "embodiment", "prediction", "compute", "memory", "resilience", "evolution"], data
assert data.get("evolution", {}).get("mode") == "advisory-only", data
print("  cognition mesh: seven angles; evolution is advisory-only")
'
curl -fsS -X POST http://127.0.0.1:8093/cognition/v1/observe \
  -H "X-Weaver-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"body":{"awake":true,"balance":0.92,"velocity_mps":0,"pose":{"leftElbow":0.1,"rightElbow":0.1},"confidence":0.95},"environment":{"zone":"center","obstacle_distance_m":5,"confidence":0.95,"objects":[{"id":"reading_book","zone":"lounge","distance_m":1.2,"visible":true,"confidence":0.95}]},"sensors":{"camera":{"confidence":0.9},"microphone":{"confidence":0.9}}}' \
  | python3 -c '
import json,sys
data=json.load(sys.stdin)
awareness=data.get("awareness", {})
assert awareness.get("body_revision", 0) > 0 and awareness.get("world_revision", 0) > 0, awareness
assert awareness.get("awareness_confidence", 0) > 0.5, awareness
assert data.get("fabric", {}).get("lane") == "embodiment", data
print("  sensor fusion: body/world revisions live; confidence:", awareness.get("awareness_confidence"))
'
curl -fsS -X POST http://127.0.0.1:8093/cognition/v1/route \
  -H "X-Weaver-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"task":"voice","deadline_ms":200,"quality_priority":0.4}' \
  | python3 -c '
import json,sys
data=json.load(sys.stdin)
assert data.get("primary", {}).get("alias") == "weaver-speed", data
assert data.get("advisory") is True, data
print("  inference governor: 200 ms voice route ->", data["primary"]["alias"])
'
curl -fsS -X POST http://127.0.0.1:8093/fabric/v1/intent/compile \
  -H "X-Weaver-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"goal":"Evaluate safe lounge reading","priority":"embodiment","ttl_ms":10000,"actions":[{"type":"navigate","zone":"lounge"},{"type":"interact","interaction":"reading_book"},{"type":"pose","values":{"leftElbow":0.3,"rightElbow":0.3,"leftKnee":0.1,"rightKnee":0.1}}]}' \
  -o /tmp/cognition-capsule.json
python3 - <<'PY'
import json
source=json.load(open("/tmp/cognition-capsule.json", encoding="utf-8"))
with open("/tmp/cognition-evaluate.json", "w", encoding="utf-8") as target:
    json.dump({"capsule": source["capsule"]}, target)
PY
curl -fsS -X POST http://127.0.0.1:8093/cognition/v1/intent/evaluate \
  -H "X-Weaver-Key: $KEY" -H 'Content-Type: application/json' \
  --data-binary @/tmp/cognition-evaluate.json \
  | python3 -c '
import json,sys
data=json.load(sys.stdin)
angles=data.get("angles", {})
assert data.get("capsule_verified") is True, data
assert data.get("decision") == "execute", data
assert angles.get("embodiment", {}).get("decision") == "approve", angles
assert angles.get("prediction", {}).get("predicted_zone") == "lounge", angles
assert len(angles) == 7, angles
print("  seven-angle intent: verified, reflex-approved, twin-predicted, execute")
'

curl --resolve weaverv3.com:443:127.0.0.1 -fsS -m 240 -X POST https://weaverv3.com/brain/v1/chat/completions \
  -H "X-Weaver-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"weaver-one","max_tokens":80,"messages":[{"role":"user","content":"Reply with the words public route connected."}]}' \
  | python3 -c '
import json,sys
data=json.load(sys.stdin)
text=data.get("choices", [{}])[0].get("message", {}).get("content", "")
assert text.strip(), data
print("  public brain route:", text[:160])
'

curl --resolve weaverv3.com:443:127.0.0.1 -fsS https://weaverv3.com/brain/realtime/voice/config -H "X-Weaver-Key: $KEY" | python3 -c '
import json,sys
data=json.load(sys.stdin)
assert data.get("mode") == "aws", data
print("  realtime voice config:", data.get("model"), data.get("voiceId"), data.get("mode"))
'
if WEAVER_TEST_KEY="$KEY" "$APP/venv/bin/python3" - <<'PY'
import asyncio
import base64
import json
import os
import websockets

async def main():
    key = os.environ["WEAVER_TEST_KEY"]
    encoded = base64.urlsafe_b64encode(key.encode()).decode().rstrip("=")
    protocols = ["weaver-realtime", f"weaver-key.{encoded}"]
    async with websockets.connect(
        "wss://weaverv3.com/brain/realtime/voice",
        host="127.0.0.1",
        server_hostname="weaverv3.com",
        subprotocols=protocols,
        open_timeout=20,
        close_timeout=10,
    ) as ws:
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
        assert first.get("status") == "connecting voice", first
        await ws.send(json.dumps({"type": "start"}))
        second = None
        for _ in range(20):
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
            if event.get("type") == "error":
                raise RuntimeError(event.get("error", event))
            if event.get("status") == "live voice ready":
                second = event
                break
        assert second is not None, "realtime voice never became ready"
        await ws.send(json.dumps({"type": "stop"}))
        print("  public realtime voice:", second.get("model"), second.get("voiceId"), "ready")

asyncio.run(main())
PY
then
  echo "  Nova Sonic realtime route connected"
else
  DEGRADED=1
  echo "  EXTERNAL BLOCKER: Nova Sonic realtime route is unavailable; core deploy remains active" >&2
fi

curl -fsS http://127.0.0.1:9990/api/state | python3 -c '
import json,sys
data=json.load(sys.stdin)
statuses={item.get("name"):item.get("status") for item in data.get("lobes", [])}
required={"Nexus Bus","AWS Brain API","Headless UI","Trained Voice","Codebase API","Quantum Soul","Quantum API","Akashic Hub","Pineal Gate","LoRA Server","Qwen3B Branch","Phone Bridge","Health Dashboard","n8n Workflow"}
bad={name:statuses.get(name) for name in required if statuses.get(name) != "online"}
assert not bad, f"non-online core lobes: {bad}"
print("  dashboard core lobes online:", len(required))
'

echo "── 9. trained voice and service verification ──"
curl -fsS -m 240 -D /tmp/tts.headers -X POST http://127.0.0.1:8092/synth \
  -H 'Content-Type: application/json' \
  -d '{"text":"Voice check, this is my real voice."}' \
  -o /tmp/synthcheck.bin
grep -qi '^content-type: audio/wav' /tmp/tts.headers
python3 - <<'PY'
from pathlib import Path
path=Path("/tmp/synthcheck.bin")
data=path.read_bytes()
assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", data[:16]
print("  trained voice: audio/wav RIFF/WAVE", len(data), "bytes")
PY
curl --resolve weaverv3.com:443:127.0.0.1 -fsS -m 240 -D /tmp/public-tts.headers -X POST https://weaverv3.com/tts/synth \
  -H "X-Weaver-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"text":"Public voice route connected."}' \
  -o /tmp/public-synthcheck.bin
grep -qi '^content-type: audio/wav' /tmp/public-tts.headers
python3 - <<'PY'
from pathlib import Path
data=Path("/tmp/public-synthcheck.bin").read_bytes()
assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", data[:16]
print("  public trained voice: audio/wav RIFF/WAVE", len(data), "bytes")
PY
vendor_sha=$(sha256sum "$DEPLOY_ROOT/avatar/vendor/three.module.js" | cut -d' ' -f1)
for host in weaverv3.com headless.weaverv3.com; do
  url="https://$host/vendor/three.module.js"
  live_vendor_sha=$(curl --resolve "$host:443:127.0.0.1" -fsS --max-time 20 "$url" | sha256sum | cut -d' ' -f1)
  [ "$live_vendor_sha" = "$vendor_sha" ] || { echo "vendor checksum mismatch: $url" >&2; exit 1; }
done
if curl -fsSI --max-time 20 https://weaver-avatar-404870839825.s3.amazonaws.com/weaver_avatar_dress.glb >/dev/null \
  && curl -fsSI --max-time 20 https://weaver-avatar-404870839825.s3.amazonaws.com/weaver_apartment.glb >/dev/null; then
  echo "  optional S3 visual-asset fallback reachable"
else
  echo "  warning: optional S3 visual-asset fallback unavailable; verified local assets remain primary" >&2
fi
for asset in \
  weaver_avatar_dress.glb \
  weaver_apartment.glb \
  weaver_avatar_dress_hifi.glb \
  textures/skin_normal_hifi.png \
  textures/skin_roughness_hifi.png \
  textures/skin_specular_hifi.png; do
  local_asset_sha=$(sha256sum "$DEPLOY_ROOT/avatar/$asset" | cut -d' ' -f1)
  live_asset_sha=$(curl --resolve weaverv3.com:443:127.0.0.1 -fsS --max-time 60 "https://weaverv3.com/$asset" | sha256sum | cut -d' ' -f1)
  [ "$live_asset_sha" = "$local_asset_sha" ] || { echo "visual asset checksum mismatch: $asset" >&2; exit 1; }
done
echo "  standard, penthouse, high-fidelity GLBs and PBR maps match deployed checksums"
for service in weaver-brain weaver-tts weaver-llm weaver n8n caddy; do
  sudo systemctl is-active --quiet "$service" || { echo "$service is not active" >&2; exit 1; }
done
echo "  services: $(systemctl is-active weaver-brain weaver-tts weaver-llm weaver n8n caddy | paste -sd' ' -)"
sudo docker exec n8n n8n --version | sed 's/^/  n8n version: /'
sudo docker inspect n8n --format '  n8n image: {{.Config.Image}} | digest: {{.Image}}'

echo "── 10. public static-page hashes ──"
for pair in "embodiment.html:weaverv3.com" "headless.html:headless.weaverv3.com"; do
  file="${pair%%:*}"
  host="${pair#*:}"
  url="https://$host/"
  local_md5=$(md5sum "$DEPLOY_ROOT/avatar/$file" | cut -d' ' -f1)
  live_md5=$(curl --resolve "$host:443:127.0.0.1" -fsS --max-time 30 "$url" | md5sum | cut -d' ' -f1)
  [ "$local_md5" = "$live_md5" ] || {
    echo "MISMATCH $file deployed=$local_md5 live=$live_md5" >&2
    exit 1
  }
  echo "  $file == live ($local_md5)"
done

printf '%s\n' "$DEPLOY_SHA" > "$DEPLOY_ROOT/.weaver-deployed-sha"
if (( DEGRADED )); then
  echo "CORE_FULLSTACK_DEGRADED sha=$DEPLOY_SHA external=novasonic"
else
  echo "CORE_FULLSTACK_OK sha=$DEPLOY_SHA"
fi
echo "── remote deployment complete ──"
REMOTE

echo "── external public-route verification ──"
for pair in "avatar/embodiment.html:https://weaverv3.com/" "avatar/headless.html:https://headless.weaverv3.com/"; do
  file="${pair%%:*}"
  url="${pair#*:}"
  expected=$(git -C "$ROOT" show "$DEPLOY_SHA:$file" | md5sum | cut -d' ' -f1)
  actual=$(curl -fsS --max-time 30 "$url" | md5sum | cut -d' ' -f1)
  [ "$actual" = "$expected" ] || { echo "external checksum mismatch: $url" >&2; exit 1; }
  echo "  $url reachable and current ($actual)"
done
echo "DEPLOY_COMMAND_OK sha=$DEPLOY_SHA"
