#!/usr/bin/env bash
# Run ON the Oracle A1 box, AFTER deploy/setup_oracle.sh (Weaver venv + llama-cpp)
# and AFTER the Oracle code is synced to ~/oracle. This wires up the extras the
# base kit doesn't cover: local expert LLM model, Oracle backend, frontend build,
# Caddy public TLS, and all systemd units.
#
# Required env:
#   ORACLE_HOST          public hostname for TLS, e.g. <PUBLIC_IP>.sslip.io
# Optional env:
#   EXPERTS_GGUF_URL     override the experts model download URL
#   CLOUD                aws|oracle (default oracle) — tunes the final firewall guidance
set -euo pipefail

WIN="$HOME/weaver/CascadeProjects/windsurf-project"
ORA="$HOME/oracle"
: "${ORACLE_HOST:?set ORACLE_HOST=<PUBLIC_IP>.sslip.io (or your domain) first}"
EXPERTS_GGUF_URL="${EXPERTS_GGUF_URL:-https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
CLOUD="${CLOUD:-oracle}"

echo "▶ 1/6  Experts GGUF for the local expert LLM (:8090)"
mkdir -p "$WIN/models"
if [ ! -f "$WIN/models/experts.gguf" ]; then
    curl -fL "$EXPERTS_GGUF_URL" -o "$WIN/models/experts.gguf"
else
    echo "   (models/experts.gguf already present — skipping)"
fi

echo "▶ 2/6  Oracle backend venv + deps"
python3 -m venv "$ORA/backend/venv"
"$ORA/backend/venv/bin/pip" install --upgrade pip wheel
"$ORA/backend/venv/bin/pip" install -r "$ORA/backend/requirements.txt"
[ -f "$ORA/backend/.env" ] || { cp "$WIN/deploy/env.oracle-backend.example" "$ORA/backend/.env"; \
    echo "   ⚠ created $ORA/backend/.env from template — set ORACLE_SECRET_KEY + admin creds before going public"; }

echo "▶ 3/6  Build the Oracle frontend (prod env baked in) → /var/www/oracle"
sudo apt-get install -y nodejs npm
cd "$ORA/oracle-app"
npm ci || npm install
VITE_WS_URL="wss://$ORACLE_HOST/ws" \
VITE_API_BASE="https://$ORACLE_HOST" \
VITE_USER_ID="operator" \
VITE_TENANT_ID="a1b2c3d4-0000-4000-8000-000000000001" \
    npm run build
sudo mkdir -p /var/www/oracle
sudo rm -rf /var/www/oracle/*
sudo cp -r dist/* /var/www/oracle/
cd - >/dev/null

echo "▶ 4/6  Caddy (automatic HTTPS)"
if ! command -v caddy >/dev/null; then
    sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl gnupg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
    sudo apt-get update -y
    sudo apt-get install -y caddy
fi
sudo cp "$WIN/deploy/Caddyfile" /etc/caddy/Caddyfile
echo "ORACLE_HOST=$ORACLE_HOST" | sudo tee /etc/default/caddy >/dev/null
# The stock caddy.service doesn't read an env file — add a drop-in so {$ORACLE_HOST} resolves.
sudo mkdir -p /etc/systemd/system/caddy.service.d
printf '[Service]\nEnvironmentFile=/etc/default/caddy\n' \
    | sudo tee /etc/systemd/system/caddy.service.d/env.conf >/dev/null

echo "▶ 5/6  Install + start systemd units"
sudo cp "$WIN/deploy/weaver-llm.service" "$WIN/deploy/weaver.service" \
        "$WIN/deploy/oracle-backend.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now weaver-llm oracle-backend caddy
echo "   …giving the experts server a few seconds to load before Weaver starts"
sleep 6
sudo systemctl enable --now weaver

if [ "$CLOUD" = "aws" ]; then
echo "▶ 6/6  Firewall (AWS — Security Group only)"
cat <<EOF

✅ Services installed and starting.
Firewall: your Terraform Security Group already allows 80 + 443 (0.0.0.0/0) and 22 (your IP).
AWS Ubuntu AMIs ship no default iptables REJECT — there is nothing to open on this box.

Then:  https://$ORACLE_HOST
Logs:  journalctl -u weaver-llm -u weaver -u oracle-backend -u caddy -f
EOF
else
echo "▶ 6/6  Firewall (the Oracle #1 gotcha — two layers)"
cat <<EOF

✅ Services installed and starting. Open 80 + 443 in BOTH layers:
  1) Oracle Cloud console → Networking → VCN → Security List →
     add Ingress  0.0.0.0/0  TCP  80,443
  2) On this box:
       sudo iptables -I INPUT 5 -p tcp --dport 80  -j ACCEPT
       sudo iptables -I INPUT 5 -p tcp --dport 443 -j ACCEPT
       sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save

Then:  https://$ORACLE_HOST
Logs:  journalctl -u weaver-llm -u weaver -u oracle-backend -u caddy -f
EOF
fi
