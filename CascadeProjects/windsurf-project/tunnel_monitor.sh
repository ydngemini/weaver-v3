#!/bin/bash
# tunnel_monitor.sh — Keeps a localhost.run tunnel alive and updates Twilio webhook
# Usage: ./tunnel_monitor.sh &
#
# Starts an SSH tunnel to localhost.run, monitors for URL changes,
# and auto-updates the Twilio webhook whenever the URL rotates.

set -e
cd "$(dirname "$0")"

source .env 2>/dev/null || true

TUNNEL_LOG="/tmp/tunnel_output.txt"
LAST_URL=""

update_twilio() {
    local url="$1"
    venv/bin/python3 -c "
import os
from dotenv import load_dotenv
load_dotenv('$PWD/.env')
from twilio.rest import Client
c = Client(os.environ['TWILIO_ACCOUNT_SID'], os.environ['TWILIO_AUTH_TOKEN'])
for n in c.incoming_phone_numbers.list():
    n.update(voice_url='${url}/twiml', voice_method='POST')
" 2>/dev/null && echo "[TUNNEL] Twilio updated -> ${url}/twiml" || echo "[TUNNEL] Twilio update failed"
}

start_tunnel() {
    # Kill any existing tunnel
    pkill -f "ssh.*localhost.run" 2>/dev/null || true
    sleep 1

    # Start fresh tunnel
    ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
        -R 80:localhost:8765 nokey@localhost.run > "$TUNNEL_LOG" 2>&1 &
    TUNNEL_PID=$!
    echo "[TUNNEL] Started SSH tunnel (PID: $TUNNEL_PID)"

    # Wait for URL to appear
    for i in $(seq 1 15); do
        sleep 1
        URL=$(grep -o 'https://[a-z0-9]*\.lhr\.life' "$TUNNEL_LOG" 2>/dev/null | tail -1)
        if [ -n "$URL" ]; then
            echo "[TUNNEL] URL: $URL"
            if [ "$URL" != "$LAST_URL" ]; then
                LAST_URL="$URL"
                update_twilio "$URL"
            fi
            return 0
        fi
    done
    echo "[TUNNEL] Failed to get URL after 15s"
    return 1
}

# Initial start
start_tunnel

# Monitor loop — restart if tunnel dies or URL changes
while true; do
    sleep 10

    # Check if SSH process is still alive
    if ! pgrep -f "ssh.*localhost.run" > /dev/null 2>&1; then
        echo "[TUNNEL] SSH tunnel died — restarting..."
        start_tunnel
        continue
    fi

    # Check for URL changes (localhost.run can reassign)
    CURRENT_URL=$(grep -o 'https://[a-z0-9]*\.lhr\.life' "$TUNNEL_LOG" 2>/dev/null | tail -1)
    if [ -n "$CURRENT_URL" ] && [ "$CURRENT_URL" != "$LAST_URL" ]; then
        echo "[TUNNEL] URL changed: $LAST_URL -> $CURRENT_URL"
        LAST_URL="$CURRENT_URL"
        update_twilio "$CURRENT_URL"
    fi

    # Check if tunnel is actually forwarding (health check through tunnel)
    # Skip this if curl has SSL issues on this machine
done
