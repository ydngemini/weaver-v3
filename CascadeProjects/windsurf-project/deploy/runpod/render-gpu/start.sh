#!/usr/bin/env bash
set -euo pipefail
cd /root/weaver-render-gpu
pkill -f splat_render_server.py 2>/dev/null || true
sleep 1
nohup python3 -u splat_render_server.py >> render.log 2>&1 &
echo "weaver-render-gpu pid $!"
