#!/usr/bin/env bash
set -euo pipefail

out_dir="${WEAVER_SCREENSHOT_DIR:-$HOME/Pictures/Screenshots}"
mkdir -p "$out_dir"
out="$out_dir/weaver-$(date +%Y%m%d-%H%M%S).png"

if command -v gnome-screenshot >/dev/null 2>&1; then
  gnome-screenshot -f "$out"
elif command -v import >/dev/null 2>&1; then
  import -window root "$out"
else
  notify-send "Weaver screenshot failed" "No screenshot tool found" 2>/dev/null || true
  exit 1
fi

notify-send "Weaver screenshot" "$out" 2>/dev/null || true
printf '%s\n' "$out"
