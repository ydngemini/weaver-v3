#!/usr/bin/env bash
set -euo pipefail

state_dir="${XDG_STATE_HOME:-$HOME/.local/state}/weaver-hotkeys"
out_dir="${WEAVER_RECORD_DIR:-$HOME/Videos/Weaver}"
mkdir -p "$state_dir" "$out_dir"

pidfile="$state_dir/screenrecord.pid"
logfile="$state_dir/screenrecord.log"

if [[ -s "$pidfile" ]]; then
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    kill -INT "$pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.1
    done
    kill -TERM "$pid" 2>/dev/null || true
    rm -f "$pidfile"
    notify-send "Weaver screen recording" "Stopped" 2>/dev/null || true
    exit 0
  fi
  rm -f "$pidfile"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  notify-send "Weaver screen recording failed" "ffmpeg is not installed" 2>/dev/null || true
  exit 1
fi

geometry="$(xdpyinfo 2>/dev/null | awk '/dimensions:/{print $2; exit}' || true)"
if [[ -z "${geometry:-}" ]]; then
  geometry="$(xrandr 2>/dev/null | awk '/\\*/{print $1; exit}' || true)"
fi
if [[ -z "${geometry:-}" ]]; then
  notify-send "Weaver screen recording failed" "Could not read X11 screen size" 2>/dev/null || true
  exit 1
fi

display="${DISPLAY:-:0}"
case "$display" in
  *.*) x11_input="${display}+0,0" ;;
  *) x11_input="${display}.0+0,0" ;;
esac

out="$out_dir/weaver-recording-$(date +%Y%m%d-%H%M%S).mp4"
audio_args=(-an)
if command -v pactl >/dev/null 2>&1 && pactl info >/dev/null 2>&1; then
  audio_args=(-f pulse -i default -c:a aac -b:a 160k)
fi

nohup ffmpeg \
  -hide_banner -loglevel warning -y \
  -f x11grab -draw_mouse 1 -framerate 60 -video_size "$geometry" -i "$x11_input" \
  "${audio_args[@]}" \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p \
  "$out" >"$logfile" 2>&1 &

pid="$!"
printf '%s\n' "$pid" >"$pidfile"
notify-send "Weaver screen recording" "Started: $out" 2>/dev/null || true
printf '%s\n' "$out"
