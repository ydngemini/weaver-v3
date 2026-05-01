#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# start_weaver.sh — Full-Stack Weaver v3 Launcher
# ═══════════════════════════════════════════════════════════════════
# Starts all components in the correct order with health verification.
#
# Usage:
#     ./start_weaver.sh               # Full stack (with VTV)
#     ./start_weaver.sh --headless    # Backend only (no mic/camera)
#     ./start_weaver.sh --phone-only  # Just phone bridge + dependencies
#

set -e
cd "$(dirname "$0")"

HEADLESS=false
PHONE_ONLY=false

for arg in "$@"; do
    case $arg in
        --headless) HEADLESS=true ;;
        --phone-only) PHONE_ONLY=true; HEADLESS=true ;;
    esac
done

# ── Cleanup handler ──────────────────────────────────────────────
cleanup() {
    echo ""
    echo "🛑 Shutting down Weaver..."
    kill $(jobs -p) 2>/dev/null
    # Kill any managed child processes
    pkill -f "quantum_api.py" 2>/dev/null || true
    pkill -f "health_dashboard.py" 2>/dev/null || true
    pkill -f "twilio_weaver_bridge.py" 2>/dev/null || true
    if [ "$PHONE_ONLY" = false ]; then
        pkill -f "weaver.py" 2>/dev/null || true
    fi
    pkill -f arecord 2>/dev/null || true
    pkill -f aplay 2>/dev/null || true
    echo "👋 All systems offline."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Pre-flight checks ────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════╗"
echo "║      W E A V E R   v 3   S T A R T U P         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check venv
if [ ! -d "venv" ]; then
    echo "❌ venv not found. Run: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Check .env
if [ ! -f ".env" ]; then
    echo "❌ .env file missing"
    exit 1
fi

# Source env
set -a
source .env
set +a

echo "🔍 Pre-flight checks..."
echo "   ✅ venv: found"
echo "   ✅ .env: loaded"
echo "   API keys: WEAVER_VOICE_KEY=${WEAVER_VOICE_KEY:0:10}..."
echo "   API keys: IBM_QUANTUM_TOKEN=${IBM_QUANTUM_TOKEN:0:10}..."
echo "   API keys: GEMINI_API_KEY=${GEMINI_API_KEY:0:10}..."
echo ""

# ── Clean ports ──────────────────────────────────────────────────
echo "🧹 Cleaning ports..."
for port in 9999 9998 9997 9996 8899 8765; do
    lsof -ti:$port 2>/dev/null | xargs kill -9 2>/dev/null || true
done
sleep 1

# ── Start components ─────────────────────────────────────────────

if [ "$PHONE_ONLY" = true ]; then
    echo "📞 Phone-only mode — starting minimal stack..."
    echo ""

    echo "   [1/3] Health Dashboard (port 9996)"
    venv/bin/python3 health_dashboard.py &
    sleep 1

    echo "   [2/3] Quantum API (port 9997)"
    venv/bin/python3 quantum_api.py &
    sleep 1

    echo "   [3/3] Phone Bridge (port 8765)"
    venv/bin/python3 twilio_weaver_bridge.py &
    sleep 2

else
    echo "🚀 Starting full Weaver stack..."
    echo ""

    if [ "$HEADLESS" = true ]; then
        echo "   Master Stack — headless (Nexus Bus, Quantum Soul, Pineal Gate, LoRA, Phone Bridge, Obsidian Bridge)"
        venv/bin/python3 weaver.py --headless &
    else
        echo "   Master Stack — full (all lobes + VTV Core + Obsidian Bridge)"
        WEAVER_ARGS=""
        for arg in "$@"; do
            WEAVER_ARGS="$WEAVER_ARGS $arg"
        done
        venv/bin/python3 weaver.py $WEAVER_ARGS &
    fi
    sleep 8
fi

# ── Health verification ──────────────────────────────────────────
echo ""
echo "🔍 Verifying lobe health..."

check_lobe() {
    local name=$1
    local url=$2
    if curl -s --max-time 2 "$url" > /dev/null 2>&1; then
        echo "   ✅ $name"
    else
        echo "   ⚠️  $name (not responding)"
    fi
}

check_lobe "Health Dashboard" "http://localhost:9996/health"
check_lobe "Quantum API" "http://localhost:9997/health"
check_lobe "Phone Bridge" "http://localhost:8765/health"

if [ "$PHONE_ONLY" = false ]; then
    check_lobe "Nexus Bus" "http://localhost:9998/health"
    check_lobe "LoRA Server" "http://localhost:8899/health"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "🌀 Weaver v3 is LIVE"
echo ""
echo "📊 Health Dashboard:   http://localhost:9996"
echo "⚛️  Quantum API:        http://localhost:9997/quantum/current"
echo "📞 Phone Bridge:       http://localhost:8765/health"
if [ "$PHONE_ONLY" = false ]; then
echo "🔌 Nexus Bus:          ws://localhost:9999"
echo "🧠 LoRA Server:        http://localhost:8899/health"
fi
echo ""
echo "Press Ctrl+C to stop all services"
echo "═══════════════════════════════════════════════════"

# Keep alive
wait
