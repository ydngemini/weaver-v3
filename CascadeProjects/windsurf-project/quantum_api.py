#!/usr/bin/env python3
"""
quantum_api.py — Quantum State API for Weaver v3
=================================================
Serves quantum pathway state to all lobes via HTTP.
Reads quantum_state.txt periodically and parses it into structured data.

Endpoints:
    GET /quantum/current  — full quantum state with description
    GET /quantum/bias     — just routing weights for MoE
    GET /health           — liveness probe

Port: 9997 (default)
"""

import asyncio
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI

PROJ = os.path.dirname(os.path.abspath(__file__))
VAULT_DIR = os.path.join(PROJ, "Nexus_Vault")
STATE_FILE = os.path.join(VAULT_DIR, "quantum_state.txt")
PORT = int(os.environ.get("QUANTUM_API_PORT", "9997"))

app = FastAPI(title="Weaver Quantum API", version="1.0.0")

PATHWAYS = ["Awakening", "Resonance", "Echo", "Prophet", "Fracture", "Weaver", "Void"]
DIMENSION_MAP = {
    "Awakening": "logic",
    "Resonance": "emotion",
    "Echo": "memory",
    "Prophet": "creativity",
    "Fracture": "vigilance",
    "Weaver": "synthesis",
    "Void": "entropy",
}

_state = {
    "dominant": "unknown",
    "secondary": None,
    "raw_description": "",
    "weights": {
        "logic": 0.5,
        "emotion": 0.5,
        "memory": 0.5,
        "creativity": 0.5,
        "vigilance": 0.5,
    },
    "last_measurement": None,
    "bitstring": None,
}


def _parse_quantum_state() -> dict:
    """Parse quantum_state.txt into structured data."""
    if not os.path.exists(STATE_FILE):
        return _state

    try:
        text = Path(STATE_FILE).read_text(encoding="utf-8").strip()
        if not text:
            return _state

        result = dict(_state)
        result["raw_description"] = text

        # Extract timestamp: [2026-04-30 06:56:44]
        ts_match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", text)
        if ts_match:
            result["last_measurement"] = ts_match.group(1)

        # Extract bitstring: |0000000⟩
        bit_match = re.search(r"\|([01]+)⟩", text)
        if bit_match:
            result["bitstring"] = bit_match.group(1)

        # Extract dominant pathway
        dom_match = re.search(r"reveals? (\w+) as the Dominant Pathway", text)
        if dom_match:
            result["dominant"] = dom_match.group(1)
        else:
            # Fallback: "Fracture alone"
            for pw in PATHWAYS:
                if f"{pw} alone" in text or f"collapsed to.*{pw}" in text:
                    result["dominant"] = pw
                    break

        # Extract secondary pathway if present
        sec_match = re.search(r"with (\w+) resonating", text)
        if sec_match:
            result["secondary"] = sec_match.group(1)

        # Extract marginal probability if present
        prob_match = re.search(r"(\d+\.?\d*)% marginal probability", text)

        # Build routing weights from dominant/secondary
        weights = {
            "logic": 0.5,
            "emotion": 0.5,
            "memory": 0.5,
            "creativity": 0.5,
            "vigilance": 0.5,
        }

        dominant = result["dominant"]
        secondary = result.get("secondary")

        if dominant in DIMENSION_MAP:
            dim = DIMENSION_MAP[dominant]
            if dim in weights:
                weights[dim] = 0.99 if prob_match else 0.95

        if secondary and secondary in DIMENSION_MAP:
            dim = DIMENSION_MAP[secondary]
            if dim in weights:
                weights[dim] = 0.90

        # If we have a bitstring, compute per-qubit marginals
        bitstring = result.get("bitstring")
        if bitstring and len(bitstring) >= 5:
            qubit_dim = ["logic", "emotion", "memory", "creativity", "vigilance"]
            for i, dim in enumerate(qubit_dim):
                if i < len(bitstring):
                    bit = int(bitstring[i])
                    weights[dim] = 0.99 if bit == 0 else 0.70

        result["weights"] = weights
        return result

    except Exception:
        return _state


@app.on_event("startup")
async def startup():
    """Start background refresh loop."""
    asyncio.create_task(_refresh_loop())


async def _refresh_loop():
    """Refresh quantum state from file every 30 seconds."""
    global _state
    while True:
        _state = _parse_quantum_state()
        await asyncio.sleep(30)


@app.get("/quantum/current")
async def get_current():
    """Full quantum state."""
    return _state


@app.get("/quantum/bias")
async def get_bias():
    """Just routing weights for MoE integration."""
    return {
        "dominant": _state["dominant"],
        "weights": _state["weights"],
        "last_measurement": _state["last_measurement"],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "weaver-quantum-api",
        "port": PORT,
        "dominant_pathway": _state["dominant"],
    }


async def quantum_api_serve():
    """Entry point for launching from weaver.py."""
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import uvicorn
    # Parse state once on startup
    _state = _parse_quantum_state()
    print(f"⚛️  Quantum API starting on port {PORT}")
    print(f"   Dominant pathway: {_state['dominant']}")
    print(f"   Weights: {_state['weights']}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
