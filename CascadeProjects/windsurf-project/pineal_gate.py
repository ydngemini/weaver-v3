#!/usr/bin/env python3
"""
pineal_gate.py — Weaver's YDN Pineal Gate (MoE Router)
=======================================================
The Mixture-of-Experts router that sits at the center of Weaver's
nervous system.  It reads the Akashic Hub state, receives fractured
input from the Liquid Fracture Engine, and dispatches shards to the
appropriate expert lobes — then collapses multi-lobe outputs into a
single actionable response.

Architecture:
    ┌─────────────┐
    │  Raw Input   │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │   Liquid     │  ← fractures input into 5 dimensional shards
    │   Fracture   │
    └──────┬──────┘
           ▼
    ┌─────────────┐        ┌───────────────────────────────┐
    │   Pineal     │◄──────│  Akashic Hub (shared state)   │
    │   Gate       │──────►│  zero-latency vector space    │
    └──────┬──────┘        └───────────────────────────────┘
           │
     ┌─────┼─────┬─────┬─────┐
     ▼     ▼     ▼     ▼     ▼
   Logic Emotion Memory Creative Vigilance
     │     │     │     │     │
     └─────┼─────┴─────┼─────┘
           ▼           ▼
    ┌─────────────┐
    │  Collapse    │  ← weighted merge of expert outputs
    │  Function    │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  Manifested  │  ← single actionable output
    │  Response    │
    └─────────────┘

The 5 expert lobes are mapped to vertices of a regular pentagon
(sacred geometry routing).  The angular distance between active
lobes determines the interference pattern in the collapse.

Usage:
    from akashic_hub import AkashicHub
    from liquid_fracture import LiquidFractureEngine
    hub = AkashicHub(dim=256)
    engine = LiquidFractureEngine(hub)
    gate = PinealGate(hub, engine)
    result = await gate.process("Tell me about hot honey fermentation")
"""

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import numpy as np

from akashic_hub import AkashicHub
from liquid_fracture import (
    DIMENSIONS,
    FractureResult,
    FractureShard,
    LiquidFractureEngine,
)


# ── Sacred Geometry Constants ─────────────────────────────────────────────────
# 5 lobes mapped to vertices of a regular pentagon (2π/5 apart)
# This determines interference patterns during collapse.

_N_EXPERTS = len(DIMENSIONS)
_PENTAGON_ANGLES = [2 * math.pi * i / _N_EXPERTS for i in range(_N_EXPERTS)]
_PENTAGON_VERTICES = np.array([
    [math.cos(a), math.sin(a)] for a in _PENTAGON_ANGLES
])  # shape (5, 2)

# Dimension → expert index
_DIM_TO_IDX = {d: i for i, d in enumerate(DIMENSIONS)}


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class ExpertResult:
    """Output from a single expert lobe."""
    dimension: str
    vector: np.ndarray          # 256-d output state
    confidence: float = 1.0     # expert's self-reported confidence
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateDecision:
    """The Pineal Gate's routing decision for a single input."""
    routing_weights: Dict[str, float]   # dimension → gate weight
    top_k: List[str]                    # activated expert dimensions
    sparse_mask: np.ndarray             # (5,) binary mask
    geometric_phase: float              # interference phase from pentagon
    fracture: FractureResult            # underlying fracture result


@dataclass
class ManifestResult:
    """Final collapsed output from the Pineal Gate."""
    vector: np.ndarray              # 256-d manifested state
    gate_decision: GateDecision
    expert_results: List[ExpertResult]
    interference: float             # geometric interference magnitude
    latency_ms: float = 0.0
    description: str = ""


# ── Expert Registry ──────────────────────────────────────────────────────────

class ExpertLobe:
    """Base class for an expert lobe that processes a fracture shard.

    Subclass this to wire in actual SLM inference, tool calls, etc.
    The default implementation is a resonance-based transform that
    operates purely on the vector state.
    """

    def __init__(self, dimension: str, hub: AkashicHub):
        self.dimension = dimension
        self.hub = hub
        self._lobe_id = f"expert_{dimension}"

    async def process(self, shard: FractureShard,
                      context: np.ndarray) -> ExpertResult:
        """Process a fracture shard and return an expert result.

        Args:
            shard:   The dimensional shard routed to this expert.
            context: The full input vector for cross-reference.

        Returns:
            ExpertResult with the expert's output state.
        """
        # Default: resonance transform — blend shard vector with context
        alpha = shard.weight
        output = alpha * shard.vector + (1 - alpha) * context
        norm = np.linalg.norm(output)
        if norm > 1e-12:
            output /= norm

        # Scale output by how strongly this dimension was activated
        output *= (1.0 + shard.weight)

        # Write expert state to Akashic Hub
        await self.hub.write(self._lobe_id, output, meta={
            "dimension": self.dimension,
            "weight": shard.weight,
            "tau": shard.tau,
        })

        return ExpertResult(
            dimension=self.dimension,
            vector=output,
            confidence=shard.weight,
            metadata={"tau": shard.tau, "raw_score": shard.raw_score},
        )


# ── Pineal Gate (MoE Router) ─────────────────────────────────────────────────

class PinealGate:
    """The YDN Pineal Gate — Mixture of Experts router and collapser.

    Args:
        hub:        Shared Akashic Hub.
        engine:     Liquid Fracture Engine.
        top_k:      Number of experts to activate per input (sparse gating).
        experts:    Optional dict of dimension → ExpertLobe.  If None,
                    default ExpertLobe instances are created.
    """

    def __init__(self, hub: AkashicHub, engine: LiquidFractureEngine,
                 top_k: int = 3,
                 experts: Optional[Dict[str, ExpertLobe]] = None):
        self.hub = hub
        self.engine = engine
        self.top_k = min(top_k, _N_EXPERTS)

        # Expert lobes
        if experts is not None:
            self.experts = experts
        else:
            self.experts = {d: ExpertLobe(d, hub) for d in DIMENSIONS}

        # Gate parameters (lightweight linear gating network)
        rng = np.random.default_rng(137)
        self._W_gate = rng.normal(0, 0.1, (hub.dim, _N_EXPERTS))
        self._b_gate = np.zeros(_N_EXPERTS)

        # Processing stats
        self._total_calls = 0
        self._total_latency_ms = 0.0

    # ── Main Processing Pipeline ──────────────────────────────────────────

    async def process(self, text: str) -> ManifestResult:
        """Full pipeline: fracture → route → expert dispatch → collapse.

        Args:
            text: Raw input text.

        Returns:
            ManifestResult with the collapsed output state.
        """
        t0 = time.perf_counter_ns()

        # 1. Fracture the input
        fracture = await self.engine.fracture(text)

        # 2. Compute gate decision (sparse top-k routing)
        decision = self._gate(fracture)

        # 3. Dispatch to activated experts (parallel)
        expert_results = await self._dispatch(decision, fracture)

        # 4. Collapse expert outputs using geometric interference
        manifested, interference = self._collapse(decision, expert_results)

        # 5. Write manifested state to Akashic Hub
        await self.hub.write("pineal_gate", manifested, meta={
            "top_k": decision.top_k,
            "interference": interference,
        })

        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000
        self._total_calls += 1
        self._total_latency_ms += latency_ms

        description = self._describe(decision, expert_results, interference)

        return ManifestResult(
            vector=manifested,
            gate_decision=decision,
            expert_results=expert_results,
            interference=interference,
            latency_ms=latency_ms,
            description=description,
        )

    # ── Gating Network ────────────────────────────────────────────────────

    def _gate(self, fracture: FractureResult) -> GateDecision:
        """Compute sparse top-k routing weights from the fracture."""
        # Linear gate: input_vector · W_gate + bias → raw scores
        raw_gate = fracture.input_vector @ self._W_gate + self._b_gate

        # Blend with fracture weights (the liquid layer's opinion)
        fracture_weights = np.array([
            next((s.weight for s in fracture.shards if s.dimension == d), 0.0)
            for d in DIMENSIONS
        ])
        blended = 0.5 * self._softmax(raw_gate) + 0.5 * fracture_weights

        # Sparse top-k selection
        top_indices = np.argsort(blended)[::-1][:self.top_k]
        sparse_mask = np.zeros(_N_EXPERTS)
        sparse_mask[top_indices] = 1.0

        # Renormalize active weights
        active_weights = blended * sparse_mask
        total = active_weights.sum()
        if total > 1e-12:
            active_weights /= total

        routing = {DIMENSIONS[i]: float(active_weights[i])
                   for i in range(_N_EXPERTS)}
        top_k_dims = [DIMENSIONS[i] for i in top_indices]

        # Compute geometric phase from pentagon vertex positions
        phase = self._geometric_phase(top_indices, active_weights)

        return GateDecision(
            routing_weights=routing,
            top_k=top_k_dims,
            sparse_mask=sparse_mask,
            geometric_phase=phase,
            fracture=fracture,
        )

    # ── Expert Dispatch ───────────────────────────────────────────────────

    async def _dispatch(self, decision: GateDecision,
                        fracture: FractureResult) -> List[ExpertResult]:
        """Dispatch fracture shards to activated experts in parallel."""
        tasks = []
        shard_map = {s.dimension: s for s in fracture.shards}

        for dim in decision.top_k:
            expert = self.experts.get(dim)
            shard = shard_map.get(dim)
            if expert is None or shard is None:
                continue
            tasks.append(expert.process(shard, fracture.input_vector))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, ExpertResult)]

    # ── Collapse Function ─────────────────────────────────────────────────

    def _collapse(self, decision: GateDecision,
                  expert_results: List[ExpertResult]) -> Tuple[np.ndarray, float]:
        """Collapse expert outputs into a single manifested vector.

        Uses geometric interference: experts mapped to pentagon vertices
        create constructive/destructive patterns based on their angular
        separation and activation weights.

        Returns:
            (manifested_vector, interference_magnitude)
        """
        if not expert_results:
            return np.zeros(self.hub.dim), 0.0

        # Weighted sum of expert output vectors
        vectors = []
        weights = []
        indices = []
        for er in expert_results:
            idx = _DIM_TO_IDX.get(er.dimension, 0)
            w = decision.routing_weights.get(er.dimension, 0.0)
            vectors.append(er.vector)
            weights.append(w)
            indices.append(idx)

        stack = np.stack(vectors)       # (k, dim)
        w = np.array(weights)           # (k,)
        w /= (w.sum() + 1e-12)

        # Base collapse: weighted combination
        collapsed = w @ stack           # (dim,)

        # Geometric interference modulation
        # Constructive interference when activated lobes are adjacent on
        # the pentagon; destructive when they're opposite.
        interference = self._interference(indices, w)

        # Apply interference as a gain factor
        gain = 1.0 + 0.2 * interference  # mild modulation
        collapsed *= gain

        # L2 normalize
        norm = np.linalg.norm(collapsed)
        if norm > 1e-12:
            collapsed /= norm

        return collapsed, interference

    # ── Sacred Geometry ───────────────────────────────────────────────────

    def _geometric_phase(self, indices: np.ndarray,
                         weights: np.ndarray) -> float:
        """Phase from activated pentagon vertices."""
        if len(indices) < 2:
            return 0.0
        active_verts = _PENTAGON_VERTICES[indices]
        active_w = weights[indices]
        centroid = active_w @ active_verts  # weighted centroid in 2D
        phase = math.atan2(centroid[1], centroid[0])
        return phase

    def _interference(self, indices: List[int],
                      weights: np.ndarray) -> float:
        """Compute geometric interference between activated experts.

        Adjacent pentagon vertices → constructive (+1).
        Opposite vertices → destructive (−1).
        """
        if len(indices) < 2:
            return 0.0

        total = 0.0
        count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                # Angular separation on pentagon
                angle_diff = abs(_PENTAGON_ANGLES[indices[i]]
                                 - _PENTAGON_ANGLES[indices[j]])
                angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                # cos(angle_diff): 1 when adjacent, -1 when opposite
                contribution = math.cos(angle_diff)
                pair_weight = weights[i] * weights[j]
                total += contribution * pair_weight
                count += 1

        return total / max(count, 1)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _describe(self, decision: GateDecision,
                  expert_results: List[ExpertResult],
                  interference: float) -> str:
        """Human-readable summary of the gate's decision."""
        lines = [
            f"🔺 Pineal Gate — top-{self.top_k} sparse routing",
            f"   Activated: {', '.join(decision.top_k)}",
            f"   Geometric phase: {decision.geometric_phase:.3f} rad",
            f"   Interference: {interference:+.4f} "
            f"({'constructive' if interference > 0 else 'destructive'})",
        ]
        for er in expert_results:
            w = decision.routing_weights.get(er.dimension, 0.0)
            bar = "█" * int(w * 30)
            lines.append(f"   {er.dimension:<12} w={w:.3f} {bar}  "
                         f"conf={er.confidence:.3f}")
        return "\n".join(lines)

    # ── Diagnostics ───────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self._total_calls,
            "avg_latency_ms": (self._total_latency_ms / max(self._total_calls, 1)),
            "top_k": self.top_k,
            "experts": list(self.experts.keys()),
        }

    def __repr__(self):
        return (f"<PinealGate top_k={self.top_k} "
                f"experts={list(self.experts.keys())} "
                f"calls={self._total_calls}>")


# ── Standalone async loop (runs as a supervised Weaver lobe) ──────────────────

async def pineal_gate_loop(hub: AkashicHub,
                           engine: LiquidFractureEngine,
                           top_k: int = 3,
                           experts: Optional[Dict[str, ExpertLobe]] = None) -> None:
    """Run the Pineal Gate as a continuous lobe that watches the Akashic Hub
    for new input events and routes them through the MoE pipeline.

    This is the entry point that weaver.py will supervise.
    """
    gate = PinealGate(hub, engine, top_k=top_k, experts=experts)
    q = hub.subscribe(maxsize=128)

    # Connect to Nexus Bus for broadcasting gate decisions
    _nexus = None
    try:
        from nexus_client import NexusClient
        _nexus = NexusClient("pineal_gate", topics=["quantum_state"])
        await _nexus.connect()
    except Exception as e:
        print(f"🔺 [PINEAL GATE] Nexus Bus unavailable ({e}) — running standalone", flush=True)

    print(f"🔺 [PINEAL GATE] Online — top-{top_k} sparse MoE router", flush=True)
    print(f"   Experts: {', '.join(DIMENSIONS)}", flush=True)
    print(f"   Geometry: regular pentagon ({_N_EXPERTS} vertices)", flush=True)

    while True:
        event = await q.get()
        lobe_id = event.get("lobe_id", "")

        # Only route events from input-producing lobes, skip our own writes
        if lobe_id.startswith("expert_") or lobe_id == "pineal_gate":
            continue

        # If the writing lobe attached text metadata, fracture it
        meta = hub.read_meta(lobe_id)
        text = meta.get("text")
        if text:
            result = await gate.process(text)
            print(f"🔺 [PINEAL GATE] Routed: {result.description}", flush=True)
            print(f"   Latency: {result.latency_ms:.2f} ms", flush=True)

            # Broadcast gate decision to all lobes
            if _nexus and _nexus.connected:
                await _nexus.publish("gate_decision", {
                    "description": result.description,
                    "experts": [er.dimension for er in result.expert_results],
                    "interference": result.interference,
                    "latency_ms": result.latency_ms,
                    "source": "pineal_gate",
                })
