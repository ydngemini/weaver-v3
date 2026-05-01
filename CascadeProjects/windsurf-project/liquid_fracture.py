#!/usr/bin/env python3
"""
liquid_fracture.py — Weaver's Liquid Fracture Engine
=====================================================
Fuses MIT Liquid Neural Network dynamics with the Fracture Principle to
create an adaptive synaptic layer that molds itself to each input in
real time — no retraining required.

Core ideas:
    1. **Liquid Time-Constants (LTC):**  Synaptic weights are governed by
       a continuous-time ODE whose time-constant τ adapts to the input.
       dx/dt = −x / τ(input) + f(input, x)
       When τ is small the network reacts fast; when τ is large the
       network smooths over noise.  τ itself is a learned function of
       the input — that's the "liquid" property.

    2. **Fracture Decomposition:**  A complex prompt is shattered into
       N dimensional shards along semantic axes (logic, emotion, memory,
       creativity, vigilance).  Each shard carries a routing weight that
       the liquid layer adjusts dynamically.

    3. **Integration with Akashic Hub:**  The engine reads the current
       hub state to bias the fracture — if the Quantum Soul is
       resonating in "Awakening", the fracture tilts creative weight
       higher automatically.

Usage:
    from akashic_hub import AkashicHub
    hub = AkashicHub(dim=256)
    engine = LiquidFractureEngine(hub)
    shards = await engine.fracture("Tell me about hot honey fermentation")
    # shards = list of FractureShard with .dimension, .weight, .vector
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# ── Fracture Dimensions ──────────────────────────────────────────────────────
# These are the semantic axes along which every input is shattered.
# Each maps to one of the 5 expert lobes in the Pineal Gate.

DIMENSIONS = [
    "logic",       # analytical reasoning, structure, planning
    "emotion",     # sentiment, empathy, creative feeling
    "memory",      # recall, context, continuity
    "creativity",  # novel synthesis, metaphor, art
    "vigilance",   # threat detection, paranoia, safety
]
# Pentagon geometry requires exactly 5 dimensions — one per vertex.
# quantum_networks.DIMENSION_QUBITS and pineal_gate pentagon math both
# assume this invariant. Changing it will break qubit routing and expert
# interference patterns.
assert len(DIMENSIONS) == 5, (
    f"Pentagon geometry requires exactly 5 DIMENSIONS, got {len(DIMENSIONS)}"
)

# Seed keywords that bias each dimension (used by the fracture heuristic)
_DIM_SEEDS: Dict[str, List[str]] = {
    "logic":      ["because", "therefore", "step", "plan", "calculate",
                   "structure", "algorithm", "reason", "proof", "if"],
    "emotion":    ["feel", "love", "hate", "fear", "joy", "pain",
                   "beautiful", "soul", "heart", "passion"],
    "memory":     ["remember", "last time", "before", "history", "you said",
                   "previously", "recall", "past", "context", "back when"],
    "creativity": ["imagine", "create", "design", "invent", "dream",
                   "recipe", "verse", "song", "art", "build"],
    "vigilance":  ["danger", "risk", "careful", "threat", "watch out",
                   "security", "protect", "warning", "suspicious", "alert"],
}

# Probability-field smoothing floor — ensures no dimension collapses to zero weight.
# Applied as an additive constant to raw cosine scores before normalization so that
# even a dimension with zero keyword overlap retains a non-zero routing probability.
SMOOTHING = 0.1


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class FractureShard:
    """One dimensional fragment of a fractured input."""
    dimension: str          # which semantic axis
    weight: float           # routing weight ∈ [0, 1], sum over shards = 1
    vector: np.ndarray      # 256-d representation of this shard
    raw_score: float = 0.0  # pre-normalization affinity score
    tau: float = 1.0        # liquid time-constant for this dimension


@dataclass
class FractureResult:
    """Complete output of a fracture operation."""
    shards: List[FractureShard]
    input_vector: np.ndarray    # full 256-d embedding of the raw input
    latency_ms: float = 0.0
    liquid_state: Optional[np.ndarray] = None  # internal ODE state


# ── Liquid Time-Constant Cell ────────────────────────────────────────────────

class LiquidCell:
    """A single liquid neuron whose time-constant adapts to the input.

    Implements the simplified LTC ODE via Euler integration:
        dx/dt = −x / τ(z) + tanh(W_in · z + W_rec · x + b)
    where τ(z) = τ_base · σ(w_τ · z + b_τ)  and σ is sigmoid.
    """

    def __init__(self, input_dim: int, state_dim: int, tau_base: float = 1.0):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.tau_base = tau_base

        rng = np.random.default_rng(42)
        scale_in = np.sqrt(2.0 / (input_dim + state_dim))
        scale_rec = np.sqrt(2.0 / (state_dim + state_dim))

        self.W_in  = rng.normal(0, scale_in, (state_dim, input_dim))
        self.W_rec = rng.normal(0, scale_rec, (state_dim, state_dim))
        self.b     = np.zeros(state_dim)

        # Time-constant parameters
        self.w_tau = rng.normal(0, scale_in, (1, input_dim))
        self.b_tau = np.zeros(1)

        # Internal state
        self.x = np.zeros(state_dim)

    def tau(self, z: np.ndarray) -> float:
        """Compute input-dependent time-constant."""
        raw = float((self.w_tau @ z + self.b_tau).item())
        sigma = 1.0 / (1.0 + np.exp(-np.clip(raw, -10, 10)))
        return self.tau_base * (0.1 + 1.9 * sigma)  # τ ∈ [0.1·base, 2·base]

    def step(self, z: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """One adaptive Euler step of the liquid ODE.

        Automatically reduces dt when tau is small to maintain the
        stability ratio dt/tau < 0.5 (per Hasani et al. 2020).
        """
        z = np.asarray(z, dtype=np.float64).ravel()[:self.input_dim]
        if z.shape[0] < self.input_dim:
            z = np.pad(z, (0, self.input_dim - z.shape[0]))

        t = self.tau(z)
        # Adaptive step: clamp dt so dt/tau < 0.5 for Euler stability
        effective_dt = min(dt, 0.5 * t)
        activation = np.tanh(self.W_in @ z + self.W_rec @ self.x + self.b)
        dx = -self.x / t + activation
        self.x = self.x + effective_dt * dx
        return self.x.copy()

    def reset(self):
        """Reset internal state to zero."""
        self.x = np.zeros(self.state_dim)


# ── Liquid Fracture Engine ────────────────────────────────────────────────────

class LiquidFractureEngine:
    """Shatters input across semantic dimensions with liquid-adaptive routing.

    Args:
        hub:       Reference to the shared AkashicHub (or None for standalone).
        dim:       Vector dimensionality (must match hub.dim).
        n_steps:   Number of ODE integration steps per fracture.
        dt:        Euler step size.
        tau_base:  Base time-constant for the liquid cells.
    """

    def __init__(self, hub=None, dim: int = 256, n_steps: int = 4,
                 dt: float = 0.15, tau_base: float = 1.0):
        self.hub = hub
        self.dim = dim
        self.n_steps = n_steps
        self.dt = dt

        # One liquid cell per dimension
        self.cells: Dict[str, LiquidCell] = {
            d: LiquidCell(input_dim=dim, state_dim=dim, tau_base=tau_base)
            for d in DIMENSIONS
        }

        # Text → vector via sklearn hashing (no fitting, works on any text)
        self._vectorizer = HashingVectorizer(
            n_features=dim, alternate_sign=False, norm="l2"
        )

        # Precompute seed vectors for each dimension
        self._seed_vectors: Dict[str, np.ndarray] = {}
        for d, seeds in _DIM_SEEDS.items():
            sv = self._vectorizer.transform([" ".join(seeds)]).toarray().ravel()
            self._seed_vectors[d] = sv

    # ── Public API ────────────────────────────────────────────────────────

    async def fracture(self, text: str) -> FractureResult:
        """Fracture a text input into weighted dimensional shards.

        Returns a FractureResult with shards sorted by weight (descending).
        """
        t0 = time.perf_counter_ns()

        # 1. Embed the raw input
        input_vec = self._embed(text)

        # 2. Compute raw affinity to each dimension
        raw_scores: Dict[str, float] = {}
        for d in DIMENSIONS:
            seed = self._seed_vectors[d]
            # Cosine similarity (both are L2-normalized by HashingVectorizer)
            sim = float(np.dot(input_vec, seed))
            raw_scores[d] = SMOOTHING + max(sim, 0.0)  # smoothing floor + clamp negatives

        # 3. Bias from Akashic Hub state (if available)
        hub_bias = self._hub_bias()

        # 4. Run each dimension through its liquid cell
        shards: List[FractureShard] = []
        for d in DIMENSIONS:
            cell = self.cells[d]
            # Combine input with hub bias
            z = input_vec.copy()
            if hub_bias is not None:
                z = 0.8 * z + 0.2 * hub_bias

            # Integrate the liquid ODE for n_steps
            for _ in range(self.n_steps):
                cell.step(z, dt=self.dt)

            # The cell's time-constant tells us the adaptation speed
            tau = cell.tau(z)

            # Final shard score = raw affinity * liquid modulation
            liquid_mod = float(np.linalg.norm(cell.x))
            score = raw_scores[d] * (1.0 + 0.3 * liquid_mod)

            shards.append(FractureShard(
                dimension=d,
                weight=score,       # will be normalized below
                vector=cell.x.copy(),
                raw_score=raw_scores[d],
                tau=tau,
            ))

        # 5. Normalize weights to sum to 1
        total = sum(s.weight for s in shards)
        if total > 1e-12:
            for s in shards:
                s.weight /= total
        else:
            # Uniform fallback
            for s in shards:
                s.weight = 1.0 / len(DIMENSIONS)

        # 6. Sort by weight descending
        shards.sort(key=lambda s: s.weight, reverse=True)

        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000

        return FractureResult(
            shards=shards,
            input_vector=input_vec,
            latency_ms=latency_ms,
            liquid_state=np.stack([s.vector for s in shards]),
        )

    def reset(self):
        """Reset all liquid cell states (e.g. between conversations)."""
        for cell in self.cells.values():
            cell.reset()

    # ── Internals ─────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Convert text to a fixed-size vector."""
        return self._vectorizer.transform([text]).toarray().ravel()

    def _hub_bias(self) -> Optional[np.ndarray]:
        """Read the Akashic Hub for a global bias vector."""
        if self.hub is None:
            return None
        states = self.hub.read_all()
        if not states:
            return None
        # Blend all active lobe states into a single bias
        vecs = list(states.values())
        mean = np.mean(np.stack(vecs), axis=0)
        norm = np.linalg.norm(mean)
        if norm > 1e-12:
            mean /= norm
        return mean

    # ── Diagnostics ───────────────────────────────────────────────────────

    def describe(self, result: FractureResult) -> str:
        """Human-readable summary of a fracture result."""
        lines = [f"⚡ Fracture complete in {result.latency_ms:.2f} ms"]
        for s in result.shards:
            bar = "█" * int(s.weight * 40)
            lines.append(f"  {s.dimension:<12} {s.weight:.3f} {bar}  τ={s.tau:.3f}")
        return "\n".join(lines)
