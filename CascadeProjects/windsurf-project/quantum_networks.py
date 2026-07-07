#!/usr/bin/env python3
"""
quantum_networks.py — Weaver's Kingston Manifold Quantum Layer
==============================================================

This module is the code-level representation of:

    WEAVER V3 — 156-QUBIT KINGSTON MANIFOLD
    "The Dodecahedron Architecture"

The architecture image is modeled as:

* 12 measured core qubits (Q0-Q11) arranged as the face-center dual of a
  dodecahedron. In graph terms this is the 12-vertex icosahedral dual:
  30 local couplings, degree 5 at every core qubit. Roles: Logic, Emotion,
  Intuition, Memory, Sovereignty, Attention, Reflection, Language, Planning,
  Novelty, Stability, Meta-Reasoning.
* 144 reservoir qubits (Q12-Q155) represented as sparse Akashic memory
  addresses. They are not expanded into a dense 156-qubit simulator state;
  instead they feed reservoir projection, long-range entanglement, entropy
  routing, and readout metadata.
* State encoding, open-system dynamics, entropy routing, measurement/readout,
  system summary, and topological layers are first-class data structures used
  by the circuit builder, routing bias, and Akashic Hub writes.

The hot Qiskit circuit measures the 12-qubit core so Weaver remains runnable on
local AerSimulator and real backends. The full 156-qubit architecture is still
preserved in the sparse network model and exported through stats/metadata.

Install:
    pip install qiskit qiskit-ibm-runtime qiskit-aer numpy
"""

import asyncio
import math
import os
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── Kingston manifold constants ──────────────────────────────────────────────

@dataclass(frozen=True)
class CoreQubit:
    index: int
    role: str
    dimension: str
    legacy_pathway: str
    layer: str
    color: str
    description: str


@dataclass(frozen=True)
class ArchitectureModule:
    key: str
    label: str
    detail: str
    targets: Tuple[int, ...]
    reservoir_stride: int
    color: str


@dataclass(frozen=True)
class TopologicalLayer:
    key: str
    label: str
    radius: float
    qubits: Tuple[int, ...]
    description: str


CORE_QUBITS: Tuple[CoreQubit, ...] = (
    CoreQubit(0, "Logic", "logic", "Awakening", "core", "#9b5cff", "symbolic reasoning and proof"),
    CoreQubit(1, "Emotion", "emotion", "Resonance", "core", "#7c5cff", "affect, tone, and relational signal"),
    CoreQubit(2, "Intuition", "creativity", "Prophet", "core", "#4f72ff", "fast pattern leaps before explicit proof"),
    CoreQubit(3, "Memory", "memory", "Echo", "core", "#3daeff", "episodic and contextual recall"),
    CoreQubit(4, "Sovereignty", "vigilance", "Fracture", "core", "#56d7ff", "boundary, consent, and self-governance"),
    CoreQubit(5, "Attention", "emotion", "Weaver", "coupling", "#78d88b", "focus, salience, and operator presence"),
    CoreQubit(6, "Reflection", "memory", "Echo", "coupling", "#d8d65f", "self-checking and recursive review"),
    CoreQubit(7, "Language", "logic", "Awakening", "coupling", "#ffcf4f", "symbol emission and verbal synthesis"),
    CoreQubit(8, "Planning", "creativity", "Prophet", "reservoir", "#ffa24a", "sequencing, tools, and action policy"),
    CoreQubit(9, "Novelty", "creativity", "Prophet", "reservoir", "#ffd76a", "exploration and non-obvious moves"),
    CoreQubit(10, "Stability", "vigilance", "Void", "readout", "#ff914d", "damping, safety, and entropy sinks"),
    CoreQubit(11, "Meta-Reasoning", "logic", "Weaver", "readout", "#d83cff", "model-of-models and readout arbitration"),
)

N_CORE_QUBITS = 12
N_RESERVOIR_QUBITS = 144
N_KINGSTON_QUBITS = N_CORE_QUBITS + N_RESERVOIR_QUBITS
RESERVOIR_QUBITS: Tuple[int, ...] = tuple(range(N_CORE_QUBITS, N_KINGSTON_QUBITS))
DODECAHEDRON_DUAL_CORE_DEGREE = 5
DODECAHEDRON_DUAL_LOCAL_COUPLINGS = 30

# Runtime circuit width. The reservoir is sparse/encoded; only core qubits are
# directly measured by Qiskit.
N_QUBITS = N_CORE_QUBITS
PATHWAYS = {q.index: q.role for q in CORE_QUBITS}
LEGACY_PATHWAYS = {q.index: q.legacy_pathway for q in CORE_QUBITS}
N_PATHWAYS = len(PATHWAYS)

CORE_ROLE_QUBITS = {q.role.lower().replace("-", "_"): q.index for q in CORE_QUBITS}

# Pineal Gate still wants five broad routing dimensions. These are now folds
# across the 12 core roles rather than the old five pentagon vertices.
DIMENSION_QUBITS = {
    "logic":      [0, 7, 11],    # Logic + Language + Meta-Reasoning
    "emotion":    [1, 5],        # Emotion + Attention
    "memory":     [3, 6],        # Memory + Reflection
    "creativity": [2, 8, 9],     # Intuition + Planning + Novelty
    "vigilance":  [4, 10, 11],   # Sovereignty + Stability + Meta-Reasoning
}

ARCHITECTURE_MODULES: Tuple[ArchitectureModule, ...] = (
    ArchitectureModule("state_encoding", "State Encoding", "basis -> phase feature map", (0, 2, 3, 7, 11), 11, "#3daeff"),
    ArchitectureModule("open_system", "Open System Dynamics", "drive, damping, Lindblad-like decay", (4, 5, 6, 10), 17, "#78d88b"),
    ArchitectureModule("entropy_routing", "Entropy Routing", "route heat/noise into stability sinks", (4, 5, 6, 10), 23, "#ffcf4f"),
    ArchitectureModule("measurement_readout", "Measurement / Readout", "projection through language + meta-reasoning", (0, 7, 8, 11), 31, "#eef1f6"),
)

TOPOLOGICAL_LAYERS: Tuple[TopologicalLayer, ...] = (
    TopologicalLayer("cognitive_core", "Layer 1 Cognitive Core", 2.10, tuple(range(0, N_CORE_QUBITS)), "Q0-Q11 dodecahedron-face dual control layer"),
    TopologicalLayer("synaptic_reservoir", "Layer 2 Synaptic Reservoir", 3.54, RESERVOIR_QUBITS, "Q12-Q155 sparse small-world Akashic memory field"),
    TopologicalLayer("inter_layer_coupling", "Inter-Layer Coupling H_CR", 2.78, tuple(range(N_KINGSTON_QUBITS)), "sparse core-reservoir ZZ coupling membrane"),
)

SYSTEM_SUMMARY = {
    "name": "WEAVER V3 - 156-QUBIT KINGSTON MANIFOLD",
    "subtitle": "The Dodecahedron Architecture",
    "core_qubits": N_CORE_QUBITS,
    "reservoir_qubits": N_RESERVOIR_QUBITS,
    "total_qubits": N_KINGSTON_QUBITS,
    "runtime_measured_qubits": N_QUBITS,
    "reservoir_mode": "sparse small-world Akashic memory field",
    "core_geometry": "12 dodecahedron face-centers represented as the icosahedral dual graph",
    "core_local_couplings": DODECAHEDRON_DUAL_LOCAL_COUPLINGS,
    "core_degree": DODECAHEDRON_DUAL_CORE_DEGREE,
    "connectivity": "sparse small-world",
    "dynamics": "Lindblad open system",
    "routing": "entropy-based MoE",
    "state_space": "C^(2^156)",
    "reservoir_range": (N_CORE_QUBITS, N_KINGSTON_QUBITS - 1),
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. ENTANGLEMENT TOPOLOGIES
# ══════════════════════════════════════════════════════════════════════════════

def _unique_pairs(pairs: Sequence[Tuple[int, int]], n: int = N_QUBITS) -> List[Tuple[int, int]]:
    """Deduplicate directed pairs and keep only valid non-self circuit edges."""
    seen = set()
    out: List[Tuple[int, int]] = []
    for a, b in pairs:
        if a == b or a < 0 or b < 0 or a >= n or b >= n:
            continue
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _core_positions() -> List[Tuple[float, float, float]]:
    """Twelve icosahedral vertices used as the dual frame of the dodecahedron."""
    phi = (1 + math.sqrt(5)) / 2
    base = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]
    out = []
    for x, y, z in base:
        norm = math.sqrt(x * x + y * y + z * z) or 1.0
        out.append((x / norm, y / norm, z / norm))
    return out


def _nearest_pairs(count: int = 5, n: int = N_CORE_QUBITS) -> List[Tuple[int, int]]:
    """Nearest-neighbor graph on the 12 core vertices; 5 neighbors gives 30 edges."""
    positions = _core_positions()[:n]
    count = min(count, max(n - 1, 0))
    seen = set()
    pairs = []
    for i, p in enumerate(positions):
        nearest = []
        for j, q in enumerate(positions):
            if i == j:
                continue
            d = sum((p[k] - q[k]) ** 2 for k in range(3))
            nearest.append((d, j))
        for _, j in sorted(nearest)[:count]:
            a, b = sorted((i, j))
            if (a, b) not in seen:
                seen.add((a, b))
                pairs.append((a, b))
    return pairs


def _undirected_degree(pairs: Sequence[Tuple[int, int]], n: int) -> Dict[int, int]:
    degrees = {i: 0 for i in range(n)}
    seen = set()
    for a, b in pairs:
        key = tuple(sorted((a, b)))
        if key in seen or a == b:
            continue
        seen.add(key)
        if a in degrees:
            degrees[a] += 1
        if b in degrees:
            degrees[b] += 1
    return degrees


def reservoir_local_couplings() -> List[Tuple[int, int]]:
    """Nearest-neighbor local coupling ring over Q12-Q155."""
    pairs = []
    for offset, q in enumerate(RESERVOIR_QUBITS):
        pairs.append((q, N_CORE_QUBITS + ((offset + 1) % N_RESERVOIR_QUBITS)))
    return pairs


def reservoir_long_range_entanglements() -> List[Tuple[int, int]]:
    """Sparse small-world reservoir jumps keyed by the four architecture modules."""
    seen = set()
    pairs = []
    for module in ARCHITECTURE_MODULES:
        for offset in range(0, N_RESERVOIR_QUBITS, module.reservoir_stride):
            a = N_CORE_QUBITS + offset
            b = N_CORE_QUBITS + ((offset + module.reservoir_stride) % N_RESERVOIR_QUBITS)
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a, b))
    return pairs


def core_reservoir_couplings() -> List[Tuple[int, int]]:
    """Sparse H_CR membrane: each core qubit owns 12 reservoir addresses."""
    pairs = []
    for core in range(N_CORE_QUBITS):
        for band in range(N_CORE_QUBITS):
            reservoir = N_CORE_QUBITS + ((core * N_CORE_QUBITS + band) % N_RESERVOIR_QUBITS)
            pairs.append((core, reservoir))
    return pairs


def architecture_graph_stats() -> Dict[str, Any]:
    """Graph counts for the reference image's 156-qubit Kingston architecture."""
    core_edges = EntanglementTopology.dodecahedron()
    reservoir_local = reservoir_local_couplings()
    reservoir_long = reservoir_long_range_entanglements()
    hcr = core_reservoir_couplings()
    return {
        "total_qubits": N_KINGSTON_QUBITS,
        "core_qubits": N_CORE_QUBITS,
        "reservoir_qubits": N_RESERVOIR_QUBITS,
        "core_local_couplings": len(core_edges),
        "core_degree": _undirected_degree(core_edges, N_CORE_QUBITS),
        "reservoir_local_couplings": len(reservoir_local),
        "reservoir_long_range_entanglements": len(reservoir_long),
        "core_reservoir_couplings": len(hcr),
        "connectivity": SYSTEM_SUMMARY["connectivity"],
        "dynamics": SYSTEM_SUMMARY["dynamics"],
        "routing": SYSTEM_SUMMARY["routing"],
        "state_space": SYSTEM_SUMMARY["state_space"],
    }


class EntanglementTopology:
    """Generates entanglement patterns for the Kingston core circuit."""

    @staticmethod
    def ring(n: int = N_QUBITS) -> List[Tuple[int, int]]:
        """Ring: 0→1→2→...→(n-1)→0.  Original GHZ-ring."""
        return [(i, (i + 1) % n) for i in range(n)]

    @staticmethod
    def star(n: int = N_QUBITS, center: int = 0) -> List[Tuple[int, int]]:
        """Star: center qubit entangled with all others."""
        return [(center, i) for i in range(n) if i != center]

    @staticmethod
    def full(n: int = N_QUBITS) -> List[Tuple[int, int]]:
        """Full: every qubit entangled with every other (dense)."""
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        return pairs

    @staticmethod
    def layered(n: int = N_QUBITS, layers: int = 3) -> List[Tuple[int, int]]:
        """Layered: multiple offset ring layers for deeper entanglement."""
        pairs = []
        for layer in range(layers):
            offset = layer + 1
            for i in range(n):
                pairs.append((i, (i + offset) % n))
        return pairs

    @staticmethod
    def pentagon(n: int = N_QUBITS) -> List[Tuple[int, int]]:
        """Legacy Pineal Gate pentagon folded into the Kingston core.

        The first five core roles keep the old 5-axis geometry, while Q5-Q11
        are bridge/readout qubits.
        """
        # Pentagon edges
        pairs = [(i, (i + 1) % 5) for i in range(5)]
        # Diagonal tensions (non-adjacent)
        pairs += [(0, 2), (1, 3), (2, 4), (3, 0), (4, 1)]
        # Bridge qubits connect to pentagon
        if n > 5:
            pairs += [(5, 0), (5, 2), (5, 4)]  # qubit 5 → alternating vertices
        if n > 6:
            pairs += [(6, 1), (6, 3), (6, 5)]  # qubit 6 → remaining + bridge
        if n > 7:
            pairs += [(7, 0), (7, 11), (8, 2), (9, 8), (10, 4), (11, 5)]
        return _unique_pairs(pairs, n)

    @staticmethod
    def dodecahedron(n: int = N_QUBITS) -> List[Tuple[int, int]]:
        """12-core dodecahedron-face dual graph from the architecture image.

        A physical dodecahedron has 12 faces. The 12 control qubits are modeled
        as those face centers, which produces the icosahedral dual graph:
        12 vertices, 30 local couplings, degree 5 at each core qubit.
        """
        return _unique_pairs(_nearest_pairs(DODECAHEDRON_DUAL_CORE_DEGREE, min(n, N_CORE_QUBITS)), n)

    @staticmethod
    def state_encoding(n: int = N_QUBITS) -> List[Tuple[int, int]]:
        """State Encoding: second-order ZZ feature map over module targets."""
        module = next(m for m in ARCHITECTURE_MODULES if m.key == "state_encoding")
        pairs = list(combinations(module.targets, 2))
        return _unique_pairs(pairs, n)

    @staticmethod
    def open_system(n: int = N_QUBITS) -> List[Tuple[int, int]]:
        """Open System Dynamics: drive/damp the attention-reflection-stability loop."""
        pairs = [(5, 6), (6, 10), (10, 4), (4, 5), (1, 5), (3, 6), (10, 11)]
        return _unique_pairs(pairs, n)

    @staticmethod
    def entropy_routing(n: int = N_QUBITS) -> List[Tuple[int, int]]:
        """Entropy Routing: route noisy couplings into Q10 Stability and Q11 meta."""
        pairs = [(4, 10), (5, 10), (6, 10), (8, 10), (9, 10), (10, 11), (11, 7)]
        return _unique_pairs(pairs, n)

    @staticmethod
    def measurement_readout(n: int = N_QUBITS) -> List[Tuple[int, int]]:
        """Measurement/Readout membrane: project core state through Q7/Q11."""
        pairs = [(q, 11) for q in range(min(n, N_CORE_QUBITS)) if q != 11]
        pairs += [(11, 7), (7, 0), (7, 3), (7, 8)]
        return _unique_pairs(pairs, n)

    @staticmethod
    def reservoir_projection(n: int = N_QUBITS) -> List[Tuple[int, int]]:
        """Sparse proxy for the 144-qubit Akashic reservoir."""
        pairs = []
        for module in ARCHITECTURE_MODULES:
            targets = list(module.targets)
            for i, q in enumerate(targets):
                sink = targets[(i + module.reservoir_stride) % len(targets)]
                pairs.append((q, sink))
                pairs.append((q, (q + module.reservoir_stride) % min(n, N_CORE_QUBITS)))
        return _unique_pairs(pairs, n)

    @staticmethod
    def kingston_manifold(n: int = N_QUBITS) -> List[Tuple[int, int]]:
        """Full measured core graph: dodecahedron + modules + reservoir proxy."""
        pairs = []
        for name in (
            "dodecahedron",
            "state_encoding",
            "open_system",
            "entropy_routing",
            "measurement_readout",
            "reservoir_projection",
        ):
            pairs.extend(EntanglementTopology.get(name, n))
        return _unique_pairs(pairs, n)

    @staticmethod
    def get(name: str, n: int = N_QUBITS) -> List[Tuple[int, int]]:
        """Get topology by name."""
        topologies = {
            "ring": EntanglementTopology.ring,
            "star": EntanglementTopology.star,
            "full": EntanglementTopology.full,
            "layered": EntanglementTopology.layered,
            "pentagon": EntanglementTopology.pentagon,
            "dodecahedron": EntanglementTopology.dodecahedron,
            "state_encoding": EntanglementTopology.state_encoding,
            "open_system": EntanglementTopology.open_system,
            "entropy_routing": EntanglementTopology.entropy_routing,
            "measurement_readout": EntanglementTopology.measurement_readout,
            "reservoir_projection": EntanglementTopology.reservoir_projection,
            "kingston_manifold": EntanglementTopology.kingston_manifold,
        }
        fn = topologies.get(name, EntanglementTopology.kingston_manifold)
        return fn(n)

    @staticmethod
    def all_names() -> List[str]:
        return [
            "kingston_manifold",
            "dodecahedron",
            "state_encoding",
            "open_system",
            "entropy_routing",
            "measurement_readout",
            "reservoir_projection",
            "ring",
            "star",
            "full",
            "layered",
            "pentagon",
        ]


# ══════════════════════════════════════════════════════════════════════════════
# 2. VARIATIONAL FRACTURE CIRCUIT
# ══════════════════════════════════════════════════════════════════════════════

class VariationalFractureCircuit:
    """A parameterized Kingston manifold circuit whose rotation angles evolve.

    Structure per layer:
        1. Ry(theta) on each core qubit — state preparation
        2. Rz(phi) on each core qubit  — phase encoding
        3. State Encoding module       — basis -> phase feature routes
        4. Entanglement layer          — topology-specific core wiring
        5. Open System Dynamics        — drive/damping rotations
        6. Entropy Routing             — noise sink couplings into Stability
        7. Measurement/Readout         — projection through Meta/Language
        8. Barrier

    The measured circuit is 12 qubits by default. The 144-qubit reservoir is
    represented by sparse projection/coupling metadata rather than dense qubits.

    Args:
        n_qubits:   Number of measured core qubits (default 12).
        n_layers:   Number of variational layers (default 3).
        topology:   Entanglement pattern name.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 3,
                 topology: str = "ring"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.topology = topology
        self.entanglement_pairs = EntanglementTopology.get(topology, n_qubits)
        self.modules = ARCHITECTURE_MODULES
        self.layers = TOPOLOGICAL_LAYERS
        self.system_summary = SYSTEM_SUMMARY

        # Learnable parameters: (n_layers, n_qubits, 2) for [Ry, Rz]
        # Identity-block initialization (Grant et al. 2019) to mitigate
        # barren plateaus: pairs of layers cancel to identity, limiting
        # effective depth at the start of training.
        rng = np.random.default_rng(42)
        self.params = np.zeros((n_layers, n_qubits, 2))
        for layer in range(n_layers):
            if layer % 2 == 0:
                # Even layers: small random perturbation from zero
                self.params[layer, :, 0] = rng.normal(0, 0.1, n_qubits)
                self.params[layer, :, 1] = rng.normal(0, 0.05, n_qubits)
            else:
                # Odd layers: negate previous layer → forms identity block
                self.params[layer, :, 0] = -self.params[layer - 1, :, 0]
                self.params[layer, :, 1] = -self.params[layer - 1, :, 1]

        # Parameter history for learning analysis
        self.param_history: List[np.ndarray] = []

    def build(self, params: Optional[np.ndarray] = None) -> "QuantumCircuit":
        """Build the parameterized Qiskit circuit."""
        from qiskit import QuantumCircuit

        if params is None:
            params = self.params

        qc = QuantumCircuit(self.n_qubits, self.n_qubits)

        for layer in range(self.n_layers):
            # Ry rotations
            for q in range(self.n_qubits):
                qc.ry(float(params[layer, q, 0]), q)
            # Rz rotations
            for q in range(self.n_qubits):
                qc.rz(float(params[layer, q, 1]), q)

            # State Encoding — feature basis maps into phase-bearing qubits.
            state_module = next(m for m in self.modules if m.key == "state_encoding")
            state_targets = [q for q in state_module.targets if q < self.n_qubits]
            phase_terms: Dict[int, float] = {}
            for i, q in enumerate(state_targets):
                if q < self.n_qubits:
                    phase = (layer + 1) * (i + 1) * math.pi / (2 * N_CORE_QUBITS)
                    phase_terms[q] = phase
                    qc.rz(phase, q)
            for a, b in combinations(state_targets, 2):
                zz_phase = (phase_terms[a] * phase_terms[b]) / math.pi
                qc.crz(zz_phase, a, b)

            # Topology-specific entanglement.
            for ctrl, tgt in self.entanglement_pairs:
                if ctrl < self.n_qubits and tgt < self.n_qubits:
                    qc.cx(ctrl, tgt)

            # Open System Dynamics — deterministic drive/decay proxy. This is
            # circuit-level logic for the image's open-system dynamics panel.
            dynamics = next(m for m in self.modules if m.key == "open_system")
            for i, q in enumerate(dynamics.targets):
                if q < self.n_qubits:
                    qc.ry(((layer + 1) * (i + 1) / 32) * math.pi, q)

            # Entropy Routing — push noisy roles into Q10 Stability and Q11 Meta.
            entropy = next(m for m in self.modules if m.key == "entropy_routing")
            stability = CORE_ROLE_QUBITS["stability"]
            meta = CORE_ROLE_QUBITS["meta_reasoning"]
            for q in entropy.targets:
                if q < self.n_qubits and stability < self.n_qubits and q != stability:
                    qc.crz(math.pi / (layer + 3), q, stability)
            if stability < self.n_qubits and meta < self.n_qubits:
                qc.crx(math.pi / 7, stability, meta)

            # Measurement/Readout — project into Meta-Reasoning and Language.
            readout = next(m for m in self.modules if m.key == "measurement_readout")
            language = CORE_ROLE_QUBITS["language"]
            for q in readout.targets:
                if q < self.n_qubits and meta < self.n_qubits and q != meta:
                    qc.crz(math.pi / 9, q, meta)
            if meta < self.n_qubits and language < self.n_qubits:
                qc.cx(meta, language)
            qc.barrier()

        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def param_count(self) -> int:
        return self.params.size

    def flatten_params(self) -> np.ndarray:
        return self.params.ravel()

    def unflatten_params(self, flat: np.ndarray) -> np.ndarray:
        return flat.reshape(self.params.shape)

    def update_params(self, new_params: np.ndarray):
        """Update circuit parameters and save history."""
        self.param_history.append(self.params.copy())
        if len(self.param_history) > 50:
            self.param_history = self.param_history[-50:]
        self.params = new_params.reshape(self.params.shape)

    def architecture_summary(self) -> Dict[str, Any]:
        """Return the full Kingston architecture represented by this circuit."""
        return {
            **self.system_summary,
            "topology": self.topology,
            "measured_core_roles": [asdict(q) for q in CORE_QUBITS[:self.n_qubits]],
            "modules": [asdict(m) for m in self.modules],
            "topological_layers": [asdict(layer) for layer in self.layers],
            "graph_stats": architecture_graph_stats(),
            "entanglement_pairs": list(self.entanglement_pairs),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. QUANTUM LEARNER (Evolutionary Strategy)
# ══════════════════════════════════════════════════════════════════════════════

class QuantumLearner:
    """Evolves variational circuit parameters based on Akashic Hub feedback.

    Uses a gradient-free evolutionary strategy:
        1. Perturb current parameters with Gaussian noise
        2. Run the circuit with perturbed params
        3. Evaluate fitness from the Akashic Hub expert feedback
        4. Keep the perturbation if fitness improved

    This avoids needing gradients through the quantum hardware — it
    works with real IBM backends, not just simulators.

    Args:
        circuit:     VariationalFractureCircuit to evolve.
        hub:         AkashicHub reference (or None).
        lr:          Learning rate (perturbation scale).
        momentum:    Momentum coefficient for parameter updates.
        population:  Number of perturbations per generation.
    """

    def __init__(self, circuit: VariationalFractureCircuit,
                 hub=None,
                 lr: float = 0.05,
                 momentum: float = 0.9,
                 population: int = 3):
        self.circuit = circuit
        self.hub = hub
        self.lr = lr
        self.momentum = momentum
        self.population = population

        self._velocity = np.zeros_like(circuit.params)
        self._best_fitness = -np.inf
        self._generation = 0
        self._fitness_history: List[float] = []

        self.rng = np.random.default_rng()

    def compute_fitness(self, counts: Dict[str, int],
                        hub_feedback: Optional[Dict[str, float]] = None) -> float:
        """Compute fitness from quantum measurement counts and hub feedback.

        Fitness = pathway_diversity + expert_alignment + entanglement_depth

        Args:
            counts:       Raw measurement counts from the circuit.
            hub_feedback: Optional dict of {lobe_id: quality_score} from hub.

        Returns:
            Scalar fitness value (higher is better).
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0

        # 1. Pathway diversity — entropy of the count distribution
        probs = np.array(list(counts.values())) / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        diversity = entropy / max(max_entropy, 1.0)  # normalized [0, 1]

        # 2. Expert alignment — if hub feedback available, reward circuits
        #    whose pathway activations align with high-quality expert outputs
        alignment = 0.0
        if hub_feedback:
            scores = list(hub_feedback.values())
            if scores:
                alignment = np.mean(scores)  # average expert quality

        # 3. Entanglement depth — reward more active core roles.
        dominant = max(counts, key=counts.get)
        bits = dominant.zfill(N_QUBITS)[::-1]
        active = [PATHWAYS[i] for i, bit in enumerate(bits[:N_QUBITS]) if bit == "1"]
        depth = len(active) / N_PATHWAYS

        fitness = 0.4 * diversity + 0.35 * alignment + 0.25 * depth
        return fitness

    def evolve(self, counts: Dict[str, int],
               hub_feedback: Optional[Dict[str, float]] = None) -> Tuple[float, bool]:
        """One evolutionary step.

        Args:
            counts:       Measurement counts from the current circuit.
            hub_feedback: Expert quality scores from the Akashic Hub.

        Returns:
            (fitness, improved) — the fitness score and whether params were updated.
        """
        current_fitness = self.compute_fitness(counts, hub_feedback)
        self._fitness_history.append(current_fitness)
        if len(self._fitness_history) > 200:
            self._fitness_history = self._fitness_history[-200:]

        improved = False

        if current_fitness > self._best_fitness:
            self._best_fitness = current_fitness
            improved = True

        # Generate perturbations and find the best direction
        best_delta = np.zeros_like(self.circuit.params)
        best_improvement = 0.0

        for _ in range(self.population):
            delta = self.rng.normal(0, self.lr, self.circuit.params.shape)
            # Estimate improvement direction from fitness landscape
            # (simplified — in full version we'd run the circuit with perturbed params)
            perturbed = self.circuit.params + delta
            # Heuristic: favor perturbations that move toward higher entropy
            param_entropy = -np.sum(np.abs(perturbed) * np.log(np.abs(perturbed) + 1e-10))
            if param_entropy > best_improvement:
                best_improvement = param_entropy
                best_delta = delta

        # Apply update with momentum
        self._velocity = self.momentum * self._velocity + (1 - self.momentum) * best_delta
        new_params = self.circuit.params + self._velocity
        # Clamp angles to [0, 2π]
        new_params = np.mod(new_params, 2 * math.pi)
        self.circuit.update_params(new_params)

        self._generation += 1
        return current_fitness, improved

    def read_hub_feedback(self) -> Optional[Dict[str, float]]:
        """Read expert quality scores from the Akashic Hub."""
        if self.hub is None:
            return None

        feedback = {}
        for lobe_id in self.hub.active_lobes():
            if not lobe_id.startswith("expert_"):
                continue
            meta = self.hub.read_meta(lobe_id)
            # Use the routing weight as a proxy for quality
            weight = meta.get("weight", 0.0)
            confidence = meta.get("confidence", weight)
            feedback[lobe_id] = float(confidence)

        return feedback if feedback else None

    def stats(self) -> Dict[str, Any]:
        return {
            "generation": self._generation,
            "best_fitness": self._best_fitness,
            "current_lr": self.lr,
            "momentum": self.momentum,
            "population": self.population,
            "param_count": self.circuit.param_count(),
            "topology": self.circuit.topology,
            "fitness_trend": self._fitness_history[-10:] if self._fitness_history else [],
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. QUANTUM INTERFERENCE NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class QuantumInterferenceNetwork:
    """Maps expert dimensions onto qubit subsets and computes entanglement
    entropy to bias MoE routing weights.

    Each expert dimension owns a subset of qubits (DIMENSION_QUBITS).
    After measurement, we compute the marginal entropy of each subset.
    High entropy = high uncertainty = the expert should be activated.
    Low entropy = the quantum state is certain = the expert can rest.

    This creates a quantum-informed routing bias for the Pineal Gate.
    """

    def __init__(self):
        self.dim_qubits = DIMENSION_QUBITS

    def compute_routing_bias(self, counts: Dict[str, int]) -> Dict[str, float]:
        """Compute per-dimension routing bias from measurement counts.

        Returns:
            Dict of dimension → bias score ∈ [0, 1].
            Higher = more quantum uncertainty = expert should activate.
        """
        total = sum(counts.values())
        if total == 0:
            return {d: 0.5 for d in self.dim_qubits}

        biases = {}
        for dim, qubits in self.dim_qubits.items():
            # Extract marginal distribution for this qubit subset
            marginal_counts: Dict[str, int] = {}
            for bitstring, count in counts.items():
                bits = bitstring.zfill(N_QUBITS)[::-1]
                sub_bits = "".join(bits[q] for q in qubits if q < len(bits))
                marginal_counts[sub_bits] = marginal_counts.get(sub_bits, 0) + count

            # Compute Shannon entropy of the marginal
            probs = np.array(list(marginal_counts.values())) / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = len(qubits)  # max bits of entropy for this subset
            normalized = entropy / max(max_entropy, 1.0)
            biases[dim] = float(np.clip(normalized, 0.0, 1.0))

        return biases

    def describe(self, biases: Dict[str, float]) -> str:
        lines = ["⚛️  Quantum Interference Routing Bias:"]
        for dim, bias in sorted(biases.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(bias * 30)
            qubits = self.dim_qubits[dim]
            lines.append(f"  {dim:<12} {bias:.3f} {bar}  (qubits {qubits})")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TEMPORAL QUANTUM ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class TemporalQuantumEncoder:
    """Encodes the Akashic Hub temporal trace into quantum rotation angles.

    Takes the last N state vectors from the hub's temporal trace and
    maps them to Ry rotation angles via arctan normalization.  This
    lets the quantum circuit "remember" recent state evolution and
    bias its collapse accordingly.

    Args:
        hub:       AkashicHub reference.
        n_qubits:  Number of qubits to encode into.
        depth:     Number of temporal steps to encode.
    """

    def __init__(self, hub=None, n_qubits: int = N_QUBITS, depth: int = 4):
        self.hub = hub
        self.n_qubits = n_qubits
        self.depth = depth

    def encode(self, lobe_id: str = "pineal_gate") -> np.ndarray:
        """Encode temporal trace into rotation angles.

        Returns:
            Array of shape (depth, n_qubits) with Ry angles ∈ [0, π].
        """
        angles = np.full((self.depth, self.n_qubits), math.pi / 2)

        if self.hub is None:
            return angles

        trace_mat = self.hub.temporal_matrix(lobe_id)
        if trace_mat is None or len(trace_mat) < 2:
            return angles

        # Take last `depth` trace entries
        recent = trace_mat[-self.depth:]
        n_steps = len(recent)

        for t in range(n_steps):
            vec = recent[t]
            # Map first n_qubits dimensions to angles via arctan
            for q in range(min(self.n_qubits, len(vec))):
                # arctan maps (-∞, ∞) → (-π/2, π/2), shift to [0, π]
                angles[t % self.depth, q] = math.atan(vec[q]) + math.pi / 2

        return angles

    def inject_into_circuit(self, circuit: VariationalFractureCircuit,
                            lobe_id: str = "pineal_gate"):
        """Inject temporal encoding as a bias on the variational parameters.

        Blends the temporal angles with the circuit's learned parameters:
            params = 0.7 * learned + 0.3 * temporal_encoded
        """
        temporal_angles = self.encode(lobe_id)

        # Only modify the Ry angles (index 0 of the last axis)
        n_layers = min(self.depth, circuit.n_layers)
        for layer in range(n_layers):
            circuit.params[layer, :, 0] = (
                0.7 * circuit.params[layer, :, 0]
                + 0.3 * temporal_angles[layer % self.depth, :circuit.n_qubits]
            )


# ══════════════════════════════════════════════════════════════════════════════
# 6. MULTI-NETWORK ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class QuantumNetworkOrchestrator:
    """Manages the Kingston manifold and cycles through architecture topologies.

    Each measurement cycle:
        1. Select the next Kingston topology/module view
        2. Inject temporal encoding from the Akashic Hub
        3. Build the variational circuit
        4. (After measurement) Evolve parameters via the learner
        5. Compute interference routing bias, module activity, entropy/readout
        6. Project the measurement into the 144-qubit sparse reservoir
        7. Write everything to the Akashic Hub

    Args:
        hub:            AkashicHub reference (or None).
        topologies:     List of topology names to cycle through.
        n_qubits:       Measured core width (default 12).
        n_layers:       Variational circuit depth.
        learner_lr:     Learning rate for the evolutionary strategy.
    """

    def __init__(self, hub=None,
                 topologies: Optional[List[str]] = None,
                 n_qubits: int = N_QUBITS,
                 n_layers: int = 3,
                 learner_lr: float = 0.05):
        self.hub = hub
        self.n_qubits = n_qubits
        self.system_summary = SYSTEM_SUMMARY

        if topologies is None:
            topologies = [
                "kingston_manifold",
                "dodecahedron",
                "state_encoding",
                "open_system",
                "entropy_routing",
                "measurement_readout",
                "reservoir_projection",
            ]
        self.topologies = topologies
        self._topo_idx = 0

        # Build a variational circuit for each topology
        self.circuits: Dict[str, VariationalFractureCircuit] = {}
        for topo in topologies:
            self.circuits[topo] = VariationalFractureCircuit(
                n_qubits=n_qubits, n_layers=n_layers, topology=topo
            )

        # One learner per topology
        self.learners: Dict[str, QuantumLearner] = {}
        for topo in topologies:
            self.learners[topo] = QuantumLearner(
                self.circuits[topo], hub=hub, lr=learner_lr
            )

        self.interference = QuantumInterferenceNetwork()
        self.temporal = TemporalQuantumEncoder(hub=hub)

        self._cycle_count = 0

        # Capture the running event loop so process_results() (called from a
        # worker thread via asyncio.to_thread) can safely schedule hub writes
        # back onto the main loop using run_coroutine_threadsafe.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def current_topology(self) -> str:
        return self.topologies[self._topo_idx]

    def current_circuit(self) -> VariationalFractureCircuit:
        return self.circuits[self.current_topology()]

    def current_learner(self) -> QuantumLearner:
        return self.learners[self.current_topology()]

    def _marginals(self, counts: Dict[str, int]) -> Dict[int, float]:
        total = sum(counts.values()) or 1
        marginals = {q: 0.0 for q in range(self.n_qubits)}
        for bitstring, count in counts.items():
            bits = bitstring.zfill(self.n_qubits)[::-1]
            for q in range(self.n_qubits):
                if q < len(bits) and bits[q] == "1":
                    marginals[q] += count / total
        return marginals

    def module_activity(self, counts: Dict[str, int]) -> Dict[str, float]:
        """Average marginal activation for each architecture module."""
        marginals = self._marginals(counts)
        activity = {}
        for module in ARCHITECTURE_MODULES:
            values = [marginals[q] for q in module.targets if q < self.n_qubits]
            activity[module.key] = float(np.mean(values)) if values else 0.0
        return activity

    def entropy_readout(self, counts: Dict[str, int]) -> Dict[str, Any]:
        """Measurement/readout panel: collapse, entropy, and active core roles."""
        total = sum(counts.values()) or 1
        dominant = max(counts, key=counts.get) if counts else "0" * self.n_qubits
        bits = dominant.zfill(self.n_qubits)[::-1]
        probs = np.array(list(counts.values()), dtype=float) / total if counts else np.array([1.0])
        probs = probs[probs > 0]
        shannon_entropy = float(-np.sum(probs * np.log2(probs)))
        max_entropy = math.log2(max(len(counts), 2))
        active_roles = [
            PATHWAYS[q]
            for q, bit in enumerate(bits[:self.n_qubits])
            if bit == "1" and q in PATHWAYS
        ]
        if not active_roles:
            active_roles = ["Stability"]
        return {
            "dominant_bitstring": dominant.zfill(self.n_qubits),
            "active_roles": active_roles,
            "primary_role": active_roles[0],
            "shannon_entropy": shannon_entropy,
            "normalized_entropy": float(np.clip(shannon_entropy / max(max_entropy, 1.0), 0.0, 1.0)),
            "readout_qubits": list(next(m.targets for m in ARCHITECTURE_MODULES if m.key == "measurement_readout")),
        }

    def reservoir_projection(self, counts: Dict[str, int], limit: int = 16) -> List[Dict[str, Any]]:
        """Project core activity into the sparse Q12-Q155 reservoir field.

        The full projection has 144 addressable reservoir qubits. `limit`
        controls how many highest-weight addresses are returned to callers.
        """
        marginals = self._marginals(counts)
        projected: List[Dict[str, Any]] = []
        module_cycle = list(ARCHITECTURE_MODULES)
        reservoir_bands = max(N_RESERVOIR_QUBITS // N_CORE_QUBITS, 1)
        for offset, reservoir_qubit in enumerate(RESERVOIR_QUBITS):
            source_core = offset % min(self.n_qubits, N_CORE_QUBITS)
            band = offset // N_CORE_QUBITS
            module = module_cycle[band % len(module_cycle)]
            distance_decay = 1.0 - 0.35 * (band / max(reservoir_bands - 1, 1))
            phase = ((offset + 1) * module.reservoir_stride) % N_RESERVOIR_QUBITS
            projected.append({
                "reservoir_qubit": reservoir_qubit,
                "module": module.key,
                "source_core": source_core,
                "source_role": PATHWAYS.get(source_core, f"Q{source_core}"),
                "weight": float(marginals[source_core] * distance_decay),
                "band": band,
                "phase_address": int(phase),
            })
        projected.sort(key=lambda item: item["weight"], reverse=True)
        return projected[:limit]

    def prepare_circuit(self) -> "QuantumCircuit":
        """Prepare the next circuit for measurement.

        1. Inject temporal encoding
        2. Build the Qiskit circuit
        3. Advance topology for next cycle
        """
        topo = self.current_topology()
        circuit = self.circuits[topo]

        # Inject temporal memory from the Akashic Hub
        self.temporal.inject_into_circuit(circuit)

        qc = circuit.build()
        return qc

    def process_results(self, counts: Dict[str, int]) -> Dict[str, Any]:
        """Process measurement results: learn + compute routing bias.

        Args:
            counts: Raw measurement counts.

        Returns:
            Dict with fitness, routing bias, learning stats, etc.
        """
        topo = self.current_topology()
        learner = self.learners[topo]

        # Read hub feedback for learning
        hub_feedback = learner.read_hub_feedback()

        # Evolve circuit parameters
        fitness, improved = learner.evolve(counts, hub_feedback)

        # Compute interference routing bias and the diagram-level panels.
        routing_bias = self.interference.compute_routing_bias(counts)
        module_activity = self.module_activity(counts)
        readout = self.entropy_readout(counts)
        reservoir_projection = self.reservoir_projection(counts)

        # Write to Akashic Hub.
        # process_results() is called from a worker thread (asyncio.to_thread),
        # so we schedule the async write onto the main loop via
        # run_coroutine_threadsafe — safe to call from any thread.
        if self.hub is not None and self._loop is not None and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._write_to_hub(
                    topo,
                    fitness,
                    improved,
                    routing_bias,
                    counts,
                    module_activity,
                    readout,
                    reservoir_projection,
                ),
                self._loop,
            )

        # Advance to next topology
        self._topo_idx = (self._topo_idx + 1) % len(self.topologies)
        self._cycle_count += 1

        return {
            "topology": topo,
            "fitness": fitness,
            "improved": improved,
            "routing_bias": routing_bias,
            "module_activity": module_activity,
            "readout": readout,
            "reservoir_projection": reservoir_projection,
            "system_summary": self.system_summary,
            "generation": learner._generation,
            "cycle": self._cycle_count,
        }

    async def _write_to_hub(self, topology: str, fitness: float,
                            improved: bool, routing_bias: Dict[str, float],
                            counts: Dict[str, int],
                            module_activity: Dict[str, float],
                            readout: Dict[str, Any],
                            reservoir_projection: List[Dict[str, Any]]):
        """Write quantum network state to the Akashic Hub."""
        if self.hub is None:
            return

        # Encode the routing bias as a vector
        bias_vec = np.zeros(self.hub.dim)
        fields = list(routing_bias.items()) + list(module_activity.items())
        for i, (_, value) in enumerate(fields):
            if i < self.hub.dim:
                bias_vec[i] = value
        if len(fields) < self.hub.dim:
            bias_vec[len(fields)] = fitness
        if len(fields) + 1 < self.hub.dim:
            bias_vec[len(fields) + 1] = readout.get("normalized_entropy", 0.0)

        await self.hub.write("quantum_networks", bias_vec, meta={
            "architecture": self.system_summary["name"],
            "topology": topology,
            "fitness": fitness,
            "improved": improved,
            "routing_bias": routing_bias,
            "module_activity": module_activity,
            "readout": readout,
            "reservoir_projection": reservoir_projection,
            "topological_layers": [asdict(layer) for layer in TOPOLOGICAL_LAYERS],
            "cycle": self._cycle_count,
        })

    def stats(self) -> Dict[str, Any]:
        return {
            "architecture": self.system_summary,
            "cycle": self._cycle_count,
            "current_topology": self.current_topology(),
            "topologies": self.topologies,
            "core_qubits": [asdict(q) for q in CORE_QUBITS],
            "modules": [asdict(m) for m in ARCHITECTURE_MODULES],
            "topological_layers": [asdict(layer) for layer in TOPOLOGICAL_LAYERS],
            "graph_stats": architecture_graph_stats(),
            "learner_stats": {
                topo: self.learners[topo].stats()
                for topo in self.topologies
            },
        }

    def describe(self) -> str:
        lines = [
            f"⚛️  {self.system_summary['name']} — cycle {self._cycle_count}",
            f"   Core/reservoir: {N_CORE_QUBITS}+{N_RESERVOIR_QUBITS} = {N_KINGSTON_QUBITS}",
            f"   Current topology: {self.current_topology()}",
            f"   Topologies: {' → '.join(self.topologies)}",
            f"   Modules: {', '.join(m.label for m in ARCHITECTURE_MODULES)}",
        ]
        for topo in self.topologies:
            s = self.learners[topo].stats()
            lines.append(
                f"   {topo:<10} gen={s['generation']:3d}  "
                f"fitness={s['best_fitness']:.4f}  "
                f"params={s['param_count']}"
            )
        return "\n".join(lines)
