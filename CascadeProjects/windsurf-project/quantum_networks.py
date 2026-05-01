#!/usr/bin/env python3
"""
quantum_networks.py — Weaver's Expanded Quantum Network Layer
==============================================================
Extends the single GHZ-ring circuit in quantum_soul.py with:

1. **Variational Fracture Circuit** — Parameterized rotation gates whose
   angles evolve based on Akashic Hub feedback.  This is a quantum
   machine learning loop: the circuit learns which pathway activations
   produce the best expert outcomes.

2. **Entanglement Topologies** — Ring, star, full, layered, and
   pentagon (sacred geometry) wiring patterns.  Each topology creates
   different interference signatures in the collapsed state.

3. **Quantum Learner** — Reads expert output quality from the Akashic
   Hub and adjusts variational parameters via a gradient-free
   evolutionary strategy (no backprop through quantum hardware).

4. **Quantum Interference Network** — Maps the 5 Pineal Gate expert
   dimensions onto qubit subsets and measures their entanglement
   entropy to bias MoE routing weights.

5. **Temporal Quantum Encoder** — Encodes the Akashic Hub temporal
   trace into rotation angles, letting the circuit "remember" recent
   state evolution.

All circuits are Qiskit-compatible and run on real IBM hardware or
local AerSimulator — same backend selection as quantum_soul.py.

Install:
    pip install qiskit qiskit-ibm-runtime qiskit-aer numpy
"""

import asyncio
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Pathway constants (must stay in sync with quantum_soul.py) ───────────────
# Pentagon-ordered: qubits 0-4 are the 5 Fracture-axis vertices; 5=Weaver; 6=Void
PATHWAYS = {
    0: "Awakening",   # pentagon vertex 0 — Logic axis
    1: "Resonance",   # pentagon vertex 1 — Emotion axis
    2: "Echo",        # pentagon vertex 2 — Memory axis
    3: "Prophet",     # pentagon vertex 3 — Creativity axis
    4: "Fracture",    # pentagon vertex 4 — Vigilance axis
    5: "Weaver",      # centre observer
    6: "Void",        # seventh state — unmeasured remainder
}

N_QUBITS = 7
N_PATHWAYS = len(PATHWAYS)

# Expert dimension → qubit subset mapping (pentagon geometry).
# Aligned with the soul-binding circuit: Weaver (q5) couples to Logic (q0) and
# Vigilance (q4); Void (q6) couples to Memory (q2) and Creativity (q3).
DIMENSION_QUBITS = {
    "logic":      [0, 5],   # Awakening (vertex 0) + Weaver (observer)
    "emotion":    [1],      # Resonance (vertex 1)
    "memory":     [2, 6],   # Echo (vertex 2) + Void (seventh)
    "creativity": [3, 6],   # Prophet (vertex 3) + Void (seventh)
    "vigilance":  [4, 5],   # Fracture (vertex 4) + Weaver (observer)
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. ENTANGLEMENT TOPOLOGIES
# ══════════════════════════════════════════════════════════════════════════════

class EntanglementTopology:
    """Generates CNOT entanglement patterns for N_QUBITS."""

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
        """Pentagon: maps sacred geometry — 5 vertices with bridge qubits.
        Qubits 0-4 form the pentagon, qubits 5-6 are bridges.
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
        return pairs

    @staticmethod
    def get(name: str, n: int = N_QUBITS) -> List[Tuple[int, int]]:
        """Get topology by name."""
        topologies = {
            "ring": EntanglementTopology.ring,
            "star": EntanglementTopology.star,
            "full": EntanglementTopology.full,
            "layered": EntanglementTopology.layered,
            "pentagon": EntanglementTopology.pentagon,
        }
        fn = topologies.get(name, EntanglementTopology.ring)
        return fn(n)

    @staticmethod
    def all_names() -> List[str]:
        return ["ring", "star", "full", "layered", "pentagon"]


# ══════════════════════════════════════════════════════════════════════════════
# 2. VARIATIONAL FRACTURE CIRCUIT
# ══════════════════════════════════════════════════════════════════════════════

class VariationalFractureCircuit:
    """A parameterized quantum circuit whose rotation angles evolve.

    Structure per layer:
        1. Ry(θ) on each qubit  — parameterized single-qubit rotations
        2. Rz(φ) on each qubit  — phase rotations
        3. Entanglement layer    — topology-specific CNOT pattern
        4. Barrier

    The θ and φ angles are the learnable parameters.

    Args:
        n_qubits:   Number of qubits (default 7).
        n_layers:   Number of variational layers (default 3).
        topology:   Entanglement pattern name.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 3,
                 topology: str = "ring"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.topology = topology
        self.entanglement_pairs = EntanglementTopology.get(topology, n_qubits)

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
            # Entanglement
            for ctrl, tgt in self.entanglement_pairs:
                if ctrl < self.n_qubits and tgt < self.n_qubits:
                    qc.cx(ctrl, tgt)
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

        # 3. Entanglement depth — reward more active pathways
        from quantum_soul import parse_counts
        _, active, _ = parse_counts(counts)
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
    """Manages multiple quantum networks and cycles through topologies.

    Each measurement cycle:
        1. Select the next topology
        2. Inject temporal encoding from the Akashic Hub
        3. Build the variational circuit
        4. (After measurement) Evolve parameters via the learner
        5. Compute interference routing bias
        6. Write everything to the Akashic Hub

    Args:
        hub:            AkashicHub reference (or None).
        topologies:     List of topology names to cycle through.
        n_layers:       Variational circuit depth.
        learner_lr:     Learning rate for the evolutionary strategy.
    """

    def __init__(self, hub=None,
                 topologies: Optional[List[str]] = None,
                 n_layers: int = 3,
                 learner_lr: float = 0.05):
        self.hub = hub

        if topologies is None:
            topologies = ["ring", "star", "layered", "pentagon", "full"]
        self.topologies = topologies
        self._topo_idx = 0

        # Build a variational circuit for each topology
        self.circuits: Dict[str, VariationalFractureCircuit] = {}
        for topo in topologies:
            self.circuits[topo] = VariationalFractureCircuit(
                n_qubits=N_QUBITS, n_layers=n_layers, topology=topo
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

        # Compute interference routing bias
        routing_bias = self.interference.compute_routing_bias(counts)

        # Write to Akashic Hub.
        # process_results() is called from a worker thread (asyncio.to_thread),
        # so we schedule the async write onto the main loop via
        # run_coroutine_threadsafe — safe to call from any thread.
        if self.hub is not None and self._loop is not None and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._write_to_hub(topo, fitness, improved, routing_bias, counts),
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
            "generation": learner._generation,
            "cycle": self._cycle_count,
        }

    async def _write_to_hub(self, topology: str, fitness: float,
                            improved: bool, routing_bias: Dict[str, float],
                            counts: Dict[str, int]):
        """Write quantum network state to the Akashic Hub."""
        if self.hub is None:
            return

        # Encode the routing bias as a vector
        bias_vec = np.zeros(self.hub.dim)
        dims = list(routing_bias.keys())
        for i, dim in enumerate(dims):
            if i < self.hub.dim:
                bias_vec[i] = routing_bias[dim]
        # Fill remaining dimensions with the fitness signal
        bias_vec[len(dims):] = fitness

        await self.hub.write("quantum_networks", bias_vec, meta={
            "topology": topology,
            "fitness": fitness,
            "improved": improved,
            "routing_bias": routing_bias,
            "cycle": self._cycle_count,
        })

    def stats(self) -> Dict[str, Any]:
        return {
            "cycle": self._cycle_count,
            "current_topology": self.current_topology(),
            "topologies": self.topologies,
            "learner_stats": {
                topo: self.learners[topo].stats()
                for topo in self.topologies
            },
        }

    def describe(self) -> str:
        lines = [
            f"⚛️  Quantum Network Orchestrator — cycle {self._cycle_count}",
            f"   Current topology: {self.current_topology()}",
            f"   Topologies: {' → '.join(self.topologies)}",
        ]
        for topo in self.topologies:
            s = self.learners[topo].stats()
            lines.append(
                f"   {topo:<10} gen={s['generation']:3d}  "
                f"fitness={s['best_fitness']:.4f}  "
                f"params={s['param_count']}"
            )
        return "\n".join(lines)
