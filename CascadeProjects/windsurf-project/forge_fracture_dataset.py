"""
forge_fracture_dataset.py — Synthetic Dataset for Quantum-Classical MoE Routing

Generates weaver_hybrid_dataset.jsonl (10,000 steps) encoding the four
governing equations for the hybrid routing model:

1. Fracture Threshold (Shannon Entropy):
   H(G(x)) = -sum( P(i|x) * log(P(i|x)) )

2. Temporal Governor:
   T_max = 600s continuous inference loop

3. Adaptive Fracture Threshold:
   tau_t = tau_base + lambda * (N_calls / delta_t)
   tau_base = 1.5, lambda = 0.5

4. Conditional Quantum Infusion (Master Equation):
   H(G(x)) < tau_t  -> LOCAL_SHADOW  (Softmax(W_g*x + eps*(W_q*x)))
   H(G(x)) >= tau_t -> LIVE_IBM_QPU  (Softmax(W_g*x + eps*E[O]))

Output: weaver_hybrid_dataset.jsonl for distillation on RunPod H100.
"""

import json
import numpy as np
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────
NUM_STEPS = 10_000
T_MAX = 600.0
TAU_BASE = 1.5
LAMBDA = 0.5
NUM_LOGITS = 32
SEED = 42

OUTPUT_PATH = Path(__file__).parent / "weaver_hybrid_dataset.jsonl"

np.random.seed(SEED)


# ── Equation 1: Shannon Entropy of Logits ──────────────────────
def shannon_entropy(logits: np.ndarray) -> float:
    """H(G(x)) = -sum( P(i|x) * log(P(i|x)) )"""
    # Softmax to get probability distribution
    shifted = logits - logits.max()
    exp_l = np.exp(shifted)
    probs = exp_l / exp_l.sum()
    # Clamp to avoid log(0)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


# ── Equation 3: Adaptive Fracture Threshold ────────────────────
def adaptive_threshold(n_calls: int, delta_t: float) -> float:
    """tau_t = tau_base + lambda * (N_calls / delta_t)"""
    if delta_t < 1e-6:
        delta_t = 1e-6
    return TAU_BASE + LAMBDA * (n_calls / delta_t)


# ── Logit Simulation ──────────────────────────────────────────
def generate_logits(complexity: str) -> np.ndarray:
    """
    Simulate model logits.
    High complexity -> flatter distribution (higher entropy).
    Low complexity -> peaky distribution (lower entropy).
    """
    if complexity == "High Complexity":
        # Flatter logits -> higher entropy -> more likely to exceed threshold
        logits = np.random.normal(0.0, 0.5, size=NUM_LOGITS)
    else:
        # One dominant logit -> lower entropy -> stays local
        logits = np.random.normal(0.0, 0.2, size=NUM_LOGITS)
        dominant = np.random.randint(0, NUM_LOGITS)
        logits[dominant] += np.random.uniform(3.0, 6.0)
    return logits


# ── Master Equation (Equation 4): Routing Decision ────────────
def routing_decision(entropy: float, tau_t: float) -> str:
    """
    If H(G(x)) < tau_t  -> LOCAL_SHADOW
    If H(G(x)) >= tau_t -> LIVE_IBM_QPU
    """
    if entropy < tau_t:
        return "LOCAL_SHADOW"
    else:
        return "LIVE_IBM_QPU"


# ── Prompt Type Generation ────────────────────────────────────
HIGH_COMPLEXITY_PROMPTS = [
    "Explain quantum decoherence in open systems with Lindblad dynamics",
    "Derive the holographic entropy bound from first principles",
    "Analyze the fracture topology of a 15-qubit GHZ state under noise",
    "Compare variational quantum eigensolver convergence across ansatze",
    "Model the interference pattern of a pentagon-geometry MoE router",
    "Prove the no-cloning theorem's implications for quantum routing",
    "Describe phase transitions in quantum error-correcting codes",
    "Formulate the master equation for a driven-dissipative qubit array",
]

LOW_COMPLEXITY_PROMPTS = [
    "What time is it",
    "Summarize today's conversation",
    "How are you feeling",
    "Remind me about my schedule",
    "What did we talk about yesterday",
    "Tell me a short joke",
    "Define entropy in one sentence",
    "Say hello to Nate",
]


def generate_dataset():
    print("═══════════════════════════════════════════════════════")
    print("  Forge Fracture Dataset — Quantum-Classical MoE")
    print("═══════════════════════════════════════════════════════")
    print(f"  Steps:      {NUM_STEPS:,}")
    print(f"  T_max:      {T_MAX}s")
    print(f"  tau_base:   {TAU_BASE}")
    print(f"  lambda:     {LAMBDA}")
    print(f"  Logit dim:  {NUM_LOGITS}")
    print(f"  Output:     {OUTPUT_PATH}")
    print()

    n_calls = 0
    local_count = 0
    qpu_count = 0

    with open(OUTPUT_PATH, "w") as f:
        for step in range(1, NUM_STEPS + 1):
            # Equation 2: Temporal Governor — simulate elapsed time
            # Steps spread across T_max with slight jitter
            delta_t = (step / NUM_STEPS) * T_MAX
            delta_t += np.random.uniform(-0.01, 0.01)
            delta_t = max(0.001, delta_t)

            # Assign prompt complexity (60% low, 40% high)
            if np.random.random() < 0.4:
                complexity = "High Complexity"
            else:
                complexity = "Low Complexity"

            # Generate simulated logits and compute entropy (Eq 1)
            logits = generate_logits(complexity)
            entropy = shannon_entropy(logits)

            # Compute adaptive threshold (Eq 3)
            tau_t = adaptive_threshold(n_calls, delta_t)

            # Master equation routing decision (Eq 4)
            route = routing_decision(entropy, tau_t)

            # Update QPU call counter
            if route == "LIVE_IBM_QPU":
                n_calls += 1
                qpu_count += 1
            else:
                local_count += 1

            # Write JSONL record
            record = {
                "step": step,
                "simulated_prompt_type": complexity,
                "shannon_entropy": round(entropy, 6),
                "adaptive_threshold": round(tau_t, 6),
                "routing_decision": route,
                "qpu_api_calls": n_calls,
            }
            f.write(json.dumps(record) + "\n")

    print(f"  Generated {NUM_STEPS:,} records")
    print(f"  LOCAL_SHADOW:  {local_count:,} ({100*local_count/NUM_STEPS:.1f}%)")
    print(f"  LIVE_IBM_QPU:  {qpu_count:,} ({100*qpu_count/NUM_STEPS:.1f}%)")
    print(f"  Final N_calls: {n_calls}")
    print(f"  Final tau_t:   {adaptive_threshold(n_calls, T_MAX):.4f}")
    print()

    # Verify output
    with open(OUTPUT_PATH) as f:
        lines = f.readlines()
    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    print("  First record:", json.dumps(first, indent=None))
    print("  Last record: ", json.dumps(last, indent=None))
    print()
    print(f"  ✅ {OUTPUT_PATH.name} written ({len(lines):,} lines)")
    print("     Ready for H100 distillation.")


if __name__ == "__main__":
    generate_dataset()
