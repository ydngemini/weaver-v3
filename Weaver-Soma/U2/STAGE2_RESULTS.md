# U2 Stage 2: Exact U1 preservation, learned gates and CPU adapters

**Executed November 15, 2026; synthetic research only.** Model predictions do not authorize robot motion. See [train_stage2.py](./train_stage2.py).

## Architecture change

The expanded U2 network now starts as a bit-for-bit functional match to the recovered U1 network:

`u_2 = u_1 + gamma_deep * F_100M(z, memory, workspace, context) + gamma_adapter * A_2K(u_1)`

Both scalar gains start at **zero**; the deep core is still randomly initialized and must not be treated as pretrained intelligence.

Recovered checkpoint SHA-256 `3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f`. The original 91 tensors loaded, and the maximum initial output difference was **0.0** on a sixteen-example test across every output; a second local test checked batch sizes 1 and 9 and also gave 0.0.

Updated total model size: **100,206,197 parameters** (100.2 million). The adapter path has **2,129 parameters** including its scalar gate. U1's original weights and all 100M deep-core weights were frozen for the adapter comparison.

## Trial

- CPU, PyTorch. Reused exact U1 checkpoint; 1,024 synthetic training episodes (512 normal, 512 changed).
- Independent development: 384 episodes (192 normal and 192 changed).
- Fresh holdout: 300 episodes each normal, changed, extreme; **never used for selecting the adapter**.
- Adapter-only training: 160 optimizer steps, learning rate 0.001, batch 32; select lowest development joint loss from steps 0, 1, 5, 10, 20, 40, 80, 120, 160.
- Frozen deep-core residual-gain pilot: 12 additional CPU steps, learning rate 0.0003, batch 16; core weights untouched.

## Measured results

| Evaluation regime | Original U1 loss | Selected adapter loss | U1 AP | Adapter AP |
|---|---:|---:|---:|---:|
| Development (normal+changed) | 0.32921 | **0.32460** | 0.94875 | 0.94740 |
| Fresh normal | 0.29831 | **0.29428** | 0.86978 | 0.86510 |
| Fresh changed | **0.26190** | 0.28079 | **0.98041** | 0.97584 |
| Fresh extreme | **0.45388** | 0.47473 | **0.99771** | 0.99735 |

Average precision (AP) measures event ranking and does not establish decision safety. In the normal-physics test both models had **1.0% avoidable violations**, at the boundary rather than below it. In changed/extreme tests the risk-based rule incurred **0% avoidable** in this tiny sample yet total violations remained **21.33% / 80.67%**. There are only 300 examples per regime, too few to infer strong safety guarantees.

The adapter selected **step 160** based on development loss. That improvement **did not generalize** to changed or extreme physics. It should **not** replace U1.

In the separate scalar-gate-only pilot, development joint loss moved from 0.32920657 to 0.32916602 by step 12, a marginal change. This is not evidence of improved generalization; the 100M deep layers are still random, frozen weights.

## Decision and next experiment

- **Do not promote** either U2 adapter or 100M core; keep the recovered U1 behavior as the default.
- Next, train across more varied changed dynamics and use a worst-regime development objective before testing on a fully locked set.
- Compare against a compact model of similar *trainable* size, not merely a 100M frozen trunk.
- Test live calibrated sensors, missing channels, stuck-at readings and actuation limits only on a guarded research rig. No trained model gets to bypass signed Intent Capsules and independent reflex safeguards.

Raw numerical results and the development-only adapter checkpoint are kept locally; no model weights or private data are committed to public GitHub.
