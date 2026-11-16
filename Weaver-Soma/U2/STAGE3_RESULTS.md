# WEAVER-SOMA U2 Stage 3: robust CPU adapter and fault-health gating

**Execution status:** 240 real CPU PyTorch optimizer updates completed. This is synthetic research only; no physical robot controlled, no consciousness claim, no production safety endorsement.

## Experimental setup

The original 121,827-parameter U1 weights (epoch 24) were recovered with verified SHA-256 (`3eb07468...5030c8f`). U2 is **100,206,197 parameters** including a 100M-parameter **frozen, randomly initialized** deep core and a **2,129-parameter trainable adapter**. Although U2 is large, this experiment updates only the small adapter. U1 remains frozen.

Six augmented training groups (512 examples each; 1,024 unique underlying physics episodes): normal+changed dynamics, clean/noisy/independently flagged channels. Development has six augmented groups (256 examples each; 512 distinct underlying physics episodes), and a worst-group validation objective selects checkpoint step 240 out of 240 CPU updates; no final test episodes used to select hyperparameters. An additional heldout test of 1,250 episodes per normal/changed/extreme regime was run before fault gating. We then added a **fixed deterministic mask** activating the adapter only when independently verified sensor health indicated a missing channel. The adapter was retrained with the deterministic health gate active and the same fixed training protocol. That retraining did not use holdout episodes. The independent second test evaluates the selected checkpoint without post-hoc tuning. The independent second test comprises **1,800 new episodes per regime (5,400 unique episodes)** with four measured sensor conditions per episode.

The mask is: `health_gate = 1 - min(quality)` over channels and history, taking values 0 or 1. The U2 state update is `u = u_U1 + health_gate * gamma * adapter(u_U1)`, where `gamma` is the trained scalar. All output tensors match U1 **exactly (maximum absolute difference 0.0)** when every channel is healthy. Original U1 trained tensors remain unchanged.

## Independent second test, changed physics

| Sensor condition | U1 loss | Health-gated U2 loss | U1 avoidable errors | U2 avoidable errors |
|---|---:|---:|---:|---:|
| Clean* | 0.3557 | 0.4037 | 0.444% | 0.389% |
| Noisy* | 0.4981 | 0.5433 | 1.278% | 1.167% |
| Flagged fault | 1.6668 | **0.8425** | 0.944% | **0.611%** |
| Silent fault | 1.4664 | 1.5072 | 3.778% | 3.667% |

*'Clean' and 'noisy' contain base-generator missing-sensor flags on approximately 12% of examples per sensor. These are not the same as **all sensors healthy**. Therefore conditional gating can alter some nominal observations and degrade their joint loss. A fully verified all-healthy batch was tested separately: **exactly unchanged predictions**.

## Other regimes

- Normal flagged: U1 joint loss **1.5959**, fault-gated U2 **0.6437**. Avoidable errors **0.50% → 0.111%**.
- Extreme flagged: U1 loss **1.3597**, gated U2 **0.9733**. Avoidable errors **0.389% → 0.333%**.
- In the changed-physics flagged regime, withdrawal rose **65.94% → 72.89%**. Better error rates cost more withdrawals. This must be measured as a meaningful task-utility tradeoff.
- In extreme synthetic physics, the unavoidable safety threshold violation floor remains **80.17%** of episodes. No neural method can overcome this fixed actuator/thermal simulator floor.
- Silent sensor faults remain serious failure modes and **cannot be solved by trusting a sensor's own health bit**; the independent sensor-health monitor and hardware safety MCU are essential.

## Scientific decision

**Do not promote this model for ordinary use.** The adaptation gives meaningful improvements on reported missing-sensor conditions but degrades ordinary joint loss. Keep U1 as the default, and evaluate health-gated U2 in read-only shadow comparisons for fault-triggered applications. Test confidence intervals and randomized independent faults before drawing safety conclusions. The original U1 model and weights remain safe and unchanged. The Stage 3 adapter is an experimental artifact and is not a promoted checkpoint.

## How to reproduce

1. Restore the separate verified U1 checkpoint at `U1/weaver_soma_unified.pt`.
2. `python U2/train_stage3.py` with numpy, torch, scikit-learn installed. Results in `results/stage3_results.json` and experimental adapter weights locally.
3. `python U2/evaluate_health_gate.py`. Outputs the independent second heldout comparison.
4. No files connect to the live Weaver runtime or sign/execute intents.

## Result files

`results/stage3_results.json`, `results/health_gate_heldout.json`, `results/heldout_comparison.csv`, `results/health_gate_changed_physics.png`, plus train/health-gate logs.