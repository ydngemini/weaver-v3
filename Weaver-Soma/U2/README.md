# WEAVER-SOMA U2: 100,206,197-parameter CPU architecture

U2 **extends U1**, retaining its sensor encoders, recurrent body memory, five experts, shared-workspace representations, counterfactual forecasts and output signatures. A ten-block 1,024-dimensional attention/MLP cognitive trunk is added before the shared output heads. It is one trainable PyTorch module, not a separate AI service.

## Current state

- Parameter count: 100,206,197. Architecture and CPU forward/backward tests have succeeded locally.
- The **trained U1 epoch-24 checkpoint has been recovered and verified** (91 matching base tensors).
- The original U1 tensors can be loaded by WeaverSOMA100M.load_u1_weights(trusted_path).
- The additional U2 deep-cognitive-core weights are **randomly initialized and untrained**. Compatibility is not proof of improved capabilities.
- This is a numerical CPU research model, not a pretrained language model, complete general intelligence, or physical robot controller.

## Quick start

From this directory with torch installed:

~~~bash
python test_u2.py
python weaver_soma_u2.py --smoke-test
python weaver_soma_u2.py --train-step
~~~

Restoring original learned weights:

~~~python
from weaver_soma_u2 import WeaverSOMA100M
model = WeaverSOMA100M()
matched_tensors = model.load_u1_weights("../U1/results/weaver_soma_unified.pt")
assert matched_tensors == 91
~~~

The restored original file is not kept in the public repository. Its SHA-256 is:
3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f.

Recommended next stage: freeze U1's existing trained layers; train the new shared U2 trunk on the same synthetic data while measuring validation loss; progressively unfreeze only after stability checks; compare action errors, sensor-failure robustness and calibration on held-out episodes. Do not claim the trained U1 metrics for the untrained U2 model.

Never let model predictions sign intents, command motors, override an emergency stop or bypass Weaver's runtime safety guards.

## Stage 2 evidence

Zero-initialized deep and adapter residual gates guarantee exact U1 output equivalence after loading the recovered weights. The resulting U2 architecture contains **100,206,197 parameters**. The CPU experiment trained a 2,129-parameter adapter for 160 updates and an independent scalar deep gate for 12 updates. Development loss improved modestly, but the changed/extreme physics holdouts got worse, so U2 has **not** been promoted. See [STAGE2_RESULTS.md](./STAGE2_RESULTS.md) and [train_stage2.py](./train_stage2.py).

## Stage 3: masked-sensor robustness (executed)

The existing U2 architecture now passes sensor quality through the unified latent pathway and applies a zero-preserving **fault-conditional adapter**. When every independently verified sensor flag is healthy, the entire output dictionary is bit-exact to the restored U1. Local behavioral tests: **3 passed**.

A frozen-base, CPU-only **240-update** experiment trained only the ~2,129-parameter adapter using six augmented synthetic groups with 1,024 distinct underlying physical episodes. The *100-million-parameter deep core remains untrained*. A separate locked holdout included 5,400 new episodes with clean, noisy, detected-fault and silent-fault variants. Flagged-fault prediction loss improved substantially, but some nominal and noisy cases regressed; unreported faults remain dangerous. **No U2 checkpoint was promoted**.

- [Stage 3 exact executed training source](./train_stage3.py)
- [Second heldout sensor-gate evaluation](./evaluate_health_gate.py)
- [Behavioral tests](./test_health_gate.py)
- [Stage 3 measured results and limitations](./STAGE3_RESULTS.md)

A local, untrusted or missing U1 checkpoint is rejected by SHA-256 validation. Restore the original recovered checkpoint to `../U1/results/weaver_soma_unified.pt`. Run `python train_stage3.py`, then `python evaluate_health_gate.py`. Their small experimental adapter checkpoint is not committed to this public repository. All models remain advisory-only.
