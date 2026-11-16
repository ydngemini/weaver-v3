# U2 CPU stage-1 experiment: recovered U1 warm start

This document records an **executed synthetic CPU trial**, not robot readiness, improved intelligence, or a model suitable for motor control. The experiment used the locally recovered U1 epoch-24 checkpoint (verified SHA-256), and loaded all 91 original matching tensors into the 100,204,067-parameter U2 model.

## Experiment

- Hardware: CPU only, no GPU.
- Original U1 parameters: 121,827, **frozen**.
- New U2 shared transformer parameters: 100,082,240, **trainable**.
- Dataset: 192 synthetic nominal train episodes, 96 separate changed-physics validation episodes, 96 extreme-physics evaluation episodes.
- Fixed seed: 20261115. Optimizer: AdamW with learning rate 3e-6.
- Training: 20 minibatch updates, minibatch 16.
- Checkpoints examined: updates 1, 2, 4, 8, 12, 16, and 20. Choose lowest validation loss, including the unmodified step-0 warm start.

## Results

| Model / step | Changed-physics validation loss | Changed risk average precision |
|---|---:|---:|
| Recovered U1 baseline | 0.219378 | 0.989747 |
| U2 warm start, before updates | 0.219471 | 0.989871 |
| U2 update 4 | 0.219593 | 0.989871 |
| U2 update 8 | 0.219636 | 0.989871 |
| U2 update 20 | 0.220138 | 0.989663 |

**Best validation checkpoint = update 0.** The 20 CPU gradient updates did not outperform the warm start. This is a **negative generalization result**, so no new U2 checkpoint was promoted as better or deployed.

On the 96 extreme simulated evaluation episodes, the unmodified U2 initial model had loss 0.359483 and risk AP 0.997156, essentially matching U1's 0.359299 / 0.997156. Such a tiny hold-out cannot establish robust performance, let alone <1% safety. Total simulated crossing rates were still 23.96% (changed) and 85.42% (extreme), respectively, under a risk-only action rule; minimizing avoidable violations alone may favor over-withdrawal.

## Next controlled questions

1. Increase training sample diversity before adding more epochs.
2. Train with multi-step action-conditioned temporal transition targets, not only single-episode risk.
3. Test a gated residual initialized at zero, and compare against a **matched-parameter** smaller model rather than assuming size improves behavior.
4. Add held-out sensor faults, calibration, uncertainty metrics, and false-alarm/withdrawal cost.
5. Keep the U1 checkpoint as the validated baseline until U2 beats it on locked test criteria.

The code to reproduce the experiment is [train_stage1.py](./train_stage1.py). The original 400 MB experimental candidate has not been committed to the public GitHub repository. No hardware or live Weaver v3 runtime was touched.
