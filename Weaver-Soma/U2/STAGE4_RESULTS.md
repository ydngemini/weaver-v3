# WEAVER-SOMA U2 Stage 4: full-core CPU training

**Status:** Executed full-core PyTorch training and independent checkpoint verification. Research only; simulated physics and sensor data, not validated hardware or consciousness.

## Actual training

- Model size: **100,206,197 parameters**.
- **100,082,241 deep-core parameters** received nonzero gradient signals after optimizer update 2. This is NOT an adapter-only experiment.
- All **91 pretrained U1 tensors** restored from original checkpoint, independently SHA-256 verified, and frozen.
- 240 AdamW CPU updates, batch 16, learning rate 0.00003, approximately 224.47 seconds including evaluation.
- Training: 2,048 distinct synthetic physical episodes with four sensor-condition augmentations; development: 512 other distinct episodes. Normal and changed physics only in training.
- Chosen checkpoint: step 240 based on development loss. The extreme-physics test was not used to select weights.

## Development loss

| Condition | U1 baseline | Trained U2 |
|---|---:|---:|
| Mean joint loss | 0.91128 | **0.86585** |
| Worst-group joint loss | 1.75984 | **1.67399** |
| Clean-condition mean | 0.29191 | **0.29038** |

## Locked synthetic holdouts

2,100 independent underlying physics episodes: 700 each normal, changed, extreme. Each episode was evaluated with four sensor conditions: clean, noisy, independently flagged dropout, and unflagged silent corruption. This creates 8,400 condition-level evaluations, not 8,400 independent physical scenarios.

| Regime, mean over four sensor conditions | U1 loss | Trained U2 loss |
|---|---:|---:|
| Normal | 0.79780 | **0.75374** |
| Changed | 1.00064 | **0.95640** |
| Extreme | 1.24259 | **1.20712** |
| All 12 groups | 1.01368 | **0.97242** |

All **12 out of 12** holdout sensor/physics groups improved in joint loss. Independent reloading verified 60/60 inspected deep weight tensors changed from deterministic initialization, and a masked-channel corruption test passed.

## Important limits

- Risk average precision moved slightly in both directions; **no uniform improvement** established.
- Withdrawal rates generally increased. Improved joint loss does not necessarily improve task utility or physical control.
- Extreme simulated total physical violations remain roughly 80% due to physical scenario constraints.
- The model is **not** trained on real sensors, video, audio, natural language or robot-world interactions. The small synthetic dataset cannot establish general intelligence or robot safety.
- No live Weaver service or actuators were touched.

## Reproduce

Sources in this branch: train_full_core.py and verify_full_core.py. Restore the trusted original U1 checkpoint to Weaver-Soma/U1/weaver_soma_unified.pt; then run:

    python U2/train_full_core.py --steps 240 --batch-size 16 --threads 4 --interval 30
    python U2/verify_full_core.py

The original U1 checkpoint SHA-256 is 3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f.

The separately saved **400,383,725-byte research checkpoint** contains deep-core weights only (not all original pretrained U1 weights) and is not committed to this public repository. To restore, first load the original U1 weights into WeaverSOMA100M, then load the new deep-core tensors using torch.load(..., weights_only=True) and model.deep_cognitive_core.load_state_dict(...).

**Decision:** retain the trained U2 as a research checkpoint, not production or hardware control. Require larger independent real/simulated datasets, replicated seeds, model ablations, risk/utility evaluation and independent actuator cutoffs before promotion.
