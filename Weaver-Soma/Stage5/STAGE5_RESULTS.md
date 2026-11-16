# WEAVER-SOMA Stage 5: actual 100M-core CPU continuation, sequence memory, and sensor bridge

**Result:** CPU PyTorch training and independent verification completed. **No physical data collected, no general-purpose multimodal foundation-model pretraining, no hardware control.** The Stage 4 U2 model is the initializer, not a model trained afresh on billions of tokens.

## Model

- Stage 4 source model: **100,206,197 parameters**. Stage 5 adds a 64-d recurrent window memory, a 24-step counterfactual hazard readout and a compact body-state readout for a total of **100,258,619 parameters**.
- **100,082,241 core parameters** received nonzero gradients on the actual training step. Including new memory/readout parameters, 100,134,663 were trainable. U1-derived 121,827 parameters were frozen and verified bit-for-bit unchanged.
- **80 CPU optimizer updates**, batch size 4, AdamW learning rate 8e-6. Four sequential 8-sample (120 ms/sample) windows give 32 sample positions, approximately **3.84 seconds of synthetic continuous sensory history**. Predictions include three alternative actions and a **24 x 120 ms = 2.88-second horizon**.
- Old core windows are evaluated without gradients to bound CPU memory; gradient on latest window updates the full shared 100M core. This is *truncated* cross-window backpropagation, not full BPTT through the entire sequence.
- **1,280 train** synthetic trajectories under normal/changed dynamics and four sensor conditions, **448 development** trajectories, and **1,200 locked test** synthetic trajectories under *extreme*, *thermal-shift* and *motor-shift* physics. Sensor conditions: clean, noise, independently flagged faults, silent faults. Holdout data were not used to select the best checkpoint.

## Development criterion and measured results

| Development metric | Stage 4 core + random temporal heads (starting point) | Stage 5 trained |
|---|---:|---:|
| Average joint loss | 0.954700 | **0.920793** |
| Worst-condition joint loss | 1.217563 | **1.168653** |
| Mean 24-step hazard Brier | 0.243399 | **0.235494** |

Selected step: 80 of 80; locked test aggregate joint loss **0.909882**, worst-condition loss **1.173813**, mean long-hazard Brier **0.237124**. Do not compare absolute losses between these and prior Stage 4's different 8-frame simulation protocol.

## Independent post-selection locked comparisons

| Unseen condition | Before 32-frame training (loss) | After (loss) | 24-step hazard Brier before | After |
|---|---:|---:|---:|---:|
| Thermal-shift, flagged sensor | 1.162056 | **1.118542** | 0.249084 | **0.239468** |
| Motor-shift, clean sensors | 0.819615 | **0.786556** | 0.244216 | **0.233823** |
| Extreme physics, silent fault | 0.900205 | **0.875451** | 0.257128 | **0.242797** |

The 'before' baseline is exactly Stage 4's trained 100M core with *untrained new temporal heads*. That comparison establishes the new temporal objectives learned in this setup. It does not prove the new 100M weights generalize better than all Stage 4 outputs. Unavoidable crossing floors remain extremely high (for example, the extreme/silent group had **100% total threshold violations** in this small subset). No physical action safety claims.

## Memory and checkpoint integrity

- **136 deep-core tensors** changed versus the Stage 4 checkpoint, max absolute tensor difference ~0.0005495.
- **All U1-derived original tensors bit-exact** relative to Stage 4.
- Adversarially corrupting every masked thermal observation/derivative led to exactly **0.0 maximal output difference**.
- Ablating the first three windows in a `thermal_shift/flagged` heldout group changed 24-step hazard Brier from **0.239468 (all four windows)** to **0.241251 (latest window only)**. This small benefit supports a functional long-memory pathway on one test, not comprehensive long-term causal understanding.
- Stage 5 checkpoint: 401,119,127 bytes; SHA-256 `dd206d9e0398f88ade46cb77e042995938dd35c6520b6fe0a4675c88bfec6605`.
- Stage 4 source checkpoint SHA-256 `14d170d70f7d16d2abc23721f4f4cc3a9b3d115651dffbb2e343ebfb1f135149`.

## Source and validation

- `U2/long_world.py` generates 32 time-contiguous action/thermal/joint trajectories, 6-step known counterfactuals, 24-step hazard labels.
- `U2/temporal_model.py` wraps **the same trained U2 core** in a trainable, persistent four-window GRU and temporal heads.
- `U2/train_stage5.py` performs CPU optimizer training and selects checkpoint **using development data only**.
- `U2/verify_stage5.py` verifies actual changes in deep weights, frozen U1, checkpoint provenance, missing-channel invariance and post-selection comparisons.
- `U2/sensor_stream.py` consumes existing SOMA-SENSE V2 gateway packets via the existing sensor health monitor and rejects unverifiable physical calibration claims.
- `U2/test_stage5.py`: **7 unit tests passed**. Synthetic CAN-FD packets are never misrepresented as hardware data.

## What's still missing

No physical PCB has been fabricated, no traceably calibrated finger force/temperature/angle data were available, and there is no actual sensor stream from hardware. No pretrained camera, speech, or language token pathways are included. The extra recurrent memory is bounded to 32 sampled frames and has not been validated for indefinite autobiographical memory. High 24-step hazard prevalence under extreme physics can make AP appear deceptively strong. Critical hardware actuator reflexes, timestamp verification and signed Intent Capsules remain outside this network. Proper *full-scale* pretraining needs far more data, diversity, independent tests and much more compute than the 80-step CPU proof of continuation.