# WEAVER-SOMA U2 Stage 5: long-sequence sensory world-model continuation

**Research-only continuation of the original trained 100,206,197-parameter U2 checkpoint.** No physical robot was connected; no sensory signal has been physically calibrated or measured; no new language, vision or audio foundation model was trained.

## Architecture

- Reuse the original U2 100.2M deep cognitive core checkpoint from Stage 4 with its trained weights, not a randomly initialized replica.
- Process **32 continuous simulated sensor samples**, grouped into four consecutive 8-frame windows; pressure/thermal/joint channels remain masked by independently assessed quality.
- Add a small trainable 64-dimensional recurrent memory cell across the four windows and two temporal readouts: 24-step counterfactual hazard across three abstract actions, and current state reconstruction.
- Continue full **100,082,241-parameter U2 deep-core** training on the latest segment while truncating gradients through earlier core windows to stay within a 4-GB CPU memory environment. Memory cell and temporal heads train jointly.
- Every model output remains advisory-only. Stage 5 has no signed intent path, actuator writes, sensor provisioning secrets or autonomous deployments.

## CPU reproduction

Set your working directory to `Weaver-Soma/U2`:

~~~bash
python -m pip install torch numpy scikit-learn pytest
python -m pytest -q test_stage5.py
python train_stage5.py --checkpoint /absolute/path/to/WEAVER_SOMA_U2_100M_Stage4_FULL_MODEL.pt --steps 80 --batch-size 4 --threads 3
python verify_stage5.py --checkpoint ../results/WEAVER_SOMA_U2_Stage5_Temporal_Research.pt
~~~

The original 400 MB checkpoint is not embedded in this source package. Source checkpoint SHA-256 and parameter counts are checked. All dataset generation is deterministic with separate seeds for train, development and unseen scenario testing. **Do not substitute the previous U1 weights for the trained Stage 4 U2 checkpoint**.

## Bench sensor wiring

`Sensors/V2/packet.py`, `gateway.py`, `bridge.py` plus `Sensors/sensor_hub.py` are copied from SOMA-SENSE V2 / V1 research sources. `U2/sensor_stream.py` wraps them, preserving packet chronology, calibration provisioning and the distinction between unknown pressure and measured calibrated pressure. The included tests never present synthetic packets as real readings.

## Interpretation

Even if a held-out synthetic dataset improves, this does not qualify as full-scale pretraining on the real world. Real sensor sessions, image/audio/text tokenization, unsupervised environment prediction and independent physical safety trials are still missing. The memory module offers a bounded 32-frame horizon, not indefinite autobiographical memory. The lab should use the [physical data collection and calibration plan](./PHYSICAL_DATA_COLLECTION.md) before collecting training data from hardware.

## Verified V2-to-U2 software smoke test

`python U2/smoke_sensor_to_u2.py` generates 460 explicitly synthetic 32-byte CAN-FD frames, validates them with the existing SOMA-SENSE gateway, assembles 32 sequential sensor samples and runs the actual Stage 5 checkpoint. It deliberately **does not** generate physical pressure when contact force/area calibration is unavailable. See `results/v2_sensor_to_stage5_smoke.json`.