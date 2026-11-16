# SOMA-SENSE V1: Sensor architecture & reliability bridge

**CPU-only engineering prototype.** Integrates *without retraining* the existing trained Weaver-SOMA U1 and maintains U2's `obs[8,7]` and `quality[8,3]` interface. No hardware assembled, no motors controlled, and no model checkpoints published.

- `SENSOR_ARCHITECTURE.md`: whole-body scalable modular topology, first finger/forearm rig, real candidate sensor families and electrical/physical caution.
- `sensor_hub.py`: strict channel schema, host clock/sequence/calibration and independently paired-channel checks, rejected-packet quarantine, bounded resampling, conservative quality masking.
- `test_sensor_hub.py`: adversarial boundary/sensor-quality unit tests.
- `fault_benchmark.py`: repeatable synthetic fault injection with **two known missed classes**.
- `benchmark_results.json`: fixed-seed reference report; these numbers have *no physical validity*.
- `integration_u1.py`: full inference with recovered, SHA256-verified **original** U1 weights; never commands actuators.

## Run

Python 3.10+, numpy and pytest. Torch is only needed for the optional real trained model compatibility test.

~~~bash
python -m pip install numpy pytest
python -m pytest -q test_sensor_hub.py
python fault_benchmark.py
~~~

To test against the recovered ORIGINAL checkpoint and the original U1 `unified_model.py`:

~~~bash
python integration_u1.py --checkpoint /trusted/path/weaver_soma_unified.pt --model-dir ../U1
~~~

The verified epoch-24 checkpoint SHA-256 is `3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f`.

**Synthetic test only.** Datasheet parameters are not device-level accuracy guarantees. A binary “healthy” flag is not a safety certificate. See architecture doc for calibration and physical safety gates.