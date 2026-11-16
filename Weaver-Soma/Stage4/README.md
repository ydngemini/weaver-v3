# Weaver-SOMA Stage 4: Sensor integrity and physical connection design

This directory is the **sensor-ingestion development layer for Weaver-SOMA U1/U2**, not another model. The code has been run against synthetic inputs and the recovered U1 checkpoint on CPU. No physical device is currently connected, and no motor action occurs.

## Run the tests

From `Weaver-Soma/Stage4/`, with the PyTorch/NumPy/pytest dependencies from `../U1/requirements.txt`:

```bash
python -m pip install -r ../U1/requirements.txt
python -m pytest -q tests/test_sensor_stack.py
python fault_benchmark.py --cases 200 --output fault_benchmark_results.json
```

The benchmark constructs deliberate fault cases, including CRC corruption, replayed sequences, stale timestamps, out-of-range values, frozen readings with independently measured motion, redundant-sensor disagreements, and plausible biases without any independent witness. It **does not** measure real sensor accuracy.

For a synthetic demonstration using the **separately recovered and SHA-256 verified U1 weights**:

```bash
python demo_to_soma.py --checkpoint ../U1/results/weaver_soma_unified.pt
```

The demo prints sensor-quality flags, predicted risks and a supervisory stop **request**, but does not execute any action. Any hardware motor-stop mechanism must remain independently wired and verified.

## Files

- [`sensors/integrity.py`](./sensors/integrity.py): calibrated sample packet format and deterministic external health monitor.
- [`sensors/assembly.py`](./sensors/assembly.py): bounded-freshness asynchronous sensor aggregation; outputs U1/U2 `(8,7)` readings and `(8,3)` independent health masks.
- [`tests/test_sensor_stack.py`](./tests/test_sensor_stack.py): offline unit, fault, freshness and U1/U2 integration tests.
- [`fault_benchmark.py`](./fault_benchmark.py): seeded synthetic detection benchmark with explicit blind spots.
- [`SENSOR_DESIGN.md`](./SENSOR_DESIGN.md): practical sensing hardware, electrical architecture and calibration/fault-injection roadmap.

## What cannot be solved in software alone

A plausible smoothly biased pressure, temperature, or joint sensor can pass every basic checksum/range/sequence test. **Redundant independent references and a safety MCU** are required to detect relevant faults. The system should be tested on a real fixture with instrument-calibrated measurements before any robotic movement is enabled.