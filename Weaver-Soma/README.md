# Weaver-SOMA

**Unified embodied cognition research inside Weaver v3.** This folder contains the PyTorch U1 model and its 100.2-million-parameter U2 expansion. These are research prototypes, not a production robot controller or evidence of subjective consciousness.

## Versions

| Version | Parameters | State |
|---|---:|---|
| [U1](./U1/) | 121,827 | Trained on synthetic episodes; recovered epoch-24 checkpoint is stored **outside this public repository** |
| [U2](./U2/) | 100,206,197 | Trainable CPU architecture; the 91 original U1 tensors have a verified warm-start path; its added deep transformer blocks remain **untrained** |

U1 learns shared body-state, short temporal history, risk, counterfactual forecasts, internal routing, and memory/attention proxies in a single PyTorch model. U2 adds a ten-block shared cognitive trunk while retaining U1's input/output interface. Stage 2 adds zero-initialized scalar residual gains and a compact adapter, proving exact U1 output equivalence before additional training. The experimental adapter improved development loss but did **not** improve changed/extreme physics generalization: see [stage 2 results](./U2/STAGE2_RESULTS.md).

## Getting started

Install Python, PyTorch, NumPy, scikit-learn and pytest, then:

~~~bash
cd Weaver-Soma/U1
python -m pip install -r requirements.txt
python train_unified.py --epochs 30 --train-n 8400 --val-n 2100
python -m pytest -q test_unified.py
python evaluate_policy.py
~~~

This trains U1 **again** on the original synthetic simulator, creating a local checkpoint in U1/results/. To use the **recovered original trained weights instead**, manually restore the trusted weaver_soma_unified.pt file from the private recovery archive to U1/results/weaver_soma_unified.pt. Reference SHA-256:
3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f.

U2 CPU structural test:

~~~bash
cd ../U2
python test_u2.py
python weaver_soma_u2.py --smoke-test
~~~

Use WeaverSOMA100M().load_u1_weights(path_to_trusted_u1_checkpoint) to copy the trained U1 layers into U2; it does **not** train U2's new deep core.

## Security and experimental status

- This repository is **public**. Do not commit model checkpoints, personal sensor streams, API keys, or credentials. See .gitignore.
- No model output authorizes hardware movement. Original IntentCompiler, ReflexKernel, independently verified sensor health, and emergency-stop code retain control.
- Results apply only to synthetic physics and are not real-world safety guarantees. Absolute violation rates in the changed and extreme simulations remain high.
- This folder is a source-code development area; it is not wired into the running Weaver v3 service and has not replaced Weaver's original runtime.

The exact recovered U1 checkpoint is retained separately for authorized local continuation.

## Stage 4: Sensor integrity and physical sensing

[Stage 4](./Stage4/) adds a host-side **independent integrity monitor**, packet freshness/CRC/sequence checks, calibrated sensor readings, bounded-time asynchronous sample fusion and missing-channel masks for existing U1/U2 model inputs. It does **not** connect live hardware or alter motor control. CPU checks: 14 Stage 4 tests and the existing 3 U2 health-gate tests passed locally. In a deliberately detectable synthetic fault benchmark, 3,600/3,600 injected cases were caught, but 0/600 plausible sensor biases without an independent witness were detected. This gap motivates redundant physical sensors and safety MCU validation. [Read the Stage 4 hardware design](./Stage4/SENSOR_DESIGN.md).
