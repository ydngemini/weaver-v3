# SOMA-SENSE V2: one-finger instrumented sensor node

**Continuation of Weaver-SOMA's real sensory hardware design.** No hardware fabricated, no real motor actions, and no claim of physical safety or subjective experience. Source is at prototype/schematic-ready level, **not a Gerber/BOM-for-order package**.

## What is here

- [BOARD_DESIGN.md](BOARD_DESIGN.md): complete component/power/bus/sensor placement, pin plan, calibration and physical bring-up checklist.
- [finger_layout.svg](finger_layout.svg) and [system_architecture.svg](system_architecture.svg): editable concept diagrams.
- [PINMAP.csv](PINMAP.csv), [CONNECTIONS.csv](CONNECTIONS.csv), [BOM.csv](BOM.csv): source for PCB capture and procurement review.
- [FIRMWARE_CONTRACT.md](FIRMWARE_CONTRACT.md): 32-byte CAN-FD telemetry and real-data acquisition timing.
- [packet.py](packet.py): executable frame codec and CRC-16; not security/authentication.
- [gateway.py](gateway.py): provenance provision requirements, clock/sequence freshness, supply/encoder/thermal checks, optional calibrated pressure proxy.
- [bridge.py](bridge.py): V2 measurements -> original `Sensors/sensor_hub.py` -> SOMA 8×7 observation + 8×3 quality mask.
- [test_sensor_v2.py](test_sensor_v2.py): 28 unit tests.
- [fault_campaign.py](fault_campaign.py), [fault_results.json](fault_results.json): 1,500 scripted fault/noise trials and explicit undetectable cases.
- [integration_model.py](integration_model.py): actual recovered 121,827-parameter U1 checkpoint smoke test using 190 synthetic CAN-FD frames. It is an inference-shape compatibility test, **not** an ability/safety evaluation.

## Running locally

From the V2 folder with Python 3.11+ and pytest:

~~~bash
python -m pytest -q test_sensor_v2.py
python fault_campaign.py
~~~

`bridge.py` expects the *existing* `Sensors/sensor_hub.py` in the GitHub repository. A local exported archive also supports `soma_sensor_v1/sensor_hub.py` in a sibling folder. It does not copy/change the old implementation.

`integration_model.py` requires the original `WeaverSOMAUnified` source and **SHA256-verified** recovered U1 weights, which are intentionally not uploaded to the public GitHub repository. By default it expects `Weaver-Soma/U1/` and `Weaver-Soma/U1/results/weaver_soma_unified.pt`, or accepts the environment variables `SOMA_U1_SOURCE_DIR` and `SOMA_U1_CHECKPOINT`. No unsigned or untrusted weight files are loaded.

## Interpretation

Scripted tests show functionality of chosen rules, not real-world sensitivity or an independent device safety certificate. A slow thermal freeze below witness drift threshold, both thermal references stuck together, and a spoofed CAN sender with valid CRC remain undetected. Hardware and authenticated provisioning are separate design obligations.