# SOMA-SENSE V2: instrumented finger node

**Engineering prototype only. Not fabricated, electrically verified, flashed, or safe for motor control.**

This is the next sensor development step for the existing WEAVER-SOMA U1/U2 project, not a new neural model. The tested local prototype contains:

- A finger with four spatially separated resistive force sensing taxels; calibration is required before converting force to contact pressure.
- Two independent TMP117 temperature sensors; each only counts as fresh when a new conversion finishes.
- An AS5048A absolute joint encoder and mechanically independent potentiometer position witness.
- An STM32G0B1CBT6 candidate controller, ADS1115 4-channel ADC, and TCAN332G CAN-FD transceiver.
- TLV75533 3.3-V regulator supplied from protected **regulated 5V only**; separate hardwired normally-closed joint motion limit.
- One 32-byte CAN-FD telemetry frame at 100 Hz with sequence, capture time, explicit per-modality freshness flags, source diagnostics and CRC16 (CRC is **not authentication**).
- A host-side quality monitor that independently checks acquisition time, repeated or replayed frames, rail voltage, encoder/witness mismatch, thermal disagreement, and calibration provisioning.

The local prototype passed **28 Python unit tests**, executed **1,500 targeted synthetic fault-injection trials**, and completed a **190-frame shape-compatibility smoke test with the recovered trained U1 weights**. These results do not validate sensors, circuit behavior, bus timing, or motor safety on hardware.

## Required fabrication work

1. Capture and electrically review schematic in PCB CAD, then run ERC/DRC and validate every component footprint, connector and pin alternate function.
2. Select real protection, filters, analog front ends, power margins, CAN transceiver ESD and termination; do thermal/EMI analysis.
3. Build a **motor-disconnected bench test** with actual sensor samples; verify all advertised sample rates and monotonic time continuity.
4. Calibrate force, temperature and position against traceable references; quantify uncertainty, latency and fault false alarms.
5. Validate independent motor cutoff and hard limits outside the MCU/CAN/software paths.
6. Only after review consider physical robot integration with read-only Weaver inference.

Documentation here is a tracking design, not Gerbers or approved electronic CAD. The full working local package is available as SOMA_SENSE_V2_Finger_Node_Engineering_Prototype.zip and includes source code, test scripts, and editable diagrams.

**Original Weaver runtime, production actuators, and U1 checkpoint remain unchanged.**
