# WEAVER-SOMA Stage 5: Physical sensor acquisition and calibration acceptance

**Status: no physical sensor recordings have been collected in this project session.** Everything labelled `synthetic` comes from a simulator or fabricated test packets, not measurements made with an actual finger.

## Hardware and source-of-truth

The proposed V2 finger node uses four individually calibrated force-sensitive tactile pads, an AS5048A magnetic encoder with an independently mounted potentiometer, two TMP117 thermometers, a monitored 5-V rail and timestamped STM32-to-CAN-FD telemetry. This is an unbuilt prototype design. The 32-byte packet format, gateway and source-health monitor are reused without redefining units. A valid CRC provides data-integrity checks, **not** signed identity or attested calibration.

## Calibration protocol before enabling any physical data for training

1. **Physical build identity.** Record node serial, board revision, sensor serial numbers, connector/power checks, firmware checksum, CAN-ID provisioning and an independently verified clock offset. Treat a reboot or clock wrap as requiring explicit resynchronization.
2. **Tactile force.** Use a trusted force reference and load each pad at multiple levels, unloading and repeating to measure nonlinearity, hysteresis, temperature dependence and drift. Do not equate force with pressure. To create pressure in kPa, independently characterize effective contact area in mm² for each active patch and record its uncertainty.
3. **Temperature.** Co-locate both TMP117 sensing elements at a controlled thermal reference at several setpoints, wait for thermal settling, and measure the thermal lag and time-dependent difference. A second probe is *not* always an equivalent reference if mounted at a different thermal depth.
4. **Joint angle.** Compare magnetic encoder and potentiometer against an independent mechanical angular reference, sweep slowly and quickly through the intended safe range, measure backlash, magnet offset and plausible velocity. Separate wiring and attachment modes so common-mode failure is harder.
5. **Power, clock and digital integrity.** Test brownouts, ADC rails, CAN errors, unplug events, bus overload and stale/replayed frames. The MCU supply must be independently protected, and motor cutoff must not depend on the model or CAN link.
6. **Environmental sessions.** Capture held-out sessions varying contact material, curvature, applied force, moisture, ambient and contact temperature, velocity and actuator loading. Keep session metadata with reference measurement and full timestamp history.
7. **Operator review.** Two authorized engineering reviewers sign off measured calibration before data are labelled `hardware`. Manual JSON flags do not constitute cryptographic attestation.

## Data schema

Each capture begins with an immutable session manifest and an independent local checksum. Each sample includes sample sequence, capture timestamp (monotonic), receipt timestamp, device/firmware reference, calibration digest, measured values with real units, independently assessed modality quality, new-conversion freshness, sensor-health reason codes and provenance (`synthetic` or `hardware`). Keep raw packet bytes in a separate access-controlled log only when consent and retention rules allow. Never store personal transcripts, faces, or biometrics by default.

## Train/validation/test separation

No data windows from the same **continuous physical run** should straddle train and validation/test splits. Hold out at least one device, material or thermal regime and entire unseen capture sessions. Report both how many physical sessions exist and how many overlapping windows they generated. Under-1% avoidable simulated errors are not physical safety assurance. Keep independent hardwired actuator stops and an emergency cutoff.

## Where this stage stands

`U2/sensor_stream.py` requires explicit provisioning and rejects `hardware` capture without externally evidenced calibration checkboxes. The test suite exercises the CAN-FD gateway with **synthetic packets only** and produces SOMA-compatible 8-frame and 32-frame windows. The new temporal model is trained on synthetic 32-frame episodes, not on captured real measurements.