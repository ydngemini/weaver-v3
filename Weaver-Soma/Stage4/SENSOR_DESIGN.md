# WEAVER-SOMA Stage 4: Actual sensor path, trust boundary and health checks

**Status:** CPU-tested deterministic gateway and synthetic fault injection. No robot sensor was physically connected; no MCU driver, electronics wiring, actuator command, safety certification or real-world fault detection rate has been measured. This design extends the same Weaver-SOMA U1/U2 sensory interface, not a new model.

## The physical stack

1. **Skin:** pressure/tactile patches (capacitive or strain-based; force-sensing resistors can drift), shear/vibration if available, local temperature and a second surface/underskin thermal probe for lag.
2. **Joints:** independently measured encoders and end stops, torque/strain where available, motor-current sensing and a physical e-stop. An actuator *command* is not a valid independent witness of actual position.
3. **Motion/environment:** 6-axis IMU, optional proximity/depth cameras and microphones. Camera and audio are not implemented in this three-channel U1 sensor bridge.
4. **Local electronics:** per-zone MCU/ADC modules with calibrated gains and serial number; appropriate voltage regulation, isolation/fuses, wiring strain relief and connector retention. Local I2C/SPI is for short local wiring; differential CAN-FD or RS-485 is more suitable between distributed body zones. Actual electrical standards, ratings, grounding and shielding require hardware review.
5. **Safety versus cognition:** an independently engineered MCU handles motor cutoffs, current/temperature thresholds, watchdog, overtravel and e-stop. A Linux controller collects synchronized packets, validates them, creates health masks, then forwards **advisory** observations into Weaver-SOMA. Neural network outputs cannot enable motor movement.

## Proposed start-small bench, then scale

For an instrumented **one-arm** fixture, start with 4–8 touch elements, two temperature probes on one patch, one encoder and independent end-stop, a joint-current sensor, a six-axis IMU, local ADC and MCU, hardware stop, and a logging PC. Run observations and fault injection while motors are disabled first. After reliable data capture, expand to multiple patches across torso/hands/feet and per-joint load/position telemetry.

Starting sensor sample rates to evaluate, not requirements: tactile 50–200 Hz; joint position/current 100–500 Hz; temperature 5–20 Hz; IMU 100–400 Hz. SOMA U1's existing eight-frame input operates at a nominal 10 Hz, so high-rate sensing must be summarized conservatively, not truncated to arbitrary first readings. For true 5-Hz thermal updates, the gateway uses a **bounded-time cached reading** (default 250 ms) and marks it missing after expiry. Holding a previous sample does *not* imply a new measurement.

## Current software input contract

**Module:** `sensors/integrity.py`: frozen `SensorPacket(sensor_id, channel, sequence, captured_ns, raw_value, checksum)`, independent `ChannelSpec`, and fail-closed `HealthMonitor`.

`physical_value = raw_value * gain + offset` where gain and offset are **local verified calibration records**, not fields untrusted devices can freely choose. The example specs use unit gain and offset; replace with hardware calibration before trusting values.

Deterministic flags check device identity, sequence replay, CRC corruption, monotonic timestamps, age, nonfinite/range/slew, and independent reference disagreement. A **stuck value** is flagged only when independent hardware witnesses nonzero physical change. A legitimately constant value is not treated as a fault. Each fault immediately sets a mask to 0; recovery requires two fresh clean packets. CRC is not cryptographic authentication. Commanded actuator movement is *not* an independent witness.

**Module:** `sensors/assembly.py`: outputs exact U1/U2 shapes `obs=(8,7)` containing `[pressure_kPa, temp_degC, joint_load_fraction, pressure_slope, temp_slope, joint_slope, independently_verified_contact]`, and `quality=(8,3)` in channel order `[pressure,temperature,joint]`. A bad channel gets both its current reading and derivative zeroed. The existing model also masks those channels before its encoders. An independently verified sensor update may be reused for slower channels only inside its freshness deadline, with derivative set to zero. Unknown contact state is 0, not inferred from bad pressure.

**Safety escalation:** `AssembledFrame.stop_requested` is a **request to an independent supervisor**, not an actuator action. Any unknown or expired critical channel sets it. Illustrative threshold flags can also request it, but U1's thresholds (60 kPa, 43 °C, load .8) are experimental and should **not** be deployed as real limits without material tests and hazard assessment. If a model sees stale sensor data, a *separate* physical controller still needs to cut motor energy appropriately.

For multiple pressure patches or multiple joints, apply integrity checks to **every source separately**, then aggregate verified **maximum** load/temperature/pressure for this legacy three-channel interface. Averages can hide a dangerous hot spot. Loss of any safety-critical source should trigger the independent safety path. The eventual full-body model should expand to per-taxel spatial tensors, encoder/camera fusion, and explicit uncertainty with proper latency budgeting.

### Time synchronization

`captured_ns` is expected to be in the **host monotonic clock domain**, after measured MCU clock-offset conversion with uncertainty bounds. Device uptime ticks are not directly comparable to Linux monotonic nanoseconds. If conversion cannot be trusted, reject data or use measured host arrival time with an appropriate delay bound; never invent clock precision.

## Synthetic test outcomes

On a frozen 200-seed-per-channel-per-fault benchmark, 3 channels × 6 **explicitly detectable** fault classes × 200 trials = **3,600 injected events**. Detection was **3,600/3,600** in this narrow synthetic benchmark. Under 600 clean trials, **0/600** false alarms were observed. Under 600 plausible unverified biases without redundant sensors, **0/600 were caught**. This last result is the important negative finding: the health system cannot tell a believable lie from an authentic measurement without more evidence.

Failure classes: modified checksum, repeated sequence, old timestamp, out-of-range reading, frozen reading **with independent evidence of movement**, and plausible bias **with independent redundant reference**. These inputs were constructed to satisfy their specific detectable conditions; this is a functional unit-test benchmark, **not a claim about unknown faults or realistic false-negative rates**. Real sensor noise, drifting thermistors, mechanical hysteresis, calibrated ranges, and adversarial forged packets remain unevaluated.

## Next real-world gates

- Capture real ADC samples and build monotonic-clock offset tracking, sensor serial identification, calibration certificate/version and authenticated MCU transport.
- Collect simultaneous primary/secondary surface and temperature readings with known reference instruments. Test stuck-at, intermittent wires, short circuits, noise, thermal lag, humidity, rebound, drift, saturation, actuator vibration and deliberate channel loss.
- Pre-register fault-detection precision/recall, detection latency, worst-case missed hazards, false-stop time and independent hardware cutoff under every test.
- Only then evaluate a trained neural sensor-fusion model as *advisory*. Do not allow it to overwrite the deterministic health mask or authorize motion.

### Bench demo

`python demo_to_soma.py --checkpoint /path/to/recovered/U1/results/weaver_soma_unified.pt`

The script verifies the saved checkpoint SHA-256 and feeds integrity-checked **synthetic** packets into the **original trained U1 network**. It prints failure flags, model risk estimates and stop requests, but does not actuate anything.
