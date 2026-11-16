# SOMA-SENSE V2: schematic-ready finger sensor node (not fab-ready)

**Design status 2026-11-15:** component selection, conceptual block diagram, pin allocation, connection map, host-side CAN-FD parser, synthetic fault tests, and recovered U1 smoke test complete. No actual copper-layout DRC/ERC, footprint verification, STM32CubeMX AF validation, EMI qualification, firmware build, fabrication, or physical sensor calibration has occurred. **Do not order a board from these files as if it is production-ready.**

## Mission

A first instrumented articulated finger with four separate tactile force zones, two temperature witnesses, a 14-bit magnetic joint encoder plus independently routed potentiometer, a normally closed joint stop sensor, timestamped telemetry and an independent safety interface. Subsequent hands/body modules repeat the same pattern with their own assigned CAN IDs, hardware calibrations, and supervised clock mapping.

## Sensor placement

See [finger_layout.svg](finger_layout.svg). Position F1 and F2 on the distal fingertip/pad (not necessarily two sensors side-by-side; actual mechanical space must be checked), F3 and F4 on the proximal/intermediate phalanges. Put T1/T2 close together on the contact surface with separate signal and temperature paths and a protective thermal interface. Mount the AS5048A coaxial to the joint magnet, and mount the witness pot with an independent coupler so a slipped magnet does not defeat both. Place the STM32/controller board at the finger base or hand, away from actuators and hot regulators. Flex cable and strain relief need bend/fatigue tests. A tiny distal TMP117 assembly connected to a base MCU over short flexible I²C is acceptable only after validating bus capacitance and motion noise; a future distal digital node may be better.

## Main components

| Function | Reference candidate | Engineering basis |
|---|---|---|
| MCU | STM32G0B1CBT6, 48-pin LQFP | 64 MHz Cortex-M0+, FDCAN, SPI, I²C, 12-bit ADC, watchdog, USB; actual pin/AF mapping in PINMAP.csv |
| CAN physical | TCAN332G | 3.3 V CAN-FD, specified up to 5 Mb/s; CAN topology and termination still need engineering |
| Four tactile channels | ADS1115 | 4-input multiplexed 16-bit ADC; **860 SPS total**, budget 4×100 SPS = 400 SPS plus reconfiguration overhead |
| Tactile media | FlexiForce A201 or comparable | Flexible film senses **force** dependent resistance; manufacturer recommends an op-amp front end for characterized performance; divider below is only a bench prototype |
| Joint primary | AS5048A SPI magnetic angle | 14-bit position code, angle resolution != calibrated accuracy |
| Joint witness | independent 3.3 V analogue potentiometer | STM32 ADC, independent fixture if possible; mechanically limited to joint arc |
| Contact + local witness temperature | 2 × TMP117 | I²C at addresses 0x48/0x49; **configure 125 ms / single average** rather than factory averaged 1 Hz |
| 5V-to-3.3V rail | TLV75533PDBVR | TI example 500 mA LDO; **maximum input 5.5 V**, so use regulated 5.0 V with protection, not a battery/servo feed |
| Motion limit | mechanically NC limit switch | Must reach a **separate** deterministic motor disable/guard, MCU reports only a secondary status |

All specs require checking on the ordered exact package and datasheet revision. Primary sources: [STM32G0B1CB](https://www.st.com/en/microcontrollers-microprocessors/stm32g0b1cb.html), [ST datasheet Rev6](https://www.st.com/resource/en/datasheet/stm32g0b1me.pdf), [ADS1115](https://www.ti.com/product/ADS1115), [TMP117](https://www.ti.com/lit/ds/symlink/tmp117.pdf), [AS5048A](https://www.infineon.com/part/AS5048A), [TCAN332G](https://www.ti.com/product/TCAN332G), [FlexiForce A201](https://www.tekscan.com/products-solutions/force-sensors/flexiforce-a201-sensor).

## GPIO, exact functional allocation

See [PINMAP.csv](PINMAP.csv) for physical LQFP48 numbers and STM32 alternate functions (crosschecked to ST DS13560). Critical routes: PA11/PA12 = FDCAN1 RX/TX AF3; PB6/PB7 = I2C1 SCL/SDA AF6; PA5/PA6/PA7 = SPI1 SCK/MISO/MOSI AF0; PA4 = encoder CS; PA0 = analog witness angle; PA1 = 5V supply divider; PA13/PA14 = SWD. Reserve PB0 for limit observation only and never claim the STM32 substitutes for the external limit guard.

## Circuits / electrical nets

See [CONNECTIONS.csv](CONNECTIONS.csv). Key engineering notes:

- **Power:** regulated fused 5 V supply -> reverse/overvoltage input protection -> TLV75533 -> separate clean 3.3 V distribution. Decouple each IC and verify LDO thermals. Sensor ground routing must not share high-motor-current loops. Design TVS/EMC on CAN and external connectors. Tie VREF+/VDDA according to ST reference design, not a guessed PCB connection.
- **I²C:** one short, local 100/400 kHz bus (start 100 kHz for flex) with one 4.7 kΩ pullup pair. Address map TMP117 contact 0x48, TMP117 nearby witness 0x49, ADS1115 0x4A via ADDR-to-SDA strap. Configure distinct addresses *before populating*.
- **Tactile analog:** proposed **experimental** 3.3V -> FSR -> divider node -> 10k pulldown -> ADC AIN0–3. Add input series resistance, filtering and input protection after analog review. This simple readout is nonlinear; Tekscan's published amplified-circuit accuracy is **not transferred** to our divider. Consider migrating to a buffered transimpedance amplifier when calibration data demonstrate a benefit. ADS1115 PGA ±4.096V with 3.3V VDD and 0–3.3V inputs: respect device input common-mode and overvoltage limits.
- **Temperature:** two co-located TMP117s need separate mounting and thermal coupling characterization. Differences can reflect genuine temperature gradients, so reject thresholds are hypotheses, not proof. Device may return cached data between conversions; set NEW_TEMP only on newly completed conversions.
- **Joint:** primary magnet requires air gap/alignment testing and absolute-angle offset. Pot must have a separate mechanical mounting path and be ratiometric to the actual ADC reference. A shared shaft jam can affect both; add limit switch and motor-current/IMU measurements at the actuator guard.
- **CAN-FD:** TCAN332G TXD/RXD to STM32 FDCAN, 120 Ω end termination at only two true bus ends; nodes in the middle must not be automatically terminated. Wire twisted CANH/CANL and an appropriately grounded cable. An application CRC is not node authentication.
- **Optional future:** forearm ICM-42688-P IMU and INA226 motor-current sensing are separate add-ons. They are intentionally not on this first 32-byte finger packet; motor electrical safety is separately implemented.

## Signal and sample rates

| Source | Target acquire | Transmit | U1 input |
|---|---:|---:|---|
| F1–F4 from ADS1115 | 100 Hz each (400 mux SPS) | 100Hz snapshot | Calibrated force vector for future body tokens; pressure only after contact area validated |
| AS5048A + pot | 200 Hz | 100 Hz | Joint fraction with verified independent witness |
| Dual TMP117 | 8 Hz | cached in 100Hz packet with NEW_TEMP bit | Celsius + true acquisition timestamp |
| Whole finger snapshot | 100 Hz / 10 ms interval | 32-byte CAN-FD frame | Current model resamples **8×120ms** historical window |

**Sample freshness bug fixed:** the packet explicitly distinguishes a newly acquired temperature from a cached one. Existing SOMA health masks must never be refreshed using repeated stale readings. The receiver rejects stale messages and preserves per-modality acquisition history.

## Calibration before interpreting force as pressure

1. Disconnect any motor drive; clamp finger mechanically.
2. Measure sensor unloaded ADC values and 5–10 traceable weights through the **same fixture and contact material**; repeat apply/remove cycles and temperature variation; store calibration table, test residuals and per-taxel hysteresis.
3. Confirm taxel-effective contact areas under representative loading. A201 sensitive diameter is not enough: pressure = force/contact area is only valid if the effective area is measured. Without it, pressure quality is **0** and the U1 pressure input stays masked, even if four force values are present.
4. Angle: independently measure joint hard stops and compare absolute magnet angle with potentiometer across the full arc and midpoints. Lock down magnet spacing and sign conventions.
5. Thermal: test T1/T2 with a reference probe at 20, 30, 40 °C and under actual heating gradients, not simply static room air. Verify 125ms fresh-data timing and mounting-induced lag.
6. Generate the device's calibration SHA-256 and provision it to an authenticated local commissioning registry. Never accept a calibration ID sent only in an untrusted CAN data frame.
7. Run 100Hz packet replay/dropout and actuator noise injection with a separate safe-state controller. No human contact testing before mechanical risk review.

## Planned physical bench acceptance criteria (not passed)

- No force extrapolation outside calibrated envelope; max force/pressure uncertainty reported at 95% CI using physical repeated loads.
- Encoder agrees with independently mounted witness within measured tolerance through full joint arc, with wear and magnet displacement tests.
- Sensor fault injection catches stale/replay/wrong calibration, disconnected and rail faults, and quantifies false alarms with motor/EMI on.
- Continuously supervised timestamp error within **an initial design target ±3ms**; this is not a measured system property.
- Record actual per-channel sample latency at p50/p95/p99 and worst-case CAN bus arbitration, tested under multiple modules and traffic.
- Dedicated guard independently disables actuator on overcurrent, limits and e-stop; loss of CAN/AI cannot defeat it.

## Remaining blockers to fabrication

- PCB CAD schematic/footprints, manufacturer DRC/ERC, CAN termination/ESD component ratings, connector selection, flex/strain relief dimensions, regulator heat simulation, sensor AFE circuit and calibration fixture are **not complete**.
- The MCU firmware listed in [FIRMWARE_CONTRACT.md](FIRMWARE_CONTRACT.md) is specified but **not written, compiled, or flashed**. The existing Python parser/bridge validates only the intended format.
- No datasheet alone certifies this as safe around people. Independent motion-limiting hardware and a validated risk assessment are mandatory.