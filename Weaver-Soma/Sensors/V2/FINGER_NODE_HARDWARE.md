# Finger Sensor Node V2: wiring and acquisition specification

## Block interfaces

```
Fused, protected regulated 5 V ---> TLV75533 3V3 ---> STM32G0B1 + ADS1115 + 2xTMP117 + AS5048A + TCAN332G
                                    |
Force-sensitive film zones F1..F4 --> individually filtered analogue divider (bench only) --> ADS1115
Joint magnet --> AS5048A SPI ----- STM32 PA5/PA6/PA7, select PA4
Independent joint pot -----------> STM32 PA0 ADC
5V divider ----------------------> STM32 PA1 ADC
TMP117 contact + witness --------> I2C1 PB6/PB7
Controller FDCAN PA11/PA12 ------> TCAN332G --> CANH/CANL shared bus
NC mechanical hard stop ---------> independent hardware guard; PB0 secondary readout only
```

## Candidate circuitry

- MCU STM32G0B1CBT6 (LQFP48) with SWD PA13/PA14, NRST and local low-impedance decoupling per ST requirements.
- ADC ADS1115, four single-ended force channels AIN0...AIN3, 400 total conversions/s at >=860 SPS capability. Mux channels are **not simultaneous**. The divider approach is an *experimental* readout, not the calibrated op-amp circuit or accuracy claimed by the force-sensor vendor.
- Force sensors: four FlexiForce A201 candidate elements. Use 3.3 V -> sensor -> individual 10k resistor to ground; sense at resistor node through reviewed input protection and filtering. **Do not interpret values as kPa until effective contact area is physically established.**
- Thermometers 2x TMP117, I2C addresses 0x48/0x49, scheduled new-conversion event approx 125ms. Cached temperatures are repeated in intervening 100Hz bus frames with NEW_TEMP=0.
- AS5048A 14-bit absolute encoder on SPI1 with a separate potentiometer on STM32 ADC; angle code resolution is not calibrated angular accuracy.
- CAN-FD: STM32 FDCAN1 with 3.3V TCAN332G; add ESD and select termination **only if actual node is at one of two backbone ends**.
- Regulator: TLV75533 with max input 5.5V, so motor supply or raw vehicle battery must never connect to its input. Validate regulator power, caps and Vref.
- Motor power and hardwired e-stop are external to this sensing board. **Loss of sensor node never authorizes additional motion.**

## STM32 pin allocation (candidate; pin mux must be confirmed again in EDA)

| Net | MCU pin | Function |
|---|---|---|
| CAN_RXD / CAN_TXD | PA11 / PA12 | FDCAN1 AF3 |
| I2C_SCL / I2C_SDA | PB6 / PB7 | I2C1 AF6 |
| ENC_SCK / MISO / MOSI | PA5 / PA6 / PA7 | SPI1 AF0 |
| ENC_CS_N | PA4 | GPIO chip-select |
| JOINT_WITNESS | PA0 | ADC |
| VIN5_SENSE | PA1 | ADC with protective divider |
| LIMIT_STATUS | PB0 | GPIO monitor only |
| TEMP_ALERT / ADC_ALERT | PB4 / PB5 | Optional GPIO inputs |
| SWDIO / SWCLK | PA13 / PA14 | Programming |
| NRST | PF2 | Reset |

All external connectors must be keyed, provide strain relief and tolerate repeated finger flexing. The sensor PCB should be located at the finger base/hand to limit moving mass. Separately mounted distal pads can connect over reviewed flex.

## Sampling and data contract

- 4 tactile inputs @ 100 Hz each; SPI encoder and ADC witness @ 200 Hz internally; 2 thermal channels @ 8 Hz.
- Finger snapshots @ 100 Hz / 10ms; CAN-FD frame uses 32-byte little-endian fixed-point layout.
- Field offsets: version (u8,0), node (u8,1), flags (u16,2), sequence (u32,4), capture_ms (u32,8), force 4 x u16 centinewtons (12-19), main encoder u16 (20), witness encoder u16 (22), contact temperature signed centi-C (24), thermal witness signed centi-C (26), 5V rail mV u16 (28), CRC16/CCITT-FALSE (30-31).
- New-conversion flags separate fresh ADC, temperature and joint state. Single frame timestamp is a **proxy** for acquisition time and does NOT establish sub-10ms causal attribution.
- Node identity and calibration digest require trusted provisioning, not a CRC or untrusted self-reported node ID. Reboots and time-counter wrapping need specified re-commissioning paths.

## Real-world acceptance gates (not yet performed)

- Force calibration under repeated known loads, hysteresis and contact-area measurement.
- Thermal two-probe characterization across realistic gradients and heat cycles, including low-drift freeze faults.
- Primary encoder vs independently mounted witness with magnet drift, cable noise and stuck-at inject.
- Rail droop, CAN errors, disconnects, bus overload and spoofed node tests.
- Independent hardwired motor cutoff during total MCU/bus loss.
- Human-contact and actuator testing only after mechanical safety review.

Datasheets and design references:
[STM32G0B1CB](https://www.st.com/en/microcontrollers-microprocessors/stm32g0b1cb.html) ·
[ADS1115](https://www.ti.com/product/ADS1115) ·
[TMP117](https://www.ti.com/lit/ds/symlink/tmp117.pdf) ·
[AS5048A](https://www.infineon.com/part/AS5048A) ·
[TCAN332G](https://www.ti.com/product/TCAN332G) ·
[FlexiForce A201](https://www.tekscan.com/products-solutions/force-sensors/flexiforce-a201-sensor).
