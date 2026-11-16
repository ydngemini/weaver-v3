# Firmware/transport contract (prototype, not deployed)

**Firmware is not flashed or compiled in this package.** `packet.py` is the executable host-side reference format; the MCU implementation must be verified against the same reference vectors.

## Schedule / timestamp discipline

- ADC ADS1115: 4 channels x **100 samples/s per channel**, sequentially multiplexed (400 conversions/s total, below the 860 SPS device maximum). No simultaneous sampling claimed. Record the acquisition completion time for each conversion in MCU memory and publish a coherent snapshot only after all four are updated.
- AS5048A SPI and independent potentiometer ADC: **200 samples/s** internally, provide freshest coherent primary/witness pair each 10 ms transmit period.
- Dual TMP117: configure single-average conversion at ~**125 ms** cycle (8 updates/s). Factory configuration commonly averages 8 at 1Hz, so default settings are not acceptable for our target. Use fresh-data status and a 125 ms schedule; verify actual chosen device configuration.
- Publish one **32-byte CAN-FD data frame every 10 ms** (100Hz). Repeated cached thermal data MUST have `NEW_TEMP=0`, not a fabricated new acquisition timestamp. Encoder and ADC flags similarly indicate true new acquisitions.
- Maintain MCU monotonically increasing `capture_ms` in the same time domain agreed with the gateway via trusted clock mapping. `seq` must increase on each transmitted frame. Reset/reboot/wrap requires recommissioning. Current demo deliberately lacks 32-bit wrap handling and secure identity.
- Surface explicit source diagnostic flags (ADC, thermal, encoder, power/calibration). Trust decisions are made independently at the gateway. Preserve original raw ADC samples in local test logs for calibration and root-cause analysis, under access control.

## Fixed data frame, little-endian, 32 bytes

| Byte offset | Field | Type | Unit |
|---:|---|---|---|
| 0 | version = 2 | u8 | constant |
| 1 | node ID | u8 | provisioned |
| 2 | flags | u16 | fault + fresh-data bitmask |
| 4 | sequence | u32 | monotonic count |
| 8 | capture_ms | u32 | node-clock milliseconds |
| 12 | force x4 | 4 x u16 | calibrated 0.01 N (centinewtons) |
| 20 | primary absolute joint angle | u16 | 14-bit magnetic code |
| 22 | witness joint angle | u16 | 14-bit-equivalent calibrated code |
| 24 | contact temperature | i16 | 0.01 °C |
| 26 | witness temperature | i16 | 0.01 °C |
| 28 | VIN5 supply voltage | u16 | millivolts |
| 30 | CRC16/CCITT-FALSE | u16 | over bytes 0–29 |

Flags: bit0 ADC fault, bit1 thermal fault, bit2 encoder fault, bit3 supply fault, bit4 calibration fault; bit8 `NEW_ADC`, bit9 `NEW_TEMP`, bit10 `NEW_ENCODER`. Unknown flags rejected. CAN-FD link CRC remains mandatory; application CRC helps detect accidental mishandling in host/storage but does **not** provide cryptographic authenticity.

**Important:** The sample frame does not include a calibration digest or node boot nonce. The trusted gateway must obtain these through a *separate verified provisioning channel* and reject unknown/rebooted nodes. A spoofed CAN sender with the correct identifier and a valid CRC can otherwise pass these checks. Do not use this bus as the only safety control.

## Zero-latency safety path forbidden

The CAN-FD protocol is for sensing and research. **No ML-controlled motor actions, enable lines, braking decisions, overcurrent cutoffs, or e-stop operations ride solely on this telemetry path.** Safety decisions must use separately validated deterministic motor electronics and hardwired fail-safe limits.