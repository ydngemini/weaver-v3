# SOMA-SENSE v1 sensor monitor: executed synthetic reliability experiment

**Executed locally on CPU; not a physical sensor evaluation.** The architecture, source, unit tests, and integration bridge are in this folder. Run `python fault_benchmark.py` for the fixed-seed test.

## Experimental protocol

Each of 12 conditions used 100 independently seeded **synthetic 2-second traces** (1,200 trials). Primary and witness channels sampled every 20 ms, then were checked on an 8-frame / 120-ms SOMA window. The monitor checked metadata, staleness, rate outliers, and co-located independent sensor disagreement. Failure meant that at least one channel was masked invalid/suspect in the final three SOMA frames. The test deliberately does **not** simulate full electrical noise, actuator temperature dynamics, clock drift, cable flex, sensor drift correlations, or hardware latency.

| Condition | Monitor detected | Sensor-self-report-only detected |
|---|---:|---:|
| Clean, fault-free | 0 / 100 (correct non-alerts) | 0 / 100 |
| Bounded random noise | 0 / 100 (correct non-alerts) | 0 / 100 |
| Joint sensor stuck while independent encoder moves | **100 / 100** | 0 / 100 |
| Pressure sensor stuck while independent reference changes | **100 / 100** | 0 / 100 |
| Slowly frozen temperature with insufficient reference separation | **0 / 100 (missed)** | 0 / 100 |
| Temperature drift by +4 °C while witness remains correct | **100 / 100** | 0 / 100 |
| Pressure stream disappears | **100 / 100** | 0 / 100 |
| Joint packets replay stale sequence numbers | **100 / 100** | 0 / 100 |
| Temperature samples use wrong calibration ID | **100 / 100** | 0 / 100 |
| Joint sensor itself reports failed | **100 / 100** | **100 / 100** |
| Joint sample becomes outside configured operating range | **100 / 100** | 0 / 100 |
| Joint and its witness freeze identically | **0 / 100 (missed)** | 0 / 100 |

The source-report-only baseline is an intentionally weak illustrative strawman, not representative of a real industrial protection system. These clean, handcrafted scenarios do **not** establish 80% real-world sensitivity, 0% false-positive rate, or any physical safety certification. Thresholds were chosen for this experiment; they are not manufacturing tolerances.

## Actual integration check

The independently recovered U1 epoch-24 checkpoint was verified by SHA-256 (`3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f`).

- Original trained model: **121,827 parameters**.
- Clean observation: sensor quality `[1,1,1]`.
- An intentionally corrupted thermal channel with independently valid witness: quality `[1,0,1]`.
- The network accepted the masked `[1,8,7]` observation and `[1,8,3]` quality arrays, returned finite risk logits, and did not need retraining.
- All **12/12 local unit tests passed**.
- No U1/U2 weights were altered; no human subjects or motors involved.

## What we must solve next

1. A frozen sensor can be plausible until sufficiently independent excitation or redundancy reveals it.
2. Common-mode failures can deceive both paired channels. Add different transduction mechanisms and separate power, mechanical paths and independent MCU error reporting when practical.
3. Calibrate time synchronization, hysteresis, effective contact area, ADC saturation and field electromagnetic interference on actual hardware.
4. Design a richer **body graph** so SOMA can reason about where contact happened; U1's present pressure/thermal/joint scalars are merely the bridge.
5. A separate hardwired physical motor stop and limits monitor must remain independent of Weaver's neural model.

**Deployment decision: no physical autonomy. Engineering validation still required.**
