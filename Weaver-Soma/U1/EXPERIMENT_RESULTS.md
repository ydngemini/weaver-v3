# Recovered U1 experiment results

The following are **historical results from synthetic physics simulations**, documented with the recovered epoch-24 U1 checkpoint, and are not safety certifications or re-runs of new hardware tests.

| Physics regime | Risk average precision | Direct-policy avoidable violations | Validation-tuned risk-decoder avoidable violations | Total violations (risk decoder) | Simulated unavoidable floor |
|---|---:|---:|---:|---:|---:|
| Nominal | 0.885 | 5.18% | 0.36% | 3.73% | 3.36% |
| Changed | 0.955 | 7.18% | 0.27% | 24.91% | 24.64% |
| Extreme | 0.996 | 6.23% | 0.64% | 80.59% | 79.95% |

The extremely high event prevalence in the extreme regime makes average precision hard to interpret in isolation. The learned direct-action head did not meet the 1% avoidable-error target; the separately tuned risk decoder did on this simulator, but total violations remained high.

Original test protocol: 8,400 training episodes, 2,100 development episodes, 2,200 held-out episodes per regime, three action alternatives (hold/ease/withdraw), 30 executed epochs, best epoch 24. Simulator parameter ranges, data generation, joint objective, and ablation scripts are included here for reproducing and challenging the claim.

Ablating the shared workspace previously reduced nominal risk AP from approximately 0.885 to 0.792. Removing the discrete latent codebook had negligible effect. Neither observation demonstrates subjective awareness, nor does the model safely control physical devices.

No recovered binary weight files or private sensor traces have been published in the public repository.
