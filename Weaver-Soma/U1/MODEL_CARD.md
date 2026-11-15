# Model card: WEAVER-SOMA U1

## Model facts

- Purpose: compact experimental **jointly trained, multi-task PyTorch neural core** for robotic sensory forecasting and cross-task representations.
- Trainable parameters: **121,827** across one `WeaverSOMAUnified(nn.Module)`.
- Data: **synthetic only**, from `physics_simulator.py`, plus synthetic disturbance and context labels. No camera pixels, raw language, audio, private chats, real motor outputs, or clinical pain measurements.
- Optimization: AdamW, 30 epochs executed, best validation checkpoint **epoch 24** selected on 2,100 development examples. Training data: 8,400 independent simulated episodes, equal nominal/changed regimes. No examples from extreme regime used for training or validation.
- Test: 2,200 held-out episodes each nominal, changed, extreme; tests can share the same protocol/simulator family and are not equivalent to validation on a different real world.
- Outputs: action-specific 6-step predictions and crossing risks, action proposal, external-observation score, 5-way attention target, 4-way compute routing advice, episodic summary, reconstructed current body, seven-dimensional abstract regulation, sensor quality score, 4 discrete body token IDs, latent vector.
- Connections: `weaver_adapter.py` implements read-only messages matching source code of v3 Cognition Mesh, v3 IntentCompiler requests, and a 256-dimensional AkashicHub vector. This is an interface-level prototype; the running Weaver service was **not** modified or accessed.

## Evaluation

Nominal / changed / extreme **risk average precision**: 0.885 / 0.955 / 0.996. Note extreme base prevalence is very high. Independent test-set action choices and counters were computed **without using future outcomes at inference time**; future labels were used for evaluation only.

| Regime | Direct trained action-head avoidable violations | Validation-tuned risk decoder avoidable violations | Risk-decoder total violations | Simulated unavoidable floor |
|---|---:|---:|---:|---:|
| Normal | 5.18% | 0.36% | 3.73% | 3.36% |
| Changed | 7.18% | 0.27% | 24.91% | 24.64% |
| Extreme | 6.23% | 0.64% | 80.59% | 79.95% |

*No learned action-selection method satisfies a 1% **absolute** violation target in these test environments. The selected risk-based decoder has <1% **avoidable** violations on these synthetic held-out episodes. Always-withdraw has 0 avoidable violations by definition but incurs 100% withdrawal rate and still cannot beat the simulated physical floor.*

## Honest interpretation

**Supported in this code:** a single shared learned state improves multiple synthetic tasks; removing workspace broadcast hurts risk and route prediction; missing-channel masking is numerically invariant; heuristic risk ranking is better than the directly trained action head on this test distribution.

**Not supported:** subjective experience, autonomous reasoning/general intelligence, genuine long-term autobiographical memory, human-like emotion, biological pain, safe physical interaction, a novel universal physical equation, trained language/voice/vision generation, demonstrably correct active-inference agency, validated uncertainty calibration, a cryptographically signed neural intent, or online execution in Weaver v3.

**Important ablation:** learned 4-slot quantized body vocabulary did **not** contribute measurably on current benchmarks. Semantic labels must not be assigned to the integer codes without probe validation and ablation improvements. The `semantic_memory_key` head is only a scaffold and has no supervised retrieval objective; it must not be treated as a reliable semantic search vector.

**Hardware caveat:** These are illustrative thermal/pressure/joint thresholds. Physical safety limits require material testing, time synchronization, accredited sensor calibrations, an independent safety controller with e-stop, and hazard analysis.

## Reproducibility

Use `python train_unified.py --epochs 30 --train-n 8400 --val-n 2100` and then `python evaluate_policy.py` and `python evaluate_ablation.py`. `pytest -q test_unified.py` covers construction, output shapes, health-mask invariance, strict adapter payloads, and invalid-input rejection.