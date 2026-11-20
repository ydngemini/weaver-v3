# PURSUIT OF AGI — SOMA U3 Through U5: Full staged architecture and engineering roadmap

**Version:** 1.0 research design, 2026-11-20. **Status:** proposed stages, not implemented or validated. **Repository:** ydngemini/weaver-v3, feature/weaver-soma-u3-multimodal-evolution. **Owner:** Nathaniel Lockwood.

## North star and non-negotiable interpretation

Build ONE natively trained, end-to-end differentiable multimodal neural system, not an agent orchestrating third-party AI products or a loose stack of pretrained external models. A single PyTorch module may contain many specialized internal neural regions, tokenizers and heads, trained together against shared objectives and stored in a coordinated model checkpoint. The reference starting point is the user's own Weaver-SOMA U2. The goal is experimentally testable learning of language/code, vision/video, environmental hearing, touch, motion, spatial representations, physical/chemical olfaction, and additional machine-native senses; later, controlled neural topology growth, connectivity changes and self-proposed learning rules. Any claims about AGI, consciousness or open-ended recursive improvement remain hypotheses, not achievements.

**Differentiate** (A) a sense (physical measurements), (B) an encoder (neural transformation), (C) a representation (learned latent), (D) a learning objective, and (E) a model architecture. Bits/bytes/FP formats are representation/compute formats, not senses. Quantum-inspired algorithms on classical silicon are distinct from calculations on actual quantum hardware.

## Verified foundation and limitations

From the existing U2 branch documentation:
- Architecture: 100,206,197 parameters, ten 1,024-wide attention/MLP cognitive blocks with 2,816 FF dimension, plus U1 sensory encoders, body-state GRU, five experts, global workspace and prediction heads.
- U2 Stage 4 reports 240 CPU AdamW updates and nonzero gradients across 100,082,241 deep-core parameters; it reports better held-out joint synthetic loss in 12/12 sensor-condition groups. These are reported results from checked-in experiment documents, **not** a fresh independent run for this roadmap.
- This is synthetic physics/sensors only, not a natural-language, image, or real-sensing foundation model. The ~400 MB single-file trained checkpoint is outside public GitHub; code alone does not reconstruct its learned values.
- Keep U2 weights and its original 15 outputs as a regression target. A compatibility test must independently verify the checkpoint SHA, load all compatible tensors strictly, and compare baseline output tensors before any U3 change.

## Target architecture (ONE model, many internal neural regions)

1. **Input contracts:** typed modalities, per-sample absolute/relative timestamps, units, sensor IDs, calibration version, data-quality mask, confidence metadata and legal data provenance.
2. **Native modality encoders:** text/code byte or subword embedding; RGB/video patch encoder; audio waveform/mel-spectrogram encoder; tactile/IMU/time-series encoder; 3D/thermal/radar and electronic-nose/chemical time-series encoder. Each is a learnable submodule within the same model.
3. **Latent token standard:** project all encoders into shared D_MODEL (initially choose and profile 512/768/1024), retaining modality/type/time/position/missingness embeddings plus optional private features to avoid information loss.
4. **Shared attention/latent resampler:** compress long asynchronous streams via cross-attention into bounded latent token slots, then exchange information through a single backbone.
5. **Preserved U2 body and cognitive regions:** maintain existing U1/U2 path through gated compatible updates while learning new pathways. Eventually jointly unfreeze with replay.
6. **Shared deep cognitive backbone:** attention as the default; recurrent/state-space, sparse experts, graph-related features and temporal memory are eligible evidence-driven modifications, never mandatory decorative branches.
7. **Shared predictive world representation:** future latent state, uncertainty, causal/action-conditioned prediction on controlled data; learn about changing environments without claiming physical agency.
8. **Native output heads inside the same network:** autoregressive text/code, event tagging, spatial or sensor prediction, reconstruction, contrastive alignment, risk/calibration and (later) image/audio generation as data and compute permit.
9. **Neural evolution region inside model:** learned assessor/proposer that outputs candidate architecture and learning-rule parameters; immutable external runner validates candidates, clones checkpoints and enforces bounds. The evolving core cannot rewrite the independent acceptance, infrastructure permissions or runtime interlocks.
10. **Optional quantum research sublayer:** a small differentiable variational circuit or quantum feature map attached to a low-dimensional latent bottleneck, first on a classical simulator, potentially a physically separate QPU. A genuine hardware QPU is not an ordinary block on a standard AI CMOS chip.

## Experiments and pass gates: Stage N00 → N18

### N00 — Baseline provenance and freeze
**Build:** inventory branches, relevant U1/U2 code, model tensors and tests; restore only trusted checkpoints, hash-verify, reproduce synthetic eval on the real target hardware, publish baseline FLOPs, speed, RAM and variance.
**Gate:** verified full U2 reload; reproduce reference output signatures/values; document missing checkpoint as blocker rather than guessing.

### N01 — Scientific success criteria and threat model
**Build:** task matrix for perception, generation, reasoning, robustness, nonhuman sensing, compute, old-task retention, false positives and calibration. Split private holdouts by subjects, time, devices and scenario; license/provenance controls.
**Gate:** locked independent eval harness; explicit hypotheses, thresholds and stopping rules *before* tuning on holdouts. No benchmark leakage.

### N02 — Single-model contracts and compute engine
**Build:** specify variable-length multimodal batch schema, typed tokens, shared dimensions, coordinate frames, timestamps, missingness, masks; CPU/GPU determinism; mixed precision and checkpoint schema. Use PyTorch single-module forward graph.
**Gate:** unit/shape/gradient tests for all input types; mixed-modality batch and empty-missing-modality runs; bounded sequence lengths; finite gradients.

### N03 — Lossless U2-to-U3 migration path
**Build:** new model container imports U2 weights without changing existing inputs/outputs. Zero-gated novel path gives legacy equivalence at initialization. Save versioned optimizer/calibration and dataset metadata.
**Gate:** bit-exact legacy output on deterministic fixtures where feasible; full report of each old/new tensor and expected compatibility; rollback works. Absolutely no overwrite of U2 checkpoint.

### N04 — Native text, bytes, code and structured data
**Build:** decide byte-level vs subword tokenizer using token-efficiency tests; text embeddings; causal and/or encoder-decoder language tasks, code syntax and JSON structures; data cleaning/dedup.
**Gate:** held-out byte/token perplexity, syntactic tests, code execution *only in isolated tests*, semantic retrieval and generalization to unseen text, compared to same-budget baseline. Do not call a third-party LLM.

### N05 — Image, video and spatial vision
**Build:** from-scratch patch/CNN frontend within U3, frame timestamps and 3D/depth where available, scene/region representations, moving-object prediction.
**Gate:** held-out image classification/localization and frame-event prediction; calibration under lighting/blur/occlusion; mixed vision + text alignment; no claims of validated autonomous driving.

### N06 — Hearing: environmental and biological events, plus speech
**Build:** audio frontend from waveform or spectrogram; multi-label overlapping events, onset/offset timestamps, sound localization where stereo data exist; cough/sneeze/siren/bird/dog/mechanical examples; optional phoneme/ASR heads.
**Gate:** per-class precision/recall/F1, event timing, robustness to noise/room changes/overlap and explicit unknown-sound detection; compare against unimodal baseline. Datasets include FSD50K, ESC-50 and appropriate spatial audio sets.

### N07 — Multimodal fusion and crossmodal learning
**Build:** shared learned tokens and bounded latent resampler; multimodal contrastive/predictive learning, masked reconstruction and language grounding; mix uni- and multimodal training without requiring every sensor present.
**Gate:** crossmodal retrieval/grounding scores, missing-sensor performance vs single modality, latency/token budget and no regression in N00 tasks.

### N08 — Touch, proprioception and physical temporal senses
**Build:** pressure, force-torque, vibration, IMU acceleration/angular velocity, GPS, temperatures and time-series frontend; learn temporal physical events, alignment with synchronized audio/video.
**Gate:** held-out device/session generalization, contact/motion recognition, missing-sensor invariance and timestamp error budgets.

### N09 — Olfaction and chemical perception, first class
**Build:** multichannel electronic-nose/gas/VOC readings with temperature/humidity, sensor identity and calibration/drift; mask-aware chemical temporal encoder; supervised targets only from measured/verified gas or odor labels, plus pair-aligned contexts.
**Gate:** measured-gas class/concentration metrics on held-out days/devices, sensor drift stress, false-alarm/unknown-class analysis, channel dropout and crossmodal tests. Text odor descriptions are NOT physical smell readings. Prototype NOT a certified smoke/CO/gas detector.

### N10 — Beyond-human / machine-native perception
**Build:** additional calibrated physical modalities not available to unaugmented humans: hyperspectral/UV/IR imaging, RF/radar, LiDAR, acoustic ultrasound, magnetic field, precision chemical and quantum-sensor measurements if actual hardware provides them. Let the network learn unsupervised latent variables and clustering without forcing human-language labels.
**Gate:** show predictive value on withheld real measurements versus classical baselines, instrument calibration and randomized controls; investigate new *measurable* features, not supernatural or unverifiable senses. Information-gain active sensing only in simulation or authorized benign laboratory setups.

### N11 — Prediction, multimodal generation and world models
**Build:** unified autoregressive text/code; optional acoustic/visual reconstruction or generative heads, conditional world-state prediction, uncertainty estimates and imagination in simulation. Prioritize forecasting/perception before costly media-generation training.
**Gate:** held-out predictive likelihood/error, long-horizon rollout stability and multi-seed scores. No claim that realistic generated worlds equal understanding.

### N12 — Continual learning and memory
**Build:** replay buffers, distillation from previous checkpoints, selective module freezing/unfreezing, adaptive memory and modality rehearsal; choose memory architecture by benchmark, not branding.
**Gate:** forward transfer, backward retention, forgetting curves, capacity growth and multi-session robustness vs static-model controls.

### N13 — Neural self-model and structural plasticity
**Build:** learned diagnostics of load, uncertainty, gradients and unused capacity; candidate neuron/expert/layer addition, pruning, sparse edge rewiring and function-preserving growth. Migration of optimizer state and checkpoint tensors; computational budgets.
**Gate:** repeatable improvement beyond predeclared margin in 3+ seeds versus equal total train FLOPs, data and tuning controls including random growth; immediate rollback for regressing candidates.

### N14 — Learning-to-learn and self-proposed training rules
**Build:** neural meta-optimizer/proposal region suggests update schedules or parametric learning rules and architecture changes *from day one on a limited research subnet*, then gradually broader candidates. Train proposals in isolated forks and never mutate production/infrastructure while studying learning-rule evolution.
**Gate:** unseen-task performance, optimizer stability, gradient sanity, numerical safety, attack/overfit resistance and reproducibility. Each proposal needs an independent acceptance signal.

### N15 — Quantum and quantum-inspired R&D, OPTIONAL
**Build in sequence:** (a) classical tensor-network and complex-valued/quantum-inspired representations; (b) compare simulator-based differentiable few-qubit variational quantum circuit as an in-model low-dimensional feature transform using PennyLane/Torch or Qiskit; (c) use an actual QPU only if real hardware budget/access and reproducible evidence justify it. Quantum sensor *input* is unrelated to quantum computation.
**Gate:** identical input preprocessing and data, matched hyperparameter search, count end-to-end cost (encoding, shots, training, latency, classical CPU/GPU overhead) and noise robustness. Reject if no fair advantage. One unified mathematical training system is possible, but an actual QPU is a separate physical processor/control environment. Do not claim it makes the system AGI.

### N16 — Safety, evaluation, observability and containment
**Build:** immutable external evaluator, frozen holdouts, dataset registry, audit trails, reproducible checkpoints, rate/resource quotas, isolation, no credential access from evolving code, human deployment approval and independent physical safety interlocks. Dashboards: parameters/active FLOPs, evolution events, throughput, loss by modality, forgetting, calibration, uncertainty, mutation acceptance/rejection, CPU/GPU/VRAM and energy.
**Gate:** adversarial/missing-input tests, tamper attempts fail, restore-to-known-good rehearsed, no safety-gate bypass, independent sign-off for external action.

### N17 — Scaling and hardware implementation
**Build:** choose CPU/GPU cluster requirements from measured workload. Mixed precision, activation checkpointing, flash attention, sparse routing and distributed training only when needed. Establish unified checkpoint storage/checksums, dataset throughput and MLOps. Prototype logical chip floorplan with sensor ingress, DMA, on-chip SRAM, neural compute tiles, shared memory, routing fabric, memory controller and host interface. An ASIC design is a separate multi-year, high-capital verification/fabrication endeavor.
**Gate:** hardware cost and utilization documented; memory fit proven; inference/training benchmarks measured rather than marketing; chip view labeled concept only.

### N18 — Integration, independent replication and release decisions
**Build:** train and validate whole U3 network with old and new modalities, run longitudinal curriculum, controlled evolution trials, and reproducible blind evaluations; publish limitations and scientific artifacts. Branch-release gates for research-only model, lab input streams, and any later controlled product use.
**Gate:** independent multi-seed results, held-out tests, risk assessment, complete rollback, no unresolved blocker. AGI/frontier competitiveness is not a default exit condition or a status label.

## Priority waves and dependencies

**Wave A / prove original model remains intact:** N00–N03.
**Wave B / add broad machine perception:** N04–N09; text+audio+vision first because paired public data are accessible, then touch/olfaction from real sensors.
**Wave C / generalization and adaptation:** N10–N14; nonhuman sensing, prediction and continual learning before unrestricted growth.
**Wave D / scientific options and integration:** N15–N18; quantum is optional, empirical scales/hardware only after proof.

The system may conduct learning-rule research earlier via a **small supervised shadow subnet**; broad weight-space auto-mutation is delayed until independent gates have evidence. Every wave preserves a stable, reversible reference checkpoint.

## Baselines and evaluation (pre-register before experiments)

- **Training metrics:** total/active parameters, training FLOPs, peak GPU/CPU memory, wall time, energy, steps, dataset size/quality, all learning curves, per-seed spread.
- **Language/code:** next-token perplexity, instruction/QA scoring, syntax and controlled execution tests.
- **Vision:** image/event accuracy, mAP/IoU when ground truth exists, temporal generalization.
- **Audio:** multilabel mAP/F1, event timestamps, localization error.
- **Sensor/olfaction:** classification/regression, per-device drift, OOD/unknown detection, sensor dropout, uncertainty calibration.
- **Crossmodal:** retrieval recall, paired temporal matching, missing-modality robustness and counterfactual verification.
- **Evolution:** delta quality under equal total experiment compute, backward retention, complexity cost, error/rollback rate and test leakage control.
- **Quantum:** fair end-to-end objective and total latency/cost against classical parameter/compute-matched alternatives, not qubit count alone.
- **External safeguards:** independent test harness and real hardware emergency cutoffs must NOT be learned or editable.

Never select a checkpoint using the final holdout. Do not extrapolate narrow synthetic performance to general intelligence or physical safety.

## Compute and software stack, initial proposal

- **Core:** PyTorch, one model source graph, reproducible config+checkpoint; tokenizer/audio/image/sensor parsers as data preprocessing, not third-party inference APIs.
- **Training:** one consumer NVIDIA GPU with ~12–24 GB VRAM can support early small/mid-size experiments, using careful batch sizes, gradient accumulation, checkpointing and mixed precision. Fit and speed must be measured; a 100M-class model is not equivalent to a competitive frontier foundation model.
- **Storage:** raw licensed datasets, schema/version tracking, test splits, weight artifact checksums, event logs and separate immutable backups.
- **Quantum:** PennyLane/Torch simulator first; hardware QPU as optional remote coprocessor, not a GPU feature and not embedded in a conventional CPU/GPU die.
- **Budget cadence:** early N00–N07 minimize cloud spending. Future scaling costs are determined by measured tokens, modalities, dataset sizes, hardware, power and trials; no blanket low-cost AGI promise.
- **Minimal proof:** keep old U2 outputs identical; train new text+audio input embeddings into a small shared latent; demonstrate on genuine held-out data; then integrate further modalities.

## Logical chip visualization contract (CONCEPT ONLY)

An AI chip representation should show: outer die/package; sensor/IO PHY at edge; modality encoder tiles (text, vision, acoustic, tactile/IMU, olfactory, RF/3D); shared latent crossbar/noC; central attention/state-space/MoE accelerator tiles; on-chip SRAM caches and HBM/DRAM controllers; learner and structural evolution compute (implemented as programs/operators, not physical rewiring of transistor neurons); safety/control host complex *outside mutable model weights*. Actual smell requires e-nose input hardware outside chip. Actual quantum computation requires a physically separate QPU or remote service; only interface logic is on the CMOS diagram. The chip does not itself learn by physically creating transistors; parameter growth uses allocated memory/compute.

## Critical engineering blockers

1. The large trained U2 checkpoint isn't in GitHub; retrieve exact trusted bytes and verify SHA before claims of preservation.
2. New multimodal encoders are not currently implemented or trained. This document and U3 architecture diagram are plans.
3. Existing U2 240-step synthetic result does not demonstrate real-world generalization.
4. Audio/vision/olfaction joint datasets are sparse and must respect device/time pairing and licenses.
5. Variable-length token fusion is an architectural change from the current fixed four-token U2 trunk.
6. Learning-rule evolution can reward-hack validation; frozen external evaluator and compute-matched controls are mandatory.
7. Quantum usefulness is a falsifiable optional experiment, not a given accelerator.

## Literature and research

- Unified-IO 2 multimodal model: https://doi.org/10.1109/CVPR52733.2024.02497
- ImageBind multimodal alignment: https://doi.org/10.1109/CVPR52729.2023.01457
- Perceiver latent attention: https://doi.org/10.48550/arXiv.2103.03206
- Transfusion unified text+image generation: https://doi.org/10.48550/arXiv.2408.11039
- FSD50K sound event data: https://doi.org/10.48550/arXiv.2010.00475
- Touch and Go: https://doi.org/10.48550/arXiv.2211.12498
- Chemical sensor drift: https://doi.org/10.5772/33411
- Net2Net growth: https://doi.org/10.48550/arXiv.1511.05641
- RigL sparse rewiring: https://doi.org/10.48550/arXiv.1911.11134
- Learned optimizers: https://doi.org/10.48550/arXiv.2009.11243
- Quantum ML documentation: https://quantum.cloud.ibm.com/docs/en/guides/machine-learning
- PyTorch quantum layer: https://docs.pennylane.ai/en/stable/code/api/pennylane.qnn.TorchLayer.html
- Recent quantum ML limits: https://link.springer.com/article/10.1007/s10462-026-11744-x

**Research note:** A broad multimodal, evolving model is an ambitious research program. Reaching AGI or competitive frontier performance is not guaranteed by adding more modalities, neurons, or quantum circuits.
