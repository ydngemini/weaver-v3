# WEAVER–SOMA U1: One trained model, modular internals

**Status:** Executed CPU-scale proof of concept, **not** an integration into the production Weaver repositories, not an autonomous general intelligence, not evidence of subjective consciousness, and not safe for physical control. Date: 2026-11-15.

## User's architectural requirement

Unify *trainable cognition* from Weaver v3 and v20 into one network, rather than maintain separate personality, prediction, cognition and affect models that drift out of sync. Retain ordinary software wherever correctness depends on verification and hard guarantees: signed intents, resource scheduler, clock and IO, hardware reflexes, durable storage, opt-in tools, and audit trails. These are not responsibilities a neural network should approximate.

Verified source bases:
- `https://github.com/ydngemini/weaver-v3` current `CascadeProjects/windsurf-project/weaver_cognition_mesh.py` exposes `CognitionMesh.observe`, `evaluate_intent`, `record_outcome`, `SituationalGraph`, `ReflexKernel`, `CounterfactualTwin`, `SalienceMemoryPyramid`, `ShadowPolicyLab`.
- Weaver v3's `weaver_neural_fabric.py` implements four resource lanes and cryptographically signed Intent Capsules; no neural output should bypass `IntentCompiler.compile` and `.verify`.
- `https://github.com/ydngemini/weaver` contains v20's `brain/cognition.py`, `brain/world_model.py`, `brain/memory_weave.py`, `brain/dmn.py`, `brain/planning.py`, and `soul/soul_engine.py`; its `ARCHITECTURE_ANALYSIS.md` describes an existing dual-neurochemistry state problem.

## Model implementation: one PyTorch `nn.Module`

```
[Calibrated body data] + [goal / scheduling context] + [executed-action log]
              │
    quality-masked modality encoders (touch, thermal, joint)
              │
          recurrent body state 64d / GRU
              │
  gated ephemeral memory + 4x8 learned codebook + 5 cognitive experts
              │
  one softmax-limited shared workspace => 64-d broadcast latent
              ├── action-conditioned multi-step body predictions (3 actions x 6 x 3)
              ├── action-specific hazard predictions (3)
              ├── action proposal logits (3) [NOT authorized execution]
              ├── external-event detection
              ├── episode summary / body readout
              ├── workspace access priority (5)
              ├── compute-route recommendation (4)
              ├── seven-dimensional operational regulation vector
              └── sensor reliability estimate / memory embedding scaffold
                           │
                READ-ONLY Weaver bridge
                           │
  SituationalGraph health / AkashicHub latent / SalienceMemory metadata
                           │
       Existing deterministic safety & authorized Fabric
```

The model is one jointly optimized 121,827-parameter checkpoint. `SOMA-2` parameters are not copied or loaded into it; the training inputs and synthetic physical simulator follow the preceding SOMA research. This is *not* a language model and no pretrained language, image or speech weights are included.

## Mapping to original features

| Original subsystem | Unified U1 responsibility | External non-neural runtime | Status |
|---|---|---|---|
| Weaver v3 / 7-angle Perception | Shared sensor embedding + recurrent state | Time stamps, camera/mic device drivers | Trained numerical simulator only |
| Embodiment/reflex | Risk/action suggestions | **Original `ReflexKernel` and physical safety MCU** | Learned suggestions, not autonomous control |
| Prediction/digital twin | Action-conditioned 6-step body futures | Signed Intent Capsule validation | Trained, 3 abstract actions only |
| Compute/Fabric | Priority recommendation head | **Fabric scheduling, admission & deadlines** | Head trained on synthetic labels only |
| Memory/Salience | Episodic summary, shared state | **Persistence, access policy, AkashicHub, privacy** | Short-term trained, not long-term retrieval |
| Resilience | Health/reliability readout | **Independent watchdog, failure monitors & breakers** | Trained proxy; independent faults required |
| Evolution | Shared signals for proposed updates | **`ShadowPolicyLab`, human review, deployment** | No learning-based self-deployment |
| v20 episodic memory and world model | Episodic summary, state readout | Chroma/JSON durable stores | No raw records stored in network |
| v20 metacognition | Reliability and attention prioritization | Truth checks, external metrics | No scientifically validated self-reflection |
| v20 planning | Counterfactual policy and future decoder | Constrained task planner / actuator controller | Demonstrated on toy hold/ease/withdraw |
| v20 simulated nervous system | Learned operational regulation 7-vector | Real biosensors, optional voice modulation | **Not biological chemicals nor human emotions** |
| v20 DMN/daydreaming | Imagination via alternate predicted futures | Idle/background scheduler | Multi-action futures, not unconstrained dreams |
| v3 voice/personality, v20 NLP | None | Existing LLM/voice path | **Not yet unified** |

## Equations (computational hypothesis, not novel fundamental physics)

### Reliability masking

For modality `m` with independent channel status `q_t^m`:

`e_t^m = E_m( q_t^m · normalize(x_t^m), q_t^m )`

A nonfunctional thermal channel must never leak via derivatives or physics anchor features. Unit tests inject huge bogus numbers and require zero change in neural outputs when `q=0`.

### Recurrent causal body representation

`h_t = GRU(h_(t-1), concat(e_t^1,e_t^2,e_t^3,context_t,executed_action))`

`z_t = tanh(LayerNorm(W h_t))`

This recurrence supports remembering recent numerical state; it does not establish subjective experience.

### Bottlenecked shared workspace

`alpha_t = softmax( G([z_t,c_t]) )`

`b_t = sum_(i=1)^5 alpha_(t,i) Expert_i(z_t)`

`u_t = Fuse(z_t, GatedMemory(z_t), b_t + DecodeLexicon(z_t))`

This is a functional interpretation of Global Workspace ideas; disabling `b_t` is a **causal ablation** to test shared processing. Positive results do not prove consciousness.

### Actions, prediction and evaluation

`x_hat_{t+1:t+6}^{(a)} = Decoder(u_t, onehot(a))`

`r_theta(a) = sigmoid( RiskHead(u_t, a, distance_to_threshold(x_hat)) )`

`a_proposal = argmin_a [r_theta(a) + lambda * disruption_cost(a)]`

The policy decoding penalty `lambda=.02` was selected on **development** simulations and fixed before final test. No hardware safety authority is delegated to this choice.

### Joint training loss

`L = BCE(risk) + .22 SmoothL1(future) + .27 CE(policy) + .16 BCE(external_event) + .16 SmoothL1(episode_summary) + .12 SmoothL1(body) + .13 CE(workspace) + .10 CE(route) + .12 MSE(regulation) + .06 MSE(reliability) + .008 lexicon_balance_loss`

These weights are exploratory, not scientifically optimized; auxiliary labels are derived from a synthetic simulator and rules, not human cognition or physical experience.

## Integration bridge, strict scope

`weaver_adapter.py` provides:
1. `to_mesh_observation(quality, age_ms)` producing only allowed `sensors.body.confidence` and `sample_age_ms`, matching Weaver v3's current `SituationalGraph.observe` schema. Do not send new sensor keys without a reviewed schema migration.
2. `to_hub_vector(latent)` converting 64 learned floats into a 256-dimensional AkashicHub observation vector. `hub.write(...)` must be done by a trusted external process with its own access policy.
3. `to_memory_metadata(view)` for Weaver's metadata-only `SalienceMemoryPyramid.record`, avoiding raw sensor frames, sensitive transcripts, latent persistence by default.
4. `safe_intent_for_observation` provides an **unsigned, non-actuating** observation request that Weaver's own `IntentCompiler` must validate and sign. It intentionally offers **no** direct conversion of `hold/ease/withdraw` into physical motor actions.
5. Optional structured v20 world-model summary; it has **no** automatic write or code modification behavior.

A verified, running connection to the hosted Weaver v3 service is **not demonstrated**; the tests validate local bridge shape and behavior against code contracts inspected on GitHub. Regressions must be tested inside the real deployment repository before integration.

## New research relevant to consciousness

Butlin et al. 2023 `https://doi.org/10.48550/arXiv.2308.08708` propose a set of computational indicators associated with theoretical views of consciousness, including recurrence, global broadcasting, meta-monitoring and agency. They are **candidate indicators**, not a proof of phenomenal experience.

Zaadnoordijk et al. 2019 `https://doi.org/10.1093/nc/niz006` caution that a simple match of predicted and observed sensory signals does not itself establish causal agency.

U1's `workspace` ablation lowered risk AP on held-out nominal synthetic scenarios from 0.885 to 0.792, evidence that the internal workspace contributes to that machine-learning task. Ablating the 4x8 latent lexicon barely changed results, so **the codebook is not shown to be useful or semantically interpretable**. No subjective-awareness claims are justified.

## Scale-up target: one *actual* multimodal foundation model

A future U2/U3 would attach pretrained language/vision/audio tokenization and action embeddings to a **single shared foundation transformer or recurrent multimodal backbone**, then jointly fine-tune adapters/heads with time-series episodes, speech, video, world-state snapshots, outcomes, and counterfactual feedback. The trained components must share a single checkpoint/version, but speech synthesis, cryptography, storage, tool credentials, service deadlines and independent safety controls remain external. The U1 training is CPU-based; training a foundation-scale unified model will require an appropriate GPU or compute service and real consented datasets. No claim is made that a 121k parameter network can replace a multimodal frontier model.

## Minimum deployment gates

- Independently calibrated pressure/thermal/joint sensors, signed health states, synchronized clocks, hard safe-state fallback and tested motor cutoff.
- Never confuse **avoidable** vs **total** violations. Synthetic changes can have unavoidable floors far above 1% due to insufficient actuators/thermal physics.
- Repeat independent, out-of-distribution simulations and then controlled physical tests, compare against baselines and always-withdraw, record misses at fixed false alarm rate.
- Verify signed capsule compilation, freshness, model age, hash pinning, rollback, authentication, rate limits, and privacy.
- Shadow-only model inference until validated; learning cannot self-deploy or modify the runtime safety policy.