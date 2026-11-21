# Weaver-SOMA U3 | Stage 03: first neural implementation

**Branch:** `feature/weaver-soma-u3-multimodal-evolution`  
**Status:** Executable research scaffold and compatibility tests. **Not multimodally pretrained, not AGI, not a real-QPU integration.**

## What has actually been implemented

`model.py` defines **one `WeaverSOMAU3(torch.nn.Module)`**, inheriting the unchanged U2 neural body, experts, memory, deep core, and 15 outputs, with new *internal trainable* paths:

- **Text/bytes:** 256-symbol byte embedding (not yet an autoregressive language model).
- **Vision/video frames:** trainable patch convolution and projection.
- **Environmental audio:** legacy waveform convolutions and an optional 16 kHz log-mel encoder. The log-mel checkpoint has five trained research labels.
- **Touch/motion:** mask-aware temporal stem.
- **Olfaction:** eight-channel learnable chemical-sensor stem, quality masks and timestamps. A real chemical sensor has not been attached.
- **RF and spatial arrays:** separate temporal encoders for **measured** signal features (no RF hardware connected).
- **Shared workspace:** typed and position/time-aware sensory tokens, learned attention query from the preserved U2 state, cross-attention, and a zero-initialized trainable residual into the existing latent.
- **Quantum:** `quantum_native.py` contains a differentiable 2–8-qubit RY/RZ/CZ statevector circuit with trainable parameters and Z-expectation measurements. **It runs on a classical PyTorch simulator.** No IBM QPU, pre-measurement quantum sensor, quantum speedup, or multiverse claim.
- **Auxiliary untrained heads:** modality-specific synthetic-task logits and per-byte text probes. Logits do not yet imply reliable audio classification, chemical recognition, or language understanding.

The original U2 block over `z`, memory, broadcast and common tokens is **unchanged**. New modality tokens are fused *after* its original shared-state computation to preserve backward compatibility. A later stage can experiment with deeper early fusion without losing this baseline.

## Recorded local verification, 2026-11-20

Tests were executed **locally on CPU** against source from the U2 Stage 4 research package and the verified U1 epoch-24 checkpoint:

| Check | Observation |
|---|---|
| Verified U1 hash | `3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f` |
| Recovered trained U1 tensors | **91** compatible tensors loaded |
| Inherited U2 model params | **100,206,197** |
| U3 model params (default configuration) | **100,610,471** |
| Additional U3 params | **404,274** |
| Inherited outputs | **15 / 15 bit-identical** with matching base and zero-gated fusion |
| Modality interfaces | **7** (text, vision, audio, touch, olfaction, RF, spatial) |
| Quantum simulator | 4 qubits by default; 3-qubit smoke test passed |
| Backpropagation | Sensory stems, quantum input encoding and quantum gate angles receive gradients |
| Missing and invalid channel masking | No change to original U2 outputs when an entire new stream is unavailable; masked chemical channel perturbation leaves model state unchanged |
| Real sensor training | **Not done** |
| Original U2 Stage 4 401 MB checkpoint | **Not recovered**; no claim its trained deep-core weights were loaded |

The source was tested with a small width/depth setting for CPU gradient/behavior checks, plus a default full-size model instantiation and legacy-output comparison. This validates **wiring and compatibility**, not learned cognitive competence.

## How to run locally

PyTorch and Python 3.10+ are required. Commands below assume the `Weaver-Soma` directory is the current working directory:

```bash
python U3/test_u3.py
# With recovered original trusted U1 artifact:
python U3/test_u3.py /path/to/weaver_soma_unified_RECOVERED.pt
```

Example use with verified U1 weights:

```python
import sys
sys.path.insert(0, "U3")
from model import WeaverSOMAU3
import torch

model = WeaverSOMAU3().eval()
model.load_verified_u1("/path/to/weaver_soma_unified_RECOVERED.pt")
obs = torch.randn(1, 8, 7)
quality = torch.ones(1, 8, 3)
context = torch.zeros(1, 8)
nose = torch.randn(1, 12, 8)      # SYNTHETIC NUMBERS, not odors or a live sensor
with torch.inference_mode():
    result = model(obs, quality, context,
        modalities={"olfaction": {"values": nose, "quality": torch.ones_like(nose)}})
print(result["risk_logits"].shape, result["quantum_expectations"].shape)
```

The model initializes its sensory heads randomly. Load a task-specific checkpoint before interpreting its predictions.

## Five-class audio benchmark

Run from `Weaver-Soma` with CPU PyTorch, NumPy, SciPy and scikit-learn.

```bash
python U3/benchmark_esc50_audio.py --data-dir ./esc50 --run-dir ./runs/audio-logmel
```

The benchmark uses 200 files from the original ESC-50 research subset and compares the native log-mel model to a pooled MFCC/RBF classifier. It selects training settings on source-grouped folds 1–4, fits the final models on folds 1–3, and scores fold 5 once per preset seed. The original waveform benchmark remains available in `train_esc50.py`.

In the CPU run, the waveform result was 8/40 and the log-mel result was 33/40 for each of three seeds. MFCC/RBF reached 34/40. All results cover the same five-class fold 5 subset. This evaluation fold had already been used by the waveform benchmark. The U2 body and multimodal fusion were outside audio training.

```python
checkpoint = torch.load("runs/audio-logmel/soma_u3_audio5_logmel.pt",
                        map_location="cpu", weights_only=True)
model = WeaverSOMAU3(audio_encoder="logmel", fusion_dim=checkpoint["fusion_dim"]).eval()
model.spectral_audio.load_state_dict(checkpoint["weights"], strict=True)
with torch.inference_mode():
    result = model(obs, quality, context, modalities={"audio": {"values": waveform_16khz}})
    logits = result["modality_logits"]["audio"][:, :len(checkpoint["classes"])]
```

Pass normalized 16 kHz mono PCM as `waveform_16khz`, shaped `[batch, samples]`. For variable lengths, pass a boolean `valid` mask alongside it. Use the checkpoint class order and restrict classification to its five trained outputs. The default waveform encoder loads earlier U3 checkpoints through its existing interface.

## Functional self-monitoring experiments

Train the opt-in audio correctness monitor using the saved seed 2501 audio
checkpoint, or train the simulated body forward model independently.

```bash
python U3/train_audio_self_monitor.py --data-dir ./esc50 \
  --audio-checkpoint ./runs/audio-logmel/soma_u3_audio5_logmel.pt \
  --run-dir ./runs/audio-self-monitor
python U3/train_body_self_model.py --run-dir ./runs/body-self-model
```

The audio monitor estimates whether the paired five-class classifier's output
is correct. It is trained on out-of-fold predictions and calibrated on fold 4.
The saved bundle includes the frozen audio weights and the monitor. Inference
returns the original predictions and an optional advisory readout.

```python
model = WeaverSOMAU3(audio_encoder="logmel").eval()
model.load_audio_self_monitor("runs/audio-self-monitor/soma_u3_audio_self_monitor.pt")
with torch.inference_mode():
    output = model(obs, quality, context, modalities={"audio": {"values": waveform_16khz}})
report = output["audio_self_monitor"]
```

Use `samples_observed` to distinguish missing data from recorded silence.
Read `p_correct` only where `score_available` is true. An abstention is
advisory. The monitor accepts too many sounds outside its five classes and has
no validated novelty detector. Recalibrate when changing the encoder, signal
pipeline, or operating environment.

Each score carries an audit trace with the original summary values, named
feature contributions, intercept, threshold, and the loaded checkpoint SHA-256.
The intercept plus feature contributions reconstructs the reported log odds.
Reason codes are defined by `AudioDecisionReason`. Traces describe the linear
correctness monitor. The acoustic encoder requires separate intervention tests
to investigate its learned representations.

The simulated body model predicts motion with a copy of the proposed command
and five recent command/response pairs. Its trainer compares paired command
reversals against a no-command ablation, a state-and-command model, and a
linear baseline. The trainer saves the linear weights and verifies the exact
input and command-reversal decompositions. It requires no hardware. The residual is a prediction
mismatch and does not identify what caused the mismatch.

## Offline audio intervention audit

Replay real ESC-50 audio through the paired correctness monitor and test the
effect of an isolated change on the audio classifier or the fusion path.

```bash
python U3/audit_audio_interventions.py --data-dir ./esc50 \
  --monitor-checkpoint ./runs/audio-self-monitor/soma_u3_audio_self_monitor.pt \
  --run-dir ./runs/audio-intervention-audit
```

The audit records the checkpoint and audio hashes, the original and changed
outputs, sham and restoration controls, matched waveform interventions, and the
weights of the diagnostic probes. It fits probes on folds 1–3 and assesses them
on fold 5, which was used in earlier audio experiments. The fusion comparison
uses an isolated copy for the explicitly opened gate. The primary model is
created for replay and its weights are kept fixed. These tests measure the
effects of controlled edits under the recorded configuration.

## Audio scope research monitor

The scope monitor measures similarity to the five trained sound categories.
It can be loaded beside the correctness monitor to expose a separate advisory
decision. A low scope score can reflect an unfamiliar sound or an impaired
recording. Use both readouts to inspect the uncertainty, and measure coverage
on the target dataset.

```bash
python U3/benchmark_audio_scope.py --data-dir ./esc50 \
  --monitor-checkpoint ./runs/audio-self-monitor/soma_u3_audio_self_monitor.pt \
  --run-dir ./runs/audio-scope
```

```python
model = WeaverSOMAU3(audio_encoder="logmel").eval()
model.load_audio_self_monitor("runs/audio-self-monitor/soma_u3_audio_self_monitor.pt")
model.load_audio_scope_monitor("runs/audio-scope/soma_u3_audio_scope.pt")
with torch.inference_mode():
    output = model(obs, quality, context, modalities={"audio": {"values": waveform_16khz}})
scope = output["audio_scope_monitor"]
```

The checkpoint binds to the audio self-monitor's SHA-256 digest. Read the
score only when `score_available` is true. The audit contains the score's
exact contributions and the threshold. For the prototype method it also
reports the nearest prototype. That class describes the scope comparison,
while the original classifier predicts the sound label independently.

The benchmark selects the candidate and thresholds on calibration clips,
then reports a held-out test with different ESC-50 sound categories. It also
reports test acceptance at equal familiar-sound coverage as a retrospective
comparison. The fixed thresholds can behave differently under noise, clipping,
packet loss and echo, and generalization to live captures remains unverified.

## Temporal look-alike benchmark

The experimental temporal scopes add mean spectral change across short and
long token intervals. The nearest-neighbor variant records its three fitted
reference clips and the exact contribution of each feature to its distance
score. A clip with too little contiguous token evidence returns an unavailable
temporal score.

```bash
python U3/benchmark_temporal_scope.py --data-dir ./esc50 \
  --monitor-checkpoint ./runs/audio-self-monitor/soma_u3_audio_self_monitor.pt \
  --incumbent-checkpoint ./runs/audio-scope/soma_u3_audio_scope.pt \
  --run-dir ./runs/audio-temporal-scope
```

The run stores an optional research checkpoint and a per-clip audit. Two
sets of look-alike classes are kept separate from the original fit and
calibration categories. The first checks the temporal candidates. The second
checks a combined rule that uses the lower of the two scaled margins. Compare
both fixed and equal-coverage results before using a temporal candidate.
The reference run favored the original prototype on both cohorts of novel categories at matched familiar coverage. Use the temporal checkpoint for research.

## Local realtime voice and environmental audio

The audio window buffer accepts mono 16 kHz PCM16 or floating-point samples
with absolute sample offsets. Seal an offset when the receiver's reorder
deadline has elapsed. Each returned window distinguishes received samples,
usable observations and gaps. Use `window.modalities()` with U3's forward
method for classification and the advisory audio monitors.

The voice session reuses the buffer with 20 ms frames. It emits
`speech_started`, `speech_frame` and `speech_stopped` so an attached
recognizer can process audio before turn end. The reference RMS detector
uses three voiced frames for onset and fifteen quiet frames for endpoint.
Missing frames retain their own status and can stop a turn with no response
eligibility. Use a separate session per caller and serialize mutations.

Responses carry their own IDs. Keep delivering microphone packets during
playback. When `playback_clear` arrives, clear queued output for that
response and discard any stale chunks. Report acknowledgments as
cumulative played samples for each response. The heard transcript includes
fully acknowledged, explicitly aligned speech segments. The interruption
event uses the last acknowledgment received when the turn switches.

The bundled demo uses PocketSphinx with a two-intent grammar and FFmpeg
with the Flite filter. Its inputs and replies are synthetic local audio.
It exercises partial transcripts and an interruption without a physical
microphone.

```bash
python -m pip install pocketsphinx==5.0.4
python U3/demo_realtime_voice.py --run-dir ./runs/realtime-demo
python U3/test_realtime_voice.py
python U3/benchmark_audio_packets.py --data-dir ./esc50 \
  --monitor-checkpoint ./runs/audio-self-monitor/soma_u3_audio_self_monitor.pt \
  --scope-checkpoint ./runs/audio-scope/soma_u3_audio_scope.pt \
  --output ./runs/packet_replay.json
```

The interaction reference is [GPT-Realtime-2.1](https://developers.openai.com/api/docs/models/gpt-realtime-2.1)
and [GPT-Live-1.1](https://developers.openai.com/api/docs/models/gpt-live-1.1).
The prototype currently exercises turn detection, cancellation, a
replaceable audio output path and acknowledgment-based history. Native
open-ended speech reasoning, hands-free echo cancellation, and physical
device latency measurements are future validation work.

## Checkpoint handling

- `load_verified_u1(path)`: hashes against known trusted U1 artifact, validates all 91 original base tensor keys/shapes. U2's deep core and U3's new weights remain **random**.
- `load_verified_u2_stage4(path)`: requires original reported **400,912,211-byte** file whose SHA-256 is `14d170d70f7d16d2abc23721f4f4cc3a9b3d115651dffbb2e343ebfb1f135149`, and validates the stage-four checkpoint metadata and exact base tensor keys/shapes. **No such file was supplied or loaded during these tests.**
- There is no baked-in path to a private local or cloud computer; model weights are not downloaded implicitly.
- No live Weaver/Neoh service, actuator or physical quantum sensor was touched.

## Immediate next gates

1. Commit the implementation and regression tests to the existing U3 branch (not `main`).
2. Train the **untrained sensory stems on legitimate task data**, especially environmental sound (sirens, coughs, sneezes, animals) and real olfactory arrays with device/time holdouts. Record train/dev/test provenance and compute.
3. Demonstrate crossmodal alignment and old-task retention, including a saved U3 research checkpoint.
4. Recover the historical trained U2 Stage 4 model from the original ChatGPT Work task **if still accessible**. Strictly verify before migrating; otherwise disclose fresh training.
5. Treat the quantum simulator as an ablation to compare against an equally resourced classical MLP. Later optional IBM and quantum-sensor work require separate access and hardware.

For the longer research scope, read [ARCHITECTURE.md](./ARCHITECTURE.md), [ROADMAP_00_TO_17.md](./ROADMAP_00_TO_17.md), and [NATIVE_QUANTUM_COSMIC_RESEARCH.md](./NATIVE_QUANTUM_COSMIC_RESEARCH.md).


## First supervised training smoke test (synthetic only)

`train_smoke.py` freezes all inherited U1/U2 tensors and trains only U3's new sensory components with 80 AdamW steps on deliberately simple, procedurally generated binary tasks. This was executed on CPU with the verified U1 file and **a small research core configuration** (`core_dim=64`, `depth=2`, `ff_dim=128`, `fusion_dim=64`, three simulated qubits); it is not a 100M-parameter training run.

On a separate synthetic evaluation draw (256 examples), olfactory-pattern accuracy moved from **70.3125% to 100%**, and synthetic sine-wave audio-pattern accuracy from **52.734375% to 100%**. The two tasks are easy by construction and use the same synthetic generator family for training and evaluation; neither number represents actual odor identification, cough/siren recognition, generalization to new sensors or quantum advantage. No trained checkpoint was saved.

```bash
python U3/train_smoke.py
# Or with recovered original U1:
python U3/train_smoke.py /path/to/weaver_soma_unified_RECOVERED.pt
```

Next **scientific** training requires appropriately licensed real audio and calibrated chemical-sensor datasets, stricter held-out devices/times, actual U3 saved weights, and compute-matched baselines.

## N13 research module added: neuron growth and learned proposals (2026-11-20)

The U3 branch now includes `evolution.py` and `test_evolution.py`.

- `widen_feedforward(ff, sources)` creates a **separate candidate copy** of a `Linear -> elementwise activation -> Linear` feed-forward block. It duplicates chosen hidden units and divides outgoing weights among copies, preserving the block's mathematical function (modulo floating-point roundoff). It never mutates the original block and does not migrate optimizer state; the candidate needs a recreated optimizer and independent evaluation.
- `EvolutionProposalHead` is **5,254 additional learnable parameters** registered under `WeaverSOMAU3.evolution_head`, not a third-party agent. It processes eight finite diagnostics and emits untrained logits for four actions (no change / widen FFN / sparse rewiring trial / learning-rate trial) plus bounded growth fraction and learning-rate multiplier suggestions.
- `WeaverSOMAU3.forward(..., evolution_diagnostics=...)` optionally adds `evolution_proposals` to the output. Calling it without diagnostics keeps the existing legacy-only interface and output semantics. The neural proposal head is not trained, and none of its proposals are automatically applied.
- Standalone PyTorch CPU tests on a separate ephemeral environment **passed** function-preserving widening on 15 combinations, input guard cases, and the differentiability of the proposal network. A *new integrated U3 test* has been committed but **not yet executed on the full U3 source at this revision**. Existing recorded CPU model tests apply to the previous scaffold.
- The previously documented U3 parameter count of 100,610,471 was measured **before** the extra 5,254-parameter proposal head. The predicted updated count is 100,615,725; remeasure when running an actual U3 test before treating that as verified.
- The evolving head does not have permission to edit model code, execute deployment commands, change safety rules, or apply structural changes. Candidate selection, mutation and independent test gates are future stages, not current capabilities.

**Blocking conditions for full integration/training:** historical trained Stage 4 U2 checkpoint not yet restored; connected Linux box had only about 142 MB free in a read-only check on November 20 and its system Python did not have PyTorch. Do not install gigabyte-scale packages or download full audio datasets on the nearly full root drive. Use an approved roomier GPU/CPU research environment for next tests; do not alter active Weaver or Neoh services.

**Next exact validation:**
```bash
cd Weaver-Soma
python U3/test_u3.py
python U3/test_evolution.py
```
These are research checks, not AGI or deployed autonomous neural evolution.
