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

### Learned transcript endpoint experiment

Train this optional SOMA speech endpoint using the text questions from OpenAI's
[GSM8K](https://github.com/openai/grade-school-math) and
[WebGPT Comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons).
The trainer labels full questions and truncated versions of the same question.
It selects the decision threshold on a separate validation split before it
scores the GSM8K test questions and plays the held-out
[OpenAI Cookbook Realtime fixtures](https://developers.openai.com/cookbook/examples/realtime_eval_guide).
The fixtures are public evaluation examples. Use the reference transcripts to
isolate endpoint decisions and the optional PocketSphinx pass to measure the
effect of streaming ASR partials.

```sh
python -m pip install pyarrow==23.0.1 pocketsphinx==5.0.4
python train_voice_endpoint.py \
  --gsm-dir /path/to/grade-school-math/grade_school_math/data \
  --webgpt /path/to/webgpt-comparisons-train.parquet \
  --cookbook-walk /path/to/openai-cookbook/examples/evals/realtime_evals/walk_harness/data \
  --compare-asr --out /path/to/new-endpoint-run
```

The run requires FFmpeg with the Flite filter and writes a JSON model and
per-clip replay report. Fetch the WebGPT Parquet from the dataset's public Hub
export. For reproducibility, the first run used source commits
`3101c7d5072418e28b9008a6636bde82a006892c` (GSM8K) and
`dfd760b4de0bd175b51ad05b9b5260c36dfea09c` (Cookbook). The report includes
the SHA-256 of the data files.

Load the candidate with `TranscriptEndpointModel.load(path)` and pass it as
`endpoint_model` to `RealtimeVoiceSession`. While a turn is active, push the
latest whole-turn ASR partial with
`session.report_transcript(turn_id, text, through_sample)`.
The watermark is the latest input sample covered by that partial. At the
normal 300 ms silence boundary, a fresh transcript can qualify for an
immediate endpoint. Otherwise the session waits for up to 600 ms of silence.
The `speech_stopped` event includes the actual rule, score, threshold, and
each active feature's contribution. Missing packets still have a separate
response-ineligible path. The default session uses the existing fixed
silence policy. The learned model is available for replay experiments.

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

## Camera motion and phone IMU

The optional camera motion path uses timestamped RGB windows to track the
center of contrast in the image and estimate horizontal and vertical motion.
It scores four directions and exposes the measured velocity, usable frame
pairs, and the exact contributions to each class score. It withholds direction
if pairs are missing, motion is weak, or directions conflict. Camera movement,
clutter and lighting can affect the center of contrast.

The optional IMU path expects six 50 Hz channels in order: acceleration xyz
in g, then angular velocity xyz in rad/s. Keep missing samples in their original
time slots and mark them invalid. Per-axis quality of zero excludes an axis.
The causal convolutional encoder emits features for each observed slot and a
six-activity window score. Read coverage and score availability in
`out["motion_evidence"]`. At least 32 observed time slots are required to
score a window. Validation uses 128-sample UCI HAR windows with the original
waist-mounted setup. The score does not establish device-placement transfer.

```python
model = WeaverSOMAU3(fusion_dim=64, vision_encoder="motion", enable_imu=True).eval()
vision = torch.load("runs/motion/vision_motion.pt", map_location="cpu", weights_only=True)
imu = torch.load("runs/motion/imu_activity.pt", map_location="cpu", weights_only=True)
model.motion_vision.load_state_dict(vision["weights"], strict=True)
model.inertial.load_state_dict(imu["weights"], strict=True)
out = model(obs, quality, context, modalities={
    "vision": {"values": rgb_frames, "timestamps": frame_times_seconds},
    "imu": {"values": accel_and_gyro, "quality": per_axis_quality},
})
```

RGB values are floats in [0,1] shaped [B,F,3,H,W]. The camera input accepts a
per-frame validity mask. IMU values and quality are [B,T,6]. When a batch mixes
scorable and unscorable windows, use each row's `score_available` flag.
The token budget prioritizes observed samples before thinning. The CPU trainer
fits separate readouts and verifies both through the normal U3 forward path.

```sh
python U3/benchmark_motion_inputs.py --mnist-dir /path/to/mnist \
  --har-dir "/path/to/UCI HAR Dataset" --run-dir runs/motion
```

[MNIST image files](https://github.com/cvdfoundation/mnist) ·
[UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

## Paired skeleton and wearable action recognition

This optional readout combines Kinect v1 skeleton joints and wearable
inertial data from [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html).
It uses whole segmented trials. It matches recordings by action, participant,
and trial ID and summarizes each stream into eight relative phases. These
phases provide trial-level alignment. Frame-level sync and streaming action
segmentation are outside this experiment.

```python
model = WeaverSOMAU3(enable_paired_actions=True).eval()
ckpt = torch.load("runs/paired/paired_action_head.pt", map_location="cpu", weights_only=True)
model.paired_action_head.load_state_dict(ckpt["weights"], strict=True)
out = model(obs, quality, context, paired_action_inputs={
    "skeleton": joints,  # float [B,T,20,3] in Kinect v1 joint order and units
    "wearable": inertial,  # float [B,S,6] in original UTD-MHAD order and units
})["paired_actions"]
```

Either sensor may be omitted. Optional boolean `skeleton_valid` has shape
[B,T,20] and `wearable_valid` has shape [B,S,6]. Non-finite readings are
excluded. The readout reports class-score contributions, observed coverage,
normalized stream weights, and whether the two sensor predictions disagree.
Its weights are proportional to squared coverage for streams with at least
half the observations. Use `score_available` before interpreting scores.
Disagreement can occur on correctly paired trials.

Install SciPy and use the official Skeleton and Inertial ZIP files directly.

```sh
python U3/benchmark_paired_actions.py --skeleton Skeleton.zip --inertial Inertial.zip --run-dir runs/paired
```

Each sensor encoder is fitted with its own classification loss and a shared
fused loss. The PairedActionHead is an auxiliary readout in the normal U3
forward pass. The general-purpose U3 latent fusion is outside this training.

## Adaptive text, touch, chemical, RF, and spatial inputs

Set `enable_adaptive_inputs=True` to use the contextual byte encoder and the
adaptive numeric stems. Keep batches on one device and use float32 for sensor
values. Existing model options continue to select the audio, video, and IMU
paths independently. Use task checkpoints before interpreting class scores.

| Input key | Profile | Evidence returned |
| --- | --- | --- |
| `text` | UTF-8 bytes [B,T], optional [B,T] validity | Selected byte offsets, separate 150-class intent logits, exact per-token class-score contributions |
| `touch` | Six normalized force/contact channels [B,T,6] | Per-channel observations, usable changes, selected slots |
| `touch_grid` | Calibrated pressure [B,T,H,W] in [0,1], optional validity per frame or per taxel | Mean pressure, contact fraction, center of pressure, peak, spread, and observed taxels |
| `olfaction` | Eight normalized chemical signal channels [B,T,8] | Per-channel observations, usable changes, selected slots |
| `gas_summary` | UCI archive format [B,16 physical sensors,8 descriptors] | Six-class research probe, per-sensor score contributions, deviation from training |
| `rf` | Eight precomputed, normalized RF feature channels [B,T,8] | Per-channel observations, usable changes, selected slots |
| `spatial` | Six normalized spatial features [B,T,6] | Per-channel observations, usable changes, selected slots |
| `spatial_points` | Metric XYZ [B,T,P,3], optional [B,T,P] point validity | Point count, centroid, and axis spread in meters |

Numeric streams exclude invalid values before constructing differences.
Supply per-channel `quality` for the four numeric streams and `timestamps`
when available. Timestamps exclude non-increasing pairs. Keep gaps represented
as masked slots. Short and longer causal filters both run, and observed
change controls their mixture. The router reserves temporal anchors, ranks
other valid tokens by observed signal change, and enforces the configured per
stream cap. Text favors digits and UTF-8 lead bytes for the unreserved slots.
The selected indices are available in `out["adaptive_evidence"]`. This is a
fusion token budget. The front end still examines every input slot.

```python
model = WeaverSOMAU3(fusion_dim=48, max_modality_tokens=48, enable_adaptive_inputs=True).eval()
text = torch.load("runs/text/adaptive_text_intent.pt", map_location="cpu", weights_only=True)
gas = torch.load("runs/gas/gas_clean_training.pt", map_location="cpu", weights_only=True)
model.adaptive_bank.text.load_state_dict(text["weights"], strict=True)
model.gas_summary_head.load_state_dict(gas["weights"], strict=True)
out = model(obs, quality, context, modalities={
    "text": {"values": utf8_bytes, "valid": text_valid},
    "gas_summary": {"values": gas_measurement},
})
```

The contextual byte model learns an intent classification task. The saved
validation threshold is experimental. Gas results are for archived sensor
descriptors, not raw chemical streams or field measurements. The raw touch,
chemical time series, RF and generic geometry stems have synthetic wiring and
fault tests. Their parameters require task training. RF data is supplied after
measurement and calibration. The point-cloud stem is a global geometry summary.

```sh
python U3/benchmark_adaptive_text.py --dataset /path/to/oos-eval/data/data_full.json --run-dir runs/text
python U3/benchmark_adaptive_text.py --dataset /path/to/oos-eval/data/data_full.json --run-dir runs/text-reference --reference-only
python U3/benchmark_gas_drift.py --dataset /path/to/gas-sensor-array-drift-dataset.zip --run-dir runs/gas
python -m unittest discover -s U3 -p test_adaptive_inputs.py
```

The optional text reference requires scikit-learn. The archived gas data
source describes a research-only use case.

[CLINC150 repository](https://github.com/clinc/oos-eval) ·
[UCI Gas Sensor Array Drift](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset)

## CPU inference speed

Long adaptive streams on CPU use a bounded selection pass. It keeps the same
anchors and resolves equal-priority samples in chronological order. Short
streams keep the existing stable sort. In eager CPU evaluation with gradients
disabled, a zero fusion gain lets SOMA reuse the original body heads and skip
cross-attention. Training still executes the fusion path, including at zero
gain.

All existing diagnostics are returned by default. In callers that do not use
the simulator readout, pass `include_quantum_diagnostics=False` to the
existing forward call. SOMA still runs the simulator whenever it contributes
to the result.

```bash
python U3/benchmark_cpu_speed.py --out /tmp/soma-speed-current.json --optional-diagnostics-off
python U3/benchmark_cpu_speed.py --source-dir /path/to/comparison-checkout/Weaver-Soma/U3 --out /tmp/soma-speed-before.json
```

The harness reports warm one-thread CPU inference latency on repeatable
synthetic workloads. The JSON includes a digest for each output tensor to
verify predictions, masks, routing traces, and simulator parity across
checkouts. Timing starts with tensors in memory and excludes capture,
transcription, voice synthesis, and any lookahead needed to fill an input
window.

## Local duplex device pilot

`live_voice.py` reuses the session coordinator with a duplex PortAudio
transport. One callback fills a 20 ms output block and records ADC, callback,
and scheduled DAC timestamps. The worker owns recognition and playback state.
Cancellation clears the next output block the worker can affect. Audio already
submitted to the device may still play. Playback acknowledgements use elapsed
DAC timestamps from the device driver. They describe a scheduling estimate.
A physical loopback microphone or electrical capture is needed for acoustic
mute measurements.

On Ubuntu, use a headset or select the matched source and sink of an existing
PipeWire echo-cancel configuration. Use the route names or indices shown by
the device listing. The pilot accepts 16 kHz mono int16 and uses a local
two-intent PocketSphinx grammar with pre-rendered Flite replies.

```bash
# PortAudio and FFmpeg with Flite must already be installed on the target.
python -m pip install pocketsphinx==5.0.4 sounddevice==0.5.6
python U3/live_voice.py --devices
python U3/live_voice.py --live --input-device INPUT --output-device OUTPUT --trace pilot.jsonl
python U3/benchmark_live_loopback.py --run-dir runs/device-loopback
```

The live trace option records recognized text and timing metadata. Use it
with consent. For the pilot, keep the microphone and output route fixed per
run. Include clean turns, labeled within-turn pauses, self-echo without user
speech, and true interruptions. Use an independent record of source speech
onset and audible output for the device acceptance test.

The six synthetic cases separate interruption, a 420 ms internal pause, and
self-echo. The two interruption tests scheduled mute 100 ms after the
fixture's energy onset and emitted no old-response blocks after that mute.
Timing from file insertion includes leading silence and measured 460 ms.
A 300 ms timeout split the 420 ms pause. A 600 ms timeout kept it together.
Injected echo interrupted playback once. Exact subtraction of the known
fixture echo prevented that interruption. The subtraction is confined to the
synthetic benchmark. Physical audio I/O and the configured PipeWire
echo-cancel pair still require laptop verification.

[PortAudio stream callback and clocks](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html) ·
[PipeWire echo-cancel module](https://docs.pipewire.org/page_module_echo_cancel.html) ·
[Full-Duplex-Bench](https://doi.org/10.1109/ASRU65441.2025.11433838) ·
[Full-Duplex-Bench-v2](https://doi.org/10.18653/v1/2026.acl-short.4) ·
[OpenAI Realtime evaluation guide](https://developers.openai.com/cookbook/examples/realtime_eval_guide)

## Native text decoder pilot

`native_text_decoder.py` implements a small causal Transformer with tied
embedding/output weights and an append-only KV cache. It loads its own
from-scratch checkpoints. The realtime voice pilot still uses its existing
two-intent backend. This decoder experiment evaluates text continuation.

The CPU trainer compares UTF-8 bytes with a 512-token byte-level BPE
tokenizer fitted on the training subset. It uses 1,536 distinct train
stories and 128 validation stories from two published TinyStories V2 files.
Exact duplicate stories across the selected splits are excluded. Both
models see the same training-token budget, with a shared 96-token context
and three logged budgets. Token counts cover different amounts of text
between tokenizers. Validation reports bits per original UTF-8 byte to
compare loss across the vocabularies.

Download the two bounded prefixes from the pinned dataset revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. Range lengths match the
measurement. The trainer discards a partial last record in each prefix and
the leading fragment in the validation file.

```sh
python -m pip install tokenizers==0.23.3
revision=f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
data_url=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/$revision
curl -fL --range 0-2599999 "$data_url/TinyStoriesV2-GPT4-train.txt" -o train.prefix.txt
curl -fL --range 0-349999 "$data_url/TinyStoriesV2-GPT4-valid.txt" -o valid.prefix.txt
python U3/train_native_text.py --train train.prefix.txt --validation valid.prefix.txt --source-revision "$revision" --run-dir runs/native-text
```

```python
from train_native_text import load_native
from native_text_decoder import complete

model, codec = load_native("runs/native-text/soma_native_bpe.pt")
prompt = [codec.bos, *codec.encode("Once upon a time")]
print(codec.decode(complete(model, prompt, max_new=32, stop=codec.eos)))
```

The report includes checkpoint hashes, held-out loss, a unigram baseline,
one-thread warm KV-cache timings with prefill, and numerical parity with
full-prefix replay. The corpus consists of simplified synthetic English
stories. Open-ended dialogue, evidence grounding, and live integration
require separate training and evaluation.

[TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) ·
[TinyStories paper](https://arxiv.org/abs/2305.07759) ·
[ByT5](https://doi.org/10.1162/tacl_a_00461) ·
[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

## Full and LoRA fine-tuning for grounded replies

This experiment starts both methods from the exact native BPE story checkpoint.
Given short records labeled A–D, it trains the decoder to return a short
value with its record label, or `unknown` when the requested field is absent.
The benchmark synthesizes distinct record/question combinations before
training and scores unrestricted greedy text. The objective covers response
tokens and EOS. Prompt and padding positions have zero loss.

Full fine-tuning updates every weight. The native rank-eight LoRA comparison
freezes the base and trains the four linear projections in each decoder
block. Both methods use the same minibatch draws and token budget. Learning
rates are fixed independently and documented in the report. The release
contains adapter-only weights tied to the base hash and a merged checkpoint
that loads with the normal native decoder loader.

```sh
python U3/train_grounded_text.py --checkpoint runs/native-text/soma_native_bpe.pt --story-validation valid.prefix.txt --run-dir runs/grounded-text
```

Add `--story-replay-train train.prefix.txt` to test full fine-tuning with an
extra next-token loss on the original training stories, weighted 0.35. This
arm uses the same supervised draws plus two story windows per update. Its
extra story tokens and runtime are reported separately. This comparison was
added after observing retention loss and uses the same held-out benchmark.

The held-out slices cover new record combinations, an unseen question
template, and values withheld from supervised training. Paired interventions
change one relevant value or remove its record while preserving other
records. The report also measures story continuation loss after adaptation
and includes every synthetic test prediction. Both methods train in native
PyTorch on CPU. This experiment does not measure Unsloth, live voice, or
open-domain evidence grounding.

[LoRA paper](https://arxiv.org/abs/2106.09685) ·
[PEFT LoRA guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora) ·
[TRL supervised fine-tuning guide](https://huggingface.co/docs/trl/sft_trainer) ·
[Replay and forgetting in language-model training](https://aclanthology.org/2025.acl-long.105/)
