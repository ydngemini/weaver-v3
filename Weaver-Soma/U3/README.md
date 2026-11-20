# Weaver-SOMA U3 | Stage 03: first neural implementation

**Branch:** `feature/weaver-soma-u3-multimodal-evolution`  
**Status:** Executable research scaffold and compatibility tests. **Not multimodally pretrained, not AGI, not a real-QPU integration.**

## What has actually been implemented

`model.py` defines **one `WeaverSOMAU3(torch.nn.Module)`**, inheriting the unchanged U2 neural body, experts, memory, deep core, and 15 outputs, with new *internal trainable* paths:

- **Text/bytes:** 256-symbol byte embedding (not yet an autoregressive language model).
- **Vision/video frames:** trainable patch convolution and projection.
- **Environmental audio:** learnable waveform convolutions; sound-event head remains untrained.
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

The new heads return random/untrained predictions until properly trained on trusted datasets. Training is an explicit next task, not something silently accomplished by inference.

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
