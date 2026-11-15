# Weaver integration: one trained core, old systems as services

## Verified integration contracts

- `weaver-v3/CascadeProjects/windsurf-project/weaver_cognition_mesh.py` has `CognitionMesh.observe(payload)` and `mesh.memory.record(...)`, with `sensors` only accepting **body**, environment, camera, microphone. No raw tactile pressure/temperature arrays accepted.
- `weaver-v3/CascadeProjects/windsurf-project/weaver_neural_fabric.py` has `IntentCompiler.compile(request)` / `verify(capsule)`; valid observation action is `{'type':'observe','sensor':'body'}`. Signing secrets must never be given to an LLM/model.
- `weaver-v3/CascadeProjects/windsurf-project/akashic_hub.py` has `AkashicHub(dim=256).write(lobe_id, vector, meta=None)`.
- `weaver-v3/CascadeProjects/windsurf-project/pineal_gate.py` routes five expert lobes; the new network has its **own** five trainable experts. Choose ONE source of cognitive ranking to prevent duplicate policy decisions. Older gate stays until A/B testing has validated replacement.
- `weaver/brain/world_model.py` supports `update_state(**updates)`, a JSON-based store. This is persistent storage, not neural weights. In the next migration, v20's two neurochemistry layers should be consolidated rather than both independently updated.

## Read-only shadow test pseudocode

```python
from weaver_adapter import WeaverSOMABridge
# import actual Weaver libraries only inside your trusted backend process:
# from weaver_cognition_mesh import CognitionMesh
# from weaver_neural_fabric import IntentCompiler
# from akashic_hub import AkashicHub

bridge = WeaverSOMABridge('results/weaver_soma_unified.pt')
view, latent = bridge.infer(obs, independently_verified_health, task_context, executed_action)

# Existing Weaver mesh must receive an allowed and validated schema.
mesh_observation = bridge.to_mesh_observation(independently_verified_health, age_ms=20)
mesh.observe(mesh_observation)

# Optional in-process state sharing only; no raw media payloads.
await hub.write('soma_unified', bridge.to_hub_vector(latent))

# Advisory-only metadata to salience memory; no automatic cloud persistence.
mesh.memory.record(**bridge.to_memory_metadata(view))

# Authorized intent is compiled and signed by the established Weaver Fabric.
# Only OBSERVE is requested, not a movement.
request = bridge.safe_intent_for_observation(mesh.awareness.snapshot())
capsule = authorized_intent_compiler.compile(request)
assert authorized_intent_compiler.verify(capsule)
# mesh.evaluate_intent(capsule, fabric=fabric.snapshot())  # may still reject
# Do not execute a capsule or command motors from neural output.
```

## Stage gates

1. Mirror only validated source files and pinned dependencies into a **side branch**, with shadow-run harness and fixed log schema. Do not change main or deployed service before review.
2. Test schema/backward compatibility against the actual v3 unit suite (this package currently tests **prototypes against inspected schemas only**).
3. Add real sensor adapter with timestamps, calibrated physical units, signed independent health bits, stale timeouts, overlap checks, and emergency stop in MCU.
4. Compare shadow model vs existing digital twin on identical tasks, test OOD and sensor-failure scenarios and p50/p95 latency under voice load.
5. If validated, migrate ONE learned representation to the shared core at a time and retain runtime protections. Make sure the trained model cannot change system files, run shell commands, forge signed intents, bypass guard, or deploy its own weights.

## Large-scale endgame

Current U1 is a 121k numerical demonstrator. The intended **single truly multimodal model** needs a pretrained transformer or hybrid recurrent foundation trunk, learned event-time/action/body token embeddings, camera and audio encoders, text generation, episodic retrieval conditioning, and calibrated uncertainty. Train/fine-tune one checkpoint with large GPU capacity, real consented sensor-labeled episodes, synthetic curricula, and external safety checks. No such full-scale model is trained in this package.