# SOMA U3 Stage 00: checkpoint recovery audit

**Status: BLOCKED ON HISTORICAL TRAINED U2 CHECKPOINT | 2026-11-20**  
**User clarification:** the historic trained checkpoint was generated under **ChatGPT Work**, not on the connected Linux machine. The physical quantum sensor is on hold. This audit is read-only and does not alter model weights or train a replacement.

## Target checkpoint
- Filename: `WEAVER_SOMA_U2_100M_Stage4_FULL_MODEL.pt`
- Claimed length: **400,912,211 bytes**
- Expected SHA-256: `14d170d70f7d16d2abc23721f4f4cc3a9b3d115651dffbb2e343ebfb1f135149`
- Expected metadata: `architecture == "WeaverSOMA100M-stage4"`, `format_version == 1`, `total_parameters == 100206197`, `no_motor_authority == True`.
- Required test from `Weaver-Soma/U2/load_stage4.py`: `torch.load(path, map_location="cpu", weights_only=True)`, verify metadata, instantiate `WeaverSOMA100M()`, and call `model.load_state_dict(obj["weights"], strict=True)`. Evaluate the original 15 outputs on a fixed input probe. **No test of the target file has yet run.**

## Confirmed available
- ChatGPT Library: `weaver_soma_unified_RECOVERED.pt`, 515,038 bytes, SHA-256 `3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f`. The hash matches the historical U1 epoch-24 weight reference, **not** U2 Stage 4.
- Prior U2 compatibility session reportedly loaded this U1 checkpoint into the 100,206,197-parameter U2 architecture and observed exact equality across the 15 legacy outputs when newly added gated paths were disabled. Keep separate from proof of U2 Stage 4 weight restoration.
- ChatGPT Library `WEAVER_SOMA_U2_Stage4_100M_CPU_Code_and_Results.zip`, 142,122 bytes compressed, contains **15 files, 218,948 bytes uncompressed**, including Stage 4 source, metrics and plots. **It does not contain the 400.9 MB checkpoint.** Confirmed by inspecting the ZIP manifest.
- Other SOMA research archives in Library are smaller and do not match the target's size. A Library listing limited to checkpoint formats `.pt/.pth/.ckpt/.safetensors` returned the 515 KB U1 checkpoint only.

## Recovery surfaces checked
- ChatGPT Library (semantic/title search and recursive listing by size and weight file extensions): target **not found**.
- Current conversation attachments: target **not present**.
- ChatGPT Space pages and Drive-backed Spaces accessible to this account: no matching Work task/page found (these are not necessarily the same thing as ChatGPT Work cloud-computer tasks).
- Connected browser tab access: unavailable outside its originating browser/extension context.
- Original connected Linux machine and external mount: no matching visible filenames in scoped search; a subset of directories was inaccessible. User states target is in ChatGPT Work, making those paths the wrong primary recovery target.

**Conclusion:** A prior Work session is the most likely location, but the current connected tools do **not** expose its cloud-computer artifact store or earlier Work task file manifest. The target binary remains **unrecovered**. Do not imply its absence from all of ChatGPT, only from currently accessible surfaces.

## Exact next recovery operation
1. Open the **November 16, 2026 ChatGPT Work task/conversation** that produced the trained 100M checkpoint.
2. Locate the generated file by exact target filename and select/download or attach it to the current conversation or save it into the ChatGPT Library. Merely posting an old `sandbox:/mnt/data/...` URL from another task may not grant access; supply the actual file bytes.
3. Verify downloaded byte length and SHA-256 **before** any PyTorch deserialization.
4. Check metadata and load via weights-only, strict state dictionary matching; reproduce fixed-seed 15-output probe against U1 and record evidence (Git commit, environment, hash, probe values).
5. If the Work task no longer exposes the binary, label it **unavailable** rather than claiming recovery. Separately consider verified fresh retraining only on explicit direction.

## Safety and resource constraints
- No checkpoint copy to connected Linux root filesystem: it had about **303 MB free**, less than the 400.9 MB source file.
- No automatically purchased quantum hardware or live quantum sensor access.
- No modification to active Weaver, Neoh or model-serving systems.
