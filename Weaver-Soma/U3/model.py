"""SOMA U3 research scaffold: one jointly trainable multimodal PyTorch model.

All new neural weights are untrained. U1/U2 legacy pathways remain unchanged
on base-only calls. No real QPU, olfactory hardware, or actuator access.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import io
import sys
from pathlib import Path
from collections.abc import Mapping
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE.parent / "U2") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "U2"))
from weaver_soma_u2 import WeaverSOMA100M, count_parameters  # noqa: E402
from unified_model import HORIZON, STATE_CENTER, STATE_SCALE  # noqa: E402
from quantum_native import VariationalQuantumFeatures  # noqa: E402
from evolution import EvolutionProposalHead  # noqa: E402
from audio_spectral import SpectralAudioStem  # noqa: E402
from audio_self_monitor import AUDIO_CLASSES, AudioSelfMonitor, spectral_input  # noqa: E402
from audio_scope import ARCHITECTURE as AUDIO_SCOPE_ARCHITECTURE  # noqa: E402
from audio_scope import AudioScopeMonitor, temporal_scope_features  # noqa: E402

U1_SHA256 = "3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f"
U2_STAGE4_SHA256 = "14d170d70f7d16d2abc23721f4f4cc3a9b3d115651dffbb2e343ebfb1f135149"
LEGACY_OUTPUTS = frozenset({
    "state", "risk_logits", "future", "policy_logits", "external_logits", "episodic_summary",
    "body_readout", "workspace_logits", "compute_logits", "regulation", "reliability",
    "memory_key", "workspace_weights", "body_tokens", "lex_regularizer",
})


class TemporalStem(nn.Module):
    """Trainable temporal encoder; invalid channels have zero input gradient."""

    def __init__(self, channels: int, dim: int):
        super().__init__()
        self.channels = channels
        self.project = nn.Sequential(nn.Linear(channels, dim), nn.SiLU())
        self.local = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, values: torch.Tensor, quality: torch.Tensor | None):
        if values.ndim != 3 or values.shape[-1] != self.channels:
            raise ValueError(f"expected temporal samples [B,T,{self.channels}]")
        if quality is not None:
            if quality.shape != values.shape:
                raise ValueError("channel quality must match temporal sample shape")
            values = torch.where(quality > 0, values * quality.clamp(0.0, 1.0),
                                 torch.zeros_like(values))
        x = self.project(values)
        return self.norm(self.local(x.transpose(1, 2)).transpose(1, 2))


class WeaverSOMAU3(WeaverSOMA100M):
    """One model with legacy body, native sensory stems, and trainable quantum simulation.

    modalities is a mapping name -> {values, valid?, quality?, timestamps?}.
    text: [B,T] integer bytes 0..255 (not pretrained language).
    vision: [B,F,3,H,W] float frames; valid [B,F].
    audio: [B,S] mono waveform; valid [B,S]. The logmel encoder expects 16 kHz.
    touch: [B,T,6], olfaction: [B,T,8], rf: [B,T,8], spatial: [B,T,6].
    These dimensions are research contracts, not universal sensor standards.
    """

    CHANNELS = {"touch": 6, "olfaction": 8, "rf": 8, "spatial": 6}
    LABELS = {"text": 256, "vision": 12, "audio": 16, "touch": 8,
              "olfaction": 8, "rf": 8, "spatial": 8}

    def __init__(self, *, fusion_dim=128, max_modality_tokens=128, quantum_qubits=4,
                 quantum_layers=2, audio_encoder="waveform", **u2_kwargs):
        super().__init__(**u2_kwargs)
        if fusion_dim % 4 or fusion_dim < 32 or max_modality_tokens < 1:
            raise ValueError("fusion_dim must be >=32, divisible by 4; token limit >=1")
        if audio_encoder not in ("waveform", "logmel"):
            raise ValueError("audio_encoder must be waveform or logmel")
        self._u2_keys = frozenset(self.state_dict())
        self.fusion_dim = fusion_dim
        self.max_modality_tokens = max_modality_tokens
        self.text_embedding = nn.Embedding(256, fusion_dim)
        self.vision_patch = nn.Conv2d(3, fusion_dim // 4, kernel_size=4, stride=4)
        self.vision_project = nn.Linear(fusion_dim // 4, fusion_dim)
        self.audio_conv = nn.Sequential(
            nn.Conv1d(1, fusion_dim // 4, kernel_size=7, stride=4, padding=3), nn.SiLU(),
            nn.Conv1d(fusion_dim // 4, fusion_dim, kernel_size=5, stride=4, padding=2))
        self.temporal_stems = nn.ModuleDict({name: TemporalStem(ch, fusion_dim)
                                             for name, ch in self.CHANNELS.items()})
        self.type_embed = nn.ParameterDict({name: nn.Parameter(torch.randn(fusion_dim) * .01)
                                             for name in self.LABELS})
        self.position_embed = nn.Embedding(max_modality_tokens, fusion_dim)
        self.time_embed = nn.Linear(2, fusion_dim, bias=False)
        self.token_norm = nn.LayerNorm(fusion_dim)
        self.state_to_query = nn.Linear(self.latent_dim, fusion_dim)
        self.cross_attention = nn.MultiheadAttention(fusion_dim, 4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_to_state = nn.Linear(fusion_dim, self.latent_dim)
        self.multimodal_gain = nn.Parameter(torch.zeros(()))
        self.quantum = VariationalQuantumFeatures(fusion_dim, quantum_qubits, quantum_layers)
        self.quantum_projection = nn.Linear(quantum_qubits, fusion_dim)
        self.quantum_gain = nn.Parameter(torch.zeros(()))
        self.label_heads = nn.ModuleDict({name: nn.Linear(fusion_dim, n)
                                          for name, n in self.LABELS.items()})
        # Trainable proposal head is part of THIS neural model; mutation execution is not.
        self.evolution_head = EvolutionProposalHead(signal_count=8, width=64)
        self.spectral_audio = (SpectralAudioStem(fusion_dim, self.LABELS["audio"])
                               if audio_encoder == "logmel" else None)
        self.audio_self_monitor: AudioSelfMonitor | None = None
        self.audio_scope_monitor: AudioScopeMonitor | None = None

    @staticmethod
    def _file_hash(path: str | Path) -> str:
        h = hashlib.sha256()
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_base_weights(self, weights: dict, wanted: frozenset[str]):
        own = self.state_dict()
        if set(weights) != set(wanted):
            raise ValueError("checkpoint base tensor key set does not match")
        for key, value in weights.items():
            if not isinstance(value, torch.Tensor) or value.shape != own[key].shape:
                raise ValueError(f"incompatible base checkpoint tensor {key}")
        result = self.load_state_dict(weights, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != set(own) - set(wanted):
            raise ValueError("mismatch loading the protected base")
        return len(weights)

    def load_verified_u1(self, path: str | Path):
        """Restore known original 91 U1 tensors; U2 deep core stays random."""
        if self._file_hash(path) != U1_SHA256:
            raise ValueError("U1 SHA-256 mismatch")
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(obj, dict) or obj.get("architecture") != "WeaverSOMAUnified-v1":
            raise ValueError("not a trusted SOMA U1 checkpoint")
        expected = frozenset(k for k in self._u2_keys if not k.startswith("deep_cognitive_core."))
        return self._load_base_weights(obj["weights"], expected)

    def load_verified_u2_stage4(self, path: str | Path):
        """Strict full U2 restore; requires historical 401MB file bytes."""
        if self._file_hash(path) != U2_STAGE4_SHA256:
            raise ValueError("U2 Stage 4 SHA-256 mismatch")
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
        if (not isinstance(obj, dict) or obj.get("architecture") != "WeaverSOMA100M-stage4"
                or obj.get("format_version") != 1 or obj.get("total_parameters") != 100206197
                or not obj.get("no_motor_authority")):
            raise ValueError("not the verified unified Stage 4 research checkpoint")
        return self._load_base_weights(obj["weights"], self._u2_keys)

    def load_audio_self_monitor(self, path: str | Path):
        """Install a paired five-class audio stem and read-only correctness monitor."""
        if self.spectral_audio is None:
            raise ValueError("the audio self-monitor requires audio_encoder='logmel'")
        # Hash and deserialize the same bytes, even if the filename is replaced during load.
        checkpoint = Path(path).read_bytes()
        obj = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True)
        if (not isinstance(obj, dict) or obj.get("architecture") != "SOMA-U3-audio-self-monitor-v1"
                or obj.get("classes") != AUDIO_CLASSES or obj.get("sample_rate") != 16000
                or obj.get("fusion_dim") != self.fusion_dim):
            raise ValueError("incompatible SOMA audio self-monitor checkpoint")
        stem = deepcopy(self.spectral_audio)
        monitor = AudioSelfMonitor().to(self.spectral_audio.local[0].weight)
        stem.load_state_dict(obj["audio_weights"], strict=True)
        monitor.load_state_dict(obj["monitor_weights"], strict=True)
        for module in (stem, monitor):
            if not all(torch.isfinite(tensor).all() for tensor in module.state_dict().values()):
                raise ValueError("audio self-monitor checkpoint contains nonfinite values")
        if (monitor.scale <= 0).any() or not 0 < monitor.threshold < 1:
            raise ValueError("invalid audio self-monitor calibration")
        self.spectral_audio.load_state_dict(stem.state_dict(), strict=True)
        monitor.loaded_checkpoint_sha256 = hashlib.sha256(checkpoint).hexdigest()
        self.audio_self_monitor = monitor.train(self.training)
        self.audio_scope_monitor = None
        return monitor

    def load_audio_scope_monitor(self, path: str | Path):
        """Attach a scope monitor calibrated against the currently loaded audio checkpoint."""
        if self.audio_self_monitor is None:
            raise ValueError("load the paired audio self-monitor first")
        data = Path(path).read_bytes()
        obj = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if (not isinstance(obj, dict) or obj.get("architecture") != AUDIO_SCOPE_ARCHITECTURE
                or obj.get("monitor_sha256") != self.audio_self_monitor.loaded_checkpoint_sha256
                or self.audio_self_monitor.loaded_checkpoint_sha256 is None):
            raise ValueError("scope and audio self-monitor checkpoints do not match")
        scope = AudioScopeMonitor(obj["method"], obj.get("reference_files", ())).to(
            self.audio_self_monitor.center,
        )
        scope.load_state_dict(obj["weights"], strict=True)
        if (not all(torch.isfinite(tensor).all() for tensor in scope.state_dict().values())
                or (scope.scale <= 0).any()):
            raise ValueError("invalid audio scope checkpoint")
        scope.loaded_checkpoint_sha256 = hashlib.sha256(data).hexdigest()
        self.audio_scope_monitor = scope.train(self.training)
        return scope

    def _encode(self, name: str, sample: Mapping[str, torch.Tensor]):
        if "values" not in sample:
            raise ValueError(f"{name} is missing values")
        values = sample["values"]
        if name == "text":
            if values.ndim != 2 or values.dtype not in (torch.long, torch.int64, torch.int32):
                raise ValueError("text values must be integer [B,T] bytes")
            if (values < 0).any() or (values > 255).any():
                raise ValueError("text bytes must be in [0,255]")
            tokens = self.text_embedding(values.long())
            default_mask = torch.ones_like(values, dtype=torch.bool)
        elif name == "vision":
            if values.ndim != 5 or values.shape[2] != 3 or min(values.shape[-2:]) < 4:
                raise ValueError("vision frames must be [B,F,3,H>=4,W>=4]")
            b, frames = values.shape[:2]
            frame_valid = sample.get("valid")
            if frame_valid is not None and frame_valid.shape != (b, frames):
                raise ValueError("vision validity must have shape [B,F]")
            safe_values = values if frame_valid is None else torch.where(
                frame_valid[:, :, None, None, None].bool(), values, torch.zeros_like(values))
            patches = self.vision_patch(safe_values.reshape(b * frames, 3, *values.shape[-2:]))
            patches = patches.flatten(2).transpose(1, 2)
            per_frame = patches.shape[1]
            tokens = self.vision_project(patches).reshape(b, frames * per_frame, -1)
            default_mask = torch.ones(b, frames, device=values.device, dtype=torch.bool)
        elif name == "audio":
            if values.ndim != 2 or values.shape[1] < 1:
                raise ValueError("mono audio samples must be [B,S]")
            valid_samples = sample.get("valid", torch.ones_like(values, dtype=torch.bool))
            if valid_samples.shape != values.shape:
                raise ValueError("audio sample mask has wrong shape")
            quality = sample.get("quality")
            if quality is not None and quality.shape != values.shape:
                raise ValueError("audio quality shape mismatch")
            if self.spectral_audio is None:
                audio = torch.where(valid_samples.bool(), values, torch.zeros_like(values))
                if quality is not None:
                    audio = torch.where(quality > 0, audio * quality.clamp(0, 1), torch.zeros_like(audio))
                tokens = self.audio_conv(audio.unsqueeze(1)).transpose(1, 2)
                default_mask = F.adaptive_max_pool1d(valid_samples.float().unsqueeze(1),
                                                      tokens.shape[1]).squeeze(1) > 0
            else:
                audio, effective_valid = spectral_input(values, valid_samples, quality)
                tokens, default_mask = self.spectral_audio(audio, effective_valid)
        else:
            if name not in self.CHANNELS:
                raise ValueError(f"unknown SOMA modality: {name}")
            mask = sample.get("valid")
            if mask is not None and mask.shape != values.shape[:2]:
                raise ValueError(f"{name} validity must have shape [B,T]")
            masked = values if mask is None else torch.where(
                mask.unsqueeze(-1).bool(), values, torch.zeros_like(values))
            tokens = self.temporal_stems[name](masked, sample.get("quality"))
            qual = sample.get("quality")
            default_mask = (qual > 0).any(dim=-1) if qual is not None else torch.ones(
                values.shape[:2], device=values.device, dtype=torch.bool)
        valid = sample.get("valid")
        if name == "audio":
            valid = default_mask
        elif name == "vision":
            if valid is not None and valid.shape != default_mask.shape:
                raise ValueError("vision validity must have shape [B,F]")
            valid = default_mask if valid is None else valid.bool()
            valid = valid.repeat_interleave(per_frame, dim=1)
        else:
            if valid is not None and valid.shape != default_mask.shape:
                raise ValueError(f"{name} validity shape mismatch")
            valid = default_mask if valid is None else valid.bool() & default_mask
        if name == "audio" and "valid" in sample and not sample["valid"].any():
            valid = torch.zeros_like(default_mask)
        timestamps = sample.get("timestamps")
        if timestamps is not None:
            if timestamps.shape != (values.shape[0], values.shape[1]):
                raise ValueError(f"{name} timestamps must be [B,T_input]")
            if name == "vision":
                timestamps = timestamps.repeat_interleave(per_frame, dim=1)
            elif name == "audio":
                timestamps = F.adaptive_avg_pool1d(timestamps.float().unsqueeze(1),
                                                   tokens.shape[1]).squeeze(1)
        if tokens.shape[1] > self.max_modality_tokens:
            select = torch.linspace(0, tokens.shape[1] - 1, self.max_modality_tokens,
                                    device=tokens.device).long()
            tokens, valid = tokens[:, select], valid[:, select]
            if timestamps is not None:
                timestamps = timestamps[:, select]
        length = tokens.shape[1]
        if length == 0:
            raise ValueError("empty sensory input")
        spectral_audio = name == "audio" and self.spectral_audio is not None
        if not spectral_audio:
            position = self.position_embed(torch.arange(length, device=tokens.device))
            tokens = tokens + position.unsqueeze(0) + self.type_embed[name]
        if timestamps is not None:
            rel = (timestamps - timestamps[:, :1]).to(tokens.dtype)
            phase = torch.stack((rel.sin(), rel.cos() - 1.0), dim=-1)
            tokens = tokens + self.time_embed(phase)
        if not spectral_audio:
            tokens = self.token_norm(tokens)
        return tokens * valid.unsqueeze(-1).to(tokens.dtype), valid

    def _refresh_legacy_heads(self, out: dict, common: torch.Tensor, obs: torch.Tensor):
        """Recompute inherited heads from fused state, same operators as U1."""
        batch = obs.shape[0]
        actions = torch.eye(3, device=obs.device, dtype=obs.dtype).unsqueeze(0).expand(batch, -1, -1)
        states = common.unsqueeze(1).expand(-1, 3, -1)
        future = self.counterfactual(torch.cat((states, actions), dim=-1)).reshape(batch, 3, HORIZON, 3)
        boundary = ((future * obs.new_tensor(STATE_SCALE) + obs.new_tensor(STATE_CENTER)
                    - obs.new_tensor([62., 43., .82])) / obs.new_tensor(STATE_SCALE)).amax(dim=2)
        out.update({
            "state": common,
            "future": future,
            "risk_logits": self.risk(torch.cat((states, actions, boundary), dim=-1)).squeeze(-1),
            "policy_logits": self.action_policy(common),
            "external_logits": self.event(common).squeeze(-1),
            "episodic_summary": self.episodic_summary(common),
            "body_readout": self.body_readout(common),
            "workspace_logits": self.workspace_task(common),
            "compute_logits": self.compute_advice(common),
            "regulation": self.regulation(common),
            "reliability": self.confidence(common).squeeze(-1),
            "memory_key": self.semantic_memory_key(common),
        })

    def forward(self, obs, quality, context, executed_action=None, ablate=None, *,
                modalities: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
                evolution_diagnostics: torch.Tensor | None = None):
        out = super().forward(obs, quality, context, executed_action, ablate)
        if evolution_diagnostics is not None:
            if evolution_diagnostics.shape[0] != obs.shape[0]:
                raise ValueError("evolution diagnostics batch size mismatch")
            out["evolution_proposals"] = self.evolution_head(evolution_diagnostics)
        if modalities is None or len(modalities) == 0:
            return out  # EXACT 15-key legacy contract when diagnostics are absent.
        tokens, masks, auxiliary = [], [], {}
        for name, sample in modalities.items():
            if name not in self.LABELS:
                raise ValueError(f"unknown SOMA modality: {name}")
            encoded, valid = self._encode(name, sample)
            if encoded.shape[0] != obs.shape[0]:
                raise ValueError("modality batch size must match body input")
            tokens.append(encoded)
            masks.append(valid)
            if name == "audio" and self.spectral_audio is not None:
                audio_logits = self.spectral_audio.classify(encoded, valid)
                if valid.any():
                    auxiliary[name] = audio_logits
                if self.audio_self_monitor is not None:
                    values = sample["values"]
                    audio, observed = spectral_input(
                        values, sample.get("valid", torch.ones_like(values, dtype=torch.bool)),
                        sample.get("quality"),
                    )
                    correctness = self.audio_self_monitor(
                        audio_logits[:, :len(AUDIO_CLASSES)], audio, observed, encoded, valid,
                    )
                    out["audio_self_monitor"] = correctness
                    if self.audio_scope_monitor is not None:
                        scope = self.audio_scope_monitor
                        temporal = temporal_scope_features(encoded, valid) if scope.uses_temporal_features else None
                        out["audio_scope_monitor"] = scope(correctness, temporal)
            elif valid.any():
                pooled = encoded.sum(dim=1) / valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
                auxiliary[name] = self.label_heads[name](pooled)
                if name == "text":
                    auxiliary["text_token_logits"] = self.label_heads["text"](encoded)
        merged = torch.cat(tokens, dim=1)
        validity = torch.cat(masks, dim=1)
        safe = validity.clone()
        empty = ~safe.any(dim=1)
        if empty.any():
            safe[empty, 0] = True
        query = self.state_to_query(out["state"]).unsqueeze(1)
        attended, _ = self.cross_attention(query, merged, merged,
                                           key_padding_mask=~safe, need_weights=False)
        attended = attended.squeeze(1)
        attended = torch.where(empty.unsqueeze(-1), torch.zeros_like(attended), attended)
        pooled = merged.sum(dim=1) / validity.sum(dim=1).clamp(min=1).unsqueeze(-1)
        qfeatures = self.quantum(pooled)
        attended = attended + self.quantum_gain * self.quantum_projection(qfeatures)
        fused = out["state"] + self.multimodal_gain * self.fusion_to_state(self.fusion_norm(attended))
        fused = torch.where(empty.unsqueeze(-1), out["state"], fused)
        self._refresh_legacy_heads(out, fused, obs)
        out["modality_logits"] = auxiliary
        out["quantum_expectations"] = qfeatures
        out["sensory_token_mask"] = validity
        return out
