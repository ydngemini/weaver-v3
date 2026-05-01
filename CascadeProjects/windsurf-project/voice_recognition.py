#!/usr/bin/env python3
"""
voice_recognition.py — Speaker Identification for Phone Calls
=============================================================
Identifies callers by voice using OpenAI's audio embeddings API.
Saves voice embeddings to voice_registry.npz alongside face embeddings.

Usage:
    from voice_recognition import VoiceRecognizer

    recognizer = VoiceRecognizer("Nexus_Vault")

    # Register a voice
    await recognizer.register_voice("Nate", audio_samples=[audio1, audio2, audio3])

    # Identify a caller
    name, confidence = await recognizer.identify_voice(audio_sample)
    # Returns: ("Nate", 0.87) or ("unknown", 0.15)
"""

import asyncio
import base64
import os
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import numpy as np


class VoiceRecognizer:
    """Speaker identification using OpenAI audio embeddings."""

    def __init__(self, vault_dir: str = "Nexus_Vault", api_key: Optional[str] = None):
        self.vault_dir = Path(vault_dir)
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.vault_dir / "voice_registry.npz"

        # API key from env or parameter
        self.api_key = api_key or os.environ.get("WEAVER_VOICE_KEY", "")

        # Load existing registry
        self.voice_registry: dict[str, List[np.ndarray]] = {}
        if self.registry_path.exists():
            loaded = np.load(self.registry_path, allow_pickle=True)
            for name in loaded.files:
                embeddings = loaded[name]
                # Convert to list of arrays
                if embeddings.ndim == 2:
                    self.voice_registry[name] = [embeddings[i] for i in range(len(embeddings))]
                else:
                    self.voice_registry[name] = [embeddings]

    async def get_voice_embedding(self, audio_b64: str) -> Optional[np.ndarray]:
        """Get voice embedding from OpenAI (placeholder - using mock embeddings for now).

        Args:
            audio_b64: Base64-encoded audio (PCM16, 24kHz, mono).

        Returns:
            512-dimensional embedding vector, or None on failure.
        """
        # TODO: OpenAI doesn't have a public voice embedding API yet
        # For now, we'll use a hash-based mock embedding
        # In production, you'd use:
        #   - Azure Speaker Recognition API
        #   - Resemblyzer (open-source speaker embeddings)
        #   - pyannote.audio speaker embeddings

        import hashlib
        audio_hash = hashlib.sha256(audio_b64.encode()).digest()
        # Generate pseudo-embedding from hash (512-d)
        embedding = np.frombuffer(audio_hash + audio_hash * 15, dtype=np.uint8)[:512].astype(np.float32)
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        return embedding

    async def register_voice(self, name: str, audio_samples: List[str]) -> bool:
        """Register a voice with multiple samples.

        Args:
            name: Person's name.
            audio_samples: List of base64-encoded audio samples (PCM16, 24kHz).

        Returns:
            True if successful.
        """
        embeddings = []
        for audio_b64 in audio_samples:
            emb = await self.get_voice_embedding(audio_b64)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return False

        # Update registry
        self.voice_registry[name] = embeddings

        # Save to disk
        save_dict = {name: np.array(embs) for name, embs in self.voice_registry.items()}
        np.savez_compressed(self.registry_path, **save_dict)

        return True

    async def identify_voice(self, audio_b64: str, threshold: float = 0.70) -> Tuple[str, float]:
        """Identify a speaker from audio.

        Args:
            audio_b64: Base64-encoded audio (PCM16, 24kHz).
            threshold: Minimum cosine similarity for positive ID.

        Returns:
            (name, confidence) tuple. Returns ("unknown", score) if below threshold.
        """
        query_emb = await self.get_voice_embedding(audio_b64)
        if query_emb is None:
            return ("unknown", 0.0)

        best_name = "unknown"
        best_score = 0.0

        for name, stored_embs in self.voice_registry.items():
            for emb in stored_embs:
                # Cosine similarity (both vectors are normalized)
                score = float(np.dot(query_emb, emb))
                if score > best_score:
                    best_score = score
                    if score > threshold:
                        best_name = name

        return (best_name, best_score)

    def list_registered_voices(self) -> List[str]:
        """List all registered voice names."""
        return list(self.voice_registry.keys())


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════════════════════

async def _example():
    recognizer = VoiceRecognizer("Nexus_Vault")

    # Mock audio samples (in reality these would be from phone calls)
    sample1 = base64.b64encode(b"mock_audio_nate_sample_1").decode()
    sample2 = base64.b64encode(b"mock_audio_nate_sample_2").decode()
    sample3 = base64.b64encode(b"mock_audio_nate_sample_3").decode()

    # Register Nate's voice
    success = await recognizer.register_voice("Nate", [sample1, sample2, sample3])
    print(f"Registration: {success}")

    # Identify from new sample
    test_sample = base64.b64encode(b"mock_audio_nate_sample_1").decode()
    name, confidence = await recognizer.identify_voice(test_sample)
    print(f"Identified: {name} (confidence: {confidence:.2f})")

    print(f"Registered voices: {recognizer.list_registered_voices()}")


if __name__ == "__main__":
    asyncio.run(_example())
