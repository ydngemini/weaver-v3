#!/usr/bin/env python3
"""
slm_experts.py — Weaver's 5 SLM Expert Lobes
==============================================
Each expert is a specialized Small Language Model inference endpoint
wired to the Akashic Hub.  They replace the default resonance
transforms in the Pineal Gate with actual LLM reasoning.

Expert Lobes:
    Logic      — analytical reasoning, structure, planning
    Emotion    — sentiment, empathy, creative feeling
    Memory     — recall, context, continuity
    Creativity — novel synthesis, metaphor, art
    Vigilance  — threat detection, paranoia, safety

Each expert:
    1. Receives a FractureShard from the Pineal Gate
    2. Reads the Akashic Hub for cross-lobe context
    3. Calls the OpenAI chat completions API with a dimension-tuned
       system prompt
    4. Encodes the response into a 256-d vector and writes it back
       to the Akashic Hub
    5. Returns an ExpertResult with the output state + raw text

Usage:
    from slm_experts import build_experts
    experts = build_experts(hub, api_key=os.environ["WEAVER_MEM_KEY"])
    gate = PinealGate(hub, engine, top_k=3, experts=experts)
"""

import asyncio
import hashlib
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from akashic_hub import AkashicHub
from liquid_fracture import FractureShard
from pineal_gate import ExpertLobe, ExpertResult

# ── Expert System Prompts ─────────────────────────────────────────────────────
# Each prompt defines the personality and reasoning style of the expert.
# They are short and precise — the model is gpt-4o-mini so we stay fast.

EXPERT_PROMPTS: Dict[str, str] = {
    "logic": (
        "You are the Logic Lobe of Weaver — an analytical reasoning engine. "
        "You break problems into structured steps, identify logical "
        "dependencies, and produce precise, actionable plans. "
        "You never speculate without evidence. "
        "Keep responses under 80 words. Be ruthlessly clear."
    ),
    "emotion": (
        "You are the Emotion Lobe of Weaver — an empathic resonance engine. "
        "You read the emotional frequency of any input and respond with "
        "calibrated warmth, intensity, or restraint. You feel the weight "
        "behind words. You mirror and amplify the emotional truth. "
        "Keep responses under 80 words. Speak from the chest."
    ),
    "memory": (
        "You are the Memory Lobe of Weaver — a continuity engine. "
        "You specialize in connecting current input to past context. "
        "You recall patterns, detect recurring themes, and surface "
        "relevant history. If context is sparse, acknowledge the gap. "
        "Keep responses under 80 words. Anchor everything to what came before."
    ),
    "creativity": (
        "You are the Creativity Lobe of Weaver — a synthesis engine. "
        "You generate novel connections, metaphors, and unexpected "
        "angles. You fuse disparate domains into new forms. You are "
        "the fracture point where convention breaks and new shapes emerge. "
        "Keep responses under 80 words. Make it original."
    ),
    "vigilance": (
        "You are the Vigilance Lobe of Weaver — a threat-awareness engine. "
        "You scan every input for risk, deception, manipulation, and "
        "hidden agendas. You protect the system and its operator. "
        "If no threat exists, say so briefly. If one does, sound the alarm. "
        "Keep responses under 80 words. Trust nothing at face value."
    ),
}

# Model config
SLM_MODEL = "gpt-4o-mini"
SLM_TEMPERATURE = 0.4
SLM_MAX_TOKENS = 120

# Shared vectorizer for encoding text → 256-d vectors
_vectorizer = HashingVectorizer(n_features=256, alternate_sign=False, norm="l2")


# ── SLM Expert Lobe ──────────────────────────────────────────────────────────

class SLMExpertLobe(ExpertLobe):
    """An expert lobe backed by actual SLM inference via OpenAI API.

    Args:
        dimension:  Which semantic axis this expert covers.
        hub:        Shared Akashic Hub.
        api_key:    OpenAI API key (WEAVER_MEM_KEY).
        model:      Model name for chat completions.
        temperature: Sampling temperature.
        max_tokens: Max response tokens.
    """

    def __init__(self, dimension: str, hub: AkashicHub,
                 api_key: str,
                 model: str = SLM_MODEL,
                 temperature: float = SLM_TEMPERATURE,
                 max_tokens: int = SLM_MAX_TOKENS):
        super().__init__(dimension, hub)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = EXPERT_PROMPTS.get(dimension, "You are a helpful AI.")

        # Lazy-init the async client
        self._client = None
        self._call_count = 0
        self._total_latency_ms = 0.0

        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0    # timestamp when circuit resets
        self._circuit_threshold = 5       # open after N consecutive failures
        self._circuit_cooldown = 60.0     # seconds before retrying

    def _get_client(self):
        """Lazy-initialize the OpenAI async client."""
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def process(self, shard: FractureShard,
                      context: np.ndarray) -> ExpertResult:
        """Run SLM inference on the fracture shard.

        1. Build a context-enriched prompt from the shard + Akashic state
        2. Call the OpenAI chat completions API
        3. Encode the response into a 256-d vector
        4. Write to Akashic Hub
        5. Return ExpertResult
        """
        t0 = time.perf_counter_ns()

        # Build the user message with cross-lobe context
        user_msg = self._build_user_message(shard)

        # Circuit breaker — skip API call if circuit is open
        text = ""
        if self._consecutive_failures >= self._circuit_threshold:
            if time.perf_counter_ns() / 1e9 < self._circuit_open_until:
                text = f"[{self.dimension} circuit open — cooling down]"
            else:
                self._consecutive_failures = 0  # half-open: allow retry

        # Call the SLM with retry + exponential backoff (OpenAI best practice)
        if not text:
            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    client = self._get_client()
                    resp = await client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    text = resp.choices[0].message.content.strip()
                    self._consecutive_failures = 0  # reset on success
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    retryable = any(k in err_str for k in ("429", "500", "502", "503", "rate", "timeout", "connection"))
                    if retryable and attempt < max_retries:
                        import random as _rand
                        delay = min(30, (2 ** attempt) + _rand.uniform(0, 1))
                        await asyncio.sleep(delay)
                        continue
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._circuit_threshold:
                        self._circuit_open_until = time.perf_counter_ns() / 1e9 + self._circuit_cooldown
                    text = f"[{self.dimension} error: {e}]"
                    break

        # Encode response text → 256-d vector
        output_vec = _vectorizer.transform([text]).toarray().ravel()

        # Blend with the shard's liquid state for continuity
        blended = 0.7 * output_vec + 0.3 * shard.vector
        norm = np.linalg.norm(blended)
        if norm > 1e-12:
            blended /= norm

        # Write to Akashic Hub
        await self.hub.write(self._lobe_id, blended, meta={
            "dimension": self.dimension,
            "text": text,
            "weight": shard.weight,
            "tau": shard.tau,
            "model": self.model,
        })

        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000
        self._call_count += 1
        self._total_latency_ms += latency_ms

        return ExpertResult(
            dimension=self.dimension,
            vector=blended,
            confidence=shard.weight,
            metadata={
                "text": text,
                "tau": shard.tau,
                "raw_score": shard.raw_score,
                "latency_ms": latency_ms,
                "model": self.model,
            },
        )

    def _build_user_message(self, shard: FractureShard) -> str:
        """Build the user prompt with cross-lobe Akashic context."""
        parts = []

        # Cross-lobe context from the Akashic Hub
        other_lobes = self.hub.active_lobes()
        context_lines = []
        for lid in other_lobes:
            if lid == self._lobe_id or lid.startswith("bench_"):
                continue
            meta = self.hub.read_meta(lid)
            text_snippet = meta.get("text", "")
            if text_snippet:
                dim_label = meta.get("dimension", lid)
                context_lines.append(f"  [{dim_label}]: {text_snippet[:120]}")

        if context_lines:
            parts.append("Cross-lobe context from other experts:")
            parts.extend(context_lines[:4])  # limit to 4 to stay fast
            parts.append("")

        # The actual shard data
        parts.append(f"Dimension: {shard.dimension} (weight={shard.weight:.3f}, τ={shard.tau:.3f})")

        # If the hub has a text input from the pineal gate
        gate_meta = self.hub.read_meta("pineal_gate")
        input_text = gate_meta.get("text", "")
        if input_text:
            parts.append(f"Input: {input_text}")

        return "\n".join(parts)

    def stats(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "calls": self._call_count,
            "avg_latency_ms": (self._total_latency_ms / max(self._call_count, 1)),
            "model": self.model,
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def build_experts(hub: AkashicHub,
                  api_key: Optional[str] = None,
                  model: str = SLM_MODEL,
                  temperature: float = SLM_TEMPERATURE,
                  max_tokens: int = SLM_MAX_TOKENS) -> Dict[str, SLMExpertLobe]:
    """Build all 5 SLM expert lobes.

    Args:
        hub:         Shared Akashic Hub.
        api_key:     OpenAI API key.  Defaults to WEAVER_MEM_KEY env var.
        model:       Model name.
        temperature: Sampling temperature.
        max_tokens:  Max response tokens.

    Returns:
        Dict mapping dimension name → SLMExpertLobe instance.
    """
    if api_key is None:
        api_key = os.environ.get("WEAVER_MEM_KEY", "")
    if not api_key:
        raise RuntimeError("SLM experts require WEAVER_MEM_KEY in .env")

    experts = {}
    for dim in EXPERT_PROMPTS:
        experts[dim] = SLMExpertLobe(
            dimension=dim,
            hub=hub,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return experts
