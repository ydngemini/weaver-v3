#!/usr/bin/env python3
"""Privacy boundary for Weaver's public headless state."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


FORBIDDEN_PUBLIC_KEYS = {
    "last_thought",
    "last_dream",
    "transcript",
    "messages",
    "prompt",
    "system_prompt",
    "draft",
    "expert",
    "lobe",
    "models",
    "model_id",
    "region",
    "voice_id",
    "signature",
    "key_id",
    "token",
    "secret",
    "password",
    "credential",
}

PUBLIC_DEGRADED_REASONS = {
    "headless-degraded",
    "fabric-pressure",
    "fabric-ledger-invalid",
    "cognition-guarded",
    "voice-degraded",
    "awareness-degraded",
}

SAFE_COGNITION_TOPICS = {
    "attention",
    "body",
    "environment",
    "latency",
    "memory",
    "privacy",
    "safety",
    "voice",
}


@dataclass(frozen=True)
class _PrivateCognitionEntry:
    kind: str
    content: str
    created_at: float
    digest: str
    topics: tuple[str, ...]


class PrivateCognitionVault:
    """Bounded raw cognition retained only for internal cortex context."""

    def __init__(
        self,
        *,
        max_entries: int = 16,
        ttl_seconds: float = 86_400,
        max_content_chars: int = 2_400,
        clock=time.time,
    ) -> None:
        self.max_entries = min(max(int(max_entries), 2), 64)
        self.ttl_seconds = min(max(float(ttl_seconds), 60), 86_400)
        self.max_content_chars = min(max(int(max_content_chars), 120), 4_000)
        self._clock = clock
        self._lock = asyncio.Lock()
        self._entries: deque[_PrivateCognitionEntry] = deque()

    @staticmethod
    def _topics(content: str) -> tuple[str, ...]:
        lowered = content.lower()
        aliases = {
            "attention": ("attention", "focus", "notice"),
            "body": ("body", "pose", "elbow", "knee", "motion"),
            "environment": ("environment", "room", "object", "penthouse"),
            "latency": ("latency", "fast", "delay", "response"),
            "memory": ("memory", "remember", "recall"),
            "privacy": ("privacy", "private", "secret"),
            "safety": ("safe", "safety", "constraint", "boundary"),
            "voice": ("voice", "speech", "speak", "audio"),
        }
        return tuple(
            topic
            for topic, words in aliases.items()
            if topic in SAFE_COGNITION_TOPICS and any(word in lowered for word in words)
        )[:4]

    def _prune(self, now: float) -> None:
        while self._entries and now - self._entries[0].created_at >= self.ttl_seconds:
            self._entries.popleft()
        while len(self._entries) > self.max_entries:
            self._entries.popleft()

    async def store(self, kind: str, content: Any) -> dict[str, Any]:
        normalized_kind = str(kind or "").strip().lower()
        if normalized_kind not in {"thought", "dream"}:
            raise ValueError("private cognition kind is invalid")
        text = " ".join(str(content or "").split())[: self.max_content_chars]
        if not text:
            raise ValueError("private cognition content is empty")
        now = float(self._clock())
        entry = _PrivateCognitionEntry(
            kind=normalized_kind,
            content=text,
            created_at=now,
            digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            topics=self._topics(text),
        )
        async with self._lock:
            self._prune(now)
            self._entries.append(entry)
            self._prune(now)
        return self._public_entry(entry)

    @staticmethod
    def _public_entry(entry: _PrivateCognitionEntry) -> dict[str, Any]:
        return {
            "available": True,
            "updated_at": datetime.fromtimestamp(entry.created_at, tz=timezone.utc),
            "topics": list(entry.topics),
            "content_hidden": True,
            "digest_prefix": entry.digest[:12],
        }

    async def latest(self, kind: str) -> str:
        normalized_kind = str(kind or "").strip().lower()
        now = float(self._clock())
        async with self._lock:
            self._prune(now)
            for entry in reversed(self._entries):
                if entry.kind == normalized_kind:
                    return entry.content
        return ""

    async def public_metadata(self) -> dict[str, dict[str, Any]]:
        now = float(self._clock())
        async with self._lock:
            self._prune(now)
            result: dict[str, dict[str, Any]] = {}
            for kind in ("thought", "dream"):
                entry = next(
                    (item for item in reversed(self._entries) if item.kind == kind),
                    None,
                )
                result[kind] = (
                    self._public_entry(entry)
                    if entry is not None
                    else {
                        "available": False,
                        "updated_at": None,
                        "topics": [],
                        "content_hidden": True,
                        "digest_prefix": "",
                    }
                )
            result["retention"] = {
                "entries": len(self._entries),
                "max_entries": self.max_entries,
                "ttl_seconds": int(self.ttl_seconds),
                "content_hidden": True,
            }
            return result


def utc_from_unix(value: Any) -> datetime | None:
    """Convert a finite, positive Unix timestamp without passing through text."""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def private_cognition_metadata(legacy: dict[str, Any]) -> dict[str, Any]:
    """Return existence/count metadata, never private thought or dream content."""

    metadata = (
        legacy.get("private_cognition")
        if isinstance(legacy.get("private_cognition"), dict)
        else {}
    )
    thought = metadata.get("thought") if isinstance(metadata.get("thought"), dict) else {}
    dream = metadata.get("dream") if isinstance(metadata.get("dream"), dict) else {}

    def _safe_topics(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            topic for topic in value[:4]
            if isinstance(topic, str) and topic in SAFE_COGNITION_TOPICS
        ]

    return {
        "thought_count": max(0, int(legacy.get("thoughts", 0) or 0)),
        "dream_count": max(0, int(legacy.get("dreams", 0) or 0)),
        "last_thought_at": utc_from_unix(legacy.get("last_thought_at")),
        "last_dream_at": utc_from_unix(legacy.get("last_dream_at")),
        "private_thought_available": bool(legacy.get("last_thought_at")),
        "private_dream_available": bool(legacy.get("last_dream_at")),
        "thought_topics": _safe_topics(thought.get("topics")),
        "dream_topics": _safe_topics(dream.get("topics")),
        "private_content_hidden": True,
    }


def degraded_reasons(
    legacy: dict[str, Any],
    fabric: dict[str, Any],
    cognition: dict[str, Any],
) -> list[str]:
    """Map internal failures to a small stable public vocabulary."""

    reasons: list[str] = []
    voice = legacy.get("voice_realtime") if isinstance(legacy.get("voice_realtime"), dict) else {}
    if legacy.get("last_error"):
        reasons.append("headless-degraded")
    if fabric.get("status") != "nominal":
        reasons.append("fabric-pressure")
    ledger = fabric.get("ledger") if isinstance(fabric.get("ledger"), dict) else {}
    if not bool(ledger.get("valid", False)):
        reasons.append("fabric-ledger-invalid")
    if cognition.get("status") != "nominal":
        reasons.append("cognition-guarded")
    if voice.get("last_error"):
        reasons.append("voice-degraded")
    return [reason for reason in reasons if reason in PUBLIC_DEGRADED_REASONS]


def ensure_public_state_safe(value: Any) -> None:
    """Fail closed if a private-content key enters a public state projection."""

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError("private field reached the public headless state")
            ensure_public_state_safe(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            ensure_public_state_safe(item)
