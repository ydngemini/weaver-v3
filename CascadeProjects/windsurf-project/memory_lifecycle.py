#!/usr/bin/env python3
"""Metadata-only lifecycle index for Weaver's persistent memories."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RETENTION_DAYS = {
    "conversation": 365,
    "person": 3650,
    "thought": 30,
    "dream": 30,
    "browser_memory": 90,
    "vision": 30,
    "quantum": 7,
    "memory": 90,
}


def _utc_iso(timestamp: float | None = None) -> str:
    value = time.time() if timestamp is None else float(timestamp)
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec="seconds")


def _safe_identifier(value: Any, limit: int = 80) -> str:
    text = "".join(
        character if character.isalnum() or character in "_.:-" else "-"
        for character in str(value or "").strip().lower()
    )
    return text[:limit] or "unknown"


def _safe_reason(value: Any, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(r"(?i)(secret|token|password|credential)\s*[:=]\s*\S+", r"\1=[REDACTED]", text)
    return text[:limit] or "operator-request"


class MemoryLifecycle:
    """Track provenance, deduplication, freshness, retention, and deletion."""

    def __init__(
        self,
        index_path: Path,
        deletion_log_path: Path,
        *,
        max_records: int = 2_048,
        dedupe_window_seconds: float = 300,
        clock=time.time,
    ) -> None:
        self.index_path = Path(index_path)
        self.deletion_log_path = Path(deletion_log_path)
        self.max_records = min(max(int(max_records), 64), 16_384)
        self.dedupe_window_seconds = min(
            max(float(dedupe_window_seconds), 1.0),
            86_400.0,
        )
        self._clock = clock
        self._lock = threading.RLock()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.deletion_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.deletion_log_path.touch(exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("records"), dict):
                raise ValueError("invalid lifecycle index")
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {"version": 1, "records": {}}
        payload["version"] = 1
        payload.setdefault("records", {})
        return payload

    def _save(self) -> None:
        self._data["updated_at"] = _utc_iso(self._clock())
        temporary = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._data, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, self.index_path)

    @staticmethod
    def _digest(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _fingerprint(kind: str, source: str, speaker: str, digest: str) -> str:
        canonical = f"{kind}\0{source}\0{speaker}\0{digest}".encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _active_by_fingerprint(self, fingerprint: str) -> tuple[str, dict[str, Any]] | None:
        for memory_id, record in self._data["records"].items():
            if record.get("status") == "active" and record.get("fingerprint") == fingerprint:
                return memory_id, record
        return None

    def _prune_index(self) -> None:
        records = self._data["records"]
        if len(records) <= self.max_records:
            return
        ordered = sorted(
            records,
            key=lambda key: (
                records[key].get("status") != "deleted",
                records[key].get("last_seen_at", ""),
            ),
        )
        for memory_id in ordered[: len(records) - self.max_records]:
            records.pop(memory_id, None)

    def admit(
        self,
        *,
        kind: str,
        content: str,
        source: str,
        speaker: str,
        meta: dict[str, Any] | None = None,
        retention_days: int | None = None,
    ) -> dict[str, Any]:
        now = float(self._clock())
        safe_kind = _safe_identifier(kind or "memory")
        safe_source = _safe_identifier(source or "brain")
        safe_speaker = _safe_identifier(speaker or "system")
        digest = self._digest(content)
        fingerprint = self._fingerprint(safe_kind, safe_source, safe_speaker, digest)
        requested_days = retention_days or DEFAULT_RETENTION_DAYS.get(safe_kind, 90)
        days = min(max(int(requested_days), 1), 3_650)
        provenance = {
            "source": safe_source,
            "speaker": safe_speaker,
            "received_at": _utc_iso(now),
            "origin": _safe_identifier((meta or {}).get("origin", safe_source)),
        }
        with self._lock:
            existing = self._active_by_fingerprint(fingerprint)
            if existing is not None:
                memory_id, record = existing
                last_seen = float(record.get("last_seen_epoch", 0) or 0)
                if now - last_seen <= self.dedupe_window_seconds:
                    record["occurrences"] = int(record.get("occurrences", 1)) + 1
                    record["last_seen_at"] = _utc_iso(now)
                    record["last_seen_epoch"] = now
                    record["expires_at"] = _utc_iso(now + days * 86_400)
                    record["expires_epoch"] = now + days * 86_400
                    trail = list(record.get("provenance", []))[-7:]
                    trail.append(provenance)
                    record["provenance"] = trail
                    self._save()
                    return self._receipt(memory_id, record, deduplicated=True)

            memory_id = f"mem-{uuid.uuid4().hex[:24]}"
            record = {
                "memory_id": memory_id,
                "kind": safe_kind,
                "content_digest": digest,
                "fingerprint": fingerprint,
                "status": "active",
                "occurrences": 1,
                "first_seen_at": _utc_iso(now),
                "first_seen_epoch": now,
                "last_seen_at": _utc_iso(now),
                "last_seen_epoch": now,
                "expires_at": _utc_iso(now + days * 86_400),
                "expires_epoch": now + days * 86_400,
                "retention_days": days,
                "provenance": [provenance],
            }
            self._data["records"][memory_id] = record
            self._prune_index()
            self._save()
            return self._receipt(memory_id, record, deduplicated=False)

    def _receipt(
        self,
        memory_id: str,
        record: dict[str, Any],
        *,
        deduplicated: bool,
    ) -> dict[str, Any]:
        return {
            "memory_id": memory_id,
            "content_digest": record["content_digest"],
            "deduplicated": deduplicated,
            "occurrences": int(record.get("occurrences", 1)),
            "provenance": dict(record.get("provenance", [{}])[-1]),
            "retention": {
                "days": int(record.get("retention_days", 90)),
                "expires_at": record.get("expires_at"),
            },
        }

    def delete_record(self, memory_id: str, *, reason: str) -> dict[str, Any] | None:
        safe_id = str(memory_id or "").strip().lower()
        if not re.fullmatch(r"mem-[0-9a-f]{24}", safe_id):
            return None
        now = float(self._clock())
        with self._lock:
            record = self._data["records"].get(safe_id)
            if record is None:
                return None
            if record.get("status") == "deleted":
                return {
                    "memory_id": safe_id,
                    "deleted": True,
                    "already_deleted": True,
                    "audit_id": record.get("deletion_audit_id", ""),
                }
            audit_id = f"del-{uuid.uuid4().hex[:24]}"
            audit = {
                "audit_id": audit_id,
                "memory_id": safe_id,
                "deleted_at": _utc_iso(now),
                "reason": _safe_reason(reason),
                "kind": record.get("kind"),
                "content_digest": record.get("content_digest"),
                "occurrences": record.get("occurrences", 1),
            }
            with open(self.deletion_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
            record["status"] = "deleted"
            record["deleted_at"] = audit["deleted_at"]
            record["deletion_audit_id"] = audit_id
            record["fingerprint"] = ""
            self._save()
            return {
                "memory_id": safe_id,
                "deleted": True,
                "already_deleted": False,
                "audit_id": audit_id,
            }

    def due_memory_ids(self) -> list[str]:
        now = float(self._clock())
        with self._lock:
            return sorted(
                memory_id
                for memory_id, record in self._data["records"].items()
                if record.get("status") == "active"
                and float(record.get("expires_epoch", math.inf)) <= now
            )

    def state(self) -> dict[str, Any]:
        now = float(self._clock())
        with self._lock:
            active = [
                record for record in self._data["records"].values()
                if record.get("status") == "active"
            ]
            deleted = sum(
                1 for record in self._data["records"].values()
                if record.get("status") == "deleted"
            )
            fresh = sum(
                1 for record in active
                if now - float(record.get("last_seen_epoch", 0)) <= 86_400
            )
            if self.deletion_log_path.exists():
                with open(self.deletion_log_path, "r", encoding="utf-8") as handle:
                    deletion_events = sum(1 for _ in handle)
            else:
                deletion_events = 0
            freshness_scores = [
                max(
                    0.0,
                    1.0 - (
                        now - float(record.get("last_seen_epoch", now))
                    ) / max(
                        float(record.get("retention_days", 90)) * 86_400,
                        1.0,
                    ),
                )
                for record in active
            ]
            return {
                "status": "connected",
                "version": 1,
                "active_records": len(active),
                "deleted_records": deleted,
                "max_records": self.max_records,
                "duplicates_consolidated": sum(
                    max(0, int(record.get("occurrences", 1)) - 1)
                    for record in active
                ),
                "fresh_records": fresh,
                "stale_records": max(0, len(active) - fresh),
                "freshness_score": round(
                    sum(freshness_scores) / len(freshness_scores), 4
                ) if freshness_scores else 0.0,
                "retention_due": len(self.due_memory_ids()),
                "deletion_audit_events": deletion_events,
            }
