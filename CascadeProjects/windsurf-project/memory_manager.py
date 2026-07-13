#!/usr/bin/env python3
"""
memory_manager.py - Unified Memory Interface for Weaver v3.

This module owns Weaver's persistent local memory contract. Runtime services
should resolve their vault through this file so local runs, Docker, and EC2
instances all write to the same structure:

  - people_memory.md
  - weaver_transcript.txt
  - weaver_phone_transcript.txt
  - weaver_dreams.md
  - weaver_headless_thoughts.md
  - weaver_browser_memory.jsonl
  - weaver_memory_events.jsonl
  - quantum_state.txt
  - cloud_vision_memory.md
  - akashic_persist/
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import numpy as np
from memory_lifecycle import MemoryLifecycle


PROJECT_DIR = Path(__file__).resolve().parent

MEMORY_FILENAMES: dict[str, str] = {
    "people": "people_memory.md",
    "transcript": "weaver_transcript.txt",
    "phone_transcript": "weaver_phone_transcript.txt",
    "dreams": "weaver_dreams.md",
    "thoughts": "weaver_headless_thoughts.md",
    "browser": "weaver_browser_memory.jsonl",
    "events": "weaver_memory_events.jsonl",
    "vision": "cloud_vision_memory.md",
    "quantum": "quantum_state.txt",
    "todos": "weaver_todos.md",
    "discord_transcript": "weaver_discord_transcript.txt",
    "state_reconciliation": "weaver_state_reconciliation.jsonl",
    "lora_versions": "weaver_lora_versions.json",
    "memory_index": "weaver_memory_index.json",
    "memory_deletions": "weaver_memory_deletions.jsonl",
}


def resolve_vault_dir(vault_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve Weaver's canonical memory vault.

    Precedence:
      1. explicit argument
      2. WEAVER_VAULT_DIR
      3. project-local Nexus_Vault
    """
    raw = str(vault_dir or os.environ.get("WEAVER_VAULT_DIR") or "Nexus_Vault").strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def default_vault_dir() -> Path:
    """Return the configured canonical vault directory."""
    return resolve_vault_dir(None)


def memory_paths(vault_dir: str | os.PathLike[str] | None = None) -> dict[str, Path]:
    """Return standard memory paths for a vault."""
    vault = resolve_vault_dir(vault_dir)
    paths = {name: vault / filename for name, filename in MEMORY_FILENAMES.items()}
    paths["vault"] = vault
    paths["akashic"] = vault / "akashic_persist"
    paths["akashic_state"] = paths["akashic"] / "akashic_state.npz"
    paths["akashic_meta"] = paths["akashic"] / "akashic_meta.json"
    return paths


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _compact(value: Any, limit: int = 1200) -> str:
    return " ".join(str(value or "").split())[:limit]


def _redact_text(value: Any, limit: int = 5000) -> str:
    text = _compact(value, limit * 2)
    patterns = [
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bASIA[0-9A-Z]{16}\b",
        r"\bsk-[A-Za-z0-9_-]{16,}\b",
        r"(?i)(api[_-]?key|secret|token|password|passphrase)(\s*[:=]\s*)([^\s,;}]+)",
    ]
    for pattern in patterns:
        if "api" in pattern.lower() or "secret" in pattern.lower():
            text = re.sub(pattern, lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
        else:
            text = re.sub(pattern, "[REDACTED]", text)
    return text[:limit]


def _sanitize_payload(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "[MAX_DEPTH]"
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in list(value.items())[:80]:
            key_text = _compact(key, 80)
            if re.search(r"(?i)(secret|token|password|key|credential)", key_text):
                clean[key_text] = "[REDACTED]"
            else:
                clean[key_text] = _sanitize_payload(item, depth + 1)
        return clean
    if isinstance(value, list):
        return [_sanitize_payload(item, depth + 1) for item in value[:80]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _redact_text(value, 1200)


def _tail_file(path: Path, chars: int = 2000) -> str:
    try:
        if not path.exists():
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - chars))
            return f.read()
    except OSError:
        return ""


def _keyword_matches(query: str, text: str) -> bool:
    words = [
        word
        for word in re.findall(r"[a-z0-9']{4,}", query.lower())
        if word not in {"what", "when", "where", "with", "this", "that", "from", "have"}
    ]
    if not words:
        return False
    lower = text.lower()
    return any(word in lower for word in words[:12])


def _search_file(path: Path, query: str, max_lines: int = 12) -> str:
    try:
        if not path.exists() or not query:
            return ""
        matches: list[str] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if _keyword_matches(query, line):
                matches.append(line)
        return "\n".join(matches[-max_lines:])
    except OSError:
        return ""


def _line_count(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _remove_jsonl_memory(path: Path, memory_id: str) -> int:
    if not path.exists():
        return 0
    kept: list[str] = []
    removed = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            item = None
        if isinstance(item, dict) and item.get("memory_id") == memory_id:
            removed += 1
        else:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def _remove_marked_lines(path: Path, memory_id: str) -> int:
    if not path.exists():
        return 0
    marker = f"[{memory_id}]"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    kept = [line for line in lines if marker not in line]
    removed = len(lines) - len(kept)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def _remove_dream_block(path: Path, memory_id: str) -> int:
    if not path.exists():
        return 0
    marker = f"[{memory_id}]"
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = text.split("\n\n---\n")
    kept = [block for block in blocks if marker not in block]
    removed = len(blocks) - len(kept)
    path.write_text("\n\n---\n".join(kept), encoding="utf-8")
    return removed


def _text_vector(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    tokens = re.findall(r"[a-z0-9']+", text.lower())[:800]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec


def _stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(_sanitize_payload(value), sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _keywords(text: str, limit: int = 16) -> list[str]:
    stop = {
        "about", "after", "again", "also", "because", "before", "being", "could",
        "should", "would", "there", "these", "those", "their", "thing", "think",
        "right", "really", "still", "where", "which", "while", "with", "your",
        "youre", "have", "what", "when", "from", "that", "this", "just", "like",
    }
    words = re.findall(r"[a-z0-9']{4,}", str(text or "").lower())
    seen: set[str] = set()
    result: list[str] = []
    for word in words:
        word = word.strip("'")
        if not word or word in stop or word in seen:
            continue
        seen.add(word)
        result.append(word)
        if len(result) >= limit:
            break
    return result


def _file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


class PeopleMemory:
    """Manages people_memory.md."""

    def __init__(self, vault_dir: str | os.PathLike[str] | None = None):
        self.vault_dir = resolve_vault_dir(vault_dir)
        self.people_path = self.vault_dir / MEMORY_FILENAMES["people"]
        self.people_path.parent.mkdir(parents=True, exist_ok=True)
        self._content = ""
        self.refresh()

    def search(self, query: str) -> str:
        self.refresh()
        if not self._content:
            return ""
        if not query:
            return self._content
        query_lower = query.lower()
        matching_lines = [line for line in self._content.splitlines() if query_lower in line.lower()]
        return "\n".join(matching_lines) if matching_lines else self._content

    def add(self, event: Dict[str, Any]) -> None:
        name = _redact_text(event.get("name", "unknown"), 120) or "unknown"
        appearance = _redact_text(event.get("appearance", ""), 500)
        notes = _redact_text(event.get("notes", ""), 800)
        memory_id = _redact_text(event.get("memory_id", ""), 40)

        entry_parts = [f"**{name}**"]
        if appearance:
            entry_parts.append(f"- {appearance}")
        if notes:
            entry_parts.append(f"({notes})")
        marker = f"[{memory_id}] " if memory_id else ""
        new_entry = f"- {marker}{' '.join(entry_parts)}"

        lines = self._content.splitlines()
        updated = False
        for index, line in enumerate(lines):
            if f"**{name}**" in line:
                lines[index] = new_entry
                updated = True
                break
        if not updated:
            lines.append(new_entry)

        self._content = "\n".join(line for line in lines if line.strip()).strip()
        self.people_path.write_text(self._content + ("\n" if self._content else ""), encoding="utf-8")

    def get_all(self) -> str:
        self.refresh()
        return self._content

    def refresh(self) -> None:
        if self.people_path.exists():
            self._content = self.people_path.read_text(encoding="utf-8", errors="replace")
        else:
            self._content = ""


class ConversationMemory:
    """Manages text transcripts."""

    def __init__(self, vault_dir: str | os.PathLike[str] | None = None):
        self.vault_dir = resolve_vault_dir(vault_dir)
        self.main_transcript = self.vault_dir / MEMORY_FILENAMES["transcript"]
        self.phone_transcript = self.vault_dir / MEMORY_FILENAMES["phone_transcript"]
        self.discord_transcript = self.vault_dir / MEMORY_FILENAMES["discord_transcript"]
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    def transcript_for_source(self, source: str = "main") -> Path:
        source_lower = (source or "main").lower()
        if source_lower in {"phone", "twilio", "call", "sms"}:
            return self.phone_transcript
        if source_lower == "discord":
            return self.discord_transcript
        return self.main_transcript

    def search(self, query: str, max_lines: int = 50) -> str:
        matching_lines: list[str] = []
        paths = [
            ("main", self.main_transcript),
            ("phone", self.phone_transcript),
            ("discord", self.discord_transcript),
        ]
        for label, path in paths:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not query or _keyword_matches(query, line) or query.lower() in line.lower():
                    matching_lines.append(f"[{label}] {line}")
        return "\n".join(matching_lines[-max_lines:])

    def add(self, event: Dict[str, Any]) -> None:
        speaker = _redact_text(event.get("speaker", "user"), 80) or "user"
        content = _redact_text(event.get("content", ""), 5000)
        timestamp = _redact_text(event.get("timestamp", _utc_iso()), 80)
        source = _redact_text(event.get("source", "main"), 80) or "main"
        memory_id = _redact_text(event.get("memory_id", ""), 40)
        if not content:
            return
        path = self.transcript_for_source(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            marker = f" [{memory_id}]" if memory_id else ""
            f.write(f"[{timestamp}]{marker} {speaker.upper()}: {content}\n")

    def get_recent(self, source: str = "main", lines: int = 40) -> str:
        transcript = self.transcript_for_source(source)
        if not transcript.exists():
            return ""
        all_lines = transcript.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(all_lines[-lines:])

    def get_summary(self, source: str = "main", chars: int = 2000) -> str:
        return _tail_file(self.transcript_for_source(source), chars)


class AkashicPersistence:
    """Manages akashic_persist/ vector snapshots."""

    def __init__(self, vault_dir: str | os.PathLike[str] | None = None):
        self.vault_dir = resolve_vault_dir(vault_dir)
        self.persist_dir = self.vault_dir / "akashic_persist"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.persist_dir / "akashic_state.npz"
        self.meta_path = self.persist_dir / "akashic_meta.json"

    async def query(self, context: str) -> Dict[str, Any]:
        files = sorted(self.persist_dir.glob("*.npz"))
        return {
            "note": "Akashic query returns persisted vector file inventory; cosine query lives in AkashicHub.",
            "persist_dir": str(self.persist_dir),
            "file_count": len(files),
            "files": [str(path) for path in files[-20:]],
        }

    async def write(self, event: Dict[str, Any]) -> None:
        await asyncio.to_thread(self.write_sync, event)

    def write_sync(self, event: Dict[str, Any]) -> None:
        lobe_id = _redact_text(event.get("lobe_id", "unknown"), 120) or "unknown"
        state_vec = event.get("state_vec")
        if state_vec is None:
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = self.persist_dir / f"{lobe_id}_{timestamp}.npz"
        np.savez_compressed(filename, state=state_vec)

    def write_text_lobe_sync(self, lobe_id: str, text: str, meta: dict[str, Any] | None = None) -> None:
        lobe_id = _redact_text(lobe_id, 120) or "weaver_memory"
        text = _redact_text(text, 5000)
        if not text:
            return
        arrays: dict[str, Any] = {}
        if self.state_path.exists():
            try:
                with np.load(self.state_path) as existing:
                    arrays.update({name: existing[name] for name in existing.files})
            except Exception:
                arrays = {}
        arrays[lobe_id] = _text_vector(text)
        np.savez_compressed(self.state_path, **arrays)

        payload: dict[str, Any] = {"dim": 256, "trace_depth": 32, "timestamps": {}, "meta": {}}
        if self.meta_path.exists():
            try:
                payload.update(json.loads(self.meta_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        payload.setdefault("timestamps", {})[lobe_id] = datetime.now(timezone.utc).timestamp()
        payload.setdefault("meta", {})[lobe_id] = {
            **payload.get("meta", {}).get(lobe_id, {}),
            **_sanitize_payload(meta or {}),
            "source": "weaver-memory-manager",
            "preview": _redact_text(text, 240),
        }
        payload["saved_at"] = datetime.now(timezone.utc).timestamp()
        self.meta_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    async def write_text_lobe(self, lobe_id: str, text: str, meta: dict[str, Any] | None = None) -> None:
        await asyncio.to_thread(self.write_text_lobe_sync, lobe_id, text, meta)

    def delete_memory_sync(self, memory_id: str) -> int:
        """Remove latest vector lobes whose metadata points at a deleted memory."""

        if not self.meta_path.exists():
            return 0
        try:
            payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return 0
        metadata = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        lobe_ids = [
            lobe_id
            for lobe_id, item in metadata.items()
            if isinstance(item, dict) and item.get("memory_id") == memory_id
        ]
        if not lobe_ids:
            return 0
        arrays: dict[str, Any] = {}
        if self.state_path.exists():
            try:
                with np.load(self.state_path) as existing:
                    arrays = {
                        name: existing[name]
                        for name in existing.files
                        if name not in lobe_ids
                    }
            except Exception:
                arrays = {}
        if arrays:
            np.savez_compressed(self.state_path, **arrays)
        elif self.state_path.exists():
            self.state_path.unlink()
        for lobe_id in lobe_ids:
            metadata.pop(lobe_id, None)
            if isinstance(payload.get("timestamps"), dict):
                payload["timestamps"].pop(lobe_id, None)
        payload["meta"] = metadata
        payload["saved_at"] = datetime.now(timezone.utc).timestamp()
        self.meta_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return len(lobe_ids)


class MemoryManager:
    """Unified interface for all Weaver memory systems."""

    def __init__(self, vault_dir: str | os.PathLike[str] | None = None):
        self.vault_dir = resolve_vault_dir(vault_dir)
        self.paths = memory_paths(self.vault_dir)
        self._ensure_layout()
        self.people = PeopleMemory(self.vault_dir)
        self.conversations = ConversationMemory(self.vault_dir)
        self.akashic = AkashicPersistence(self.vault_dir)
        self.lifecycle = MemoryLifecycle(
            self.paths["memory_index"],
            self.paths["memory_deletions"],
        )

    def _ensure_layout(self) -> None:
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.paths["akashic"].mkdir(parents=True, exist_ok=True)
        for key in (
            "people",
            "transcript",
            "phone_transcript",
            "dreams",
            "thoughts",
            "browser",
            "events",
            "vision",
            "quantum",
            "discord_transcript",
            "state_reconciliation",
            "lora_versions",
            "memory_index",
            "memory_deletions",
        ):
            self.paths[key].parent.mkdir(parents=True, exist_ok=True)
            if key == "lora_versions" and not self.paths[key].exists():
                self.paths[key].write_text(
                    json.dumps({"active_version": "", "versions": []}, indent=2),
                    encoding="utf-8",
                )
            elif key == "memory_index" and not self.paths[key].exists():
                self.paths[key].write_text(
                    json.dumps({"version": 1, "records": {}}, indent=2),
                    encoding="utf-8",
                )
            else:
                self.paths[key].touch(exist_ok=True)

    def sources(self) -> dict[str, str]:
        return {name: str(path) for name, path in self.paths.items()}

    def state(self) -> dict[str, Any]:
        return {
            "status": "connected",
            "vault": str(self.vault_dir),
            "sources": self.sources(),
            "counts": {
                "transcript_lines": _line_count(self.paths["transcript"]),
                "phone_lines": _line_count(self.paths["phone_transcript"]),
                "discord_lines": _line_count(self.paths["discord_transcript"]),
                "dream_lines": _line_count(self.paths["dreams"]),
                "thought_lines": _line_count(self.paths["thoughts"]),
                "browser_memory_events": _line_count(self.paths["browser"]),
                "memory_events": _line_count(self.paths["events"]),
                "akashic_files": len(list(self.paths["akashic"].glob("*.npz"))),
            },
            "recent": {
                "transcript": _redact_text(_tail_file(self.paths["transcript"], 1200), 1200),
                "phone": _redact_text(_tail_file(self.paths["phone_transcript"], 900), 900),
                "dream": _redact_text(_tail_file(self.paths["dreams"], 1200), 1200),
                "thought": _redact_text(_tail_file(self.paths["thoughts"], 900), 900),
                "browser": _redact_text(_tail_file(self.paths["browser"], 900), 900),
            },
            "lifecycle": self.lifecycle.state(),
        }

    async def recall(self, context: str) -> Dict[str, Any]:
        return {
            "people": self.people.search(context),
            "conversations": self.conversations.search(context),
            "vectors": await self.akashic.query(context),
        }

    async def remember(self, event: Dict[str, Any]) -> None:
        event_type = str(event.get("type", "")).lower()
        if event_type == "person":
            receipt = self.append_event_sync(
                "person",
                f"{event.get('name', 'unknown')}: {event.get('appearance', '')} {event.get('notes', '')}",
                source=event.get("source", "memory-manager"),
                speaker=event.get("speaker", "system"),
                meta=event,
            )
            if receipt and not receipt.get("deduplicated"):
                self.people.add({
                    **event,
                    "memory_id": receipt.get("memory_id", ""),
                })
        elif event_type == "conversation":
            self.append_event_sync(
                "conversation",
                str(event.get("content", "")),
                source=str(event.get("source", "main")),
                speaker=str(event.get("speaker", "user")),
                meta=event,
            )
        elif event_type == "vector":
            await self.akashic.write(event)
        else:
            self.append_event_sync(
                event_type or "memory",
                str(event.get("content", event)),
                source=str(event.get("source", "memory-manager")),
                speaker=str(event.get("speaker", "system")),
                meta=event,
            )

    def append_event_sync(
        self,
        kind: str,
        content: str,
        *,
        source: str = "brain",
        speaker: str = "",
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        content = _redact_text(content, 5000)
        if not content:
            return None

        kind = _redact_text(kind or "memory", 80)
        source = _redact_text(source or "brain", 80)
        speaker = _redact_text(speaker or "", 80)
        timestamp = _utc_iso()
        raw_retention = (meta or {}).get("retention_days")
        retention_days = (
            int(raw_retention)
            if isinstance(raw_retention, (int, float)) and not isinstance(raw_retention, bool)
            else None
        )
        lifecycle = self.lifecycle.admit(
            kind=kind,
            content=content,
            source=source,
            speaker=speaker,
            meta=meta,
            retention_days=retention_days,
        )
        if lifecycle["deduplicated"]:
            return {
                "timestamp": timestamp,
                "kind": kind,
                "source": source,
                "speaker": speaker,
                "memory": lifecycle,
                "deduplicated": True,
            }
        event = {
            "timestamp": timestamp,
            "memory_id": lifecycle["memory_id"],
            "kind": kind,
            "source": source,
            "speaker": speaker,
            "content": content,
            "content_digest": lifecycle["content_digest"],
            "provenance": lifecycle["provenance"],
            "retention": lifecycle["retention"],
            "deduplicated": False,
            "meta": _sanitize_payload(meta or {}),
        }

        with open(self.paths["events"], "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

        if kind == "conversation":
            self.conversations.add(
                {
                    "timestamp": timestamp,
                    "speaker": speaker or source,
                    "content": content,
                    "source": source,
                    "memory_id": lifecycle["memory_id"],
                }
            )
        elif kind == "thought":
            with open(self.paths["thoughts"], "a", encoding="utf-8") as f:
                f.write(
                    f"\n- [{timestamp}] [{lifecycle['memory_id']}] ({source}) {content}\n"
                )
        elif kind == "dream":
            with open(self.paths["dreams"], "a", encoding="utf-8") as f:
                f.write(
                    f"\n\n---\n### Headless Dream - {timestamp} "
                    f"[{lifecycle['memory_id']}] ({source})\n{content}\n"
                )
        elif kind == "browser_memory":
            with open(self.paths["browser"], "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        elif kind == "vision":
            with open(self.paths["vision"], "a", encoding="utf-8") as f:
                f.write(
                    f"\n- [{timestamp}] [{lifecycle['memory_id']}] ({source}) {content}\n"
                )
        elif kind == "quantum":
            with open(self.paths["quantum"], "a", encoding="utf-8") as f:
                f.write(f"\n[{timestamp}] [{lifecycle['memory_id']}] {content}\n")

        lobe_map = {
            "conversation": "aws_brain_conversation_memory",
            "thought": "aws_headless_thought_memory",
            "dream": "aws_headless_dream_memory",
            "browser_memory": "browser_evolution_memory",
            "person": "people_memory",
            "vision": "cloud_vision_memory",
            "quantum": "quantum_state_memory",
        }
        self.akashic.write_text_lobe_sync(lobe_map.get(kind, "weaver_memory"), content, event)
        return event

    async def append_event(
        self,
        kind: str,
        content: str,
        *,
        source: str = "brain",
        speaker: str = "",
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(
            self.append_event_sync,
            kind,
            content,
            source=source,
            speaker=speaker,
            meta=meta,
        )

    def delete_memory_sync(self, memory_id: str, *, reason: str) -> dict[str, Any] | None:
        """Delete one indexed memory from canonical and derived stores with an audit tombstone."""

        receipt = self.lifecycle.delete_record(memory_id, reason=reason)
        if receipt is None:
            return None
        removed = 0
        storage_errors: list[str] = []
        operations = (
            ("events", lambda: _remove_jsonl_memory(self.paths["events"], memory_id)),
            ("browser", lambda: _remove_jsonl_memory(self.paths["browser"], memory_id)),
            ("people", lambda: _remove_marked_lines(self.paths["people"], memory_id)),
            ("transcript", lambda: _remove_marked_lines(self.paths["transcript"], memory_id)),
            ("phone", lambda: _remove_marked_lines(self.paths["phone_transcript"], memory_id)),
            ("discord", lambda: _remove_marked_lines(self.paths["discord_transcript"], memory_id)),
            ("thoughts", lambda: _remove_marked_lines(self.paths["thoughts"], memory_id)),
            ("dreams", lambda: _remove_dream_block(self.paths["dreams"], memory_id)),
            ("vision", lambda: _remove_marked_lines(self.paths["vision"], memory_id)),
            ("quantum", lambda: _remove_marked_lines(self.paths["quantum"], memory_id)),
        )
        for label, operation in operations:
            try:
                removed += int(operation())
            except OSError:
                storage_errors.append(label)
        try:
            removed += self.akashic.delete_memory_sync(memory_id)
        except OSError:
            storage_errors.append("akashic")
        self.people.refresh()
        return {
            **receipt,
            "storage_records_removed": removed,
            "storage_errors": storage_errors,
        }

    async def delete_memory(self, memory_id: str, *, reason: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(
            self.delete_memory_sync,
            memory_id,
            reason=reason,
        )

    def expire_due_sync(self, *, limit: int = 32) -> list[dict[str, Any]]:
        receipts: list[dict[str, Any]] = []
        for memory_id in self.lifecycle.due_memory_ids()[: max(1, min(int(limit), 128))]:
            receipt = self.delete_memory_sync(memory_id, reason="retention-expired")
            if receipt is not None:
                receipts.append(receipt)
        return receipts

    def build_context(self, query: str = "", max_chars: int = 7200) -> str:
        parts: list[str] = []
        people = _tail_file(self.paths["people"], 1600)
        if people:
            parts.append(f"People memory:\n{people[-1600:]}")
        recent = _tail_file(self.paths["transcript"], 2600)
        if recent:
            parts.append(f"Recent transcript:\n{recent[-2600:]}")
        phone = _tail_file(self.paths["phone_transcript"], 1000)
        if phone:
            parts.append(f"Recent phone memory:\n{phone[-1000:]}")
        discord = _tail_file(self.paths["discord_transcript"], 900)
        if discord:
            parts.append(f"Recent Discord memory:\n{discord[-900:]}")
        dreams = _tail_file(self.paths["dreams"], 2200)
        if dreams:
            parts.append(f"Dream memory:\n{dreams[-2200:]}")
        thoughts = _tail_file(self.paths["thoughts"], 1400)
        if thoughts:
            parts.append(f"Headless thought memory:\n{thoughts[-1400:]}")
        browser = _tail_file(self.paths["browser"], 1800)
        if browser:
            parts.append(f"Browser evolution memory:\n{browser[-1800:]}")
        quantum = _tail_file(self.paths["quantum"], 700)
        if quantum:
            parts.append(f"Quantum state memory:\n{quantum[-700:]}")
        if re.search(r"(?i)\b(see|saw|visual|camera|face|screenshot|image|room|body|avatar)\b", query):
            vision = _tail_file(self.paths["vision"], 1200)
            if vision:
                parts.append(f"Visual memory:\n{vision[-1200:]}")

        matches = "\n".join(
            block
            for block in [
                _search_file(self.paths["transcript"], query),
                _search_file(self.paths["phone_transcript"], query),
                _search_file(self.paths["discord_transcript"], query),
                _search_file(self.paths["dreams"], query),
                _search_file(self.paths["thoughts"], query),
                _search_file(self.paths["browser"], query),
            ]
            if block
        )
        if matches:
            parts.append(f"Memory search hits:\n{matches[-1800:]}")
        return _redact_text("\n\n".join(parts), max_chars)

    async def memory_context(self, query: str = "", max_chars: int = 7200) -> str:
        return await asyncio.to_thread(self.build_context, query, max_chars)

    async def fetch_quantum_bias(
        self,
        url: str = "http://127.0.0.1:9997/quantum/bias",
        timeout: float = 3.0,
    ) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=timeout)
                if response.status_code == 200:
                    return response.json()
        except Exception:
            pass
        return {"dominant": "unknown", "weights": {}, "last_measurement": None}

    async def update_people_from_transcript(self, lc_llm, transcript_block: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        current = self.people.get_all()
        prompt = [
            SystemMessage(
                content=(
                    "You are Weaver's people memory. Extract every person mentioned "
                    "in the conversation: their name, relationship to the caller, "
                    "and any key facts. Merge updates into existing entries. "
                    "Return only the updated people list in markdown bullet format."
                )
            )
        ]
        if current:
            prompt.append(HumanMessage(content=f"Existing people list:\n{current}"))
        prompt.append(HumanMessage(content=f"Conversation:\n{transcript_block}"))
        result = await lc_llm.ainvoke(prompt)
        updated = str(result.content).strip()
        if updated and updated != current:
            self.people._content = updated
            self.people.people_path.write_text(updated + "\n", encoding="utf-8")
        return updated

    def build_phone_context(self, quantum_bias: Optional[Dict[str, Any]] = None) -> str:
        parts: list[str] = []
        people = self.people.get_all()
        if people:
            parts.append(f"## PEOPLE YOU KNOW:\n{people}\n")
        phone_history = self.conversations.get_recent(source="phone", lines=40)
        if phone_history:
            parts.append(f"## RECENT PHONE CONVERSATIONS:\n{phone_history}\n")
        main_summary = self.conversations.get_summary(source="main", chars=2000)
        if main_summary:
            parts.append(f"## CONVERSATION MEMORY:\n{main_summary[-1000:]}\n")
        if quantum_bias and quantum_bias.get("dominant", "unknown") != "unknown":
            dom = quantum_bias["dominant"]
            weights = quantum_bias.get("weights", {})
            w_str = ", ".join(f"{key}={value:.2f}" for key, value in weights.items()) if weights else "N/A"
            parts.append(f"## QUANTUM STATE:\nDominant pathway: {dom}\nWeights: {w_str}\n")
        return "\n".join(parts)

    def snapshot_conversation_state(
        self,
        *,
        caller: str = "unknown",
        latest_user: str = "",
        conversation_history: list[dict[str, Any]] | None = None,
        quantum_bias: Optional[Dict[str, Any]] = None,
        source: str = "phone",
        turn_id: str | int | None = None,
    ) -> dict[str, Any]:
        """Capture the current turn contract for cross-service reconciliation.

        This is intentionally small and hash-heavy. Services can pass the snapshot
        through slow paths (n8n, Pineal Gate, LoRA) without copying the whole
        transcript, then verify the returned thought still belongs to this exact
        caller/turn before it is allowed to steer speech.
        """
        history = list(conversation_history or [])[-8:]
        recent_text = "\n".join(
            f"{item.get('role', 'unknown')}: {item.get('content', '')}"
            for item in history
        )
        latest_user = _redact_text(latest_user, 2000)
        caller = _redact_text(caller or "unknown", 120) or "unknown"
        quantum_bias = _sanitize_payload(quantum_bias or {})
        keywords = _keywords(" ".join([latest_user, recent_text]), 20)
        payload = {
            "caller": caller.lower(),
            "latest_user_hash": _stable_hash(latest_user),
            "recent_hash": _stable_hash(recent_text),
            "history_len": len(history),
            "quantum_dominant": str(quantum_bias.get("dominant", "unknown")).lower(),
            "keywords": keywords,
            "turn_id": str(turn_id or ""),
            "source": source,
        }
        payload["signature"] = _stable_hash(payload, 24)
        payload["timestamp"] = _utc_iso()
        payload["latest_user_preview"] = latest_user[:240]
        return payload

    def reconcile_thought(
        self,
        *,
        thought: str,
        expected: dict[str, Any],
        caller: str = "unknown",
        latest_user: str = "",
        conversation_history: list[dict[str, Any]] | None = None,
        quantum_bias: Optional[Dict[str, Any]] = None,
        source: str = "pineal_gate",
        min_score: float = 0.62,
    ) -> dict[str, Any]:
        """Verify a slow-path thought still matches the live conversation state."""
        current = self.snapshot_conversation_state(
            caller=caller,
            latest_user=latest_user,
            conversation_history=conversation_history,
            quantum_bias=quantum_bias,
            source=source,
            turn_id=expected.get("turn_id"),
        )
        reasons: list[str] = []
        score = 0.0

        if current.get("latest_user_hash") == expected.get("latest_user_hash"):
            score += 0.38
        else:
            reasons.append("stale_user_turn")

        if current.get("caller") == expected.get("caller"):
            score += 0.17
        else:
            reasons.append("caller_changed")

        expected_keywords = set(expected.get("keywords") or [])
        thought_keywords = set(_keywords(thought, 24))
        context_keywords = set(current.get("keywords") or [])
        overlap = len((thought_keywords & expected_keywords) | (thought_keywords & context_keywords))
        if expected_keywords:
            keyword_score = min(0.28, 0.28 * (overlap / max(3, min(len(expected_keywords), 8))))
            score += keyword_score
            if overlap == 0:
                reasons.append("no_topic_overlap")
        else:
            score += 0.12

        if current.get("quantum_dominant") == expected.get("quantum_dominant"):
            score += 0.07
        elif expected.get("quantum_dominant") in {"", "unknown", None}:
            score += 0.04
        else:
            reasons.append("quantum_context_shift")

        if current.get("history_len", 0) >= expected.get("history_len", 0):
            score += 0.10
        else:
            reasons.append("history_regressed")

        hard_stale = any(reason in reasons for reason in ("stale_user_turn", "caller_changed"))
        ok = score >= min_score and not hard_stale
        result = {
            "timestamp": _utc_iso(),
            "ok": ok,
            "action": "inject" if ok else "quarantine",
            "score": round(score, 3),
            "min_score": min_score,
            "reasons": reasons,
            "source": source,
            "expected_signature": expected.get("signature", ""),
            "current_signature": current.get("signature", ""),
            "turn_id": expected.get("turn_id", ""),
            "caller": current.get("caller", "unknown"),
            "thought_preview": _redact_text(thought, 400),
        }
        with open(self.paths["state_reconciliation"], "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        return result

    def _read_lora_registry(self) -> dict[str, Any]:
        path = self.paths["lora_versions"]
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
            if not isinstance(data, dict):
                raise ValueError("registry is not an object")
        except Exception:
            data = {}
        data.setdefault("active_version", "")
        data.setdefault("versions", [])
        if not isinstance(data["versions"], list):
            data["versions"] = []
        return data

    def _write_lora_registry(self, data: dict[str, Any]) -> None:
        data["updated_at"] = _utc_iso()
        self.paths["lora_versions"].write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def register_lora_version(
        self,
        *,
        adapter_path: str = "",
        gguf_path: str = "",
        backend: str = "transformers",
        base_model: str = "",
        notes: str = "",
        activate: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a LoRA/GGUF soul checkpoint so it can be rolled back later."""
        adapter = Path(adapter_path).expanduser() if adapter_path else Path("")
        gguf = Path(gguf_path).expanduser() if gguf_path else Path("")
        cfg_hash = ""
        cfg_path = adapter / "adapter_config.json" if adapter_path else Path("")
        if adapter_path and cfg_path.is_file():
            cfg_hash = _stable_hash(cfg_path.read_text(encoding="utf-8", errors="replace"), 20)
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                base_model = base_model or str(cfg.get("base_model_name_or_path", ""))
            except Exception:
                pass
        fingerprints = {
            "adapter": _file_fingerprint(adapter) if adapter_path else {},
            "adapter_config": _file_fingerprint(cfg_path) if adapter_path else {},
            "adapter_model": _file_fingerprint(adapter / "adapter_model.safetensors") if adapter_path else {},
            "gguf": _file_fingerprint(gguf) if gguf_path else {},
        }
        version_seed = {
            "adapter_path": str(adapter) if adapter_path else "",
            "gguf_path": str(gguf) if gguf_path else "",
            "backend": backend,
            "base_model": base_model,
            "config_hash": cfg_hash,
            "fingerprints": fingerprints,
        }
        version_id = _stable_hash(version_seed, 20)
        record = {
            "version_id": version_id,
            "created_at": _utc_iso(),
            "backend": _redact_text(backend, 80),
            "adapter_path": str(adapter) if adapter_path else "",
            "gguf_path": str(gguf) if gguf_path else "",
            "base_model": _redact_text(base_model, 300),
            "config_hash": cfg_hash,
            "notes": _redact_text(notes, 500),
            "metadata": _sanitize_payload(metadata or {}),
            "fingerprints": fingerprints,
        }
        data = self._read_lora_registry()
        versions = [item for item in data["versions"] if item.get("version_id") != version_id]
        versions.append(record)
        data["versions"] = versions[-80:]
        if activate or not data.get("active_version"):
            data["active_version"] = version_id
        self._write_lora_registry(data)
        return record

    def lora_versions(self) -> dict[str, Any]:
        return self._read_lora_registry()

    def active_lora_version(self) -> dict[str, Any] | None:
        data = self._read_lora_registry()
        active = data.get("active_version")
        for record in data.get("versions", []):
            if record.get("version_id") == active:
                return record
        return None

    def rollback_lora_version(self, version_id: str = "") -> dict[str, Any]:
        data = self._read_lora_registry()
        versions = data.get("versions", [])
        target = None
        if version_id:
            target = next((item for item in versions if item.get("version_id") == version_id), None)
        elif len(versions) >= 2:
            current = data.get("active_version")
            previous = [item for item in versions if item.get("version_id") != current]
            target = previous[-1] if previous else None
        if not target:
            raise ValueError("No LoRA version available for rollback")
        data["active_version"] = target["version_id"]
        data.setdefault("rollback_events", []).append({
            "timestamp": _utc_iso(),
            "version_id": target["version_id"],
            "backend": target.get("backend", ""),
            "notes": "active LoRA version changed",
        })
        data["rollback_events"] = data["rollback_events"][-100:]
        self._write_lora_registry(data)
        return target

    def build_vtv_context(self) -> str:
        parts: list[str] = []
        people = self.people.get_all()
        if people:
            parts.append(f"People You Know:\n{people}\n")
        recent = self.conversations.get_summary(source="main", chars=4000)
        if recent:
            parts.append(f"Recent Conversation:\n{recent}\n")
        return "\n".join(parts)

    def refresh(self) -> None:
        self.people.refresh()


async def _example() -> None:
    mem = MemoryManager()
    await mem.remember(
        {
            "type": "person",
            "name": "Nate",
            "appearance": "short brown hair, glasses",
            "notes": "AI researcher, works on quantum systems",
        }
    )
    await mem.remember(
        {
            "type": "conversation",
            "speaker": "user",
            "content": "How do I test the phone bridge?",
            "source": "phone",
        }
    )
    print(await mem.recall("Nate"))


if __name__ == "__main__":
    asyncio.run(_example())
