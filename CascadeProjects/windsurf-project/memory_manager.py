#!/usr/bin/env python3
"""
memory_manager.py — Unified Memory Interface for Weaver v3
==========================================================
Consolidates 3 separate memory systems into a single interface:
  1. PeopleMemory (people_memory.md) — face IDs + descriptions
  2. ConversationMemory (weaver_transcript.txt, weaver_phone_transcript.txt)
  3. AkashicPersistence (akashic_persist/) — vector state snapshots

Usage:
    from memory_manager import MemoryManager

    mem = MemoryManager(vault_dir="Nexus_Vault")

    # Recall context
    context = await mem.recall("What did Nate say about quantum?")

    # Remember events
    await mem.remember({
        "type": "person",
        "name": "Nate",
        "appearance": "short brown hair, glasses",
        "notes": "works on AI systems",
    })

    await mem.remember({
        "type": "conversation",
        "speaker": "user",
        "content": "How do I test the phone bridge?",
        "timestamp": "2026-04-30T06:30:00",
    })
"""

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 1. PEOPLE MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class PeopleMemory:
    """Manages people_memory.md — persistent face ID + descriptions."""

    def __init__(self, vault_dir: str):
        self.vault_dir = Path(vault_dir)
        self.people_path = self.vault_dir / "people_memory.md"
        self.people_path.parent.mkdir(parents=True, exist_ok=True)

        # Load on init
        self._content = ""
        if self.people_path.exists():
            self._content = self.people_path.read_text(encoding="utf-8")

    def search(self, query: str) -> str:
        """Return all people matching the query (case-insensitive)."""
        if not self._content:
            return ""

        query_lower = query.lower()
        matching_lines = []
        for line in self._content.splitlines():
            if query_lower in line.lower():
                matching_lines.append(line)

        return "\n".join(matching_lines) if matching_lines else self._content

    def add(self, event: Dict[str, Any]) -> None:
        """Add or update a person entry."""
        name = event.get("name", "unknown")
        appearance = event.get("appearance", "")
        notes = event.get("notes", "")

        # Build new entry
        entry_parts = [f"**{name}**"]
        if appearance:
            entry_parts.append(f"— {appearance}")
        if notes:
            entry_parts.append(f"({notes})")
        new_entry = f"- {' '.join(entry_parts)}"

        # Update existing entry if name exists
        lines = self._content.splitlines()
        updated = False
        for i, line in enumerate(lines):
            if f"**{name}**" in line:
                lines[i] = new_entry
                updated = True
                break

        if not updated:
            lines.append(new_entry)

        self._content = "\n".join(lines).strip()
        self.people_path.write_text(self._content, encoding="utf-8")

    def get_all(self) -> str:
        """Return full people memory."""
        return self._content

    def refresh(self) -> None:
        """Reload from disk."""
        if self.people_path.exists():
            self._content = self.people_path.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONVERSATION MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """Manages weaver_transcript.txt and weaver_phone_transcript.txt."""

    def __init__(self, vault_dir: str):
        self.vault_dir = Path(vault_dir)
        self.main_transcript = self.vault_dir / "weaver_transcript.txt"
        self.phone_transcript = self.vault_dir / "weaver_phone_transcript.txt"

        self.main_transcript.parent.mkdir(parents=True, exist_ok=True)
        self.phone_transcript.parent.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, max_lines: int = 50) -> str:
        """Search both transcripts for the query (case-insensitive)."""
        query_lower = query.lower()
        matching_lines = []

        # Search main transcript
        if self.main_transcript.exists():
            lines = self.main_transcript.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if query_lower in line.lower():
                    matching_lines.append(f"[main] {line}")

        # Search phone transcript
        if self.phone_transcript.exists():
            lines = self.phone_transcript.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if query_lower in line.lower():
                    matching_lines.append(f"[phone] {line}")

        return "\n".join(matching_lines[-max_lines:])

    def add(self, event: Dict[str, Any]) -> None:
        """Append a conversation turn."""
        speaker = event.get("speaker", "user")
        content = event.get("content", "")
        timestamp = event.get("timestamp", datetime.now().isoformat())
        source = event.get("source", "main")  # "main" or "phone"

        if not content:
            return

        line = f"[{timestamp}] {speaker.upper()}: {content}\n"

        if source == "phone":
            with open(self.phone_transcript, "a", encoding="utf-8") as f:
                f.write(line)
        else:
            with open(self.main_transcript, "a", encoding="utf-8") as f:
                f.write(line)

    def get_recent(self, source: str = "main", lines: int = 40) -> str:
        """Get last N lines from a transcript."""
        transcript = self.phone_transcript if source == "phone" else self.main_transcript

        if not transcript.exists():
            return ""

        all_lines = transcript.read_text(encoding="utf-8").splitlines()
        return "\n".join(all_lines[-lines:])

    def get_summary(self, source: str = "main", chars: int = 2000) -> str:
        """Get last N chars from a transcript (for context injection)."""
        transcript = self.phone_transcript if source == "phone" else self.main_transcript

        if not transcript.exists():
            return ""

        with open(transcript, "r", encoding="utf-8") as f:
            f.seek(0, 2)  # End of file
            size = f.tell()
            f.seek(max(0, size - chars))
            return f.read()


# ══════════════════════════════════════════════════════════════════════════════
# 3. AKASHIC PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

class AkashicPersistence:
    """Manages akashic_persist/ — vector state snapshots."""

    def __init__(self, vault_dir: str):
        self.vault_dir = Path(vault_dir)
        self.persist_dir = self.vault_dir / "akashic_persist"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    async def query(self, context: str) -> Dict[str, Any]:
        """Query persisted vector states (placeholder — needs AkashicHub integration)."""
        # TODO: Load .npz files, compute cosine similarity with query embedding
        return {
            "note": "Akashic query not yet implemented",
            "persist_dir": str(self.persist_dir),
            "files": list(self.persist_dir.glob("*.npz")),
        }

    async def write(self, event: Dict[str, Any]) -> None:
        """Save a vector state snapshot."""
        lobe_id = event.get("lobe_id", "unknown")
        state_vec = event.get("state_vec")

        if state_vec is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.persist_dir / f"{lobe_id}_{timestamp}.npz"

        np.savez_compressed(filename, state=state_vec)


# ══════════════════════════════════════════════════════════════════════════════
# 4. UNIFIED MEMORY MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class MemoryManager:
    """Unified interface for all memory systems."""

    def __init__(self, vault_dir: str = "Nexus_Vault"):
        self.vault_dir = vault_dir
        self.people = PeopleMemory(vault_dir)
        self.conversations = ConversationMemory(vault_dir)
        self.akashic = AkashicPersistence(vault_dir)

    async def recall(self, context: str) -> Dict[str, Any]:
        """Query all memory sources, return unified context.

        Args:
            context: Search query (name, topic, keyword).

        Returns:
            Dict with keys: people, conversations, vectors.
        """
        return {
            "people": self.people.search(context),
            "conversations": self.conversations.search(context),
            "vectors": await self.akashic.query(context),
        }

    async def remember(self, event: Dict[str, Any]) -> None:
        """Save to all relevant memory stores.

        Args:
            event: Dict with keys:
                - type: "person" | "conversation" | "vector"
                - (other fields depend on type)
        """
        event_type = event.get("type", "")

        if event_type == "person":
            self.people.add(event)

        elif event_type == "conversation":
            self.conversations.add(event)

        elif event_type == "vector":
            await self.akashic.write(event)

    def build_phone_context(self) -> str:
        """Build memory context for phone calls (combines all sources)."""
        parts = []

        people = self.people.get_all()
        if people:
            parts.append(f"## PEOPLE YOU KNOW:\n{people}\n")

        phone_history = self.conversations.get_recent(source="phone", lines=40)
        if phone_history:
            parts.append(f"## RECENT PHONE CONVERSATIONS:\n{phone_history}\n")

        main_summary = self.conversations.get_summary(source="main", chars=2000)
        if main_summary:
            parts.append(f"## CONVERSATION MEMORY:\n{main_summary[-1000:]}\n")

        return "\n".join(parts)

    def build_vtv_context(self) -> str:
        """Build memory context for VTV (voice/video/text)."""
        parts = []

        people = self.people.get_all()
        if people:
            parts.append(f"People You Know:\n{people}\n")

        recent = self.conversations.get_summary(source="main", chars=4000)
        if recent:
            parts.append(f"Recent Conversation:\n{recent}\n")

        return "\n".join(parts)

    def refresh(self) -> None:
        """Reload all memory from disk."""
        self.people.refresh()


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ══════════════════════════════════════════════════════════════════════════════

async def _example():
    mem = MemoryManager("Nexus_Vault")

    # Add a person
    await mem.remember({
        "type": "person",
        "name": "Nate",
        "appearance": "short brown hair, glasses",
        "notes": "AI researcher, works on quantum systems",
    })

    # Add conversation turn
    await mem.remember({
        "type": "conversation",
        "speaker": "user",
        "content": "How do I test the phone bridge?",
        "source": "phone",
    })

    # Recall context
    context = await mem.recall("Nate")
    print(context)


if __name__ == "__main__":
    asyncio.run(_example())
