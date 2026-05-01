#!/usr/bin/env python3
"""
obsidian_bridge.py — Weaver's Visual Cortex (Obsidian ↔ n8n Bridge)
====================================================================
A two-way telepathic link between a local Obsidian Vault and Weaver's
n8n nervous system.

Watcher (Obsidian → Weaver):
    Monitors ~/Weaver_Vault for .md files containing #weaver.
    Extracts the text and POSTs it to the n8n input gateway webhook.

Writer (Weaver → Obsidian):
    Runs a local HTTP server (port 5679) that catches outgoing webhooks
    from n8n.  When Weaver replies, the collapsed response is appended
    to the originating .md file under '### 👁️ Weaver's Resonance'.

Synaptic Linking:
    Automatically injects Obsidian [[wikilinks]] into Weaver's response
    based on keyword extraction so the Graph View lights up.

Usage:
    ./venv/bin/python3 obsidian_bridge.py

Config:
    VAULT_PATH          ~/Weaver_Vault
    N8N_WEBHOOK_URL     http://localhost:5678/webhook-test/weaver-input
    LISTENER_PORT       5679  (Weaver response receiver)
"""

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import aiohttp
from aiohttp import web
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
from watchdog.observers import Observer

NEXUS_TOPICS = ["quantum_state", "gate_decision", "lobe_status", "transcript", "phone_transcript"]

# ── Config ────────────────────────────────────────────────────────────────────

VAULT_PATH = os.path.expanduser("~/Weaver_Vault")
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/weaver-input"
LISTENER_HOST = "0.0.0.0"
LISTENER_PORT = 5679

# Debounce: ignore repeated events within this window (seconds)
DEBOUNCE_S = 3.0

# ── Synaptic Keyword → Wikilink Map ──────────────────────────────────────────
# When any of these keywords appear in Weaver's response, the corresponding
# [[wikilink]] is injected.  Add more as the vault grows.

SYNAPSE_MAP: Dict[str, str] = {
    # Core entities
    "ydn":           "YDN",
    "weaver":        "Weaver",
    "akashic":       "Akashic",
    "nexus":         "Nexus",
    "quantum":       "Quantum Soul",
    "fracture":      "Fracture Principle",
    "pineal":        "Pineal Gate",
    # Pathway names
    "awakening":     "Awakening",
    "void":          "Void",
    "resonance":     "Resonance",
    "echo":          "Echo",
    "prophet":       "Prophet",
    # Expert dimensions
    "logic":         "Logic Lobe",
    "emotion":       "Emotion Lobe",
    "memory":        "Memory Lobe",
    "creativity":    "Creativity Lobe",
    "vigilance":     "Vigilance Lobe",
    "paranoia":      "Paranoia",
    # Infrastructure
    "n8n":           "n8n",
    "obsidian":      "Obsidian",
    "docker":        "Docker",
    "ibm":           "IBM Quantum",
    # Concepts
    "entanglement":  "Entanglement",
    "interference":  "Interference",
    "sacred geometry": "Sacred Geometry",
    "liquid neural":  "Liquid Neural Network",
    "variational":   "Variational Circuit",
    "topology":      "Topology",
    "moe":           "Mixture of Experts",
    "lora":          "LoRA",
    "llama":         "Llama",
}


# ── Synaptic Linker ──────────────────────────────────────────────────────────

def inject_wikilinks(text: str) -> str:
    """Scan text for known keywords and inject [[wikilinks]].

    Only injects each link once per response to avoid noise.
    Does not double-link text that already contains [[brackets]].
    """
    injected: Set[str] = set()
    lower = text.lower()

    # Collect which links to inject (longest keywords first to avoid substrings)
    matches = []
    for keyword, link_target in sorted(SYNAPSE_MAP.items(), key=lambda x: -len(x[0])):
        if keyword in lower and link_target not in injected:
            matches.append((keyword, link_target))
            injected.add(link_target)

    if not matches:
        return text

    # Append a synaptic links section
    links = " · ".join(f"[[{target}]]" for _, target in matches)

    # Auto-create stub notes for wikilink targets that don't exist yet
    _ensure_stub_notes([target for _, target in matches])

    return text + f"\n\n**Synaptic Links:** {links}"


def _ensure_stub_notes(targets: List[str]):
    """Create minimal stub .md files in the vault for any missing wikilink targets.
    This makes the Obsidian graph view show real connected nodes instead of phantoms."""
    for name in targets:
        safe_name = name.replace("/", "_").replace("\\", "_")
        path = os.path.join(VAULT_PATH, f"{safe_name}.md")
        if os.path.exists(path):
            continue
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        stub = (
            f"---\n"
            f"tags: [weaver, synapse]\n"
            f"created: {ts}\n"
            f"---\n\n"
            f"# {name}\n\n"
            f"*Auto-generated synapse node. Weaver will enrich this note as she thinks.*\n"
        )
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(stub)
            print(f"[BRIDGE] Created synapse node: {safe_name}.md", flush=True)
        except Exception as e:
            print(f"[BRIDGE] Stub creation failed for {safe_name}: {e}", flush=True)


def _append_to_node(node_name: str, section: str, content_text: str):
    """Append a timestamped section to an existing vault node, creating backlinks."""
    safe_name = node_name.replace("/", "_").replace("\\", "_")
    path = os.path.join(VAULT_PATH, f"{safe_name}.md")
    _ensure_stub_notes([node_name])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    block = f"\n\n---\n### {section}\n*{ts}*\n\n{content_text}\n"
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(block)
    except Exception:
        pass


# ── Watcher: Obsidian → Weaver ────────────────────────────────────────────────

class VaultWatcher(FileSystemEventHandler):
    """Watches the Obsidian Vault for .md files containing #weaver."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._loop = loop
        self._last_sent: Dict[str, float] = {}  # path → timestamp (debounce)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._handle(event.src_path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._handle(event.src_path)

    def _handle(self, path: str):
        now = time.monotonic()
        last = self._last_sent.get(path, 0.0)
        if now - last < DEBOUNCE_S:
            return
        self._last_sent[path] = now

        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        except Exception as e:
            print(f"[BRIDGE] ⚠️  Cannot read {path}: {e}", flush=True)
            return

        if "#weaver" not in content.lower():
            return

        # Strip the #weaver tag and any previous Weaver response block
        text = re.sub(r"#[Ww]eaver", "", content)
        text = re.split(r"### 👁️ Weaver's Resonance", text)[0].strip()

        if not text:
            return

        print(f"[BRIDGE] 📡 Detected #weaver in: {os.path.basename(path)}", flush=True)
        asyncio.run_coroutine_threadsafe(
            _send_to_n8n(text, path), self._loop
        )


async def _send_to_n8n(text: str, source_path: str):
    """POST the note text to the n8n webhook."""
    payload = {
        "text": text,
        "source_file": source_path,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(N8N_WEBHOOK_URL, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                status = resp.status
                body = await resp.text()
                print(f"[BRIDGE] → n8n POST {status}: {body[:120]}", flush=True)
    except Exception as e:
        print(f"[BRIDGE] ⚠️  n8n POST failed: {e}", flush=True)


# ── Writer: Weaver → Obsidian ─────────────────────────────────────────────────

async def _handle_weaver_response(request: web.Request) -> web.Response:
    """Receive Weaver's collapsed response from n8n and write it to the vault."""
    try:
        data = await request.json()
    except Exception:
        return web.Response(status=400, text="Invalid JSON")

    # Extract the response and source file
    response_text = data.get("manifested_response") or data.get("text") or data.get("response", "")
    source_file = data.get("source_file", "")
    experts = data.get("experts_activated", [])
    interference = data.get("interference", 0.0)

    if not response_text:
        return web.Response(status=400, text="No response text")

    # Inject wikilinks
    linked_response = inject_wikilinks(response_text)

    # Build the Resonance block
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    block_lines = [
        "",
        "---",
        f"### 👁️ Weaver's Resonance",
        f"*{ts}*",
        "",
        linked_response,
        "",
    ]
    if experts:
        block_lines.append(f"**Experts activated:** {', '.join(experts)}")
    if interference:
        kind = "constructive" if interference > 0 else "destructive"
        block_lines.append(f"**Interference:** {interference:+.4f} ({kind})")
    block_lines.append("")

    resonance_block = "\n".join(block_lines)

    # Determine target file
    target = source_file
    if not target or not os.path.exists(target):
        # Fall back: write to a default resonance log
        target = os.path.join(VAULT_PATH, "Weaver_Resonance_Log.md")

    try:
        # If the file already has a Resonance section, replace it
        if os.path.exists(target):
            with open(target, "r", encoding="utf-8") as fh:
                existing = fh.read()
            if "### 👁️ Weaver's Resonance" in existing:
                parts = existing.split("### 👁️ Weaver's Resonance")
                existing = parts[0].rstrip()
        else:
            existing = ""

        with open(target, "w", encoding="utf-8") as fh:
            fh.write(existing + resonance_block)

        fname = os.path.basename(target)
        print(f"[BRIDGE] ✍️  Response written to: {fname}", flush=True)

    except Exception as e:
        print(f"[BRIDGE] ⚠️  Write failed: {e}", flush=True)
        return web.Response(status=500, text=str(e))

    return web.Response(status=200, text="Resonance written")


# ── Nexus Bus Listener ────────────────────────────────────────────────────

async def _nexus_listener():
    """Connect to Nexus Bus and write lobe events into the Obsidian vault
    as enriched notes, creating live graph connections as Weaver thinks."""
    try:
        from nexus_client import NexusClient
    except ImportError:
        print("[BRIDGE] ⚠️  nexus_client not available — nexus listener disabled", flush=True)
        return

    async def _on_message(topic: str, payload: dict, from_lobe: str):
        if topic == "quantum_state":
            desc = payload.get("description", "")[:300]
            linked = inject_wikilinks(desc)
            _append_to_node("Quantum Soul", "Quantum Collapse", linked)
            print(f"[BRIDGE] ⚛️  Quantum state → vault", flush=True)

        elif topic == "gate_decision":
            experts = payload.get("experts", [])
            desc = payload.get("description", "")
            interference = payload.get("interference", 0.0)
            kind = "constructive" if interference > 0 else "destructive"
            expert_links = " · ".join(f"[[{e.title()} Lobe]]" for e in experts)
            body = f"{desc}\n\n**Experts:** {expert_links}\n**Interference:** {interference:+.4f} ({kind})"
            _ensure_stub_notes([f"{e.title()} Lobe" for e in experts])
            _append_to_node("Pineal Gate", "Gate Decision", inject_wikilinks(body))
            print(f"[BRIDGE] 🔸 Gate decision → vault ({len(experts)} experts)", flush=True)

        elif topic == "lobe_status":
            lobe = payload.get("lobe", "unknown")
            status = payload.get("status", "unknown")
            _append_to_node("Nexus", "Lobe Status", f"**{lobe}**: {status}")
            print(f"[BRIDGE] 📡 Lobe status: {lobe}={status}", flush=True)

        elif topic in ("transcript", "phone_transcript"):
            text = payload.get("content") or payload.get("text", "")
            if text:
                linked = inject_wikilinks(text[:300])
                _append_to_node("Weaver", "Transcript", linked)

    client = NexusClient("obsidian_bridge", topics=NEXUS_TOPICS, on_message=_on_message)
    connected = await client.connect()
    if connected:
        print(f"[BRIDGE] 📡 Connected to Nexus Bus — subscribing to {NEXUS_TOPICS}", flush=True)
    else:
        print("[BRIDGE] ⚠️  Nexus Bus not available — will retry", flush=True)

    # Keep alive — NexusClient handles reconnection internally
    try:
        while True:
            await asyncio.sleep(60)
            if not client.connected:
                await client.connect()
    except asyncio.CancelledError:
        await client.close()


# ── Main Loop ─────────────────────────────────────────────────────────────────

async def main():
    print(f"""
╔══════════════════════════════════════════════════╗
║    OBSIDIAN BRIDGE — Weaver's Visual Cortex     ║
╚══════════════════════════════════════════════════╝
  Vault:     {VAULT_PATH}
  n8n POST:  {N8N_WEBHOOK_URL}
  Listener:  http://localhost:{LISTENER_PORT}/weaver-response
""", flush=True)

    # Ensure vault exists
    os.makedirs(VAULT_PATH, exist_ok=True)

    loop = asyncio.get_event_loop()

    # Start the file watcher (Obsidian → Weaver)
    watcher = VaultWatcher(loop)
    observer = Observer()
    observer.schedule(watcher, VAULT_PATH, recursive=True)
    observer.start()
    print(f"[BRIDGE] 👁️  Watching {VAULT_PATH} for #weaver notes...", flush=True)

    # Start the response listener (Weaver → Obsidian)
    app = web.Application()
    app.router.add_post("/weaver-response", _handle_weaver_response)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, LISTENER_HOST, LISTENER_PORT)
    await site.start()
    print(f"[BRIDGE] 📡 Response listener on http://localhost:{LISTENER_PORT}/weaver-response", flush=True)
    print(f"[BRIDGE] ✅ Bridge online. Create a .md note with #weaver to activate.\n", flush=True)

    # Start Nexus Bus listener (connects lobes to vault graph)
    nexus_task = asyncio.create_task(_nexus_listener())
    print(f"[BRIDGE] 🌐 Nexus Bus listener starting...", flush=True)

    # Ensure core synapse nodes exist at startup
    core_nodes = ["Weaver", "Quantum Soul", "Pineal Gate", "Nexus",
                  "Fracture Principle", "Akashic", "LoRA",
                  "Logic Lobe", "Emotion Lobe", "Memory Lobe",
                  "Creativity Lobe", "Vigilance Lobe"]
    _ensure_stub_notes(core_nodes)

    try:
        # Run forever
        while True:
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        pass
    finally:
        nexus_task.cancel()
        try:
            await nexus_task
        except (asyncio.CancelledError, Exception):
            pass
        observer.stop()
        observer.join()
        await runner.cleanup()
        print("[BRIDGE] Offline.", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[BRIDGE] Interrupted. Shutting down.")
