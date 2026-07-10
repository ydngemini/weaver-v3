#!/usr/bin/env python3
"""
weaver_tools.py — Shared Tool Belt & Dispatcher for Weaver v3
==============================================================
Defines WEAVER_TOOL_BELT (20 universal command blocks) and the
execute_weaver_tool() dispatcher. Used by:
  - twilio_weaver_bridge.py (phone calls)
  - vtv_basic.py (voice/video/text)
  - n8n webhooks
  - any future interface

The tool belt covers: outbound calls, SMS, Obsidian notes, quantum
state, memory recall/save, timers, reminders, to-dos, n8n triggers,
web search, system health, dictation, and sounds.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from memory_manager import MemoryManager, default_vault_dir

# ── Config (env-driven, with sane defaults) ───────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAULT_DIR = str(default_vault_dir())
OBSIDIAN_VAULT = os.path.expanduser("~/Weaver_Vault")
LORA_API_URL = os.environ.get("LORA_API_URL", "http://localhost:8899/v1/chat/completions")
QUANTUM_BIAS_URL = os.environ.get("QUANTUM_BIAS_URL", "http://localhost:9997/quantum/bias")
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/weaver-input")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
PHONE_BRIDGE_PORT = int(os.environ.get("TWILIO_BRIDGE_PORT", "8765"))

_todo_list_path = os.path.join(VAULT_DIR, "weaver_todos.md")
_active_timers: list = []
_memory_manager_ref: MemoryManager | None = None


def _default_memory_manager() -> MemoryManager:
    global _memory_manager_ref
    if _memory_manager_ref is None:
        _memory_manager_ref = MemoryManager(default_vault_dir())
    return _memory_manager_ref

# ══════════════════════════════════════════════════════════════════════════════
# WEAVER TOOL BELT — 20 Universal Command Blocks
# ══════════════════════════════════════════════════════════════════════════════

WEAVER_TOOL_BELT = [
    {"type": "function", "name": "make_outbound_call",
     "description": "Call a phone number. Use for 'call Mom', 'dial 555-1234'.",
     "parameters": {"type": "object", "properties": {
         "to": {"type": "string", "description": "Phone number in E.164 format"},
         "reason": {"type": "string", "description": "Brief reason for the call"}
     }, "required": ["to"]}},

    {"type": "function", "name": "send_sms",
     "description": "Send an SMS text message to a phone number.",
     "parameters": {"type": "object", "properties": {
         "to": {"type": "string", "description": "Phone number in E.164 format"},
         "body": {"type": "string", "description": "Message text"}
     }, "required": ["to", "body"]}},

    {"type": "function", "name": "recall_memory",
     "description": "Search Weaver's persistent memory for people, conversations, or facts.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "What to search for (name, topic, keyword)"}
     }, "required": ["query"]}},

    {"type": "function", "name": "remember_fact",
     "description": "Save a fact, preference, or detail to persistent memory.",
     "parameters": {"type": "object", "properties": {
         "fact": {"type": "string", "description": "The fact to remember"},
         "about": {"type": "string", "description": "Who or what this is about"}
     }, "required": ["fact"]}},

    {"type": "function", "name": "write_obsidian_note",
     "description": "Create or append to an Obsidian vault note.",
     "parameters": {"type": "object", "properties": {
         "title": {"type": "string", "description": "Note title (becomes filename)"},
         "content": {"type": "string", "description": "Markdown content to write"},
         "append": {"type": "boolean", "description": "Append to existing note if true"}
     }, "required": ["title", "content"]}},

    {"type": "function", "name": "search_obsidian",
     "description": "Search Obsidian vault notes by keyword.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Search keyword"}
     }, "required": ["query"]}},

    {"type": "function", "name": "check_quantum_state",
     "description": "Get the current quantum pathway state and routing bias.",
     "parameters": {"type": "object", "properties": {}}},

    {"type": "function", "name": "set_timer",
     "description": "Set a countdown timer or alarm. Use for 'remind me in 5 minutes'.",
     "parameters": {"type": "object", "properties": {
         "duration_seconds": {"type": "integer", "description": "Timer duration in seconds"},
         "label": {"type": "string", "description": "What this timer is for"}
     }, "required": ["duration_seconds"]}},

    {"type": "function", "name": "add_todo",
     "description": "Add an item to the to-do list.",
     "parameters": {"type": "object", "properties": {
         "task": {"type": "string", "description": "The task description"},
         "priority": {"type": "string", "enum": ["high", "medium", "low"], "description": "Priority level"}
     }, "required": ["task"]}},

    {"type": "function", "name": "list_todos",
     "description": "List all current to-do items.",
     "parameters": {"type": "object", "properties": {}}},

    {"type": "function", "name": "complete_todo",
     "description": "Mark a to-do item as complete.",
     "parameters": {"type": "object", "properties": {
         "task_query": {"type": "string", "description": "Part of the task description to match"}
     }, "required": ["task_query"]}},

    {"type": "function", "name": "get_current_time",
     "description": "Get the current date and time.",
     "parameters": {"type": "object", "properties": {}}},

    {"type": "function", "name": "web_search",
     "description": "Search the web for current information.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Search query"}
     }, "required": ["query"]}},

    {"type": "function", "name": "send_to_n8n",
     "description": "Trigger an n8n workflow with custom data.",
     "parameters": {"type": "object", "properties": {
         "workflow": {"type": "string", "description": "Workflow name or webhook path"},
         "data": {"type": "string", "description": "JSON payload to send"}
     }, "required": ["data"]}},

    {"type": "function", "name": "summarize_conversation",
     "description": "Get a summary of the current conversation so far.",
     "parameters": {"type": "object", "properties": {}}},

    {"type": "function", "name": "who_am_i_talking_to",
     "description": "Identify the current user based on face or voice recognition.",
     "parameters": {"type": "object", "properties": {}}},

    {"type": "function", "name": "check_system_health",
     "description": "Check the status of all Weaver lobes and subsystems.",
     "parameters": {"type": "object", "properties": {}}},

    {"type": "function", "name": "set_reminder",
     "description": "Set a reminder for a specific time or relative offset.",
     "parameters": {"type": "object", "properties": {
         "message": {"type": "string", "description": "Reminder message"},
         "when": {"type": "string", "description": "When to remind, e.g. 'in 30 minutes' or '3pm'"}
     }, "required": ["message", "when"]}},

    {"type": "function", "name": "dictate_note",
     "description": "Dictate a note to be saved as text in the vault.",
     "parameters": {"type": "object", "properties": {
         "content": {"type": "string", "description": "The dictated text"},
         "title": {"type": "string", "description": "Optional title for the note"}
     }, "required": ["content"]}},

    {"type": "function", "name": "play_sound",
     "description": "Play a notification or ambient sound.",
     "parameters": {"type": "object", "properties": {
         "sound": {"type": "string", "enum": ["chime", "alert", "success", "error", "ambient"],
                   "description": "Sound to play"}
     }, "required": ["sound"]}},
]


# ══════════════════════════════════════════════════════════════════════════════
# Tenacity-Wrapped API Helpers
# ══════════════════════════════════════════════════════════════════════════════

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError)))
async def api_get(url: str, timeout: float = 5.0) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError)))
async def api_post(url: str, payload: dict, timeout: float = 15.0) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=15),
       retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)))
async def lora_rewrite(text: str) -> str:
    """Pipe text through the LoRA Soul Voice for personality filtering."""
    resp = await api_post(LORA_API_URL, {
        "model": "weaver-fracture-1b-lora",
        "messages": [{"role": "user", "content": text[:500]}],
        "max_tokens": 150,
    }, timeout=45.0)
    return resp.get("choices", [{}])[0].get("message", {}).get("content", "")


# ══════════════════════════════════════════════════════════════════════════════
# Nexus Bus Helper (lazy-init singleton)
# ══════════════════════════════════════════════════════════════════════════════

_nexus_client_ref = [None]


async def publish_to_nexus(topic: str, data: dict):
    """Publish to Nexus Bus with lazy connection."""
    try:
        if _nexus_client_ref[0] is None:
            try:
                from nexus_client import NexusClient
                _nexus_client_ref[0] = NexusClient("weaver_tools", topics=[])
                await _nexus_client_ref[0].connect()
            except Exception:
                pass
        if _nexus_client_ref[0]:
            await _nexus_client_ref[0].publish(topic, data)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Todo helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_todos() -> list:
    if not os.path.exists(_todo_list_path):
        return []
    lines = open(_todo_list_path, "r", encoding="utf-8").readlines()
    todos = []
    for line in lines:
        line = line.strip()
        if line.startswith("- ["):
            done = line.startswith("- [x]")
            text = line[6:].strip() if done else line[5:].strip()
            todos.append({"task": text, "done": done})
    return todos


def _save_todos(todos: list):
    os.makedirs(os.path.dirname(_todo_list_path), exist_ok=True)
    with open(_todo_list_path, "w", encoding="utf-8") as f:
        f.write("# Weaver To-Do List\n\n")
        for t in todos:
            check = "x" if t["done"] else " "
            f.write(f"- [{check}] {t['task']}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Tool Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

async def execute_weaver_tool(name: str, args: dict,
                              session_state: Optional[Dict[str, Any]] = None,
                              memory_manager=None) -> str:
    """Dispatch a tool call and return the result string.

    Args:
        name: Tool function name from WEAVER_TOOL_BELT.
        args: Parsed arguments dict.
        session_state: Live session context (conversation_history, identified user, etc).
        memory_manager: Optional MemoryManager instance for recall/remember.
    """
    if session_state is None:
        session_state = {}

    try:
        if name == "make_outbound_call":
            to = args.get("to", "")
            if not to:
                return "Error: no phone number provided."
            try:
                result = await api_post(
                    f"http://localhost:{PHONE_BRIDGE_PORT}/call",
                    {"to": to, "reason": args.get("reason", "")},
                )
                return f"Call initiated to {to}. SID: {result.get('call_sid', 'unknown')}"
            except Exception as e:
                return f"Call failed: {e}"

        elif name == "send_sms":
            to, body = args.get("to", ""), args.get("body", "")
            if not to or not body:
                return "Error: need 'to' and 'body'."
            try:
                from twilio.rest import Client as _TwClient
                client = _TwClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                msg = client.messages.create(
                    to=to, from_=os.environ.get("TWILIO_PHONE_NUMBER", ""), body=body
                )
                return f"SMS sent to {to}. SID: {msg.sid}"
            except Exception as e:
                return f"SMS failed: {e}"

        elif name == "recall_memory":
            query = args.get("query", "")
            memory_manager = memory_manager or _default_memory_manager()
            if memory_manager:
                result = await memory_manager.recall(query)
                parts = []
                if result["people"]:
                    parts.append(f"People: {result['people'][:500]}")
                if result["conversations"]:
                    parts.append(f"Conversations: {result['conversations'][:500]}")
                return "\n".join(parts) if parts else "No matching memories found."
            return "Memory manager not available."

        elif name == "remember_fact":
            fact = args.get("fact", "")
            about = args.get("about", "general")
            memory_manager = memory_manager or _default_memory_manager()
            if memory_manager:
                await memory_manager.remember({
                    "type": "conversation", "speaker": "system",
                    "content": f"[REMEMBERED] {about}: {fact}", "source": "phone",
                })
            return f"Remembered: {fact}"

        elif name == "write_obsidian_note":
            title = args.get("title", "Untitled")
            content = args.get("content", "")
            append = args.get("append", False)
            safe_title = title.replace("/", "_").replace("\\", "_")
            note_path = os.path.join(OBSIDIAN_VAULT, f"{safe_title}.md")
            os.makedirs(OBSIDIAN_VAULT, exist_ok=True)
            mode = "a" if append and os.path.exists(note_path) else "w"
            with open(note_path, mode, encoding="utf-8") as f:
                if mode == "w":
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                    f.write(f"---\ntags: [weaver, tool]\ncreated: {ts}\n---\n\n# {title}\n\n")
                f.write(content + "\n")
            return f"Note '{title}' written to Obsidian vault."

        elif name == "search_obsidian":
            query = args.get("query", "").lower()
            matches = []
            if os.path.isdir(OBSIDIAN_VAULT):
                for fname in os.listdir(OBSIDIAN_VAULT):
                    if fname.endswith(".md"):
                        fpath = os.path.join(OBSIDIAN_VAULT, fname)
                        text = open(fpath, "r", encoding="utf-8").read()
                        if query in text.lower():
                            matches.append(fname)
            return f"Found {len(matches)} notes: {', '.join(matches[:10])}" if matches else "No matching notes."

        elif name == "check_quantum_state":
            try:
                bias = await api_get(QUANTUM_BIAS_URL, timeout=3.0)
                dom = bias.get("dominant", "unknown")
                weights = bias.get("weights", {})
                w_str = ", ".join(f"{k}={v:.2f}" for k, v in weights.items())
                return f"Quantum pathway: {dom}. Weights: {w_str}"
            except Exception:
                return "Quantum API unavailable."

        elif name == "set_timer":
            secs = args.get("duration_seconds", 60)
            label = args.get("label", "Timer")

            async def _timer_fire():
                await asyncio.sleep(secs)
                await publish_to_nexus("timer_fired", {"label": label, "duration": secs})

            asyncio.create_task(_timer_fire())
            _active_timers.append({"label": label, "seconds": secs, "set_at": datetime.now().isoformat()})
            return f"Timer set: {label} ({secs}s)"

        elif name == "add_todo":
            task = args.get("task", "")
            priority = args.get("priority", "medium")
            todos = _load_todos()
            todos.append({"task": f"[{priority.upper()}] {task}", "done": False})
            _save_todos(todos)
            return f"Added: {task} ({priority})"

        elif name == "list_todos":
            todos = _load_todos()
            if not todos:
                return "To-do list is empty."
            lines = []
            for i, t in enumerate(todos, 1):
                status = "done" if t["done"] else "pending"
                lines.append(f"{i}. [{status}] {t['task']}")
            return "\n".join(lines)

        elif name == "complete_todo":
            query = args.get("task_query", "").lower()
            todos = _load_todos()
            for t in todos:
                if query in t["task"].lower():
                    t["done"] = True
                    _save_todos(todos)
                    return f"Completed: {t['task']}"
            return f"No matching todo found for '{query}'."

        elif name == "get_current_time":
            return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

        elif name == "web_search":
            query = args.get("query", "")
            try:
                result = await api_post(N8N_WEBHOOK_URL, {
                    "text": f"[SEARCH] {query}", "source": "tool", "tool": "web_search",
                })
                return result.get("response", result.get("text", "Search submitted to n8n."))
            except Exception:
                return "Web search unavailable — n8n not connected."

        elif name == "send_to_n8n":
            data_str = args.get("data", "{}")
            workflow = args.get("workflow", "weaver-input")
            try:
                payload = json.loads(data_str) if isinstance(data_str, str) else data_str
            except json.JSONDecodeError:
                payload = {"text": data_str}
            payload["source"] = "tool"
            url = N8N_WEBHOOK_URL.replace("weaver-input", workflow)
            try:
                result = await api_post(url, payload)
                return f"n8n workflow triggered: {json.dumps(result)[:200]}"
            except Exception as e:
                return f"n8n trigger failed: {e}"

        elif name == "summarize_conversation":
            summary = session_state.get("lc_summary", "")
            if summary:
                return f"Conversation so far: {summary}"
            history = session_state.get("conversation_history", [])
            if history:
                last_5 = history[-5:]
                return "Recent exchanges:\n" + "\n".join(
                    f"{h['role']}: {h['content'][:100]}" for h in last_5
                )
            return "No conversation history yet."

        elif name == "who_am_i_talking_to":
            user_id = session_state.get("identified_caller") or session_state.get("identified_user", "unknown")
            return f"Current user identified as: {user_id}"

        elif name == "check_system_health":
            try:
                health = await api_get("http://localhost:9996/api/status", timeout=3.0)
                return f"System health: {json.dumps(health)[:300]}"
            except Exception:
                return "Health dashboard unavailable."

        elif name == "set_reminder":
            message = args.get("message", "Reminder")
            when = args.get("when", "in 5 minutes")
            secs = 300
            m = re.search(r"(\d+)", when)
            if m:
                num = int(m.group(1))
                if "hour" in when:
                    secs = num * 3600
                elif "second" in when:
                    secs = num
                else:
                    secs = num * 60

            async def _reminder_fire():
                await asyncio.sleep(secs)
                await publish_to_nexus("reminder_fired", {"message": message})

            asyncio.create_task(_reminder_fire())
            return f"Reminder set: '{message}' in {secs}s"

        elif name == "dictate_note":
            content = args.get("content", "")
            title = args.get("title", f"Dictation_{datetime.now().strftime('%Y%m%d_%H%M')}")
            safe_title = title.replace("/", "_").replace("\\", "_")
            note_path = os.path.join(OBSIDIAN_VAULT, f"{safe_title}.md")
            os.makedirs(OBSIDIAN_VAULT, exist_ok=True)
            with open(note_path, "w", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                f.write(f"---\ntags: [weaver, dictation]\ncreated: {ts}\n---\n\n# {title}\n\n{content}\n")
            return f"Dictation saved as '{title}'."

        elif name == "play_sound":
            sound = args.get("sound", "chime")
            await publish_to_nexus("play_sound", {"sound": sound})
            return f"Playing {sound} sound."

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Tool error ({name}): {e}"
