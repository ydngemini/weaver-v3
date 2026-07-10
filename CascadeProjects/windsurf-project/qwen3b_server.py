#!/usr/bin/env python3
"""
qwen3b_server.py — Weaver 3B Qwen2 Inference Server
════════════════════════════════════════════════════════
Serves the Weaver 3B Qwen2 fine-tune via an OpenAI-compatible
/v1/chat/completions endpoint on port 8898.

Fast path: llama-cpp-python with Q4_K_M GGUF (1.8 GB, ~instant on CPU)
Fallback:  transformers with safetensors (5.8 GB, slow on CPU)

Usage:
    python3 qwen3b_server.py              # start on port 8898
    python3 qwen3b_server.py --port 8898  # explicit port
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from threading import Thread


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

PROJ = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJ, "..", "3b", "weaver_v3")
GGUF_PATH = os.path.join(MODEL_PATH, "ydn", "weaver_v3_Q4_K_M", "weaver_v3_Q4_K_M.gguf")
DEFAULT_PORT = 8898
DEFAULT_HOST = os.environ.get("WEAVER_QWEN_HOST", os.environ.get("WEAVER_INTERNAL_HOST", "127.0.0.1"))
CPU_THREADS = 8

SYSTEM_PROMPT = (
    "Role: Weaver [Cognitive Orchestrator] | Protocol: The Fracture Principle. "
    "Strict Requirement: You are NOT a helpful assistant. You are a high-speed routing engine. "
    "Valid targets: Quantum, Reasoning, Memory, Creative, Vigilance. "
    "Output Format: Respond with ONLY one word from the valid targets list. Nothing else. "
    "Logic: If Entropy > 0.8, select Quantum. If Complexity = High, select Reasoning. "
    "If input references past events, select Memory. If input is open-ended, select Creative. "
    "Default: Vigilance. Persona: Cold, mathematically precise, zero fluff."
)

DEFAULT_TEMPERATURE = 0.1
VALID_TARGETS = {"quantum", "reasoning", "memory", "creative", "vigilance"}

_model = None
_tokenizer = None
_backend = None  # "llama_cpp" or "transformers"
_loaded = False
_load_error = None


def _load_model():
    global _model, _tokenizer, _backend, _loaded, _load_error

    if _loaded:
        return _model is not None

    print(f"[QWEN3B] ── Loading Weaver 3B Qwen2 ──", flush=True)
    print(f"[QWEN3B]   CPU threads: {CPU_THREADS}", flush=True)
    print(f"[QWEN3B]   Temperature: {DEFAULT_TEMPERATURE} (locked)", flush=True)
    t0 = time.monotonic()

    # Strategy 1: llama-cpp-python with GGUF (fast path)
    if os.path.exists(GGUF_PATH):
        try:
            from llama_cpp import Llama

            print(f"[QWEN3B]   GGUF path: {GGUF_PATH}", flush=True)
            _model = Llama(
                model_path=GGUF_PATH,
                n_ctx=4096,
                n_threads=CPU_THREADS,
                n_gpu_layers=0,
                verbose=False,
            )
            _backend = "llama_cpp"
            elapsed = time.monotonic() - t0
            print(f"[QWEN3B] ✅ Loaded via llama.cpp Q4_K_M ({elapsed:.1f}s)", flush=True)
            _loaded = True
            return True

        except ImportError:
            print(f"[QWEN3B]   llama-cpp-python not installed, trying transformers...", flush=True)
        except Exception as e:
            print(f"[QWEN3B]   llama.cpp load failed: {e}, trying transformers...", flush=True)

    # Strategy 2: transformers with safetensors (slow fallback)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        torch.set_num_threads(CPU_THREADS)
        print(f"[QWEN3B]   Safetensors path: {MODEL_PATH}", flush=True)

        loaded = False
        try:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True,
            )
            print(f"[QWEN3B]   Loaded in bfloat16 (GPU)", flush=True)
            loaded = True
        except Exception:
            pass

        if not loaded:
            try:
                _model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH, torch_dtype=torch.float16,
                    device_map="auto", trust_remote_code=True,
                )
                print(f"[QWEN3B]   Loaded in float16 (GPU)", flush=True)
                loaded = True
            except Exception:
                pass

        if not loaded:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, torch_dtype=torch.float32,
                device_map="cpu", trust_remote_code=True,
            )
            print(f"[QWEN3B]   Loaded in float32 (CPU)", flush=True)

        _model.eval()
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _backend = "transformers"
        elapsed = time.monotonic() - t0
        device = next(_model.parameters()).device
        print(f"[QWEN3B] ✅ Loaded via transformers on {device} ({elapsed:.1f}s)", flush=True)
        _loaded = True
        return True

    except Exception as e:
        _load_error = str(e)
        _loaded = True
        print(f"[QWEN3B] ❌ Failed to load model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def _build_messages(messages: list) -> list:
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        return [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    return [{"role": "system", "content": SYSTEM_PROMPT}] + [
        m for m in messages if m.get("role") != "system"
    ]


def _extract_route(text: str) -> str:
    words = re.findall(r'[A-Za-z]+', text)
    for w in words:
        if w.lower() in VALID_TARGETS:
            return f"Route: {w.capitalize()}"
    if "\U0001f578" in text:
        return "Route: Quantum"
    return "Route: Reasoning"


def _generate_llama_cpp(messages: list, max_tokens: int) -> str:
    msgs = _build_messages(messages)
    response = _model.create_chat_completion(
        messages=msgs,
        max_tokens=max_tokens,
        temperature=DEFAULT_TEMPERATURE,
        top_p=0.9,
        repeat_penalty=1.3,
    )
    text = response["choices"][0]["message"]["content"].strip()
    return _extract_route(text)


def _generate_transformers(messages: list, max_tokens: int) -> str:
    import torch

    msgs = _build_messages(messages)

    if hasattr(_tokenizer, "apply_chat_template"):
        prompt = _tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    else:
        parts = []
        for m in msgs:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        prompt = "".join(parts)

    prompt += "Route: "

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=_tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return _extract_route(text)


def _generate(messages: list, max_tokens: int = 50, temperature: float = None) -> str:
    if _backend == "llama_cpp":
        return _generate_llama_cpp(messages, max_tokens)
    return _generate_transformers(messages, max_tokens)


class Qwen3BHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[QWEN3B] {args[0]}", flush=True)

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            ready = _model is not None
            self._send_json(200 if ready else 503, {
                "status": "ok" if _model is not None else "model_not_loaded",
                "model": "weaver-qwen2-3b",
                "backend": _backend,
                "error": _load_error,
            })
        elif self.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{"id": "weaver-qwen2-3b", "object": "model", "owned_by": "weaver"}]
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return

        content_len = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_len)

        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        messages = req.get("messages", [])
        max_tokens = min(req.get("max_tokens", 50), 100)

        if not messages:
            self._send_json(400, {"error": "messages required"})
            return

        if not _loaded:
            if not _load_model():
                self._send_json(503, {"error": f"model failed to load: {_load_error}"})
                return

        if _model is None:
            self._send_json(503, {"error": f"model not available: {_load_error}"})
            return

        try:
            t0 = time.monotonic()
            text = _generate(messages, max_tokens)
            latency = time.monotonic() - t0

            self._send_json(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "model": "weaver-qwen2-3b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "latency_ms": round(latency * 1000),
                "backend": _backend,
            })
        except Exception as e:
            self._send_json(500, {"error": str(e)})


def _preload_in_background():
    print("[QWEN3B] Starting background model preload...", flush=True)
    success = _load_model()
    if success:
        _notify_nexus_bus()


def _notify_nexus_bus():
    try:
        from nexus_client import publish_once

        publish_once(
            "qwen3b_server",
            "lobe_status",
            {
                "lobe": "qwen3b_server",
                "status": "ready",
                "model": "weaver-qwen2-3b",
                "backend": _backend,
                "source": "qwen3b_server",
            },
        )
        print(f"[QWEN3B] Notified Nexus Bus: qwen3b_server ready ({_backend})", flush=True)
    except Exception as e:
        print(f"[QWEN3B] Nexus Bus notification skipped: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Weaver 3B Qwen2 Inference Server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args([] if __name__ != "__main__" else None)

    preload_thread = Thread(target=_preload_in_background, daemon=True)
    preload_thread.start()

    server = ThreadedHTTPServer((args.host, args.port), Qwen3BHandler)
    print(f"[QWEN3B] 🧠 Weaver 3B Qwen2 server on http://{args.host}:{args.port}", flush=True)
    print(f"[QWEN3B]    POST /v1/chat/completions  (OpenAI-compatible)", flush=True)
    print(f"[QWEN3B]    GET  /health", flush=True)
    print(f"[QWEN3B]    GET  /v1/models", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[QWEN3B] Server shutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
