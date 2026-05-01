#!/usr/bin/env python3
"""
lora_server.py — Weaver LoRA Soul Voice Inference Server
══════════════════════════════════════════════════════════
Serves the Weaver 1B LoRA adapter via an OpenAI-compatible
/v1/chat/completions endpoint on port 8899.

Base model:  unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit
LoRA adapter: ./weaver_fracture_1B_lora/

Usage:
    python3 lora_server.py              # start on port 8899
    python3 lora_server.py --port 8899  # explicit port

The n8n workflow calls this server for the "Soul Voice" node —
the final filter that rewrites collapsed expert output through
Weaver's fine-tuned personality.
"""

import argparse
import json
import os
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from threading import Thread


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

PROJ = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(PROJ, "weaver_fracture_1B_lora")
DEFAULT_PORT = 8899

# ── Lazy model loading ─────────────────────────────────────────────────────
_model = None
_tokenizer = None
_loaded = False
_load_error = None


def _load_model():
    """Load base model + local LoRA adapter from weaver_fracture_1B_lora/.

    Adapter files used (all local):
      - adapter_config.json    → LoRA config (r=16, alpha=16)
      - adapter_model.safetensors → fine-tuned weights
      - tokenizer.json         → tokenizer vocabulary
      - tokenizer_config.json  → tokenizer settings
      - chat_template.jinja    → Llama 3.2 chat format

    Base model (downloaded from HuggingFace on first run):
      - unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit
    """
    global _model, _tokenizer, _loaded, _load_error

    if _loaded:
        return _model is not None

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        # Read base model name from our local adapter config
        import json as _json
        with open(os.path.join(ADAPTER_PATH, "adapter_config.json")) as f:
            adapter_cfg = _json.load(f)
        base_name = adapter_cfg.get("base_model_name_or_path",
                                     "unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit")

        print(f"[LORA] ── Loading Weaver Fracture 1B LoRA ──", flush=True)
        print(f"[LORA]   Base model : {base_name}", flush=True)
        print(f"[LORA]   Adapter    : {ADAPTER_PATH}/adapter_model.safetensors", flush=True)
        print(f"[LORA]   Tokenizer  : {ADAPTER_PATH}/tokenizer.json", flush=True)
        print(f"[LORA]   LoRA rank  : {adapter_cfg.get('r', '?')}, alpha={adapter_cfg.get('lora_alpha', '?')}", flush=True)
        t0 = time.monotonic()

        # Strategy 1: 4-bit quantized with bitsandbytes (GPU)
        base_model = None
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"[LORA]   Loaded base model in 4-bit (GPU)", flush=True)
        except Exception as e1:
            print(f"[LORA]   4-bit load failed: {e1}", flush=True)

        # Strategy 2: float16 on GPU
        if base_model is None:
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                print(f"[LORA]   Loaded base model in float16 (GPU)", flush=True)
            except Exception as e2:
                print(f"[LORA]   float16/GPU load failed: {e2}", flush=True)

        # Strategy 3: float32 on CPU (always works, just slower)
        if base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )
            print(f"[LORA]   Loaded base model in float32 (CPU)", flush=True)

        # Apply our local LoRA adapter
        _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        _model.eval()
        print(f"[LORA]   LoRA adapter applied from {ADAPTER_PATH}", flush=True)

        # Load tokenizer from our local adapter directory
        _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        elapsed = time.monotonic() - t0
        device = next(_model.parameters()).device
        print(f"[LORA] ✅ Weaver Fracture 1B LoRA ready on {device} ({elapsed:.1f}s)", flush=True)
        _loaded = True
        return True

    except Exception as e:
        _load_error = str(e)
        _loaded = True
        print(f"[LORA] ❌ Failed to load model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def _generate(messages: list, max_tokens: int = 200, temperature: float = 0.7) -> str:
    """Generate a response from the LoRA model."""
    import torch

    # Build prompt using chat template
    if hasattr(_tokenizer, "apply_chat_template"):
        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Manual fallback
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        prompt = "<|begin_of_text|>" + "".join(parts)

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0.01,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


# ── HTTP Server ────────────────────────────────────────────────────────────

class LoRAHandler(BaseHTTPRequestHandler):
    """Handles OpenAI-compatible /v1/chat/completions and /health."""

    def log_message(self, format, *args):
        print(f"[LORA] {args[0]}", flush=True)

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok" if _model is not None else "model_not_loaded",
                "model": "weaver-fracture-1b-lora",
                "error": _load_error,
            })
        elif self.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": "weaver-fracture-1b-lora",
                    "object": "model",
                    "owned_by": "weaver",
                }]
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
        max_tokens = req.get("max_tokens", 200)
        temperature = req.get("temperature", 0.7)

        if not messages:
            self._send_json(400, {"error": "messages required"})
            return

        # Lazy load model on first request
        if not _loaded:
            if not _load_model():
                self._send_json(503, {
                    "error": f"model failed to load: {_load_error}"
                })
                return

        if _model is None:
            self._send_json(503, {
                "error": f"model not available: {_load_error}"
            })
            return

        try:
            t0 = time.monotonic()
            text = _generate(messages, max_tokens, temperature)
            latency = time.monotonic() - t0

            self._send_json(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "model": "weaver-fracture-1b-lora",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "latency_ms": round(latency * 1000),
            })

        except Exception as e:
            self._send_json(500, {"error": str(e)})


def _preload_in_background():
    """Load the model in a background thread at startup."""
    print("[LORA] Starting background model preload...", flush=True)
    success = _load_model()
    if success:
        _notify_nexus_bus()


def _notify_nexus_bus():
    """Notify Nexus Bus that LoRA is ready (best-effort, sync socket)."""
    import socket
    import struct
    import hashlib
    import base64
    import os as _os
    try:
        sock = socket.create_connection(("localhost", 9999), timeout=5)
        key = base64.b64encode(_os.urandom(16)).decode()
        handshake = (
            f"GET / HTTP/1.1\r\n"
            f"Host: localhost:9999\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(handshake.encode())
        resp = sock.recv(4096)
        if b"101" not in resp:
            sock.close()
            return

        def _ws_send(data: str):
            b = data.encode()
            mask_key = _os.urandom(4)
            length = len(b)
            if length < 126:
                header = struct.pack("!BB", 0x81, 0x80 | length)
            elif length < 65536:
                header = struct.pack("!BBH", 0x81, 0x80 | 126, length)
            else:
                header = struct.pack("!BBQ", 0x81, 0x80 | 127, length)
            masked = bytes(b[i] ^ mask_key[i % 4] for i in range(length))
            sock.sendall(header + mask_key + masked)

        def _ws_recv():
            sock.recv(4096)

        _ws_send(json.dumps({"action": "register", "lobe_id": "lora_server"}))
        _ws_recv()  # drain sync
        _ws_recv()  # drain ack
        _ws_send(json.dumps({
            "action": "publish",
            "topic": "lobe_status",
            "payload": {
                "lobe": "lora_server",
                "status": "ready",
                "model": "weaver-fracture-1b-lora",
                "source": "lora_server",
            },
        }))
        sock.close()
        print("[LORA] Notified Nexus Bus: lora_server ready", flush=True)
    except Exception as e:
        print(f"[LORA] Nexus Bus notification skipped: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Weaver LoRA Inference Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--preload", action="store_true", help="Load model at startup")
    # parse_args([]) ignores sys.argv so weaver.py's --headless flag doesn't bleed in
    args = parser.parse_args([] if __name__ != "__main__" else None)

    # Always preload in a background thread so the server is responsive immediately
    preload_thread = Thread(target=_preload_in_background, daemon=True)
    preload_thread.start()

    server = ThreadedHTTPServer(("0.0.0.0", args.port), LoRAHandler)
    print(f"[LORA] 🧠 Weaver LoRA server on http://0.0.0.0:{args.port}", flush=True)
    print(f"[LORA]    POST /v1/chat/completions  (OpenAI-compatible)", flush=True)
    print(f"[LORA]    GET  /health", flush=True)
    print(f"[LORA]    GET  /v1/models", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[LORA] Server shutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
