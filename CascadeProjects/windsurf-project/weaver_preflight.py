#!/usr/bin/env python3
"""
weaver_preflight.py — Startup Config Validator
================================================
Validates environment, ports, venv, and directories before launch.
Called by: make preflight, weaver.py startup
"""

import os
import socket
import sys

PROJ = os.path.dirname(os.path.abspath(__file__))

REQUIRED_ENV = [
    ("WEAVER_VOICE_KEY", "OpenAI Realtime + voice"),
    ("WEAVER_MEM_KEY", "Memory/summarization calls"),
]

OPTIONAL_ENV = [
    ("WEAVER_VISION_KEY", "Vision model (gpt-4o)"),
    ("GEMINI_API_KEY", "Gemini vision"),
    ("IBM_QUANTUM_TOKEN", "IBM Quantum hardware"),
    ("TWILIO_ACCOUNT_SID", "Twilio telephony"),
    ("TWILIO_AUTH_TOKEN", "Twilio telephony"),
    ("TWILIO_PHONE_NUMBER", "Twilio telephony"),
    ("NATE_PHONE_NUMBER", "Proactive outbound calls"),
]

PORTS = [
    (9999, "Nexus Bus"),
    (9995, "Akashic Hub API"),
    (9998, "Nexus Bus Health"),
    (9997, "Quantum API"),
    (9996, "Health Dashboard"),
    (8899, "LoRA Server"),
    (8765, "Phone Bridge"),
    (5678, "n8n Workflow"),
]

VAULT_DIRS = [
    "Nexus_Vault",
    "Nexus_Vault/akashic_persist",
]

CRITICAL_FILES = [
    "weaver.py",
    "nexus_bus.py",
    "akashic_hub.py",
    "health_dashboard.py",
]


def _check_port(port: int) -> bool:
    """Return True if port is free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def preflight_check(verbose: bool = True) -> dict:
    """Run all preflight checks. Returns summary dict."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJ, ".env"))

    results = {"errors": [], "warnings": [], "ok": []}

    # ── venv ──
    venv_path = os.path.join(PROJ, "venv")
    if os.path.isdir(venv_path):
        results["ok"].append("venv exists")
    else:
        results["errors"].append("venv/ not found — run: python3 -m venv venv && venv/bin/pip install -r requirements-core.txt")

    # ── .env file ──
    env_path = os.path.join(PROJ, ".env")
    if os.path.isfile(env_path):
        results["ok"].append(".env file exists")
    else:
        results["errors"].append(".env not found — run: cp .env.example .env")

    # ── Required env vars ──
    for var, desc in REQUIRED_ENV:
        val = os.environ.get(var, "")
        if val and not val.startswith("sk-..."):
            results["ok"].append(f"{var} set")
        else:
            results["errors"].append(f"{var} missing or placeholder ({desc})")

    # ── Optional env vars ──
    for var, desc in OPTIONAL_ENV:
        val = os.environ.get(var, "")
        if val and "..." not in val:
            results["ok"].append(f"{var} set")
        else:
            results["warnings"].append(f"{var} not configured ({desc})")

    # ── Ports ──
    for port, svc in PORTS:
        if _check_port(port):
            results["ok"].append(f"Port {port} ({svc}) available")
        else:
            results["warnings"].append(f"Port {port} ({svc}) already in use")

    # ── Vault directories ──
    for d in VAULT_DIRS:
        full = os.path.join(PROJ, d)
        if os.path.isdir(full):
            results["ok"].append(f"{d}/ exists")
        else:
            try:
                os.makedirs(full, exist_ok=True)
                results["ok"].append(f"{d}/ created")
            except OSError as e:
                results["errors"].append(f"Cannot create {d}/: {e}")

    # ── Critical files ──
    for f in CRITICAL_FILES:
        if os.path.isfile(os.path.join(PROJ, f)):
            results["ok"].append(f"{f} present")
        else:
            results["errors"].append(f"{f} missing from project directory")

    # ── LoRA adapter ──
    lora_dir = os.path.join(PROJ, "weaver_fracture_1B_lora")
    if os.path.isdir(lora_dir):
        adapter_cfg = os.path.join(lora_dir, "adapter_config.json")
        if os.path.isfile(adapter_cfg):
            results["ok"].append("LoRA adapter ready")
        else:
            results["warnings"].append("LoRA adapter dir exists but missing adapter_config.json")
    else:
        results["warnings"].append("LoRA adapter not found (weaver_fracture_1B_lora/)")

    # ── Print report ──
    if verbose:
        n_err = len(results["errors"])
        n_warn = len(results["warnings"])
        n_ok = len(results["ok"])
        total = n_err + n_warn + n_ok

        print(f"\n{'=' * 56}")
        print(f"  Weaver v3 Preflight Check — {total} checks")
        print(f"{'=' * 56}")

        for msg in results["ok"]:
            print(f"  \033[32m✓\033[0m {msg}")
        for msg in results["warnings"]:
            print(f"  \033[33m⚠\033[0m {msg}")
        for msg in results["errors"]:
            print(f"  \033[31m✗\033[0m {msg}")

        print(f"\n  \033[32m{n_ok} passed\033[0m  \033[33m{n_warn} warnings\033[0m  \033[31m{n_err} errors\033[0m")

        if n_err > 0:
            print(f"\n  \033[31m✗ Preflight FAILED — fix {n_err} error(s) above.\033[0m\n")
        else:
            print(f"\n  \033[32m✓ Preflight PASSED — system ready to launch.\033[0m\n")

    return results


if __name__ == "__main__":
    results = preflight_check(verbose=True)
    sys.exit(1 if results["errors"] else 0)
