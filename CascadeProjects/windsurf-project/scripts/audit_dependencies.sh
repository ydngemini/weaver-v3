#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$(dirname "$HERE")"
PYTHON="${1:-$APP/venv/bin/python}"

"$PYTHON" -m pip check
"$PYTHON" -m pip_audit --progress-spinner off \
  --ignore-vuln PYSEC-2026-2447 \
  --ignore-vuln CVE-2025-3000

# PYSEC-2026-2447: diskcache has no fixed release. It is a transitive
# llama-cpp dependency, but Weaver never enables llama_cpp.server's disk cache.
# The production unit also combines ProtectSystem=strict, ProtectHome=read-only,
# PrivateTmp=true, and UMask=0077, so another service/user cannot plant a cache.
#
# CVE-2025-3000: the OSV affected range is PyTorch 2.6.0 only. Weaver pins
# 2.11.0; pip-audit currently reports the GIT-only advisory against later wheels.
