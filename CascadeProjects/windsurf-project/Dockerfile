# ═══════════════════════════════════════════════════════════════
# Weaver v3 — Multi-Stage Dockerfile
# ═══════════════════════════════════════════════════════════════
#
# Build:
#   docker build -t weaver:latest .
#
# Run (headless — backend lobes only):
#   docker run -d --name weaver \
#     --env-file .env \
#     -p 8765:8765 -p 9996:9996 -p 9997:9997 -p 9998:9998 -p 9999:9999 \
#     -v weaver_vault:/app/Nexus_Vault \
#     -v weaver_obsidian:/root/Weaver_Vault \
#     weaver:latest
#
# Full stack (with mic/camera — requires host devices):
#   docker run -d --name weaver \
#     --env-file .env \
#     --device /dev/snd --device /dev/video0 \
#     -p 8765:8765 -p 9996:9996 -p 9997:9997 -p 9998:9998 -p 9999:9999 \
#     -v weaver_vault:/app/Nexus_Vault \
#     weaver:latest --heartbeat

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements-core.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-core.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="ydngemini"
LABEL description="Weaver v3 — Async Multi-Modal AI Hive-Mind"
LABEL version="3.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    alsa-utils curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY *.py ./
COPY weaver_fracture_1B_lora/ ./weaver_fracture_1B_lora/

# Create persistent directories
RUN mkdir -p /app/Nexus_Vault /root/Weaver_Vault

# Expose all service ports
EXPOSE 8765 8899 9995 9996 9997 9998 9999

# Health checks for critical lobes
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:9996/health && \
        curl -sf http://localhost:8765/health || exit 1

# Default: headless mode (no mic/camera)
ENTRYPOINT ["python3", "weaver.py"]
CMD ["--headless"]
