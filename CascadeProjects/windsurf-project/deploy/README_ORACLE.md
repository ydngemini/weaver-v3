# Deploying Weaver (headless) to Oracle Cloud — Always Free ARM

**Goal:** run the Weaver backend brain 24/7 for **$0/month**, forever.

**What runs:** Nexus Bus, Akashic Hub, Quantum Soul, Pineal Gate, Liquid Fracture,
5 Experts, Soul Voice (LoRA), Obsidian Bridge, optional n8n.
**What does NOT run:** VTV voice/vision (needs a physical mic+camera — impossible on a
headless cloud box). That's why we deploy `--headless`.

**Why Oracle:** the A1 "Always Free" tier gives **4 ARM cores + 24 GB RAM, free forever** —
the only free tier big enough for this stack (measured ~4 GB; 24 GB leaves room for the 3B later).

---

## 0. Provision the box (one time, ~10 min)
1. Sign up at cloud.oracle.com. **Upgrade to Pay-As-You-Go** (still free under limits; this
   stops Oracle from reclaiming idle "Always Free" instances and unlocks A1 capacity).
2. Compute → Create Instance:
   - Shape → **Ampere → VM.Standard.A1.Flex**, set **4 OCPU / 24 GB**.
   - Image → **Ubuntu 24.04 (aarch64)**.
   - Save the SSH private key.
   - If you hit **"Out of host capacity"**, retry later or pick a quieter home region — this is
     the #1 A1 annoyance. (A retry script/loop usually wins within a day.)
3. Boot volume 100 GB is fine (free up to 200 GB total).

**Security:** leave the default ingress **closed** (only SSH/22). Everything Weaver exposes is
localhost-only; reach dashboards via SSH tunnel (step 5). Don't open 8899/9990/9996 publicly —
`lora_server` binds `0.0.0.0`, so an open port = an exposed model endpoint.

---

## 1. Build the Soul Voice GGUF — **on your local x86 box**
The ARM box can't run bitsandbytes, so we serve the LoRA as a merged GGUF instead.
Bake it locally (where transformers + llama.cpp already work) and upload the ~0.8 GB file:

```bash
cd "weaver v3/CascadeProjects/windsurf-project"
./deploy/build_soul_gguf.sh          # → weaver_merged_1B_Q4_K_M.gguf
```

## 2. Get the code + model onto the box
```bash
# from local:
rsync -avz --exclude venv --exclude '*.gguf' --exclude Nexus_Vault \
    "weaver v3/CascadeProjects" ubuntu@<ORACLE_IP>:~/weaver/
scp weaver_merged_1B_Q4_K_M.gguf \
    ubuntu@<ORACLE_IP>:~/weaver/CascadeProjects/windsurf-project/
```

## 3. Configure + install — **on the Oracle box**
```bash
cd ~/weaver/CascadeProjects/windsurf-project
cp deploy/env.oracle.example .env     # then edit: GEMINI_API_KEY, IBM_QUANTUM_TOKEN
bash deploy/setup_oracle.sh           # venv + ARM-safe deps + llama-cpp-python(OpenBLAS)
```

## 4. Run it
```bash
# foreground smoke test (wait ~50s for all lobes):
./start_weaver.sh --headless

# make it persistent + auto-restart:
sudo cp deploy/weaver.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now weaver
journalctl -u weaver -f                # watch boot
```

## 5. Reach the dashboards (no open ports)
```bash
ssh -L 9996:localhost:9996 -L 9990:localhost:9990 -L 9997:localhost:9997 ubuntu@<ORACLE_IP>
# then browse http://localhost:9996  (health)  /  9990 (live)  /  9997 (quantum)
```

---

## Cost = $0 — how
| Piece | Free path | Set by |
|---|---|---|
| Compute | Oracle A1 Always Free (24 GB) | provisioning |
| 5 Experts | Gemini free tier (or local llama.cpp) | `WEAVER_LLM_BACKEND=gemini`\|`local` |
| Soul Voice | merged LoRA GGUF via llama.cpp | `WEAVER_LORA_BACKEND=gguf` |
| Quantum | IBM Quantum free open plan | `IBM_QUANTUM_TOKEN` |
| n8n | self-hosted OSS (optional) | `npm i -g n8n` |

Gemini free tier has ~15 req/min limits — fine for personal use (5 expert calls/request).
For zero quota, set `WEAVER_LLM_BACKEND=local` and run a small GGUF with
`llama-server -m smollm2-360m...gguf --port 8090` (trades API quota for CPU time).

## Optional: n8n nervous system
```bash
sudo apt install -y nodejs npm && sudo npm i -g n8n
n8n start &                            # http://localhost:5678
# import n8n_weaver_v5.json via the n8n UI; weaver.py posts to its webhook.
```
Weaver degrades gracefully if n8n is absent.

## Optional: add the 3B later
Build a Q4_K_M GGUF of the 3B the same way (it's a Qwen2 fine-tune), set its path, and
raise `MemoryMax` in weaver.service. 24 GB has the headroom.

## Troubleshooting
- **llama-cpp-python build fails** → ensure `build-essential cmake libopenblas-dev` installed (setup does this).
- **A package in requirements fails on ARM** → it's likely a VTV/audio dep; comment it out, headless doesn't need it.
- **Experts 400 on Gemini** → the code already sends `max_tokens` (not `max_completion_tokens`) for non-Azure backends; if a model name is rejected, set `WEAVER_LLM_MODEL=gemini-1.5-flash`.
- **Health checks show "not responding" right after boot** → expected; `start_weaver.sh` probes at +12s but the stack needs ~50s. Use `journalctl -u weaver -f`.
