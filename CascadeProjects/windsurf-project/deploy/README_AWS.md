# Deploying Weaver to AWS — EC2 Graviton (ARM), full public stack

This is the **AWS** counterpart to `README_ORACLE.md`. Same stack, same systemd units — the
only differences are **provisioning** (Terraform → EC2 instead of the OCI console) and
**networking** (an AWS Security Group replaces Oracle's VCN + on-box `iptables`).

**What runs:** Nexus Bus, Akashic Hub, Quantum Soul, Pineal Gate, Liquid Fracture, 5 Experts
(local llama.cpp), Soul Voice (LoRA GGUF), Obsidian Bridge, the Oracle command-center
frontend + backend, and Caddy public TLS. Deployed `--headless` (no VTV mic/camera on a
cloud box).

> **"Oracle" is two things here.** This guide swaps **Oracle _Cloud_ (OCI)** → **AWS**. The
> **Oracle _app_** (the real-estate frontend) still deploys on the box exactly as before.

## Cost reality — AWS is **not** free (Oracle A1 was)

There is no permanent AWS free tier big enough for this. Realistic monthly cost, us-east-1,
on-demand:

| Piece | ~Cost/mo |
|---|---|
| `t4g.large` (2 vCPU / 8 GB, ARM) compute | ~$49 |
| Public IPv4 (Elastic IP, charged even when attached since Feb 2024) | ~$3.60 |
| 50 GB gp3 EBS | ~$4 |
| **Total** | **~$56/mo** |

Levers: a **1-yr Compute Savings Plan** cuts compute ~30-40%. Drop to `t4g.medium` (4 GB) +
`WEAVER_LLM_BACKEND=gemini` for ~$28/mo if you don't need on-box experts.

---

## 0. Provision the box with Terraform (one time, ~3 min)

Authored in `deploy/aws-terraform/`. **You run `apply`** — it creates real, billable infra.

```bash
cd deploy/aws-terraform
cp terraform.tfvars.example terraform.tfvars
#   → edit: set ssh_ingress_cidr to "$(curl -s ifconfig.me)/32", confirm public_key_path
terraform init
terraform apply            # review the plan, type "yes"
terraform output           # note: public_ip, ssh_command, and public_urls
```

This creates: 1× `t4g.large` (Ubuntu 24.04 arm64, IMDSv2, encrypted gp3), an Elastic IP, a
key pair, and a Security Group that opens **only 22 (your IP) + 80/443**. Everything Weaver
exposes internally (9999/8899/8090/8091/8000/9996/9997) stays localhost-only — `lora_server`
binds `0.0.0.0`, so an open port would be an exposed model endpoint.

Set shell vars for the rest of this guide:
```bash
export ORACLE_HOST=weaverv3.com
export WEAVER_HEADLESS_HOST=headless.weaverv3.com
export WEAVER_DASH_HOST=dash.weaverv3.com
export WEAVER_STATUS_HOST=status.weaverv3.com
export BOX="ubuntu@$(terraform output -raw public_ip)"
cd -
```

## 1. Build the Soul Voice GGUF — **on your local x86 box**

The ARM box can't run bitsandbytes, so we serve the LoRA as a merged GGUF. Bake it locally
and upload the ~0.8 GB file:

```bash
cd "weaver v3/CascadeProjects/windsurf-project"
./deploy/build_soul_gguf.sh          # → weaver_merged_1B_Q4_K_M.gguf
```

## 2. Get the code + model onto the box

```bash
# from local:
rsync -avz --exclude venv --exclude '*.gguf' --exclude Nexus_Vault \
    "weaver v3/CascadeProjects" "$BOX":~/weaver/
rsync -avz --exclude node_modules --exclude venv --exclude .next \
    "/media/ydn/SYPHER_CORE2/Oracle/" "$BOX":~/oracle/
scp weaver_merged_1B_Q4_K_M.gguf \
    "$BOX":~/weaver/CascadeProjects/windsurf-project/
```

## 3. Configure + install the base stack — **on the box**

```bash
ssh "$BOX"
cd ~/weaver/CascadeProjects/windsurf-project
cp deploy/env.oracle.example .env     # defaults to WEAVER_LLM_BACKEND=local; fill IBM_QUANTUM_TOKEN (optional)
bash deploy/setup_oracle.sh           # venv + ARM-safe deps + llama-cpp-python(OpenBLAS)
```

## 4. Wire the full public stack (experts + Oracle backend/frontend + Caddy)

```bash
ORACLE_HOST=weaverv3.com \
WEAVER_HEADLESS_HOST=headless.weaverv3.com \
WEAVER_DASH_HOST=dash.weaverv3.com \
WEAVER_STATUS_HOST=status.weaverv3.com \
CLOUD=aws bash deploy/setup_oracle_extras.sh
#   → then edit ~/oracle/backend/.env: ORACLE_SECRET_KEY, ORACLE_ENCRYPTION_MASTER_KEY,
#     ORACLE_ADMIN_ID/PASSPHRASE, and keep ORACLE_ENABLE_DEMO_LOGINS=0
sudo systemctl restart oracle-backend
```

`CLOUD=aws` makes the script skip the Oracle-only `iptables` guidance — the Security Group
from step 0 already opened 80/443, and AWS Ubuntu AMIs have no default firewall to poke.

This installs + starts four units: `weaver-llm` (experts :8090), `weaver` (`--headless`),
`oracle-backend` (:8000), and `caddy` (:443).

## 5. Verify

```bash
# on the box — all four green:
journalctl -u weaver-llm -u weaver -u oracle-backend -u caddy -f
curl 127.0.0.1:8090/v1/models          # experts server up
curl 127.0.0.1:8091/health             # read-only codebase API up
curl 127.0.0.1:8000/health             # oracle backend up

# from your laptop:
#   https://weaverv3.com            → embodied avatar
#   https://headless.weaverv3.com   → headless quantum presence
#   https://dash.weaverv3.com       → protected operator dashboard
#   https://status.weaverv3.com     → protected health dashboard
```

**Security:** `9999`/`8899`/`8090`/`8091`/`8000`/`9996`/`9997` stay localhost-only — only `443`/`22`
are reachable. Before going public: set `ORACLE_SECRET_KEY` + `ORACLE_ENCRYPTION_MASTER_KEY`,
an admin login, and keep `ORACLE_ENABLE_DEMO_LOGINS=0`.

`/codebase/*` is proxied through Caddy to `127.0.0.1:8091` and requires the same
`X-Weaver-Key` as `/llm/*`. It is read-only and serves capped, redacted source/doc
snippets only; `.env`, vaults, models, assets, Terraform plans, hidden files, and large
artifacts are excluded. `weaver.service` sets `WEAVER_CODEBASE_ROOT=/home/ubuntu/weaver`
so Weaver can inspect the deployed source tree rather than only the `windsurf-project`
subfolder. Both `weaver.service` and `weaver-brain.service` set
`WEAVER_VAULT_DIR=/home/ubuntu/weaver/CascadeProjects/windsurf-project/Nexus_Vault`
so headless cognition, quantum state, browser memory, and Akashic persistence use one
shared vault across restarts.

---

## Production URLs

The current production DNS records all point at the Elastic IP:

| URL | Purpose |
|---|---|
| `https://weaverv3.com` | embodied avatar UI |
| `https://headless.weaverv3.com` | headless 3D quantum presence |
| `https://dash.weaverv3.com` | protected live operator dashboard |
| `https://status.weaverv3.com` | protected health dashboard |
| `https://weaverv3.com/brain/*` | key-gated Bedrock brain API |
| `wss://weaverv3.com/brain/realtime/voice` | Nova Sonic realtime voice |
| `https://weaverv3.com/tts/*` | key-gated AWS Polly TTS |
| `https://weaverv3.com/codebase/*` | key-gated read-only source context |

The Elastic IP (`terraform output -raw public_ip`) is stable, so these DNS
records should not churn.

## DNS Maintenance

At your DNS host, keep these records pointed at the Elastic IP:

| Type | Name | Value |
|---|---|---|
| A | `@` (the apex, e.g. `weaverv3.com`) | `<Elastic IP>` |
| A | `headless` | `<Elastic IP>` |
| A | `dash` | `<Elastic IP>` |
| A | `status` | `<Elastic IP>` |

Then verify from your laptop. Each command should print the Elastic IP:

```bash
dig +short weaverv3.com
dig +short headless.weaverv3.com
dig +short dash.weaverv3.com
dig +short status.weaverv3.com
```

`www.weaverv3.com` is intentionally not listed unless you add DNS for it and a
redirect block in Caddy.

The Terraform `hostname` output remains useful for a brand-new box before custom
DNS is ready. Use it only as a temporary sslip.io bootstrap host; the canonical
Weaver URLs above are the production paths.

## Reapply Canonical URLs

To rewrite the box env and reload Caddy after a host change:

```bash
ssh "$BOX"
cd ~/weaver/CascadeProjects/windsurf-project
ORACLE_HOST=weaverv3.com \
WEAVER_HEADLESS_HOST=headless.weaverv3.com \
WEAVER_DASH_HOST=dash.weaverv3.com \
WEAVER_STATUS_HOST=status.weaverv3.com \
CLOUD=aws bash deploy/setup_oracle_extras.sh
sudo systemctl reload caddy
```

Verify: `https://weaverv3.com`, `https://headless.weaverv3.com`,
`https://dash.weaverv3.com`, and `https://status.weaverv3.com` all load with
valid certs.

---

## What differs from the Oracle deploy

| | Oracle Cloud | AWS |
|---|---|---|
| Provision | A1 Flex console (out-of-capacity retries) | `terraform apply` (`deploy/aws-terraform/`) |
| Firewall | VCN Security List **+** on-box `iptables` | Security Group only (no iptables) |
| Public IP | ephemeral | Elastic IP (stable) |
| Cost | $0 forever (24 GB) | ~$56/mo (`t4g.large` 8 GB) |
| SSH user | `ubuntu` | `ubuntu` (same) |

Everything else — `setup_oracle.sh`, the three `.service` units, the `Caddyfile`, the GGUF
build — is byte-for-byte identical. See `README_ORACLE.md` for module-level detail,
troubleshooting, local-vs-Gemini experts, and adding the 3B later.

## Teardown

```bash
cd deploy/aws-terraform && terraform destroy   # stops all AWS charges
```
