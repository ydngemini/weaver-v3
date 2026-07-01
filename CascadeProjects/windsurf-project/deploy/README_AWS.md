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
terraform output           # note: hostname (<EIP>.sslip.io), public_ip, ssh_command
```

This creates: 1× `t4g.large` (Ubuntu 24.04 arm64, IMDSv2, encrypted gp3), an Elastic IP, a
key pair, and a Security Group that opens **only 22 (your IP) + 80/443**. Everything Weaver
exposes internally (9999/8899/8090/8000/9996/9997) stays localhost-only — `lora_server`
binds `0.0.0.0`, so an open port would be an exposed model endpoint.

Set a shell var for the rest of this guide:
```bash
export ORACLE_HOST="$(terraform output -raw hostname)"   # e.g. 203.0.113.4.sslip.io
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
export ORACLE_HOST=<EIP>.sslip.io          # the terraform output from step 0
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
curl 127.0.0.1:8000/health             # oracle backend up

# from your laptop:
#   https://<EIP>.sslip.io   → Oracle UI loads, valid Let's Encrypt cert, wss://.../ws connects
```

**Security:** `9999`/`8899`/`8090`/`8000`/`9996`/`9997` stay localhost-only — only `443`/`22`
are reachable. Before going public: set `ORACLE_SECRET_KEY` + `ORACLE_ENCRYPTION_MASTER_KEY`,
an admin login, and keep `ORACLE_ENABLE_DEMO_LOGINS=0`.

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
