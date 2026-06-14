#!/usr/bin/env bash
# ===========================================================================
# run_kaggle.sh — CLI launcher/loop for the HEADLESS Kaggle MoE GPT pretrain.
#
# Corrected, code-as-DATASET flow. Three Kaggle facts shape this:
#   1. FORCE A T4. Kaggle's torch is 2.10.0+cu128 (sm_70/75/80/86/90, NOT sm_60).
#      A P100 is sm_60 and fails "no kernel image". The train kernel-metadata sets
#      "machine_shape":"NvidiaTeslaT4" AND we push with --accelerator NvidiaTeslaT4.
#   2. A SCRIPT kernel runs ONLY its code_file; sibling .py are not importable. So
#      the project code (model/prepare_data/train) ships as a DATASET
#      (weaver-moe-code) and the kernels sys.path.insert it.
#   3. Two kernels: a CPU PREP kernel packs train.bin/val.bin -> weaver-moe-data;
#      a GPU TRAIN kernel resumes from weaver-moe-ckpt and writes a new ckpt.
#
# SAFETY MODEL — DRY-RUN BY DEFAULT:
#   * Local-only steps (copying source into the code dataset dir, syntax) RUN.
#   * Every `kaggle` command that spends quota / touches your account is GUARDED:
#       - default                         -> PRINTED only (nothing executed).
#       - RUN=1 ./run_kaggle.sh <cmd>  OR  ./run_kaggle.sh <cmd> --run  -> executed.
#
# SUBCOMMANDS:
#   bootstrap-code   sync source into kaggle/code_dataset, `datasets create` it
#   prep             push the CPU prep kernel (--accelerator none); after it
#                    completes, `kernels output` + `datasets create` -> weaver-moe-data
#   bootstrap-ckpt   create an empty/placeholder weaver-moe-ckpt dataset
#   train            push the GPU train kernel with --accelerator NvidiaTeslaT4
#   status           poll a kernel run status  (--kernel prep|train, default train)
#   output           download a kernel version output (--kernel prep|train)
#   version-ckpt     `datasets version` weaver-moe-ckpt from the train output
#   resume-cycle     document the push -> status -> output -> version -> repeat loop
#   build            local-only: sync source into kaggle/code_dataset (no kaggle calls)
#   help             this help
#
# ENV OVERRIDES:
#   KAGGLE_BIN   path to the kaggle binary (default: $HOME/.venvs/kaggle/bin/kaggle)
#   PYTHON_BIN   python used for local syntax checks (default: python3)
#   RUN=1        actually execute the guarded kaggle commands (else dry-run/print)
# ===========================================================================
set -euo pipefail

# --- resolve this script's dir (handles the SPACE in the project path) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAGGLE_DIR="${SCRIPT_DIR}/kaggle"

# --- new code-as-dataset layout ---
CODE_DATASET_DIR="${KAGGLE_DIR}/code_dataset"     # source bundle -> weaver-moe-code
PREP_DIR="${KAGGLE_DIR}/prep"                      # CPU prep kernel
TRAIN_DIR="${KAGGLE_DIR}/train"                    # GPU train kernel

# --- binaries as ARRAYS so a path with spaces / a custom interpreter is safe ---
KAGGLE_BIN_DEFAULT="${HOME}/.venvs/kaggle/bin/kaggle"
KAGGLE_BIN=("${KAGGLE_BIN:-$KAGGLE_BIN_DEFAULT}")
PYTHON_BIN=("${PYTHON_BIN:-python3}")

# --- identifiers (must match the *-metadata.json files) ---
CODE_DATASET="nathaniellockwood/weaver-moe-code"
DATA_DATASET="nathaniellockwood/weaver-moe-data"
CKPT_DATASET="nathaniellockwood/weaver-moe-ckpt"
PREP_KERNEL="nathaniellockwood/weaver-moe-prep"
TRAIN_KERNEL="nathaniellockwood/weaver-moe-train"

# --- output staging dirs ---
PREP_OUTPUT_DIR="${SCRIPT_DIR}/kaggle_prep_output"     # prep kernel output -> data dataset
TRAIN_OUTPUT_DIR="${SCRIPT_DIR}/kaggle_train_output"   # train kernel output -> ckpt version
DATA_STAGE="${SCRIPT_DIR}/kaggle_data_dataset"         # staged weaver-moe-data dataset dir
CKPT_STAGE="${SCRIPT_DIR}/kaggle_ckpt_dataset"         # staged weaver-moe-ckpt dataset dir

# --- RUN gate: RUN=1 env OR a trailing --run flag flips dry-run -> execute ---
RUN="${RUN:-0}"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
log()  { printf '%s\n' "[run_kaggle] $*"; }
hr()   { printf '%s\n' "----------------------------------------------------------------------"; }

# guarded(): print a kaggle command; execute it ONLY if RUN=1. Quoting preserved.
guarded() {
  hr
  log "kaggle command:"
  printf '    '
  printf '%q ' "$@"
  printf '\n'
  hr
  if [[ "${RUN}" == "1" ]]; then
    log "RUN=1 -> EXECUTING (this spends Kaggle quota / touches your account)."
    "$@"
  else
    log "DRY-RUN (default): not executed. Set RUN=1 (or pass --run) to run it."
  fi
}

# pull a trailing --run flag (and capture --kernel <name>) out of positional args.
KERNEL_SEL="train"
strip_flags() {
  local out=()
  local a
  local want_kernel=0
  for a in "$@"; do
    if [[ "${want_kernel}" == "1" ]]; then
      KERNEL_SEL="$a"; want_kernel=0; continue
    fi
    case "$a" in
      --run)     RUN=1 ;;
      --kernel)  want_kernel=1 ;;
      --kernel=*) KERNEL_SEL="${a#--kernel=}" ;;
      *)         out+=("$a") ;;
    esac
  done
  STRIPPED_ARGS=("${out[@]:-}")
}

# ---------------------------------------------------------------------------
# build: sync source into kaggle/code_dataset (LOCAL ONLY — no kaggle calls)
# ---------------------------------------------------------------------------
cmd_build() {
  log "syncing source into the code dataset bundle: ${CODE_DATASET_DIR}"
  mkdir -p "${CODE_DATASET_DIR}"
  local f
  for f in model.py prepare_data.py train.py; do
    if [[ ! -f "${SCRIPT_DIR}/${f}" ]]; then
      log "ERROR: missing source ${SCRIPT_DIR}/${f}"; exit 1
    fi
    cp -f "${SCRIPT_DIR}/${f}" "${CODE_DATASET_DIR}/${f}"
    log "copied ${f} -> code_dataset/${f}"
  done
  # optional architecture config, if present alongside the kit
  if [[ -f "${SCRIPT_DIR}/winner_config.json" ]]; then
    cp -f "${SCRIPT_DIR}/winner_config.json" "${CODE_DATASET_DIR}/winner_config.json"
    log "copied winner_config.json -> code_dataset/winner_config.json"
  fi
  if [[ ! -f "${CODE_DATASET_DIR}/dataset-metadata.json" ]]; then
    log "ERROR: ${CODE_DATASET_DIR}/dataset-metadata.json missing."; exit 1
  fi
  # local syntax sanity for the whole bundle + both kernels (no kaggle calls)
  "${PYTHON_BIN[@]}" -m py_compile \
    "${CODE_DATASET_DIR}/model.py" \
    "${CODE_DATASET_DIR}/prepare_data.py" \
    "${CODE_DATASET_DIR}/train.py" \
    "${PREP_DIR}/kaggle_prep.py" \
    "${TRAIN_DIR}/kaggle_train.py"
  log "py_compile OK for the code bundle + both kernels."
  log "code_dataset contents:"
  ls -la "${CODE_DATASET_DIR}"
}

# ---------------------------------------------------------------------------
# bootstrap-code: sync source then `datasets create` the weaver-moe-code dataset
# ---------------------------------------------------------------------------
cmd_bootstrap_code() {
  cmd_build
  hr
  log "ONE-TIME (or re-run after code changes -> use 'version-code' style update):"
  log "create the read-only CODE dataset '${CODE_DATASET}' from the bundle so the"
  log "kernels can sys.path it (Kaggle script-kernels can't import sibling .py)."
  guarded "${KAGGLE_BIN[@]}" datasets create -p "${CODE_DATASET_DIR}"
  hr
  log "NOTE: if the code dataset ALREADY exists and you changed source, push a new"
  log "version instead:"
  log "    RUN=1 ./run_kaggle.sh build   # re-sync source"
  printf '    '
  printf '%q ' "${KAGGLE_BIN[@]}" datasets version -p "${CODE_DATASET_DIR}" -m "code update"
  printf '\n'
}

# ---------------------------------------------------------------------------
# prep: push the CPU prep kernel; on completion pull output + create data dataset
# ---------------------------------------------------------------------------
cmd_prep() {
  log "PREP (CPU kernel): packs train.bin/val.bin via prepare_data.py --corpus kaggle."
  log "Prereq: weaver-moe-code dataset must EXIST (bootstrap-code) — it's the only"
  log "dataset_source of the prep kernel."
  if [[ ! -f "${PREP_DIR}/kernel-metadata.json" ]]; then
    log "ERROR: ${PREP_DIR}/kernel-metadata.json missing."; exit 1
  fi
  hr
  log "1) push the CPU prep kernel (no GPU accelerator):"
  guarded "${KAGGLE_BIN[@]}" kernels push -p "${PREP_DIR}"
  hr
  log "2) poll until it completes:"
  printf '    '
  printf '%q ' "${KAGGLE_BIN[@]}" kernels status "${PREP_KERNEL}"
  printf '\n'
  hr
  log "3) when COMPLETE, pull its output (train.bin/val.bin/meta.json/tokenizer):"
  mkdir -p "${PREP_OUTPUT_DIR}"
  guarded "${KAGGLE_BIN[@]}" kernels output "${PREP_KERNEL}" -p "${PREP_OUTPUT_DIR}"
  hr
  log "4) stage the packed data + a dataset-metadata.json, then CREATE weaver-moe-data."
  log "   stage layout expected in ${DATA_STAGE}:"
  log "       dataset-metadata.json  (id: ${DATA_DATASET})"
  log "       train.bin  val.bin  meta.json  tokenizer/  cursor.json"
  log "   NB: cursor.json is REQUIRED for cross-session resume — the prep kernel copies"
  log "       it back into WORKING next session to continue toward the 6B budget. The"
  log "       'cp -rf <output>/*' below already includes it. After session 1 use"
  log "       'datasets version' (not create) and add weaver-moe-data to the prep"
  log "       kernel-metadata dataset_sources so the NEXT prep run mounts the partial."
  log "   (copy the pulled files in by hand — shown for reference, NOT executed):"
  log "       mkdir -p \"${DATA_STAGE}\""
  log "       cp -rf \"${PREP_OUTPUT_DIR}/\"* \"${DATA_STAGE}/\""
  log "       # write ${DATA_STAGE}/dataset-metadata.json with id ${DATA_DATASET}"
  log "   (or init a template with: ${KAGGLE_BIN[*]} datasets init -p \"${DATA_STAGE}\")"
  hr
  log "5) create the weaver-moe-data dataset version 1 from the staged bins:"
  guarded "${KAGGLE_BIN[@]}" datasets create -p "${DATA_STAGE}" --dir-mode zip
}

# ---------------------------------------------------------------------------
# bootstrap-ckpt: create an empty/placeholder weaver-moe-ckpt dataset
# ---------------------------------------------------------------------------
cmd_bootstrap_ckpt() {
  log "ONE-TIME: create the read-only CHECKPOINT dataset '${CKPT_DATASET}'."
  log "It starts EMPTY (a placeholder file) — the first train run is a FRESH"
  log "random-init that writes the real first checkpoint; you version it afterward."
  log "Stage layout expected in ${CKPT_STAGE}:"
  log "    dataset-metadata.json   (id: ${CKPT_DATASET})"
  log "    README.txt              (placeholder so the dataset is non-empty)"
  hr
  log "1) init a dataset-metadata.json template (then edit id/title):"
  guarded "${KAGGLE_BIN[@]}" datasets init -p "${CKPT_STAGE}"
  log "   then set \"id\":\"${CKPT_DATASET}\" and a \"title\", and:"
  log "       echo 'placeholder' > \"${CKPT_STAGE}/README.txt\""
  hr
  log "2) create the dataset version 1 (empty/placeholder):"
  guarded "${KAGGLE_BIN[@]}" datasets create -p "${CKPT_STAGE}"
  hr
  log "DATASET-SOURCE ORDERING: the TRAIN kernel-metadata lists all three datasets"
  log "(code, data, ckpt) in dataset_sources — they must ALL EXIST before the first"
  log "train push. So the order is: bootstrap-code -> prep (-> create data) ->"
  log "bootstrap-ckpt -> train."
}

# ---------------------------------------------------------------------------
# train: push the GPU train kernel, FORCING a T4 via --accelerator NvidiaTeslaT4
# ---------------------------------------------------------------------------
cmd_train() {
  log "TRAIN (GPU kernel): resumes from weaver-moe-ckpt, reads weaver-moe-data,"
  log "writes a new atomic checkpoint to /kaggle/working/ckpts under the 11.5h guard."
  if [[ ! -f "${TRAIN_DIR}/kernel-metadata.json" ]]; then
    log "ERROR: ${TRAIN_DIR}/kernel-metadata.json missing."; exit 1
  fi
  log "Prereqs: weaver-moe-code, weaver-moe-data, weaver-moe-ckpt ALL exist."
  hr
  log "push the GPU train kernel — FORCE a T4 (P100 sm_60 fails on cu128):"
  guarded "${KAGGLE_BIN[@]}" kernels push -p "${TRAIN_DIR}" --accelerator NvidiaTeslaT4
}

# ---------------------------------------------------------------------------
# status: poll a kernel run status   (--kernel prep|train, default train)
# ---------------------------------------------------------------------------
cmd_status() {
  local kid="${TRAIN_KERNEL}"
  [[ "${KERNEL_SEL}" == "prep" ]] && kid="${PREP_KERNEL}"
  log "poll the ${KERNEL_SEL} kernel run status (queued / running / complete / error):"
  guarded "${KAGGLE_BIN[@]}" kernels status "${kid}"
}

# ---------------------------------------------------------------------------
# output: download a kernel version output   (--kernel prep|train, default train)
# ---------------------------------------------------------------------------
cmd_output() {
  local kid="${TRAIN_KERNEL}"
  local odir="${TRAIN_OUTPUT_DIR}"
  if [[ "${KERNEL_SEL}" == "prep" ]]; then
    kid="${PREP_KERNEL}"; odir="${PREP_OUTPUT_DIR}"
  fi
  mkdir -p "${odir}"
  log "download the ${KERNEL_SEL} kernel VERSION output to:"
  log "    ${odir}"
  guarded "${KAGGLE_BIN[@]}" kernels output "${kid}" -p "${odir}"
}

# ---------------------------------------------------------------------------
# version-ckpt: push the train output as a NEW version of weaver-moe-ckpt
# ---------------------------------------------------------------------------
cmd_version_ckpt() {
  log "stage the pulled TRAIN checkpoint into the ckpt dataset dir and version it,"
  log "so the NEXT train push resumes exactly from it."
  log "Expected: ${CKPT_STAGE}/dataset-metadata.json present (id ${CKPT_DATASET}),"
  log "with the pulled ckpts/*.pt copied in from ${TRAIN_OUTPUT_DIR}/ckpts/."
  hr
  log "(local prep you run by hand, shown for reference — NOT executed here):"
  log "    cp -f \"${TRAIN_OUTPUT_DIR}/ckpts/\"*.pt \"${CKPT_STAGE}/\""
  hr
  log "push a NEW VERSION of the checkpoint dataset:"
  guarded "${KAGGLE_BIN[@]}" datasets version -p "${CKPT_STAGE}" \
    -m "resume checkpoint $(date -u +%Y-%m-%dT%H:%M:%SZ)" --dir-mode zip
}

# ---------------------------------------------------------------------------
# resume-cycle: document the full loop (prints; executes nothing)
# ---------------------------------------------------------------------------
cmd_resume_cycle() {
  cat <<EOF
======================================================================
 FULL KAGGLE PRETRAIN LOOP  (code-as-dataset, forced T4)
======================================================================
 ONE-TIME BOOTSTRAP (in this order — train needs all 3 datasets to exist):

   1. ship the project code as a dataset (kernels can't import sibling .py):
        RUN=1 ./run_kaggle.sh bootstrap-code

   2. pack the data ONCE on a CPU kernel, then make the data dataset:
        RUN=1 ./run_kaggle.sh prep
        # poll:   RUN=1 ./run_kaggle.sh status --kernel prep
        # pull:   RUN=1 ./run_kaggle.sh output --kernel prep
        # stage ${PREP_OUTPUT_DIR} -> ${DATA_STAGE} (+ dataset-metadata.json),
        # then the create step inside 'prep' makes ${DATA_DATASET}.

   3. create an empty checkpoint dataset (first train run is fresh init):
        RUN=1 ./run_kaggle.sh bootstrap-ckpt

 THEN repeat this GPU cycle until the token budget is met (~weeks on free quota):

   A. push the GPU train kernel — FORCED T4 (P100 sm_60 fails on cu128):
        RUN=1 ./run_kaggle.sh train

   B. poll until the run finishes (complete / error):
        RUN=1 ./run_kaggle.sh status            # default --kernel train

   C. pull the train kernel version output (fresh /kaggle/working/ckpts bundle):
        RUN=1 ./run_kaggle.sh output            # default --kernel train

   D. stage the pulled checkpoint, then version weaver-moe-ckpt from it:
        cp -f "${TRAIN_OUTPUT_DIR}/ckpts/"*.pt "${CKPT_STAGE}/"
        RUN=1 ./run_kaggle.sh version-ckpt

   E. GOTO A. The next train push mounts the new ckpt version at
      /kaggle/input/weaver-moe-ckpt and train.py resumes at the exact step.

 Honest budget: ~12h/session (hard kill; guard stops cleanly at ~11.5h),
 ~30h free GPU/week -> ~2-3 sessions/week. STOP-ANYTIME: every session ends
 with a clean resumable checkpoint, so pausing for days costs nothing.
======================================================================
EOF
}

# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------
cmd_help() {
  sed -n '2,49p' "${BASH_SOURCE[0]}"
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
main() {
  local sub="${1:-help}"
  shift || true
  strip_flags "$@"
  set -- "${STRIPPED_ARGS[@]:-}"

  case "${sub}" in
    build)            cmd_build ;;
    bootstrap-code)   cmd_bootstrap_code ;;
    prep)             cmd_prep ;;
    bootstrap-ckpt)   cmd_bootstrap_ckpt ;;
    train)            cmd_train ;;
    status)           cmd_status ;;
    output)           cmd_output ;;
    version-ckpt)     cmd_version_ckpt ;;
    resume-cycle)     cmd_resume_cycle ;;
    help|-h|--help)   cmd_help ;;
    *)
      log "unknown subcommand: ${sub}"
      cmd_help
      exit 1
      ;;
  esac
}

main "$@"
