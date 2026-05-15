#!/usr/bin/env bash
set -euo pipefail

# Evaluate a trained MYDS checkpoint with HOICLIP.
# Metrics are computed by datasets/myds_eval_finalversion.py (wired in engine.py).
# Built-in defaults mirror:
#   EVAL_SPLIT=test
#   NNODES=2
#   NPROC_PER_NODE=4
#   NODE_RANK=${SLURM_NODEID}
#   MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
#   MASTER_PORT=29531

HOICLIP_DIR="${HOICLIP_DIR:-/hkfs/work/workspace/scratch/uhfpp-hoi_data/uhfpp-hoi_data-1773972484/HOICLIP}"
CONDA_BASE="${CONDA_BASE:-/hkfs/home/project/hk-project-test-p0025524/uhfpp/miniforge3}"
MYDS_PATH="${MYDS_PATH:-/hkfs/work/workspace/scratch/uhfpp-hoi_data/uhfpp-hoi_data-1773972484/datasets/myds}"
export MYDS_PATH

CKPT_PATH="${CKPT_PATH:-/hkfs/work/workspace/scratch/uhfpp-hoi_data/uhfpp-hoi_data-1773972484/HOICLIP/logs/myds_2node_8gpu_bs4_20260513_213720/checkpoint_last.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOICLIP_DIR}/logs/eval_myds_$(date +%Y%m%d_%H%M%S)}"
PRETRAINED="${PRETRAINED:-${HOICLIP_DIR}/params/detr-r50-pre-2branch-hico.pth}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"   # one of: both|test|val
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null | head -n1)}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29531}"
ENABLE_GROUP_EVAL="${ENABLE_GROUP_EVAL:-0}"
EVAL_DEBUG="${EVAL_DEBUG:-0}"
ENABLE_ROLE_PRIOR_EVAL="${ENABLE_ROLE_PRIOR_EVAL:-0}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
# Conda activate scripts may reference unset vars (e.g. MKL_INTERFACE_LAYER),
# which conflicts with `set -u`. Temporarily relax nounset only for activation.
set +u
conda activate hoiclip
set -u

cd "${HOICLIP_DIR}"
mkdir -p "${OUTPUT_DIR}" tmp

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

read -r NUM_OBJ_CLASSES NUM_VERB_CLASSES <<<"$(python - <<'PY'
from datasets.myds_meta import load_myds_meta
import os
meta = load_myds_meta(os.environ['MYDS_PATH'])
print(len(meta['objects']), len(meta['verbs']))
PY
)"

if [[ -z "${NUM_OBJ_CLASSES}" || -z "${NUM_VERB_CLASSES}" ]]; then
  echo "[ERROR] failed to infer NUM_OBJ_CLASSES/NUM_VERB_CLASSES from MYDS_PATH=${MYDS_PATH}"
  exit 1
fi

if [[ "${EVAL_SPLIT}" != "test" && "${EVAL_SPLIT}" != "val" && "${EVAL_SPLIT}" != "both" ]]; then
  echo "[ERROR] invalid EVAL_SPLIT=${EVAL_SPLIT}, expected one of both|test|val"
  exit 1
fi

GROUP_EVAL_ARGS=()
if [[ "${ENABLE_GROUP_EVAL}" == "1" ]]; then
  GROUP_EVAL_ARGS+=(--enable_group_eval)
fi
DEBUG_ARGS=()
if [[ "${EVAL_DEBUG}" == "1" ]]; then
  DEBUG_ARGS+=(--eval_debug)
fi
if [[ "${ENABLE_ROLE_PRIOR_EVAL}" == "1" ]]; then
  DEBUG_ARGS+=(--enable_role_prior_eval)
fi

LAUNCHER=(python main.py)
if [[ "${NPROC_PER_NODE}" -gt 1 || "${NNODES}" -gt 1 ]]; then
  LAUNCHER=(
    python -m torch.distributed.launch
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    --use_env
    main.py
  )
fi

"${LAUNCHER[@]}" \
  --eval \
  --eval_split "${EVAL_SPLIT}" \
  --dataset_file myds \
  --hoi_path "${MYDS_PATH}" \
  --myds_train_anno "${MYDS_PATH}/annotations/train_20k.json" \
  --myds_val_anno "${MYDS_PATH}/annotations/val.json" \
  --myds_test_anno "${MYDS_PATH}/annotations/test.json" \
  --num_obj_classes "${NUM_OBJ_CLASSES}" \
  --num_verb_classes "${NUM_VERB_CLASSES}" \
  --backbone resnet50 \
  --num_queries 64 \
  --dec_layers 3 \
  --batch_size 4 \
  --num_workers 4 \
  --dataset_root GEN \
  --model_name HOICLIP \
  --zero_shot_type default \
  --use_nms_filter \
  --fix_clip \
  --with_clip_label \
  --with_obj_clip_label \
  --pretrained "${PRETRAINED}" \
  --resume "${CKPT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --verb_pth ./tmp/verb.pth \
  "${GROUP_EVAL_ARGS[@]}" \
  "${DEBUG_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo "[INFO] Eval finished. Log: ${OUTPUT_DIR}/eval.log"
