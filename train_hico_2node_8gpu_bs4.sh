#!/bin/bash
set -eo pipefail

# =========================
# Paths
# =========================
HOICLIP_DIR="/hkfs/work/workspace/scratch/uhfpp-hoi_data/uhfpp-hoi_data-1773972484/HOICLIP"
CONDA_BASE="/hkfs/home/project/hk-project-test-p0025524/uhfpp/miniforge3"

HOI_PATH="${HOICLIP_DIR}/data/hico_20160224_det"
PRETRAINED="${HOICLIP_DIR}/params/detr-r50-pre-2branch-hico.pth"

# =========================
# Run config
# =========================
RUN_TAG="${RUN_TAG:-hico_2node_8gpu_bs4_lr2e4_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${HOICLIP_DIR}/exps/${RUN_TAG}"

NPROC_PER_NODE=4
PER_GPU_BATCH=4
BASE_GLOBAL_BATCH=16
BASE_LR=0.0001

NNODES="${SLURM_NNODES:-2}"
NODE_RANK="${SLURM_NODEID:-0}"
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
GLOBAL_BATCH=$((WORLD_SIZE * PER_GPU_BATCH))

LR=$(python - <<PY
base_lr = ${BASE_LR}
global_batch = ${GLOBAL_BATCH}
base_global_batch = ${BASE_GLOBAL_BATCH}
print(base_lr * global_batch / base_global_batch)
PY
)

# =========================
# Distributed setup
# =========================
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
else
  MASTER_ADDR="127.0.0.1"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
else
  MASTER_PORT="${MASTER_PORT:-29531}"
fi

# =========================
# NCCL / CPU settings
# =========================
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING=1

# HoreKa 上如果你之前遇到 NET/IB vendor err，可以保留 1，更稳但可能慢。
# 如果之后确认 IB 正常，可以运行前 export NCCL_IB_DISABLE=0。
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

ulimit -n 4096

# =========================
# Conda environment
# =========================
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate hoiclip

cd "${HOICLIP_DIR}"
mkdir -p "${OUTPUT_DIR}" tmp

# =========================
# Sanity checks
# =========================
if [[ ! -d "${HOI_PATH}" ]]; then
  echo "[ERROR] HICO path not found: ${HOI_PATH}"
  exit 1
fi

if [[ ! -f "${PRETRAINED}" ]]; then
  echo "[ERROR] pretrained checkpoint not found: ${PRETRAINED}"
  exit 1
fi

if [[ ! -f "${HOI_PATH}/annotations/trainval_hico.json" ]]; then
  echo "[ERROR] missing train annotation: ${HOI_PATH}/annotations/trainval_hico.json"
  exit 1
fi

if [[ ! -f "${HOI_PATH}/annotations/test_hico.json" ]]; then
  echo "[ERROR] missing test annotation: ${HOI_PATH}/annotations/test_hico.json"
  exit 1
fi

RESUME_ARGS=()
if [[ -f "${OUTPUT_DIR}/checkpoint_last.pth" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/checkpoint_last.pth")
fi

LOG_FILE="${OUTPUT_DIR}/train.node${NODE_RANK}.log"

if [[ "${NODE_RANK}" == "0" ]]; then
  echo "[INFO] HOICLIP_DIR=${HOICLIP_DIR}"
  echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
  echo "[INFO] NNODES=${NNODES}"
  echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "[INFO] WORLD_SIZE=${WORLD_SIZE}"
  echo "[INFO] PER_GPU_BATCH=${PER_GPU_BATCH}"
  echo "[INFO] GLOBAL_BATCH=${GLOBAL_BATCH}"
  echo "[INFO] BASE_LR=${BASE_LR}"
  echo "[INFO] SCALED_LR=${LR}"
  echo "[INFO] MASTER_ADDR=${MASTER_ADDR}"
  echo "[INFO] MASTER_PORT=${MASTER_PORT}"
  echo "[INFO] NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
fi

python -m torch.distributed.launch \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --use_env \
  main.py \
  --output_dir "${OUTPUT_DIR}" \
  --dataset_file hico \
  --hoi_path "${HOI_PATH}" \
  --num_obj_classes 80 \
  --num_verb_classes 117 \
  --backbone resnet50 \
  --num_queries 64 \
  --dec_layers 3 \
  --epochs 90 \
  --lr_drop 60 \
  --lr "${LR}" \
  --use_nms_filter \
  --fix_clip \
  --batch_size "${PER_GPU_BATCH}" \
  --pretrained "${PRETRAINED}" \
  --with_clip_label \
  --with_obj_clip_label \
  --gradient_accumulation_steps 1 \
  --num_workers 8 \
  --opt_sched "multiStep" \
  --dataset_root GEN \
  --model_name HOICLIP \
  --zero_shot_type default \
  --verb_pth ./tmp/verb.pth \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
