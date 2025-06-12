#! /bin/bash
#SBATCH --job-name=vit_s_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --output=%x_%j.out
#SBATCH --partition=gpu_short
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=16


CONDA_ROOT_DIR=$(conda info --base)
SUFFIX="/envs/swin_latest"

if [[ "$CONDA_ROOT_DIR" == */envs/swin_latest ]]; then
    # Remove the suffix
    CONDA_ROOT_DIR="${CONDA_ROOT_DIR%$SUFFIX}"
fi

source $CONDA_ROOT_DIR/etc/profile.d/conda.sh
conda activate swin_latest
PY_BIN=${CONDA_ROOT_DIR}/${SUFFIX}/bin

# export WANDB_API_KEY=""

EVAL_CKPT_DIR=$1
MYPORT=${2:-8775}
######################
while ss -tulnp | grep -q ":$MYPORT"; do
    echo "Port $PORT is occupied. Trying next..."
    ((MYPORT++))
done

EVAL_BATCH_SIZE=128
EVAL_IMG_SIZES="224;320;384;448;512;640;672" # eval size of images. please use `;` as the seperator. the size could be both int, e.g., 224, or a tuple, e.g., (224, 224)
EVAL_CKPT_NAMES="best_model.pth" # we support the evaluation of multiple models. please use `;` as the seperator
EVAL_EXTRAPOLATION_METHOD="extend" # "interpolate" or "extend".
EVAL_INTERPOLATE_MODE="bicubic" # "bicubic", "bilinear", or "linear". The rotary method only supports "linear".
NPROC_PER_NODE=1

$PY_BIN/torchrun --nproc_per_node=${NPROC_PER_NODE} \
--nnodes=1 \
--node_rank=0 \
--master_port=$MYPORT eval.py \
--eval_ckpt_dir "${EVAL_CKPT_DIR}" \
--eval_ckpt_names $EVAL_CKPT_NAMES \
--eval_img_sizes $EVAL_IMG_SIZES \
--eval_extrapolation_method $EVAL_EXTRAPOLATION_METHOD \
--eval_interpolate_mode $EVAL_INTERPOLATE_MODE \
--eval_batch_size $EVAL_BATCH_SIZE