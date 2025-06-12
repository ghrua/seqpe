#! /bin/bash
#SBATCH --job-name=gpt_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --output=%x_%j.out
#SBATCH --partition=gpu_short
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:6000:4
#SBATCH --cpus-per-task=8


RAW_CMD="bash eval.sh $@"
echo "RAW_CMD: $RAW_CMD"

######################
# ENV Config
######################
CONDA_ROOT_DIR=$(conda info --base)
source $CONDA_ROOT_DIR/etc/profile.d/conda.sh
conda activate text_seqpe
PY_BIN=${CONDA_ROOT_DIR}/${SUFFIX}/bin

# export WANDB_API_KEY=""
######################
# LM Config
######################
EVAL_CKPT_DIR=$1

CUDA_VISIBLE_DEVICES=0,1,2,3
EVAL_BLOCK_SIZES="512,1024,2048,4096,8192,16384"
EVAL_CKPT_NAME="best_model"
EVAL_BATCH_SIZE=1
EVAL_EXTRAPOLATION_METHOD="extend"
WANDB_PROJECT_NAME=gpt2_eval
EVAL_ATTN_METHOD=sdpa
EVAL_INTERPOLATE_MODE=linear


MYPORT=8775
while ss -tulnp | grep -q ":$MYPORT"; do
    echo "Port $PORT is occupied. Trying next..."
    ((MYPORT++))
done

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --main_process_port $MYPORT eval.py \
--eval_block_sizes $EVAL_BLOCK_SIZES \
--eval_ckpt_dir $EVAL_CKPT_DIR \
--eval_ckpt_name $EVAL_CKPT_NAME \
--eval_batch_size $EVAL_BATCH_SIZE \
--eval_extrapolation_method $EVAL_EXTRAPOLATION_METHOD \
--eval_interpolate_mode $EVAL_INTERPOLATE_MODE \
--eval_attn_method $EVAL_ATTN_METHOD \
--use_wandb \
--wandb_project_name $WANDB_PROJECT_NAME \
--wandb_run_name "$RAW_CMD"