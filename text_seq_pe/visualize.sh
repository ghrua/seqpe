#! /bin/bash
#SBATCH --job-name=gpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --output=%x_%j.out
#SBATCH --partition=gpu_long
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:6000:4
#SBATCH --cpus-per-task=16

######################
# ENV Config
######################
CONDA_ROOT_DIR=$(conda info --base)
source $CONDA_ROOT_DIR/etc/profile.d/conda.sh
conda activate text_seqpe
PY_BIN=${CONDA_ROOT_DIR}/${SUFFIX}/bin

export WANDB_API_KEY="511a2692431f62cfb93c7c1f8bbe78bae0b9c3df"
######################
# LM Config
######################
# EVAL_CKPT_DIR=".//text_seq_pe_out/250415_g4G5xnnd" # ours
# EVAL_CKPT_DIR=".//text_seq_pe_out/250420_n0nPYOQY"
# EVAL_CKPT_DIR=".//text_seq_pe_out/250422_K0Vmvki" # baseline
EVAL_CKPT_DIR=$1 # baseline

EVAL_CKPT_NAME=${2:-best_model}
EVAL_BLOCK_SIZES="4096" # 512, 1024
EVAL_EXTRAPOLATION_METHOD="extend"
WANDB_PROJECT_NAME=gpt2_eval
CUDA_VISIBLE_DEVICES=${3:-0}
EVAL_INTERPOLATE_MODE=linear

# for VISUALIZE_START_POS in 0 512 1024 10000
# do
VISUALIZE_START_POS=0
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch visualize.py \
--eval_block_sizes $EVAL_BLOCK_SIZES \
--eval_ckpt_dir $EVAL_CKPT_DIR \
--eval_ckpt_name $EVAL_CKPT_NAME \
--eval_extrapolation_method $EVAL_EXTRAPOLATION_METHOD \
--eval_interpolate_mode $EVAL_INTERPOLATE_MODE \
--visualize_start_pos $VISUALIZE_START_POS \
--visualize_output_dir ./visualization
# done

# for VISUALIZE_LAYER_INDEX in -1 3 7 11
# do
#     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch visualize.py \
#     --eval_block_sizes $EVAL_BLOCK_SIZES \
#     --eval_ckpt_dir $EVAL_CKPT_DIR \
#     --eval_ckpt_name $EVAL_CKPT_NAME \
#     --eval_extrapolation_method $EVAL_EXTRAPOLATION_METHOD \
#     --eval_interpolate_mode $EVAL_INTERPOLATE_MODE \
#     --visualize_layer_index $VISUALIZE_LAYER_INDEX \
#     --visualize_pos_index -1 \
#     --visualize_output_dir ./visualization
# done