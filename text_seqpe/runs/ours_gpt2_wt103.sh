#! /bin/bash
#SBATCH --job-name=gpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --output=%x_%j.out
#SBATCH --partition=gpu_week
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16

RAW_CMD="bash runs/ours_gpt2_wt103.sh $@"
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

MODEL_NAME_OR_PATH=openai-community/gpt2
# arguments
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=4
DATA_DIR=./
PE_MAX_POSITION=20000
SEQPE_CONTRASTIVE_WEIGHT=0.1
SEQPE_TRANSFER_WEIGHT=0.1
SEQPE_TRANSFER_METRIC=kl_div # mse, kl_div
PE_RANDOM_SHIFT_RATE=0.1
MAX_TRAIN_STEPS=100000
PE_RANDOM_SHIFT_DOWNSAMPLE=320
CUSTOM_PE_MAIN_BATCH_SIZE=0
DATALOADER_PREFETCH_FACTOR=4
DATALOADER_NUM_WORKERS=6
CUSTOM_TASK_MAIN_BATCH_SIZE=0
PE_EMBED_DIM=768
BLOCK_SIZE=512
MIXED_PRECISION="bf16" # choices in `no`, 'bf16', and 'fp16'
WANDB_PROJECT_NAME="gpt2"
PE_APPLY_METHOD=attn_scalar # input_add, attn_mul, attn_add, attn_scalar
USE_PE_QK_PER_LAYER=single # none, single, multi
PE_USE_RANDOM_SHIFT=true
SEQPE_LOGIT_SCALED_LOSS=1.0
SEQPE_DIST_SAMPLE_RANGE=256
SEQPE_MULTI_HEAD_LOSS=true
SEQPE_SCALAR_SCALE=1.0
# Usage function
usage() {
    echo "Usage: $0 \  
        [-d DATA_DIR, default='./'] \  
        [-b BLOCK_SIZE, default=256] \  
        [-p PE_MAX_POSITION, default=10000] \  
        [-q SEQPE_DIST_SAMPLE_RANGE, default=512] \  
        [-w SEQPE_CONTRASTIVE_WEIGHT, default=0.1] \  
        [-t SEQPE_TRANSFER_WEIGHT, default=0.1] \  
        [-r PE_RANDOM_SHIFT_RATE, default=0.1] \
        [-s SEQPE_SCALAR_SCALE, default=1.0] \  
        [-n NPROC_PER_NODE, default=4] \  
        [-e MAX_TRAIN_STEPS, default=100000] \  
        [-m PE_APPLY_METHOD, default='attn_mul'] \  
        [-D PE_RANDOM_SHIFT_DOWNSAMPLE, default=320] \  
        [-B CUSTOM_PE_MAIN_BATCH_SIZE, default='auto'] \  
        [-S CUSTOM_TASK_MAIN_BATCH_SIZE, default='auto'] \  
        [-U PE_USE_RANDOM_SHIFT, default=true] \  
        [-U USE_PE_QK_PER_LAYER, default='multi'] \  
        [-T SEQPE_TRANSFER_METRIC, default=0.1] \  
        [-L SEQPE_MULTI_HEAD_LOSS, default=true] \
        [-M MIXED_PRECISION, default=0.1] \    
        [-W DATALOADER_NUM_WORKERS, default=6]"
    exit 1
}

# Parse command-line arguments
while getopts "b:d:p:q:w:r:e:n:m:t:s:D:B:W:S:U:R:T:L:M:" opt; do
  case "$opt" in
    b) BLOCK_SIZE="$OPTARG" ;;
    d) DATA_DIR="$OPTARG" ;;
    p) PE_MAX_POSITION="$OPTARG" ;;
    q) SEQPE_DIST_SAMPLE_RANGE="$OPTARG" ;;
    w) SEQPE_CONTRASTIVE_WEIGHT="$OPTARG" ;;
    t) SEQPE_TRANSFER_WEIGHT="$OPTARG" ;;
    s) SEQPE_SCALAR_SCALE="$OPTARG" ;;
    r) PE_RANDOM_SHIFT_RATE="$OPTARG" ;;
    e) MAX_TRAIN_STEPS="$OPTARG" ;;
    n) NPROC_PER_NODE="$OPTARG" ;;
    m) PE_APPLY_METHOD="$OPTARG" ;;
    D) PE_RANDOM_SHIFT_DOWNSAMPLE="$OPTARG" ;;
    B) CUSTOM_PE_MAIN_BATCH_SIZE="$OPTARG" ;;
    W) DATALOADER_NUM_WORKERS="$OPTARG" ;;
    U) USE_PE_QK_PER_LAYER="$OPTARG" ;;
    S) CUSTOM_TASK_MAIN_BATCH_SIZE="$OPTARG" ;;
    T) SEQPE_TRANSFER_METRIC="$OPTARG" ;;
    L) SEQPE_MULTI_HEAD_LOSS="$OPTARG" ;;
    M) MIXED_PRECISION="$OPTARG" ;;
    R) PE_USE_RANDOM_SHIFT="$OPTARG" ;;
    *) usage ;;
  esac
done


if [[ "$NPROC_PER_NODE" -ne 4 && "$NPROC_PER_NODE" -ne 8 && "$NPROC_PER_NODE" -ne 2 && "$NPROC_PER_NODE" -ne 1 ]]; then
    echo "Error: NPROC_PER_NODE must be either 1, 2, 4 or 8. NPROC_PER_NODE=1 or 2 is for debugging"
    exit 1
fi

if [ "$NPROC_PER_NODE" -eq 8 ]; then
    
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    BATCH_SIZE=16
    PE_MAIN_BATCH_SIZE=8
    SEQPE_TRANSFER_BATCH_SIZE=8
    SEQPE_CONTRASTIVE_BATCH_SIZE=8
elif [ "$NPROC_PER_NODE" -eq 4 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3
    BATCH_SIZE=32
    PE_MAIN_BATCH_SIZE=16
    SEQPE_TRANSFER_BATCH_SIZE=16
    SEQPE_CONTRASTIVE_BATCH_SIZE=16
elif [ "$NPROC_PER_NODE" -eq 2 ]; then
    # debug
    CUDA_VISIBLE_DEVICES=0,1
    BATCH_SIZE=32
    PE_MAIN_BATCH_SIZE=16
    SEQPE_TRANSFER_BATCH_SIZE=16
    SEQPE_CONTRASTIVE_BATCH_SIZE=16
    WANDB_PROJECT_NAME="gpt2_debug"
else
    # debug
    CUDA_VISIBLE_DEVICES=0
    BATCH_SIZE=16
    PE_MAIN_BATCH_SIZE=8
    SEQPE_TRANSFER_BATCH_SIZE=8
    SEQPE_CONTRASTIVE_BATCH_SIZE=8
    WANDB_PROJECT_NAME="gpt2_debug"
fi

if [ "$CUSTOM_PE_MAIN_BATCH_SIZE" -ne 0 ]; then
    PE_MAIN_BATCH_SIZE=$CUSTOM_PE_MAIN_BATCH_SIZE
    SEQPE_TRANSFER_BATCH_SIZE=$CUSTOM_PE_MAIN_BATCH_SIZE
    SEQPE_CONTRASTIVE_BATCH_SIZE=$CUSTOM_PE_MAIN_BATCH_SIZE
fi 

if [ "$CUSTOM_TASK_MAIN_BATCH_SIZE" -ne 0 ]; then
    BATCH_SIZE=$CUSTOM_TASK_MAIN_BATCH_SIZE
fi


echo "DATA_DIR: $DATA_DIR"
echo "PE_MAX_POSITION: $PE_MAX_POSITION"
echo "SEQPE_CONTRASTIVE_WEIGHT: $SEQPE_CONTRASTIVE_WEIGHT"
echo "PE_RANDOM_SHIFT_RATE: $PE_RANDOM_SHIFT_RATE"
echo "MAX_TRAIN_STEPS: $MAX_TRAIN_STEPS"
echo "BLOCK_SIZE: $BLOCK_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "PE_MAIN_BATCH_SIZE: $PE_MAIN_BATCH_SIZE"
echo "PE_RANDOM_SHIFT_DOWNSAMPLE: $PE_RANDOM_SHIFT_DOWNSAMPLE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PE_USE_RANDOM_SHIFT: $PE_USE_RANDOM_SHIFT"
echo "SEQPE_LOGIT_SCALED_LOSS: $SEQPE_LOGIT_SCALED_LOSS"
echo "SEQPE_DIST_SAMPLE_RANGE: $SEQPE_DIST_SAMPLE_RANGE"
echo "SEQPE_MULTI_HEAD_LOSS: $SEQPE_MULTI_HEAD_LOSS"
echo "MIXED_PRECISION: $MIXED_PRECISION"
echo "SEQPE_SCALAR_SCALE: $SEQPE_SCALAR_SCALE"
########################################
PE_TYPE=seq_pe # seq_pe, rotary, vanilla, sin, alibi
PE_DATA_DIM=1 # NOTE: seq_pe could use both 1d and 2d PE, but other pe_types can only use 1d

SEQPE_TRANSFER_BETA=1.0

SEQPE_TEMPERATURE=1.0

SEQPE_RESID_PDROP=0.1
SEQPE_ATTN_PDROP=0
SEQPE_INIT_NORM_WEIGHT=1.0
SEQPE_ATTN_DIRECTION=causal

current_date=$(date +"%y%m%d")
project_random_id=$(openssl rand -base64 6 | tr -dc 'A-Za-z0-9')

OUTPUT_DIR="${DATA_DIR}/text_seq_pe_out/${current_date}_${project_random_id}"
mkdir -p $OUTPUT_DIR


MYPORT=8775
while ss -tulnp | grep -q ":$MYPORT"; do
    echo "Port $PORT is occupied. Trying next..."
    ((MYPORT++))
done

######################
# Run
######################
PYTHON_CMD=(accelerate launch --main_process_port $MYPORT train.py \
--mixed_precision $MIXED_PRECISION \
--dataset_name Salesforce/wikitext \
--dataset_config_name wikitext-103-raw-v1 \
--model_name_or_path $MODEL_NAME_OR_PATH \
--output_dir $OUTPUT_DIR \
--preprocessing_num_workers $DATALOADER_NUM_WORKERS \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--block_size $BLOCK_SIZE \
--max_train_steps $MAX_TRAIN_STEPS \
--pe_type $PE_TYPE \
--pe_apply_method $PE_APPLY_METHOD \
--pe_embed_dim $PE_EMBED_DIM \
--pe_data_dim $PE_DATA_DIM \
--pe_max_position $PE_MAX_POSITION \
--seqpe_dist_sample_range $SEQPE_DIST_SAMPLE_RANGE \
--seqpe_transfer_batch_size $SEQPE_TRANSFER_BATCH_SIZE \
--seqpe_transfer_weight $SEQPE_TRANSFER_WEIGHT \
--seqpe_contrastive_batch_size $SEQPE_CONTRASTIVE_BATCH_SIZE \
--seqpe_contrastive_weight $SEQPE_CONTRASTIVE_WEIGHT \
--seqpe_attn_pdrop $SEQPE_ATTN_PDROP \
--seqpe_resid_pdrop $SEQPE_RESID_PDROP \
--seqpe_attn_direction $SEQPE_ATTN_DIRECTION \
--use_wandb \
--wandb_project_name $WANDB_PROJECT_NAME \
--wandb_run_name "$RAW_CMD" \
--seqpe_transfer_metric $SEQPE_TRANSFER_METRIC \
--seqpe_transfer_beta $SEQPE_TRANSFER_BETA \
--seqpe_data_size_multiplier 1 \
--seqpe_init_norm_weight $SEQPE_INIT_NORM_WEIGHT \
--seqpe_last_layernorm \
--seqpe_logit_scaled_loss $SEQPE_LOGIT_SCALED_LOSS \
--pe_main_batch_size $PE_MAIN_BATCH_SIZE \
--pe_random_shift_rate $PE_RANDOM_SHIFT_RATE \
--pe_random_shift_downsample $PE_RANDOM_SHIFT_DOWNSAMPLE \
--use_pe_qk_per_layer $USE_PE_QK_PER_LAYER \
--seqpe_scalar_scale $SEQPE_SCALAR_SCALE \
--use_pe_multi_head)

if [ "$PE_USE_RANDOM_SHIFT" = true ]; then
  PYTHON_CMD+=("--pe_use_random_shift")
fi

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES "${PYTHON_CMD[@]}"
######################
# Eval
######################
MYPORT=8775
while ss -tulnp | grep -q ":$MYPORT"; do
    echo "Port $PORT is occupied. Trying next..."
    ((MYPORT++))
done

EVAL_BLOCK_SIZES="512,1024,2048,4096,8192,16384,32768"
EVAL_CKPT_NAME="best_model"
EVAL_BATCH_SIZE=1
EVAL_EXTRAPOLATION_METHOD="extend"
WANDB_PROJECT_NAME=gpt2_eval
EVAL_INTERPOLATE_MODE=linear

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --main_process_port $MYPORT eval.py \
--eval_block_sizes $EVAL_BLOCK_SIZES \
--eval_ckpt_dir $OUTPUT_DIR \
--eval_ckpt_name $EVAL_CKPT_NAME \
--eval_batch_size $EVAL_BATCH_SIZE \
--eval_extrapolation_method $EVAL_EXTRAPOLATION_METHOD \
--eval_interpolate_mode $EVAL_INTERPOLATE_MODE \
--use_wandb \
--wandb_project_name $WANDB_PROJECT_NAME \
--wandb_run_name "$RAW_CMD"
