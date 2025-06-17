#! /bin/bash
#SBATCH --job-name=gpt_qa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --output=%x_%j.out
#SBATCH --partition=gpu_long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16

RAW_CMD="bash runs/ours_gpt2_qa.sh $@"
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
TASK=squad_qa # squad_qa, hotpot_qa
SEQPE_CONTRASTIVE_WEIGHT=0.1
SEQPE_TRANSFER_WEIGHT=0.1
SEQPE_TRANSFER_METRIC=kl_div # mse, kl_div
PE_RANDOM_SHIFT_RATE=0.1
MAX_TRAIN_STEPS=10000
PE_RANDOM_SHIFT_DOWNSAMPLE=160
CUSTOM_PE_MAIN_BATCH_SIZE=0
DATALOADER_PREFETCH_FACTOR=4
DATALOADER_NUM_WORKERS=6
CUSTOM_TASK_MAIN_BATCH_SIZE=0
PE_EMBED_DIM=768
BLOCK_SIZE=1024
MIXED_PRECISION="bf16" # choices in `no`, 'bf16', and 'fp16'
WANDB_PROJECT_NAME="gpt2_qa"
PE_APPLY_METHOD=attn_scalar # input_add, attn_mul, attn_add, attn_scalar
USE_PE_QK_PER_LAYER=none # none, single, multi
PE_USE_RANDOM_SHIFT=true
SEQPE_SCALAR_SCALE=1.0
SEQPE_DIST_SAMPLE_RANGE=512
PRETRAINED_DIR=""
PROJ_DIR="./"
LEARNING_RATE=0.00005
TRAIN_ON_PROMPT=false
GRADIENT_ACCUMULATION_STEPS=1
ANSWER_LOSS_RATIO=-1
# Usage function
usage() {
    echo "Usage: $0 \  
        [-d PROJ_DIR, default='./'] \  
        [-a TASK, default='squad_qa'] \  
        [-q SEQPE_DIST_SAMPLE_RANGE, default=512] \  
        [-o ANSWER_LOSS_RATIO, default=-1] \  
        [-w SEQPE_CONTRASTIVE_WEIGHT, default=0.1] \  
        [-t SEQPE_TRANSFER_WEIGHT, default=0.1] \  
        [-r PE_RANDOM_SHIFT_RATE, default=0.1] \
        [-l LEARNING_RATE, default=0.000005] \
        [-s SEQPE_SCALAR_SCALE, default=1.0] \  
        [-n NPROC_PER_NODE, default=4] \  
        [-e MAX_TRAIN_STEPS, default=100000] \  
        [-D PE_RANDOM_SHIFT_DOWNSAMPLE, default=160] \  
        [-B CUSTOM_PE_MAIN_BATCH_SIZE, default='auto'] \  
        [-S CUSTOM_TASK_MAIN_BATCH_SIZE, default='auto'] \  
        [-U PE_USE_RANDOM_SHIFT, default=true] \  
        [-P PRETRAINED_DIR, default=''] \  
        [-Q GRADIENT_ACCUMULATION_STEPS, default=1] \  
        [-T TRAIN_ON_PROMPT, default=false] \  
        [-W DATALOADER_NUM_WORKERS, default=6]"
    exit 1
}

# Parse command-line arguments
while getopts "a:b:d:q:w:r:e:n:t:s:l:o:D:B:W:S:R:P:T:Q:" opt; do
  case "$opt" in
    a) TASK="$OPTARG" ;;
    b) BLOCK_SIZE="$OPTARG" ;;
    d) PROJ_DIR="$OPTARG" ;;
    q) SEQPE_DIST_SAMPLE_RANGE="$OPTARG" ;;
    w) SEQPE_CONTRASTIVE_WEIGHT="$OPTARG" ;;
    t) SEQPE_TRANSFER_WEIGHT="$OPTARG" ;;
    s) SEQPE_SCALAR_SCALE="$OPTARG" ;;
    r) PE_RANDOM_SHIFT_RATE="$OPTARG" ;;
    l) LEARNING_RATE="$OPTARG" ;;
    o) ANSWER_LOSS_RATIO="$OPTARG" ;;
    e) MAX_TRAIN_STEPS="$OPTARG" ;;
    n) NPROC_PER_NODE="$OPTARG" ;;
    D) PE_RANDOM_SHIFT_DOWNSAMPLE="$OPTARG" ;;
    B) CUSTOM_PE_MAIN_BATCH_SIZE="$OPTARG" ;;
    W) DATALOADER_NUM_WORKERS="$OPTARG" ;;
    S) CUSTOM_TASK_MAIN_BATCH_SIZE="$OPTARG" ;;
    R) PE_USE_RANDOM_SHIFT="$OPTARG" ;;
    Q) GRADIENT_ACCUMULATION_STEPS="$OPTARG" ;;
    T) TRAIN_ON_PROMPT="$OPTARG" ;;
    P) PRETRAINED_DIR="$OPTARG" ;;
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
    BATCH_SIZE=16
    PE_MAIN_BATCH_SIZE=16
    SEQPE_TRANSFER_BATCH_SIZE=16
    SEQPE_CONTRASTIVE_BATCH_SIZE=16
elif [ "$NPROC_PER_NODE" -eq 2 ]; then
    # debug
    CUDA_VISIBLE_DEVICES=0,1
    BATCH_SIZE=8
    PE_MAIN_BATCH_SIZE=8
    SEQPE_TRANSFER_BATCH_SIZE=8
    SEQPE_CONTRASTIVE_BATCH_SIZE=8
    WANDB_PROJECT_NAME="gpt2_qa_debug"
else
    # debug
    CUDA_VISIBLE_DEVICES=0
    BATCH_SIZE=8
    PE_MAIN_BATCH_SIZE=8
    SEQPE_TRANSFER_BATCH_SIZE=8
    SEQPE_CONTRASTIVE_BATCH_SIZE=8
    WANDB_PROJECT_NAME="gpt2_qa_debug"
fi

if [ "$CUSTOM_PE_MAIN_BATCH_SIZE" -ne 0 ]; then
    PE_MAIN_BATCH_SIZE=$CUSTOM_PE_MAIN_BATCH_SIZE
    SEQPE_TRANSFER_BATCH_SIZE=$CUSTOM_PE_MAIN_BATCH_SIZE
    SEQPE_CONTRASTIVE_BATCH_SIZE=$CUSTOM_PE_MAIN_BATCH_SIZE
fi 

if [ "$CUSTOM_TASK_MAIN_BATCH_SIZE" -ne 0 ]; then
    BATCH_SIZE=$CUSTOM_TASK_MAIN_BATCH_SIZE
fi

DATA_DIR=$PROJ_DIR/data/${TASK}/

echo "DATA_DIR: $DATA_DIR"
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
echo "SEQPE_SCALAR_SCALE: $SEQPE_SCALAR_SCALE"
echo "SEQPE_DIST_SAMPLE_RANGE: $SEQPE_DIST_SAMPLE_RANGE"
echo "PRETRAINED_DIR: $PRETRAINED_DIR"
########################################
current_date=$(date +"%y%m%d")
project_random_id=$(openssl rand -base64 6 | tr -dc 'A-Za-z0-9')

OUTPUT_DIR="${PROJ_DIR}/text_seq_pe_out/${current_date}_${project_random_id}"
mkdir -p $OUTPUT_DIR
echo "OUTPUT_DIR: $OUTPUT_DIR"

MYPORT=8775
while ss -tulnp | grep -q ":$MYPORT"; do
    echo "Port $PORT is occupied. Trying next..."
    ((MYPORT++))
done

######################
# Run
######################

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --main_process_port $MYPORT qa_sft.py \
--mixed_precision $MIXED_PRECISION \
--train_file $DATA_DIR/train.jsonl \
--validation_file $DATA_DIR/validation.jsonl \
--model_name_or_path $MODEL_NAME_OR_PATH \
--output_dir $OUTPUT_DIR \
--preprocessing_num_workers $DATALOADER_NUM_WORKERS \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--block_size $BLOCK_SIZE \
--max_train_steps $MAX_TRAIN_STEPS \
--pretrained_dir $PRETRAINED_DIR \
--learning_rate $LEARNING_RATE \
--train_on_prompt $TRAIN_ON_PROMPT \
--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
--answer_loss_ratio $ANSWER_LOSS_RATIO \
--use_wandb \
--wandb_project_name $WANDB_PROJECT_NAME \
--wandb_run_name "$RAW_CMD" \
--pe_config_override "$(cat <<EOF
{
  "PE_MAIN_BATCH_SIZE": "$PE_MAIN_BATCH_SIZE",
  "SEQPE_TRANSFER_BATCH_SIZE": "$SEQPE_TRANSFER_BATCH_SIZE",
  "SEQPE_CONTRASTIVE_BATCH_SIZE": "$SEQPE_CONTRASTIVE_BATCH_SIZE",
  "PE_RANDOM_SHIFT_DOWNSAMPLE": "$PE_RANDOM_SHIFT_DOWNSAMPLE"
}
EOF
)"

######################
# Eval
######################
MYPORT=8875
while ss -tulnp | grep -q ":$MYPORT"; do
    echo "Port $PORT is occupied. Trying next..."
    ((MYPORT++))
done

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
EVAL_DATA_PREF=./data/${TASK}/validation
EVAL_BLOCK_SIZES="1024,2048,4096,8192,16384"
EVAL_CKPT_NAME="best_model"
EVAL_BATCH_SIZE=1
EVAL_EXTRAPOLATION_METHOD="extend"
WANDB_PROJECT_NAME=gpt2_qa_eval
EVAL_INTERPOLATE_MODE=linear
EVAL_SIMPLE_QA=false

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --main_process_port $MYPORT qa_eval.py \
--eval_data_pref $EVAL_DATA_PREF \
--eval_block_sizes $EVAL_BLOCK_SIZES \
--eval_ckpt_dir $OUTPUT_DIR \
--eval_ckpt_name $EVAL_CKPT_NAME \
--eval_extrapolation_method $EVAL_EXTRAPOLATION_METHOD \
--eval_interpolate_mode $EVAL_INTERPOLATE_MODE \
--eval_simple_qa $EVAL_SIMPLE_QA \
--use_wandb \
--wandb_project_name $WANDB_PROJECT_NAME \
--wandb_run_name "$RAW_CMD"
