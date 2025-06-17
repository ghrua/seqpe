#! /bin/bash
#SBATCH --job-name=vit_s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --output=%x_%j.out
#SBATCH --partition=gpu_week
#SBATCH --time=200:00:00
#SBATCH --gres=gpu:6000:4
#SBATCH --cpus-per-task=32

######################
# RUNNING CMD
######################

RAW_CMD="bash runs/ours_vit.sh $@"
echo "RAW_CMD: $RAW_CMD"
######################


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

########################################
# arguments
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=4
DATA_DIR=./
PE_MAX_POSITION=10000
SEQPE_CONTRASTIVE_WEIGHT=0.1
PE_RANDOM_SHIFT_RATE=0.1
EPOCH=400
CONFIG=vit_s_16 # swin_tiny_patch4_window7, cvt_13, t2tvit_14, resnet_50, vit_base_16
PE_RANDOM_SHIFT_DOWNSAMPLE=320
CUSTOM_PE_MAIN_BATCH_SIZE=0
DATALOADER_PREFETCH_FACTOR=4
DATALOADER_NUM_WORKERS=6
CUSTOM_TASK_MAIN_BATCH_SIZE=0
BASE_LR=0.0006
SEQPE_TRANSFER_WEIGHT=0.1
SEQPE_LOGIT_SCALED_LOSS=1.0
CUSTOM_SEQPE_RESID_PDROP=-1
CUSTOM_PE_CLIP_GRAD=-1
CUSTOM_SEQPE_WARMUP_STEPS=-1
CUSTOM_PE_EMBED_DIM=-1
PE_WEIGHT_DECAY=0
PE_LR=-1
CUSTOM_VIT_DROP_PATH_RATE=-1
PE_APPLY_METHOD=attn_mul # input_add, rotary, attn_mul, attn_add, attn_scalar
USE_PE_QK_PER_LAYER=multi # single, multi
# Usage function
usage() {
    echo "Usage: $0 \  
        [-d DATA_DIR, default='./'] \  
        [-p PE_MAX_POSITION, default=10000] \  
        [-w SEQPE_CONTRASTIVE_WEIGHT, default=0.1] \  
        [-t SEQPE_TRANSFER_WEIGHT, default=0.1] \  
        [-q SEQPE_LOGIT_SCALED_LOSS, default=1.0] \ 
        [-r PE_RANDOM_SHIFT_RATE, default=0.1] \  
        [-o SEQPE_RESID_PDROP, default='auto'] \  
        [-l BASE_LR, default=0.0006] \  
        [-L PE_LR, default=0.0012] \  
        [-n NPROC_PER_NODE, default=4] \  
        [-e EPOCH, default=400] \  
        [-m PE_APPLY_METHOD, default='attn_mul'] \  
        [-c CONFIG, default='vit_s_16'] \
        [-A CUSTOM_SEQPE_WARMUP_STEPS, default='auto'] \    
        [-E CUSTOM_PE_EMBED_DIM, default='auto'] \    
        [-G CUSTOM_PE_CLIP_GRAD, default='auto'] \    
        [-D PE_RANDOM_SHIFT_DOWNSAMPLE, default=320] \  
        [-Y PE_WEIGHT_DECAY, default='0'] \  
        [-B CUSTOM_PE_MAIN_BATCH_SIZE, default='auto'] \  
        [-S CUSTOM_TASK_MAIN_BATCH_SIZE, default='auto'] \  
        [-O CUSTOM_VIT_DROP_PATH_RATE, default='auto'] \  
        [-U USE_PE_QK_PER_LAYER, default='multi'] \  
        [-W DATALOADER_NUM_WORKERS, default=6]"
    exit 1
}

# Parse command-line arguments
while getopts "d:p:w:t:q:r:e:n:c:l:o:m:A:E:G:D:B:W:S:Y:L:U:O:" opt; do
  case "$opt" in
    d) DATA_DIR="$OPTARG" ;;
    p) PE_MAX_POSITION="$OPTARG" ;;
    w) SEQPE_CONTRASTIVE_WEIGHT="$OPTARG" ;;
    t) SEQPE_TRANSFER_WEIGHT="$OPTARG" ;;
    q) SEQPE_LOGIT_SCALED_LOSS="$OPTARG" ;;
    r) PE_RANDOM_SHIFT_RATE="$OPTARG" ;;
    o) CUSTOM_SEQPE_RESID_PDROP="$OPTARG" ;;
    m) PE_APPLY_METHOD="$OPTARG" ;;
    e) EPOCH="$OPTARG" ;;
    n) NPROC_PER_NODE="$OPTARG" ;;
    c) CONFIG="$OPTARG" ;;
    A) CUSTOM_SEQPE_WARMUP_STEPS="$OPTARG" ;;
    E) CUSTOM_PE_EMBED_DIM="$OPTARG" ;;
    G) CUSTOM_PE_CLIP_GRAD="$OPTARG" ;;
    D) PE_RANDOM_SHIFT_DOWNSAMPLE="$OPTARG" ;;
    B) CUSTOM_PE_MAIN_BATCH_SIZE="$OPTARG" ;;
    W) DATALOADER_NUM_WORKERS="$OPTARG" ;;
    S) CUSTOM_TASK_MAIN_BATCH_SIZE="$OPTARG" ;;
    Y) PE_WEIGHT_DECAY="$OPTARG" ;;
    l) BASE_LR="$OPTARG" ;;
    L) PE_LR="$OPTARG" ;;
    U) USE_PE_QK_PER_LAYER="$OPTARG" ;;
    O) CUSTOM_VIT_DROP_PATH_RATE="$OPTARG" ;;
    *) usage ;;
  esac
done


if [[ "$NPROC_PER_NODE" -ne 4 && "$NPROC_PER_NODE" -ne 8 && "$NPROC_PER_NODE" -ne 2 && "$NPROC_PER_NODE" -ne 1 ]]; then
    echo "Error: NPROC_PER_NODE must be either 1, 2, 4 or 8. NPROC_PER_NODE=1 or 2 is for debugging"
    exit 1
fi

if [ "$NPROC_PER_NODE" -eq 8 ]; then
    MAIN_TASK_BATCH_SIZE=128
    PE_MAIN_BATCH_SIZE=16
    SEQPE_TRANSFER_BATCH_SIZE=16
    SEQPE_CONTRASTIVE_BATCH_SIZE=16
elif [ "$NPROC_PER_NODE" -eq 4 ]; then
    MAIN_TASK_BATCH_SIZE=256
    PE_MAIN_BATCH_SIZE=32
    SEQPE_TRANSFER_BATCH_SIZE=32
    SEQPE_CONTRASTIVE_BATCH_SIZE=32
elif [ "$NPROC_PER_NODE" -eq 2 ]; then
    # debug
    MAIN_TASK_BATCH_SIZE=256
    PE_MAIN_BATCH_SIZE=32
    SEQPE_TRANSFER_BATCH_SIZE=32
    SEQPE_CONTRASTIVE_BATCH_SIZE=32
else
    # debug
    MAIN_TASK_BATCH_SIZE=16
    PE_MAIN_BATCH_SIZE=8
    SEQPE_TRANSFER_BATCH_SIZE=8
    SEQPE_CONTRASTIVE_BATCH_SIZE=8
fi

if [ "$CUSTOM_PE_MAIN_BATCH_SIZE" -ne 0 ]; then
    SEQPE_TRANSFER_BATCH_SIZE=$CUSTOM_PE_MAIN_BATCH_SIZE
    SEQPE_CONTRASTIVE_BATCH_SIZE=$SEQPE_CONTRASTIVE_BATCH_SIZE
    PE_MAIN_BATCH_SIZE=$CUSTOM_PE_MAIN_BATCH_SIZE
fi 

if [ "$CUSTOM_TASK_MAIN_BATCH_SIZE" -ne 0 ]; then
    MAIN_TASK_BATCH_SIZE=$CUSTOM_TASK_MAIN_BATCH_SIZE
fi

if [ "$CONFIG" = "vit_s_16" ]; then
    PE_EMBED_DIM=384
    PE_CLIP_GRAD=5.0
    SEQPE_RESID_PDROP=0.1
    SEQPE_WARMUP_STEPS=0
    VIT_DROP_PATH_RATE=0
elif [ "$CONFIG" = "vit_b_16" ]; then
    PE_EMBED_DIM=384
    PE_CLIP_GRAD=5.0
    SEQPE_RESID_PDROP=0.3
    SEQPE_WARMUP_STEPS=0
    VIT_DROP_PATH_RATE=0.1
else
    # debug
    echo "Error: CONFIG must be either vit_s_16 or vit_b_16"
    exit 1
fi

if [ "$CUSTOM_SEQPE_RESID_PDROP" != "-1" ]; then
    SEQPE_RESID_PDROP=$CUSTOM_SEQPE_RESID_PDROP
fi 

if [ "$CUSTOM_PE_CLIP_GRAD" != "-1" ]; then
    PE_CLIP_GRAD=$CUSTOM_PE_CLIP_GRAD
fi

if [ "$CUSTOM_VIT_DROP_PATH_RATE" != "-1" ]; then
    VIT_DROP_PATH_RATE=$CUSTOM_VIT_DROP_PATH_RATE
fi

if [ "$CUSTOM_SEQPE_WARMUP_STEPS" != "-1" ]; then
    SEQPE_WARMUP_STEPS=$CUSTOM_SEQPE_WARMUP_STEPS
fi


if [ "$CUSTOM_PE_EMBED_DIM" != "-1" ]; then
    PE_EMBED_DIM=$CUSTOM_PE_EMBED_DIM
fi

echo "DATA_DIR: $DATA_DIR"
echo "PE_MAX_POSITION: $PE_MAX_POSITION"
echo "SEQPE_CONTRASTIVE_WEIGHT: $SEQPE_CONTRASTIVE_WEIGHT"
echo "SEQPE_TRANSFER_WEIGHT: $SEQPE_TRANSFER_WEIGHT"
echo "PE_RANDOM_SHIFT_RATE: $PE_RANDOM_SHIFT_RATE"
echo "EPOCH: $EPOCH"
echo "CONFIG: $CONFIG"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MAIN_TASK_BATCH_SIZE: $MAIN_TASK_BATCH_SIZE"
echo "PE_MAIN_BATCH_SIZE: $PE_MAIN_BATCH_SIZE"
echo "PE_RANDOM_SHIFT_DOWNSAMPLE: $PE_RANDOM_SHIFT_DOWNSAMPLE"
echo "BASE_LR: $BASE_LR"
echo "PE_CLIP_GRAD: $PE_CLIP_GRAD"
echo "SEQPE_RESID_PDROP: $SEQPE_RESID_PDROP"
echo "SEQPE_WARMUP_STEPS: $SEQPE_WARMUP_STEPS"
echo "PE_EMBED_DIM: $PE_EMBED_DIM"
echo "PE_WEIGHT_DECAY: $PE_WEIGHT_DECAY"
echo "PE_LR: $PE_LR"
echo "VIT_DROP_PATH_RATE: $VIT_DROP_PATH_RATE"
echo "PE_APPLY_METHOD: $PE_APPLY_METHOD"
echo "USE_PE_QK_PER_LAYER: $USE_PE_QK_PER_LAYER"

########################################
# SEQPE_PRETRAINED="/cl/work4/huayang-l/image_seq_pe_out/250214_rrNDQozF/ckpt_epoch_299.pth" # e.g., /cl/work4/huayang-l/image_seq_pe_pretrained/vit_small_seqpe_mul_ct_weight_0.05_trans_weight_0.05.pth
SEQPE_PRETRAINED=""

MODE=$CONFIG # swintiny, cvt13, t2t, resnet50, vit
IMG_SIZE=224 # 224, 384
LAMBDA_DRLOC=0.01 # swin: 0.5, t2t: 0.1, cvt: 0.1
DRLOC_MODE=l1 # l1, ce, cbr

DATASET=imagenet # imagenet-100, imagenet, cifar-10, cifar-100, svhn, places365, flowers102, clipart, infograph, painting, quickdraw, real, sketch
NUM_CLASSES=1000

PE_TYPE=seq_pe # seq_pe, rotary, vanilla, sin
PE_DATA_DIM=2 # NOTE: seq_pe could use both 1d and 2d PE, but other pe_types can only use 1d


SEQPE_TRANSFER_BETA=1.0
SEQPE_TRANSFER_NUM=64

SEQPE_TEMPERATURE=1.0
SEQPE_ATTN_PDROP=0
SEQPE_INIT_NORM_WEIGHT=1.0
SEQPE_ATTN_DIRECTION=causal

current_date=$(date +"%y%m%d")
project_random_id=$(openssl rand -base64 6 | tr -dc 'A-Za-z0-9')
WANDB_PROJECT_NAME="${CONFIG}"

PROJECT_NAME="${DATA_DIR}/image_seq_pe_out/${current_date}_${project_random_id}"
mkdir -p $PROJECT_NAME


DISK_DATA=${DATA_DIR}/datasets/${DATASET}
TARGET_FOLDER=${DATASET}-${MODE}_24only_seq_pe
SAVE_DIR=${DATA_DIR}/swintransformer-expr/${TARGET_FOLDER}

MYPORT=8775
while ss -tulnp | grep -q ":$MYPORT"; do
    echo "Port $PORT is occupied. Trying next..."
    ((MYPORT++))
done

######################
# Train
######################

$PY_BIN/torchrun --nproc_per_node=${NPROC_PER_NODE} \
--nnodes=1 \
--node_rank=0 \
--master_port=$MYPORT main.py \
--cfg ./configs/${CONFIG}_${IMG_SIZE}.yaml \
--dataset ${DATASET} \
--num_classes ${NUM_CLASSES} \
--data-path ${DISK_DATA} \
--batch-size $MAIN_TASK_BATCH_SIZE \
--accumulation-steps 1 \
--output $PROJECT_NAME \
--lambda_dlocr ${LAMBDA_DRLOC} \
--dlocr_mode nonlinear \
--mask-ratio 0.1 \
--mask_use False \
--vit_drop_path_rate $VIT_DROP_PATH_RATE \
--base_lr $BASE_LR \
--pe_type $PE_TYPE \
--pe_clip_grad $PE_CLIP_GRAD \
--pe_apply_method $PE_APPLY_METHOD \
--pe_embed_dim $PE_EMBED_DIM \
--pe_data_dim $PE_DATA_DIM \
--pe_max_position $PE_MAX_POSITION \
--seqpe_warmup_steps $SEQPE_WARMUP_STEPS \
--seqpe_transfer_batch_size $SEQPE_TRANSFER_BATCH_SIZE \
--seqpe_transfer_weight $SEQPE_TRANSFER_WEIGHT \
--seqpe_transfer_num $SEQPE_TRANSFER_NUM \
--seqpe_contrastive_batch_size $SEQPE_CONTRASTIVE_BATCH_SIZE \
--seqpe_contrastive_weight $SEQPE_CONTRASTIVE_WEIGHT \
--seqpe_attn_pdrop $SEQPE_ATTN_PDROP \
--seqpe_resid_pdrop $SEQPE_RESID_PDROP \
--seqpe_attn_direction $SEQPE_ATTN_DIRECTION \
--seqpe_decay $PE_WEIGHT_DECAY \
--seqpe_lr $PE_LR \
--use_wandb \
--wandb_project_name $WANDB_PROJECT_NAME \
--wandb_run_name "$RAW_CMD" \
--seqpe_transfer_metric kl_div \
--seqpe_transfer_beta $SEQPE_TRANSFER_BETA \
--seqpe_pretrained "${SEQPE_PRETRAINED}" \
--seqpe_data_size_multiplier 1 \
--seqpe_init_norm_weight $SEQPE_INIT_NORM_WEIGHT \
--use_pe_qk_per_layer $USE_PE_QK_PER_LAYER \
--seqpe_last_layernorm \
--use_pe_multi_head \
--pe_use_random_shift \
--pe_main_batch_size $PE_MAIN_BATCH_SIZE \
--pe_random_shift_rate $PE_RANDOM_SHIFT_RATE \
--total_epochs $EPOCH \
--pe_random_shift_downsample $PE_RANDOM_SHIFT_DOWNSAMPLE \
--dataloader_prefetch_factor $DATALOADER_PREFETCH_FACTOR \
--dataloader_num_workers $DATALOADER_NUM_WORKERS


######################
# EVAL
######################
while ss -tulnp | grep -q ":$MYPORT"; do
    echo "Port $PORT is occupied. Trying next..."
    ((MYPORT++))
done

EVAL_BATCH_SIZE=64
# PROJECT_NAME="/cl/work4/huayang-l/image_seq_pe_out/250214_rrNDQozF"
EVAL_IMG_SIZES="224;320;384;448;512;640;672" # eval size of images. please use `;` as the seperator. the size could be both int, e.g., 224, or a tuple, e.g., (224, 224)
EVAL_CKPT_NAMES="best_model.pth" # we support the evaluation of multiple models. please use `;` as the seperator
EVAL_EXTRAPOLATION_METHOD="extend" # "interpolate" or "extend".
EVAL_INTERPOLATE_MODE="bicubic" # "bicubic", "bilinear", or "linear". The rotary method only supports "linear".

$PY_BIN/torchrun --nproc_per_node=${NPROC_PER_NODE} \
--nnodes=1 \
--node_rank=0 \
--master_port=$MYPORT eval.py \
--eval_ckpt_dir $PROJECT_NAME  \
--eval_ckpt_names $EVAL_CKPT_NAMES \
--eval_img_sizes $EVAL_IMG_SIZES \
--eval_extrapolation_method $EVAL_EXTRAPOLATION_METHOD \
--eval_interpolate_mode $EVAL_INTERPOLATE_MODE \
--eval_batch_size $EVAL_BATCH_SIZE \
--use_wandb \
--wandb_project_name "${WANDB_PROJECT_NAME}_eval" \
--wandb_run_name "$RAW_CMD"