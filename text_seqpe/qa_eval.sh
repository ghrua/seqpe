#! /bin/bash
#SBATCH --job-name=gpt_qa_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --output=%x_%j.out
#SBATCH --partition=gpu_short
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8

# RAW_CMD=$(sacct -j "$SLURM_JOB_ID" -o submitline -P | head -n 2 | tail -n 1)
RAW_CMD="bash qa_eval.sh $@"
echo $RAW_CMD

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
# EVAL_CKPT_DIR=".//text_seq_pe_out/250505_DlisHrOU/"
EVAL_CKPT_DIR=$1


# CUDA_VISIBLE_DEVICES=0
# EVAL_BLOCK_SIZES="512,1024,2048,4096,8192,16384,32768"
TASK=squad_qa # wt103_gen, squad_qa, hotpot_qa
EVAL_DATA_PREF=./data/${TASK}/validation
EVAL_BLOCK_SIZES="1024,2048,4096,8192"
EVAL_CKPT_NAME=${2:-best_model} # best_model
EVAL_BATCH_SIZE=1
WANDB_PROJECT_NAME=gpt2_qa_eval
EVAL_MODE=${3:-ppl} # ppl, gen
EVAL_INTERPOLATE_MODE=linear
DO_SAMPLE=${4:-false}
EVAL_SIMPLE_QA=${5:-false}
EVAL_EXTRAPOLATION_METHOD=${6:-extend}


echo "EVAL_MODE: $EVAL_MODE"
if [ "$EVAL_MODE" = "gen" ]; then
    echo "EVAL_SIMPLE_QA: $EVAL_SIMPLE_QA"
    echo "DO_SAMPLE: $DO_SAMPLE"
fi

# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch qa_eval.py \
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES 
python qa_eval.py \
--eval_data_pref $EVAL_DATA_PREF \
--eval_block_sizes $EVAL_BLOCK_SIZES \
--eval_ckpt_dir $EVAL_CKPT_DIR \
--eval_ckpt_name $EVAL_CKPT_NAME \
--eval_extrapolation_method $EVAL_EXTRAPOLATION_METHOD \
--eval_interpolate_mode $EVAL_INTERPOLATE_MODE \
--eval_simple_qa $EVAL_SIMPLE_QA \
--eval_mode $EVAL_MODE \
--do_sample $DO_SAMPLE \
--use_wandb \
--wandb_project_name $WANDB_PROJECT_NAME \
--wandb_run_name "${RAW_CMD}"
