#! /bin/bash
#SBATCH --job-name=vit_s_seqpe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --output=%x_%j.out
#SBATCH --partition=gpu_week
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=16

source /home/is/huayang-l/anaconda3/etc/profile.d/conda.sh
conda activate swin
export WANDB_API_KEY="511a2692431f62cfb93c7c1f8bbe78bae0b9c3df"


MYPORT=${2:-8775}
######################
# VISUALIZE_CKPT_DIR="/cl/work4/huayang-l/image_seq_pe_out/250109_ARPvKI8" # seq_pe attn_add
# VISUALIZE_CKPT_DIR="/cl/work4/huayang-l/image_seq_pe_out/250117_UdGQmlzL" # seq_pe attn_scalar trans1.0
# VISUALIZE_CKPT_DIR="/cl/work4/huayang-l/image_seq_pe_out/250121_yqjtTqVv" # seq_pe attn_scalar trans2.0
# VISUALIZE_CKPT_DIR="/cl/work4/huayang-l/image_seq_pe_out/250124_A9lwGd7F" # seq_pe attn_scalar trans4.0
# VISUALIZE_CKPT_DIR="/cl/work4/huayang-l/image_seq_pe_out/250118_2lzHLye1" # seq_pe input_add trans0.1
VISUALIZE_CKPT_DIR="/cl/work4/huayang-l/image_seq_pe_out/250125_SNEPiUp5" # seq_pe attn_scalar ct0.1 random shift
# VISUALIZE_CKPT_DIR="/cl/work4/huayang-l/image_seq_pe_out/250109_zeHbNXMM" # rotary
# VISUALIZE_CKPT_DIR="/cl/work4/huayang-l/image_seq_pe_out/250110_PZ8ovTa" # vanilla input_add
######################
VISUALIZE_IMG_SIZES="224" # eval size of images. please use `;` as the seperator. the size could be both int, e.g., 224, or a tuple, e.g., (224, 224)
VISUALIZE_POS_INDEX="0,0;7,7;13,0;13,13"
VISUALIZE_EXTRAPOLATION_METHOD="extend" # "interpolate" or "extend".

# VISUALIZE_IMG_SIZES="512" # eval size of images. please use `;` as the seperator. the size could be both int, e.g., 224, or a tuple, e.g., (224, 224)
# VISUALIZE_POS_INDEX="0,0;16,16;31,0;31,31"
# VISUALIZE_EXTRAPOLATION_METHOD="extend" # "interpolate" or "extend".

VISUALIZE_CKPT_NAMES="best_model.pth" # we support the evaluation of multiple models. please use `;` as the seperator
VISUALIZE_INTERPOLATE_MODE="bicubic" # "bicubic", "bilinear", or "linear". The rotary method only supports "linear".

python visualize.py \
--visualize_ckpt_dir $VISUALIZE_CKPT_DIR \
--visualize_ckpt_names $VISUALIZE_CKPT_NAMES \
--visualize_img_sizes $VISUALIZE_IMG_SIZES \
--visualize_extrapolation_method $VISUALIZE_EXTRAPOLATION_METHOD \
--visualize_interpolate_mode $VISUALIZE_INTERPOLATE_MODE \
--visualize_pos_index $VISUALIZE_POS_INDEX