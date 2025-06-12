# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from collections.abc import Iterable
import os
import time
import argparse
import datetime
import numpy as np
import json
from yacs.config import CfgNode as CN
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from pe_sampler import SeqPeMainTaskPosSampler, SeqPeContrstiveDataSampler, SeqPeTransferDataSampler
from models.pe_utils import PeUtils
from models.pe_criterion import SeqPeContrastiveCriterion, SeqPeTransferCriterion
from config import get_config
from models import build_model, build_pe_model
from data.build import build_loader, build_dataset
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import (
    get_git_commit_hash,
    load_pe_pretrained,
    load_checkpoint,
    load_eval_checkpoint,
    save_checkpoint, 
    save_checkpoint_best,
    get_grad_norm, 
    auto_resume_helper, 
    reduce_tensor
)
from torch.nn import MSELoss

from data.masking_generator import JigsawPuzzleMaskedRegion
from ssl_mjp import cal_selfsupervised_loss

import utils 
import warnings
import wandb
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import seaborn as sns


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)

    # easy config modification
    parser.add_argument('--visualize_img_sizes', default="224", type=str, help='test image size. If you have multiple img sizes, please use `;` as the seperator.')
    parser.add_argument('--visualize_ckpt_dir', default="", required=True, type=str, help='directory of the ckpt')
    parser.add_argument('--visualize_ckpt_names', default="best_model.pth", type=str, help='please use `;` as the seperator.')
    parser.add_argument('--visualize_layer_index', type=int, default=-1, help="visualize the PE of a specific layer")
    parser.add_argument('--visualize_pos_index', type=str, default="None", help="visualize the PE of a specific position. 2d pos could be '7,7' ")
    parser.add_argument('--visualize_extrapolation_method', type=str, default="extend", choices=["interpolate", "extend"])
    parser.add_argument('--visualize_interpolate_mode', type=str, default='bicubic', choices=["linear", "bicubic", "bilinear", 'none'])
    parser.add_argument('--visualize_save_dir', type=str, default='./figs/')

    ## setup 
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()
    with open(os.path.join(args.visualize_ckpt_dir, "config.json")) as fin:
        config = CN.load_cfg(fin.read())
    config.EVAL_MODE = True
    config.EVAL.IMG_SIZES = args.visualize_img_sizes
    config.EVAL.CKPT_DIR = args.visualize_ckpt_dir
    config.EVAL.CKPT_NAMES = args.visualize_ckpt_names
    config.LOCAL_RANK = args.local_rank

    if args.visualize_img_sizes != "224" and args.visualize_extrapolation_method == "vanilla" and config.PE_TYPE == "vanilla":
        raise ValueError("The vanilla PE only supports the extrapolation by interpolation")
    return args, config

torch.autograd.set_detect_anomaly(True)
from safetensors.torch import safe_open


# def visualize_rotary(pe):
#     # e^{im\theta} = cosm\theta 

@torch.no_grad()
def main():
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"visualize_{config.MODEL.NAME}")

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")


    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    pe_model = build_pe_model(config)

    model = model.cuda()
    pe_model = build_pe_model(config)

    if pe_model is not None:
        pe_model = pe_model.cuda()
        logger.info(str(pe_model))

    ####################################################
    # prepare pe data samplers
    device = torch.device("cuda")
    """
    NOTE: explanation for add_cls_token_pe
    add_cls_token_pe=Ture: the PE model will generate the PE of cls.
    add_cls_token_pe=Ture: we use a special pos embedding to model the pos of cls.

    This is because that seq_pe is hard to represent cls's pos in 2d situation.
    """
    add_cls_token_pe = True if config.PE_TYPE != 'seq_pe' else False
    ####################################################
    pe_type = config.PE_TYPE
    pe_apply_method = config.PE_APPLY_METHOD
    ckpt_dir = config.EVAL.CKPT_DIR
    ckpt_names = config.EVAL.CKPT_NAMES
    img_sizes = config.EVAL.IMG_SIZES
    train_image_size = config.DATA.IMG_SIZE
    patch_size = config.TRAIN.PATCH_SIZE
    use_pe_multi_head = config.USE_PE_MULTI_HEAD
    extrapolation_method = args.visualize_extrapolation_method
    train_input_shape = (train_image_size//patch_size, train_image_size//patch_size)

    interpolate_mode = args.visualize_interpolate_mode
    layer_index = args.visualize_layer_index
    pos_index = [eval(it) for it in args.visualize_pos_index.split(";")] if args.visualize_pos_index != "None" else None
    save_dir = args.visualize_save_dir
    
    ckpt_dir_basename = os.path.basename(ckpt_dir)
    for ckpt in ckpt_names.split(";"):
        config.MODEL.RESUME = os.path.join(ckpt_dir, ckpt)
        _ = load_eval_checkpoint(config, model, pe_model, logger)
        for img_size in img_sizes.split(";"):

            
            img_size = eval(img_size)
            new_shape = (img_size//patch_size, img_size//patch_size)

            # build eval dataset for corresponding img_size
            pe_main_task_sampler = SeqPeMainTaskPosSampler(
                new_shape if extrapolation_method == "extend" else train_input_shape,
                config.PE_MAX_POSITION, config.SEQPE_MAX_DIGITS, data_dim=config.PE_DATA_DIM,
                add_cls_token_pe=add_cls_token_pe,
            ) if pe_type != "vanilla" else None

            pe_main_inputs = pe_main_task_sampler.next(device=device) if pe_main_task_sampler is not None else None
            if pe_type == "seq_pe":
                pos_seq_data, pad_mask = pe_main_inputs['pos_seq_data'], pe_main_inputs['pad_mask']
                pe_main = pe_model(pos_seq_data, pad_mask)
            elif pe_type == "rotary" or pe_type == "sin":
                pe_main = pe_model(pe_main_inputs['pos_ids'])
            else:
                pe_main = model.pos_embed[:, 1:, :]

            if extrapolation_method == "interpolate":
                orig_input_max_len = train_input_shape[0] * train_input_shape[1]
                pe_main = PeUtils.full_interpolation(pe_main[:orig_input_max_len], train_input_shape, new_shape, mode=interpolate_mode)
            
            pe_main = pe_main.squeeze(0)
            if layer_index >= 0:
                assert pe_apply_method.startswith("attn")
                pe_q, pe_k = model.blocks[layer_index].attn.pe_qk(pe_main)
            else:
                pe_q, pe_k = pe_main, pe_main

            heatmap = pe_q @ pe_k.transpose(-2, -1)
            fig_dir = os.path.join(save_dir, ckpt_dir_basename, ckpt.split(".")[0], extrapolation_method, f"size{img_size}")
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            
            if pos_index is not None:
                for pos in pos_index:
                    fig = plt.figure(figsize=(8, 8))
                    pos_2d_str = "-".join([str(i) for i in pos])
                    if isinstance(pos, Iterable):
                        pos = new_shape[0] * pos[0] + pos[1]
                    elif isinstance(pos, int):
                        pos = pos
                    pos_heatmap = heatmap[pos].reshape(*new_shape)
                    pos_heatmap = pos_heatmap.cpu().numpy()
                    plt.imshow(pos_heatmap, cmap='hot', aspect='auto', vmin=0, vmax=250) 
                    plt.colorbar()
                    fig_fname = fig_fname = f"pos{pos_2d_str}.jpg" if pos >= 0 else f"all.jpg" 
                    fig_fname = f"layer{layer_index}_{fig_fname}" if layer_index > 0 else f"raw_{fig_fname}"
                    logger.info(f"Saving plot to {os.path.join(fig_dir, fig_fname)}")
                    plt.savefig(os.path.join(fig_dir, fig_fname), dpi=300)
                    plt.close(fig)
            else:
                fig = plt.figure(figsize=(8, 8))
                heatmap = heatmap.cpu().numpy()
                # sns.heatmap(heatmap, linewidths=0.5)  # Heatmap
                plt.imshow(heatmap, cmap='hot', aspect='auto', vmin=0, vmax=250)
                plt.colorbar()
                fig_fname = f"layer{layer_index}_all.jpg" if layer_index > 0 else f"raw_all.jpg"
                logger.info(f"Saving plot to {os.path.join(fig_dir, fig_fname)}")
                plt.savefig(os.path.join(fig_dir, fig_fname), dpi=300)
                plt.close(fig)


if __name__ == '__main__':
    main()
