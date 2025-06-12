# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np
import math
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
    init_wandb,
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

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)

    # easy config modification
    parser.add_argument('--eval_img_sizes', default="224", type=str, help='test image size. If you have multiple img sizes, please use `;` as the seperator.')
    parser.add_argument('--eval_ckpt_dir', default="", required=True, type=str, help='directory of the ckpt')
    parser.add_argument('--eval_ckpt_names', default="best_model.pth", type=str, help='please use `;` as the seperator.')
    parser.add_argument('--eval_batch_size', type=int, default=256, help="batch size for single GPU")
    parser.add_argument('--eval_extrapolation_method', type=str, default="extend", choices=["interpolate", "extend"])
    parser.add_argument('--eval_interpolate_mode', type=str, default='bicubic', choices=["linear", "bicubic", "bilinear", 'none'])

    ## setup 
    # parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    ## wandb
    parser.add_argument("--use_wandb", action='store_true', help='use wandb to log results')
    parser.add_argument("--wandb_project_name", type=str, default="vit", help='name of the wandb project')
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--start_pos", type=str, default=None, help='start position for testing, default (0, 0)')

    args, unparsed = parser.parse_known_args()
    with open(os.path.join(args.eval_ckpt_dir, "config.json")) as fin:
        config = CN.load_cfg(fin.read())
    config.EVAL_MODE = True
    config.EVAL.IMG_SIZES = args.eval_img_sizes
    config.EVAL.CKPT_DIR = args.eval_ckpt_dir
    config.EVAL.CKPT_NAMES = args.eval_ckpt_names
    config.DATA.BATCH_SIZE = args.eval_batch_size
    config.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    if args.start_pos is None:
        config.START_POS = 0 if config.PE_DATA_DIM == 1 else (0, 0)
    else:
        config.START_POS = eval(args.start_pos)
    args.eval_interpolate_mode = None if args.eval_interpolate_mode == 'none' else args.eval_interpolate_mode

    if config.PE_APPLY_METHOD in ["rotary_2d", "rotary_mixed"]:
        assert args.eval_extrapolation_method == "extend"

    if config.PE_TYPE in ['rotary', 'sin'] and config.PE_APPLY_METHOD not in ["rotary_2d", "rotary_mixed"] and args.eval_extrapolation_method == "interpolate":
        assert args.eval_interpolate_mode == "linear"

    if args.eval_img_sizes != "224" and args.eval_extrapolation_method == "extend" and config.PE_TYPE == "vanilla":
        raise ValueError("The vanilla PE only supports the extrapolation by interpolation")
    return args, config

torch.autograd.set_detect_anomaly(True)
from safetensors.torch import safe_open


def main():
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"eval_{config.MODEL.NAME}")

    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    config.MODEL.NUM_HEADS = model.num_heads
    config.MODEL.DEPTH = model.depth
    pe_model = build_pe_model(config)

    model.cuda()
    logger.info(str(model))

    if pe_model is not None:
        pe_model.cuda()
        logger.info(str(pe_model))
    
    model = torch.nn.parallel.DistributedDataParallel(model, 
        device_ids=[config.LOCAL_RANK], 
        broadcast_buffers=False,
        find_unused_parameters=True)
    logger.info("Successfully init DDP for model")

    if config.PE_TYPE == 'seq_pe':
        pe_model = torch.nn.parallel.DistributedDataParallel(pe_model,
            device_ids=[config.LOCAL_RANK],
            broadcast_buffers=False,
            find_unused_parameters=True)
        logger.info("Successfully init DDP for pe model")
        pe_model_wo_ddp = pe_model.module
    else:
        pe_model_wo_ddp = pe_model

    model_wo_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_wo_ddp, 'flops'):
        flops = model_wo_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    if args.use_wandb and dist.get_rank() == 0:
        init_wandb(config, args.wandb_project_name, args.wandb_run_name)

    ####################################################
    # prepare pe data samplers
    device = model.device
    """
    NOTE: explanation for add_cls_token_pe
    add_cls_token_pe=Ture: the PE model will generate the PE of cls.
    add_cls_token_pe=Ture: we use a special pos embedding to model the pos of cls.

    This is because that seq_pe is hard to represent cls's pos in 2d situation.
    """
    add_cls_token_pe = True if config.PE_TYPE != 'seq_pe' and config.PE_APPLY_METHOD not in ["rotary_2d", "rotary_mixed"] else False
    ####################################################

    ckpt_dir = config.EVAL.CKPT_DIR
    ckpt_names = config.EVAL.CKPT_NAMES
    img_sizes = config.EVAL.IMG_SIZES
    train_image_size = config.DATA.IMG_SIZE
    train_input_shape = model_wo_ddp.input_shape
    for ckpt in ckpt_names.split(";"):
        config.MODEL.RESUME = os.path.join(ckpt_dir, ckpt)
        _ = load_eval_checkpoint(config, model_wo_ddp, pe_model_wo_ddp, logger)
        for img_size in img_sizes.split(";"):
            img_size = eval(img_size)
            config.DATA.IMG_SIZE = img_size
            # build eval dataset for corresponding img_size
            dataset_val, _ = build_dataset(is_train=False, config=config)
            if config.TEST.SEQUENTIAL:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            else:
                sampler_val = torch.utils.data.distributed.DistributedSampler(
                    dataset_val, shuffle=False
                )
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=config.DATA.BATCH_SIZE,
                shuffle=False,
                num_workers=config.DATA.NUM_WORKERS,
                pin_memory=config.DATA.PIN_MEMORY,
                drop_last=False
            )
            model_wo_ddp.reset_patch_embed(img_size)
            pe_main_task_sampler = SeqPeMainTaskPosSampler(
                    model_wo_ddp.input_shape if args.eval_extrapolation_method != "interpolate" else train_input_shape,
                    config.PE_MAX_POSITION, config.SEQPE_MAX_DIGITS, data_dim=config.PE_DATA_DIM,
                    add_cls_token_pe=add_cls_token_pe, device=device, use_random_shift=False, default_start_pos=config.START_POS
            ) if config.PE_TYPE not in ["vanilla", "nope"] else None
            
            acc1, acc5, loss = validate(config, data_loader_val, model, model_wo_ddp, pe_model, pe_main_task_sampler, logger, epoch=None, use_wandb=args.use_wandb, eval_extrapolation_method=args.eval_extrapolation_method, train_input_shape=train_input_shape, interpolate_mode=args.eval_interpolate_mode)
            logger.info(f"Accuracy of {ckpt} on the {len(dataset_val)} test images with size ({img_size}, {img_size}): {acc1:.3f}%")
            

@torch.no_grad()
def validate(config, data_loader, model, model_wo_ddp, pe_model, pe_main_task_sampler, logger, epoch=None, use_wandb=False, eval_extrapolation_method="extend", train_input_shape=(14, 14), interpolate_mode=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    if pe_model is not None:
        pe_model.eval()

    device = torch.device('cuda', dist.get_rank())

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    pe_norm_meter = AverageMeter()

    end = time.time()
    orig_input_max_len = train_input_shape[0] * train_input_shape[1]
    new_shape = model_wo_ddp.input_shape
    new_input_max_len = new_shape[0] * new_shape[1]
    pe_main_inputs = pe_main_task_sampler.next(device=device) if pe_main_task_sampler is not None else None
    if config.PE_TYPE == "seq_pe":
        pos_seq_data, pad_mask = pe_main_inputs['pos_seq_data'], pe_main_inputs['pad_mask']
        pe_main = pe_model(pos_seq_data, pad_mask)
    elif config.PE_TYPE == "rotary" or config.PE_TYPE == "sin":
        pe_main = pe_model(pe_main_inputs['pos_ids'])
    else:
        pe_main = None
    
    if eval_extrapolation_method == "interpolate" and config.PE_TYPE != "nope" and orig_input_max_len != new_input_max_len:
        if pe_main is None and config.PE_TYPE == "vanilla":
            # vanilla pe
            pe_main = model_wo_ddp.pos_embed[:, 1:, :]
            interpolate_mode = "bicubic" if interpolate_mode is None else interpolate_mode
            if interpolate_mode in ["bicubic", "bilinear"]:
                pe_main = PeUtils.full_interpolation(pe_main, train_input_shape, new_shape, mode=interpolate_mode)
                pe_main = torch.cat([model_wo_ddp.pos_embed[:, :1, :], pe_main], dim=1).squeeze(0)
            elif interpolate_mode == "linear":
                pe_main = PeUtils.full_interpolation(pe_main, orig_input_max_len, new_input_max_len, mode=interpolate_mode)
                pe_main = torch.cat([model_wo_ddp.pos_embed[:, 0, :], pe_main], dim=0)
        elif config.PE_TYPE == "seq_pe":
            if interpolate_mode in ["bicubic", "bilinear"]:
                pe_main = PeUtils.full_interpolation(pe_main, train_input_shape, new_shape, mode=interpolate_mode).squeeze(0)
            elif interpolate_mode == "linear":
                pe_main = PeUtils.full_interpolation(pe_main, pe_main.size(0), new_input_max_len, mode=interpolate_mode).squeeze(0)
        elif config.PE_TYPE == "rotary" or config.PE_TYPE == "sin":
            # interpolation for 1-dim pe
            pe_main = PeUtils.full_interpolation(pe_main, pe_main.size(0), new_input_max_len+1, mode="linear" if interpolate_mode is None else interpolate_mode).squeeze(0)
        else:
            raise NotImplementedError

    # compute output
    if pe_main is not None:
        if config.PE_APPLY_METHOD == "rotary_mixed":
            pe_main = pe_main.unsqueeze(1)
        else:
            pe_main = pe_main.unsqueeze(0) 
        
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if pe_main is not None:
            pe_norm_meter.update(torch.mean(torch.norm(pe_main.detach(), dim=-1)).item(), pe_main.size(0))

        output = model(images, pe_main)

        # measure accuracy and record loss
        loss = criterion(output.sup, target)
        acc1, acc5 = accuracy(output.sup, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    if use_wandb and dist.get_rank() == 0:
        wandb.log({
            'valid/loss': loss_meter.avg,
            'valid/acc@1': acc1_meter.avg,
            'valid/acc@5': acc5_meter.avg,
        }, step=config.DATA.IMG_SIZE)
    
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg



@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


# please surround your entry function with
#
#   if __name__ == "__main__":
#
# otherwise, some multiprocess issues may be triggered.
if __name__ == '__main__':
    main()
