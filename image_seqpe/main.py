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
import json

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from pe_sampler import SeqPeMainTaskPosSampler, SeqPeContrstiveDataSampler, SeqPeTransferDataSampler
from models.pe_utils import PeUtils, PeWeightScheduler
from models.pe_criterion import SeqPeContrastiveCriterion, SeqPeTransferCriterion
from config import get_config
from models import build_model, build_pe_model
from data.build import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
import wandb
from utils import (
    init_wandb,
    get_git_commit_hash,
    load_pe_pretrained,
    load_checkpoint, 
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
warnings.filterwarnings("ignore", category=UserWarning)

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, default=2, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--repeated-aug', type=bool, default=False)
    # arguments for vit_base
    parser.add_argument('--base_lr', default=6e-4, type=float)

    # distributed training
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="imagenet", help="imagenet, imagenet-100, cifar-10, cifar-100, svhn, places365")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--ape", action="store_true", help="using absolute position embedding")
    parser.add_argument("--rpe", action="store_false", help="using relative position embedding")
    parser.add_argument("--total_epochs", type=int, default=300)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--use-jigsaw", action="store_true")
    parser.add_argument("--use-unk-pos", action="store_true")
    parser.add_argument("--eval-mode", type=str, default="none")
    parser.add_argument("--use-idx-emb", action="store_true")
    parser.add_argument("--use-dlocr", action="store_true")
    parser.add_argument("--lambda_dlocr", type=float, default=0.01)
    parser.add_argument("--dlocr-type", type=str, default="linear", help="linear, nonlinear")
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--mask-type", type=str, default="mjp", help="[mjp, pps]")
    parser.add_argument("--mask_use", type=bool, default=False, help="")
    parser.add_argument("--mask-ratio", type=float, default=-1)
    parser.add_argument("--num_attention_heads", type=int, default=-1, help="Number of attention heads")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--vit_drop_path_rate", type=float, default=-1)

    # pe args
    parser.add_argument("--pe_type", type=str, choices=["seq_pe", "sin", "rotary", "vanilla", "nope"], default="vanilla", help="Position encoding method.")
    parser.add_argument("--pe_apply_method", type=str, choices=["input_add", "rotary", "rotary_2d", "rotary_mixed", "attn_mul", "attn_add", "attn_scalar", "nope"], default="input_add", help="Position encoding method.")
    parser.add_argument("--pe_embed_dim", type=int, default=384, help="Dimension of position encoding embeddings.")
    parser.add_argument("--pe_clip_grad", type=float, default=5.0, help="Dimension of position encoding embeddings.")
    parser.add_argument("--pe_data_dim", type=int, default=1, choices=[1, 2], help="1d or 2d data")
    parser.add_argument("--pe_max_position", type=int, default=10000, help="max position embeddings")
    parser.add_argument("--pe_main_batch_size", type=int, default=-1, help="default pe_main_batch_size is set to batch_size later")
    parser.add_argument("--pe_use_random_shift", action="store_true", help="whether shift the training position embedding")
    parser.add_argument("--pe_random_shift_rate", default=1.0, type=float, help="when pe_use_random_shift is set, how much percent of the batch will use the random shift. default is 100%.")
    parser.add_argument("--pe_random_shift_downsample", default=320, type=int, help="only sample pe_random_shift_downsample blocks for training")
    parser.add_argument("--sinusoidal_pe_base", type=int, default=10000, help="base of sin & rotary pe")
    parser.add_argument("--use_pe_multi_head", action="store_true", help='whether split pe for different head. If set, we will only generate pe with dim=pe_embed_dim//head_num. Note that this cannot be used together with pe_apply_method=input_add')
    parser.add_argument("--use_pe_qk_per_layer", type=str, default=None, choices=['multi', 'single'], help='whether create weights that map a single PE to query and key PEs at each layer')

    # seqpe args
    parser.add_argument("--seqpe_multi_head_loss", action="store_true")
    parser.add_argument("--seqpe_logit_scaled_loss", type=float, default=1.0)
    parser.add_argument("--seqpe_pretrained", type=str, default=None, help="the path to pre-trained seq_pe. ")
    parser.add_argument("--seqpe_max_digits", type=int, default=-1, help="Maximum number of digits.")
    parser.add_argument("--seqpe_layer_num", type=int, default=2, help="Number of sequence position encoding layers.")
    
    parser.add_argument("--seqpe_last_layernorm", action="store_true", help="Whether to constrain the last layer's norm.")
    parser.add_argument("--seqpe_scale_attn_weights", type=bool, default=True, help="last layer's ln switch")
    parser.add_argument("--seqpe_attn_pdrop", type=float, default=0.1)
    parser.add_argument("--seqpe_resid_pdrop", type=float, default=0.1)
    parser.add_argument("--seqpe_decay", type=float, default=-1)
    parser.add_argument("--seqpe_lr", type=float, default=-1, help="learning_rate for seq_pe")
    parser.add_argument("--seqpe_temperature", type=float, default=1.0, help="a temperature is to scale the representation of seq_pe output by 1/seqpe_temperature")
    parser.add_argument("--seqpe_freeze_epoch_num", type=int, default=-1, help="if set, the pe model will not be trained before seqpe_freeze_epoch_num. This var starts from 1.")
    parser.add_argument("--seqpe_init_norm_weight", type=float, default=1, help="the default value for the weight of layernorm in the seq_pe model. We find that the l2 norm of the output PE representation has a big impact on the final performace. Thus, we wonder whether 1.0 is the best norm value for seq_pe model.")
    # parser.add_argument("--seqpe_norm_scale", type=float, default=-1, help="a temperature is to scale the representation of seq_pe output by 1/seqpe_temperature")

    parser.add_argument("--seqpe_activation_function", type=str, default="gelu_new", help="activation function selection")
    parser.add_argument("--seqpe_attn_direction", type=str, default="causal", choices=['causal', 'bi'], help="define the integrate way.")
    parser.add_argument("--seqpe_mask_padding", type=bool, default=False, help="Whether to mask_padding")
    parser.add_argument("--seqpe_add_out_proj", type=bool, default=True, help="add a out projection layer for seq_pe to adjust the scale")
    
    ## seqpe sampler
    parser.add_argument("--seqpe_data_size_multiplier", type=int, default=4, help="setting the pe dataset size by a multiplier to the size of the main task.")

    parser.add_argument("--seqpe_transfer_weight", type=float, default=0.1)
    parser.add_argument("--seqpe_transfer_beta", type=float, default=0.8)
    parser.add_argument("--seqpe_transfer_metric", type=str, default="kl_div", choices=['kl_div','mse'])
    parser.add_argument("--seqpe_transfer_batch_size", type=int, default=32)
    parser.add_argument("--seqpe_transfer_num", type=int, default=32)

    parser.add_argument("--seqpe_contrastive_weight", type=float, default=0.1)
    parser.add_argument("--seqpe_contrastive_batch_size", type=int, default=32, help="batch size of contrastive loss")
    parser.add_argument("--seqpe_contrastive_num", type=int, default=64, help="number of neg examples")

    parser.add_argument("--seqpe_warmup_steps", type=int, default=0, help="warmup for the weight of seq pe losses")

    ## wandb
    parser.add_argument("--use_wandb", action='store_true', help='use wandb to log results')
    parser.add_argument("--wandb_project_name", type=str, default="vit", help='name of the wandb project')
    parser.add_argument("--wandb_run_name", type=str, default="")

    ## legacy
    # parser.add_argument("--seqpe_epoch_shuffle", action="store_true", help="shuffle the data each epoch")

    args, unparsed = parser.parse_known_args()
    if args.seqpe_max_digits < 0:
        if args.pe_data_dim == 1:
            args.seqpe_max_digits = PeUtils.get_digit_num(args.pe_max_position)
        else:
            args.seqpe_max_digits = PeUtils.get_digit_num(int(np.sqrt(args.pe_max_position)))
    
    if args.num_attention_heads < 0:
        # will will assert that pe_embed_dim == model.embed_dim in main
        assert (args.pe_embed_dim % 64) == 0
        args.num_attention_heads = args.pe_embed_dim // 64
    
    if args.pe_type == "rotary":
        assert args.pe_apply_method in ["rotary", "rotary_2d", "rotary_mixed"]

    if args.pe_type == "vanilla":
        assert args.pe_apply_method == "input_add"

    if args.pe_type == "nope":
        assert args.pe_apply_method == "nope"

    if args.pe_type != 'seq_pe':
        if args.pe_apply_method in ["rotary_2d", "rotary_mixed"]:
            assert args.pe_data_dim == 2
        else:
            assert args.pe_data_dim == 1
    
    if args.use_pe_multi_head:
        if args.pe_apply_method == "input_add" or args.pe_apply_method == "nope" or args.pe_apply_method == "rotary":
            raise ValueError
        elif args.pe_type == "seq_pe" and not args.seqpe_add_out_proj:
            raise ValueError
    if args.use_pe_qk_per_layer == 'multi':
        assert args.pe_apply_method not in ['rotary', "rotary_2d", "rotary_mixed", 'input_add']

    # if args.seqpe_norm_scale > 0:
    #     assert args.seqpe_temperature == 1
    assert args.seqpe_temperature > 1e-2
    config = get_config(args)

    if config.PE_TYPE == "seq_pe" and config.MODEL.TYPE == "swin":
        assert config.USE_PE_QK_PER_LAYER and config.PE_APPLY_METHOD.startswith("attn")

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

    config.defrost()
    config.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    config.CODE_VERSION = get_git_commit_hash()
    config.freeze()

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.SEQPE_LR = linear_scaled_lr if config.SEQPE_LR < 0 else config.SEQPE_LR
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    if config.MODEL.TYPE == "swin" and config.PE_TYPE == "seq_pe":
        config.MODEL.SWIN.APE = False
        config.MODEL.SWIN.RPE = False
        logger.warning(f"Reseted APE and RPE to False because of the use of seq_pe")
    config.freeze()

    # print config
    logger.info(config.dump())

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    # dataset_train, data_loader_train, mixup_fn = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    if config.MODEL.TYPE != "swin":
        assert config.MODEL.NUM_HEADS == model.num_heads and config.MODEL.EMBED_DIM == model.embed_dim
    config.defrost()
    config.MODEL.DEPTH = model.depth
    config.freeze()
    pe_model = build_pe_model(config)
    input_shape = model.input_shape

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    model.cuda()
    logger.info(str(model))

    if pe_model is not None:
        pe_model.cuda()
        logger.info(str(pe_model))

    optimizer = build_optimizer(config, model, pe_model)
    logger.info("Successfully set optimizer")
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, 
        device_ids=[config.LOCAL_RANK], 
        broadcast_buffers=False,
        find_unused_parameters=True)
    logger.info("Successfully init DDP for model")

    if config.PE_TYPE == 'seq_pe' or config.PE_APPLY_METHOD == "rotary_mixed":
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

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_wo_ddp, pe_model_wo_ddp, optimizer, lr_scheduler, logger)
        
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
    use_random_shift = config.PE_USE_RANDOM_SHIFT if hasattr(config, 'PE_USE_RANDOM_SHIFT') else False
    pe_main_task_sampler = SeqPeMainTaskPosSampler(
        input_shape, config.PE_MAX_POSITION, config.SEQPE_MAX_DIGITS, data_dim=config.PE_DATA_DIM,
        add_cls_token_pe=add_cls_token_pe, device=device, use_random_shift=use_random_shift,
        random_shift_rate=config.PE_RANDOM_SHIFT_RATE, random_shift_downsample=config.PE_RANDOM_SHIFT_DOWNSAMPLE,
        start_epoch=config.TRAIN.START_EPOCH
    ) if config.PE_TYPE not in ["vanilla", "nope"] else None
    data_size_multiplier = args.seqpe_data_size_multiplier # intuitively set to increase data diversity by creating more random data
    pe_ct_sampler = SeqPeContrstiveDataSampler(
        len(data_loader_train) * data_size_multiplier, config.SEQPE_CONTRASTIVE_BATCH_SIZE,
        config.PE_MAX_POSITION, config.SEQPE_MAX_DIGITS, config.SEQPE_CONTRASTIVE_NUM, data_dim=config.PE_DATA_DIM,
        start_epoch=config.TRAIN.START_EPOCH, seed=seed
    ) if config.PE_TYPE == "seq_pe" else None
    pe_trans_sampler = SeqPeTransferDataSampler(
        input_shape, len(data_loader_train) * data_size_multiplier, config.SEQPE_TRANSFER_BATCH_SIZE, config.PE_MAX_POSITION,
        config.SEQPE_MAX_DIGITS, transfer_size=config.SEQPE_TRANSFER_NUM, data_dim=config.PE_DATA_DIM,
        start_epoch=config.TRAIN.START_EPOCH
    ) if config.PE_TYPE == "seq_pe" else None

    num_heads = config.NUM_ATTENTION_HEADS
    pe_ct_criterion = SeqPeContrastiveCriterion(num_heads=num_heads, seqpe_logit_scaled_loss=1.0, seqpe_multi_head_loss=False) if config.PE_TYPE == "seq_pe" else None
    pe_trans_criterion = SeqPeTransferCriterion(config.SEQPE_TRANSFER_BETA, config.SEQPE_TRANSFER_METRIC,num_heads=num_heads, seqpe_logit_scaled_loss=1.0, seqpe_multi_head_loss=True) if config.PE_TYPE == "seq_pe" else None

    pe_scheduler = PeWeightScheduler(config.SEQPE_CONTRASTIVE_WEIGHT, config.SEQPE_TRANSFER_WEIGHT, config.SEQPE_WARMUP_STEPS)
    if args.use_wandb and dist.get_rank() == 0:
        init_wandb(config, args.wandb_project_name, args.wandb_run_name)
    ####################################################

    # supervised criterion
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion_sup = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion_sup = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion_sup = torch.nn.CrossEntropyLoss()

    # self-supervised criterion
    criterion_ssup = cal_selfsupervised_loss
    # cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=-100)
    # mse_loss = MSELoss()
    max_accuracy = 0.0

    if config.MODEL.RESUME:
        acc1, acc5, loss = validate(config, data_loader_val, model, pe_model, pe_main_task_sampler, logger, epoch=None, use_wandb=False)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return
    
    if config.SEQPE_PRETRAINED:
        if config.MODEL.RESUME and config.MODEL.RESUME != config.SEQPE_PRETRAINED:
            logger.warning(
                f"The resumed model ({config.MODEL.RESUME}) is different from that in args.seqpe_pretrained ({config.SEQPE_PRETRAINED}), but only the args.seqpe_pretrained will be used! Make sure you understand what you are doing !!!"
            )
        load_pe_pretrained(config.SEQPE_PRETRAINED, pe_model_wo_ddp, logger)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()

    jigsaw_pullzer = JigsawPuzzleMaskedRegion(
        config.DATA.IMG_SIZE,
        config.TRAIN.PATCH_SIZE,
        mask_type=config.TRAIN.MASK_TYPE
    )
    init_lambda_dlocr = 0.0
    if config.TRAIN.USE_DLOCR and config.TRAIN.LAMBDA_DLOCR > 0:
        lambda_dlocr_schedule = utils.cosine_scheduler(
                base_value=config.TRAIN.LAMBDA_DLOCR,
                final_value=config.TRAIN.LAMBDA_DLOCR * 0.5,
                epochs=config.TRAIN.EPOCHS,
                warmup_epochs=config.TRAIN.WARMUP_EPOCHS)
    
    ####################################################
    # freeze seq_pe according to seqpe_freeze_epoch_num
    if config.PE_TYPE == "seq_pe" and config.SEQPE_FREEZE_EPOCH_NUM > config.TRAIN.START_EPOCH:
        for param in pe_model.parameters():
            param.requires_grad = False
        dist.barrier()
        logger.info(f"The pe_model has been frozen before the finish of {config.SEQPE_FREEZE_EPOCH_NUM}-th epoch")

    ####################################################


    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        if config.TRAIN.USE_DLOCR and config.TRAIN.LAMBDA_DLOCR > 0:
            init_lambda_dlocr = lambda_dlocr_schedule[epoch]

        jigsaw_pullzer._update_masking_generator(config.TRAIN.EPOCHS, epoch, config.TRAIN.MASK_RATIO)

        if epoch == config.SEQPE_FREEZE_EPOCH_NUM:
            for param in pe_model.parameters():
                param.requires_grad = True
            logger.info(f"The pe_model re-joint the training since {epoch}-th epoch")
            dist.barrier()

        train_one_epoch(
            config, 
            model,
            pe_model,
            criterion_sup, 
            criterion_ssup,
            data_loader_train, 
            optimizer, 
            epoch, 
            mixup_fn, 
            lr_scheduler, 
            logger,
            jigsaw_pullzer,
            init_lambda_dlocr,
            pe_main_task_sampler,
            pe_ct_sampler,
            pe_trans_sampler,
            pe_ct_criterion,
            pe_trans_criterion,
            use_wandb=args.use_wandb,
            pe_scheduler=pe_scheduler
        )

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_wo_ddp, pe_model_wo_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        acc1, acc5, loss = validate(config, data_loader_val, model, pe_model, pe_main_task_sampler, logger, epoch=epoch, use_wandb=args.use_wandb)

        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if dist.get_rank() == 0 and acc1 > max_accuracy:
            save_checkpoint_best(config, epoch, model_wo_ddp, pe_model_wo_ddp, max_accuracy, optimizer, lr_scheduler, logger)
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(
    config, 
    model,
    pe_model,
    criterion_sup,
    criterion_ssup, 
    data_loader, 
    optimizer, 
    epoch, 
    mixup_fn, 
    lr_scheduler, 
    logger,
    jigsaw_pullzer,
    lambda_dlocr,
    pe_main_task_sampler: SeqPeMainTaskPosSampler,
    pe_ct_sampler: SeqPeContrstiveDataSampler,
    pe_trans_sampler: SeqPeTransferDataSampler,
    pe_ct_criterion: SeqPeContrastiveCriterion,
    pe_trans_criterion: SeqPeTransferCriterion,
    use_wandb: bool = False,
    pe_scheduler: PeWeightScheduler = None,
):
    model.train()
    if pe_model is not None:
        pe_model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    main_task_meter = AverageMeter()
    batch_img_data_time = AverageMeter()
    batch_pe_data_time = AverageMeter()
    batch_fwd_time = AverageMeter()
    batch_bwd_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    pe_grad_norm_meter = AverageMeter()

    pe_ct_meter = AverageMeter()
    pe_trans_meter = AverageMeter()
    pe_norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    device = torch.device('cuda', dist.get_rank())

    end_time_tmp = time.time()
    pe_main_batch_size = config.PE_MAIN_BATCH_SIZE

    batch_img_data_start = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        batch_img_data_time.update(time.time() - batch_img_data_start)
        img_batch_size = samples.shape[0]

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        unk_mask = None
        if epoch >= config.TRAIN.WARMUP_EPOCHS:
            if config.TRAIN.USE_JIGSAW and torch.rand(1) > config.AUG.JIGSAW:
                samples, unk_mask = jigsaw_pullzer(samples)
                unk_mask = torch.from_numpy(unk_mask).long().to(device) if config.TRAIN.USE_UNK_POS else None
        ct_weight, transfer_weight = pe_scheduler.step()
        batch_pe_data_start = time.time()
        if config.PE_TYPE not in ["vanilla", "nope"]:
            cur_pe_main_batch_size = pe_main_batch_size if (img_batch_size%pe_main_batch_size) == 0 else 1
            pe_main_inputs = pe_main_task_sampler.next(device=device, batch_size=cur_pe_main_batch_size) if pe_main_task_sampler is not None else None
            cur_pe_main_batch_size = pe_main_inputs["batch_size"]
        else:
            cur_pe_main_batch_size = 1
        use_pe_ct = config.PE_TYPE == "seq_pe" and ct_weight > 0
        use_pe_transfer = config.PE_TYPE == "seq_pe" and transfer_weight > 0 and cur_pe_main_batch_size == 1
        batch_pe_data_end = time.time()
        if config.PE_TYPE == "seq_pe":
            if use_pe_ct and use_pe_transfer:
                pe_ct_inputs = pe_ct_sampler.next(device)
                pe_trans_inputs = pe_trans_sampler.next(device)
                pos_seq_data, pad_mask, sizes = PeUtils.merge_pe_data(pe_main_inputs, pe_ct_inputs, pe_trans_inputs)
            elif use_pe_ct:
                pe_ct_inputs = pe_ct_sampler.next(device)
                pos_seq_data, pad_mask, sizes = PeUtils.merge_pe_data(pe_main_inputs, pe_ct_inputs, None)
            elif use_pe_transfer:
                pe_trans_inputs = pe_trans_sampler.next(device)
                pos_seq_data, pad_mask, sizes = PeUtils.merge_pe_data(pe_main_inputs, None, pe_trans_inputs)
            else:
                pos_seq_data, pad_mask, sizes = PeUtils.merge_pe_data(pe_main_inputs, None, None)
            batch_pe_data_end = time.time()
            pe_out = pe_model(pos_seq_data, pad_mask)
            pe_splits = PeUtils.split_pe_data(pe_out, sizes)
            pe_main = pe_splits[0].reshape(cur_pe_main_batch_size, -1, pe_splits[0].shape[-1])
        elif config.PE_TYPE == "rotary" or config.PE_TYPE == "sin":
            pe_main = pe_model(pe_main_inputs['pos_ids'])
            if config.PE_APPLY_METHOD == "rotary_mixed":
                pe_main = pe_main.unsqueeze(1)
            else:
                pe_main = pe_main.reshape(cur_pe_main_batch_size, -1, pe_main.shape[-1])
        else:
            pe_main = None
        batch_pe_data_time.update(batch_pe_data_end-batch_pe_data_start)
        
        if pe_main is not None:
            pe_main_ = pe_main.clone().detach()
            if config.PE_APPLY_METHOD == "rotary_mixed":
                pe_main_ = torch.view_as_real(pe_main_).flatten(0, 2).flatten(-2, -1)
            pe_norm_meter.update(torch.mean(torch.norm(pe_main_, dim=-1)).item(), pe_main_.size(0) * pe_main_.size(1))

        fwd_time = time.time()
        outputs = model(samples, pe=pe_main, unk_mask=unk_mask)
        batch_fwd_time.update(time.time() - fwd_time)

        bwd_time = time.time()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion_sup(outputs.sup, targets)
            main_task_meter.update(loss.item(), targets.size(0))
            if config.TRAIN.USE_DLOCR:
                loss_ssup, ssup_items = criterion_ssup(outputs, config, lambda_dlocr)
                loss += loss_ssup
            
            if use_pe_ct:
                ct_loss = pe_ct_criterion(pe_splits[1], pe_splits[2], pe_ct_inputs['labels'])
                loss += ct_weight * ct_loss
                pe_ct_meter.update(ct_loss.item(), pe_splits[1].size(0))
            
            if use_pe_transfer:
                trans_loss = pe_trans_criterion(pe_main, config.SEQPE_TRANSFER_BATCH_SIZE, pe_trans_inputs['pivot_indices'], pe_splits[-1])
                loss += transfer_weight * trans_loss
                pe_trans_meter.update(trans_loss.item(), pe_splits[-1].size(0) / config.SEQPE_TRANSFER_BATCH_SIZE)

            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.PE_TYPE == "seq_pe":
                    if config.PE_CLIP_GRAD > 0:
                        pe_grad_norm = torch.nn.utils.clip_grad_norm_(pe_model.parameters(), config.PE_CLIP_GRAD)
                    else:
                        pe_grad_norm = get_grad_norm(pe_model.parameters())
                else:
                    pe_grad_norm = 0
                if config.TRAIN.CLIP_GRAD > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion_sup(outputs.sup, targets)
            main_task_meter.update(loss.item(), targets.size(0))
            if config.TRAIN.USE_DLOCR:
                loss_ssup, ssup_items = criterion_ssup(outputs, config, lambda_dlocr)
                loss += loss_ssup

            if use_pe_ct:
                ct_loss = pe_ct_criterion(pe_splits[1], pe_splits[2], pe_ct_inputs['labels'])
                loss += ct_weight * ct_loss
                pe_ct_meter.update(ct_loss.item(), pe_splits[1].size(0))
            
            if use_pe_transfer:
                trans_loss = pe_trans_criterion(pe_main, config.SEQPE_TRANSFER_BATCH_SIZE, pe_trans_inputs['pivot_indices'], pe_splits[-1])
                loss += transfer_weight * trans_loss
                pe_trans_meter.update(trans_loss.item(), pe_splits[-1].size(0) / config.SEQPE_TRANSFER_BATCH_SIZE)

            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.PE_TYPE == "seq_pe":
                    if config.PE_CLIP_GRAD > 0:
                        pe_grad_norm = torch.nn.utils.clip_grad_norm_(pe_model.parameters(), config.PE_CLIP_GRAD)
                    else:
                        pe_grad_norm = get_grad_norm(pe_model.parameters())
                else:
                    pe_grad_norm = 0
                if config.TRAIN.CLIP_GRAD > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        batch_bwd_time.update(time.time() - bwd_time)

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        pe_grad_norm_meter.update(pe_grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'|pe data time| {batch_pe_data_time.val:.4f} ({batch_pe_data_time.avg:.4f})\t'
                f'|img data time| {batch_img_data_time.val:.4f} ({batch_img_data_time.avg:.4f})\t'
                f'|model fwd time| {batch_fwd_time.val:.4f} ({batch_fwd_time.avg:.4f})\t'
                f'|model bwd & opt time| {batch_bwd_time.val:.4f} ({batch_bwd_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'main loss {main_task_meter.val:.4f} ({main_task_meter.avg:.4f})\t'
                f'pe_ct loss {pe_ct_meter.val:.4f} ({pe_ct_meter.avg:.4f})\t'
                f'pe_trans loss {pe_trans_meter.val:.4f} ({pe_trans_meter.avg:.4f})\t'
                f'pe_norm {pe_norm_meter.val:.4f} ({pe_norm_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'pe_grad_norm {pe_grad_norm_meter.val:.4f} ({pe_grad_norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            if config.TRAIN.USE_DLOCR:
                logger.info(f'weights: drloc {lambda_dlocr:.4f}')
                logger.info(f' '.join(['%s: [%.4f]' % (key, value) for key, value in ssup_items.items()]))

        end_time_tmp = time.time()
        batch_img_data_start = time.time()
    epoch_time = time.time() - start
    if use_wandb and dist.get_rank() == 0:
        wandb.log({
            'train/loss': loss_meter.avg,
            'train/pe_data_time': batch_pe_data_time.avg,
            'train/img_data_time': batch_img_data_time.avg,
            'train/model_fwd_time': batch_fwd_time.avg,
            'train/model_bwd_time': batch_bwd_time.avg,
            'train/main_loss': main_task_meter.avg,
            'train/pe_ct_loss': pe_ct_meter.avg,
            'train/pe_trans_loss': pe_trans_meter.avg,
            'train/pe_norm': pe_norm_meter.avg,
            'train/grad_norm': norm_meter.avg,
            'train/pe_grad_norm': pe_grad_norm_meter.avg,
        }, step=epoch)
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, pe_model, pe_main_task_sampler, logger, epoch=None, use_wandb=False):
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
    if config.PE_TYPE not in ["vanilla", "nope"]:
        pe_main_inputs = pe_main_task_sampler.next(device=device) if pe_main_task_sampler is not None else None
        pe_main_batch_size = pe_main_inputs["batch_size"]
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if config.PE_TYPE == "seq_pe":
            pos_seq_data, pad_mask = pe_main_inputs['pos_seq_data'], pe_main_inputs['pad_mask']
            pe_main = pe_model(pos_seq_data, pad_mask)
            pe_main = pe_main.reshape(pe_main_batch_size, -1, pe_main.shape[-1])
        elif config.PE_TYPE == "rotary" or config.PE_TYPE == "sin":
            pe_main = pe_model(pe_main_inputs['pos_ids'])
            if config.PE_APPLY_METHOD == "rotary_mixed":
                pe_main = pe_main.unsqueeze(1)
            else:
                pe_main = pe_main.reshape(pe_main_batch_size, -1, pe_main.shape[-1])
        else:
            pe_main = None

        if pe_main is not None:
            pe_main_ = pe_main.clone().detach()
            if config.PE_APPLY_METHOD == "rotary_mixed":
                pe_main_ = torch.view_as_real(pe_main_).flatten(0, 2).flatten(-2, -1)
            pe_norm_meter.update(torch.mean(torch.norm(pe_main_, dim=-1)).item(), pe_main_.size(0) * pe_main_.size(1))

        # compute output
        output = model(images, pe=pe_main)

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
        }, step=epoch)
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
