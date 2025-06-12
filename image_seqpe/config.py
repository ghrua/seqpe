# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN
import pdb
_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 256
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 4 #8, it should be with a larger /dev/shm for using docker images
_C.DATA.PREFETCH_FACTOR = 2 #8, it should be with a larger /dev/shm for using docker images

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0
_C.MODEL.ATTN_DROP_RATE = 0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.EMBED_DIM = 384
_C.MODEL.NUM_HEADS = 6

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False # absolute position embedding
_C.MODEL.SWIN.RPE = True  # reletive position embedding
_C.MODEL.SWIN.PATCH_NORM = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300 
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 6e-4 ###!!! origin 5e-4 try 3e-4 & 7e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9


# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'
_C.AUG.REPEATED_AUG = False

_C.AUG.JIGSAW = 0.5
_C.TRAIN.PATCH_SIZE = 16
_C.TRAIN.USE_JIGSAW = False
_C.TRAIN.USE_UNK_POS = False
_C.TRAIN.USE_IDX_EMB = False
_C.TRAIN.USE_DLOCR = False
_C.TRAIN.LAMBDA_DLOCR = 0.0
_C.TRAIN.DLOCR_TYPE = "linear" # nonlinear, linear, pca
_C.TRAIN.USE_PCA = False
_C.TRAIN.MASK_RATIO=-1.0
_C.TRAIN.MASK_TYPE="mjp" # mjp, pps

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
_C.TEST.SEQUENTIAL = False

_C.EVAL = CN()
_C.EVAL.MODE = "none"
_C.EVAL.EPOCHS_SHIFT = 5
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 5
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    config.DATA.DATASET = args.dataset
    config.MODEL.NUM_CLASSES = args.num_classes
    # config.AUG.REPEATED_AUG = args.repeated_aug

    # merge from specific arguments
    if args.vit_drop_path_rate >= 0:
        config.MODEL.DROP_PATH_RATE = args.vit_drop_path_rate
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    config.MODEL.SWIN.APE    = args.ape
    config.MODEL.SWIN.RPE    = args.rpe

    config.TRAIN.EPOCHS       = args.total_epochs

    if args.eval_mode != "none": 
        config.EVAL.EPOCHS_SHIFT = args.epochs_shift
        config.EVAL.MODE    = args.eval_mode

    config.TRAIN.USE_JIGSAW = args.use_jigsaw
    config.TRAIN.PATCH_SIZE = args.patch_size
    config.TRAIN.USE_UNK_POS = args.use_unk_pos
    config.TRAIN.USE_IDX_EMB = args.use_idx_emb
    if args.lambda_dlocr > 0 and args.use_dlocr:
        config.TRAIN.USE_DLOCR  = True
    config.TRAIN.LAMBDA_DLOCR = args.lambda_dlocr
    config.TRAIN.DLOCR_TYPE = args.dlocr_type
    config.TRAIN.USE_PCA = args.use_pca
    config.TRAIN.MASK_RATIO = args.mask_ratio

    config.TRAIN.BASE_LR = args.base_lr
    # DDP
    # config.LOCAL_RANK = args.local_rank
    config.DATA.NUM_WORKERS = args.dataloader_num_workers
    config.DATA.PREFETCH_FACTOR = args.dataloader_prefetch_factor #8, it should be with a larger /dev/shm for using docker images

    # pe config
    config.PE_CLIP_GRAD = args.pe_clip_grad
    config.PE_EMBED_DIM = args.pe_embed_dim
    config.NUM_ATTENTION_HEADS = args.num_attention_heads
    config.PE_TYPE = args.pe_type
    config.PE_APPLY_METHOD = args.pe_apply_method
    config.PE_DATA_DIM = args.pe_data_dim
    config.PE_MAX_POSITION = args.pe_max_position
    config.PE_USE_RANDOM_SHIFT = args.pe_use_random_shift
    config.PE_RANDOM_SHIFT_RATE = args.pe_random_shift_rate
    config.PE_RANDOM_SHIFT_DOWNSAMPLE = args.pe_random_shift_downsample
    config.SINUSOIDAL_PE_BASE = args.sinusoidal_pe_base
    config.USE_PE_MULTI_HEAD = args.use_pe_multi_head
    config.USE_PE_QK_PER_LAYER = args.use_pe_qk_per_layer

    if args.pe_use_random_shift < 0:
        config.PE_MAIN_BATCH_SIZE = config.DATA.BATCH_SIZE
    else:
        config.PE_MAIN_BATCH_SIZE = args.pe_main_batch_size
    assert (config.DATA.BATCH_SIZE % config.PE_MAIN_BATCH_SIZE) == 0
    
    if config.PE_APPLY_METHOD == "input_add":
        out_proj_dim = config.MODEL.EMBED_DIM
    elif config.PE_APPLY_METHOD in ["rotary", "rotary_2d", "rotary_mixed"]:
        out_proj_dim = config.MODEL.EMBED_DIM // config.MODEL.NUM_HEADS
    elif config.PE_APPLY_METHOD.startswith("attn"):
        if config.USE_PE_MULTI_HEAD:
            # proj_dim = head_dim * head
            out_proj_dim = config.MODEL.EMBED_DIM
        else:
            # proj_dim = head_dim
            out_proj_dim = config.MODEL.EMBED_DIM // config.MODEL.NUM_HEADS
    elif config.PE_APPLY_METHOD == "nope":
        out_proj_dim = -1
    else:
        raise NotImplementedError
    config.PE_OUT_PROJ_DIM = out_proj_dim

    # seqpe config
    config.SEQPE_MAX_DIGITS = args.seqpe_max_digits
    config.SEQPE_LAYER_NUM = args.seqpe_layer_num
    config.SEQPE_LAST_LAYERNORM = args.seqpe_last_layernorm
    config.SEQPE_SCALE_ATTN_WEIGHTS = args.seqpe_scale_attn_weights
    config.SEQPE_ATTN_PDROP = args.seqpe_attn_pdrop
    config.SEQPE_RESID_PDROP = args.seqpe_resid_pdrop
    config.SEQPE_DECAY = args.seqpe_decay if args.seqpe_decay >= 0 else config.TRAIN.WEIGHT_DECAY
    config.SEQPE_LR = args.seqpe_lr
    config.SEQPE_ACTIVATION_FUNCTION = args.seqpe_activation_function
    config.SEQPE_ATTN_DIRECTION = args.seqpe_attn_direction
    config.SEQPE_MASK_PADDING = args.seqpe_mask_padding
    config.SEQPE_ADD_OUT_PROJ = args.seqpe_add_out_proj
    config.SEQPE_PRETRAINED = args.seqpe_pretrained
    config.SEQPE_TEMPERATURE = args.seqpe_temperature
    config.SEQPE_FREEZE_EPOCH_NUM = args.seqpe_freeze_epoch_num
    config.SEQPE_INIT_NORM_WEIGHT = args.seqpe_init_norm_weight
    config.SEQPE_LOGIT_SCALED_LOSS = args.seqpe_logit_scaled_loss
    config.SEQPE_MULTI_HEAD_LOSS = args.seqpe_multi_head_loss
    
    # seqpe transfer loss config
    config.SEQPE_TRANSFER_WEIGHT = args.seqpe_transfer_weight
    config.SEQPE_TRANSFER_BETA = args.seqpe_transfer_beta
    config.SEQPE_TRANSFER_METRIC = args.seqpe_transfer_metric
    config.SEQPE_TRANSFER_BATCH_SIZE = args.seqpe_transfer_batch_size
    config.SEQPE_TRANSFER_NUM = args.seqpe_transfer_num
    
    # seqpe contrastive loss config
    config.SEQPE_CONTRASTIVE_WEIGHT = args.seqpe_contrastive_weight
    config.SEQPE_CONTRASTIVE_BATCH_SIZE = args.seqpe_contrastive_batch_size
    config.SEQPE_CONTRASTIVE_NUM = args.seqpe_contrastive_num
    config.SEQPE_WARMUP_STEPS = args.seqpe_warmup_steps

    # legacy
    # config.SEQPE_EPOCH_SHUFFLE = args.seqpe_epoch_shuffle

    # output folder
    # pdb.set_trace()
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    config.OUTPUT = args.output
    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
