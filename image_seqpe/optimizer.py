# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim


def build_optimizer(config, model, pe_model=None):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """

    parameters = set_weight_decay(model)
    if pe_model is not None and (config.PE_TYPE == "seq_pe" or config.PE_APPLY_METHOD == "rotary_mixed"):
        pe_parameters = set_pe_model_weight_decay(pe_model, config=config)
        parameters = parameters + pe_parameters

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    skip_list = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip_list = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        # print(f"{name} | min: {param.min()} | max: {param.max()}")
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def set_pe_model_weight_decay(pe_model, config):
    skip_list = {}
    skip_keywords = {}
    if hasattr(pe_model, 'no_weight_decay'):
        skip_list = pe_model.no_weight_decay()
    if hasattr(pe_model, 'no_weight_decay_keywords'):
        skip_keywords = pe_model.no_weight_decay_keywords()

    has_decay, has_decay_names = [], []
    no_decay, no_decay_names = [], []
    
    for name, param in pe_model.named_parameters():
        # print(f"{name} | min: {param.min()} | max: {param.max()}")
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_names.append(name)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
            has_decay_names.append(name)
    return [{'params': has_decay, 'weight_decay': config.SEQPE_DECAY, 'lr': config.SEQPE_LR},
            {'params': no_decay, 'weight_decay': 0., 'lr': config.SEQPE_LR}]

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin