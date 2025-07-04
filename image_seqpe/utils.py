# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import numpy as np
import pdb
import subprocess
import yaml
import wandb

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def get_git_commit_hash():
    try:
        # Run the git command and capture its output
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).strip()
        # Decode from bytes to string
        return commit_hash.decode("utf-8")
    except subprocess.CalledProcessError as e:
        # Handle errors (e.g., not a git repository)
        return f"-1"
    except FileNotFoundError:
        # Handle the case where git is not installed
        return "-2"


def get_slurm_job_info():
    job_id = os.getenv("SLURM_JOB_ID")  # The unique job ID
    array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")  # The array job ID (same for all)
    task_id = os.getenv("SLURM_ARRAY_TASK_ID")  # The specific task ID
    return {"SLURM_JOB_ID": job_id, "SLURM_ARRAY_JOB_ID": array_job_id, "SLURM_ARRAY_TASK_ID": task_id}


def init_wandb(config, proj_name, run_name=''):
    api_key = os.getenv("WANDB_API_KEY")
    job_info = get_slurm_job_info()
    config_json = yaml.safe_load(config.dump())
    config_json['CODE_VERSION'] = get_git_commit_hash()
    config_json['SLURM_JOB_INFO'] = job_info
    wandb.login(key=api_key)
    wandb.init(
        # set the wandb project where this run will be logged
        project=proj_name,
        # track hyperparameters and run metadata
        config=config_json,
        name=run_name if run_name.strip() else None
    )


def load_pe_pretrained(seqpe_pretrained, pe_model, logger):
    logger.info(f"==============> Resuming SeqPE form {seqpe_pretrained}....................")
    # file = 'all_newgl_relative_rotary_text_64_32_real_1int_0.05ctt_loss_vit_s_16_16k_32_16_e-1transloss_0.25detach/ckpt_epoch_15.pth'
    if seqpe_pretrained.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            seqpe_pretrained, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(seqpe_pretrained, map_location='cpu', weights_only=False)
        # checkpoint = torch.load(file, map_location='cpu')
    msg = pe_model.load_state_dict(checkpoint['pe_model'], strict=False)
    logger.info(msg)
    del checkpoint
    torch.cuda.empty_cache()


def load_eval_checkpoint(config, model, pe_model, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    # file = 'all_newgl_relative_rotary_text_64_32_real_1int_0.05ctt_loss_vit_s_16_16k_32_16_e-1transloss_0.25detach/ckpt_epoch_15.pth'
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu', weights_only=False)
        # checkpoint = torch.load(file, map_location='cpu')
    if config.MODEL.TYPE == "swin":
        # skip buffer, whose size changes during extrapolation evaluation
        checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if "attn_mask" not in k}
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    if pe_model is not None:
        pe_msg = pe_model.load_state_dict(checkpoint['pe_model'], strict=False)
        logger.info(pe_msg)
    max_accuracy = checkpoint.get('max_accuracy', 0.0)
    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_checkpoint(config, model, pe_model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    # file = 'all_newgl_relative_rotary_text_64_32_real_1int_0.05ctt_loss_vit_s_16_16k_32_16_e-1transloss_0.25detach/ckpt_epoch_15.pth'
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu', weights_only=False)
        # checkpoint = torch.load(file, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    if pe_model is not None:
        pe_msg = pe_model.load_state_dict(checkpoint['pe_model'], strict=False)
        logger.info(pe_msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print('optimizer doesnt match')
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

def load_checkpoint_ft(pretrain_path, model, pe_model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Loading pretrained model form {pretrain_path}....................")
    ckpt = torch.load(pretrain_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model']
    pe_state_dict = ckpt['pe_model']
    own_state_dict = model.state_dict()
    own_pe_state_dict = pe_model.state_dict() if pe_model is not None else None
    for name, param in state_dict.items():
        if name in own_state_dict and "head" not in name:
            own_state_dict[name].copy_(param)
    if pe_state_dict is not None:
        for name, param in pe_state_dict.items():
            own_pe_state_dict[name].copy_(param)
    logger.info(f"=> loaded successfully")
    max_accuracy = 0.0
    del state_dict
    torch.cuda.empty_cache()
    return max_accuracy

def save_checkpoint(config, epoch, model, pe_model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'pe_model': pe_model.state_dict() if pe_model is not None else None,
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()
    
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def save_checkpoint_best(config, epoch, model, pe_model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'pe_model': pe_model.state_dict() if pe_model is not None else None,
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()
    
    save_path = os.path.join(config.OUTPUT, 'best_model.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs #* niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs
    return schedule
