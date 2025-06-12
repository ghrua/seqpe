import os
import torch
import torch.distributed as dist
import numpy as np
import pdb
import subprocess
import yaml
import wandb

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


def init_wandb(config, args, proj_name, run_name=''):
    api_key = os.getenv("WANDB_API_KEY")
    job_info = get_slurm_job_info()
    config_json = yaml.safe_load(config.dump())
    config_json['args'] = {}
    for k, v in vars(args).items():
        config_json['args'][k] = v
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


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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