#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import datetime
import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import time

import datasets
import torch
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import shutil
import os
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import numpy as np
from transformers.custom_modules.pe_utils import PeUtils
from transformers.custom_modules.pe_sampler import SeqPeMainTaskPosSampler, SeqPeContrstiveDataSampler, SeqPeTransferDataSampler
from transformers.custom_modules.pe_criterion import SeqPeContrastiveCriterion, SeqPeTransferCriterion
from transformers.custom_modules.config import get_pe_config, save_pe_config
from transformers.custom_modules.utils import init_wandb, wandb, AverageMeter, get_grad_norm
from transformers.custom_modules.build import build_pe_model, build_pe_sampler_and_criterion

from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import json
from yacs.config import CfgNode as CN


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.51.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

STUDY_MODE = False

def parse_args():

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--pretrained_ckpt_path",
        type=str,
        default="",
        help="Path to pretrained model",
        required=False,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=-1, help="will be automatically set according to max_train_steps")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=80000,
        required=True,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=10086, help="A seed for reproducible training.")
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--attn_method",
        type=str,
        default="eager",
        help="eager training is more stable",
    )
    parser.add_argument(
        "--train_on_prompt", type=str2bool, default=False
    )
    parser.add_argument(
        "--eval_stride", type=int, default=-1
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=250,
        help="Whether the various states should be saved at the end of every n steps",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "bf16", "fp16"]
    )
    parser.add_argument(
        "--clip_grad", type=float, default=5.0, 
    )
    parser.add_argument(
        "--pretrained_dir", type=str, default="", required=True, help="model used for fine-tuning. we only load the weight of models. Cannot be set together with resume_from_checkpoint."
    )
    parser.add_argument(
        "--pretrained_ckpt", type=str, default="best_model"
    )
    parser.add_argument(
        "--pe_config_override", type=str, default="", help="model used for fine-tuning. we only load the weight of models. Cannot be set together with resume_from_checkpoint."
    )
    parser.add_argument(
        "--answer_loss_ratio", type=float, default=-1
    )

    ## wandb
    parser.add_argument("--use_wandb", action='store_true', help='use wandb to log results')
    parser.add_argument("--wandb_project_name", type=str, default="vit", help='name of the wandb project')
    parser.add_argument("--wandb_run_name", type=str, default="", help='name of the wandb run')
    #################################

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt", "jsonl"]:
                raise ValueError("`train_file` should be a csv, json, jsonl, or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt", "jsonl"]:
                raise ValueError("`validation_file` should be a csv, jsonl, or txt file.")
    args.eval_stride = args.block_size if args.eval_stride < 1 else args.eval_stride
    if args.pretrained_dir is None and args.resume_from_checkpoint is None:
        raise ValueError
    if args.pretrained_dir is not None and args.resume_from_checkpoint is not None:
        raise ValueError

    with open(os.path.join(args.pretrained_dir, "pe_config.json")) as fin:
        pe_config = CN.load_cfg(fin.read())
        if args.pe_config_override:
            pe_config.defrost()
            for k, v in json.loads(args.pe_config_override).items():
                orig_key = '' if k not in pe_config else pe_config[k]
                print(f"updating pe argument: {k} from {orig_key} -> {v}")
                if v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
                pe_config.merge_from_list([k, v])
            pe_config.freeze()

    return args, pe_config


def build_optimizer(model, pe_model, args, pe_config):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    model_no_decay = model.no_weight_decay_keywords()
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in model_no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in model_no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if pe_model is not None and pe_config.PE_TYPE == "seq_pe":
        pe_model_no_decay = pe_model.no_weight_decay_keywords()
        pe_parameters = [
            {
                "params": [p for n, p in pe_model.named_parameters() if not any(nd in n for nd in pe_model_no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in pe_model.named_parameters() if any(nd in n for nd in pe_model_no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_grouped_parameters = optimizer_grouped_parameters + pe_parameters
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer


def build_model_and_tokenizer(args, pe_config, max_position_embeddings=None, attn_method=None):
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, trust_remote_code=True
    )
    config.pe_config = pe_config # allow the model to access pe config during traing
    if max_position_embeddings is not None:
        config.max_position_embeddings = max_position_embeddings
        tokenizer.model_max_length = max_position_embeddings
    if attn_method:
        config._attn_implementation = attn_method
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name_or_path,
    #     config=config,
    #     trust_remote_code=True,
    # )
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return model, config, tokenizer


def build_dataset(args):
    assert args.validation_file is not None and args.train_file is not None
    extension = args.train_file.split(".")[-1]
    data_files, dataset_args = {}, {}
    data_files["train"] = args.train_file
    data_files["validation"] = args.validation_file
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
    elif extension == "jsonl":
        extension = "json"
    else:
        raise NotImplementedError
    raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)

    return raw_datasets


def process_dataset(accelerator, raw_datasets, tokenizer, args):
    column_names = raw_datasets["train"].column_names
    block_size = args.block_size

    def tokenize_function(examples, eos_idx, ignore_index=-100, train_on_prompt=False, is_gen=False):
        example_inputs = tokenizer([it.strip() for it in examples['input']])
        example_outputs = tokenizer([" " + it[0].strip() for it in examples['outputs']])
        n_ex = len(example_inputs['input_ids'])
        input_ids, index, raw_outputs, raw_input, labels, attention_mask, answer_mask = [], [], [], [], [], [], []
        for i in range(n_ex):
            cur_input, cur_output = example_inputs['input_ids'][i], example_outputs['input_ids'][i]
            cur_answer_mask = [0] * (len(cur_input)-1) + [1] * len(cur_output + [eos_idx])

            if train_on_prompt:
                cur_labels = example_inputs['input_ids'][i][1:] + cur_output + [eos_idx]
            else:
                cur_labels = [ignore_index] * (len(cur_input)-1) + cur_output + [eos_idx]
            
            if len(cur_labels) > block_size:
                continue
            index.append(examples['index'][i])
            raw_outputs.append(examples['outputs'][i])
            raw_input.append(examples['input'][i])
            if is_gen:
                input_ids.append(cur_input)
                attention_mask.append(example_inputs['attention_mask'][i])
            else:
                input_ids.append(cur_input + cur_output)
                answer_mask.append(cur_answer_mask)
                attention_mask.append(example_inputs['attention_mask'][i] + example_outputs['attention_mask'][i])
            labels.append(cur_labels)
        data = {
            'index': index,
            'raw_outputs': raw_outputs,
            'raw_input': raw_input,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
        if answer_mask:
            data['answer_mask'] = answer_mask
        return data

    with accelerator.main_process_first():
        train_dataset = raw_datasets['train'].map(
            partial(tokenize_function, eos_idx=tokenizer.eos_token_id, train_on_prompt=args.train_on_prompt),
            batched=True,
            batch_size=5000,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        eval_dataset = raw_datasets['validation'].map(
            partial(tokenize_function, eos_idx=tokenizer.eos_token_id, train_on_prompt=False),
            batched=True,
            batch_size=5000,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    return train_dataset, eval_dataset


def get_word_num(dataset, tokenizer):
    word_count = 0
    for i, it in enumerate(dataset):
        valid_token_num = sum(it['attention_mask'])
        input_ids = it['input_ids'][:valid_token_num]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        word_count += sum(1 for token in tokens if token.startswith("Ġ"))
        if i == 0 and not tokens[0].startswith("Ġ"):
            word_count += 1
    return word_count


def data_collator(features, ignore_index=-100, padding_value=0):
    first = features[0]
    batch = {}
    input_ids = [torch.LongTensor(it['input_ids']) for it in features]
    labels = [torch.LongTensor(it['labels']) for it in features]
    attention_mask = [torch.FloatTensor(it['attention_mask']) for it in features]
    indices = torch.LongTensor([it['index'] for it in features]) if 'index' in first else None
    batch['input_ids'] = pad_sequence(input_ids, batch_first=True, padding_value=padding_value, padding_side='left')
    batch['attention_mask'] = pad_sequence(attention_mask, batch_first=True, padding_value=0, padding_side='left')
    batch['labels'] = pad_sequence(labels, batch_first=True, padding_value=ignore_index, padding_side='left')
    if indices is not None:
        batch['indices'] = indices
    
    if 'answer_mask' in first:
        answer_mask = [torch.FloatTensor(it['answer_mask']) for it in features]
        batch['answer_mask'] = pad_sequence(answer_mask, batch_first=True, padding_value=0, padding_side='left')

    batch['is_label_shifted'] = True # in our implementation, we shift the labels during data processing period
    batch['raw_input'] = [it['raw_input'] for it in features]
    batch['raw_outputs'] = [it['raw_outputs'] for it in features]
    return batch


@torch.no_grad()
def validate(accelerator: Accelerator, model, config, eval_dataloader, pe_model, pe_config, pe_main_task_sampler, **kwargs):
    model.eval()
    if pe_model is not None:
        pe_model.eval()
    device = model.device
    ignore_index = -100
    if pe_config.PE_TYPE != "vanilla":
        pe_main_inputs = pe_main_task_sampler.next(device=device) if pe_main_task_sampler is not None else None
        pe_main_batch_size = pe_main_inputs["batch_size"]

    if pe_config.PE_TYPE == "seq_pe":
        pos_seq_data, pad_mask = pe_main_inputs['pos_seq_data'], pe_main_inputs['pad_mask']
        pe_main = pe_model(pos_seq_data, pad_mask)
        pe_main = pe_main.reshape(pe_main_batch_size, -1, pe_main.shape[-1])
    elif pe_config.PE_TYPE == "rotary" or pe_config.PE_TYPE == "sin" or pe_config.PE_TYPE == "alibi":
        pe_main = pe_model(pe_main_inputs['pos_ids'])
        if pe_config.PE_APPLY_METHOD == "rotary_mixed":
            pe_main = pe_main.unsqueeze(1)
        else:
            pe_main = pe_main.reshape(pe_main_batch_size, -1, pe_main.shape[-1])
    else:
        pe_main = None
    dataset_indices, losses, num_tokens = [], [], []
    for step, batch in enumerate(eval_dataloader):
        if pe_main is not None:
            batch['pe'] = pe_main
        indices = batch.pop('indices')
        B, N = batch['input_ids'].shape
        labels = batch.pop('labels')
        label_mask = torch.ne(labels, ignore_index) # real token -> 1, pad token -> 0
        outputs = model(**batch, return_dict=True)
        logits = outputs.logits
        logits = logits.view(-1, config.vocab_size)
        labels = labels.view(-1).to(logits.device)
        loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction='none')
        loss = loss.reshape(B, N) * label_mask.to(loss.dtype)
        loss = loss.sum(dim=-1)
        loss = accelerator.pad_across_processes(loss, pad_index=ignore_index)
        batch_tokens = accelerator.pad_across_processes(label_mask.sum(dim=-1), pad_index=0)
        indices = accelerator.pad_across_processes(indices, pad_index=ignore_index)
        losses.append(accelerator.gather_for_metrics(loss))
        dataset_indices.append(accelerator.gather_for_metrics(indices))
        num_tokens.append(accelerator.gather_for_metrics(batch_tokens))
    losses, dataset_indices, num_tokens = torch.cat(losses), torch.cat(dataset_indices), torch.cat(num_tokens)
    eval_loss, n = 0, 0
    visited_indices = set()
    for i in range(len(losses)):
        data_id = dataset_indices[i].item()
        if data_id != ignore_index and data_id not in visited_indices:
            eval_loss += losses[i]
            n += num_tokens[i]
            visited_indices.add(data_id)
    try:
        eval_loss = eval_loss / n
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    result = {
        'loss': eval_loss,
        'token_ppl': perplexity,
        'token_num': n
    }
    
    return result


def copy_directory(source_dir, destination_dir):
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)


def save_train_args(args, output_dir):
    train_args = vars(args)
    with open(os.path.join(output_dir, 'train_args.json'), 'w') as fout:
        json.dump(train_args, fout, indent="  ")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pretrained_checkpoint(pretrain_dir, model, pe_model, model_file_name="model.safetensors", pe_file_name="model_1.safetensors"):
    from safetensors.torch import load_file
    model.load_state_dict(load_file(os.path.join(pretrain_dir, model_file_name)), strict=False)
    if pe_model is not None:
        pe_model.load_state_dict(load_file(os.path.join(pretrain_dir, pe_file_name)), strict=False)
    torch.cuda.empty_cache()


def main():
    args, pe_config = parse_args()

    # init distributed training
    accelerator_log_kwargs = {}
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    seed = args.seed + accelerator.process_index
    set_seed(seed)

    # build pe model
    model, config, tokenizer = build_model_and_tokenizer(args, pe_config, attn_method=args.attn_method)
    pe_model = build_pe_model(pe_config)
    optimizer = build_optimizer(model, pe_model, args, pe_config)

    # Handle the repository creation
    if accelerator.is_main_process:
        logger.info(f"Model\n{model}")
        logger.info(f"PE Model\n{pe_model}")
        if args.use_wandb:
            init_wandb(pe_config, args, args.wandb_project_name, args.wandb_run_name)
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            config.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            save_train_args(args, args.output_dir)
            save_pe_config(pe_config, args.output_dir)
    accelerator.wait_for_everyone()

    # Preprocessing the datasets.
    raw_datasets = build_dataset(args)
    train_dataset, eval_dataset = process_dataset(accelerator, raw_datasets, tokenizer, args)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True,
        collate_fn=partial(data_collator, padding_value=tokenizer.eos_token_id),
        batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=partial(data_collator, padding_value=tokenizer.eos_token_id),
        batch_size=args.per_device_eval_batch_size
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )
    # Prepare everything with our `accelerator`.
    model, pe_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, pe_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    device = model.device
    if pe_model is not None:
        pe_model = pe_model.to(device)

    pe_main_task_sampler, pe_ct_sampler, pe_trans_sampler, pe_ct_criterion, pe_trans_criterion = build_pe_sampler_and_criterion(
        pe_config, args.block_size, len(train_dataloader), args=args, add_cls_token_pe=False, start_epoch=0, seed=seed
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    checkpointing_steps = args.checkpointing_steps

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_pe_model = accelerator.unwrap_model(pe_model)
    if args.pretrained_dir:
        load_pretrained_checkpoint(os.path.join(args.pretrained_dir, args.pretrained_ckpt), unwrapped_model, unwrapped_pe_model)

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        training_difference = os.path.splitext(path)[0]
        resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
        starting_epoch = resume_step // len(train_dataloader)
        completed_steps = resume_step // args.gradient_accumulation_steps
        resume_step -= starting_epoch * len(train_dataloader)

    ct_weight = pe_config.SEQPE_CONTRASTIVE_WEIGHT
    transfer_weight = pe_config.SEQPE_TRANSFER_WEIGHT
    use_pe_ct = pe_config.PE_TYPE == "seq_pe" and ct_weight > 0
    use_pe_transfer = pe_config.PE_TYPE == "seq_pe" and transfer_weight > 0
    pe_main_batch_size = pe_config.PE_MAIN_BATCH_SIZE

    best_ckpt = {"step": -1, "ppl": float("inf")}
    loss_meter, main_task_meter, norm_meter, batch_time, pe_ct_meter, pe_trans_meter, pe_norm_meter = [AverageMeter() for _ in range(7)]

    if STUDY_MODE:
        eval_out = validate(accelerator, model, config, eval_dataloader, pe_model, pe_config, pe_main_task_sampler)

    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        
        for step, batch in enumerate(active_dataloader):
            cur_batch_size = batch['input_ids'].shape[0]
            end = time.time()
            with accelerator.accumulate(model):
                if pe_config.PE_TYPE != "vanilla":
                    cur_pe_main_batch_size = pe_main_batch_size if (cur_batch_size%pe_main_batch_size) == 0 else 1
                    pe_main_inputs = pe_main_task_sampler.next(device=device, batch_size=cur_pe_main_batch_size) if pe_main_task_sampler is not None else None
                    cur_pe_main_batch_size = pe_main_inputs["batch_size"]
                if pe_config.PE_TYPE == "seq_pe":
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
                    
                    pe_out = pe_model(pos_seq_data, pad_mask)
                    pe_splits = PeUtils.split_pe_data(pe_out, sizes)
                    pe_main = pe_splits[0].reshape(cur_pe_main_batch_size, -1, pe_splits[0].shape[-1])
                elif pe_config.PE_TYPE == "rotary" or pe_config.PE_TYPE == "sin" or pe_config.PE_TYPE == "alibi":
                    pe_main = pe_model(pe_main_inputs['pos_ids'])
                    if pe_config.PE_APPLY_METHOD == "rotary_mixed":
                        pe_main = pe_main.unsqueeze(1)
                    else:
                        pe_main = pe_main.reshape(cur_pe_main_batch_size, -1, pe_main.shape[-1])
                else:
                    pe_main = None

                if pe_main is not None:
                    batch['pe'] = pe_main
                    pe_main_ = pe_main.clone().detach()
                    pe_norm_meter.update(torch.mean(torch.norm(pe_main_, dim=-1)).item(), pe_main_.size(0) * pe_main_.size(1))

                outputs = model(**batch, answer_loss_ratio=args.answer_loss_ratio)
                loss = outputs.loss
                main_task_meter.update(loss.item(), cur_batch_size)
                if use_pe_ct:
                    ct_loss = pe_ct_criterion(pe_splits[1], pe_splits[2], pe_ct_inputs['labels'])
                    loss = loss + ct_weight * ct_loss
                    pe_ct_meter.update(ct_loss.detach().item(), pe_splits[1].size(0))
                
                if use_pe_transfer:
                    trans_loss = pe_trans_criterion(pe_main, pe_config.SEQPE_TRANSFER_BATCH_SIZE, pe_trans_inputs['pivot_indices'], pe_splits[-1])
                    loss = loss + transfer_weight * trans_loss
                    pe_trans_meter.update(trans_loss.detach().item(), pe_splits[-1].size(0) / pe_config.SEQPE_TRANSFER_BATCH_SIZE)
                loss_meter.update(loss.detach().item(), cur_batch_size)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.clip_grad > 0:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.clip_grad)
                        if pe_config.PE_TYPE == "seq_pe":
                            _ = accelerator.clip_grad_norm_(pe_model.parameters(), args.clip_grad)
                    else:
                        accelerator.unscale_gradients()
                        grad_norm = get_grad_norm(model.parameters())
                    norm_meter.update(grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                batch_time.update(time.time() - end)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                completed_steps += 1

            if completed_steps % 100 == 0 and accelerator.sync_gradients:
                logger.info(
                    f'train: [{completed_steps}/{args.max_train_steps}]\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'main loss {main_task_meter.val:.4f} ({main_task_meter.avg:.4f})\t'
                    f'pe_ct loss {pe_ct_meter.val:.4f} ({pe_ct_meter.avg:.4f})\t'
                    f'pe_trans loss {pe_trans_meter.val:.4f} ({pe_trans_meter.avg:.4f})\t'
                    f'pe_norm {pe_norm_meter.val:.4f} ({pe_norm_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'mem {torch.cuda.max_memory_allocated() / (1024.0 * 1024.0):.0f}MB'
                )

            if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                eval_out = validate(accelerator, model, config, eval_dataloader, pe_model, pe_config, pe_main_task_sampler)
                output_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                accelerator.save_state(output_dir)
                if accelerator.is_main_process:
                    if eval_out['token_ppl'] < best_ckpt['ppl']:
                        logger.info(f"Found best model | step {completed_steps}\ttoken ppl: {eval_out['token_ppl']:.4f}")
                        copy_directory(output_dir, os.path.join(args.output_dir, "best_model"))
                        best_ckpt['ppl'] = eval_out['token_ppl']
                        best_ckpt['step'] = completed_steps
                    logger.info(
                        f'Eval: [{completed_steps}/{args.max_train_steps}]\t'
                        f'Time: [{datetime.timedelta(seconds=int(batch_time.sum))}]\t'
                        f"loss {eval_out['loss']:.4f}\t"
                        f"token ppl {eval_out['token_ppl']:.4f}\t"
                        f"token num {eval_out['token_num']}\t"
                        f"best token ppl {best_ckpt['ppl']:.4f}\t"
                    )

                if accelerator.is_main_process and args.use_wandb:
                    wandb.log(
                        {
                            "valid/token_ppl": eval_out['token_ppl'],
                            "train/loss": loss_meter.avg,
                            "train/main loss": main_task_meter.avg,
                            "train/grad_norm": norm_meter.avg,
                            "train/pe_ct": pe_ct_meter.avg,
                            "train/pe_trans": pe_trans_meter.avg,
                            "train/pe_norm": pe_norm_meter.avg,
                        },
                        step=completed_steps,
                    )
                loss_meter, main_task_meter, norm_meter, batch_time, pe_ct_meter, pe_trans_meter, pe_norm_meter = [AverageMeter() for _ in range(7)]
                model.train()
                if pe_model is not None:
                    pe_model.train()

            if completed_steps >= args.max_train_steps:
                break

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            save_pe_config(pe_config, args.output_dir)
            tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
