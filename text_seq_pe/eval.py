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
from accelerate.utils import set_seed
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
from transformers.custom_modules.build import build_pe_model,build_eval_pe_sampler

from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from yacs.config import CfgNode as CN
import json
from argparse import Namespace


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.51.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

STUDY_MODE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--eval_block_sizes",
        type=str,
        default="512,1024,2048,4096,8192",
    )
    parser.add_argument(
        "--eval_ckpt_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--eval_ckpt_name",
        type=str,
        default="best_model",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--eval_attn_method",
        type=str,
        default="sdpa",
        help="The sdpa is more memory efficient, but will cause minor change to the final results",
    )

    parser.add_argument('--eval_extrapolation_method', type=str, default="extend", choices=["interpolate", "extend"])
    parser.add_argument('--eval_interpolate_mode', type=str, default='none', choices=["linear", "bicubic", "bilinear", 'none'])
    parser.add_argument('--eval_start_pos', type=int, default=None)

    parser.add_argument("--use_wandb", action='store_true', help='use wandb to log results')
    parser.add_argument("--wandb_project_name", type=str, default="gpt", help='name of the wandb project')
    parser.add_argument("--wandb_run_name", type=str, default="", help='name of the wandb run')

    #################################
    args, unparsed = parser.parse_known_args()
    args.eval_interpolate_mode = None if args.eval_interpolate_mode == 'none' else args.eval_interpolate_mode

    with open(os.path.join(args.eval_ckpt_dir, "pe_config.json")) as fin:
        pe_config = CN.load_cfg(fin.read())

    with open(os.path.join(args.eval_ckpt_dir, "train_args.json")) as fin:
        train_args = Namespace(**json.load(fin))

    if pe_config.PE_TYPE == 'vanilla':
        assert args.eval_extrapolation_method == "interpolate" and args.eval_interpolate_mode == "linear"
    else:
        assert args.eval_extrapolation_method == "extend"

    return args, pe_config, train_args


def build_model_and_tokenizer(args, pe_config, max_position_embeddings=None, eval_attn_method=None):
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, trust_remote_code=True
    )
    config.pe_config = pe_config # allow the model to access pe config during traing
    if max_position_embeddings is not None:
        config.max_train_position_embeddings = config.max_position_embeddings
        config.max_position_embeddings = max_position_embeddings
        tokenizer.model_max_length = max_position_embeddings
    if eval_attn_method:
        config._attn_implementation = eval_attn_method
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name_or_path,
    #     config=config,
    #     trust_remote_code=True,
    # )
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return model, config, tokenizer


def group_texts(examples, block_size: int, stride: int, drop_last: bool, pad_value=-100):
    concatenated_examples = {k: list(chain(*v)) for k, v in examples.items()}
    input_ids = concatenated_examples['input_ids']
    attention_mask = concatenated_examples['attention_mask']
    labels = input_ids[1:] + [pad_value]
    total_length = len(input_ids)
    total_length = (total_length // block_size) * block_size if drop_last else total_length
    result = {'input_ids': [], 'attention_mask': [], 'labels': []}
    i = block_size
    while i < total_length:
        result['input_ids'].append(input_ids[i-block_size:i])
        result['labels'].append(labels[i-block_size:i])
        result['attention_mask'].append(attention_mask[i-block_size:i])
        i += stride

    if not drop_last and i > total_length:
        result['input_ids'].append(input_ids[i-stride:i])
        result['labels'].append(labels[i-stride:i])
        result['attention_mask'].append(attention_mask[i-stride:i])
    return result


def build_dataset(args):
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, trust_remote_code=True
        )
        assert "validation" in raw_datasets.keys() and "test" in raw_datasets.keys()
    else:
        assert args.validation_file is not None and args.train_file is not None
        extension = args.test_file.split(".")[-1]
        data_files, dataset_args = {}, {}
        data_files["test"] = args.test_file
        data_files["validation"] = args.validation_file
        extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)

    return raw_datasets


def process_dataset(accelerator, raw_datasets, tokenizer, args, eval_block_size=None):
    column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            batch_size=5000,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    block_size = args.block_size if eval_block_size is None else eval_block_size
    with accelerator.main_process_first():
        assert len(tokenized_datasets["validation"]) <= 5000 and len(tokenized_datasets["test"]) <= 5000
        eval_dataset = tokenized_datasets["validation"].map(
            partial(group_texts, block_size=block_size, stride=block_size, drop_last=False),
            batched=True,
            batch_size=5000,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        eval_dataset = eval_dataset.map(lambda example, idx: {"index": idx}, with_indices=True)
        test_dataset = tokenized_datasets["test"].map(
            partial(group_texts, block_size=block_size, stride=block_size, drop_last=False),
            batched=True,
            batch_size=5000,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        test_dataset = test_dataset.map(lambda example, idx: {"index": idx}, with_indices=True)

    return eval_dataset, test_dataset


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
    batch['input_ids'] = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
    batch['attention_mask'] = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    batch['labels'] = pad_sequence(labels, batch_first=True, padding_value=ignore_index)
    if indices is not None:
        batch['indices'] = indices
    batch['is_label_shifted'] = True # in our implementation, we shift the labels during data processing period
    return batch


@torch.no_grad()
def validate(accelerator: Accelerator, model, config, eval_dataloader, pe_model, pe_config, pe_main_task_sampler, eval_args, train_args, eval_block_size, **kwargs):
    model.eval()
    model_wo_ddp = accelerator.unwrap_model(model)
    if pe_model is not None:
        pe_model.eval()
    device = model.device
    ignore_index = -100
    if pe_config.PE_TYPE not in ["vanilla", "nope"]:
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

    if eval_args.eval_extrapolation_method == "interpolate" and pe_config.PE_TYPE != "nope":
        if pe_main is None and pe_config.PE_TYPE == "vanilla":
            # vanilla pe
            pe_main = model_wo_ddp.transformer.wpe.weight[:train_args.block_size]
            if eval_block_size > train_args.block_size:
                pe_main = PeUtils.full_interpolation(pe_main, train_args.block_size, eval_block_size, mode='linear')
            pe_main = pe_main.unsqueeze(0) # add dim for batch
        else:
            raise NotImplementedError

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
    if kwargs.get('word_num', None):
        result['word_ppl'] = math.exp(eval_loss * n / kwargs['word_num'])
    else:
        result['word_ppl'] = -1
    
    return result


def copy_directory(source_dir, destination_dir):
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)


# def load_pretrained_checkpoint(pretrain_dir, model, pe_model, model_file_name="model.safetensors", pe_file_name="model_1.safetensors"):
#     from safetensors.torch import load_file
#     model_state_dict = load_file(os.path.join(pretrain_dir, model_file_name)
#     model.load_state_dict(), strict=False)
#     if pe_model is not None:
#         pe_model.load_state_dict(load_file(os.path.join(pretrain_dir, pe_file_name)), strict=False)
#     torch.cuda.empty_cache()

def main():
    args, pe_config, train_args = parse_args()

    # init distributed training
    accelerator_log_kwargs = {}
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=train_args.mixed_precision, **accelerator_log_kwargs)

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
    set_seed(train_args.seed)

    # build pe model
    eval_block_sizes = [eval(s) for s in args.eval_block_sizes.split(",")]
    model, config, tokenizer = build_model_and_tokenizer(
        train_args, pe_config, max(eval_block_sizes), eval_attn_method=args.eval_attn_method
    )
    pe_model = build_pe_model(pe_config)

    # Handle the repository creation
    if accelerator.is_main_process:
        logger.info(f"Model\n{model}")
        logger.info(f"PE Model\n{pe_model}")
        if args.use_wandb:
            init_wandb(pe_config, args, args.wandb_project_name, args.wandb_run_name)
    accelerator.wait_for_everyone()

    # Prepare everything with our `accelerator`.
    model, pe_model = accelerator.prepare(
        model, pe_model
    )
    device = model.device
    if pe_model is not None:
        pe_model = pe_model.to(device)

    # Potentially load in the weights and states from a previous save
    if args.eval_ckpt_dir:
        resume_path = os.path.join(args.eval_ckpt_dir, args.eval_ckpt_name)
        accelerator.print(f"Resumed from checkpoint: {resume_path}")
        accelerator.load_state(resume_path)

    for eval_block_size in eval_block_sizes:
        # Preprocessing the datasets.
        raw_datasets = build_dataset(train_args)
        eval_dataset, test_dataset = process_dataset(accelerator, raw_datasets, tokenizer, train_args, eval_block_size)
        eval_word_num = get_word_num(eval_dataset, tokenizer)
        test_word_num = get_word_num(test_dataset, tokenizer)

        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=partial(data_collator, padding_value=tokenizer.eos_token_id),
            batch_size=args.eval_batch_size
        )
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=partial(data_collator, padding_value=tokenizer.eos_token_id),
            batch_size=args.eval_batch_size
        )
        eval_dataloader, test_dataloader = accelerator.prepare(eval_dataloader, test_dataloader)
        pe_main_task_sampler = build_eval_pe_sampler(
            pe_config, eval_block_size, add_cls_token_pe=False, start_pos=args.eval_start_pos
        )
        eval_start = time.time()
        eval_out = validate(accelerator, model, config, eval_dataloader, pe_model, pe_config, pe_main_task_sampler, word_num=eval_word_num, eval_args=args, train_args=train_args, eval_block_size=eval_block_size)
        eval_time = time.time() - eval_start
        test_start = time.time()
        test_out = validate(accelerator, model, config, test_dataloader, pe_model, pe_config, pe_main_task_sampler, word_num=test_word_num, eval_args=args, train_args=train_args, eval_block_size=eval_block_size)
        test_time = time.time() - test_start
        if accelerator.is_main_process:
            logger.info(
                f'Eval: [{eval_block_size}]\t'
                f'Time: [{datetime.timedelta(seconds=int(eval_time))}]\t'
                f"loss {eval_out['loss']:.4f}\t"
                f"token ppl {eval_out['token_ppl']:.4f}\t"
                f"token num {eval_out['token_num']}\t"
                f"word ppl {eval_out['word_ppl']:.4f}\t"
                f'word num {eval_word_num}'
            )
            logger.info(
                f'Test: [{eval_block_size}]\t'
                f'time: [{datetime.timedelta(seconds=int(test_time))}]\t'
                f"loss {test_out['loss']:.4f}\t"
                f"token ppl {test_out['token_ppl']:.4f}\t"
                f"token num {test_out['token_num']}\t"
                f"word ppl {test_out['word_ppl']:.4f}\t"
                f'word num {test_word_num}'
            )


if __name__ == "__main__":
    main()
