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
from accelerate.utils import set_seed, gather_object
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--eval_block_sizes",
        type=str,
        default="512,1024,2048,4096,8192",
    )
    parser.add_argument(
        "--eval_data_pref",
        type=str,
        default="./data/squad_qa/validation",
    )
    parser.add_argument(
        "--eval_data_ext",
        type=str,
        default="jsonl",
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
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval_attn_method",
        type=str,
        default="sdpa",
        help="The sdpa is more memory efficient, but will cause minor change to the final results",
    )

    parser.add_argument(
        "--eval_simple_qa",
        type=str2bool,
        default=False,
        help="whether to evalaute the simpler qa data",
    )

    parser.add_argument(
        "--eval_mode",
        type=str,
        default="ppl",
        choices=['ppl', 'gen'],
    )

    parser.add_argument(
        "--do_sample",
        type=str2bool,
        default=False,
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

    # assert args.eval_batch_size == 1
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


def build_dataset(eval_data_path):
    extension = eval_data_path.split(".")[-1]
    if extension == "txt":
        extension = "text"
    elif extension == "jsonl":
        extension = "json"
    else:
        raise NotImplementedError
    data_files, dataset_args = {}, {}
    data_files["validation"] = eval_data_path
    raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
    return raw_datasets


def process_gen_dataset(accelerator, raw_datasets, tokenizer, args):
    column_names = raw_datasets["validation"].column_names

    def tokenize_function(examples, eos_idx, ignore_index=-100, train_on_prompt=False, is_gen=False):
        example_inputs = tokenizer([it.strip() for it in examples['input']])
        example_outputs = tokenizer([" " + it[0].strip() for it in examples['outputs']])
        n_ex = len(example_inputs['input_ids'])
        input_ids, labels, attention_mask = [], [], []
        for i in range(n_ex):
            cur_input, cur_output = example_inputs['input_ids'][i], example_outputs['input_ids'][i]
            if train_on_prompt:
                cur_labels = example_inputs['input_ids'][i][1:] + cur_output + [eos_idx]
            else:
                cur_labels = [ignore_index] * (len(cur_input)-1) + cur_output + [eos_idx]

            if is_gen:
                input_ids.append(cur_input)
                attention_mask.append(example_inputs['attention_mask'][i])
            else:
                input_ids.append(cur_input + cur_output)
                attention_mask.append(example_inputs['attention_mask'][i] + example_outputs['attention_mask'][i])
            labels.append(cur_labels)
        return {
            'index': examples['index'],
            'raw_outputs': examples['outputs'],
            'raw_input': examples['input'],
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            partial(tokenize_function, eos_idx=tokenizer.eos_token_id, is_gen=True),
            batched=True,
            batch_size=5000,
            # num_proc=args.preprocessing_num_workers,
            num_proc=1,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    eval_dataset = tokenized_datasets["validation"]

    return eval_dataset


def process_ppl_dataset(accelerator, raw_datasets, tokenizer, args):
    column_names = raw_datasets["validation"].column_names

    def tokenize_function(examples, eos_idx, ignore_index=-100, train_on_prompt=False, is_gen=False):
        example_inputs = tokenizer([it.strip() for it in examples['input']])
        example_outputs = tokenizer([" " + it[0].strip() for it in examples['outputs']])
        n_ex = len(example_inputs['input_ids'])
        input_ids, labels, attention_mask = [], [], []
        for i in range(n_ex):
            cur_input, cur_output = example_inputs['input_ids'][i], example_outputs['input_ids'][i]
            if train_on_prompt:
                cur_labels = example_inputs['input_ids'][i][1:] + cur_output + [eos_idx]
            else:
                cur_labels = [ignore_index] * (len(cur_input)-1) + cur_output + [eos_idx]

            if is_gen:
                input_ids.append(cur_input)
                attention_mask.append(example_inputs['attention_mask'][i])
            else:
                input_ids.append(cur_input + cur_output)
                attention_mask.append(example_inputs['attention_mask'][i] + example_outputs['attention_mask'][i])
            labels.append(cur_labels)
        return {
            'index': examples['index'],
            'raw_outputs': examples['outputs'],
            'raw_input': examples['input'],
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            partial(tokenize_function, eos_idx=tokenizer.eos_token_id, is_gen=False),
            batched=True,
            batch_size=5000,
            # num_proc=args.preprocessing_num_workers,
            num_proc=1,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    eval_dataset = tokenized_datasets["validation"]

    return eval_dataset

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
    batch['is_label_shifted'] = True # in our implementation, we shift the labels during data processing period
    batch['raw_input'] = [it['raw_input'] for it in features]
    batch['raw_outputs'] = [it['raw_outputs'] for it in features]
    return batch


def dump_jsonl(data, fname):
    with open(fname, 'w') as fout:
        for it in data:
            fout.write(json.dumps(it) + "\n")


def is_simple_qa(outputs):
    for it in outputs:
        if len(it.strip().split()) > 3:
            return False
    return True


@torch.no_grad()
def validate_ppl(accelerator: Accelerator, model, config, eval_dataloader, pe_model, pe_config, pe_main_task_sampler, eval_args, train_args, eval_block_size, **kwargs):
    model.eval()
    if pe_model is not None:
        pe_model.eval()
    model_wo_ddp = accelerator.unwrap_model(model)
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

    if eval_args.eval_extrapolation_method == "interpolate":
        if pe_main is None:
            assert pe_config.PE_TYPE == "vanilla"
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
    
    return perplexity, n


@torch.no_grad()
def validate_gen(accelerator: Accelerator, model, config, eval_dataloader, pe_model, pe_config, pe_main_task_sampler, eval_args, train_args, eval_block_size, tokenizer, save_gen_to="", use_simple_qa=False, num_gen_samples=5, do_sample=False, **kwargs):
    model.eval()
    model_wo_ddp = accelerator.unwrap_model(model)
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

    if eval_args.eval_extrapolation_method == "interpolate":
        if pe_main is None:
            assert pe_config.PE_TYPE == "vanilla"
            # vanilla pe
            pe_main = model_wo_ddp.transformer.wpe.weight[:train_args.block_size]
            if eval_block_size > train_args.block_size:
                pe_main = PeUtils.full_interpolation(pe_main, train_args.block_size, eval_block_size, mode='linear')
            pe_main = pe_main.unsqueeze(0) # add dim for batch
        else:
            raise NotImplementedError

    dataset_indices, em_scores, all_generations, all_outputs, all_inputs = [], [], [], [], []
    for step, batch in enumerate(eval_dataloader):
        indices = batch.pop('indices')
        raw_input = batch.pop('raw_input')
        raw_outputs = batch.pop('raw_outputs')
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        B, N = batch['input_ids'].shape
        if do_sample:
            generations = [model_wo_ddp.generate(
                input_ids,
                max_new_tokens=16,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                num_beams=1,
                pe=pe_main,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id
            ) for _ in range(num_gen_samples)]
        else:
            generations = [model_wo_ddp.generate(
                input_ids,
                max_new_tokens=16,
                num_return_sequences=1,
                do_sample=False,
                num_beams=1,
                pe=pe_main,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id)]            
        for bi in range(B):
            all_outputs.append(raw_outputs[bi])
            all_inputs.append(raw_input[bi])
            multi_return_gen = []
            em = 0
            for ti in range(len(generations)):
                gen_text = tokenizer.decode(generations[ti][bi][N:], skip_special_tokens=True)
                multi_return_gen.append(gen_text)
                gen_text = gen_text.lower().strip()
                for output in raw_outputs[bi]:
                    if output.lower().strip() in gen_text:
                        em = 1
            all_generations.append(multi_return_gen)
            em_scores.append(em)
        dataset_indices += [it.item() for it in indices]
    all_generations = gather_object(all_generations)
    dataset_indices = gather_object(dataset_indices)
    all_outputs = gather_object(all_outputs)
    all_inputs = gather_object(all_inputs)
    em_scores = gather_object(em_scores)

    visited_indices = set()
    eval_em, n = 0, 0
    dump_gen_data = []
    for i in range(len(dataset_indices)):
        data_id = dataset_indices[i]
        if data_id != ignore_index and data_id not in visited_indices:
            if use_simple_qa and not is_simple_qa(all_outputs[i]):
                continue
            eval_em += em_scores[i]
            n += 1
            visited_indices.add(data_id)
            dump_gen_data.append(
                {"em_score": em_scores[i], "index": dataset_indices[i], "hyp": all_generations[i], "ref": all_outputs[i], 'input': all_inputs[i]}
            )
    if save_gen_to and accelerator.is_main_process:
        dump_jsonl(dump_gen_data, save_gen_to)
    try:
        eval_em = eval_em / n * 100
    except OverflowError:
        eval_em = 0
    return eval_em, n


def copy_directory(source_dir, destination_dir):
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)


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

    if accelerator.is_main_process:
        logger.info(f"Model\n{model}")
        logger.info(f"PE Model\n{pe_model}")
        if args.use_wandb:
            init_wandb(pe_config, args, args.wandb_project_name, args.wandb_run_name)
    accelerator.wait_for_everyone()

    model, pe_model = accelerator.prepare(
        model, pe_model
    )
    device = model.device
    if pe_model is not None:
        pe_model = pe_model.to(device)

    if args.eval_ckpt_dir:
        resume_path = os.path.join(args.eval_ckpt_dir, args.eval_ckpt_name)
        accelerator.print(f"Resumed from checkpoint: {resume_path}")
        accelerator.load_state(resume_path)
    else:
        resume_path = ""

    for eval_block_size in eval_block_sizes:
        pref_name = args.eval_data_pref.lstrip(".").replace("/", "-").strip("-")
        eval_data_path = f"{args.eval_data_pref}.{eval_block_size}.{args.eval_data_ext}"
        raw_datasets = build_dataset(eval_data_path)
        pe_main_task_sampler = build_eval_pe_sampler(
            pe_config, eval_block_size, add_cls_token_pe=False, start_pos=args.eval_start_pos
        )
        if args.eval_mode == "gen":
            ## generation evaluation
            eval_dataset = process_gen_dataset(accelerator, raw_datasets, tokenizer, train_args)
            eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=partial(data_collator, padding_value=tokenizer.eos_token_id),
                batch_size=args.eval_batch_size
            )
            eval_dataloader = accelerator.prepare(eval_dataloader)
            eval_start = time.time()
            if resume_path:
                save_gen_to = f"{resume_path}/{pref_name}.{eval_block_size}.gen.jsonl"
                logger.info(f"Generations will be saved to {save_gen_to}", main_process_only=True)
            else:
                save_gen_to = ""
            eval_em, eval_num = validate_gen(
                accelerator, model, config, eval_dataloader, pe_model, pe_config, 
                pe_main_task_sampler, eval_args=args, train_args=train_args,
                eval_block_size=eval_block_size, tokenizer=tokenizer,
                save_gen_to=save_gen_to, use_simple_qa=args.eval_simple_qa, do_sample=args.do_sample
            )
            eval_time = time.time() - eval_start
            if accelerator.is_main_process:
                logger.info(
                    f'Eval: [{eval_block_size}]\t'
                    f'Time: [{datetime.timedelta(seconds=int(eval_time))}]\t'
                    f"EM {eval_em:.4f}\t"
                    f"Eval Data Num {eval_num}\t"
                )
        elif args.eval_mode == "ppl":
            ## ppl evaluation
            eval_dataset = process_ppl_dataset(accelerator, raw_datasets, tokenizer, train_args)
            eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=partial(data_collator, padding_value=tokenizer.eos_token_id),
                batch_size=args.eval_batch_size
            )
            eval_dataloader = accelerator.prepare(eval_dataloader)

            eval_start = time.time()
            eval_ppl, eval_num = validate_ppl(
                accelerator, model, config, eval_dataloader, pe_model, pe_config, 
                pe_main_task_sampler, eval_args=args, train_args=train_args,
                eval_block_size=eval_block_size, tokenizer=tokenizer,
            )
            eval_time = time.time() - eval_start
            if accelerator.is_main_process:
                logger.info(
                    f'Eval: [{eval_block_size}]\t'
                    f'Time: [{datetime.timedelta(seconds=int(eval_time))}]\t'
                    f"eval_ppl {eval_ppl:.4f}\t"
                    f"Eval Data Num {eval_num}\t"
                )
        else:
            raise NotImplementedError



if __name__ == "__main__":
    main()
