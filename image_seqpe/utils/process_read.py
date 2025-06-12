import subprocess
import pdb
import argparse
from glob import glob
import time
import os
import shutil
import json
import random
def read_process(args, id):
    with open(f"log_file/output_{id}.txt", "r") as file:
        lines = file.readlines()

    # 去除空行，获取有内容的最后一行
    last_non_empty_line = None
    for line in reversed(lines):
        if line.strip():  # 检查行是否为空
            last_non_empty_line = line.strip()
            break
    message = f"{args.project_name} ID: {id} {last_non_empty_line}"
    curl_command = f'curl -X POST --data-urlencode "payload={{\\"channel\\": \\"shy-program-notification\\", \\"text\\": \\"{message}\\"}}" https://hooks.slack.com/services/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'
    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
    
def generate_torchrun_command(args, id, account_text):

    files_list = glob(f'{args.project_name}*/*')
    key_word = 'checkpoint'
    # 解析 project_name
    params = args.project_name.split('_')
    max_num = 0
    base_select = args.base_select
    for file in files_list:
        sep_file_name = file.split('/')[-1]
        if key_word in sep_file_name:
            num = int(sep_file_name.split('-')[-1])
            if num > max_num:
                max_num = num
    # 提取参数
    try:
        random_select_length, block_size = params[0].split('-')
        if '___' in random_select_length or '***' in random_select_length or '+++' in random_select_length:
            random_select_length = random_select_length[3:]
        relative_type = params[1][4:]
        pe_method = params[3]
        max_digits = params[4]
        ln_switch = params[5]
        seq_pe_layer_num = params[6]
        attention_direction = params[7]
        solo_ln_switch = params[8]
    except:
        block_size = 512
        pe_method = 'alibi'
        relative_type = 2
        max_digits = 5
        ln_switch = 'True'
        random_select_length = 10000
        attention_direction = 'single'
        seq_pe_layer_num = 2
        solo_ln_switch = 'True'
    dataset = args.dataset
    # 检查是否包含 ctt_loss
    contrastive_loss_switch = True if "ctt_" in args.project_name else False
    random_select = "True" if ('random_selectized' in args.project_name.lower() or 'random_select' in args.project_name.lower()) else "False"

    pad = 10
    if 'pad0' in args.project_name or '0pad' in args.project_name:
        pad = 0
    if args.visualization:
        block_size = 8192
    if pe_method == 'rotary':
        random_select == False
    # 生成命令
    # if contrastive_loss_switch:
        # random_select = True
    if 'rotary' in args.project_name:
        random_select = False
    if 'mse' in args.project_name:
        transfer_loss_select = 'mse'
    elif 'kl' in args.project_name:
        transfer_loss_select = 'kl'
    else:
        transfer_loss_select = 'void'
    
    smoothing_switch = True if 'smoothing' in args.project_name else False

    model_type = 'gpt2'
    if 'roformer' in args.project_name.lower():
        model_type = 'roformer'
    elif 'vit_' in args.project_name.lower():
        model_type = 'vit'
    elif 'swin' in args.project_name.lower():
        model_type = 'swin'
    # if dataset == 'needle':
    #     dataset = 'needle_injected_dataset'
    # else:
    #     dataset = 'wikitext'
    random_select_port_num = random.randint(2008, 20000)
    if args.steps != 0:
        max_num = args.steps
    if 'checkpoint' not in args.project_name:
        args.project_name = args.project_name + '_checkpoint'
    if 'subtractive' in args.project_name or 'subtract' in args.project_name:
        subtractive_switch = True
    else:
        subtractive_switch = False
    if model_type == 'vit' or model_type == 'swin':
        pe_method = args.pe_method
        root_path = '/project/nlp-work5/hongyu-s/gpt2_test/transformers/examples/pytorch/image-classification/'
        
            
        if args.debug:
            
            visualization_switch = '--visualization_switch True' if args.visualization else ''
            command = f"""
srun -p gpu_intr --gres=gpu:{args.GPU}:1 --pty {root_path}run_image_classification.py --model_type vit --dataset_name {args.dataset} \\
--output_dir outputs --remove_unused_columns False --image_column_name {args.image_column_name} --label_column_name {args.label_column_name} --resolution {args.resolution} --trust_remote_code {args.trust_remote_code} \\
--do_train --do_eval --learning_rate {args.lr_rate}  --lr_scheduler_type {args.lr_scheduler_type} --max_steps {args.max_steps} --max_steps 2500 --per_device_train_batch_size {args.batch_size} \\
--per_device_eval_batch_size {args.batch_size} --seq_pe_layer_num 2 --logging_strategy steps --logging_steps 250 --eval_strategy steps --eval_steps 250 --save_strategy steps --save_steps 250 --output_dir "{root_path}{args.project_name}" --logging_dir {root_path}{args.project_name}_log \\
--load_best_model_at_end True     --save_total_limit 10 --seed 42 --overwrite_output_dir --attention_direction single_direction --max_digits 3 --data_dim 2 {visualization_switch} \\
--pe_method {pe_method} --sample_range {args.sample_range} --warmup_steps {args.warmup_steps} --gradient_accumulation_steps {args.gradient_accumulation_steps}
"""
            print('\n\ngenerated command is: \n', command, '\n\n')
            exit()         
        else:
            resume_file = f'--resume_from_checkpoint ./{args.project_name}/checkpoint-{max_num}' if args.resume == 1 else ''
            # --max_steps {args.max_steps}
            training_num = f'--num_train_epochs {args.num_train_epochs}' if args.num_train_epochs else f'--max_steps {args.max_steps}'
            use_drloc = '--use-dlocr' if args.use_dlocr else ''
# --model_type vit --dataset_name {args.dataset} {resume_file} \\
# --output_dir outputs --remove_unused_columns False --image_column_name {args.image_column_name} --label_column_name {args.label_column_name} --resolution {args.resolution} --trust_remote_code {args.trust_remote_code} \\
# --do_train     --do_eval --learning_rate {args.lr_rate}  --lr_scheduler_type {args.lr_scheduler_type}  {training_num} --per_device_train_batch_size {args.batch_size} \\
# --per_device_eval_batch_size {args.batch_size}  --seq_pe_layer_num {args.seq_pe_layer_num}   --logging_strategy steps     --logging_steps 250    --eval_strategy steps --eval_steps 250 --save_strategy steps --save_steps 2500 --output_dir "{root_path}{args.project_name}" --logging_dir "{root_path}{args.project_name}"_log \\
# --load_best_model_at_end True     --save_total_limit 10 --seed 42 --overwrite_output_dir --attention_direction single_direction --max_digits 3 --data_dim 2 --pre_train_switch {args.pre_train_switch} \\
# --pe_method {pe_method} --dataloader_num_workers {args.dataloader_num_workers} --sample_range {args.sample_range} --warmup_steps {args.warmup_steps}  --pe_stop_step {args.pe_stop_step} --image_size {args.image_size} \\
# --weight_decay {args.weight_decay} --gradient_accumulation_steps {args.gradient_accumulation_steps} --baseline_add_switch {args.baseline_add_switch} \\
# --contrastive_loss_switch {contrastive_loss_switch} --ctt_loss_ratio {args.ctt_loss_ratio} --pivots_num {args.pivots_num} --contrastive_num_list_len {args.contrastive_num_list_len} --max_grad_norm 100.0   
# {use_drloc} \\     
            ct_loss_switch = '--ct_loss_switch True' if contrastive_loss_switch else ''      
            command = f"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA_DIR=./

IMG_SIZE=224 # 224, 384
MODE={args.model_type} # swintiny, cvt13, t2t, resnet50, vit
CONFIG={args.model_type} # swin_tiny_patch4_window7, cvt_13, t2tvit_14, resnet_50, vit_base_16
LAMBDA_DRLOC=0.01 # swin: 0.5, t2t: 0.1, cvt: 0.1
DRLOC_MODE=l1 # l1, ce, cbr

DATASET={args.dataset} # imagenet-100, imagenet, cifar-10, cifar-100, svhn, places365, flowers102, clipart, infograph, painting, quickdraw, real, sketch
NUM_CLASSES={args.num_class}

DISK_DATA=${{DATA_DIR}}/datasets/${{DATASET}}
TARGET_FOLDER=${{DATASET}}-${{MODE}}_24only_seq_pe
SAVE_DIR=${{DATA_DIR}}/swintransformer-expr/${{TARGET_FOLDER}}
python -m torch.distributed.launch \\
--nproc_per_node={args.per_node} \\
--nnodes 1 \\
--node_rank 0 \\
--master_port {random_select_port_num} main.py \\
--cfg ./configs/${{CONFIG}}_${{IMG_SIZE}}.yaml \\
--dataset ${{DATASET}} \\
--num_classes ${{NUM_CLASSES}} \\
--data-path ${{DISK_DATA}} \\
--batch-size {args.batch_size} \\
--accumulation-steps {args.gradient_accumulation_steps} \\
--output {'_'.join(args.project_name.split('_')[:-1])} \\
--lambda_dlocr ${{LAMBDA_DRLOC}} \\
--dlocr_mode nonlinear \\
--mask-ratio 0.1 \\
--use_abs \\
--data_dim {args.data_dim} \\
--mask_use {args.mask_use} \\
--pe_method {args.pe_method} \\
--pe_mode {args.pe_mode} \\
--seq_pe_layer_num {args.seq_pe_layer_num} \\
--ct_ratio {args.ct_ratio} \\
--PE_embd_dim {args.PE_embd_dim} \\
--sample_range {args.sample_range} \\
--alpha {args.alpha} \\
--pad_value {args.pad_value} \\
--interval_range {args.interval_range} \\
--pivot_batch_size {args.pivot_batch_size} \\
--save_epoch_num {args.save_epoch_num} \\
--contrastive_list_length {args.contrastive_list_length} \\
--seq_pe_decay {args.seq_pe_decay} \\
--max_digits {args.max_digits} \\
--attn_pdrop {args.dropout_rate} \\
--resid_pdrop {args.dropout_rate} \\
--pe_dropout_rate {args.dropout_rate} \\
--seq_pe_lr {args.seq_pe_lr} \\
--random_length {args.random_length} \\
--reg_factor {args.reg_factor} \\
--kl_batch_size {args.kl_batch_size} \\
--kl_corresp_list_len {args.kl_corresp_list_len} \\
--transfer_main_ratio {args.transfer_main_ratio} \\
--transfer_ratio {args.transfer_ratio} \\
--contrastive_position_obtain_method {args.contrastive_position_obtain_method} \\
{ct_loss_switch}
"""
        print('\n\ngenerated command is: \n' + command, '\n\n')
        return command
      
    if len(args.PE_pretrain_file) != 0:
        command = f"""
torchrun --nproc_per_node={args.per_node} ./run_clm.py --model_type {model_type} --tokenizer_name gpt2 \\
--learning_rate {args.lr_rate} --per_device_train_batch_size {args.batch_size} --per_device_eval_batch_size {args.batch_size} --block_size {block_size} --dataset_name {dataset} \\
--dataset_config_name wikitext-103-raw-v1 --do_train --do_eval --overwrite_output_dir --num_train_epochs 30 --output_dir "./{args.project_name}" \\
--evaluation_strategy steps --logging_dir ./{args.project_name}_log --logging_steps 200 --save_steps 2000 --random_select {random_select} --random_select_length {random_select_length} \\
--visualization_switch {args.visualization} --visualization_file_name {args.project_name} \\
--contrastive_loss_switch {contrastive_loss_switch} --temperature 1 --ctt_loss_ratio {args.ctt_loss_ratio} --hard_negative_switch {args.hard_negative_switch} --requires_grad {args.requires_grad} \\
--PE_pretrain_file {args.PE_pretrain_file} --PE_training_start_step {args.PE_training_start_step} \\
--pe_method {pe_method} --relative_type {relative_type} --max_digits {max_digits} --ln_switch {ln_switch} --seq_pe_layer_num {seq_pe_layer_num} --relative_attention_num_buckets {args.relative_attention_num_buckets} \\
--attention_direction {attention_direction}_direction --solo_ln_switch {solo_ln_switch} --pe_dropout_rate 0.1 --pad_value {pad} --smoothing_switch {smoothing_switch} --kl_beta {args.kl_beta} --report_to none
"""
    elif 'res50' in args.project_name:
        root_path = '/project/nlp-work5/hongyu-s/gpt2_test/transformers/examples/pytorch/image-classification/'
        command = f""" 
torchrun --nproc_per_node={args.per_node}  /project/nlp-work5/hongyu-s/gpt2_test/transformers/examples/pytorch/image-classification/run_image_classification.py --model_type resnet --dataset_name {args.dataset} \\
--output_dir outputs --remove_unused_columns False --image_column_name {args.image_column_name} --label_column_name {args.label_column_name} --resolution {args.resolution} --trust_remote_code {args.trust_remote_code} \\
--do_train     --do_eval --learning_rate {args.lr_rate}  --lr_scheduler_type {args.lr_scheduler_type}  --num_train_epoch {args.num_train_epochs} --per_device_train_batch_size {args.batch_size} \\
--per_device_eval_batch_size {args.batch_size}  --seq_pe_layer_num 2   --logging_strategy steps     --logging_steps 200     --eval_strategy steps --eval_steps 200 --save_strategy steps --save_steps 1000 --output_dir "{root_path}{args.project_name}" --logging_dir "{root_path}{args.project_name}"_log \\
--load_best_model_at_end True     --save_total_limit 10 --seed 42 --overwrite_output_dir --attention_direction single_direction --max_digits 3 --data_dim 2 \\
--pe_method {pe_method} --dataloader_num_workers {args.dataloader_num_workers} --sample_range {args.sample_range} --warmup_ratio {args.warmup_ratio} --model_name_or_path microsoft/resnet-50
"""             
    print()
    print()
    print('generated command is: \n', command)
    print()
    print()
    return command
# 要执行的命令
def make_tansorboard_record(args, id):
    while 1:
        fname = args.project_name + '_checkpoint' if "checkpoint" not in args.project_name else args.project_name
        files_list = glob(f'{fname}/checkpoint*')
        key_word = 'checkpoint'
        max_num = 0
        
        for file in files_list:
                
            sep_file_name = file.split('/')[-1]
            if key_word in sep_file_name:
                num = int(sep_file_name.split('-')[-1])
                if num > max_num:
                    max_num = num
        if args.steps != 0:
            max_num = args.steps
        name = f'{fname}/{key_word}-{max_num}'
        # name = f'{fname}/{key_word}-12500'
        # print(f'Checking {name}')
        # if count == 0:
        #     name = f'{fname}/{key_word}-{39000}'
        # else:
        #     name = f'{fname}/{key_word}-{107550}'
        try:
            os.makedirs(name, exist_ok = True)
            with open(f'{name}/trainer_state.json', 'r') as f:
                trainer_state = json.load(f)
                log_history = trainer_state['log_history']
            jj = fname.split('/')[-1]
            if os.path.isdir(f'runs/{jj}'):
                shutil.rmtree(f'runs/{jj}')
            writer = SummaryWriter(f'runs/{jj}')
            for log in log_history:
                epoch = log.get('epoch', 0.0)  # 确保epoch是浮点数
                if not isinstance(epoch, float):
                    epoch = float(epoch)  # 将epoch转换为浮点数
                epoch_step = epoch * 1000 
                step = log.get('step', 0)
                for key, value in log.items():
                    if key not in ['epoch', 'step']:
                        writer.add_scalar(f'{key}', value, epoch_step)  # 使用浮点数epoch作为横坐标

            # count += 1
            
            print(f'Successfully processed {name}')
            writer.close()
            time.sleep(1200)
            read_process(args, id)
        except:
            time.sleep(1200)
            read_process(args, id)
def generate_parser():
    parser = argparse.ArgumentParser(description="Generate torchrun command based on project name.")
    parser.add_argument('--project_name', type=str, required=True, help='Project name containing parameters.')
    parser.add_argument('--visualization', type=bool, default=False, help='Include visualization flag in the command.')
    parser.add_argument('--test', type=bool, default=False, help='Include test flag in the command.')
    parser.add_argument('--resume', type=bool, default=False, help='Include test flag in the command.')
    parser.add_argument('--GPU', type=str, default='', help='specific GPU requirment')
    parser.add_argument('--dataset', type=str, default='', help='dataset')
    parser.add_argument('--num_pre_train', type=bool, default=False, help='dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Include test flag in the command.')
    parser.add_argument('--base_select', type=int, default=10000, help='Include test flag in the command.')
    parser.add_argument('--PE_training_start_step', type=int, default= -1, help='the beginning step of the training of the PE.')
    parser.add_argument('--kl_beta', type=float, default=1.0, help='kl_beta')
    parser.add_argument('--PE_pretrain_file', type=str, default='', help='addr of the PE parameters')
    parser.add_argument('--requires_grad', type=bool, default=False, help='require grads')
    parser.add_argument('--ctt_loss_ratio', type=float, default=0.1, help='beta for contrastive loss')
    parser.add_argument('--per_node', type=int, default=4, help='grab n GPU per node')
    parser.add_argument('--debug', type=bool, default=False, help='debug mode to generate srun instruction')
    parser.add_argument('--get_grad', type=bool, default=False, help='switch to get grad')
    parser.add_argument('--lr_rate', type=str, default='1e-4', help='learning rate')
    parser.add_argument('--model_name_or_path', type=str, default='', help='pre_train_file')
    parser.add_argument('--alpha', type=float, default=1.0, help='exp distribution parameter')
    parser.add_argument('--steps', type=int, default=0, help='mannully setted steps')
    parser.add_argument('--entire_test', type=bool, default=False, help='switch that controls the steps of checkpoints')
    parser.add_argument('--interpolation', type=bool, default=False, help='interpolation')
    parser.add_argument('--PE_freeze_switch', type=bool, default=False, help='switch of the pe freeze')
    parser.add_argument('--hard_negative_switch', type=bool, default=False, help='control the hard negatives')
    parser.add_argument('--shift', type=bool, default=False, help='position random_select shift switch')
    parser.add_argument('--test_start_step', type=int, default=0, help='ignore the checkpoint before the setted steps')
    parser.add_argument('--test_end_step', type=int, default=10000000000000, help='ignore the checkpoint after the setted steps')
    parser.add_argument('--max_position_embeddings', type=int, default=8192, help='max_position_embeddings for roformer')
    parser.add_argument('--pivots_num', type=int, default=512, help='max_position_embeddings for roformer')
    parser.add_argument('--KL_batch_size', type=int, default = 16, help='max_position_embeddings for roformer')
    parser.add_argument('--contrastive_num_list_len', type=int, default = 32, help='contrastive_num_list_len')
    parser.add_argument('--l2_lambda', type=float, default=0.01, help='L2 penalty')
    parser.add_argument('--warmup_steps', type=int, default=0, help='L2 penalty')
    parser.add_argument('--subtractive_ratio', type=float, default=0.1, help='subtractive ratio')
    parser.add_argument('--relative_attention_num_buckets', type=int, default = 32, help='number of buckets for T5 position encoding')
    parser.add_argument('--detach', type=bool, default=False, help='')
    parser.add_argument('--extra_params', type=str, default='', help='')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='')
    parser.add_argument('--pe_method', type=str, default='relative', help='')
    parser.add_argument('--contrastive_position_obtain_method', type=str, default='text', help='')
    parser.add_argument('--pe_mode', type=str, default='rotary', help='')
    parser.add_argument('--model_type', type=str, default='vit_s_16', help='')
    parser.add_argument('--resolution', type=int, default=224, help='resolution of the image')
    parser.add_argument('--pe_stop_step', type=int, default=224, help='resolution of the image')
    parser.add_argument('--num_train_epochs', type=int, default=0, help='resolution of the image')
    parser.add_argument('--dataloader_num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--max_steps', type=int, default=0, help='max_steps')
    parser.add_argument('--image_size', type=int, default=224, help='max_steps')
    parser.add_argument('--sample_range', type=int, default=100, help='beginning of the position')
    parser.add_argument('--image_column_name', type=str, default='image', help='image_column_name')
    parser.add_argument('--label_column_name', type=str, default='label', help='image_column_name')
    parser.add_argument('--trust_remote_code', type=bool, default=False, help='')
    parser.add_argument('--mask_use', type=bool, default=False, help='')
    parser.add_argument('--pre_train_switch', type=bool, default=False, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--baseline_add_switch', type=bool, default=False, help='')
    parser.add_argument('--use_dlocr', type=bool, default=False, help='')
    parser.add_argument('--seq_pe_layer_num', type=int, default=1, help='')
    parser.add_argument('--num_class', type=int, default=100, help='')
    parser.add_argument('--data_dim', type=int, default=2, help='')
    parser.add_argument('--ct_ratio', type=float, default=0.01, help='ct ratio')
    parser.add_argument('--PE_embd_dim', type=int, default=64, help='')
    parser.add_argument('--interval_range', type=int, default=1, help='interval range')
    parser.add_argument('--pivot_batch_size', type=int, default=128, help='pivot_batch_size')
    parser.add_argument('--pad_value', type=int, default=0, help='pad_value')
    parser.add_argument('--contrastive_list_length', type=int, default=128, help='contrastive_list_length')
    parser.add_argument('--save_epoch_num', type=int, default=128, help='save_epoch_num')
    parser.add_argument('--max_digits', type=int, default=5, help='max digits length')
    parser.add_argument('--random_length', type=int, default=900, help='random_length')
    parser.add_argument('--kl_corresp_list_len', type=int, default=16, help='kl_corresp_list_len')
    parser.add_argument('--kl_batch_size', type=int, default=32, help='kl_batch_size')
    parser.add_argument("--seq_pe_decay", type=float, default=0)
    parser.add_argument("--reg_factor", type=float, default=0)
    parser.add_argument("--dropout_rate", type=float, default=0)
    parser.add_argument("--seq_pe_lr", type=float, default=1e-4)
    parser.add_argument("--transfer_ratio", type=float, default=0.01)
    parser.add_argument("--transfer_main_ratio", type=float, default=0.25)
    
    

    return parser.parse_args()


def get_valuable_GPU(args):

    command = "bash /project/nlp-work5/hongyu-s/gpt2_test/transformers/examples/pytorch/language-modeling/GPU_info.sh"


    # 执行命令并获取返回结果
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    data = result.stdout.split('\n')
    result = {}
    # 遍历数据列表
    for item in data:
        
        # 检查是否包含a6000或a100
        if args.GPU:
            if args.GPU in item:
                # 使用split()方法获取需要的部分
                parts = item.split()
                if parts:
                    # 提取第一个部分（比如elm72）
                    result[parts[0]] = parts[1]            
        else:
            parts = []
            # if 'a6000' in item or 'a100' in item or '6000' in item or 'v100' in item:
            if args.test:
                if ('a6000' in item or 'a100' in item or '6000' in item or 'a100-80' in item):
                    parts = item.split()
                    if parts:
                        # 提取第一个部分（比如elm72）
                        result[parts[0]] = parts[1]                    
            else:
                if ('a6000' in item or 'a100' in item or '6000' in item or 'a100-80' in item):
                # if 'a6000' in item:
                    # 使用split()方法获取需要的部分
                    parts = item.split()
                    if parts:
                        # 提取第一个部分（比如elm72）
                        result[parts[0]] = parts[1]
            # if len(parts) == 0:
            #     if 'a6000' in item or 'a100' in item or '6000' in item or '3090' in item or 'v100' in item:
            #         parts = item.split()
            #         if parts:
            #             # 提取第一个部分（比如elm72）
            #             result[parts[0]] = parts[1]                    
            # pdb.set_trace()
            # try:
            #     if not parts:
            #         pass
            # except:

            #     if '3090' in item or 'v100' in item:
            #         parts = item.split()
            #         if parts:
            #             # 提取第一个部分（比如elm72）
            #             result[parts[0]] = parts[1]
    return result

def run_bash_file(args):
    # pdb.set_trace()
    command = f"sbatch {args.project_name}.sh"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # pdb.set_trace()
    try:
        id = result.stdout.split()[-1]
    except:
        print(result)
    print(result.stdout)
    return id
# pdb.set_trace()
args = generate_parser()

GPU_dict = get_valuable_GPU(args)
gress = None
account_text = ''
partion = 'gpu_long'

for key, value in GPU_dict.items():
    if 'lang' in key:
        partion = 'lang_gpu_long'
        account_text = '#SBATCH --account=lang'
        # account_text = '--account lang'
    try:
        if int(value.split(':')[-1]) < args.per_node:
            print(f'{value.split(":")[-1]} valuable GPU is less than your expectation!!!!!!!! ')
    except:
        gress = 'gpu:6000:4'
    gress = value
    if args.GPU == 'a100' and 'a100-80' in value:
        continue
    # node = key   
    break

if gress is None:
    gress = "gpu:a6000:4"
gress = gress.split('(')[0]
gress = gress[:-1] 
gress = gress + '4'
if args.test:
    show_result_in_excel_form = 'python ./log_file/info_extraction.py --id {}'
command = generate_torchrun_command(args, id, account_text)

initial_start_step = f'_start_from_{args.test_start_step}' if args.test_start_step else ''
specific_step = f'_step{args.steps}' if args.steps else ''
record_name = f"{args.project_name}{initial_start_step}{specific_step}_test" if args.test else f"{args.project_name}"
ID = '${SLURM_JOB_ID}'
record_command = f"python log_file/info_extraction.py --project_name {record_name}_{ID}" if args.test else ""
# gress = 'gpu:6000:3'
test_folder = 'test' if args.test else 'train'
# if '(' in gress:
#     gress = 'gpu:6000:4'
# gres = 'gpu:a100:8'
# partion = 'gpu_week'
# account_text = '#SBATCH --account=lang'
# gres = 'a100'
# partion = 'lang_gpu_long'
#SBATCH --nodes=1
#SBATCH --gres={gress}
#SBATCH --partition={partion}
#SBATCH --ntasks={args.per_node}
#SBATCH --account=lang
# {account_text}
#SBATCH --ntasks=8
content = f"""#! /bin/bash
#SBATCH --job-name=
#SBATCH --output=log_file/{test_folder}/%j_{record_name}.txt
#SBATCH --partition=gpu_week
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --gres={gress}
#SBATCH --ntasks={args.per_node}
#SBATCH --cpus-per-task=4
{account_text}

echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_PROCID: $SLURM_PROCID"

if [ -z "$SLURM_NODELIST" ]; then
    echo "Error: SLURM_NODELIST is empty"
    exit 1
fi

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# 打印调试信息
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"

project_name="{args.project_name}"
MESSAGE="${{project_name}} start running"
curl -X POST --data-urlencode "payload={{\\"channel\\": \\"shy-program-notification\\", \\"text\\": \\"${{MESSAGE}}\\"}}" https://hooks.slack.com/services/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

{command}
# {record_command}
if [ $? -eq 0 ]
then
MESSAGE="${{project_name}} program finished successfully!!!!!!! "
else
MESSAGE="${{project_name}} program failed!!!!!!! "
fi

# 发送Slack通知
curl -X POST --data-urlencode "payload={{\\"channel\\": \\"shy-program-notification\\", \\"text\\": \\"$MESSAGE\\n\\n\\n\\n\\n\\"}}" https://hooks.slack.com/services/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
"""

# 打开文件并写入内容
with open(f"{args.project_name}.sh", "w") as file:
    file.write(content)
id = run_bash_file(args)
if not args.test:
    make_tansorboard_record(args, id)