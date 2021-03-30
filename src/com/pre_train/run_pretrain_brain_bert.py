# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""BrainBERT Pretrain TASKs runner.
   include MLM task
   could include BTM task
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
import sys
sys.path.append(Proj_dir)
import time
import json
import torch
import random
import numpy as np
from apex import amp
from tqdm import tqdm
from horovod import torch as hvd
from os.path import join
from collections import defaultdict
from torch.nn import functional as F
from src.com.pre_train.pretrain_model import BrainBertForPreTraining
from src.com.util.misc import NoOp, set_dropout
from src.com.util.optimizer import get_lr_sched, build_optimizer
from src.com.util.save import ModelSaver, save_training_meta
from src.com.util.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                                      broadcast_tensors)
from src.com.util.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file, make_print_to_file
from src.com.pre_train.dataset import create_dataloaders
from src.com.pre_train.base_data import (MetaLoader,PrefetchLoader,clip_grad_norm_)
IMG_DIM = 1024

@torch.no_grad()
def validate_btm(model, val_loader):
    save_flag = False
    LOGGER.info("start running BTM validation...")
    val_loss = 0
    tot_ot_loss = 0
    tot_ot_pos = 0
    tot_ot_neg = 0
    tot_score = 0
    n_example = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        scores, ot_loss = model(batch=batch, task='btm', compute_loss=False)
        if ot_loss is not None:
            if isinstance(ot_loss, tuple):
                ot_pos, ot_neg = ot_loss
                ot_pos = ot_pos.sum().item()
                ot_neg = ot_neg.sum().item()
                tot_ot_pos += ot_pos
                tot_ot_neg += ot_neg
                tot_ot_loss += ot_pos - ot_neg
            else:
                tot_ot_loss += ot_loss.sum().item()
        targets = batch['targets']
        loss = F.cross_entropy(scores, targets, reduction='sum')
        val_loss += loss.item()

        tot_score += (scores.max(dim=-1)[1] == targets).sum().item()
        if targets is not None:
            n_example += len(targets)
        else:
            n_example += 1
    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_example = sum(all_gather_list(n_example))
    tot_time = time.time() - st
    val_loss /= n_example
    # ZeroDivisionError: division by zero
    val_acc = tot_score / n_example
    val_log = {'valid/loss': val_loss,
               'valid/acc': val_acc,
               'valid/ex_per_s': n_example / tot_time}

    if ot_loss is not None:
        tot_ot_loss = sum(all_gather_list(tot_ot_loss))
        tot_ot_pos = sum(all_gather_list(tot_ot_pos))
        tot_ot_neg = sum(all_gather_list(tot_ot_neg))
        val_log['valid/ot_loss'] = tot_ot_loss / n_example
        val_log['valid/ot_pos'] = tot_ot_pos / n_example
        val_log['valid/ot_neg'] = tot_ot_neg / n_example

    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc * 100:.2f}")
    print(f"validation finished in {int(tot_time)} seconds, "
          f"score: {val_acc * 100:.2f}")
    if val_acc > 0.5:
        save_flag = True
    return val_log,val_acc, save_flag


def validate(model, val_dataloaders):
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate on {task} task")
        if task.startswith('mlm'):
            val_log, acc, save_flag = validate_mlm(model, loader)
        # elif task.startswith('mrfr'):
        #     val_log = validate_mrfr(model, loader)
        # elif task.startswith('mrc'):
        #     val_log = validate_mrc(model, loader, task)
        elif task.startswith('btm'):
            val_log, acc, save_flag = validate_btm(model, loader)
        # else:
        #     raise ValueError(f'Undefined task {task}')
        val_log = {f'{task}_{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scaler_dict(
            {f'valid_{task}/{k}': v for k, v in val_log.items()})
    model.train()
    return save_flag,acc


@torch.no_grad()
def validate_mlm(model, val_loader):
    LOGGER.info("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        #  batch = {key: batch[key].cuda() for key in batch}
        batch = {key: batch[key].cuda() for key in batch}
        scores = model(batch=batch, task='mlm', compute_loss=False)  # 网络输出的概率向量
        labels = batch['txt_labels']
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()
    val_loss = sum(all_gather_list(val_loss))
    n_correct = sum(all_gather_list(n_correct))
    n_word = sum(all_gather_list(n_word))
    tot_time = time.time() - st
    if n_word == 0:
        val_loss = 0
        acc = 0
    else:
        val_loss /= n_word
        acc = n_correct / n_word

    print("acc:",acc,",n_correct：",n_correct,",n_word:",n_word)
    val_log = {'loss': val_loss,
               'acc': acc}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc * 100:.2f}")  # 目标提升到60+
    save_flag = False
    if acc > 0.1:
        save_flag = True
    return val_log, acc, save_flag


def main():
    hvd.init()
    n_examples = defaultdict(int)
    n_in_units = defaultdict(int)
    n_loss_units = defaultdict(int)
    param = sys.argv[2]
    args = json.loads(open(param, 'r', encoding='utf-8').read())
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cudas']
    if args['gradient_accumulation_steps'] < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args['gradient_accumulation_steps']))

    if args['local_rank'] == -1 or args['no_cuda']:
        device = torch.device("cuda" if torch.cuda.is_available() and not args['no_cuda'] else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args['local_rank'])
        device = torch.device("cuda", args['local_rank'])
        n_gpu = 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(args['master_port'])
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=hvd.rank(),
                                             store=None,
                                             group_name='')
        random.seed(args['seed'])
        np.random.seed(args['seed'])
        torch.manual_seed(args['seed'])
    if n_gpu > 0:
        # #为所有GPU设置随机种子
        torch.cuda.manual_seed_all(args['seed'])

    if not args['do_train'] and not args['do_eval']:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args['output_dir']) and not os.listdir(args['output_dir']):
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args['output_dir']))
        os.makedirs(args['output_dir'], exist_ok=True)
    # rank 参数-> GPU id
    # 必须在 rank==0 的进程内保存参数。
    if args['local_rank'] == 0:
        # save_training_meta(args)
        TB_LOGGER.create(join(args['output_dir'], 'log'))
        # pbar = tqdm(total=args['num_train_steps'])
        model_saver = ModelSaver(join(args['output_dir'], 'ckpt'))
        add_log_to_file(join(args['output_dir'], 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        # pbar = NoOp()
        model_saver = NoOp()

    if args['checkpoint']:
        # checkpoint = torch.load(args['checkpoint'])
        checkpoint = {k.replace('module.', ''): v for k, v in torch.load(os.path.join(args['checkpoint'])).items()}
    else:
        checkpoint = {}
    model = BrainBertForPreTraining.from_pretrained(
        args['model_config'], checkpoint,
        img_dim=IMG_DIM)
    if args['fp16']:
        model.half()

    LOGGER.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args['local_rank'] != -1), args['fp16']))
    model.to(device)

    # # DistributedDataParallel（DDP）：All-Reduce模式，本意是用来分布式训练，但是也可用于单机多卡。
    if args['local_rank'] != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        # DataParallel（DP）：Parameter Server模式，一张卡位reducer
        gpus = range(0, n_gpu)
        model = torch.nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
        model = model.cuda()

    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, args['dropout'])

    global_step = 0
    # build data loaders
    train_dataloader, all_img_dbs = create_dataloaders(
        datasets=args['train_datasets'], is_train=True, opts=args)
    val_dataloader, _ = create_dataloaders(
        datasets=args['val_datasets'], is_train=False, opts=args, all_img_dbs=all_img_dbs)
    meta_loader = MetaLoader(loader=train_dataloader,
                             accum_steps=args['gradient_accumulation_steps'],
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(loader=meta_loader)
    model.train()
    task2loss = {task: RunningMeter(f'loss/{task}') for task in train_dataloader.keys()}
    optimizer = build_optimizer(model, args)
    task2scaler = {t: i for i, t in enumerate(train_dataloader.keys())}
    # 获取task name dict👆
    model, optimizer = amp.initialize(model, optimizer,
                                      num_losses=len(task2scaler),
                                      enabled=args['fp16'], opt_level='O2')
    pbar = tqdm(total=args['num_train_steps'])
    train_loss =[]
    validation_acc = []
    # save_training_meta(args)
    for step, (task_name, batch) in enumerate(tqdm(meta_loader, desc="iter")):
        task = task_name.split('_')[0]
        # 核心的训练
        loss = model(batch=batch, task=task, compute_loss=True)
        if task.startswith('mlm') and loss is not None:
            n_loss_units[task_name] += loss.size(0)
            loss = loss.mean()
            train_loss.append(loss.item())
            # loss is not normalized in model
        if task.startswith('btm') and loss is not None:
            # OT
            btm_loss, ot_loss = loss
            n_loss_units[task_name] += btm_loss.size(0)
            btm_loss = btm_loss.mean()
            if ot_loss is not None:
                ot_pos, ot_neg = ot_loss
                ot_loss = (ot_pos.sum() - ot_neg.sum()
                           ) / (ot_pos.size(0) + ot_neg.size(0))
                loss = btm_loss + args['btm_ot_lambda'] * ot_loss
                # task2loss[f'{task_name}_xe'](btm_loss.item())
                # task2loss[f'{task_name}_ot'](ot_loss.item())
            else:
                loss = btm_loss

        # backward pass
        delay_unscale = (step + 1) % args['gradient_accumulation_steps'] != 0
        with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale,
                                loss_id=task2scaler[task_name]) as scaled_loss:
                if scaled_loss is not None:
                    scaled_loss.backward()
                if not delay_unscale:
                    # gather gradients from every processes
                    # do this before unscaling to make sure every process uses
                    # the same gradient scale
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))
        # optimizer update and logging
        if (step + 1) % args['gradient_accumulation_steps'] == 0:
            global_step += 1
            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, args)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # log loss
            # NOTE: not gathered across GPUs for efficiency
            TB_LOGGER.log_scaler_dict({ll.name: ll.val
                                       for ll in task2loss.values()
                                       if ll.val is not None})
            TB_LOGGER.step()

            # update model params
            if args['grad_norm'] != -1:
                grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                            args['grad_norm'])
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)
            if global_step % args['valid_steps'] == 0:
                LOGGER.info(f'Step {global_step}: start validation')
                save_flag, acc = validate(model, val_dataloader)
                validation_acc.append(acc)
                if save_flag:
                    model_saver.save(model, round(acc*100,2), global_step)
        if global_step >= args['num_train_steps']:
            break
    args['loss'] = train_loss
    args['acc'] = validation_acc
    save_training_meta(args)
    print(args)



if __name__ == "__main__":
    start = time.perf_counter()
    make_print_to_file(path='.')
    main()
    end = time.perf_counter()
    time_cost = str((end - start) / 60)
    print("time-cost:", time_cost)
