"""
 finetuning for Image-Text Retrieval
"""
import argparse
import os
from os.path import join
from time import time
import random
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from apex import amp
from horovod import torch as hvd
from tqdm import tqdm
# sys.path.append(os.path.abspath('../../../'))

from src.com.pre_train.base_data import (PrefetchLoader, TxtTokLmdb, ImageLmdbGroup)
from com.fine_tune.bt_model import BrainBertForBTRetrieval
from src.com.pre_train.dataset import (BtmRankDataset, btm_rank_collate,
                                       BtmValDataset, btm_val_collate,
                                       BtmEvalDataset, btm_eval_collate)
from src.com.util.optimizer import get_lr_sched, build_optimizer
from src.com.util.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from src.com.util.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                                      broadcast_tensors)
from src.com.util.save import ModelSaver, save_training_meta
from src.com.util.misc import NoOp, set_dropout, set_random_seed
from src.com.util.btm_eval import evaluate

Proj_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))

IMG_DIM = 1024


def build_dataloader(dataset, collate_fn, is_train, opts):
    batch_size = opts.train_batch_size if is_train else 1
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=is_train, drop_last=is_train,
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader


def main(opts):
    # n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    if hvd.rank() == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
        # store BTM predictions
        os.makedirs(join(opts.output_dir, 'results_val'), exist_ok=True)
        os.makedirs(join(opts.output_dir, 'results_test'), exist_ok=True)
        os.makedirs(join(opts.output_dir, 'results_train'), exist_ok=True)
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # train_examples = None
    LOGGER.info(f"Loading Train Dataset {opts.train_txt_dbs}, "
                f"{opts.train_brain_dbs}")
    # check multiple DBs
    assert len(opts.train_txt_dbs) == len(opts.train_brain_dbs), \
        "pre_train txt_db and img_db have different length"

    # load DBs and image dirs ,
    #                                  opts.num_bb, opts.compressed_db
    all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb)
    # pre_train
    LOGGER.info(f"Loading Train Dataset "
                f"{opts.train_txt_dbs}, {opts.train_brain_dbs}")
    train_datasets = []
    for txt_path, img_path in zip(opts.train_txt_dbs, opts.train_brain_dbs):
        img_db = all_img_dbs[img_path]
        txt_db = TxtTokLmdb(txt_path, opts.max_txt_len)
        train_datasets.append(BtmRankDataset(txt_db, img_db,
                                             opts.negative_size))
    train_dataset = ConcatDataset(train_datasets)

    # val
    LOGGER.info(f"Loading Val Dataset {opts.val_txt_db}, {opts.val_brain_db}")
    val_brain_db = all_img_dbs[opts.val_brain_db]
    val_txt_db = TxtTokLmdb(opts.val_txt_db, -1)
    val_dataset = BtmValDataset(val_txt_db, val_brain_db,
                                opts.inf_minibatch_size)
    val_dataloader = build_dataloader(val_dataset, btm_val_collate,
                                      False, opts)
    # eval
    LOGGER.info(f"Loading val, test Dataset for full evaluation: "
                f"{opts.val_txt_db}, {opts.val_brain_db}"
                f"{opts.test_txt_db}, {opts.test_brain_db}")
    eval_dataset_val = BtmEvalDataset(val_txt_db, val_brain_db,
                                      opts.inf_minibatch_size)
    eval_loader_val = build_dataloader(eval_dataset_val, btm_eval_collate,
                                       False, opts)
    test_brain_db = all_img_dbs[opts.test_brain_db]
    test_txt_db = TxtTokLmdb(opts.test_txt_db, -1)
    eval_dataset_test = BtmEvalDataset(test_txt_db, test_brain_db,
                                       opts.inf_minibatch_size)
    eval_loader_test = build_dataloader(eval_dataset_test, btm_eval_collate,
                                        False, opts)

    # Prepare model
    if opts.checkpoint:
        # checkpoint = torch.load(opts.checkpoint)
        checkpoint = {k.replace('module.', ''): v for k, v in torch.load(os.path.join(opts.checkpoint)).items()}

    else:
        checkpoint = {}

    model = BrainBertForBTRetrieval.from_pretrained(
        opts.model_config, state_dict=checkpoint,
        img_dim=IMG_DIM, margin=opts.margin)

    model.init_output()  # pretrain BTM head is different from ranking head
    model.to(device)
    if opts.local_rank != -1:
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
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=opts.fp16, opt_level='O2')

    global_step = 0
    LOGGER.info(f"***** Running training on {n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_dataset) * hvd.size())
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    running_loss = RunningMeter('loss')
    model.train()

    n_examples = 0
    n_epoch = 0
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    while True:
        train_dataloader = build_dataloader(
            train_dataset, btm_rank_collate, True, opts)
        for step, batch in enumerate(train_dataloader):
            n_examples += batch['input_ids'].size(0)
            loss = model(batch=batch, task='btm', compute_loss=True)
            loss = loss.mean()
            delay_unscale = (step + 1) % opts.gradient_accumulation_steps != 0
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale) as scaled_loss:
                scaled_loss.backward()
                if not delay_unscale:
                    # gather gradients from every processes
                    # do this before unscaling to make sure every process uses
                    # the same gradient scale
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))

            running_loss(loss.item())
            if (step + 1) % opts.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, opts)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
                TB_LOGGER.step()

                # update model params
                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

                if global_step % 100 == 0:
                    # monitor training throughput
                    LOGGER.info(f'------------Step {global_step}-------------')
                    tot_ex = sum(all_gather_list(n_examples))
                    ex_per_sec = int(tot_ex / (time() - start))
                    LOGGER.info(f'{tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar('perf/ex_per_s',
                                         ex_per_sec, global_step)
                    LOGGER.info(f'-------------------------------------------')

                if global_step % opts.valid_steps == 0:
                    if opts.full_val:
                        LOGGER.info(
                            f"========================== Step {global_step} "
                            f"==========================")
                        val_log = evaluate(model, eval_loader_val)
                        TB_LOGGER.log_scaler_dict(
                            {f"valid/{k}": v for k, v in val_log.items()})
                        LOGGER.info(f"brain image retrieval R1: "
                                    f"{val_log['img_r1'] * 100:.2f},\n"
                                    f"brain image retrieval R5: "
                                    f"{val_log['img_r5'] * 100:.2f},\n"
                                    f"brain image retrieval R10: "
                                    f"{val_log['img_r10'] * 100:.2f}\n"
                                    f"text retrieval R1: "
                                    f"{val_log['txt_r1'] * 100:.2f},\n"
                                    f"text retrieval R5: "
                                    f"{val_log['txt_r5'] * 100:.2f},\n"
                                    f"text retrieval R10: "
                                    f"{val_log['txt_r10'] * 100:.2f}")
                        LOGGER.info("================================="
                                    "=================================")
                    else:
                        val_log = validate(model, val_dataloader)
                        TB_LOGGER.log_scaler_dict(val_log)
                    model_saver.save(model, global_step)

            if global_step >= opts.num_train_steps:
                break

        if global_step >= opts.num_train_steps:
            break
        n_epoch += 1
        LOGGER.info(f"finished {n_epoch} epochs")

    pbar.close()
    if opts.num_train_steps % opts.valid_steps != 0:
        # final validation
        val_log = validate(model, val_dataloader)
        TB_LOGGER.log_scaler_dict(val_log)
        model_saver.save(model, global_step)

    # evaluation
    for split, loader in [('val', eval_loader_val),
                          ('test', eval_loader_test)]:
        eval_log = evaluate(model, loader)
        TB_LOGGER.log_scaler_dict({f"eval/{split}_{k}": v
                                   for k, v in eval_log.items()})
        if hvd.rank() != 0:
            continue
        LOGGER.info(
            f"========================= {split} ===========================\n"
            f"brain image retrieval R1: {eval_log['img_r1'] * 100:.2f},\n"
            f"brain image retrieval R5: {eval_log['img_r5'] * 100:.2f},\n"
            f"brain image retrieval R10: {eval_log['img_r10'] * 100:.2f}\n"
            f"text retrieval R1: {eval_log['txt_r1'] * 100:.2f},\n"
            f"text retrieval R5: {eval_log['txt_r5'] * 100:.2f},\n"
            f"text retrieval R10: {eval_log['txt_r10'] * 100:.2f}")
    LOGGER.info("=========================================================")


@torch.no_grad()
def validate(model, val_loader):
    if hvd.rank() == 0:
        pbar = tqdm(total=len(val_loader))
    else:
        pbar = NoOp()
    LOGGER.info("start running Brain Retrieval validation ...")
    model.eval()
    n_ex = 0
    st = time()

    recall_at_1, recall_at_5, recall_at_10 = 0, 0, 0
    for batch in val_loader:
        scores = model(batch=batch, task='btm', compute_loss=False)
        _, indices = scores.squeeze(1).topk(10, dim=0)
        rank = (indices == 0).nonzero()
        if rank.numel():
            rank = rank.item()
            if rank < 1:
                recall_at_1 += 1
            if rank < 5:
                recall_at_5 += 1
            if rank < 10:
                recall_at_10 += 1
        n_ex += 1
        pbar.update(1)
    n_ex = sum(all_gather_list(n_ex))
    recall_at_1 = sum(all_gather_list(recall_at_1)) / n_ex
    recall_at_5 = sum(all_gather_list(recall_at_5)) / n_ex
    recall_at_10 = sum(all_gather_list(recall_at_10)) / n_ex
    tot_time = time() - st
    val_log = {'valid/ex_per_s': n_ex / tot_time,
               'valid/recall_1': recall_at_1,
               'valid/recall_5': recall_at_5,
               'valid/recall_10': recall_at_10}
    model.train()
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"recall_1: {recall_at_1 * 100:.2f}, "
                f"recall_5: {recall_at_5 * 100:.2f}, "
                f"recall_10: {recall_at_10 * 100:.2f}")
    pbar.close()
    return val_log


if __name__ == "__main__":
    hvd.init()
    parser = argparse.ArgumentParser()

    # # Required parameters
    #
    # parser.add_argument('--compressed_db', action='store_true',
    #                     help='use compressed LMDB')
    # parser.add_argument("--checkpoint",
    #                     default=None, type=str,
    #                     help="pretrained MLM")
    #
    # parser.add_argument("--output_dir", default=None, type=str,
    #                     help="The output directory where the model "
    #                          "checkpoints will be written.")
    #
    # # Prepro parameters
    # parser.add_argument('--max_txt_len', type=int, default=60,
    #                     help='max number of tokens in text (BERT BPE)')
    # parser.add_argument('--conf_th', type=float, default=0.2,
    #                     help='threshold for dynamic bounding boxes '
    #                          '(-1 for fixed)')
    # parser.add_argument('--max_bb', type=int, default=100,
    #                     help='max number of bounding boxes')
    # parser.add_argument('--min_bb', type=int, default=10,
    #                     help='min number of bounding boxes')
    # parser.add_argument('--num_bb', type=int, default=36,
    #                     help='static number of bounding boxes')
    #
    # # training parameters
    # parser.add_argument("--train_batch_size", default=128, type=int,
    #                     help="Total batch size for training. "
    #                          "(batch by examples)")
    # parser.add_argument("--negative_size", default=1, type=int,
    #                     help="Number of negative samples per positive sample")
    # parser.add_argument("--inf_minibatch_size", default=400, type=int,
    #                     help="batch size for running inference. "
    #                          "(used for validation, and evaluation)")
    #
    # parser.add_argument("--margin", default=0.2, type=float,
    #                     help="margin of ranking loss")
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
    #                     help="Number of updates steps to accumualte before "
    #                          "performing a backward/update pass.")
    # parser.add_argument("--learning_rate", default=3e-5, type=float,
    #                     help="The initial learning rate for Adam.")
    # parser.add_argument("--valid_steps", default=1000, type=int,
    #                     help="Run validation every X steps")
    # parser.add_argument("--num_train_steps", default=100000, type=int,
    #                     help="Total number of training updates to perform.")
    # parser.add_argument("--optim", default='adam',
    #                     choices=['adam', 'adamax', 'adamw'],
    #                     help="optimizer")
    # parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
    #                     help="beta for adam optimizer")
    # parser.add_argument("--dropout", default=0.1, type=float,
    #                     help="tune dropout regularization")
    # parser.add_argument("--weight_decay", default=0.01, type=float,
    #                     help="weight decay (L2) regularization")
    # parser.add_argument("--grad_norm", default=0.25, type=float,
    #                     help="gradient clipping (-1 for no clipping)")
    # parser.add_argument("--warmup_steps", default=4000, type=int,
    #                     help="Number of training steps to perform linear "
    #                          "learning rate warmup for.")
    #
    # # device parameters
    # parser.add_argument('--seed', type=int, default=42,
    #                     help="random seed for initialization")
    # parser.add_argument('--full_val', action='store_true',
    #                     help="Always run full evaluation during training")
    # parser.add_argument('--fp16', action='store_true',
    #                     help="Whether to use 16-bit float precision instead "
    #                          "of 32-bit")
    # parser.add_argument('--n_workers', type=int, default=4,
    #                     help="number of data workers")
    # parser.add_argument('--pin_mem', action='store_true',
    #                     help="pin memory")
    #
    # # can use config files
    # parser.add_argument('--config', help='JSON config files')

    # args = parse_with_config(parser)
    args = parser.parse_args()
    args.warmup_steps = 100
    args.btm_neg_prob = 0.5
    args.optim = "adamw"
    args.local_rank = -1  # hvd.local_rank()
    args.n_workers = 2
    args.num_train_steps = 10
    args.valid_steps = 5
    args.pin_mem = True
    args.grad_norm = 5.0
    args.no_cuda = False
    args.gradient_accumulation_steps = 1
    args.val_batch_size = 32
    args.train_batch_size = 32
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    args.val_batch_size = int(args.val_batch_size / args.gradient_accumulation_steps)
    args.seed = random.randint(3, 5)
    print("seed= ", args.seed)
    args.do_train = True
    args.do_eval = True
    args.output_dir = "../../../output/"
    args.do_lower_case = True
    # args.train_file = "../pre_train/alice.txt"
    args.on_memory = True
    args.dropout = 0.1
    args.max_txt_len = 200
    args.conf_th = 0.2
    args.max_bb = int(1024 * 0.2)
    args.min_bb = 10
    args.model_config = Proj_dir + '/config/brainbert-large.json'
    args.betas = [
        0.9,
        0.98
    ]
    args.fp16 = False
    args.loss_scale = 0
    args.learning_rate = 1e-5
    args.weight_decay = 0.01

    args.btm_ot_lambda = 0.1
    args.checkpoint = Proj_dir + "/output/ckpt/model_step_240.pt"
    args.train_txt_dbs = [
        Proj_dir + "/txt/alice/pretrain_train.db",
        Proj_dir + "/txt/pereira/pretrain_train.db"
    ]
    args.train_brain_dbs = [
        Proj_dir + "/brain/alice/pretrain_train.db",
        Proj_dir + "/brain/pereira/pretrain_train.db"
    ]
    args.val_txt_db = Proj_dir + f"/txt/pereira/pretrain_val.db"
    args.val_brain_db = Proj_dir + f"/brain/pereira/pretrain_val.db"
    args.test_txt_db = Proj_dir + f"/txt/pereira/pretrain_test.db"
    # Proj_dir + "/txt/pereira/pretrain_test.db"]
    args.test_brain_db = Proj_dir + f"/brain/pereira/pretrain_test.db"
    # Proj_dir + "/brain/pereira/pretrain_test.db"]
    # args.checkpoint = "output/ckpt/model_step_50.pt"
    # args.train_txt_dbs = [
    #     "txt/alice/pretrain_train.db"
    # ]
    # args.train_brain_dbs = [
    #     "brain/alice/pretrain_train.db"
    # ]
    # args.val_txt_db = "txt/alice/pretrain_val.db"
    # args.val_brain_db = "brain/alice/pretrain_val.db"
    # args.test_txt_db = "txt/alice/pretrain_test.db"
    # args.test_brain_db = "brain/alice/pretrain_test.db"
    args.negative_size = 1
    args.inf_minibatch_size = 400
    args.margin = 0.2
    args.full_val = True
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=hvd.rank(),
                                             store=None,
                                             group_name='')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    if n_gpu > 0:
        # #为所有GPU设置随机种子
        torch.cuda.manual_seed_all(args.seed)

    # if exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not "
    #                      "empty.".format(args.output_dir))

    # # options safe guard
    # if args.conf_th == -1:
    #     assert args.max_bb + args.max_txt_len + 2 <= 512
    # else:
    #     assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
