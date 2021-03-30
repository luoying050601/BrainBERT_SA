import os
import json
import lmdb
from os.path import exists
from tqdm import tqdm
import numpy as np
import msgpack
import random
from torch.utils.data import ConcatDataset
from lz4.frame import compress, decompress
import torch
import argparse
from torch._six import inf
# sys.path.append(os.path.abspath('../../../'))

from src.com.util.distributed import any_broadcast

def _fp16_to_fp32(feat_dict):
    out = {k: arr.astype(np.float32)
    if arr.dtype == np.float16 else arr
           for k, arr in feat_dict.items()}
    return out

def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm


def compute_num_bb(confs, conf_th, min_bb, max_bb):
    num_bb = max(min_bb, sum(np.array(confs['data']) > conf_th))
    num_bb = min(max_bb, num_bb)
    return int(num_bb)


class ConcatDatasetWithLens(ConcatDataset):
    """ A thin wrapper on pytorch concat dataset for lens batching """

    def __init__(self, datasets):
        super().__init__(datasets)
        self.lens = [l for dset in datasets for l in dset.lens]

    def __getattr__(self, name):
        return self._run_method_on_all_dsets(name)

    def _run_method_on_all_dsets(self, name):
        def run_all(*args, **kwargs):
            return [dset.__getattribute__(name)(*args, **kwargs)
                    for dset in self.datasets]

        return run_all


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, argparse.Namespace):
            return str(vars(obj), encoding='utf-8')

        else:
            return super(MyEncoder, self).default(obj)


class DetectFeatLmdb(object):
    def __init__(self, img_dir, conf_th, max_bb, min_bb):
        self.img_dir = img_dir
        self.conf_th = conf_th
        self.max_bb = max_bb
        self.min_bb = min_bb
        nbb = f'nbb.json'
        # nbb_path = f'{img_dir}/{nbb}'
        # npz_path = img_dir

        if not exists(f'{img_dir}/{nbb}'):
            # nbb is not pre-computed
            # 第一次执行才会需要重新生成
            self.name2nbb = None
        else:
            #     # f = open(f'{img_dir}/{nbb}')
            self.name2nbb = json.load(open(f'{img_dir}/{nbb}'))

        self.env = lmdb.open(img_dir, readonly=True, create=False, lock=False)  # ,
        #                      readonly=True, create=False,
        #                      readahead=not _check_distributed())
        self.txn = self.env.begin(buffers=True)
        if self.name2nbb is None:
            from src.com.util.data_format import chmod
            if os.path.exists(f'{img_dir}/{nbb}'):
                # 删除文件,path为文件路径
                chmod(f'{img_dir}/{nbb}', 777)
                os.remove(f'{img_dir}/{nbb}')
            file = open(f'{img_dir}/{nbb}', 'w', encoding='utf-8')
            chmod(f'{img_dir}/{nbb}', 755)
            self.name2nbb = self._compute_nbb()
            json.dump(self.name2nbb, file, ensure_ascii=False, cls=MyEncoder)
            file.close()

    def _compute_nbb(self):
        name2nbb = {}
        for fname, dump in tqdm(self.txn.cursor(), desc='reading brain images...'):
            img_dump = msgpack.loads(decompress(dump), raw=False)
            fname = bytes(fname.tolist()).decode('utf-8')
            confs = img_dump['features']
            name2nbb[fname] = np.int64(compute_num_bb(confs, self.conf_th,
                                                      self.min_bb, self.max_bb))
        return name2nbb

    def __del__(self):
        self.env.close()

    # def get_dump(self, file_name):
    #     # hack for MRC
    #     dump = self.txn.get(file_name.encode('utf-8'))
    #     nbb = self.name2nbb[file_name]
    #     # if self.compress:
    #     with io.BytesIO(dump) as reader:
    #         img_dump = np.load(reader, allow_pickle=True)
    #         img_dump = _fp16_to_fp32(img_dump)
    #     # else:
    #     #     img_dump = msgpack.loads(dump, raw=False)
    #     #     img_dump = _fp16_to_fp32(img_dump)
    #     img_dump = {k: arr[:nbb, ...] for k, arr in img_dump.items()}
    #     return img_dump

    def __getitem__(self, file_name):
        nbb = self.name2nbb[file_name]
        dump = self.txn.get(file_name.encode('utf-8'))
        img_dump = msgpack.loads(decompress(dump), raw=False)
        img_dump = {'features': np.reshape(img_dump['features']['data'], (-1, 1))
                    # ,
                    #         'norm_bb': np.reshape(img_dump['norm_bb']['data'], (72, 96, 64))
                    }
        img_feat = torch.tensor(img_dump['features'].T[0][:nbb]).float()
        img_bb = torch.tensor(img_dump['features']).float()
        # data, ranges, minVals = noramlization((img_dump['norm_bb'][36:39, 48:65, 32:36]).reshape(-1, 1))
        # img_bb = torch.tensor(data).float()
        return img_feat, img_bb


class ImageLmdbGroup(object):
    def __init__(self, conf_th, max_bb, min_bb):
        self.path2imgdb = {}
        self.conf_th = conf_th
        self.max_bb = max_bb
        self.min_bb = min_bb
        # self.num_bb = num_bb
        # self.compress = compress

    def __getitem__(self, path):
        img_db = self.path2imgdb.get(path, None)
        if img_db is None:
            img_db = DetectFeatLmdb(path, self.conf_th, self.max_bb,
                                    self.min_bb)
        return img_db


class TxtLmdb(object):
    def __init__(self, db_dir, readonly=True):
        self.readonly = readonly
        if readonly:
            # training
            self.env = lmdb.open(db_dir, readonly=True, create=False, lock=False)
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            # prepro
            self.env = lmdb.open(db_dir, readonly=False, create=True,
                                 map_size=4 * 1024 ** 4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0

    def __del__(self):
        if self.write_cnt:
            self.txn.commit()
        self.env.close()

    def __getitem__(self, key):
        return msgpack.loads(decompress(self.txn.get(key.encode('utf-8'))),
                             raw=False)

    def __setitem__(self, key, value):
        # NOTE: not thread safe
        if self.readonly:
            raise ValueError('readonly text DB')
        ret = self.txn.put(key.encode('utf-8'),
                           compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        #  这里干了啥不太懂，每1000次推一下？
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        return ret

def dict_slice(adict, start, end):
    keys = list(adict.keys())
    dict_slice = {}
    for k in keys[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

class TxtTokLmdb(object):
    def __init__(self, db_dir, max_txt_len=60):
        if max_txt_len == -1:
            self.id2len = json.load(open(f'{db_dir}/id2len.json'))
        else:
            self.id2len = {
                id_: len_
                for id_, len_ in json.load(open(f'{db_dir}/id2len.json')
                                           ).items()
                if len_ <= max_txt_len
            }
        # ceshidaima
        self.id2len = dict_slice(self.id2len,0,50000)
        self.db_dir = db_dir
        self.ids = self.id2len.keys()
        self.db = TxtLmdb(db_dir, readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

    def __getitem__(self, id_):
        txt_dump = self.db[id_]
        return txt_dump

    def combine_inputs(self, *inputs):
        input_ids = [self.cls_]
        for ids in inputs:
            input_ids.extend(ids + [self.sep])
        return torch.tensor(input_ids)

    @property
    def txt2brain(self):
        txt2brain = json.load(open(f'{self.db_dir}/txt2brain.json'))
        return txt2brain

    @property
    def brain2txt(self):
        brain2txt = json.load(open(f'{self.db_dir}/brain2txt.json'))
        return brain2txt


def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def record_cuda_stream(batch):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)
    else:
        pass


class PrefetchLoader(object):
    """
    overlap compute and cuda data transfer
    (copied and then modified from nvidia apex)
    """

    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        # if record_stream() doesn't work, another option is to make sure
        # device inputs are created on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input,
        #                                        device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target,
        #                                         device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use
        # by the main stream at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this
            # side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

    def next(self, it):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


class MetaLoader(object):
    """ wraps multiple data loaders """

    def __init__(self, loader, accum_steps=1, distributed=False):
        assert isinstance(loader, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.sampling_pools = []
        self.accum_steps = accum_steps
        self.distributed = distributed
        self.step = 0
        for n, l in loader.items():
            # TODO 目前代码结构移除了mix ratio参数
            # if isinstance(l, tuple):
            #     l, r = l
            # elif isinstance(l, DataLoader):
            #     r = 1
            # else:
            #     raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n])

    def __iter__(self):
        """ this iterator will run indefinitely """
        task = self.sampling_pools[0]
        while True:
            if self.step % self.accum_steps == 0:
                task = random.choice(self.sampling_pools)
                if self.distributed:
                    # make sure all process is training same task
                    task = any_broadcast(task, 0)
            self.step += 1
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:
                iter_ = iter(self.name2loader[task])
                batch = next(iter_)
                self.name2iter[task] = iter_

            yield task, batch
