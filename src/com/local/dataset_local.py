"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

task dataset
"""
import copy
import random
from horovod import torch as hvd
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from src.com.util.sampler import TokenBucketSampler, TokenBucketSamplerForBtm
from src.com.pre_train.base_data import (PrefetchLoader,TxtTokLmdb, DetectFeatLmdb,ImageLmdbGroup,ConcatDatasetWithLens)
from src.com.util.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file, make_print_to_file

BUCKET_SIZE = 128

def pad_tensors(tensors, lens=None, pad=0, type=None):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    # max_len = max(lens)
    max_len = max(max(lens), 1024)
    bs = len(tensors)
    # hid = tensors[0].size(-1)
    dtype = tensors[0].dtype

    output = torch.zeros(bs, max_len, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data

    return output


def get_gather_index(batch_size, out_size):
    gather_index = torch.arange(0, out_size, dtype=torch.long,
                                ).unsqueeze(0).repeat(batch_size, 1)
    return gather_index


def mlm_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim) (1,1,1024)
    :img_pos_feat (n, max_num_bb, )
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    if len(inputs) > 0:
        # print(len(inputs))
        (id, input_ids, img_feats, img_pos_feats, attn_masks, txt_labels
         ) = map(list, unzip(inputs))
        # text batches
        # txt_lens = [i.size(0) for i in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        # image batches
        num_bbs = [f.size(0) for f in img_feats]
        img_feat = pad_tensors(img_feats, num_bbs, type='img_feat')
        img_pos_feat = torch.arange(0, img_feat.size(1), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

        bs, max_tl = input_ids.size()
        out_size = attn_masks.size(1)
        gather_index = get_gather_index(bs,out_size)

        batch = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'txt_labels': txt_labels,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'attn_masks': attn_masks,
            'gather_index': gather_index}
        return batch
    else:
        return None


def mlm_text_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    if len(inputs) > 0:
        # print(len(inputs))
        (id, input_ids, attn_masks, txt_labels
         ) = map(list, unzip(inputs))
        # text batches
        # txt_lens = [i.size(0) for i in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)
        attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
        batch = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'txt_labels': txt_labels,
            'attn_masks': attn_masks,
        }
        return batch
    else:
        return None
#
# def btm_ot_collate(inputs):
#     if len(inputs) > 0:
#         (input_ids, img_feats, img_pos_feats, attn_masks, targets
#          ) = map(list, unzip(inputs))
#
#         txt_lens = [i.size(0) for i in input_ids]
#
#         input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)  # (list纯粹换成tensor)
#         position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
#                                     ).unsqueeze(0)
#
#         num_bbs = [f.size(0) for f in img_feats]
#         img_feat = pad_tensors(img_feats, num_bbs, type='img_feat')
#         img_pos_feat = torch.arange(0, img_feat.size(1), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
#         attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
#         targets = torch.cat(targets, dim=0)
#         bs, max_tl = input_ids.size()
#         out_size = attn_masks.size(1)
#         # gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
#         gather_index = get_gather_index(bs,out_size)
#
#         # OT inputs
#         max_tl = max(txt_lens)
#         max_nbb = max(max(num_bbs), 1024)
#         ot_scatter = _compute_ot_scatter(txt_lens, max_tl, attn_masks.size(1))
#         txt_pad = _compute_pad(txt_lens, max_tl)
#         img_pad = _compute_pad(num_bbs, max_nbb)
#         ot_inputs = {'ot_scatter': ot_scatter,
#                      'scatter_max': ot_scatter.max().item(),
#                      'txt_pad': txt_pad,
#                      'img_pad': img_pad}
#
#         batch = {'input_ids': input_ids,
#                  'position_ids': position_ids,
#                  'img_feat': img_feat,
#                  'img_pos_feat': img_pos_feat,
#                  'attn_masks': attn_masks,
#                  'gather_index': gather_index,
#                  'targets': targets,
#                  'ot_inputs': ot_inputs}
#         return batch
#     else:
#         return None
#
#
# def btm_collate(inputs):
#     (input_ids, img_feats, img_pos_feats, attn_masks, targets
#      ) = map(list, unzip(inputs))
#
#     txt_lens = [i.size(0) for i in input_ids]
#
#     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
#     position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
#                                 ).unsqueeze(0)
#
#     num_bbs = [f.size(0) for f in img_feats]
#     img_feat = pad_tensors(img_feats, num_bbs, type='img_feat')
#     img_pos_feat = torch.arange(0, img_feat.size(1), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
#     attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
#     targets = torch.cat(targets, dim=0)
#     bs, max_tl = input_ids.size()
#     out_size = attn_masks.size(1)
#     # gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
#     gather_index = get_gather_index(bs, out_size)
#
#     batch = {'input_ids': input_ids,
#              'position_ids': position_ids,
#              'img_feat': img_feat,
#              'img_pos_feat': img_pos_feat,
#              'attn_masks': attn_masks,
#              'gather_index': gather_index,
#              'targets': targets}
#     return batch
#

class DetectFeatTxtTokDataset(Dataset):
    def __init__(self, txt_db, brain_db):
        assert isinstance(txt_db, TxtTokLmdb)
        # assert isinstance(brain_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.brain_db = brain_db
        txt_lens, self.ids = get_ids_and_lens(txt_db)

        txt2brain = txt_db.txt2brain
        if self.brain_db is not None:
            self.lens = [tl + self.brain_db.name2nbb[txt2brain[id_]]
                         for tl, id_ in zip(txt_lens, self.ids)]
        else:
            self.lens = txt_lens



    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        return example

    def _get_img_feat(self, fname):
        img_feat, bb = self.brain_db[fname]
        # TODO 这里也不确定是啥😂 pos useless
        img_pos_bb = None
        # torch.cat([img_feat, img_feat[:, 4:5] * img_feat[:, 5:]], dim=-1)
        num_bb = img_feat.size(0)
        return img_feat, img_pos_bb, num_bb


class DetectFeatTxtDataset(Dataset):
    def __init__(self, txt_db, brain_db):
        assert isinstance(txt_db, TxtTokLmdb)
        # assert isinstance(brain_db, DetectFeatLmdb)
        self.txt_db = txt_db
        # self.brain_db = brain_db
        txt_lens, self.ids = get_ids_and_lens(txt_db)

        # txt2brain = txt_db.txt2brain
        # if self.brain_db is not None:
        #     self.lens = [tl + self.brain_db.name2nbb[txt2brain[id_]]
        #                  for tl, id_ in zip(txt_lens, self.ids)]
        # else:
        self.lens = txt_lens



    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        return example

    def _get_img_feat(self, fname):
        img_feat, bb = self.brain_db[fname]
        # TODO 这里也不确定是啥😂 pos useless
        img_pos_bb = None
        # torch.cat([img_feat, img_feat[:, 4:5] * img_feat[:, 5:]], dim=-1)
        num_bb = img_feat.size(0)
        return img_feat, img_pos_bb, num_bb


def _has_overlap(la, lb):
    if len(la) < len(lb):
        la, lb = lb, la
    s = set(la)
    return any(b in s for b in lb)


def sample_negative(sample_pool, ground_truths, num_sample):
    """ random and retry """
    outputs = ground_truths
    while _has_overlap(outputs, ground_truths):
        outputs = random.sample(sample_pool, num_sample)
    return outputs


class BtmDataset(DetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """

    def __init__(self, txt_db, brain_db, neg_sample_p=0.5):
        super().__init__(txt_db, brain_db)
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(brain_db, DetectFeatLmdb)

        self.txt_db = txt_db
        self.brain_db = brain_db
        self.train_imgs = []

        self.txt_lens, self.ids = get_ids_and_lens(txt_db)
        self.all_imgs = list(txt_db[id_]['img_fname'] for id_ in self.ids)

        self.neg_sample_p = neg_sample_p
        self.new_epoch()

    def new_epoch(self):
        """ should be called every epoch for more randomness"""
        # 随机生成0，1 label
        self.labels = np.random.choice(
            [0, 1], size=len(self.ids),
            p=[self.neg_sample_p, 1 - self.neg_sample_p])

        self.lens = []
        self.train_imgs = []
        for i, (id_, tl) in enumerate(zip(self.ids, self.txt_lens)):
            img_fname = super().__getitem__(i)['img_fname']
            # print("original name:", img_fname)
            if self.labels[i] == 0:
                # 负label 做negative处理
                img_fname = sample_negative(self.all_imgs, [img_fname], 1)[0]
                # print("negative name:", img_fname)

            self.train_imgs.append(img_fname)
            # self.train_imgs.extend(img_fname)
            max_nbb = 0
            # for j in img_fname:
            max_nbb = max(max_nbb, self.brain_db.name2nbb[img_fname])
            # TODO 改了源代码
            self.lens.append(tl + max_nbb)

    def __getitem__(self, i):
        example = super().__getitem__(i)
        # labels and negative images should be sampled every epoch
        ground_truth_label = self.labels[i]
        img_fname = self.train_imgs[i]
        img_feat, img_pos_feat, num_bb = self._get_img_feat(img_fname)

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        attn_masks = torch.ones(len(input_ids) + 1, dtype=torch.long)
        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)

        return input_ids, img_feat, img_pos_feat, attn_masks, target


def _compute_ot_scatter(txt_lens, max_txt_len, joint_len):
    ot_scatter = torch.arange(0, joint_len, dtype=torch.long
                              ).unsqueeze(0).repeat(len(txt_lens), 1)
    for i, tl in enumerate(txt_lens):
        max_ind = max_txt_len + (joint_len - tl)
        ot_scatter.data[i, tl:] = torch.arange(max_txt_len, max_ind,
                                               dtype=torch.long).data
    return ot_scatter


def _compute_pad(lens, max_len):
    pad = torch.zeros(len(lens), max_len, dtype=torch.uint8)
    for i, l in enumerate(lens):
        pad.data[i, l:].fill_(1)
    return pad

def build_mlm_dataset(txt_db, brain_db, is_train, opts):

    if brain_db is not None:
        collate_fn = mlm_collate
        if is_train:
            datasets = [MlmDataset(t, i) for t, i in zip(txt_db, brain_db)]
            dataset = ConcatDatasetWithLens(datasets)
        # else:
        else:
            dataset = MlmDataset(txt_db, brain_db)
    else:
        collate_fn = mlm_text_collate
        if is_train:
            datasets = [MlmTextDataset(t) for t in txt_db]
            dataset = ConcatDatasetWithLens(datasets)
        # else:
        else:
            dataset = MlmTextDataset(txt_db)
    return dataset, collate_fn


def build_dataloader_btm(dataset, collate_fn, is_train, opts):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size
    sampler = TokenBucketSamplerForBtm(
        dataset=dataset, bucket_size=BUCKET_SIZE,
        batch_size=batch_size, droplast=is_train)
    loader = DataLoader(dataset=dataset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return loader


def build_dataloader(dataset, collate_fn, is_train, opts):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size
    sampler = TokenBucketSampler(lens=dataset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=batch_size, droplast=is_train)
    loader = DataLoader(dataset=dataset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    # loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
    #                     num_workers=opts.n_workers, pin_memory=opts.pin_mem,
    #                     collate_fn=collate_fn)
    # collate_fn的作用就是将一个batch的数据进行合并操作。
    # 默认的collate_fn是将img和label分别合并成imgs和labels，
    # 所以如果你的__getitem__方法只是返回 img, label,那么你可以使用默认的collate_fn方法，
    # 但是如果你每次读取的数据有img, box, label等等，
    # 那么你就需要自定义collate_fn来将对应的数据合并成一个batch数据，
    # 这样方便后续的训练步骤。
    return loader

# def build_btm_dataset(txt_db, brain_db, is_train, opts):
#     if is_train:
#         datasets = [BtmDataset(txt_db=t, brain_db=i, neg_sample_p=opts.btm_neg_prob)
#                     for t, i in zip(txt_db, brain_db)]
#         dataset = ConcatDatasetWithLens(datasets)
#     else:
#         dataset = BtmDataset(txt_db, brain_db, opts.btm_neg_prob)
#     collate_fn = btm_ot_collate if opts.btm_ot_lambda > 0 else btm_collate
#     return dataset, collate_fn
#

class MlmDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)

    def __getitem__(self, index):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(index)
        input_ids, txt_labels = self.create_mlm_io(example['input_ids'])
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])
        attn_masks = torch.ones(len(input_ids) + 1, dtype=torch.long)
        return self.ids[int(index)], input_ids, img_feat, img_pos_feat, attn_masks, txt_labels

    def create_mlm_io(self, input_ids):
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels

def create_dataloaders(datasets, is_train, opts, all_img_dbs=None):
    if all_img_dbs is None:
        all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb)
    dataloader = {}
    for dset in datasets:
        if len(dset['brain'])>0:
            if is_train:
                img_db = [all_img_dbs[path] for path in dset['brain']]
            else:
                # assert len(dset['text']) == len(dset['brain']) == 1
                img_db = all_img_dbs[dset['brain'][0]]
                # LOGGER.info(f"Loading {task} validation dataset, "
                #             f"{dset['text']}, {img_db.img_dir}")
        else:
            img_db = None

        for i, t in enumerate(dset['tasks']):
            task = f'{t}_{dset["name"]}'

        if is_train:
            txt_db = [TxtTokLmdb(path, opts.max_txt_len)
                      for path in dset['text']]
        else:
            txt_db = TxtTokLmdb(dset['text'][0], -1)

        if task.startswith('mlm'):
            dataset = build_mlm_dataset(txt_db, img_db, is_train, opts)
        # elif task.startswith('btm'):
        #     dataset = build_btm_dataset(txt_db, img_db, is_train, opts)
        LOGGER.info(f"{len(dataset[0]) * hvd.size()} samples loaded")

        if task.startswith('btm'):
            # btm handles distributed training in dset not sampler
            loader = build_dataloader_btm(*dataset, is_train, opts)
        else:
            loader = build_dataloader(*dataset, is_train, opts)
        if is_train:  # pre_train test
            # ratio = dset['mix_ratio'][i]
            dataloader[task] = loader
        else:  # validation
            dataloader[task] = PrefetchLoader(loader)
    LOGGER.info(f"***** Running training with GPUs *****")
    LOGGER.info("  Num examples = %d", len(dataset[0]))
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)
    return dataloader, all_img_dbs



class MlmTextDataset(DetectFeatTxtDataset):
    def __init__(self, txt_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db,None)

    def __getitem__(self, index):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(index)
        input_ids, txt_labels = self.create_mlm_io(example['input_ids'])
        attn_masks = torch.ones(len(input_ids), dtype=torch.long)
        return self.ids[int(index)], input_ids, attn_masks, txt_labels

    def create_mlm_io(self, input_ids):
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels


def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label


class BtmRankDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, \
            "BtmRankDataset need at least 1 negative sample"
        super().__init__(txt_db, img_db)

        txt2brain = self.txt_db.txt2brain
        self.txt2brain = {id_: txt2brain[id_] for id_ in self.ids}
        # images partitioned by rank
        self.brain2txt = self.txt_db.brain2txt
        # defaultdict(list)
        # for id_, img_list in self.txt2brain.items():
        #     for img in img_list:
        #         self.brain2txt[img].append(id_)
        self.img_name_list = list(self.brain2txt.keys())

        assert neg_sample_size > 0
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        text_list = []
        gt_img_fname = self.txt2brain[gt_txt_id]
        # for brain_frame in gt_img_fname:
        text_list.append(self.brain2txt[gt_img_fname])

        id_pairs = [(gt_txt_id, gt_img_fname)]
        # sample negatives
        neg_sample_img_ids = sample_negative(
            self.img_name_list, [gt_img_fname], self.neg_sample_size)
        neg_sample_txt_ids = sample_negative(
            self.ids, text_list, self.neg_sample_size)
        id_pairs.extend([(gt_txt_id, neg_img_id)
                         for neg_img_id in neg_sample_img_ids] +
                        [(neg_txt_id, gt_img_fname)
                         for neg_txt_id in neg_sample_txt_ids])
        inputs = self._collect_inputs(id_pairs)
        assert len(inputs) == (1 + 2 * self.neg_sample_size)
        return inputs

    def _collect_inputs(self, id_pairs):
        # create input features
        inputs = []
        for txt_id, img_fname in id_pairs:
            example = self.txt_db[txt_id]
            # text input
            input_ids = example['input_ids']
            input_ids = self.txt_db.combine_inputs(input_ids)
            img_feat, img_pos_feat, num_bb = self._get_img_feat(img_fname)
            # mask
            attn_masks = torch.ones(len(input_ids) + 1, dtype=torch.long)

            inputs.append((input_ids, img_feat, img_pos_feat, attn_masks))

        return inputs
#
#
# def btm_rank_collate(inputs):
#     (input_ids, img_feats, img_pos_feats, attn_masks,
#      ) = map(list, unzip(concat(i for i in inputs)))
#
#     txt_lens = [i.size(0) for i in input_ids]
#     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
#     position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
#                                 ).unsqueeze(0)
#
#     num_bbs = [f.size(0) for f in img_feats]
#     img_feat = pad_tensors(img_feats, num_bbs, type='img_feat')
#     img_pos_feat = torch.arange(0, img_feat.size(1), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
#     attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
#     sample_size = len(inputs[0])
#     assert all(sample_size == len(i) for i in inputs)
#
#     bs, max_tl = input_ids.size()
#     out_size = attn_masks.size(1)
#     # gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
#     gather_index = get_gather_index(bs, out_size)
#
#     batch = {'input_ids': input_ids,
#              'position_ids': position_ids,
#              'img_feat': img_feat,
#              'img_pos_feat': img_pos_feat,
#              'attn_masks': attn_masks,
#              'gather_index': gather_index,
#              'sample_size': sample_size}
#     return batch
#

class BtmRankDatasetHardNegFromText(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, "need at least 1 negative sample"
        super().__init__(txt_db, img_db)

        txt2brain = self.txt_db.txt2brain
        self.txt2img = {id_: txt2brain[id_] for id_ in self.ids}
        self.brain2txt = self.txt_db.brain2txt
        self.img_name_list = list(self.brain2txt.keys())
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_fname = self.txt2img[gt_txt_id]

        input_ids = self.txt_db[gt_txt_id]['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        neg_img_ids = sample_negative(
            self.img_name_list, [gt_img_fname], self.neg_sample_size)
        img_ids = [gt_img_fname] + neg_img_ids
        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs, type='img_feat')
        img_pos_feat = torch.arange(0, img_feat.size(1), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        tl = input_ids.size(1)
        attn_masks = torch.zeros(len(img_ids), max(num_bbs) + tl).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks.data[i, :tl + nbb].fill_(1)
        out_size = attn_masks.size(1)
        # gather_index = get_gather_index([tl] * len(img_ids), num_bbs,
        #                                 len(img_ids), tl, out_size)
        # gather_index = get_gather_index(bs,out_size)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': None
                 }
        return batch

#
# class BtmRankDatasetHardNegFromImage(DetectFeatTxtTokDataset):
#     def __init__(self, txt_db, img_db, neg_sample_size=1):
#         assert neg_sample_size > 0, "need at least 1 negative sample"
#         super().__init__(txt_db, img_db)
#
#         txt2brain = self.txt_db.txt2brain
#         self.txt2brain = {id_: txt2brain[id_] for id_ in self.ids}
#         self.brain2txt = self.txt_db.brain2txt
#         self.txt_name_list = list(self.txt2brain.keys())
#         self.neg_sample_size = neg_sample_size
#
#     def __getitem__(self, i):
#         gt_txt_id = self.ids[i]
#         gt_img_id = self.txt2brain[gt_txt_id]
#         gt_txt_ids = self.brain2txt[gt_img_id]
#
#         # process image features (gt always first)
#         img_feat, img_pos_feat, nbb = self._get_img_feat(gt_img_id)
#         img_feat = img_feat.unsqueeze(0)
#         img_pos_feat = img_pos_feat.unsqueeze(0)
#
#         # sample negative
#         neg_txt_ids = sample_negative(
#             self.txt_name_list, gt_txt_ids, self.neg_sample_size)
#         txt_ids = [gt_txt_id] + neg_txt_ids
#
#         # process text inputs
#         all_inputs = []
#         txt_lens = []
#         for txt_id in txt_ids:
#             input_ids = self.txt_db.combine_inputs(
#                 self.txt_db[txt_id]['input_ids'])
#             all_inputs.append(input_ids)
#             txt_lens.append(len(input_ids))
#         input_ids = pad_sequence(all_inputs, batch_first=True, padding_value=0)
#         position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
#                                     ).unsqueeze(0)
#
#         attn_masks = torch.zeros(len(txt_ids), max(txt_lens) + nbb).long()
#         for i, tl in enumerate(txt_lens):
#             attn_masks.data[i, :tl + nbb].fill_(1)
#         out_size = attn_masks.size(1)
#         # gather_index = get_gather_index(txt_lens, [nbb] * len(txt_ids),
#         #                                 len(txt_ids), tl, out_size)
#         gather_index = get_gather_index(bs,out_size)
#
#         batch = {'input_ids': input_ids,
#                  'position_ids': position_ids,
#                  'img_feat': img_feat,
#                  'img_pos_feat': img_pos_feat,
#                  'attn_masks': attn_masks,
#                  'gather_index': gather_index}
#         return batch
#

def btm_rank_hn_collate(inputs):
    assert len(inputs) == 1
    return inputs[0]


class BtmValDataset(DetectFeatTxtTokDataset):
    """ For evaluating Image-Text-Retrieval task """

    def __init__(self, db_dir, img_dir, mini_batch_size=400):
        super().__init__(db_dir, img_dir)
        del self.lens
        self.txt2brain = self.txt_db.txt2brain
        self.brain2txt = self.txt_db.brain2txt
        self.all_img_ids = list(self.brain2txt.keys())

        assert len(self.brain2txt) >= mini_batch_size > 0
        self.bs = mini_batch_size

    def _get_batch_ids(self, i):
        gt_txt_id = self.ids[i]
        gt_img_id = self.txt2brain[gt_txt_id]

        # sample fixed negatives for each gt image
        i = self.all_img_ids.index(gt_img_id)
        neg_st = i + 1
        neg_end = neg_st + self.bs - 1
        if neg_end > len(self.all_img_ids):
            # warp around
            neg_end -= len(self.all_img_ids)
            neg_img_ids = (self.all_img_ids[neg_st:]
                           + self.all_img_ids[:neg_end])
        else:
            neg_img_ids = self.all_img_ids[neg_st:neg_end]

        assert len(neg_img_ids) == (self.bs - 1), \
            "Did not sample enough neg samples"

        return gt_img_id, neg_img_ids

    def __getitem__(self, i):
        """ this returns list of mini-batches """
        gt_img_id, neg_img_ids = self._get_batch_ids(i)
        # NOTE 1st one is gt img
        batch = self.get_batch(i, [gt_img_id] + neg_img_ids)
        return batch

    def get_batch(self, i, img_ids):
        example = super().__getitem__(i)

        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0).expand(len(img_ids), -1).clone()
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs, type='img_feat')
        img_pos_feat = torch.arange(0, img_feat.size(1), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        tl = input_ids.size(1)
        attn_masks = torch.zeros(len(img_ids), max(num_bbs) + tl).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks.data[i, :tl + nbb].fill_(1)
        out_size = attn_masks.size(1)
        # gather_index = get_gather_index([tl] * len(img_ids), num_bbs,
        #                                 len(img_ids), tl, out_size)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': None
                 }
        return batch


def btm_val_collate(inputs):
    assert len(inputs) == 1, "input batch size > 1"
    return inputs[0]


def get_ids_and_lens(db):
    assert isinstance(db, TxtTokLmdb)
    lens = []
    ids = []
    for id_ in list(db.id2len.keys())[hvd.rank()::hvd.size()]:
        lens.append(db.id2len[id_])
        ids.append(id_)
    return lens, ids


class BtmEvalDataset(BtmValDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_img_ids = sorted(copy.deepcopy(self.all_img_ids),
                                  key=lambda i: self.brain_db.name2nbb[i])

    def __getitem__(self, i):
        mini_batches = []
        for st in range(0, len(self.all_img_ids), self.bs):
            mini_batches.append(
                self.get_batch(i, self.all_img_ids[st:st + self.bs]))
        return mini_batches


btm_eval_collate = btm_val_collate
btrc_eval_collate = btm_val_collate

btrc_val_collate = btm_val_collate
# btrc_train_collate = btm_rank_collate
