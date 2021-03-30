from torch.utils.data import DistributedSampler
from cytoolz import partition_all
import random
import math


class TokenBucketSampler(DistributedSampler):
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, size_multiple=1):
        self._lens = lens
        self._max_tok = max(batch_size, 1024)
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._size_mul = size_multiple

    def _create_ids(self):
        return list(range(len(self._lens)))

    def _sort_fn(self, i):
        return self._lens[i]

    def __iter__(self):
        ids = self._create_ids()  # 创建了0-23115的id seq
        random.shuffle(ids)  # id 随机洗牌
        buckets = [sorted(ids[i:i + self._bucket_size],
                          key=self._sort_fn, reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]  # 按照桶大小，分成n组。
        # fill batches until max_token (include padding)
        batches = []
        for bucket in buckets:  # 取 每个桶
            max_len = 0
            batch_indices = []
            for indices in partition_all(self._size_mul, bucket):
                # 每个桶里取8个数据
                max_len = max(max_len, max(self._lens[i] for i in indices))
                # 找到当前index下，len最大的以及目前为止最大的len标记，
                max_l = max_len * (len(batch_indices) + self._size_mul)
                if max_l > self._max_tok:
                    # if not batch_indices:
                    #     # print("max_len=", max_len)
                    #     print("max_len * (len(batch_indices) + self._size_mul)=", max_l)
                    #     # batch设置必须不小于8*94；第一次如果就超了，说明batch设置太小，或者 seq的长度太长了；
                    #     raise ValueError(
                    #         "max_tokens too small / max_seq_len too long")
                    # assert len(batch_indices) % self._size_mul == 0
                    batches.append(batch_indices)
                    batch_indices = list(indices)
                else:
                    batch_indices.extend(indices)
            if not self._droplast and batch_indices:
                batches.append(batch_indices)
        random.shuffle(batches)
        for b in batches:
            # return iter(b)
            # for i in b:
            yield iter(b)  # 20210110困扰了我2天的问题 不知道为什么之前的return i 不可以了。
        # yield [batches[i] for i in batches]
        # return iter(batches)

    def __len__(self):
        if self._droplast:
            return len(self._lens) // self._max_tok
        else:
            return math.ceil(len(self._lens) / self._max_tok)
        # raise ValueError("NOT supported. "
        #                  "This has some randomness across epochs")


# TokenBucketSampler
class TokenBucketSamplerForBtm(TokenBucketSampler):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.dataset = dataset

    def __iter__(self):
        it = super().__iter__()
        self.dataset.new_epoch()
        self._lens = self.dataset.lens
        return it
