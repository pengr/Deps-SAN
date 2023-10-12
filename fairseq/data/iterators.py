# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import os

import numpy as np
import torch

from . import data_utils


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count.关于可保持迭代数的迭代器的包装器

    Args:
        iterable (iterable): iterable to wrap　　# 需要包装的迭代器,根据torch.data.DataLoader创建数据加载器,并在给定的数据集<LanguagePairDataset>类上提供的迭代器
        start (int): starting iteration count   #　开始的迭代数,即offset
        override_len (int): override the iterator length
            returned by ``__len__``             #　覆盖由__len__返回的迭代器长度

    Attributes:
        count (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, start=0, override_len=None):
        self.iterable = iterable
        self.count = start
        self.itr = iter(self)
        if override_len is None: # 默认情况,则由开始的迭代数(offset)+传入的迭代器长度作为self.len
            self.len = start + len(iterable)
        else:
            self.len = override_len

    def __len__(self):
        return self.len

    def __iter__(self):
        for x in self.iterable:
            if self.count >= self.len:
                return
            self.count += 1
            yield x

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        """Whether the iterator has been exhausted."""
        return self.count < len(self)

    def skip(self, num_to_skip):
        """Fast-forward the iterator by skipping *num_to_skip* elements."""
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        return self

    def take(self, n):
        """
        Truncates the iterator to n elements at most.
        """
        self.len = min(self.len, n)


class EpochBatchIterating(object):
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def next_epoch_idx(self):
        raise NotImplementedError

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus: ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
        """
        raise NotImplementedError

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        raise NotImplementedError

    @property
    def iterations_in_epoch(self) -> int:
        """The number of consumed batches in the current epoch."""
        raise NotImplementedError

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        raise NotImplementedError


class StreamingEpochBatchIterator(EpochBatchIterating):
    def __init__(
        self, dataset, epoch=1, num_shards=1, shard_id=0,
    ):
        assert isinstance(dataset, torch.utils.data.IterableDataset)
        self.dataset = dataset
        self.epoch = max(epoch, 1)  # we use 1-based indexing for epochs
        self._current_epoch_iterator = None
        self.num_shards = num_shards
        self.shard_id = shard_id

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called."""
        if self._current_epoch_iterator is not None and self.end_of_epoch():
            return self.epoch + 1
        else:
            return self.epoch

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        self.epoch = self.next_epoch_idx
        self.dataset.set_epoch(self.epoch)
        self._current_epoch_iterator = CountingIterator(
            iterable=ShardedIterator(
                iterable=self.dataset,
                num_shards=self.num_shards,
                shard_id=self.shard_id,
            ),
        )
        return self._current_epoch_iterator

    def end_of_epoch(self) -> bool:
        return not self._current_epoch_iterator.has_next()

    @property
    def iterations_in_epoch(self) -> int:
        if self._current_epoch_iterator is not None:
            return self._current_epoch_iterator.count
        return 0

    def state_dict(self):
        return {
            'epoch': self.epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']


class EpochBatchIterator(EpochBatchIterating):
    """A multi-epoch iterator over a :class:`torch.utils.data.Dataset`. # 在：class：`torch.utils.data.Dataset上的多轮迭代器

    Compared to :class:`torch.utils.data.DataLoader`, this iterator:

    - can be reused across multiple epochs with the :func:`next_epoch_itr`
      method (optionally shuffled between epochs)
    - can be serialized/deserialized with the :func:`state_dict` and
      :func:`load_state_dict` methods
    - supports sharding with the *num_shards* and *shard_id* arguments

    相比于：torch.utils.data.DataLoader`,此迭代器：
    -可以使用：func：`next_epoch_itr`方法在多个epoch之间重复使用（可选地在epochs之间打乱）
    -可以使用：func序列化/反序列化：`state_dict`和：func：`load_state_dict`方法
    -支持使用* num_shards *和* shard_id *参数进行分片

    Args:
        dataset (~torch.utils.data.Dataset): dataset from which to load the data # 用于分批的数据集,<LanguagePairDataset>
        collate_fn (callable): merges a list of samples to form a mini-batch  # 合并示例列表来形成小批次,LanguagePairDataset.collater
        batch_sampler (~torch.utils.data.Sampler): an iterator over batches of
            indices　　# 在indices的批次上的迭代器,即存储有所有批次的列表,每个批次中有句子的索引号
        seed (int, optional): seed for random number generator for
            reproducibility (default: 1).　# 用于重现的随机数生成器种子
        num_shards (int, optional): shard the data iterator into N
            shards (default: 1).         # 将数据迭代器分片为N个分片
        shard_id (int, optional): which shard of the data iterator to
            return (default: 0).        # 返回数据迭代器的哪个分片
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).       # 多少个子进程用于数据加载, 0表示将在主进程中加载数据
        epoch (int, optional): the epoch to start the iterator from
            (default: 1).      # 从当前轮数启动数据迭代器
    """

    def __init__(
        self, dataset, collate_fn, batch_sampler, seed=1, num_shards=1, shard_id=0,
        num_workers=0, epoch=1,
    ):
        assert isinstance(dataset, torch.utils.data.Dataset)  # 判断当前dataset类LanguagePairDataset,是否属于torch.utils.data.Dataset
        self.dataset = dataset                      # 用于分批的数据集,<LanguagePairDataset>
        self.collate_fn = collate_fn                # 合并示例列表来形成小批次,LanguagePairDataset.collater
        self.frozen_batches = tuple(batch_sampler)  # 将存储有所有批次的列表变成元组,每个批次中有句子的索引号
        self.seed = seed                           # 用于重现的随机数生成器种子
        self.num_shards = num_shards               # 将数据迭代器分片为N个分片
        self.shard_id = shard_id                 # 返回数据迭代器的哪个分片
        self.num_workers = num_workers          # 多少个子进程用于数据加载, 0表示将在主进程中加载数据

        self.epoch = max(epoch, 1)  # we use 1-based indexing for epochs 我们将从1开始的索引用于epoch计数
        self.shuffle = True         # 是否打乱数据集的标记
        self._cur_epoch_itr = None  # 定义_cur_epoch_itr和_next_epoch_itr的方法标记
        self._next_epoch_itr = None
        self._supports_prefetch = getattr(dataset, 'supports_prefetch', False)  # 获取当前dataset类的'supports_prefetch'参数,是否支持预取的标记,默认为False

    def __len__(self):  # 返回当前数据迭代器的大小,即当前数据集总共有多少个batch
        return len(self.frozen_batches)

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called. 调用*next_epoch_itr*后,返回epoch编号"""
        if self._next_epoch_itr is not None:
            return self.epoch
        elif self._cur_epoch_itr is not None and self.end_of_epoch():
            return self.epoch + 1
        else:
            return self.epoch  # 返回当前epoch编号

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        """Return a new iterator over the dataset. 在数据集上返回一个新的迭代器

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True). # 在返回迭代器之前先随机打乱全部批次
            fix_batches_to_gpus: ensure that batches are always # 确保在各epochs,全部批次始终分配给相同的分片,要求：attr：`dataset`支持预取
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
        """
        self.epoch = self.next_epoch_idx  # 获取当前epoch编号
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else: # 若_next_epoch_itr标记为None,则定义_cur_epoch_itr为可记录迭代数的迭代器,基于torch.data.DataLoader在给定的数据集<LanguagePairDataset>类上创建的数据加载器
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch, shuffle, fix_batches_to_gpus=fix_batches_to_gpus,
            )
        self.dataset.set_epoch(self.epoch) # 将在epoch开始时收到更新的epoch编号,什么也没做pass
        self.shuffle = shuffle  # 获取是否打乱数据集的标记
        return self._cur_epoch_itr  # 返回可记录迭代数的迭代器,基于torch.data.DataLoader在给定的数据集<LanguagePairDataset>类上创建的数据加载器

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted 返回是否轮数迭代器已到最后"""
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.count
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.count
        return 0

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
            'shuffle': self.shuffle,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict['epoch']
        itr_pos = state_dict.get('iterations_in_epoch', 0)
        if itr_pos > 0:
            # fast-forward epoch iterator
            self._next_epoch_itr = self._get_iterator_for_epoch(
                self.epoch,
                shuffle=state_dict.get('shuffle', True),
                offset=itr_pos,
            )
            if self._next_epoch_itr is None:
                # we finished the epoch, increment epoch counter
                self.epoch += 1

    def _get_iterator_for_epoch(self, epoch, shuffle, fix_batches_to_gpus=False, offset=0):

        def shuffle_batches(batches, seed):
            with data_utils.numpy_seed(seed):  # 首先固定numpy数组的随机数生成(每设置一次后仅生效一次)
                np.random.shuffle(batches)     # 随机打乱所有的批次
            return batches   # 返回经过随机打乱的所有批次

        if self._supports_prefetch:  # 若`dataset`支持预取:
            batches = self.frozen_batches

            if shuffle and not fix_batches_to_gpus:
                batches = shuffle_batches(list(batches), self.seed + epoch)

            batches = list(ShardedIterator(
                batches, self.num_shards, self.shard_id, fill_value=[]
            ))
            self.dataset.prefetch([i for s in batches for i in s])

            if shuffle and fix_batches_to_gpus:
                batches = shuffle_batches(batches, self.seed + epoch + self.shard_id)
        else:  # 默认情况:
            if shuffle:  # 若选择打乱数据集(训练时),则将存储有所有批次的元组(每个批次中有句子的索引号)全部打乱
                batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)  # 返回经过随机打乱的所有批次
            else:       # 若选择不打乱数据集(验证,测试时),则存储有所有批次的元组(每个批次中有句子的索引号)不变
                batches = self.frozen_batches
            batches = list(ShardedIterator(
                batches, self.num_shards, self.shard_id, fill_value=[]
            ))  # ShardedIterator为一个迭代器,其中为编号i以及i对应的batch(装满句子编号的列表);batches将其列表化

        if offset > 0 and offset >= len(batches):  # 若遍历完了一整轮,则返回None
            return None

        if self.num_workers > 0:  # 默认情况,则定义pythonwarnig类型
            os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

        return CountingIterator(
            torch.utils.data.DataLoader(
                self.dataset,             # 从中加载数据的数据集,即<LanguagePairDataset>类
                collate_fn=self.collate_fn,  # 合并示例列表来形成小批次,LanguagePairDataset.collater
                batch_sampler=batches[offset:],  # batches为一个列表,其中为编号i以及i对应的batch(装满句子编号的列表),我们选择从offset开始
                num_workers=self.num_workers,    # 所选择的进程数
            ),                  # 根据torch.data.DataLoader创建数据加载器,并在给定的数据集<LanguagePairDataset>类上提供的迭代器
            start=offset,
        )  # 返回可记录迭代数的迭代器,基于torch.data.DataLoader在给定的数据集<LanguagePairDataset>类上创建的数据加载器


class GroupedIterator(object):
    """Wrapper around an iterable that returns groups (chunks) of items. 关于可以返回items的组(块)的迭代器的包装器

    Args:
        iterable (iterable): iterable to wrap　# self._cur_epoch_itr,可记录迭代数的迭代器,基于torch.data.DataLoader在给定的数据集<LanguagePairDataset>类上创建的数据加载器
        chunk_size (int): size of each chunk  # 每个块的大小,默认为1
    """

    def __init__(self, iterable, chunk_size):
        self._len = int(math.ceil(len(iterable) / float(chunk_size)))  # 返回分块后的每一块迭代器的长度(向上取整)
        self.offset = int(math.ceil(getattr(iterable, 'count', 0) / float(chunk_size)))  # 若迭代器的count参数除以块数,得到每一块的offset;默认为0
        self.itr = iterable
        self.chunk_size = chunk_size

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        chunk = []
        try:
            for _ in range(self.chunk_size):
                chunk.append(next(self.itr))
        except StopIteration as e:
            if len(chunk) == 0:
                raise e
        return chunk


class ShardedIterator(object):
    """A sharded wrapper around an iterable, padded to length. # 关于可迭代,填充长度的分片包装器

    Args:
        iterable (iterable): iterable to wrap  # 需要包装的迭代器
        num_shards (int): number of shards to split the iterable into  # 将迭代器切片的分片数,默认为1
        shard_id (int): which shard to iterator over   　　　　　　　　　 # 在迭代器的哪块分片上,默认为0
        fill_value (Any, optional): padding value when the iterable doesn't
            evenly divide *num_shards* (default: None).　　　　　　　　　 # 当iterable不能平均分配*num_shards*个分片时的填充值
    """

    def __init__(self, iterable, num_shards, shard_id, fill_value=None):
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError('shard_id must be between 0 and num_shards')

        self._sharded_len = len(iterable) // num_shards  # 每个分片的批次总数,默认仅为1个分片
        if len(iterable) % num_shards > 0:  # 若迭代器不能平均分配*num_shards*个分片时:
            self._sharded_len += 1   # 将当前分配的批次总数+1

        # 创建一个迭代器,从每个可迭代对象中收集元素,如果可迭代对象的长度未对齐,将根据fillvalue填充缺失值,迭代持续到耗光最长的可迭代对象;
        # 如zip_longest('ABCD', 'xy', fillvalue='-') --> Ax By C- D-;
        # self.itr -> 编号i以及i对应的batch(装满句子编号的列表)
        self.itr = itertools.zip_longest(
            range(self._sharded_len),
            itertools.islice(iterable, shard_id, len(iterable), num_shards), # 创建一个迭代器返回从iterable里选中的元素(切片方式),将shard_id~shard_id+len(iterable)分成num_shards片
            fillvalue=fill_value,  # fillvalue为空列表[],用于填充缺少值
        )

    def __len__(self):
        return self._sharded_len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.itr)[1]
