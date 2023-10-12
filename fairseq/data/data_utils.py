# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import contextlib
import itertools
import logging
import os
import sys
import types

import numpy as np


logger = logging.getLogger(__name__)


def infer_language_pair(path):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    # 从上述文件名推断语言对
    src, dst = None, None
    for filename in os.listdir(path):
        parts = filename.split('.')
        if len(parts) >= 3 and len(parts[1].split('-')) == 2:
            return parts[1].split('-')
    return src, dst


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor. 将一维张量的列表转换为填充的二维张量"""
    size = max(v.size(0) for v in values)  # 遍历当前batch中的全部源/目标句子列表,得到该批的最大句子长度srclen
    res = values[0].new(len(values), size).fill_(pad_idx) # len(values)即为当前batch的句子数(batch size),创建一个值全为填充符号idx的tensor矩阵->(batch,srclen/tgtlen)

    def copy_tensor(src, dst):  # 将src复制到dst中
        assert dst.numel() == src.numel()  # 首先判断src和dst中元素个数是否一致
        if move_eos_to_beginning:  # 训练时为true,将src末尾的eos标记移到dst开头(第一个timestep),其余按顺序填充到dst中
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:  # 测试时为True,则dst与传入的src一致
            dst.copy_(src)

    for i, v in enumerate(values):  # 取出当前遍历的batch中的每个源/目标句子v,以及在该批次中其为第i句
        # 若left_pad为True/Fasle,源和目标句子v对应的res矩阵行(第i行)的开头/末尾全部为pad_idx(1),我们选择将从右/左侧开始数句子长度;
        # 若move_eos_to_beginning为True,则无论该句子放置在res矩阵行的开头或末尾,均在该句子的开头填充eos标记,句子v剩余按从左->右顺序填充到eos标记后
        # 若move_eos_to_beginning为False,则无论该句子放置在res矩阵行的开头或末尾,直接将整个句子v按从左->右顺序填充到eos标记后
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def load_indexed_dataset(path, dictionary, dataset_impl=None, combine=False, default='cached'):
    """A helper function for loading indexed datasets. 用于加载索引数据集的辅助函数

    Args:
        path (str): path to indexed dataset (e.g., 'data-bin/train') # 索引数据集的路径
        dictionary (~fairseq.data.Dictionary): data dictionary  # 数据字典
        dataset_impl (str, optional): which dataset implementation to use. If   # 使用怎样的数据输出格式,若未提供则根据infer_dataset_impl自动推算得到
            not provided, it will be inferred automatically. For legacy indexed # 对于legacy indexed data,我们采用'cached'作为默认
            data we use the 'cached' implementation by default.
        combine (bool, optional): automatically load and combine multiple  自动加载并合并多个数据集
            datasets. For example, if *path* is 'data-bin/train', then we will
            combine 'data-bin/train', 'data-bin/train1', ... and return a
            single ConcatDataset instance.
    """
    from fairseq.data.concat_dataset import ConcatDataset
    import fairseq.data.indexed_dataset as indexed_dataset

    datasets = []   # 用于存储path路径下的所有dataset(针对如'data-bin/train', 'data-bin/train1'情况)
    for k in itertools.count():
        path_k = path + (str(k) if k > 0 else '')  # 取出当前数据路径,如验证集'data-bin/iwslt14.tokenized.de-en/valid.de-en.de'

        dataset_impl_k = dataset_impl
        if dataset_impl_k is None:  # 根据infer_dataset_impl自动推算得到当前输出格式,判断是否为raw文本,以及根据其_HDR_MAGIC类型判断为cached, mmap, None
            dataset_impl_k = indexed_dataset.infer_dataset_impl(path_k)

        dataset = indexed_dataset.make_dataset(
            path_k,
            impl=dataset_impl_k or default,
            fix_lua_indexing=True,
            dictionary=dictionary,
        )  # 完成对<MMapIndexedDataset>类的初始化, 对path路径下.idx和.bin文件的二进制读取,采用np.memmap()方法小段读入
        if dataset is None:
            break
        logger.info('loaded {} examples from: {}'.format(len(dataset), path_k))  # 打印在当前数据路径path_k下有多少个句子示例
        datasets.append(dataset)
        if not combine:  # 若该path路径下不存在多个数据集,则直接break
            break
    if len(datasets) == 0:  # 若datasets为空,则返回None
        return None
    elif len(datasets) == 1:  # 若datasets存在一个dataset,则返回它(默认情况)
        return datasets[0]
    else:                    # 若datasets存在多个dataset的情况,则进行合并(定义ConcatDataset类)
        return ConcatDataset(datasets)


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds): # 使用指定的seed为所有NumPy参数seed,然后设定状态以记录下数组被打乱的操作
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.
     类似于：func：`filter`, 但是在``filtered''中收集被过滤掉的元素
    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered  # 为应过滤的元素返回``False''的函数,即传入的_filter_by_size_dynamic.check_size
        iterable (iterable): iterable to filter  # 用于过滤的迭代器,即数据集索引的有序列表
        filtered (list): list to store filtered elements  # 用于存储被过滤掉的句子的列表
    """
    for el in iterable:
        if function(el):  # 返回该idx对应的源和目标示例是否为None或少于最大tokens数的bool标记,满足则不被当成过长序列
            yield el
        else:           # 返回该idx对应的源和目标示例大于最大tokens数的bool标记,满足则被当成过长序列,在这里我们收集需被过滤掉的示例
            filtered.append(el)


def _filter_by_size_dynamic(indices, size_fn, max_positions, raise_exception=False): # 根据数据集索引的有序列表,数据集类的size(),以及句子最大tokens数元组,以及是否为长句子过滤raise异常
    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):  # 若max_positions类型为float或者int
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, dict):   # 若max_positions类型为dict
            idx_size = size_fn(idx)
            assert isinstance(idx_size, dict)
            intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
            return all(
                all(a is None or b is None or a <= b
                    for a, b in zip(idx_size[key], max_positions[key]))
                for key in intersect_keys
            )
        else:  # max_positions默认为tuple类型
            # Hacky as heck, for the specific case of multilingual training with RoundRobin. 对于使用RoundRobin进行多语种训练的特定案例
            if isinstance(size_fn(idx), dict) and isinstance(max_positions, tuple):  # size_fn(idx)以tuple返回当前idx对应的源和目标示例的大小
                return all(
                    a is None or b is None or a <= b
                    for a, b in zip(size_fn(idx).values(), max_positions)
                )
            # For MultiCorpusSampledDataset, will generalize it later 对于MultiCorpusSampledDataset,稍后将对其进行概括
            if not isinstance(size_fn(idx), Iterable):  # size_fn(idx)以tuple返回当前idx对应的源和目标示例的大小
                return all(size_fn(idx) <= b for b in max_positions)
            return all(  # 默认情况,遍历zip(当前idx对应的源和目标示例的大小,源和目标最大tokens数),返回该idx对应的源和目标示例是否为None或少于最大tokens数的bool标记,满足则不被当成过长序列
                a is None or b is None or a <= b
                for a, b in zip(size_fn(idx), max_positions)
            )
    ignored = []  # 用于存储被过滤掉的句子
    itr = collect_filtered(check_size, indices, ignored)  # 遍历数据集索引的有序列表,若当前遍历的源和目标示例大于最大tokens数,则被当成过长序列不保存在indices列表,而被收集到ignored中,其余句子仍保留在indices列表
    indices = np.fromiter(itr, dtype=np.int64, count=-1)  # np.fromiter从上述非过长句子的索引有序列表中建立ndarray一维数组
    return indices, ignored  # 返回非过长句子的索引有序列表的ndarray一维数组,以及收集了过长句子示例的列表


def filter_by_size(indices, dataset, max_positions, raise_exception=False):
    """
    Filter indices based on their size.

    Args:
        indices (List[int]): ordered list of dataset indices # 数据集索引的有序列表
        dataset (FairseqDataset): fairseq dataset instance   # 用于分批的数据集,<fairseq.data.language_pair_dataset.LanguagePairDataset>
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.             # 过滤掉大于此大小的句子示例,比较是按组件进行的
        raise_exception (bool, optional): if ``True``, raise an exception if
            any elements are filtered (default: False).     # 如果为True,则在过滤任何元素时引发异常
    """
    if isinstance(max_positions, float) or isinstance(max_positions, int):  # 若max_positions类型为float或者int
        if hasattr(dataset, 'sizes') and isinstance(dataset.sizes, np.ndarray):
            ignored = indices[dataset.sizes[indices] > max_positions].tolist()
            indices = indices[dataset.sizes[indices] <= max_positions]
        elif hasattr(dataset, 'sizes') and isinstance(dataset.sizes, list) and len(dataset.sizes) == 1:
            ignored = indices[dataset.sizes[0][indices] > max_positions].tolist()
            indices = indices[dataset.sizes[0][indices] <= max_positions]
        else:
            indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)
    else:                                                                 # max_positions默认为tuple类型
        indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions) # 返回非过长句子的索引有序列表的ndarray一维数组,以及收集了过长句子示例的列表

    if len(ignored) > 0 and raise_exception:  # 若收集了过长句子示例的列表不为空,则报告如下:
        raise Exception((
            'Size of sample #{} is invalid (={}) since max_positions={}, '
            'skip this example with --skip-invalid-size-inputs-valid-test'
        ).format(ignored[0], dataset.size(ignored[0]), max_positions)) # 由于max_positions,样本{该句子}的大小无效(= {}),使用--skip-invalid-size-inputs-valid-test=True跳过此示例
    if len(ignored) > 0:
        logger.warning((
            '{} samples have invalid sizes and will be skipped, '
            'max_positions={}, first few sample ids={}'
        ).format(len(ignored), max_positions, ignored[:10]))
    return indices  # 返回非过长句子的索引有序列表的ndarray一维数组


def batch_by_size(
    indices, num_tokens_fn, max_tokens=None, max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.
    产生按size分类的indices的mini-batches, 批次中可能包含不同长度的序列
    Args:
        indices (List[int]): ordered list of dataset indices  # 数据集索引的有序列表,即非过长句子的索引有序列表的ndarray一维数组
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index　　　　　　　　　　　　　　　　　　　　　　# 返回给定idx对应的示例的token数的函数,即dataset.num_tokens
        max_tokens (int, optional): max number of tokens in each batch　
            (default: None).　　　　　　　　　　　　　　　　　　　　# 每批中的最大token数
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).                           # 每批中的最大sents数
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).                 # 要求一个批次数为N的倍数
    """
    try:
        from fairseq.data.data_utils_fast import batch_by_size_fast
    except ImportError:
        raise ImportError(
            'Please build Cython components with: `pip install --editable .` '
            'or `python setup.py build_ext --inplace`'
        )

    max_tokens = max_tokens if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):  # 若传入的数据集索引的有序列表不是python生成器类型,则再np.fromiter一次;非默认情况
        indices = np.fromiter(indices, dtype=np.int64, count=-1)
    # 注!!!!!!!!!!由于batch_by_size_fast是由cython代码编写,无法调试
    return batch_by_size_fast(indices, num_tokens_fn, max_tokens, max_sentences, bsz_mult) # 返回存储所有批次的列表,每个批次中有句子的编号


def process_bpe_symbol(sentence: str, bpe_symbol: str):  # sentence为所遍历到的源/目标句子的原始token索引tensor经过对应转换后的string序列
    if bpe_symbol == 'sentencepiece':
        sentence = sentence.replace(' ', '').replace('\u2581', ' ').strip()
    elif bpe_symbol == '_EOW':
        sentence = sentence.replace(' ', '').replace('_EOW', ' ').strip()
    elif bpe_symbol is not None:  # 默认情况;若设置--remove_bpe,则arg.remove_bpe即bpe_symbol为"@@ "
        sentence = (sentence + ' ').replace(bpe_symbol, '').rstrip() # 将sentence中所出现的"@@ "替换为'',实则就是去掉"@@ "
    return sentence # 返回移除bpe符号的所遍历到的源/目标句子的原始token索引tensor经过对应转换后的string序列
