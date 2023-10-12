# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from . import data_utils, FairseqDataset


logger = logging.getLogger(__name__)


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    # 利用当前遍历的batch中的全部源/目标句子的tensor列表,pad,eos符号,以及是否在源/目标左边填充; 将一维张量的列表转换为填充的二维张量
    # 训练时:若move_eos_to_beginning为true,则将全部句子的eos标记移到句子开头(第一个timestep);源端在左边填充,目标端不在左边填充;
    # 测试时:move_eos_to_beginning为False;源端在左边填充,目标端不在左边填充;
    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],  # 将当前遍历的batch中的全部源/目标句子tensor收集组成一个List
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])   # 将当前遍历的batch中全部句子对的ID转换为Tensor
    src_tokens = merge('source', left_pad=left_pad_source) # 利用当前遍历的batch中的全部源/目标句子的tensor列表,pad,eos符号,以及是否在源/目标左边填充; 将一维张量的列表转换为填充的二维张量
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])  # 将当前遍历的batch中全部源句子的长度转换为Tensor
    src_lengths, sort_order = src_lengths.sort(descending=True)  # 将当前遍历的batch中全部源句子的长度Tensor,按降序重新排序
    id = id.index_select(0, sort_order)   # 根据源句子的降序排序,重新排序当前batch中的所有句子对的ID
    src_tokens = src_tokens.index_select(0, sort_order)  # 将由源句子tensor列表转换后的源句子tensor填充二维张量,按照源句子降序的order重新排列每行

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:  # 默认情况,若存在目标句子的tensor列表:(测试时也存在)
        target = merge('target', left_pad=left_pad_target) # 和源端一样,利用当前遍历的batch中的全部源/目标句子的tensor列表.pad/eos符号,以及是否在源/目标左边填充; 将一维张量的列表转换为填充的二维张量
        target = target.index_select(0, sort_order) # 将由目标句子tensor列表转换后的目标句子tensor填充二维张量,按照源句子降序的order重新排列每行
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order) # 将当前遍历的batch中全部目标句子的长度转换为Tensor,并按源句子降序的order重新排序
        ntokens = sum(len(s['target']) for s in samples)  # ntokens为该批次所有目标句子的tokens总数

        if input_feeding:  # 若进行input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step # 创建目标的移位版本(即全部目标句子的eos移到开头),以将先前的输出令牌馈送到下一个解码器步骤
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )  # 和源端一样,利用当前遍历的batch中的全部源/目标句子的tensor列表.pad/eos符号,以及是否在源/目标左边填充; 将一维张量的列表转换为填充的二维张量
               # 区别: move_eos_to_beginning为True, 则将全部句子的eos标记移到句子开头(第一个timestep)
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)  # 将prev_output_tokens的tensor,按照源句子降序的order重新排列每行
    else:  # 若不存在目标句子的tensor列表,则定义ntokens为该批次所有源句子的tokens总数
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,  # 当前遍历的batch中全部句子对的ID的Tensor,按照源句子降序的order重新排列每行
        'nsentences': len(samples),  # 句子对数
        'ntokens': ntokens,    # 若存在目标句子的tensor列表,则为该批次所有目标句子的tokens总数
        'net_input': {
            'src_tokens': src_tokens,    # 当前遍历的batch中全部源句子的tensor所转换的二维填充向量,按照源句子降序的order重新排列每行
            'src_lengths': src_lengths,  # 将当前遍历的batch中全部源句子的长度的Tensor
        },
        'target': target,  # 当前遍历的batch中全部目标句子的tensor所转换的二维填充向量,按照源句子降序的order重新排列每行
    }
    if prev_output_tokens is not None: # 若prev_output_tokens不为空,则添加到['net_input']内
        batch['net_input']['prev_output_tokens'] = prev_output_tokens # 创建目标的移位版本(即全部目标句子的eos移到开头)

    if samples[0].get('alignment', None) is not None:  # 若当前遍历的batch中存在'alignment';非默认情况
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        append_bos=False, eos=None
    ):
        if tgt_dict is not None:  # 若目标字典不为None,则保证两个字典的3种特殊符号一致
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src  # 源和目标端数据集,prefix+src/tgt下.idx和.bin文件读取的MMapIndexedDataset类,
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)  # 将源和目标句子长度列表,转换成np数组
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict  # 源和目标字典Dictionray类
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source  # 源和目标是否从左开始填充(pad符号)
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions  # 最大源和目标句子长度
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle                            # 批处理前是否将数据集句子打乱,默认为True
        self.input_feeding = input_feeding                # 是否进行input_feeding,即创建一个目标的shifted版本传递到模型用于teacher forcing,默认为True
        self.remove_eos_from_source = remove_eos_from_source  # 是否从源末尾删除eos(如果存在),默认为False
        self.append_eos_to_target = append_eos_to_target    # 是否将eos附加到目标句子末尾(如果不存在),默认为False
        self.align_dataset = align_dataset                  # 包含对齐的数据集
        if self.align_dataset is not None:                  # 若对齐数据集不为空,则同时需要源和目标
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos                       # 是否将bos附加到源/目标句子的开头
        self.eos = (eos if eos is not None else src_dict.eos())   # 若所传入eos符号对应idx不为None,则用所传入的,否则为源/目标的eos符号对应idx

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch. 合并样本列表以形成一个mini-batch

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order   # 原始输入顺序中的示例ID
                - `ntokens` (int): total number of tokens in the batch         # 在该批次中tokens总数
                - `net_input` (dict): the input to the Model, containing keys: # 模型的输入,包含如下三者:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                    # 形状为（bsz,src_len）的源句子的tokens填充后的二维张量, 如果* left_pad_source*为True,则填充将出现在左侧
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                    # 形状为（（bsz）`的每个源句的未填充长度的一维张量
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                    # 目标句子的tokens填充后的二维张量,向右移动一个位置以进行教师强迫,形状为（bsz,tgt_len）
                    # 若*input_feeding*为False,则此键将不存在;如果*left_pad_target*为True,填充将出现在左侧

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                  # 形状为（bsz,tgt_len）的目标句子的tokens填充后的二维张量,如果*left_pad_target*为True,填充将出现在左侧
        """
        # 通过当前遍历到的batch,pad/eos符号,以及是否在源/目标左边填充,以及是否进行input_feeding,即创建一个目标的shifted版本传递到模型用于teacher forcing
        # 返回: batch = {
        #     'id': id,  # 当前遍历的batch中全部句子对的ID的Tensor,按照源句子降序的order重新排列每行
        #     'nsentences': len(samples),  # 句子对数
        #     'ntokens': ntokens,  # 若存在目标句子的tensor列表,则为该批次所有目标句子的tokens总数
        #     'net_input': {
        #         'src_tokens': src_tokens,  # 当前遍历的batch中全部源句子的tensor所转换的二维填充向量,按照源句子降序的order重新排列每行
        #         'src_lengths': src_lengths,  # 将当前遍历的batch中全部源句子的长度的Tensor(带eos)
        #     },
        #     'target': target,  # 当前遍历的batch中全部目标句子的tensor所转换的二维填充向量,按照源句子降序的order重新排列每行
        # }
        # if prev_output_tokens is not None:  # 若prev_output_tokens不为空,则添加到['net_input']内
        #     batch['net_input']['prev_output_tokens'] = prev_output_tokens  # 创建目标的移位版本(即全部目标句子的eos移到开头)
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching. 返回一个示例中的token数(源和目标之间的最大值),此值用于在批处理期间强制执行``--max-tokens''"""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``. 以tuple返回当前idx对应的源和目标示例的大小,该值用于在使用--max-positions过滤数据集时"""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order. 返回索引的有序列表,将根据此顺序构建所有批次"""
        if self.shuffle: # 若打乱数据集
            indices = np.random.permutation(len(self))  # 输入一个数,在此数范围内生成一个随机序列
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:  # 若目标数据集所有句子长度列表不为空,在随机打乱数据集后,按照目标句子的长度从小到大再排序
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')] # 在随机打乱数据集后,按照目标句子的长度从小到大再排序,最后按照源句子的长度从小到大再排序,返回当前数据集的句子序列号

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)
