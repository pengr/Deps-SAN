# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from .learned_positional_embedding import LearnedPositionalEmbedding
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding


def PositionalEmbedding(
        num_embeddings: int,  # 为源/目标端最大的tokens数
        embedding_dim: int,
        padding_idx: int,
        learned: bool = False,
):
    if learned:  # 非默认情况
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately # 如果指定padding_idx,则将嵌入ID偏移此索引,并适当调整num_embeddings
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation. 学习型的位置嵌入(学习pos值)
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:  # 默认为vaswani的正弦函数的postionalEmbedding,通过词嵌入维数,填充符号idx,以及输入大小(源/目标端最大的tokens数+填充符号idx+1)->1026
        m = SinusoidalPositionalEmbedding(
            embedding_dim, padding_idx, init_size=num_embeddings + padding_idx + 1,
        )  # 返回由max_source/target_position决定长度的正弦型位置嵌入,填充符号(pad)的嵌入向量被置0
    return m
