# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional

import torch
import torch.onnx.operators
from fairseq import utils
from torch import Tensor, nn


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
       该模块产生任意长度(非1e5,由max_source/target_position决定)的正弦型位置嵌入,填充符号(pad)的嵌入向量被置0
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim  # 词嵌入维数
        self.padding_idx = padding_idx      # 填充符号<pad>的idx(1)
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )  # 返回正弦曲线嵌入(填充符号<pad>对应idx行被置0)
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))  # 在内存中定一个常量FloatTensor(1),用于为其他tensor提供类型
        self.max_positions = int(1e5)  # 手动设置最大的源位置(1e5),由于我们使用max_source/target_position代替,故不起作用

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings. 建立正弦曲线嵌入

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        # 这与tensor2tensor中的实现相匹配,但与vaswani论文第3.5节描述稍有不同
        所复现的效果如[sin(0),...,sin(0), cos(0),...,cos(0)
                    sin(1/10000^(1/255)),...,sin(1/10000^(256/255)), cos(1/10000^(1/255)),...,cos(1/10000^(256/255))
                                                                ...
                    sin(1026/10000^(1/255)),...,sin(1026/10000^(256/255)), cos(1026/10000^(256/255)),...,cos(1026/10000^(256/255))
        与vaswani论文区别: pos范围为0~1026; 原来2i/d(model)=i/((d(model))/2)=i/256,在这里我们为255;
        原来是sin()和cos()元素穿插,如sin(1/10000^(1/255)),cos(1/10000^(1/255)),sin(2/10000^(1/255));现在是sin矩阵和cos矩阵整个级联
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:  # 若所传入的词嵌维数不能分半,则需要补充一个零行
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:  # 若填充符号不为空,则该填充符号的对应索引idx,所映射到词嵌入矩阵为零向量
            emb[padding_idx, :] = 0
        return emb  # 返回正弦曲线嵌入(填充符号<pad>对应idx行被置0)

    def forward(
        self,
        input,  # 源/目标端句子tensor的二维填充向量
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]. 输入的大小应为[bsz x seqlen]"""
        bspair = torch.onnx.operators.shape_as_tensor(input)  # 获得input的两个维度
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len  # 最大pos为该批的最大句子长度+1+填充符号idx
        if self.weights is None or max_pos > self.weights.size(0):  # 若正弦曲线嵌入为空,或者最大pos>所设置的最大tokens数+1+填充符号idx(1026);非默认情况
            # recompute/expand embeddings if needed 必要时需要重新计算/扩展嵌入
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)  # 将正弦曲线嵌入(填充符号<pad>对应idx行置0)全部元素,与torch自带_float_tensor类型保持一致

        if incremental_state is not None:  # 若所传入的incremental_state不为None;测试时为{}
            # positions is the same for every token when decoding a single step 当解码单个步骤时,对于每个token的位置均相同
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len # 测试时,pos为seq_len(根据待生成目标词idx序列解码到了第几个时间步)
            if self.onnx_trace: # 测试时为False;非默认情况
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)
            # 测试时此处即返回,从正弦曲线嵌入矩阵的第0个维度中,当前解码时间步+pad_idx上选取PE(x)数据,由于仅选取单个时间步需重新扩维为[bsz,embbed_dim]

        positions = utils.make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )  # 将源句子tensor的二维填充向量中,非填充符号替换为其位置编号,位置编号从padding_idx+1开始,填充符号将被忽略(位置编号还是1)
            # 类似:               (累加)            (float)               (*mask)                   (加pad_idx)
            # mask: [[0,0,1,1,1], -> [[0,0,1,2,3],  -> [[0.,0.,1.,2.,3.], -> [[0.,0.,1.,2.,3.], -> [[1.,1.,2.,3.,4.],
            #        [1,1,1,1,1]]     [1,2,3,4,5]]      [1.,2.,3.,4.,5.]]     [1.,2.,3.,4.,5.]]     [2.,3.,4.,5.,6.]]
        if self.onnx_trace:  # 若onnx_trace为True;非默认情况
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, positions.view(-1)) # 将源句子tensor的二维填充向量的位置编号展平
            .view(bsz, seq_len, -1)
            .detach()
        )  # 从正弦曲线嵌入矩阵的第0个维度中,在展平后的源句子tensor的二维填充向量的位置编号中选取PE(x)数据,并扩维+detach(从计算图复制,共用地址内存,不进行梯度更新)
