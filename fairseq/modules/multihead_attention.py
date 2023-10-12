# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor, nn
from torch.nn import Parameter
from fairseq.incremental_decoding_utils import with_incremental_state


@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 获取词嵌入维数d(model)
        self.kdim = kdim if kdim is not None else embed_dim  # 获取键的维度dk,若未给定则与词嵌入维数相同,暂时未分为d(model)/h
        self.vdim = vdim if vdim is not None else embed_dim  # 获取值的维度dv,若未给定则与词嵌入维数相同,暂时未分为d(model)/h
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim  # 判断键/值的维度dk和dv是否均和词嵌入维数一致的标记

        self.num_heads = num_heads  # 获取注意力头数
        self.dropout = dropout      # 获取self-attn的dropout率,由args.attention_dropout给定
        self.head_dim = embed_dim // num_heads  # 获取每个注意力头的隐藏态维数,即为d(model)/h
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5   # 获取self-attn的缩放比例dk^(-0.5)

        self.self_attention = self_attention   # 获取是否为self-attn的标记
        self.encoder_decoder_attention = encoder_decoder_attention  # 获取是否为enc-dec-side self-attn的标记

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )  # 若为self-attn(无论源或目标端),查询,键,值三者的维度必须一致

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)  # 获取用于映射查询,键,值的WQ,WK,WV三个矩阵,带偏置向量
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 获取用于映射输出O(即每个注意力头的输出Z的级联)的WO的矩阵,带偏置向量

        if add_bias_kv:  # 将在dim = 0处添加的键和值序列的偏置(独立于nn.Linear自带的bias);非默认情况
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn  # 是否在dim = 1处将新的零批次添加到键和值序列;非默认情况

        self.reset_parameters()   # 重新设置(初始化)multi-headed attn的参数,其中WO的bias初始化为0向量

        self.onnx_trace = False  # 设置onnx_trace标记为False

        self.enable_torch_version = False  # 是否允许采用pytorch底层对multi-headed attn的实现
        if hasattr(F, "multi_head_attention_forward"):  # 若torch.nn.functional存在multi_head_attention_forward,则调用其进行操作
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:  # 若查询,键,值三者维度均一致(无论是源和目标的self-attn,还是目标的enc-dec attn),则按带gain缩放的xavier初始化来初始化WQ,WK,WV
            # Empirically observed the convergence to be much better with
            # the scaled initialization 凭经验观察到, 按比例缩放初始化会更好地收敛
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))  # xavier初始化服从均匀分布U(−a,a),a = gain * sqrt(6/fan_in+fan_out)
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:   # 若查询,键,值三者维度并非一致(即目标的enc-dec-side self-attn),则直接xavier初始化WQ,WK,WV
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)  # 直接xavier初始化WO
        if self.out_proj.bias is not None:  # 若输出O(即每个注意力头的输出Z的级联)的矩阵WO存在偏置,则将该偏置的初始化设0
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:  # 若手动为键,值添加了偏置(即不用nn.Linear自带的bias);非默认情况
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,  # 用于注意力的静态键和值,默认为False
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel 输入形状: src_len/tgt_len * batch * embedd_size

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.  # 值掩码矩阵以排除被填充的键(适用源/目标),形状为(batch,src_len),其中填充元素被指定为1
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).  # 是否返回在所有注意力头上的平均注意力权重
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            # 通常用于实现causal attention,time-shifted未来掩码矩阵可防止注意力在未来出现,;默认为None,仅当decoder的self-attn中不为None(右上三角矩阵,对角线及以下均为0,其余为-inf);
            # 2D/3D掩码用于屏蔽注意某些位置;将为所有批次广播2D掩码,而3D掩码允许为每个批次的条目指定不同的掩码
            before_softmax (bool, optional): return the backup attention
                weights and values before the attention softmax.     # 在attention进行softmax之前返回原始注意力权重和values
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads. # 返回每个注意力头的注意力权重,必然包含*need_weights*; 默认值: 返回在所有注意力头上的平均注意力权重
        """
        if need_head_weights:    # 若需返回每个注意力头的注意力权重:
            need_weights = True  # 则首先需返回在所有注意力头上的平均注意力权重

        tgt_len, bsz, embed_dim = query.size()  # 从查询中取出目标序列长度(解码端),批次大小,隐藏态向量维度
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]  # 确认查询的维度无误

        if (
            self.enable_torch_version      # 若允许采用pytorch底层对multi-headed attn的实现
            and not self.onnx_trace        # onnx_trace标记为False
            and incremental_state is None  # incremental_state为None;测试时为{}
            and not static_kv              # static_kv为False(用于注意力的静态键和值),enc-dec self-attn时为True
        ):
            assert key is not None and value is not None  # 若键,值万不为None:
            return F.multi_head_attention_forward(  # 通过如下参数调用pytorch底层对multi-headed attn进行实现
                query,            # 查询
                key,              # 键
                value,            # 值
                self.embed_dim,   # 隐藏态向量维度d(model)
                self.num_heads,   # 注意力头数
                torch.empty([0]),  # 输入映射权重
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),  # 输入映射偏置
                self.bias_k,           # 将在dim = 0处添加的键和值序列的偏置(独立于nn.Linear自带的bias)
                self.bias_v,
                self.add_zero_attn,    # 在dim = 1处将新的零批次添加到键和值序列
                self.dropout,          # self-attn的dropout率,由args.attention_dropout给定
                self.out_proj.weight,  # 用于映射输出O(即每个注意力头的输出Z的级联)的WO的矩阵,带偏置向量
                self.out_proj.bias,    # 用于映射输出O(即每个注意力头的输出Z的级联)的WO的偏置向量,初始化为0
                self.training,         # 如果为``True'',则采用dropout(验证/测试时不采用Dropout)
                key_padding_mask,      # 值掩码矩阵以排除被填充的键(适用源/目标),形状为(batch,src_len),其中填充元素被指定为1
                need_weights,          # 是否返回在所有注意力头上的平均注意力权重
                attn_mask,             # 类似future_mask, 通常用于实现causal attention,time-shifted未来掩码矩阵可防止注意力在未来出现;
                                       #  2D/3D掩码用于屏蔽注意某些位置;将为所有批次广播2D掩码,而3D掩码允许为每个批次的条目指定不同的掩码
                use_separate_proj_weight=True,     # 该功能接受用于查询,键,值的不同形式的映射权重;若为False,则使用in_proj_weight(即torch.empty([0])),它是WQ,WK,WV的组合
                q_proj_weight=self.q_proj.weight,  # 输入投影权重(WQ,WK,WV),注: 输入映射偏置为torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias))
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )  # 返回multi-headed attn的输出attn_output->Concat(head1, ..., headh)WQ, attn_output_weights->softmax((QK^T)/dk^(-0.5))
               # 若need_weights=True,后者为在所有注意力头上的平均注意力权重(即attn_output_weights.view(bsz, num_heads, tgt_len, src_len)按dim=1取平均)
            # 解码器与编码器的self-attn区别: query,key,value,key_padding_mask均为目标端的;由于need_weights为False,作为中间模块不返回注意力权重;
            # attn_mask为time-shifted的未来掩码矩阵(右上三角矩阵,对角线及以下均为0,其余为-inf),通过attn_output_weights += attn_mask对当前目标词的未来序列进行掩码

        ############### 不调用torch底层手动实现multi-headed attn,适用于enc-dec self-attn(其中static_kv为True); ###################
        ############### 也适用于测试时,解码端的self-attn,enc-dec attn,因为测试时解码端incremental_state不为None ###################
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            # 训练/验证时incremental_state为None故值返回{};
            # 测试时,第一个解码时间步self-attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
            #  incremental_state为{}故值返回{};
            # 测试时,第一个解码时间步enc-dec attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
            #  incremental_state存储了{第一层self-attn的唯一码+'attn_state'：prev_key/prev_value/prev_key_padding_mask};
            #  由于第一层enc-dec attn的唯一码+'attn_state'不在incremental_state内,故值返回{}; 以此类推,第一个时间步内6层self-attn和enc-dec attn中值返回{};

            # 后续每个解码时间步,incremental_state键名不会增加,仍是第一个解码时间步6个self-attn和6个enc-dec attn对应的唯一码+'attn_state',
            # 但self-attn对应的值发生改变:到目前解码时间步为止(包括当前)的所存储的prev_key/prev_value/prev_key_padding_mask的级联;enc-dec attn对应的值与第一个解码时间步一样;
            # 故result返回为对应第几层的self-attn或enc-dec attn的state字典{唯一码+'attn_state'：存储prev_key/prev_value/prev_key_padding_mask}
            if saved_state is not None and "prev_key" in saved_state: # 测试时,若此时"prev_key"在saved_state内;(后续解码时间步self-attn和enc-dec attn均成立,但self-attn的static_kv为False)
                # previous time steps are cached - no need to recompute
                # key and value if they are static # 先前的时间步已缓存-如果它们是静态的,则无需重新计算key和value
                if static_kv:  # 测试时,若static_kv标记为True,且当前为enc dec attn而非self-attn,则无需对key和value进行计算
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:   # 训练,验证时为None,saved_state用于缓存先前时间步的key,value,padding_mask
            saved_state = None

        if self.self_attention:  # 若为self-attention,由于query,key,value相同,直接由query则完成WQ/WK/WV(Ex+PE(x))->Q,K,V;
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:  # 若为encoder-decoder,由于query与key,value不同:
            # encoder-decoder attention
            q = self.q_proj(query)  # 单独由query完成WQ(Ex+PE(x))->Q
            if key is None:   # 若key和value均为None,则无需计算K,V
                assert value is None
                k = v = None
            else:           # 由于key,value相同,直接由key完成WK/WV(Ex+PE(x))->K,V;
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:  # 若没有设置为self-attn或者enc-dec attn标记:
            assert key is not None and value is not None  # 只要key,value不为空:
            q = self.q_proj(query)   # 单独由query完成WQ(Ex+PE(x))->Q
            k = self.k_proj(key)     # 单独由key完成WK(Ex+PE(x))->K
            v = self.v_proj(value)   # 单独由value完成WV(Ex+PE(x))->V
        q *= self.scaling     # 完成q/dk^(-0.5)的缩放

        if self.bias_k is not None:  # 若将在dim = 0处添加的键和值序列的偏置(独立于nn.Linear自带的bias);非默认情况
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])  # 将自定义的bias_k/bias_v扩维成(1, bsz, embed_dim),在dim=0中添加进k/v
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:  # 若attn_mask不为None,则在dim=1中为attn_mask添加一列零向量;测试时为None,无需未来掩码矩阵(无论sef-attn或enc-dec attn)
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:  # 若key_padding_mask不为None,则在dim=1中为key_padding_mask添加一列零向量
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()    # 注意需要做view().transpose()操作前,需要先对tensor进行拷贝,contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )  # 将Q转换形状,[tgt_len, bsz, embed_dim]->[bsz * num_heads, tgt_len, head_dim]
        if k is not None:  # 若K,V不为None,则将K,V转换形状,[src_len, bsz, embed_dim]->[bsz * num_heads, src_len, head_dim]
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # 训练/验证时incremental_state为None故值返回{};
            # 测试时,第一个解码时间步self-attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
            #  incremental_state为{}故值返回{};
            # 测试时,第一个解码时间步enc-dec attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
            #  incremental_state存储了{第一层self-attn的唯一码+'attn_state'：prev_key/prev_value/prev_key_padding_mask};
            #  由于第一层enc-dec attn的唯一码+'attn_state'不在incremental_state内,故值返回{}; 以此类推,第一个时间步内6层self-attn和enc-dec attn中值返回{};

            # 后续每个解码时间步,incremental_state键名不会增加,仍是第一个解码时间步6个self-attn和6个enc-dec attn对应的唯一码+'attn_state',
            # 但self-attn对应的值发生改变:到目前解码时间步为止(包括当前)的所存储的prev_key/prev_value/prev_key_padding_mask的级联;enc-dec attn对应的值与第一个解码时间步一样;
            # 故result返回为对应第几层的self-attn或enc-dec attn的state字典{唯一码+'attn_state'：存储prev_key/prev_value/prev_key_padding_mask}
            # ---------------------------------------------------------------------------------------------------------------------
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            # 第一个解码时间步saved_state为{}(无论self-attn/enc-dec attn),故为Fasle; 后续的解码时间步(无论self-attn/enc-dec attn)均成立;
            # 从saved_state中取出前一时间步的prev_key,prev_value,也即先前解码时间步的k和v,并重新调整回(bsz*num_heads, src_len/tgt_len, head_dim)
            # 从saved_state中取出前一时间步的prev_key_padding_mask,也即先前解码时间步的key_padding_mask
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:  # 若为后续的解码时间步的enc-dec attn,则直接将先前解码时间步的k和v复制给当前解码时间步,因为enc-dec attn中k,v来自编码器的输出(即最后一个编码器输出)encoder_output
                    k = prev_key
                else:  # 若为后续的解码时间步的self-attn,则将先前解码时间步的k和v和当前解码时间步进行级联,因为self-attn中k,v来自目标序列的学习,需要考虑先前生成的全部目标词信息
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]  # 若为后续的解码时间步的self-attn,则由于是顺序解码不对目标进行填充掩码一直为None;若为后续的解码时间步的enc-dec attn,则一直是源端填充掩码矩阵
            assert k is not None and v is not None  # 此时k和v不能为None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,  # 测试时,self-attn由于是顺序解码不对目标进行填充掩码故为None,enc-dec attn为源端填充掩码矩阵
                prev_key_padding_mask=prev_key_padding_mask,  # 测试时第一个解码时间步为None(无论self-attn/enc-dec attn),之后与key_padding_mask一致
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,  # 测试时,self-attn中static_kv为Fasle,enc-dec attn中static_kv为为True
            )  # 测试时,self-attn中key_padding_mask返回None, enc-dec attn中key_padding_mask返回float32类型的源端填充掩码矩阵

            # 将当前解码时间步的k和v转换形状为[bsz, num_heads, src_len/tgt_len, head_dim],key_padding_mask
            # 存储到saved_state["prev_key"/"prev_value"/"prev_key_padding_mask"],作为先前解码时间步的k,v,源端端填充矩阵
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
            # 训练/验证时incremental_state为None;
            # 测试时,第一个解码时间步self-attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
            #  incremental_state为{};
            # 测试时,第一个解码时间步enc-dec attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
            #  incremental_state存储了{第一层self-attn的唯一码+'attn_state'：prev_key/prev_value/prev_key_padding_mask};
            # 测试时,第一个解码时间步第二层self-attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
            #  incremental_state存储了{第一层self-attn和enc-dec attn的唯一码+'attn_state'：prev_key/prev_value/prev_key_padding_mask}; 以此类推;

            # 后续每个解码时间步,incremental_state键名不会增加,仍是第一个解码时间步6个self-attn和6个enc-dec attn对应的唯一码+'attn_state',
            # 但self-attn对应的值发生改变:到目前解码时间步为止(包括当前)的所存储的prev_key/prev_value/prev_key_padding_mask的级联;enc-dec attn对应的值与第一个解码时间步一样;

        assert k is not None  # 从这里开始,K不能继续为None; K可以由外部传入并计算,也可以由saved_state从先前缓存的字典中提取
        src_len = k.size(1)  # 取出K中第1个维度src_len(解码器中self-attn为目标句子长度,enc-dec attn为源句子长度)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types. 这是解决不支持Optional类型的fork/join并行性的一种解决方法
        if key_padding_mask is not None and key_padding_mask.dim() == 0:  # 若key_padding_mask的维数为0,则key_padding_mask为None
            key_padding_mask = None

        if key_padding_mask is not None:  # 测试时,self-attn由于是顺序解码不对目标进行填充掩码为None;enc-dec attn为float32类型的源端填充掩码矩阵
            assert key_padding_mask.size(0) == bsz  # 确认key_padding_mask的两个维度w无误;
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:   # 若add_zero_attn不为空;非默认情况
            assert v is not None
            src_len += 1     # 若存在新的零批次的添加,则K,V序列在src_len这一维度上需要+1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)  # 在dim = 1处将新的零批次添加到K和V序列
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:  # 若attn_mask不为None,由于存在新的零批次的添加,则在dim=1中为attn_mask添加一列零向量
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:  # 若key_padding_mask不为None,由于存在新的零批次的添加,则在dim=1中为key_padding_mask添加一列零向量
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # 采用批量矩阵乘法,完成(QK^T)/dk^(-0.5)的计算
        attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)  # 此函数直接返回attn_weights

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]  # 确认attn_weights的维度无误

        if attn_mask is not None:  # 训练/验证时,仅当decoder self-attn不为None;测试时(无论sef-attn或enc-dec attn)为None,无需用于解码器self-attn的time-shifted的未来掩码矩阵
            attn_mask = attn_mask.unsqueeze(0)  # 将为所有批次广播2D掩码,将attn_mask转为3D
            if self.onnx_trace:   # 若onnx_trace为True,则对attn_mask的第0个维度进行扩维
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask  # 与torch底层multi-headed self-attn中attn_mask为float类型的操作一致,则直接加到attn_weights中

        if key_padding_mask is not None:  # 若key_padding_mask不为空;测试时,self-attn由于是顺序解码不对目标进行填充掩码为None;enc-dec attn为float32类型的源端填充掩码矩阵
            # don't attend to padding symbols # 用值掩码矩阵(适用源/目标)以屏蔽被填充的键,先扩维,然后将被指定为1的填充元素的位置上的attn_weights改为"-inf" 　
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:  # 若设置before_softmax,在attention进行softmax之前返回原始注意力权重和values;非默认设置
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )  # 对attn_weights在最后一个dim(src_len)上进行softmax,返回softmax((QK^T)/dk^(-0.5))为float32类型,形状不变
        attn_weights = attn_weights_float.type_as(attn_weights)  # 将attn_weights_float转换回未softmax的attn_weights的数据类型
        # 这里选择创建attn_weights_float,是因为下面attn_weights要设置为None,通过need_weights是否为Ture,决定返回在所有注意力头上的平均注意力权重或None;
        # attn_weights在这里仅仅为一个中间值;
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )  # 对attn_weights的副本进行dropout,training决定是否使用dropout
        assert v is not None  # 从这里开始,V不能继续为None;
        attn = torch.bmm(attn_probs, v)  # 通过批次矩阵乘法来完成softmax((QK^T)/dk^(-0.5))V
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]  # 确认attn的形状无误
        if self.onnx_trace and attn.size(1) == 1: # 当ONNX跟踪单个解码步骤(tgt_len=1)时,transpose是view之前的无操作拷贝,因此不必要执行
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)  # 转换attn的形状,实则完成Concat(head1, ..., headh)操作
        attn = self.out_proj(attn)  # 完成Concat(head1, ..., headh)WQ的操作
        attn_weights: Optional[Tensor] = None  # 将attn_weights重新定义为一个可选Tensor,默认为None
        if need_weights:   # 是否返回在所有注意力头上的平均注意力权重
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)         # 若need_head_weights为true,则返回每个注意力头的注意力权重,形状为(self.num_heads, bsz, tgt_len, src_len)
            if not need_head_weights: # 若need_head_weights为false,则返回在所有注意力头上的平均注意力权重
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights  # 返回multi-headed attn的输出Concat(head1, ..., headh)WQ,以及None
        # 最后一层存在need_weights和need_head_weights时,则将softmax((QK^T)/dk^(-0.5)))展开为多个注意力头,或进一步在全部注意力头上取平均

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv: # 测试时enc-dec attn下static_kv为True,但prev_key_padding_mask也为None:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None: # 测试时enc-dec attn下key_padding_mask为源端填充掩码矩阵
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            ) # 创建一个[bsz,src_len-src_len]的零矩阵,即tensor([], device='cuda:0', size=(500, 0))
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            ) # new_key_padding_mask与key_padding_mask源端填充掩码矩阵一致,只是为float32类型
        else:  # 测试时self-attn下,由于是顺序解码不对目标进行填充掩码key_padding_mask为None
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation). 重新排序缓冲的内部状态(用于增量生成)"""
        input_buffer = self._get_input_buffer(incremental_state) # 从incremental_state中返回当前时间步该层的self-attn/enc-dec attn的prev_key,prev_value,prev_key_padding_mask
        if input_buffer is not None: # input_buffer不为None
            for k in input_buffer.keys(): # 遍历当前时间步该层的self-attn/enc-dec attn的prev_key,prev_value,prev_key_padding_mask
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None: # 防止遍历到self-attn中的prev_key_padding_mask,自回归解码不需要目标端填充矩阵
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(0): # 若当前为enc-dec-attn
                        break # 则不对其的prev_key,prev_value,prev_key_padding_mask重新排序,永远保持与前一时间步该层的enc-dec-attn的k,v一致,其中第一层为encoder_out映射而来的
                    input_buffer[k] = input_buffer_k.index_select(0, new_order) # 对当前时间步该层的self-attn中的prev_key,prev_value重新排序
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
            # 将改动后的input_buffer,仅改动了<对当前时间步该层的self-attn中的prev_key,prev_value重新排序>存入到incremental_state中
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        # 训练/验证时incremental_state为None故值返回{};
        # 测试时,第一个解码时间步self-attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
        #  incremental_state为{}故值返回{};
        # 测试时,第一个解码时间步enc-dec attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
        #  incremental_state存储了{第一层self-attn的唯一码+'attn_state'：prev_key/prev_value/prev_key_padding_mask};
        #  由于第一层enc-dec attn的唯一码+'attn_state'不在incremental_state内,故值返回{}; 以此类推,第一个时间步内6层self-attn和enc-dec attn中值返回{};

        # 后续每个解码时间步,incremental_state键名不会增加,仍是第一个解码时间步6个self-attn和6个enc-dec attn对应的唯一码+'attn_state',
        # 但self-attn对应的值发生改变:到目前解码时间步为止(包括当前)的所存储的prev_key/prev_value/prev_key_padding_mask的级联;enc-dec attn对应的值与第一个解码时间步一样;
        # 故result返回为对应第几层的self-attn或enc-dec attn的state字典{唯一码+'attn_state'：存储prev_key/prev_value/prev_key_padding_mask}
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        # buffer为传入的saved_state,存储到当目前解码时间步为止(包括当前)的'prev_key'/'prev_value'/'prev_key_padding_mask'字典;

        # 训练/验证时incremental_state为None;
        # 测试时,第一个解码时间步self-attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
        #  incremental_state为{};
        # 测试时,第一个解码时间步enc-dec attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
        #  incremental_state存储了{第一层self-attn的唯一码+'attn_state'：prev_key/prev_value/prev_key_padding_mask};
        # 测试时,第一个解码时间步第二层self-attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
        #  incremental_state存储了{第一层self-attn和enc-dec attn的唯一码+'attn_state'：prev_key/prev_value/prev_key_padding_mask}; 以此类推;

        # 后续每个解码时间步,incremental_state键名不会增加,仍是第一个解码时间步6个self-attn和6个enc-dec attn对应的唯一码+'attn_state',
        # 但self-attn对应的值发生改变:到目前解码时间步为止(包括当前)的所存储的prev_key/prev_value/prev_key_padding_mask的级联;enc-dec attn对应的值与第一个解码时间步一样;
        return self.set_incremental_state(incremental_state, "attn_state", buffer) # 将唯一码+'attn_state'：buffer的键值对存入incremental_state字典内

    def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
