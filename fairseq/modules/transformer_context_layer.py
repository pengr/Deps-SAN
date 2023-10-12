# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention, MultiheadContextAttention  # <<改写>>,Context模型的模块化
from torch import Tensor


class TransformerContextEncoderLayer(nn.Module):  # <<改写>>,Context模型的模块化
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    # 在原始论文中,每个操作(多头注意或FFN)后都经过以下处理：`dropout->添加残差-> layernorm`;
    # 在tensor2tensor代码中,建议使用layernorm预处理每个层,(多头注意或FFN)后,再使用`dropout->增加残差'进行后处理,学习更加健壮;
    # 默认使用原始论文方法,但可以通过将*args.encoder_normalize_before*设置为``True''来启用tensor2tensor方法;
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, num_context_heads, num_layers):  # <<改写>>,context模型的模块化
        super().__init__()
        self.embed_dim = args.encoder_embed_dim   # 编码器词嵌入维数
        self.num_context_heads = num_context_heads  # <<改写>>,context模型的模块化
        self.num_layers = num_layers  # <<改写>>,context模型的模块化
        self.self_attn = self.build_self_attention(self.embed_dim, args, self.num_context_heads, self.num_layers)  # <<改写>>,context模型的模块化 # 返回所定义好的MultiheadAttention类(其中由enable_torch_version来决定是否使用torch底层的multi_headed attn实现)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)  # 返回所定义好的torch.nn.LayerNorm归一化类
        self.dropout = args.dropout  # 获取dropout率,与用于self_attn的attention_dropout不同,其用于编码器层内的dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")  # 获取args的activation_fn参数,没有则设置为"relu"
        )  # 返回与`activation`相对应的激活函数,如F.relu
        self.activation_dropout = getattr(args, "activation_dropout", 0)  # 取出对激活函数的dropout率,无则默认为0
        if self.activation_dropout == 0:  # 默认为0.,与使用args.relu_dropout的模型向后兼容,即activation_dropout=relu_dropout
            # for backwards compatibility with models that use args.relu_dropout　
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.encoder_normalize_before  # 使用layernorm预处理/后处理每个编码器模块,(预处理则为tensor2tensor代码,学习更为强健;后处理则默认vaswanai论文)
        self.fc1 = self.build_fc1(self.embed_dim, args.encoder_ffn_embed_dim)  # 由编码器词嵌入维数和编码器前馈网络维度定义encoder中Position-wise FFN网络的两个矩阵,带偏置向量
        self.fc2 = self.build_fc2(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)  # 返回所定义好的torch.nn.LayerNorm归一化类,用于编码器的最后进行层归一化

    def build_fc1(self, input_dim, output_dim):  # 创建一个nn.Linear(input_dim, output_dim)的前馈网络
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):  # 与build_fc1一致,创建一个nn.Linear(input_dim, output_dim)的前馈网络
        return nn.Linear(input_dim, output_dim)

    def build_self_attention(self, embed_dim, args, num_context_heads, num_layers):  # <<改写>>,context模型的模块化  # 根据编码器词嵌入维数,注意力头数,注意力dropout率,以及是否为self-attn的标记返回MultiheadAttention类
        return MultiheadContextAttention(  # <<改写>>,Context模型的模块化
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            num_context_heads=num_context_heads,
            num_layers=num_layers,
            context_type=args.context_type,
        )  # 返回所定义好的MultiheadAttention类(其中由enable_torch_version来决定是否使用torch底层的multi_headed attn实现)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, context, attn_mask: Optional[Tensor] = None):  # <<改写>>,加入编码器的上下文向量C
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)` # 编码器层输入,即源端词嵌入向量为dropout(Ex+PE(x)),形状(src_len, batch, embed_dim)
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``. # 源端填充掩码矩阵,其中填充元素被指定为1,形状(batch, src_len)
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention
            # 类似future_mask, 二进制张量形状为(T_tgt, T_src), 其中T_tgt,T_src分别为查询,键的长度,尽管这里查询和键都是x;
            # attn_mask[t_tgt, t_src] = 1表示在计算t_tgt的嵌入时,则t_src被屏蔽;= 0表示已将其包括在内

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)` # 返回编码器输出,即最后一个编码器层的输出,形状为(src_len,batch,embed_dim)
        """
        residual = x  # 定义用作残余连接的编码器层输入x
        if self.normalize_before:  # 使用layernorm预处理/后处理每个编码器模块,(预处理则为tensor2tensor代码,学习更为强健;后处理则默认vaswanai论文)
            x = self.self_attn_layer_norm(x)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        # anything in original attn_mask = 1, becomes -1e8 任何在原始attn_mask = 1的元素, 变为-1e8
        # anything in original attn_mask = 0, becomes 0   任何在原始attn_mask = 0的元素, 仍然为0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # TODO: to formally solve this problem, we need to change fairseq's
        # MultiheadAttention. We will do this later on.
        # 请注意此处无法使用-inf,因为在某些情况下,查询中某些填充元素的注意力权重(在softmax之前)将变为-inf,这将导致模型参数中出现NaN
        # 要正式解决此问题,我们需要更改fairseq的MultiheadAttention,以后会做
        x, _ = self.self_attn( # 由编码器层输入x作为query,key,value,再将源端填充掩码矩阵,以及attn_mask(在计算t_tgt的嵌入时用于掩码t_src)
            query=x,
            key=x,
            value=x,
            context=context,  # <<改写>>,加入编码器的上下文向量C
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )   # 返回multi-headed attn的输出attn_output->Concat(head1, ..., headh)WQ, attn_output_weights->softmax((QK^T)/dk^(-0.5))
            # 若need_weights=True,后者为在所有注意力头上的平均注意力权重(即attn_output_weights.view(bsz, num_heads, tgt_len, src_len)按dim=1取平均)
        x = F.dropout(x, p=self.dropout, training=self.training)  # 对multi-headed attn的输出attn_output进行dropout
        x = residual + x    # 对dropout(attn_output)进行残余联接,即加上最开始的编码器层输入x
        if not self.normalize_before:  # 使用layernorm后处理每个编码器模块,默认vaswanai论文
            x = self.self_attn_layer_norm(x)  # 返回layer_norm(dropout(attn_output)+x)

        residual = x   # 将layer_norm(dropout(attn_output)+x)-> x'作为position-wise FFN的残余连接
        if self.normalize_before:  # 使用layernorm预处理/后处理每个编码器模块,(预处理则为tensor2tensor代码,学习更为强健;后处理则默认vaswanai论文)
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))  # 完成vaswanai论文postion-wise FFN中,max(0, x'W1+b1),其中activation_fn为RELU
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)  # 若对激活函数的dropout率不为0,则对max(0, x'W1+b1)进行dropout
        x = self.fc2(x)      # 完成vaswanai论文postion-wise FFN中,max(0, x'W1+b1)W2+b2
        x = F.dropout(x, p=self.dropout, training=self.training) # 对position-wise FFN的输出max(0, x'W1+b1)W2+b2进行dropout
        x = residual + x         # 对max(0, x'W1 + b1)W2 + b2进行dropout后进行残余联接,即加上最开始的x'
        if not self.normalize_before:     # 使用layernorm后处理每个编码器模块,默认vaswanai论文
            x = self.final_layer_norm(x)  # 返回layer_norm(dropout(max(0, x'W1+b1)W2+b2) + x')
        return x  # 返回当前编码器层的输出layer_norm(dropout(max(0, x'W1+b1)W2+b2) + x')


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
