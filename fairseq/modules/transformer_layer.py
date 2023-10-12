# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from torch import Tensor


class TransformerEncoderLayer(nn.Module):
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

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim   # 编码器词嵌入维数
        self.self_attn = self.build_self_attention(self.embed_dim, args)  # 返回所定义好的MultiheadAttention类(其中由enable_torch_version来决定是否使用torch底层的multi_headed attn实现)
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

    def build_self_attention(self, embed_dim, args):  # 根据编码器词嵌入维数,注意力头数,注意力dropout率,以及是否为self-attn的标记返回MultiheadAttention类
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
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

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
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


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    # 在原始论文中,每个操作(多头注意或FFN）都经过以下处理：`dropout->添加残差-> layernorm`;
    # 在tensor2tensor代码中,建议使用layernorm预处理每个层,再使用`dropout->增加残差'进行后处理,学习更加健壮;
    # 默认使用原始论文方法,但可以通过将*args.decoder_normalize_before*设置为``True''来启用tensor2tensor方法;

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs  # 是否参加编码器输出
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim   # 解码器词嵌入维数
        self.cross_self_attention = getattr(args, "cross_self_attention", False)  # 是否执行cross+self-attention(Peitz et al., 2019论文参数)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )  # 返回所定义好的MultiheadAttention类(其中由enable_torch_version来决定是否使用torch底层的multi_headed attn实现)
        self.dropout = args.dropout   # 获取dropout率,与用于self_attn的attention_dropout不同,其用于解码器层内的dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu") # 获取args的activation_fn参数,没有则设置为"relu"
        )  # 返回与`activation`相对应的激活函数,如F.relu
        self.activation_dropout = getattr(args, "activation_dropout", 0) # 取出对激活函数的dropout率,无则默认为0
        if self.activation_dropout == 0:  # 默认为0.,与使用args.relu_dropout的模型向后兼容,即activation_dropout=relu_dropout
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.decoder_normalize_before  # 是否使用layernorm预处理每个解码器层,(设置即为tensor2tensor代码,学习更为强健),默认使用vaswanai论文

        # use layerNorm rather than FusedLayerNorm for exporting. # 使用layerNorm而不是FusedLayerNorm进行导出
        # char_inputs can be used to determint this.  # char_inputs可用于确定这一点
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)  # args中无"char_inputs"参数,则返回False
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export) # 返回所定义好的torch.nn.LayerNorm归一化类,用于解码器的self-attn进行层归一化

        if no_encoder_attn:  # 若不加入编码器输出,则不要enc-dec-side self-attn
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:  # 若加入编码器输出,则加入enc-dec-side self-attn(默认设置),与self-attn初始化一样;
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)  # 返回所定义好的MultiheadAttention类(其中由enable_torch_version来决定是否使用torch底层的multi_headed attn实现)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export) # 返回所定义好的torch.nn.LayerNorm归一化类,用于解码器的enc-dec-attn进行层归一化

        self.fc1 = self.build_fc1(self.embed_dim, args.decoder_ffn_embed_dim)  # 由解码器词嵌入维数和解码器前馈网络维度定义decoder中Position-wise FFN网络的两个矩阵,带偏置向量
        self.fc2 = self.build_fc2(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)  # 返回所定义好的torch.nn.LayerNorm归一化类,用于解码器的最后进行层归一化
        self.need_attn = True  # 返回在全部注意头上的平均注意力权重

        self.onnx_trace = False  # 返回onnx_trace标记为False

    def build_fc1(self, input_dim, output_dim):  # 创建一个nn.Linear(input_dim, output_dim)的前馈网络
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):  # 与build_fc1一致,创建一个nn.Linear(input_dim, output_dim)的前馈网络
        return nn.Linear(input_dim, output_dim)

    # 根据解码器词嵌入维数,注意力头数,注意力dropout率,是否手动为k,v添加偏置,以及是否添加零attn,以及是否为self-attn的标记返回MultiheadAttention类
    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),  # 若没有"cross_self_attention"则定义是否为self-attn的标记为True
        )  # 返回所定义好的MultiheadAttention类(其中由enable_torch_version来决定是否使用torch底层的multi_headed attn实现)

    # 根据解码器词嵌入维数,注意力头数,注意力dropout率,以及是否为enc-dec-side attn的标记返回MultiheadAttention类
    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),  # 提前由args中"encoder_embed_dim"给键,值的维度赋值,若没有则返回None
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        ) # 返回所定义好的MultiheadAttention类(其中由enable_torch_version来决定是否使用torch底层的multi_headed attn实现),enc-dec attn和self-attn初始化一样

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)` # 解码器层输入,即目标端词嵌入向量为dropout(Ey+PE(y)),形状(tgt_len, batch, embed_dim)
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.  # 源端填充掩码矩阵,其中填充元素被指定为1,形状(batch, src_len)
            need_attn (bool, optional): return attention weights　# 返回注意力权重,即此层中在全部注意里头上的平均注意力权重,默认最后一个解码器层(idx==5);
            need_head_weights (bool, optional): return attention weights  # 返回每个注意力头的注意力权重
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`　# 返回解码器输出,即最后一个解码器层的输出,形状为(tgt_len,batch,embed_dim)
        """
        if need_head_weights:  # 若需返回每个注意力头的注意力权重,默认在最后一个解码器层(idx==5)生效
            need_attn = True   # 则首先需返回在所有注意力头上的平均注意力权重

        residual = x  # 定义用作残余连接的解码器层输入x,即目标端词嵌入向量为dropout(Ey+PE(y))
        if self.normalize_before:  # 使用layernorm预处理/后处理每个编码器模块,(预处理则为tensor2tensor代码,学习更为强健;后处理则默认vaswanai论文)
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None: # 非默认情况,若prev_self_attn_state不为None:
            prev_key, prev_value = prev_self_attn_state[:2] # 将其先前时间步的prev_key,prev_value,prev_key_padding_mask缓存进saved_state
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        # 训练/验证时incremental_state为None故值返回{};
        # 测试时,第一个解码时间步self-attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
        #  incremental_state为{}故值返回{};
        # 测试时,第一个解码时间步enc-dec attn中,在saved_state存储当目前解码时间步为止(包括当前)的q,k,v并_set_input_buffer之前,
        #  incremental_state存储了{第一层self-attn的唯一码+'attn_state'：prev_key/prev_value/prev_key_padding_mask};
        #  由于第一层enc-dec attn的唯一码+'attn_state'不在incremental_state内,故值返回{}; 以此类推,第一个时间步内6层self-attn和enc-dec attn中值返回{};

        # 后续每个解码时间步,incremental_state键名不会增加,仍是第一个解码时间步6个self-attn和6个enc-dec attn对应的唯一码+'attn_state',
        # 但self-attn对应的值发生改变:到目前解码时间步为止(包括当前)的所存储的prev_key/prev_value/prev_key_padding_mask的级联;enc-dec attn对应的值与第一个解码时间步一样;
        # 故result返回为对应第几层的self-attn或enc-dec attn的state字典{唯一码+'attn_state'：存储prev_key/prev_value/prev_key_padding_mask}

        if self.cross_self_attention and not (  # 非默认情况,若执行cross+self-attention,且incremental_state,_self_attn_input_buffer均为空:
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None: # 若用于解码器self-attn的time-shifted的未来掩码矩阵不为空;测试时为None
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                ) # 在dim=1中给未来掩码矩阵self_attn_mask添加一个tgt_len,src_len的零矩阵
            if self_attn_padding_mask is not None:  # 若目标端填充掩码矩阵,形状为(batch, tgt_len)不为空;测试时为None
                if encoder_padding_mask is None:    # 若源端填充掩码矩阵,形状(batch, src_len)为空:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )  # 创建一个(batch, src_len)的源端填充掩码矩阵,类型与按目标端填充掩码矩阵一致
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )  # 目标端填充掩码矩阵更新为[源端填充掩码矩阵,目标端填充掩码矩阵]的级联
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)  # y为最后一个编码器层的输出和解码器层输入x在dim=0上的级联
        else:
            y = x  # 默认情况,y即为解码器层输入x

        x, attn = self.self_attn(  # 根据如下参数来运行decoder的self-attn
            query=x,  # 解码器层输入x(y=x)作为query,key,value
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask, # 目标端填充掩码矩阵;测试时为None,由于是顺序解码故不对目标进行填充掩码
            incremental_state=incremental_state,  # 训练/验证时为None;测试时初始化为{},用于存储当前解码时间步中self-attn和enc-dec attn的{唯一码+'attn_state'：存储prev_key/prev_value/prev_key_padding_mask}键值对的字典
            need_weights=False,                  # 不返回在所有注意力头上的平均注意力权重,因为decoder中self-attn是中间模块(由于函数内默认为True,必须设置False)
            attn_mask=self_attn_mask,           # time-shifted的未来掩码矩阵(右上三角矩阵,对角线及以下均为0,其余为-inf);测试时为None,无需用于解码器self-attn的time-shifted的未来掩码矩阵
        ) # 返回multi-headed attn的输出attn_output->Concat(head1, ..., headh)WQ, None;
        # 解码器与编码器的self-attn区别: query,key,value,key_padding_mask均为目标端的;由于need_weights为False,作为中间模块不返回注意力权重;
        # attn_mask为time-shifted的未来掩码矩阵(右上三角矩阵,对角线及以下均为0,其余为-inf),通过attn_output_weights += attn_mask对当前目标词的未来序列进行掩码
        x = F.dropout(x, p=self.dropout, training=self.training)  # 对multi-headed attn的输出attn_output进行dropout
        x = residual + x   # 对dropout(attn_output)进行残余联接,即加上最开始的解码器层输入x
        if not self.normalize_before:  # 使用layernorm后处理每个编码器模块,默认vaswanai论文
            x = self.self_attn_layer_norm(x)  # 返回layer_norm(dropout(attn_output)+x)

        if self.encoder_attn is not None:  # 若enc-dec-side self-attn(默认设置)不为空:
            residual = x   # 将layer_norm(dropout(attn_output)+x)-> x'作为enc-dec-side self-attn的残余连接
            if self.normalize_before:  # 使用layernorm预处理/后处理每个解码器模块,(预处理则为tensor2tensor代码,学习更为强健;后处理则默认vaswanai论文)
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:  # 非默认情况;若prev_attn_state不为None;
                prev_key, prev_value = prev_attn_state[:2]  # 将其先前时间步的prev_key,prev_value,prev_key_padding_mask缓存进saved_state
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,  # layer_norm(dropout(attn_output)+x),其中attn_output为解码器self-att模块的输出,x为原始解码器层输入
                key=encoder_out,    # 最后一层编码器的输出
                value=encoder_out,  # 最后一层编码器的输出
                key_padding_mask=encoder_padding_mask,  # 源端填充元素(<pad>)的位置为源端填充掩码矩阵,形状为(batch, src_len)
                incremental_state=incremental_state,  # 训练/验证时为None;测试时初始化为{},用于存储当前解码时间步中self-attn和enc-dec attn的{唯一码+'attn_state'：存储prev_key/prev_value/prev_key_padding_mask}键值对的字典
                static_kv=True,                  # 用于注意力的静态键和值,给定为True即enc-dec self-attn不由torch底层的multi-headed self-attn实现
                need_weights=need_attn or (not self.training and self.need_attn),
                # need_attn返回此层中在全部注意里头上的平均注意力权重,仅在最后一个解码器层(idx==5)为True;若为训练,则not self.training始终为False,若为验证,则后项始终为True
                need_head_weights=need_head_weights,    # 返回每个注意力头的注意力权重,仅在最后一个解码器层(idx==5)为True;
            )  # 返回multi-headed attn的输出Concat(head1, ..., headh)WQ,以及None
               # 最后一层存在need_weights和need_head_weights时,则将softmax((QK^T)/dk^(-0.5)))展开为多个注意力头,或进一步在全部注意力头上取平均
            # 解码器enc-dec attn的注意点: query初始为解码器层的输入,后续为上一层解码器的输出,key,value为最后一层编码器的输出;key_padding_mask为源端填充掩码矩阵;
            # static_kv为True,不采用torch底层的multi-headed self-attn实现; 仅最后一个解码器层的enc-dec attn need_weights为True,返回在所有注意力头上的平均注意力权重;
            x = F.dropout(x, p=self.dropout, training=self.training)  # 对multi-headed attn的输出attn进行dropout
            x = residual + x     # 对dropout(attn)进行残余联接,即加上enc-dec-side self-attn的残余连接x'
            if not self.normalize_before:  # 使用layernorm后处理每个编码器模块,默认vaswanai论文
                x = self.encoder_attn_layer_norm(x)  # 返回layer_norm(dropout(attn) + x')

        residual = x    # 将layer_norm(dropout(attn) + x')-> x''作为position-wise FFN的残余连接
        if self.normalize_before:  # 使用layernorm预处理/后处理每个编码器模块,(预处理则为tensor2tensor代码,学习更为强健;后处理则默认vaswanai论文)
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))  # 完成vaswanai论文postion-wise FFN中,max(0, x''W1+b1),其中activation_fn为RELU
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training) # 若对激活函数的dropout率不为0,则对max(0, x''W1+b1)进行dropout
        x = self.fc2(x)  # 完成vaswanai论文postion-wise FFN中,max(0, x''W1+b1)W2+b2
        x = F.dropout(x, p=self.dropout, training=self.training)  # 对position-wise FFN的输出max(0, x''W1+b1)W2+b2进行dropout
        x = residual + x           # 对max(0, x''W1 + b1)W2 + b2进行dropout后进行残余联接,即加上position-wise FFN的残余连接x''
        if not self.normalize_before: # 使用layernorm后处理每个编码器模块,默认vaswanai论文
            x = self.final_layer_norm(x)  # 返回layer_norm(dropout(max(0, x''W1+b1)W2+b2) + x'')
        if self.onnx_trace and incremental_state is not None:  # 非默认情况,若onnx_trace标记为True,且incremental_state不为None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None  # 返回当前解码器层输出layer_norm(dropout(max(0, x''W1+b1)W2+b2) + x''),None(最后一个解码器层为enc-dec self-attn的输出attn_weights->softmax((QK^T)/dk^(-0.5))),None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Scriptable reorder incremental state in transformer layers. Transformer layer中可编写脚本的重新排序增量状态"""
        # 返回根据*new_order*重新排序过的self-attn和enc-dec attn中的incremental_state,主要对bsz所在维度进行重复扩展beam size次
        self.self_attn.reorder_incremental_state(incremental_state, new_order)

        if self.encoder_attn is not None:
            self.encoder_attn.reorder_incremental_state(incremental_state, new_order)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
