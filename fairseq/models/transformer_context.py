# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,   # context模型的编码器的父类
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerDecoder  # context模型的解码器
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,   # context模型的san编码层
    TransformerContextEncoderLayer,   # <<改写>>,context模型的sdsa编码层
)
from torch import Tensor


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("transformer_context")  # <<改写>>,context模型的模块化
class TransformerContextModel(FairseqEncoderDecoderModel):  # <<改写>>,context模型的模块化
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)  # 调用父类FairseqEncoderDecoderModel(),写入与原始论文一致的Transformer Encoder和Decoder
        self.args = args  # 将args Self化用于transformer的调用,如forward()
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')  # 所使用的激活函数
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')         # dropout率
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')  # 注意力权重所用的dropout率
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')  # 在position-wise全连接前馈网络中经过激活函数后的dropout率
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')  # 预训练的编码器嵌入的路径
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')  # 编码器嵌入维数
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')  # 用于position-wise全连接前馈网络的编码器嵌入维数
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')  # 编码器的层数
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')  # 编码器注意力头数
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')  # 设置在每个编码块(层)前采用层归一化
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')  # 在编码器中使用学习好的positional embeddings
        # <<改写>>,设置TransformerPascalEncoderLayer的层数和注意力头数,列表类型(默认:None)
        parser.add_argument('--enc-context-heads', type=int, nargs='+',
                            help='list of context aware self-attention heads per layer')
        # <<改写>>,设置context-aware self-attn的三种窗口类型
        parser.add_argument('--context-type',type=str, metavar='STR',
                            choices=["global", "deep", "deep_global"],
                            help="Which context types to use for context-aware self-attn")
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')  # 预训练的解码器嵌入的路径
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')  # 解码器嵌入维数
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')  # 用于position-wise全连接前馈网络的解码器嵌入维数
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')  # 解码器的层数
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')  # 解码器注意力头数
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')  # 在解码器中使用学习好的positional embeddings
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')  # 设置在每个解码块(层)前采用层归一化
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')  # 共享解码器的输入和输出词嵌入
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')  # 共享编码器.解码器的输入和输出三个词嵌入,需要共享字典以及相同的embed_dim
        # 下面三个参数用于禁止positional embeedings,从源和目标端,源端,目标端三类出发
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--no-src-positional-embeddings', default=False, action='store_true',
                            help='if set, disables source positional embeddings (outside self attention)')
        parser.add_argument('--no-tgt-positional-embeddings', default=False, action='store_true',
                            help='if set, disables target positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),  # 以逗号分隔的自适应softmax截止点列表,必须与adaptive_loss指标一起使用
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')  # 为tail projections设置自适应softmax丢弃率
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')  # 用于编码器的layerdrop率
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')  # 用于解码器的layerdrop率
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')  # 在修剪时保存哪些编码层
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')  # 在修剪时保存哪些解码层
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')      # 添加层归一化到嵌入中
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')  # 若为True,不缩放embeddings
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models 确保所有参数都存在于较旧的模型中(若模型为'transformer_wmt_en_de',则已设置过一轮完全一样的)
        base_architecture(args)

        if args.encoder_layers_to_keep:  # 若修剪时所保存哪些encoder/decoder层不变为None(Fan et al., 2019论文设定):
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:  # 若源和目标句子的最大token数为None,则手动设置为默认值
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary  # 从当前任务<fTranslationTask>中取出源和目标字典类(Dictionary)

        if args.share_all_embeddings:  # 若共享源,目标输入,目标输出3个词嵌入:
            if src_dict != tgt_dict:  # 需要源和目标字典一致(共用字典)
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:  # 需要编码器和解码器的嵌入维数相同
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):  # 共享全部词嵌入必须无给定好的目标词嵌入,即使有也不能和给定的源端词嵌入不同
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(  # 通过args,以及源字典和编码器词嵌入大小创建源端词嵌入,默认给定编码器嵌入路径为None
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )  # 返回源端嵌入矩阵(填充padding_idx,嵌入值为0;正态分布初始化x~N(0,embedd_dim))
            decoder_embed_tokens = encoder_embed_tokens   # 由于共用全部词嵌入,故目标端词嵌入与源端词嵌入一致
            args.share_decoder_input_output_embed = True  # 由于共用全部词嵌入,故设置目标端输入和输出嵌入一致的标记为True
        else:  # 若不共享源,目标端的词嵌入,则分别由通过args,以及源字典和编码器词嵌入大小创建源端词嵌入,默认给定编码器嵌入路径为None
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )  # 返回源端嵌入矩阵(填充padding_idx,嵌入值为0;正态分布初始化x~N(0,embedd_dim))
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )  # 返回目标端嵌入矩阵(填充padding_idx,嵌入值为0;正态分布初始化x~N(0,embedd_dim))

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)  # 通过args,源字典以及源端词嵌入返回一个与原始论文一致的Transformer Encoder,每一层TransformerEncoderLayer(Multi-headed attn->dropout->添加残差-> layernorm)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)  # 通过args,目标字典以及目标端词嵌入返回一个与原始论文一致的Transformer Decoder,每一层TransformerDecoderLayer(Multi-headed attn->dropout->添加残差-> layernorm)
        return cls(args, encoder, decoder)  # 调用__init__(),写入与原始论文一致的Transformer Encoder和Decoder

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)  # 获得词嵌入的第一个维数,即词汇表大小
        padding_idx = dictionary.pad()    # 定义填充符号(pad token)的idx,默认为1

        # 根据词汇表大小,编码器词嵌入大小,以及填充符号的idx创建词嵌入矩阵;
        emb = Embedding(num_embeddings, embed_dim, padding_idx)  # 返回嵌入矩阵(填充padding_idx,嵌入值为0;正态分布初始化x~N(0,embedd_dim))
        # if provided, load from preloaded dictionaries
        if path:  # 若提供了给定的嵌入路径,则下载预先加载好的字典,默认为None
            embed_dict = utils.parse_embedding(path)  # 将给定的嵌入文本文件解析为单词字典和嵌入张量
            utils.load_embedding(embed_dict, dictionary, emb)  # 将所得的嵌入字典和现有的字典及其此前入矩阵,一起加载(具体细节未看)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):  # 通过args,源字典以及源端词嵌入建立编码器,初始化TransformerEncoder类
        return TransformerContextEncoder(args, src_dict, embed_tokens)  # <<改写>>,Context模型的模块化 # 返回一个与原始论文一致的Transformer Encoder,每一层TransformerEncoderLayer(Multi-headed attn->dropout->添加残差-> layernorm)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens): # 通过args,目标字典以及目标端词嵌入建立解码器,初始化TransformerDecoder类
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),  # args.no_cross_attention为是否参加编码器输出的标记(Peitz et al., 2019论文参数),默认为False
        )  # 返回一个与原始论文一致的Transformer Decoder,每一层TransformerDecoderLayer(Multi-headed attn->dropout->添加残差-> layernorm)

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model. 对编码器-解码器模型运行前向传递

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript. 从基类复制,但没有TorchScript不支持的** kwargs
        """
        encoder_out = self.encoder(
            src_tokens,               # 当前遍历的batch中全部源句子的tensor所转换的二维填充向量
            src_lengths=src_lengths,  # 当前遍历的batch中全部源句子的长度的Tensor
            cls_input=cls_input,
            return_all_hiddens=return_all_hiddens,  # 是否返回全部中间隐藏态(即每一层编码器的输出)
        )   # 返回NamedTuple类型,即
            # EncoderOut(
            #     encoder_out=x,  # T x B x C  # 返回最后一层编码器的输出layer_norm(dropout(max(0, x'W1+b1)W2+b2) + x')
            #     encoder_padding_mask=encoder_padding_mask,  # B x T # 计算源端填充元素(<pad>)的位置为源端填充掩码矩阵,形状为(batch, src_len)
            #     encoder_embedding=encoder_embedding,  # B x T x C   # 未进行PE操作的原始源端词嵌入向量Ex
            #     encoder_states=encoder_states,  # List[T x B x C]  # 返回存有每一层编码器的输出的列表
            # )
        decoder_out = self.decoder(
            prev_output_tokens,          # 目标句子tensor的二维填充向量的移位版本(即全部目标句子的eos移到开头)
            encoder_out=encoder_out,     # 最后一层编码器的输出
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,      # 当前遍历的batch中全部源句子的长度的Tensor
            return_all_hiddens=return_all_hiddens,  # 是否返回全部中间隐藏态(即每一层解码器的输出)
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Dict[str, List[Optional[Tensor]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output. 从模型的输出中获取归一化的概率(或对数概率)"""
        # net_output->模型的输出(经过"Linear层"的解码器输出,最后一个解码器层enc-dec self-attn的输出attn_weights在全部注意力头上平均注意力权重,存储每一层解码器的输出列表)
        # log_probs为True,sample为None
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        # 若选择返回log概率,则对经过"Linear层"的解码器输出logits进行log_softmax(经过vaswani论文的"Softmax"),否则对其进行softmax


class TransformerContextEncoder(FairseqEncoder):  # <<改写>>,Context模型的模块化
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    # 由*args.encoder_layers*层组成的Transformer encoder,每一层为一个TransformerEncoderLayer类
    Args:
        args (argparse.Namespace): parsed command-line arguments    # 全部命令行参数
        dictionary (~fairseq.data.Dictionary): encoding dictionary  # 源端字典
        embed_tokens (torch.nn.Embedding): input embedding          # 输入嵌入
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)   # 由父类FairseqEncoder初始化源端字典
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout = args.dropout  # 获取dropout率
        self.encoder_layerdrop = args.encoder_layerdrop  # 获取编码器的layer_dropout率(Fan et al., 2019论文参数)

        # <<改写>>,确认context的注意力头和层数
        if args.enc_context_heads is None:
            self.enc_context_heads = [0] * args.encoder_layers
        else:
            self.enc_context_heads = args.enc_context_heads
        assert len(self.enc_context_heads) == args.encoder_layers
        # <<改写>>,self化窗口类型context_type
        self.context_type = args.context_type

        # <<改写>>,计算高斯分布时torch.exp()只支持sdsa_matrix为float类型,而—fp16参数将float32->float16;
        if args.fp16:
            self.matrix_dtype = torch.half
        else:
            self.matrix_dtype = torch.float

        embed_dim = embed_tokens.embedding_dim  # 获取源端字典的第二个维数(源端词嵌入维度)作为encoder的dim
        self.padding_idx = embed_tokens.padding_idx  # 获取源端字典的填充符号idx(1)
        self.max_source_positions = args.max_source_positions  # 获取源句子最大tokens数(用于positional embeddings)

        self.embed_tokens = embed_tokens  # 获取源端词嵌入
        # 若不缩放embeddings(Fan et al., 2019论文参数),embed_scale为1,否则为dmodel^-0.5(默认)
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )  # 通过源句子最大tokens数,源端词嵌入维数,padding_idx(1),以及在编码器中使用学习好的positional embeddings来得到源端的位置嵌入PositionalEmbedding
            if not args.no_src_positional_embeddings  # 若no_src_positional_embeddings=True则禁止使用源位置嵌入,默认为False
            else None
        )  # 返回由max_source/target_position决定长度的正弦型位置嵌入,填充符号(pad)的嵌入向量被置0

        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)  # 执行逐层的slef-attn(Peitz et al., 2019论文参数)

        self.layers = nn.ModuleList([])  # 用于存放全部的transformer Encoder Layer
        self.layers.extend(
            [self.build_context_encoder_layer(args, h, i) if h != 0 \
                 else self.build_encoder_layer(args) for i, h in enumerate(self.enc_context_heads)]
        )   # <<改写>>,context模型的模块化 # 根据args中编码器层数,由args构建每一层编码器层(与原始论文一致,Multi-headed attn->dropout->添加残差-> layernorm)
        self.num_layers = len(self.layers)  # 记录encoder的层数

        if args.encoder_normalize_before:  # 使用layernorm预处理/后处理每个编码器模块,(预处理则为tensor2tensor代码,学习更为强健;后处理则默认vaswanai论文)
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):  # 添加layernorm到嵌入(Fan et al., 2019论文参数)
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def build_encoder_layer(self, args):  # 由args初始化一个TransformerEncoderLayer类
        return TransformerEncoderLayer(args)  # 返回一个与原始论文一致的Transformer编码器层,Multi-headed attn->dropout->添加残差-> layernorm

    def build_context_encoder_layer(self, args, h, i):  # <<改写>>,context模型的模块化# 由args初始化一个TransformerEncoderLayer类
        return TransformerContextEncoderLayer(args, h, i)  # <<改写>>,context模型的模块化  # 返回一个与原始论文一致的Transformer编码器层,Multi-headed attn->dropout->添加残差-> layernorm

    def forward_embedding(self, src_tokens):
        # embed tokens and positions  # embed_scale为dmodel^-0.5, 将源句子tensor的二维填充向量喂入nn.Embedding(),得到原始源端词嵌入向量Ex
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:  # 若正弦型位置编码嵌入不为空,则源端词嵌入向量为Ex+PE(x)
            x = embed + self.embed_positions(src_tokens) # 从正弦曲线嵌入的第0个维度中,源句子tensor的二维填充向量的位置编号中选取PE(x)数据,并扩维+detach(从计算图复制,不进行梯度更新)
        if self.layernorm_embedding is not None: # 若添加layernorm到嵌入(Fan et al., 2019论文参数);非默认参数
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  # 对所得到的Ex+PE(x)按照args.dropout进行丢弃
        return x, embed  # 返回源端词嵌入向量为Ex+PE(x),未进行PE操作的原始源端词嵌入向量Ex

    def forward(
        self,
        src_tokens,
        src_lengths,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)` # 当前遍历的batch中全部源句子的tensor所转换的二维填充向量,形状为(batch, src_len)
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)` # 当前遍历的batch中全部源句子的长度的Tensor,形状为(batch)
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).  # 返回所有中间隐藏状态,传入True值

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)` # 最后一个编码器层的输出,形状为(src_len,batch,embed_dim)
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)` # 填充元素(<pad>)的位置,形状为(batch, src_len)
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup #
                  of shape `(batch, src_len, embed_dim)` # (缩放的)嵌入查找表,形状为(batch, src_len, embed_dim)
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True. # 所有中间隐藏状态,形状为(src_len, batch, embed_dim),仅在*return_all_hiddens*为True时生效
        """
        if self.layer_wise_attention:  # (Fan et al., 2019论文参数)
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens)  # 返回源端词嵌入向量为Ex+PE(x),未进行PE操作的原始源端词嵌入向量Ex

        # B x T x C -> T x B x C # batch,src_len,embed_dim -> src_len,batch,embed_dim
        x = x.transpose(0, 1)

        # compute padding mask # 计算源端填充元素(<pad>)的位置为源端填充掩码矩阵,形状为(batch, src_len)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None  # 当确认要返回所有中间隐藏态时,由encoder_states列表存储

        # <<改写>>,计算高斯分布时torch.exp()只支持sdsa_matrix为float类型,而—fp16参数将float32->float16;
        # matrix = matrix.to(self.matrix_dtype)

        # <<改写>>,由窗口类型context_type来定义multi-layer self-attn,
        context = None
        input_states = []
        # encoder layers
        for i, h in enumerate(self.enc_context_heads):  # <<改写>>, 由enc_context_heads遍历每一个编码器层,并添加sdsa和pascal矩阵;
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # 采用层Dropout: 即设置不为0的encoder_layerdrop值,则均匀分布随机初始化的dropout值需要超过所设layer_dropout值,才会运行encoder layer forward
            dropout_probability = torch.empty(1).uniform_()  # 由均匀分布随机初始化一个dropout值
            if not self.training or (
                dropout_probability > self.encoder_layerdrop):  # 若非训练阶段,或者无layer_dropout(0),即均匀随机初始化的dropout值肯定>0:
                # 通过源端词嵌入向量为Ex+PE(x),以及源端填充掩码矩阵进入TransformerEncoderLayer
                if h != 0:
                    # Global Context: 当前层的输入做平均(第一层为源嵌入,其他为下一层的输出); 1,batch,embed_dim
                    if self.context_type == "global":
                        context = x.mean(0, keepdim=True)
                    # Deep Context: 当前层以下的输入的级联(第一层无,第二层为源嵌入,其他为下面层的级联); src_len,batch,(l-1)*embed_dim
                    elif self.context_type == "deep":
                        input_states.append(x)
                        context = torch.cat(input_states[:-1], dim=2) if i > 0 else None
                    # Deep-Global Context: 当前层及其以下的输入均做平均的级联(第一层为源嵌入做平均,第二层为源嵌入和第一层输出做平均,其他为下面层做平均的级联);
                    elif self.context_type == "deep_global":
                        input_states.append(x.mean(0, keepdim=True))
                        context = torch.cat(input_states, dim=2)
                    x = self.layers[i](x, encoder_padding_mask, context)
                else:
                    x = self.layers[i](x, encoder_padding_mask)  # 返回当前编码器层的输出layer_norm(dropout(max(0, x'W1+b1)W2+b2) + x')
                if return_all_hiddens:  # 当确认要返回所有中间隐藏态时,将每一层编码器的输出都存在encoder_states中;测试时为False
                    assert encoder_states is not None
                    encoder_states.append(x)

        if self.layer_norm is not None:  # 使用layernorm预处理/后处理每个编码器模块,(预处理则为tensor2tensor代码,学习更为强健;后处理则默认vaswanai论文)
            x = self.layer_norm(x)  # 若采用tensor2tensor为预处理,则对最后一层编码器的输出,进行layer_norm以进入下一个模块
            if return_all_hiddens:  # 在存有每一层编码器的输出的encoder_states中,将最后一层编码器输出更新为layer_norm后的
                encoder_states[-1] = x

        return EncoderOut(
            encoder_out=x,  # T x B x C  # 返回最后一层编码器的输出layer_norm(dropout(max(0, x'W1+b1)W2+b2) + x')
            encoder_padding_mask=encoder_padding_mask,  # B x T # 计算源端填充元素(<pad>)的位置为源端填充掩码矩阵,形状为(batch, src_len)
            encoder_embedding=encoder_embedding,  # B x T x C   # 未进行PE操作的原始源端词嵌入向量Ex
            encoder_states=encoder_states,  # List[T x B x C]  # 返回存有每一层编码器的输出的列表,测试时为None
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*. # 根据*new_order*重新排序编码器输出

        Args:
            encoder_out: output from the ``forward()`` method # 当前批次的编码器输出
            new_order (LongTensor): desired order             # 当前批次的句子行号新顺序new_order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out: Dict[str, Tensor] = {}

        new_encoder_out["encoder_out"] = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None  # 若当前批次的编码器输出中encoder_out不为空:(T x B x C)
            else encoder_out.encoder_out.index_select(1, new_order) # 则从encoder_out的第1个维度的指定位置(new_order)中选取数据
            # 由于new_order为[0,0,0,1,1,1],若encoder_out为[[[a],  ->[[[a,a,a]
            #                                              [b]]]    [b,b,b]]]
        )
        new_encoder_out["encoder_padding_mask"] = (
            encoder_out.encoder_padding_mask
            if encoder_out.encoder_padding_mask is None # 若当前批次的编码器输出中encoder_padding_mask不为空:(B x T)
            else encoder_out.encoder_padding_mask.index_select(0, new_order)  # 则从encoder_padding_mask的第0个维度的指定位置(new_order)中选取数据
        )
        new_encoder_out["encoder_embedding"] = (
            encoder_out.encoder_embedding
            if encoder_out.encoder_embedding is None  # 若当前批次的编码器输出中encoder_embedding不为空:(B x T x C )
            else encoder_out.encoder_embedding.index_select(0, new_order) # 则从encoder_embedding的第0个维度的指定位置(new_order)中选取数据
        )

        encoder_states = encoder_out.encoder_states # 取出存有每一层编码器的输出的列表,测试时为None
        if encoder_states is not None:  # 测试时encoder_states为None
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out["encoder_out"],  # T x B*beam_size x C
            encoder_padding_mask=new_encoder_out["encoder_padding_mask"],  # B*beam_size x T
            encoder_embedding=new_encoder_out["encoder_embedding"],  # B*beam_size x T x C
            encoder_states=encoder_states,  # List[T x B*beam_size x C]
        )

    def max_positions(self):
        """Maximum input length supported by the encoder.编码器支持的最大输入长度"""
        if self.embed_positions is None:  # 若不存在由max_source/target_position决定长度的正弦型位置嵌入,则编码器的最大输入长度为max_source_positions
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions) # 默认为max_source_positions,因为正弦型位置嵌入的max_positions(10^5)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(
                    utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
                )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq. 升级（可能是旧的）状态字典以获取新版本的fairseq,新版本的fairseq生成的状态字典不做任何操作"""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    # 创建嵌入矩阵,由于每个句子长度不1,所以需要设置padding_idx(注:不是在m.weight矩阵中会出现padding_idx),是在后续使用
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)  # 对词嵌入矩阵m.weight进行正态分布初始化(x~N(0,embedd_dim开放)
    nn.init.constant_(m.weight[padding_idx], 0)  # 使词嵌入矩阵m.weight中padding_idx位置的值为0
    return m  # 返回嵌入矩阵(填充padding_idx,嵌入值为0;正态分布初始化x~N(0,embedd_dim))


def Linear(in_features, out_features, bias=True):  # 用于建立一个nn.Linear()的矩阵映射,由xavier_uniform_初始化,且若存在bias则对其初始化为0
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("transformer_context", "transformer_context")  # <<改写>>,context模型的模块化
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    # <<改写>>,context模型的模块化
    args.enc_context_heads = getattr(args, "enc_context_heads", None)
    args.context_type = getattr(args, "context_type", "deep_global")
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    ) # 若args中未设置decoder_output_dim,则默认为decoder_embed_dim
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim) # 若args中未设置decoder_input_dim,则默认为decoder_embed_dim

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)


@register_model_architecture("transformer_context", "transformer_context_iwslt_de_en")  # <<改写>>,context模型的模块化
def transformer_context_iwslt_de_en(args):  # <<改写>>,context模型的模块化
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("transformer_context", "transformer_context_wmt_en_de")  # <<改写>>,context模型的模块化
def transformer_context_wmt_en_de(args):  # 通过-arch选项选择合适的transformer参数  # <<改写>>,localness模型的模块化
    base_architecture(args)  # 针对args提供进来的参数,已存在的不变,未设置的则按base_architecture设置


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer_context", "transformer_context_vaswani_wmt_en_de_big")  # <<改写>>,context模型的模块化
def transformer_context_vaswani_wmt_en_de_big(args):  # <<改写>>,pascal模型的模块化
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


@register_model_architecture("transformer_context", "transformer_context_vaswani_wmt_en_fr_big")  # <<改写>>,context模型的模块化
def transformer_context_vaswani_wmt_en_fr_big(args):  # <<改写>>,context模型的模块化
    args.dropout = getattr(args, "dropout", 0.1)
    transformer_context_vaswani_wmt_en_de_big(args)  # <<改写>>,context模型的模块化


@register_model_architecture("transformer_context", "transformer_context_wmt_en_de_big")  # <<改写>>,context模型的模块化
def transformer_context_wmt_en_de_big(args):  # <<改写>>,context模型的模块化
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    transformer_context_vaswani_wmt_en_de_big(args)  # <<改写>>,context模型的模块化


# default parameters used in tensor2tensor implementation
@register_model_architecture("transformer_context", "transformer_context_wmt_en_de_big_t2t")  # <<改写>>,context模型的模块化
def transformer_context_wmt_en_de_big_t2t(args):  # <<改写>>,context模型的模块化
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_context_vaswani_wmt_en_de_big(args)  # <<改写>>,context模型的模块化