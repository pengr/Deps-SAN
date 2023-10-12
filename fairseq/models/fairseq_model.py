# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base classes for various fairseq models.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.checkpoint_utils import prune_state_dict
from fairseq.data import Dictionary
from fairseq.models import FairseqDecoder, FairseqEncoder
from torch import Tensor


logger = logging.getLogger(__name__)


class BaseFairseqModel(nn.Module):
    """Base class for fairseq models."""

    def __init__(self):
        super().__init__()
        self._is_generation_fast = False

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError("Model must implement the build_model method")

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output. 从sample或模型的输出(即log_softmax(logits))中获取目标序列"""
        return sample["target"]  # 返回sample字典中"target"对应的目标序列

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Dict[str, List[Optional[Tensor]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Dict[str, List[Optional[Tensor]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel 〜BaseFairseqModel中的get_normalized_probs可编写脚本的辅助函数"""
        if hasattr(self, "decoder"):  # 若模型中存在decoder模块,由于transformer,FairseqIncrementalDecoder无该函数,模型这里调用的是最底层的fairseq_model.py中decoder
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
            # 若选择返回log概率, 则对经过"Linear层"的解码器输出logits进行log_softmax(经过vaswani论文的"Softmax"),否则对其进行softmax
        elif torch.is_tensor(net_output): # 非默认情况;若net_output为一个tensor,则直接float(),然后根据对float(net_output)进行log_softmax或者softmax(取决于log_probs)
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    def load_state_dict(self, state_dict, strict=True, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants. # 将参数和缓冲区从*state_dict*复制到此模块及其后代

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.覆盖nn.Module中的方法,且此方法还从旧检查点“升级” *state_dicts*
        """
        self.upgrade_state_dict(state_dict) # 升级旧state字典以使用新代码,新代码的state字典不做任何操作
        new_state_dict = prune_state_dict(state_dict, args) # 如果需要LayerDrop,则修剪给定的state_dict;默认不用
        return super().load_state_dict(new_state_dict, strict) # 将state_dict中模型全部训练参数和缓冲区加载self(即当前模型)中

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code.　升级旧state字典以使用新代码,新代码的state字典不做任何操作"""
        self.upgrade_state_dict_named(state_dict, "")

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code. 升级旧state字典以使用新代码,新代码的state字典不做任何操作

        Args:
            state_dict (dict): state dictionary to upgrade, in place
            name (str): the state dict key corresponding to the current module
        """
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += "."

            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, "upgrade_state_dict_named"):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, "upgrade_state_dict"):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)

    def set_num_updates(self, num_updates):
        """ State from trainer to pass along to model at every update """
        # 在每次更新时从trainer中传递当前更新批次步骤给模型中所有模块,若该模块存在set_num_updates属性
        def _apply(m):
            if hasattr(m, 'set_num_updates') and m != self:
                m.set_num_updates(num_updates)
        self.apply(_apply)


    def make_generation_fast_(self, **kwargs):
        """Optimize model for faster generation. 优化模型以加快生成速度"""
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True  # 加快生成速度标记为true

        # remove weight norm from all modules in the network 删除网络中所有模块的权重归一化WeightNorm
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return

        # 遍历self.children()(即当前模型架构的全部模块),再以module.apply()遍历到(最小的组成模块),以fn()删除当前模块中所有模块的权重归一化
        # 注: 这里所删除的权重归一化为WeightNorm类型,整个模型架构都由LayerNorm()构成,故不进行任何操作
        self.apply(apply_remove_weight_norm)

        seen = set()

        def apply_make_generation_fast_(module):
            if (
                module != self
                and hasattr(module, "make_generation_fast_")
                and module not in seen # 若module不为整个模型架构,且包含"make_generation_fast_"属性,以及不在seen集合内:
            ):
                seen.add(module) # 将当前遍历的模块module加到seen集合内
                module.make_generation_fast_(**kwargs) # 将当前遍历的模块module的need_attn设置为True

        self.apply(apply_make_generation_fast_) # 将整个模型架构中的所有模块module的need_attn都设置为True

        def train(mode=True):
            if mode:
                raise RuntimeError("cannot train after make_generation_fast")

        # this model should no longer be used for training # 此模型不应再用于训练
        self.eval()  # 将整个模型设置为eval模式
        self.train = train # 若self.train(True),则报错"在make_generation_fast后不能在训练"

    def prepare_for_onnx_export_(self, **kwargs):
        """Make model exportable via ONNX trace."""
        seen = set()

        def apply_prepare_for_onnx_export_(module):
            if (
                module != self
                and hasattr(module, "prepare_for_onnx_export_")
                and module not in seen
            ):
                seen.add(module)
                module.prepare_for_onnx_export_(**kwargs)

        self.apply(apply_prepare_for_onnx_export_)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        **kwargs,
    ):
        """
        Load a :class:`~fairseq.models.FairseqModel` from a pre-trained model
        file. Downloads and caches the pre-trained model file if needed.

        The base implementation returns a
        :class:`~fairseq.hub_utils.GeneratorHubInterface`, which can be used to
        generate translations or sample from language models. The underlying
        :class:`~fairseq.models.FairseqModel` can be accessed via the
        *generator.models* attribute.

        Other models may override this to implement custom hub interfaces.

        Args:
            model_name_or_path (str): either the name of a pre-trained model to
                load or a path/URL to a pre-trained model state dict
            checkpoint_file (str, optional): colon-separated list of checkpoint
                files in the model archive to ensemble (default: 'model.pt')
            data_name_or_path (str, optional): point args.data to the archive
                at the given path/URL. Can start with '.' or './' to reuse the
                model archive path.
        """
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            **kwargs,
        )
        logger.info(x["args"])
        return hub_utils.GeneratorHubInterface(x["args"], x["task"], x["models"])

    @classmethod
    def hub_models(cls):
        return {}


class FairseqEncoderDecoderModel(BaseFairseqModel):
    """Base class for encoder-decoder models.

    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)  # 调用decoder.forward()

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model.模型支持的最大长度"""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder. 解码器支持的最大长度"""
        return self.decoder.max_positions()


class FairseqModel(FairseqEncoderDecoderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        utils.deprecation_warning(
            "FairseqModel is deprecated, please use FairseqEncoderDecoderModel "
            "or BaseFairseqModel instead",
            stacklevel=4,
        )


class FairseqMultiModel(BaseFairseqModel):
    """Base class for combining multiple encoder-decoder models."""

    def __init__(self, encoders, decoders):
        super().__init__()
        assert encoders.keys() == decoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            assert isinstance(encoders[key], FairseqEncoder)
            assert isinstance(decoders[key], FairseqDecoder)

        self.models = nn.ModuleDict(
            {
                key: FairseqEncoderDecoderModel(encoders[key], decoders[key])
                for key in self.keys
            }
        )

    @staticmethod
    def build_shared_embeddings(
        dicts: Dict[str, Dictionary],
        langs: List[str],
        embed_dim: int,
        build_embedding: callable,
        pretrained_embed_path: Optional[str] = None,
    ):
        """
        Helper function to build shared embeddings for a set of languages after
        checking that all dicts corresponding to those languages are equivalent.

        Args:
            dicts: Dict of lang_id to its corresponding Dictionary
            langs: languages that we want to share embeddings for
            embed_dim: embedding dimension
            build_embedding: callable function to actually build the embedding
            pretrained_embed_path: Optional path to load pretrained embeddings
        """
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError(
                "--share-*-embeddings requires a joined dictionary: "
                "--share-encoder-embeddings requires a joined source "
                "dictionary, --share-decoder-embeddings requires a joined "
                "target dictionary, and --share-all-embeddings requires a "
                "joint source + target dictionary."
            )
        return build_embedding(shared_dict, embed_dim, pretrained_embed_path)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        decoder_outs = {}
        for key in self.keys:
            encoder_out = self.models[key].encoder(src_tokens, src_lengths, **kwargs)
            decoder_outs[key] = self.models[key].decoder(
                prev_output_tokens, encoder_out, **kwargs
            )
        return decoder_outs

    def max_positions(self):
        """Maximum length supported by the model."""
        return {
            key: (
                self.models[key].encoder.max_positions(),
                self.models[key].decoder.max_positions(),
            )
            for key in self.keys
        }

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return min(model.decoder.max_positions() for model in self.models.values())

    @property
    def encoder(self):
        return self.models[self.keys[0]].encoder

    @property
    def decoder(self):
        return self.models[self.keys[0]].decoder

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def load_state_dict(self, state_dict, strict=True, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        new_state_dict = prune_state_dict(state_dict, args)
        return super().load_state_dict(new_state_dict, strict)


class FairseqLanguageModel(BaseFairseqModel):
    """Base class for decoder-only models.

    Args:
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        assert isinstance(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, **kwargs):
        """
        Run the forward pass for a decoder-only model.

        Feeds a batch of tokens through the decoder to predict the next tokens.

        Args:
            src_tokens (LongTensor): tokens on which to condition the decoder,
                of shape `(batch, tgt_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, seq_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        return self.decoder(src_tokens, **kwargs)

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def extract_features(self, src_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, seq_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        return self.decoder.extract_features(src_tokens, **kwargs)

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.decoder.max_positions()

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()

    @property
    def supported_targets(self):
        return {"future"}


class FairseqEncoderModel(BaseFairseqModel):
    """Base class for encoder-only models.

    Args:
        encoder (FairseqEncoder): the encoder
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        assert isinstance(self.encoder, FairseqEncoder)

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        Run the forward pass for a encoder-only model.

        Feeds a batch of tokens through the encoder to generate features.

        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            the encoder's output, typically of shape `(batch, src_len, features)`
        """
        return self.encoder(src_tokens, src_lengths, **kwargs)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output["encoder_out"]
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions()
