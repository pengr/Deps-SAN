# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch.nn as nn
from fairseq import utils
from torch import Tensor


class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False

    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def output_layer(self, features, **kwargs):
        """
        Project features to the default output size, e.g., vocabulary size.

        Args:
            features (Tensor): features returned by *extract_features*.
        """
        raise NotImplementedError

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Dict[str, List[Optional[Tensor]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]],
    ):
        """Get normalized probabilities (or log probs) from a net's output. 从模型的输出中获取归一化的概率(或对数概率)"""
        # net_output->模型的输出(经过"Linear层"的解码器输出,最后一个解码器层enc-dec self-attn的输出attn_weights在全部注意力头上平均注意力权重,存储每一层解码器的输出列表)
        # log_probs为True,sample为None
        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None: # 非默认情况
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]  # 从net_output中取出->经过"Linear层"的解码器输出logits
        if log_probs:  # 若选择返回log概率, 则对经过"Linear层"的解码器输出logits进行log_softmax(经过vaswani论文的"Softmax"),否则对其进行softmax
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
