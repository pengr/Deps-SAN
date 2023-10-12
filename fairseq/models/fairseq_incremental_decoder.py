# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import FairseqDecoder
from fairseq.incremental_decoding_utils import with_incremental_state


@with_incremental_state
class FairseqIncrementalDecoder(FairseqDecoder):
    """Base class for incremental decoders. 增量解码器的基类

    Incremental decoding is a special mode at inference time where the Model
    only receives a single timestep of input corresponding to the previous
    output token (for teacher forcing) and must produce the next output
    *incrementally*. Thus the model must cache any long-term state that is
    needed about the sequence, e.g., hidden states, convolutional states, etc.

    Compared to the standard :class:`FairseqDecoder` interface, the incremental
    decoder interface allows :func:`forward` functions to take an extra keyword
    argument (*incremental_state*) that can be used to cache state across
    time-steps.

    The :class:`FairseqIncrementalDecoder` interface also defines the
    :func:`reorder_incremental_state` method, which is used during beam search
    to select and reorder the incremental state based on the selection of beams.
　　 增量解码是推理时的一种特殊模式,其中模型仅接收与上一个输出token(用于教师强制)相对应的单个输入时间步长,并且必须“增量”生成下一个输出
    因此,该模型必须缓存序列所需的任何长期状态, 例如隐藏状态,卷积状态等.

    与标准的FairseqDecoder类相比, 增量解码器接口允许`forward`函数采用额外的关键字参数（* incremental_state *）,
    该参数可用于跨时间步长缓存状态。

    FairseqIncrementalDecoder类接口还定义了reorder_incremental_state方法,
    该方法在波束搜索期间用于根据波束的选择来选择增量状态并对其进行重新排序
    To learn more about how incremental decoding works, refer to `this blog
    <http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/>`_.
    """

    def __init__(self, dictionary):
        super().__init__(dictionary)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        seen = set()
        for module in self.modules():
            if (
                module != self
                and hasattr(module, 'reorder_incremental_state')
                and module not in seen
            ):
                seen.add(module)
                result = module.reorder_incremental_state(incremental_state, new_order)
                if result is not None:
                    incremental_state = result

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        if getattr(self, '_beam_size', -1) != beam_size:
            seen = set()

            def apply_set_beam_size(module):
                if module != self and hasattr(module, 'set_beam_size') \
                        and module not in seen:
                    seen.add(module)
                    module.set_beam_size(beam_size)

            self.apply(apply_set_beam_size)
            self._beam_size = beam_size
