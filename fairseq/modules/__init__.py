# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .adaptive_input import AdaptiveInput
from .adaptive_softmax import AdaptiveSoftmax
from .beamable_mm import BeamableMM
from .character_token_embedder import CharacterTokenEmbedder
from .conv_tbc import ConvTBC
from .downsampled_multihead_attention import DownsampledMultiHeadAttention
from .dynamic_convolution import DynamicConv, DynamicConv1dTBC
from .dynamic_crf_layer import DynamicCRF
from .fp32_group_norm import Fp32GroupNorm
from .gelu import gelu, gelu_accurate
from .grad_multiply import GradMultiply
from .gumbel_vector_quantizer import GumbelVectorQuantizer
from .kmeans_vector_quantizer import KmeansVectorQuantizer
from .layer_norm import Fp32LayerNorm, LayerNorm
from .learned_positional_embedding import LearnedPositionalEmbedding
from .lightweight_convolution import LightweightConv, LightweightConv1dTBC
from .linearized_convolution import LinearizedConvolution
from .multihead_attention import MultiheadAttention
from .multihead_sdsa_attention import MultiheadSdsaAttention  # <<改写>>,sdsa模型的模块化
from .multihead_sdsa_pad_attention import MultiheadSdsaPadAttention  # <<改写>>,sdsa模型的模块化
from .multihead_pascal_attention import MultiheadPascalAttention  # <<改写>>,pascal模型的模块化
from .multihead_localness_attention import MultiheadLocalnessAttention  # <<改写>>,localness模型的模块化
from .multihead_context_attention import MultiheadContextAttention  # <<改写>>,context模型的模块化
from .multihead_rpe_attention import MultiheadRpeAttention  # <<改写>>,rpe模型的模块化
from .multihead_dep_intvl_rpe_attention import MultiheadDepIntvlRpeAttention # <<改写>>,DepIntvlRpe模型的模块化
from .positional_embedding import PositionalEmbedding
from .scalar_bias import ScalarBias
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from .transformer_sentence_encoder_layer import TransformerSentenceEncoderLayer
from .transformer_sentence_encoder import TransformerSentenceEncoder
from .unfold import unfold1d
from .transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer
from .transformer_sdsa_layer import TransformerSdsaEncoderLayer  # <<改写>>,sdsa模型的模块化
from .transformer_sdsa_pad_layer import TransformerSdsaPadEncoderLayer  # <<改写>>,sdsa模型的模块化
from .transformer_pascal_layer import TransformerPascalEncoderLayer  # <<改写>>,pascal模型的模块化
from .transformer_localness_layer import TransformerLocalnessEncoderLayer  # <<改写>>,localness模型的模块化
from .transformer_context_layer import TransformerContextEncoderLayer  # <<改写>>,context模型的模块化
from .transformer_rpe_layer import TransformerRpeDecoderLayer, TransformerRpeEncoderLayer  # <<改写>>,rpe模型的模块化
from .transformer_dep_intvl_rpe_layer import TransformerDepIntvlRpeEncoderLayer # <<改写>>,DepIntvlRpe模型的模块化
from .vggblock import VGGBlock

__all__ = [
    'AdaptiveInput',
    'AdaptiveSoftmax',
    'BeamableMM',
    'CharacterTokenEmbedder',
    'ConvTBC',
    'DownsampledMultiHeadAttention',
    'DynamicConv1dTBC',
    'DynamicConv',
    'DynamicCRF',
    'Fp32GroupNorm',
    'Fp32LayerNorm',
    'gelu',
    'gelu_accurate',
    'GradMultiply',
    'GumbelVectorQuantizer',
    'KmeansVectorQuantizer',
    'LayerNorm',
    'LearnedPositionalEmbedding',
    'LightweightConv1dTBC',
    'LightweightConv',
    'LinearizedConvolution',
    'MultiheadAttention',
    'MultiheadSdsaAttention',  # <<改写>>,sdsa模型的模块化
    'MultiheadSdsaPadAttention',   # <<改写>>,sdsa模型的模块化
    'MultiheadPascalAttention',  # <<改写>>,pascal模型的模块化
    'MultiheadLocalnessAttention',  # <<改写>>,localness模型的模块化
    'MultiheadContextAttention',  # <<改写>>,context模型的模块化
    'MultiheadRpeAttention',  # <<改写>>,rpe模型的模块化
    'MultiheadDepIntvlRpeAttention',  # <<改写>>,DepIntvlRpe模型的模块化
    'PositionalEmbedding',
    'ScalarBias',
    'SinusoidalPositionalEmbedding',
    'TransformerSentenceEncoderLayer',
    'TransformerSentenceEncoder',
    'TransformerDecoderLayer',
    'TransformerEncoderLayer',
    'TransformerSdsaEncoderLayer',  # <<改写>>,sdsa模型的模块化
    'TransformerSdsaPadEncoderLayer',   # <<改写>>,sdsa模型的模块化
    'TransformerPascalEncoderLayer',  # <<改写>>,pascal模型的模块化
    'TransformerLocalnessEncoderLayer',  # <<改写>>,localness模型的模块化
    'TransformerContextEncoderLayer',  # <<改写>>,context模型的模块化
    'TransformerRpeDecoderLayer',  # <<改写>>,rpe模型的模块化
    'TransformerRpeEncoderLayer',  # <<改写>>,rpe模型的模块化
    'TransformerDepIntvlRpeEncoderLayer', # <<改写>>,DepIntvlRpe模型的模块化
    'VGGBlock',
    'unfold1d',
]
