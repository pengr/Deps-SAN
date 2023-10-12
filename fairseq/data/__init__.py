# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .dictionary import Dictionary, TruncatedDictionary

from .fairseq_dataset import FairseqDataset, FairseqIterableDataset

from .base_wrapper_dataset import BaseWrapperDataset

from .append_token_dataset import AppendTokenDataset
from .audio.raw_audio_dataset import FileAudioDataset
from .backtranslation_dataset import BacktranslationDataset
from .colorize_dataset import ColorizeDataset
from .concat_dataset import ConcatDataset
from .concat_sentences_dataset import ConcatSentencesDataset
from .denoising_dataset import DenoisingDataset
from .id_dataset import IdDataset
from .indexed_dataset import IndexedCachedDataset, IndexedDataset, IndexedRawTextDataset, MMapIndexedDataset
from .language_pair_dataset import LanguagePairDataset
from .language_sdsa_dataset import LanguageSdsaDataset   # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
from .language_sdsa_new_dataset import LanguageSdsaNewDataset   # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
from .language_sdsa_maxlen_dataset import LanguageSdsaMaxlenDataset   # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
from .language_sdsa_opennmt_dataset import LanguageSdsaOpenNMTDataset   # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
from .language_sdsa_diag_dataset import LanguageSdsaDiagDataset   # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
from .language_pascal_dataset import LanguagePascalDataset   # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
from .list_dataset import ListDataset
from .lm_context_window_dataset import LMContextWindowDataset
from .lru_cache_dataset import LRUCacheDataset
from .mask_tokens_dataset import MaskTokensDataset
from .monolingual_dataset import MonolingualDataset
from .multi_corpus_sampled_dataset import MultiCorpusSampledDataset
from .nested_dictionary_dataset import NestedDictionaryDataset
from .noising import NoisingDataset
from .numel_dataset import NumelDataset
from .num_samples_dataset import NumSamplesDataset
from .offset_tokens_dataset import OffsetTokensDataset
from .pad_dataset import LeftPadDataset, PadDataset, RightPadDataset
from .prepend_dataset import PrependDataset
from .prepend_token_dataset import PrependTokenDataset
from .raw_label_dataset import RawLabelDataset
from .replace_dataset import ReplaceDataset
from .resampling_dataset import ResamplingDataset
from .roll_dataset import RollDataset
from .round_robin_zip_datasets import RoundRobinZipDatasets
from .sort_dataset import SortDataset
from .strip_token_dataset import StripTokenDataset
from .subsample_dataset import SubsampleDataset
from .token_block_dataset import TokenBlockDataset
from .transform_eos_dataset import TransformEosDataset
from .transform_eos_lang_pair_dataset import TransformEosLangPairDataset
from .truncate_dataset import TruncateDataset

from .iterators import (
    CountingIterator,
    EpochBatchIterator,
    GroupedIterator,
    ShardedIterator,
)

__all__ = [
    'AppendTokenDataset',
    'BacktranslationDataset',
    'BaseWrapperDataset',
    'ColorizeDataset',
    'ConcatDataset',
    'ConcatSentencesDataset',
    'CountingIterator',
    'DenoisingDataset',
    'Dictionary',
    'EpochBatchIterator',
    'FairseqDataset',
    'FairseqIterableDataset',
    'GroupedIterator',
    'IdDataset',
    'IndexedCachedDataset',
    'IndexedDataset',
    'IndexedRawTextDataset',
    'LanguagePairDataset',
    'LanguageSdsaDataset',     # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
    'LanguagePascalDataset',   # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
    'LanguageSdsaNewDataset',   # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
    'LanguageSdsaMaxlenDataset',  # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
    'LanguageSdsaDiagDataset',   # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
    'LanguageSdsaOpenNMTDataset',   # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
    'LeftPadDataset',
    'ListDataset',
    'LMContextWindowDataset',
    'LRUCacheDataset',
    'MaskTokensDataset',
    'MMapIndexedDataset',
    'MonolingualDataset',
    'MultiCorpusSampledDataset',
    'NestedDictionaryDataset',
    'NoisingDataset',
    'NumelDataset',
    'NumSamplesDataset',
    'OffsetTokensDataset',
    'PadDataset',
    'PrependDataset',
    'PrependTokenDataset',
    'ReplaceDataset',
    'RollDataset',
    'FileAudioDataset',
    'RawLabelDataset',
    'ResamplingDataset',
    'RightPadDataset',
    'RoundRobinZipDatasets',
    'ShardedIterator',
    'SortDataset',
    'StripTokenDataset',
    'SubsampleDataset',
    'TokenBlockDataset',
    'TransformEosDataset',
    'TransformEosLangPairDataset',
    'TruncateDataset',
    'TruncatedDictionary',
]
