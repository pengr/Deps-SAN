# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import json
import itertools
import logging
import os

import numpy as np

from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from fairseq.tasks import FairseqTask, register_task

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang)) # 由split(分片名)和源,目标句子,以及数据集的dir路径得到split的数据路径前缀
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)  # 再由split的数据路径前缀,以及输出数据集格式判断该路径下.idx和.bin文件是否存在

    src_datasets = []  # 存储源端全部数据集(针对vaild,vaild1等情况),均为dataset类,比如MMapIndexedDataset类
    tgt_datasets = []

    for k in itertools.count(): # 创建一个从0开始的遍历器,0~...
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))  # 得到split_k的数据路径前缀,如'data-bin/iwslt14.tokenized.de-en/valid.de-en.'
        elif split_exists(split_k, tgt, src, src, data_path):  # 若src-tgt语言对下未查找到,则更换语言对为tgt-src来查找
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:  # 若不存在多个数据分片,如果"train","train1",则break
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        # 根据源数据集完整路径,源字典,以及输出数据格式(未给定可在函数内推理得到)
        # 完成对<MMapIndexedDataset>类的初始化, 对prefix+src路径下.idx和.bin文件的二进制读取,采用np.memmap()方法小段读入
        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)

        if truncate_source:  # 若选择截断源句子,非默认情况
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)  # 将所得对prefix+src路径下.idx和.bin文件读取的MMapIndexedDataset类,存储到src_datasets

        # 根据目标数据集完整路径,源字典,以及输出输出格式(未给定可在函数内推理得到)
        # 完成对<MMapIndexedDataset>类的初始化, 对prefix+tgt路径下.idx和.bin文件的二进制读取,采用np.memmap()方法小段读入
        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)  # 将所得对prefix+tgt.idx和.bin文件读取的MMapIndexedDataset类,存储到tgt_datasets

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))  # 打印数据集dir路径,所处理的数据集split名称("valid"),源和目标语言,源数据集句子总数

        if not combine:  # 若未存在多个数据集情况:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:  # 默认情况,一个源和目标数据集(验证集),均为所得对prefix+src/tgt.idx和.bin文件读取的MMapIndexedDataset类
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:  # 若预先考虑bos(句子开始标记);非默认情况
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:  # 若附加源ID;非默认情况
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    align_dataset = None
    if load_alignments:  # 若加载对齐文本;非默认情况
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None  # 获取目标句子长度列表
    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict, # prefix+src下.idx和.bin文件读取的MMapIndexedDataset类,源句子长度列表,源字典Dictionray类
        tgt_dataset, tgt_dataset_sizes, tgt_dict, # prefix+tgt下.idx和.bin文件读取的MMapIndexedDataset类,目标句子长度列表,目标字典Dictionray类
        left_pad_source=left_pad_source,  # 源端从左开始填充pad符
        left_pad_target=left_pad_target,  # 目标端从左开始填充pad符
        max_source_positions=max_source_positions, # 最大源句子长度
        max_target_positions=max_target_positions, # 最大目标句子长度
        align_dataset=align_dataset, eos=eos  #对齐数据,eos符号对应idx,此处均为None
    )  # 通过上述参数->得到源和目标的句子对数据集LanguagePairDataset类


@register_task('translation')
class TranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')  # 冒号分隔的数据目录列表路径,将在循环期间以循环方式进行迭代
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')  # 源语言
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')  # 目标语言
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')  # 加载数字化的对齐
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')  # 在左侧填充源句子(开头)
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')  # 在左侧填充目标句子(开头)
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')  # 在源句子中最大tokens数,用于positional encnding
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')  # 在目标句子中最大tokens数,用于positional encnding
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')  # 向上采样主数据集的数量
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')  # 截断源位置在max-source-positions内;默认为False

        # options for reporting BLEU during validation 在验证过程中用于报告BLEU的选项
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')  # 以BLEU分进行评估,需要设置--eval-bleu生效
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenizer before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')  # 如果使用--eval-bleu则需要在计算BLEU之前先detok(如mose);使用"space"来禁止detok,具体见fairseq.data.encoders的其他选项
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')  # 如果需要设置,则作为用于构建tokenizer的args
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='if setting, we compute tokenized BLEU instead of sacrebleu')  # 若设置,我们计算tokenized BLEU来代替sacrebleu
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')    # 在计算BLEU前,移除BPE
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')  # 用于BLUE打分的测试参数,如{"beam": 4, "lenpen": 0.6}
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')  # 在验证期间打印生成的样本;默认false
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):  # TranslationTask类的初始化,并用于加载字典(Dictionary类)
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).设置任务(例如,加载词典)

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source) # 将args.left_pad_source和left_pad_target从str类型变成bool类型
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)  # 得到数据集dir路径,并确保不为空
        assert len(paths) > 0
        # find language pair automatically 自动确定语言对
        if args.source_lang is None or args.target_lang is None:  # 根据数据集dir路径来搜索源和目标语言
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries 根据数据集dir路径和源/目标语言确定字典路径,并调用父类FairseqTask的load_dictionary,
        # 从文本文件(即dict.en/de.txt,文本格式为:<symbol0> <count0>)加载预先存在的字典,并将其符号添加到当前实例(Dictionary)
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad() # 确认源和目标字典中三种填充符一致,我们采用jointed dict肯定一致
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict))) # 打印源和目标字典的词汇表大小
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)  # 返回初始化且加载源/目标字典(两个Dictionary类)后的TranslationTask类

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split. 加载一个给定的数据分片(分片的名称:train, valid, test)

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)  # 返回数据集的dir路径(列表形式),并确认不为空
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]  # 由(epoch-1)除以数据集个数作为索引,从数据集的dir路径列表取出路径

        # infer langcode 得到语言对
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
        )  # 返回源和目标的句子对数据集LanguagePairDataset类

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_model(self, args):
        if getattr(args, 'eval_bleu', False):  # 取出args.eval_bleu参数,默认为False
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator(Namespace(**gen_args))
        return super().build_model(args)  # 调用Translation的父类Fairseqtask.build_model来建立模型,这里返回一个TransformerModel类,由与原始论文一致的Transfromer Encoder和Decoder组成

    def valid_step(self, sample, model, criterion): # 与train的区别,不需要优化器更新模型参数,设置当前更新批次步骤数和忽略计算grad标记
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)  # 调用父类FairseqTask的方法
        #  返回当前验证批次所有目标词对应的lable_smoothing loss,当前批次目标tokens数,以及输出记录元组
        if self.args.eval_bleu:  # 若采用BLEU分进行评估
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output  #  返回当前验证批次所有目标词对应的lable_smoothing loss,当前批次目标tokens数,以及输出记录元组

    def reduce_metrics(self, logging_outputs, criterion): # 通过存储当前批次的输出记录元组的列表,和当前LabelSmoothedCrossEntropyCriterion类记录
        super().reduce_metrics(logging_outputs, criterion)
        # 调用父类FairseqTask方法,将如下输出值以及优先级,由log_scalar添加到MetersDict中实时记录并更新
        # "wpb":当前遍历批次的所有目标token数总和; "bsz":当前遍历批次的所有目标句子数;"loss":当前批次平均目标词对应的－lable_smoothing loss/log(2);
        # "nll_loss":当前批次平均目标词对应的－nll_loss(y_hot损失)/log(2);"ppl":2^(－nll_loss)模型的ppl
        if self.args.eval_bleu:  # 若采用BLEU分进行评估

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu) # 将"bleu":所计算得到的bleu分,以及优先级,由log_scalar添加到MetersDict中实时记录并更新

    def max_positions(self):
        """Return the max sentence length allowed by the task. 返回task允许的最大句子长度"""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`. # 返回源端字典Dictionary类"""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`.# 返回目标端字典Dictionary类"""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                escape_unk=escape_unk,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        tokenize = sacrebleu.DEFAULT_TOKENIZER if not self.args.eval_tokenized_bleu else 'none'
        return sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)
