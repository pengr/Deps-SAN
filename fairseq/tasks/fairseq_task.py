# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch

from fairseq import metrics, search, tokenizer, utils
from fairseq.data import data_utils, FairseqDataset, iterators, Dictionary


class FairseqTask(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        pass

    @staticmethod
    def logging_outputs_can_be_summed(criterion) -> bool:
        """
        Whether the logging outputs returned by `train_step` and `valid_step` can
        be summed across workers prior to calling `aggregate_logging_outputs`.
        Setting this to True will improves distributed training speed.
        """
        return criterion.logging_outputs_can_be_summed()

    def __init__(self, args):
        self.args = args
        self.datasets = {}
        self.dataset_to_epoch_iter = {}  # 用于存储dataset数据类:其对应的多轮数据迭代器的键值对的字典

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename 从文件名中加载字典

        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename) # 从文本文件(即dict.en/de.txt,文本格式为:<symbol0> <count0>)加载预先存在的字典,并将其符号添加到Dictionary实例

    @classmethod
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = Dictionary()  # Initialize fairseq.data.Dictionary Class, mapping from symbols to consecutive integers
        for filename in filenames:  # 遍历源和目标训练集路径
            Dictionary.add_file_to_dictionary(
                filename, d, tokenizer.tokenize_line, workers
            )  # 导入路径,字典,以及标记器的函数,进程数; 统计整个源/目标训练集得到字典,包括每个单词的词频计数器d.count
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)  # 得到最终的字典(nwords个单词),由特殊符号开头,去除词频阈值以下的单词,并填充到可被padding_factor整除
        return d

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args, **kwargs)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        raise NotImplementedError

    def dataset(self, split):
        """
        Return a loaded dataset split.返回加载的数据集split

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from fairseq.data import FairseqDataset

        if split not in self.datasets:  # 判断当前dataset中是否存在当前split名称,如datasets['train']
            raise KeyError("Dataset not loaded: " + split)
        if not isinstance(self.datasets[split], FairseqDataset): # 判断当前dataset中当前split对应的类是否为FairseqDataset:
            raise TypeError("Datasets are expected to be of type FairseqDataset")
        return self.datasets[split]  #　返回加载的数据集split类,<fairseq.data.language_pair_dataset.LanguagePairDataset>

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.
        获取一个迭代器, 该迭代器从给定的数据集中生成数据批次
        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch # 用于分批的数据集,<fairseq.data.language_pair_dataset.LanguagePairDataset>
            max_tokens (int, optional): max number of tokens in each batch  # 批次大小(token-level,默认)
                (default: None).
            max_sentences (int, optional): max number of sentences in each   # 批次大小(sents-level)
                batch (default: None).
            max_positions (optional): max sentence length supported by the  # model允许的最大句子长度,即(args.max_source_positions, args.max_target_positions)
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for  # 不为太长的句子引发异常
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to  # 需要一个批次是这个数的整数
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for  # 给定用于可重现的随机数生成器种子
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N  # 将数据迭代器分片为N个分片
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to　# 返回数据迭代器的哪个分片
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process # 多少个子进程用于数据加载,0表示将在主进程中加载数据
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from # 启动数据迭代器的当前轮数
                (default: 1).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split  返回给定数据集split上的批处理迭代器->fairseq.iterators.EpochBatchIterator类
        """
        # For default fairseq task, return same iterator across epochs
        # as datasets are not dynamic, can be overridden in task specific
        # setting.　
        if dataset in self.dataset_to_epoch_iter: # 对于默认的fairseq任务,由于数据集不是动态的,因此可以跨多个epochs返回相同的迭代器,可以在task specific的设置中覆盖该迭代器
            return self.dataset_to_epoch_iter[dataset] # 覆盖方法即直接从self.dataset_to_epoch_iter中仍选出该数据迭代器

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch 用正确的开始时期初始化数据集
        dataset.set_epoch(epoch) # 将在epoch开始时收到更新的epoch编号(什么都没做pass)

        # get indices ordered by example size 获取按句子示例大小排序的索引
        with data_utils.numpy_seed(seed):  # 首先使用指定的seed为所有NumPy参数seed,然后设定状态以记录下数组被打乱的操作
            indices = dataset.ordered_indices()  # 在随机打乱数据集后,按照目标句子的长度从小到大再排序,最后按照源句子的长度从小到大再排序,返回当前数据集的句子序列号

        # filter examples that are too large 过滤掉过长的句子示例
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices,
                dataset,
                max_positions,
                raise_exception=(not ignore_invalid_inputs),
            )  # 返回非过长句子的索引有序列表的ndarray一维数组

        # create mini-batches with given size constraints 在给定size限制的情况下创建mini-batches
        batch_sampler = data_utils.batch_by_size(
            indices,
            dataset.num_tokens,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )  # 返回存储所有批次的列表,每个批次中有句子的编号

        # return a reusable, sharded iterator 返回一个可重用的分片迭代器
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )  # 返回一个基于torch.utils.data.Dataset上的多epoch数据迭代器
        self.dataset_to_epoch_iter[dataset] = epoch_iter  # 将当前dataset类:对应的多epoch数据迭代器的键值对,添加进dataset_to_epoch_iter字典
        return epoch_iter

    def build_model(self, args):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from fairseq import models  # 导入fairseq.models包

        return models.build_model(args, self)  # 通过args参数(-arch)选择所设定的models,这里返回一个TransformerModel类,由与原始论文一致的Transfromer Encoder和Decoder组成

    def build_criterion(self, args):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions  # 导入fairseq.criterions包

        return criterions.build_criterion(args, self) # 通过args参数(-arch)选择所设定的criterions,由criterion对应类的init_args完成对criterion对应类的初始化

    def build_generator(self, args):
        if getattr(args, "score_reference", False):  # 若args中有"score_reference"参数;非默认情况
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search. 选择搜索策略,默认为“波束搜索”,以下均为非默认情况
        sampling = getattr(args, "sampling", False)          # 是否采用随机搜索假设来替代beam search
        sampling_topk = getattr(args, "sampling_topk", -1)   # 是否选择从前K个最可能的词中随机搜索,而不是所有单词
        sampling_topp = getattr(args, "sampling_topp", -1.0)  # 是否选择从最小集合中随机搜索,其中对下一个单词的累积概率值超过p
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1) # 选择用于Diverse Beam Search的组数
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5) # 选择用于Diverse Beam Search的多样性惩罚强度
        match_source_len = getattr(args, "match_source_len", False) # 选择用于Diverse Beam Search的多样性惩罚强度
        diversity_rate = getattr(args, "diversity_rate", -1)  # 选择生成的预测翻译序列应该匹配的最大源长度
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):  # 报错:"提供的搜索参数是互斥的";非默认情况
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        else:  # 默认情况,通过目标端字典Dictionary类来初始化BeamSearch类
            search_strategy = search.BeamSearch(self.target_dictionary)  # 返回加载目标字典的pad/unk/eos符号对应idx,词汇表大小,初始化一个源句子长度的BeamSearch类

        if getattr(args, "print_alignment", False): # 若选择使用注意力反馈来计算和打印对源tokens的对齐(对齐/注意力权重);非默认情况
            seq_gen_cls = SequenceGeneratorWithAlignment
        else:
            seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            self.target_dictionary,  # 目标端字典Dictionary类
            beam_size=getattr(args, "beam", 5),  # 波束搜索的波束大小
            max_len_a=getattr(args, "max_len_a", 0),  # 生成最大长度为ax + b的序列, 其中x为源句子长度(a默认为0)
            max_len_b=getattr(args, "max_len_b", 200),  # 生成最大长度为ax + b的序列, 其中x为源句子长度(b默认为200)
            min_len=getattr(args, "min_len", 1),  # 最小生成序列长度
            normalize_scores=(not getattr(args, "unnormalized", False)), # 通过输出假设的长度归一化分数
            len_penalty=getattr(args, "lenpen", 1), # 长度惩罚α：<1.0有利于较短的句子,>1.0有利于较长的句子
            unk_penalty=getattr(args, "unkpen", 0), # 未知词惩罚：<0产生更多unks，>0产生更少unks
            temperature=getattr(args, "temperature", 1.0),  # 用于预测过程的温度
            match_source_len=getattr(args, "match_source_len", False), # 生成的预测翻译序列应该匹配源长度
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0), # ngram块,使得这样大小的ngram不会在生成的预测翻译序列中被重复
            search_strategy=search_strategy, # 加载目标字典的pad/unk/eos符号对应idx,词汇表大小,初始化一个源句子长度的BeamSearch类
        )  # 返回一个用于生成给定源句子的翻译类SequenceGenerator,其中加载了目标字典的pad/unk/eos符号对应idx,词汇表大小,波束大小,生成最大长度为ax + b的序列,
           # 最小生成序列长度,通过输出假设的长度归一化分数,长度惩罚α,未知词惩罚,生成时是否仍使用dropout,用于预测过程的温度,生成的预测翻译序列应该匹配源长度,
           # ngram块,加载目标字典的pad/unk/eos符号对应idx,词汇表大小,初始化一个源句子长度的BeamSearch类

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True # 若设置ignore_grad为True,则将loss*0来忽略loss

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()  # 将model切换为train模式
        model.set_num_updates(update_num) # 在每次更新时从trainer中传递当前更新批次步骤给模型中所有模块,若该模块存在set_num_updates属性
        loss, sample_size, logging_output = criterion(model, sample) # 返回当前批次所有目标词对应的lable_smoothing loss,当前批次目标tokens数,以及输出记录元组
        if ignore_grad:  # 若设置ignore_grad为True,则将loss*0来忽略loss
            loss *= 0
        optimizer.backward(loss) # 根据链式法则自动计算出计算图叶子节点,给定张量w.r.t.的梯度总和; 当前优化器类<FairseqAdam>调用其父类FairseqOptimizer,
        return loss, sample_size, logging_output # 返回当前批次所有目标词对应的lable_smoothing loss,当前批次目标tokens数,以及输出记录元组

    def valid_step(self, sample, model, criterion):
        model.eval()  # 将model切换为eavl模式
        with torch.no_grad():  # 在验证过程下不修改模型参数,仅对当前模型参数的效果评估
            loss, sample_size, logging_output = criterion(model, sample) # 返回当前验证批次所有目标词对应的lable_smoothing loss,当前批次目标tokens数,以及输出记录元组
        return loss, sample_size, logging_output # 返回当前验证批次所有目标词对应的lable_smoothing loss,当前批次目标tokens数,以及输出记录元组

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad(): # 在测试过程下不修改模型参数,仅对当前模型参数的效果评估
            # 调用用于生成给定源句子的翻译类SequenceGenerator的genrate(),
            # 由存储用于评估的模型(TransformerModel类,模型全部训练参数和缓冲区从模型文件中加载)的列表,一个批次的示例以及目标前缀标识符(强制解码器从这些标识符开始)
            # 得到该批次的最终翻译结果,其中每个句子对应的beam_size个翻译结果按'scores'从小到大排序过
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch. 在每轮开始之前调用的Hook函数"""
        pass

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        """[deprecated] Aggregate logging outputs from data parallel training."""
        utils.deprecation_warning(
            "The aggregate_logging_outputs API is deprecated. "
            "Please use the reduce_metrics API instead."
        )
        with metrics.aggregate() as agg:
            self.reduce_metrics(logging_outputs, criterion)
            return agg.get_smoothed_values()

    def reduce_metrics(self, logging_outputs, criterion):  # 通过存储当前批次的输出记录元组的列表logging_outputs,和当前LabelSmoothedCrossEntropyCriterion类记录
        """Aggregate logging outputs from data parallel training. 汇总来自数据并行训练的输出记录"""
        # backward compatibility for tasks that override aggregate_logging_outputs
        base_func = FairseqTask.aggregate_logging_outputs # 定义两个上下文管理器
        self_func = getattr(self, "aggregate_logging_outputs").__func__
        if self_func is not base_func: # 非默认情况
            utils.deprecation_warning(
                "Tasks should implement the reduce_metrics API. "
                "Falling back to deprecated aggregate_logging_outputs API."
            )
            agg_logging_outputs = self.aggregate_logging_outputs(
                logging_outputs, criterion
            )
            for k, v in agg_logging_outputs.items():
                metrics.log_scalar(k, v)
            return

        if not any("ntokens" in log for log in logging_outputs): # 若logging_outputs不存在ntokens:
            warnings.warn(
                "ntokens not found in Criterion logging outputs, cannot log wpb or wps"
            )
        else:
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs) # 从logging_outputs中取出当前遍历批次的所有目标token数总和
            metrics.log_scalar("wpb", ntokens, priority=180, round=1) # 将"wpb":当前遍历批次的所有目标token数总和,以及优先级,由log_scalar添加到MetersDict中实时记录并更新
            metrics.log_speed("wps", ntokens, priority=90, round=1)

        if not any("nsentences" in log for log in logging_outputs): # 若logging_outputs不存在nsentences:
            warnings.warn(
                "nsentences not found in Criterion logging outputs, cannot log bsz"
            )
        else:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs) # 从logging_outputs中取出当前遍历批次的所有目标句子数
            metrics.log_scalar("bsz", nsentences, priority=190, round=1) # 将"bsz":当前遍历批次的所有目标句子数,以及优先级,由log_scalar添加到MetersDict中实时记录并更新

        criterion.__class__.reduce_metrics(logging_outputs)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError
