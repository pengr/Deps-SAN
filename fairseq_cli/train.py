# #!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import logging
import math
import os
import random
import sys

import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.trainer import Trainer


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.train')


def main(args, init_distributed=False):
    utils.import_user_module(args)  # 由"--user_dir"import可选用户模块(非参数),默认--user_dir为None

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences' # 必须设置bacth size,无论是token-level还是sent-level

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu: # 进行cuda计算(单GPU),设置cuda设备号
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)  # 针对numpy参数,设置固定的随机数生成,仅该次有效
    torch.manual_seed(args.seed) # 针对torch参数,设置固定的随机数生成
    if init_distributed:  # 若进行分布式训练:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):  # 检查save_dir是否存在,由一个dummy文件做写入操作进行确认,正常则删除
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args 打印全部args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc. 设置task,如translation等
    task = tasks.setup_task(args)  # 返回初始化且加载源/目标字典后的TranslationTask类

    # Load valid dataset (we load training data below, based on the latest checkpoint) 加载验证集(我们根据最新的检查点在下面加载训练数据)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)  # 返回源和目标验证集的句子对数据集

    # Build model and criterion 根据全部参数args来建立模型和criterion(损失)
    model = task.build_model(args) # 返回一个TransformerModel类,由与原始论文一致的Transfromer Encoder和Decoder组成
    criterion = task.build_criterion(args) # 通过args参数(-arch)选择所设定的criterions,由criterion对应类(LabelSmoothedCrossEntropyCriterion)的init_args完成对criterion对应类的初始化
    logger.info(model)  # 打印模型结构
    logger.info('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))  # 打印所建立的模型和criterion(损失)
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))  # 打印所建立的模型的参数大小和训练参数大小

    # Build trainer # 由args,task,所建立的模型和criterion(损失)来建立训练器
    trainer = Trainer(args, task, model, criterion)
    logger.info('training on {} GPUs'.format(args.distributed_world_size))  # 打印在几台gpu上训练
    logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))  # 打印在每台gpu上的训练批次(token-level and sents-level)

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator 加载最新的检查点（如果有）并恢复对应的训练迭代器
    # 在该函数下还建立: 训练数据集的epoch数据迭代器,优化器optimizer,以及学习率调度器lr_scheduler
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small　训练直到学习率变得太小,到达args.min-lr(1e-09)
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf  # 设置训练的最大更新批次数
    lr = trainer.get_lr()  # 返回当前的学习率
    train_meter = meters.StopwatchMeter()  # 初始化一个计算事件的持续时间的计时器
    train_meter.start()  # 开始记录代码所持续时间
    valid_subsets = args.valid_subset.split(',')  # 获取验证集split名称['vaild']
    while (
        lr > args.min_lr
        and epoch_itr.next_epoch_idx <= max_epoch
        and trainer.get_num_updates() < max_update   # 若学习率未达到最低学习率min_lr, 当前epoch编号未超过最大轮数,当前更新批次步骤未超过最大更新批次步骤:
    ):
        # train for one epoch 由args,初始化且加载源/目标字典后的TranslationTask类,
        # 由args,task,所建立的模型和criterion(损失)来建立训练器,训练数据集的epoch数据迭代器来训练一轮
        train(args, trainer, task, epoch_itr)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0: # 若不禁止验证,且默认每一轮验证一次
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets) # 故每训练完一轮要进行一次验证,与在一轮内的验证一样
            # 返回验证集平均目标词对应的－lable_smoothing loss/log(2)列表
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate　
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0]) # 每训练完一轮后,根据验证集损失计算当前更新批次步下的学习率,并给优化器更新学习率以及更新MetersDict中lr值

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0: # 若当前轮数满足每多少轮存储一个检查点文件,默认为每轮
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
            # 将当前轮数,更新批次下的所有训练状态和vaild_loss等存到检查点目录中如下检查点文件中: (其中save_checkpoint.best默认为最小val_loss)
            # checkpoint_epoch_updates.pt,checkpoint_best.pt,checkpoint_last.pt文件下(已设好目录)
            # 同时仅保留最后keep_interval_updates个检查点(间隔为--save-interval-updates),且检查点以降序排列

        # early stop 提早停止,非默认设置
        if should_stop_early(args, valid_losses[0]):
            logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(args.patience))
            break

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in getattr(args, 'data', '')),
        ) # 由下一轮轮数来更新epoch_itr,返回一个基于torch.utils.data.Dataset上的多epoch数据迭代器
    train_meter.stop() # 记录代码从开始到全部训练结束总共所持续的时间,并打印出来
    logger.info('done training in {:.1f} seconds'.format(train_meter.sum))


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        return should_stop_early.num_runs >= args.patience


@metrics.aggregate('train')
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # 由args,初始化且加载源/目标字典后的TranslationTask类, 由args,task,所建立的模型和criterion(损失)来建立训练器,训练数据集的epoch数据迭代器来训练一轮
    # Initialize data iterator 初始化数据迭代器
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),  # 若当前epoch编号>args.curriculum,则设置打乱标记为True;默认为True
    )  # 返回self._cur_epoch_itr,可记录迭代数的迭代器,基于torch.data.DataLoader在给定的数据集<LanguagePairDataset>类上创建的数据加载器
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)  # 若当前轮数大于所设置的每N个批次更新一次参数(梯度累积更新),则返回args.update_freq[-1]
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)  # 返回分好块的迭代器GroupedIterator类,基于self._cur_epoch_itr
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,      # 记录所用格式,json
        log_interval=args.log_interval,  # 每多少个batches打印进度,等价于disp_freq
        epoch=epoch_itr.epoch,           # 当前epoch编号
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),  # 用tensorboard保存日志的路径
        default_log_format=('tqdm' if not args.no_progress_bar else 'simple'), # 若设置不报告进度信息,则默认log格式为tadm
    )  # 返回以JSON格式记录输出JsonProgressBar类,加载分好块的迭代器GroupedIterator类,log_interval,当前epoch编号,offset等

    # task specific setup per epoch 每epoch的任务特定设置
    task.begin_epoch(epoch_itr.epoch, trainer.get_model())  # trainer.get_model()获取非warpped模型

    valid_subsets = args.valid_subset.split(',')  # 存储全部验证集spilt名称的列表
    max_update = args.max_update or math.inf  # 获得最大更新批次步骤数
    for samples in progress: # 遍历JsonProgressBar类中的分好块的迭代器GroupedIterator类,其中通过了多个类的__iter__(),得到一个批次的示例
        with metrics.aggregate('train_inner'):  # 通过aggregate()创建一个给定名称下用于整合指标的上下文管理器(meterdicts)
            log_output = trainer.train_step(samples)  # 通过一个批次的示例进行训练,
              # 返回有序字典logging_output,其中键值对有,｛loss":当前批次平均目标词对应的－lable_smoothing loss/log(2);
              # "nll_loss":当前批次平均目标词对应的－nll_loss(y_hot损失)/log(2);'sample_size':当前批次目标tokens数｝
            if log_output is None:  # OOM, overflow, ...
                continue

        # log mid-epoch stats 记录一轮训练中的统计
        num_updates = trainer.get_num_updates()
        if num_updates % args.log_interval == 0:  # 若当前更新批次步骤可以整除进度打印频率:
            stats = get_training_stats(metrics.get_smoothed_values('train_inner')) # 获取在'train_inner'下所汇聚的平滑值,即MetersDict中非"_"开头的key,以及其smoothed_value所组成的有序字典
            # OrderedDict([('loss', 13.988), ('nll_loss', 13.982), ('ppl', 16175.65), ('wps', 0.0), ('ups', 0.0), ('wpb', 2931.0),
            #  ('bsz', 192.0), ('num_updates', 1), ('lr', 1.874875e-07), ('gnorm', 8.824), ('train_wall', 22.0), ('wall', 99.0)])
            progress.log(stats, tag='train_inner', step=num_updates)
            # 'epoch':当前epoch和'update':所完成第几轮的百分之几批次添加进上述'train_inner'下所汇聚的平滑值,并打印出来:
            # OrderedDict([("epoch": 1), ("update": 0.001),('loss', 13.988), ('nll_loss', 13.982), ('ppl', 16175.65), ('wps', 0.0), ('ups', 0.0),
            #  ('wpb', 2931.0), ('bsz', 192.0), ('num_updates', 1), ('lr', 1.874875e-07), ('gnorm', 8.824), ('train_wall', 22.0), ('wall', 99.0)])

            # reset mid-epoch stats after each log interval 在每个进度打印间隔后重置一轮训练中的统计信息, 仍将保留一轮训练完的统计
            # the end-of-epoch stats will still be preserved
            metrics.reset_meters('train_inner')

        if (
            not args.disable_validation  # 若不禁止验证,
            and args.save_interval_updates > 0  # 若每多少updates存储一个检查点文件大于0(存储频率),注:validate-interval为验证频率,默认为1
            and num_updates % args.save_interval_updates == 0  # 若当前更新批次步骤可以整除存储频率:
            and num_updates > 0  # 若当前更新批次步骤>0：
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets) # 返回验证集平均目标词对应的－lable_smoothing loss/log(2)列表
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
            # 将当前轮数,更新批次下的所有训练状态和vaild_loss等存到检查点目录中如下检查点文件中: (其中save_checkpoint.best默认为最小val_loss)
            # checkpoint_epoch_updates.pt,checkpoint_best.pt,checkpoint_last.pt文件下(已设好目录)
            # 同时仅保留最后keep_interval_updates个检查点(间隔为--save-interval-updates),且检查点以降序排列
        if num_updates >= max_update:  # 若超过最大更新批次步骤数,则停止训练
            break

    # log end-of-epoch stats　记录一轮训练完的统计
    stats = get_training_stats(metrics.get_smoothed_values('train')) # 获取在'train'下所汇聚的平滑值,即MetersDict中非"_"开头的key,以及其smoothed_value所组成的有序字典
    # OrderedDict([('loss', 13.215), ('nll_loss', 13.204), ('ppl', 9438.41), ('wps', 0.9), ('ups', 0.0), ('wpb', 2804.0),
    #              ('bsz', 132.0), ('num_updates', 2), ('lr', 2.74975e-07), ('gnorm', 8.076), ('train_wall', 48.0), ('wall', 0.0)])
    progress.print(stats, tag='train', step=num_updates)
    # 将'train(训练完)/root(验证完)'下所汇聚的平滑值,以及'epoch':当前epoch, (训练完特有)'update':所完成第几轮的百分之几批次添加进一个OrderDict()
    # {"epoch": 1, "train_loss": "13.215", "train_nll_loss": "13.204", "train_ppl": "9438.41", "train_wps": "0.9", "train_ups": "0",
    # "train_wpb": "2804", "train_bsz": "132", "train_num_updates": "2", "train_lr": "2.74975e-07",
    # "train_gnorm": "8.076", "train_train_wall": "48", "train_wall": "0"}

    # reset epoch-level meters 重置轮数级别(训练完一轮)的指标器
    metrics.reset_meters('train')


def get_training_stats(stats):  # stats为在'train_inner'下所汇聚的平滑值,即MetersDict中非"_"开头的key,以及其smoothed_value所组成的有序字典
    if 'nll_loss' in stats and 'ppl' not in stats:  # 若stats不存在'ppl',则由所存在'nll_loss'计算'ppl'
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)  # 将"wall":总运行时间,以及优先级,由log_scalar添加到MetersDict中实时记录并更新
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:  # 为每一次验证过程设置固定的随机数生成器种子,非默认情况
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []  # 用于存储验证集平均目标词对应的－lable_smoothing loss/log(2)的列表
    for subset in subsets:  # 遍历存储全部验证集spilt名称的列表
        # Initialize data iterator
        # 初始化验证集的数据迭代器,与训练集基本一致,区别:ignore_invalid_inputs=False(仅当max_positions小于验证集中句子长度时生效),shuffle=False(验证集不打乱)
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset), #　根据subset返回加载的数据集split类,<fairseq.data.language_pair_dataset.LanguagePairDataset>
            max_tokens=args.max_tokens_valid,  # 批次大小(token-level,默认与max_tokens一致)
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(), # 返回task允许的最大句子长度,即(self.args.max_source_positions, self.args.max_target_positions)
                trainer.get_model().max_positions(), # 返回model允许的最大句子长度,即(self.args.max_source_positions, self.args.max_target_positions)
            ), # 通过resolve_max_positions解决来自多个来源的排名限制,返回(self.args.max_source_positions, self.args.max_target_positions)
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,  # 为False,为太长的句子引发异常
            required_batch_size_multiple=args.required_batch_size_multiple, # 需要一个批次是这个数的整数
            seed=args.seed,  # 给定随机数种子
            num_shards=args.distributed_world_size,  # 将数据集分成distributed_world_size片
            shard_id=args.distributed_rank,         # 将数据集的所有分片带上序号
            num_workers=args.num_workers,          # 多少个子进程用于数据加载,0表示将在主进程中加载数据
        ).next_epoch_itr(shuffle=False)  # next_epoch_itr前,返回一个基于torch.utils.data.Dataset上的多epoch数据迭代器(验证集)
        # next_epoch_itr后, 返回self._cur_epoch_itr可记录迭代数的迭代器,基于torch.data.DataLoader在给定的数据集<LanguagePairDataset>类上创建的数据加载器(验证集)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,      # 记录所用格式,json
            log_interval=args.log_interval,  # 每多少个batches打印进度,等价于disp_freq
            epoch=epoch_itr.epoch,           # 当前epoch编号,由多epoch数据迭代器(训练集)给定
            prefix=f"valid on '{subset}' subset",  #
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),      # 用tensorboard保存日志的路径
            default_log_format=('tqdm' if not args.no_progress_bar else 'simple'), # 若设置不报告进度信息,则默认log格式为tadm
        )  # 返回以JSON格式记录输出JsonProgressBar类(验证集),加载self._cur_epoch_itr可记录迭代数的迭代器,log_interval,当前epoch编号,offset等

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)　
        with metrics.aggregate(new_root=True) as agg: # 创建一个新的root指标汇聚器(上下文管理器),以便验证指标不影响其他汇聚器(如训练指标)
            for sample in progress: # 遍历JsonProgressBar类中的self._cur_epoch_itr可记录迭代数的迭代器,其中通过了多个类的__iter__(),得到一个批次的示例
                # id': 当前遍历的batch中全部句子对的ID的Tensor,按照源句子降序的order重新排列每行;'nsentences':句子对数;'ntokens':该批次所有目标句子的tokens总数;
                # 'net_input':
                #    'src_tokens':当前遍历的batch中全部源句子的tensor所转换的二维填充向量,按照源句子降序的order重新排列每行;
                #    'src_lengths':将当前遍历的batch中全部源句子的长度的Tensor;
                #    'prev_output_tokens':创建目标的移位版本(即全部目标句子的eos移到开头)
                # 'target': 当前遍历的batch中全部目标句子的tensor所转换的二维填充向量,按照源句子降序的order重新排列每行;
                trainer.valid_step(sample)
                # 返回有序字典logging_output,其中键值对有,｛loss":当前验证批次平均目标词对应的－lable_smoothing loss/log(2);
                # "nll_loss":当前验证批次平均目标词对应的－nll_loss(y_hot损失)/log(2);'sample_size':当前验证批次目标tokens数｝

        # log validation stats 记录一轮验证完的统计
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values()) # 获取在root指标下所汇聚的平滑值,即MetersDict中非"_"开头的key,以及其smoothed_value所组成的有序字典
            # OrderedDict([('loss', 13.211), ('nll_loss', 13.2), ('ppl', 9412.54), ('wps', 1.3), ('wpb', 648.0), ('bsz', 25.0), ('num_updates', 1)])
        progress.print(stats, tag=subset, step=trainer.get_num_updates())# 将'root(验证完)'下所汇聚的平滑值(带上tag='vaild'),以及'epoch':当前epoch, 添加进一个OrderDict()
        # {"epoch": 1, "valid_loss": "13.211", "valid_nll_loss": "13.2", "valid_ppl": "9412.54", "valid_wps": "3.5", "valid_wpb": "648",
        #  "valid_bsz": "25", "valid_num_updates": "1"}
        valid_losses.append(stats[args.best_checkpoint_metric]) # 默认best_checkpoint_metric为loss,即将stats中vaild_loss存储进验证集平均目标词对应的－lable_smoothing loss/log(2)的列表
    return valid_losses  # 返回验证集平均目标词对应的－lable_smoothing loss/log(2)列表


def get_valid_stats(args, trainer, stats): # stats为在root指标下下所汇聚的平滑值,即MetersDict中非"_"开头的key,以及其smoothed_value所组成的有序字典
    if 'nll_loss' in stats and 'ppl' not in stats:  # 若stats不存在'ppl',则由所存在'nll_loss'计算'ppl'
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates() # 将"num_updates":当前批次更新步骤添加到MetersDict中实时记录并更新
    if hasattr(checkpoint_utils.save_checkpoint, 'best'): # 若checkpoint_utils.save_checkpoint存在'best';非默认情况
        key = 'best_{0}'.format(args.best_checkpoint_metric) # 默认最佳检查点指标为loss
        best_function = max if args.maximize_best_checkpoint_metric else min # 默认最佳检查点指标越小越好
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        ) # 将到目前为止的最佳检查点指标添加进stats中
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main(modify_parser=None):
    parser = options.get_training_parser()  # 根据特定的任务得到训练设置
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser) # 返回全部的arguments设置

    if args.distributed_init_method is None: # 若不采用分布式训练,infer_init_method()不执行任何操作
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:  # 若分布式训练存在建立初始连接:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:    # 若分布式训练设置了多个GPU
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
