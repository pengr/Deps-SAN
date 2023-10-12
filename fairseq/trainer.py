# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

import contextlib
from itertools import chain
import logging
import sys
from typing import Any, Dict, List

import torch

from fairseq import checkpoint_utils, distributed_utils, models, optim, utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics
from fairseq.nan_detector import NanDetector
from fairseq.optim import lr_scheduler


logger = logging.getLogger(__name__)


class Trainer(object):
    """Main class for data parallel training. 数据并行训练的主要类

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.

    此类支持同步分布式数据并行训练,其中每个worker都有完整的模型副本,并且在每次更新之前在每个worker之间累积梯度,
    我们使用：class：`〜torch.nn.parallel.DistributedDataParallel`来处理跨workers的梯度积累
    """

    def __init__(self, args, task, model, criterion):
        self.args = args  # 获取args
        self.task = task  # 获取当前task-><fairseq.tasks.translation.TranslationTask object at 0x7f956cc4c908>

        self.cuda = torch.cuda.is_available() and not args.cpu  # 获取是否存在可用GPU
        if self.cuda:  # 若存在可用GPU,定义设备号
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # copy model and criterion to current device
        self._criterion = criterion  # 获取所建立的损失
        self._model = model  # 获取所建立的模型
        if args.fp16:  # 若模型所用数据类型为fp16,则将所建立的模型和损失全部.to(torch.float16)
            self._criterion = self._criterion.half()
            self._model = self._model.half()
        self._criterion = self._criterion.to(device=self.device)  # 将所建立的模型和损失复制到当前设备
        self._model = self._model.to(device=self.device)

        self._dummy_batch = "DUMMY"  # indicates we don't have a dummy batch at first　表示我们一开始没有空批次
        self._lr_scheduler = None    # 初始化学习率策略类
        self._num_updates = 0        # 初始化当前所更新批次
        self._optim_history = None   #
        self._optimizer = None       # 初始化优化器类
        self._warn_once = set()      #
        self._wrapped_criterion = None  # 初始化包装后的损失
        self._wrapped_model = None   # 初始化包装后的模型

        if self.cuda and args.distributed_world_size > 1:  # 若存在可用GPU,且总共可用Gpu数超过1,则在每次更新之前在每个worker之间累积梯度
            self._grad_norm_buf = torch.cuda.DoubleTensor(args.distributed_world_size)
        else:
            self._grad_norm_buf = None

        metrics.log_start_time("wall", priority=790, round=0)  # 记录某些事件的持续时间（以秒为单位）

    @property
    def criterion(self):
        if self._wrapped_criterion is None:  # 若_wrapped_criterion为None:
            if (
                utils.has_parameters(self._criterion)
                and self.args.distributed_world_size > 1
                and not self.args.use_bmuf  # 若self._criterion已有parameters(),且可利用的gpu数>1,且use_bmuf为False:
            ):
                self._wrapped_criterion = models.DistributedFairseqModel(
                    self.args, self._criterion
                )
            else:  # 默认情况,将所建立的criterion赋给self._wrapped_criterion
                self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def model(self):
        if self._wrapped_model is None:  # 若_wrapped_model为None:
            if self.args.distributed_world_size > 1 and not self.args.use_bmuf:  # 若可利用的gpu数>1,且use_bmuf为False:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.args, self._model
                )
            else:  # 默认情况,将所建立的model赋给self._wrapped_model
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:  # 若_optimizer为None,则重新建立一遍optimizer
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None: # 若_lr_scheduler为空,则需先进行初始化一个InverseSquareRootSchedule类
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler  # 返回所初始化的学习率调度器,如一个InverseSquareRootSchedule类

    def _build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), self.criterion.parameters()),
            )
        )  # 遍历所建立的model和criterion的全部参数(仅参数值),选择出所有需要计算梯度的参数

        if self.args.fp16:  # 若所选择模型的数据类型为fp16:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                logger.info(
                    "NOTE: your device does NOT support faster training with --fp16, "
                    "please switch to FP32 which is likely to be faster"
                )
            if self.args.memory_efficient_fp16:
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(
                    self.args, params
                )
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:  # 若存在可用gpu且cuda计算能力超过7:
                logger.info("NOTE: your device may support faster training with --fp16")
            self._optimizer = optim.build_optimizer(self.args, params)  # 默认情况,由args(具体为lr, betas, eps, weight_decay)和所有model,criterion中所有需要计算梯度的参数来建立优化器adam类

        if self.args.use_bmuf:
            self._optimizer = optim.FairseqBMUF(self.args, self._optimizer)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.　构建优化器后,应立即初始化lr scheduler,以便设置初始学习率
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer) # 根据args(具体为warmup-init-lr,lr,warmup-updates)和所建立的优化器adam类来建立InverseSquareRootSchedule类
        self._lr_scheduler.step_update(0)  # 返回当前更新批次步骤(0表示初始化)给优化器设置的学习率

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file. 将所有训练状态保存在检查点文件中"""
        if distributed_utils.is_master(self.args):  # only save one checkpoint,仅保存一个检查点文件
            extra_state["metrics"] = metrics.state_dict()
            checkpoint_utils.save_state(
                filename,
                self.args,
                self.get_model().state_dict(),
                self.get_criterion(),
                self.optimizer,
                self.lr_scheduler,
                self.get_num_updates(),
                self._optim_history,
                extra_state,
            )

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """Load all training state from a checkpoint file. 从检查点文件加载所有训练状态"""
        extra_state, self._optim_history, last_optim_state = None, [], None

        bexists = PathManager.isfile(filename) # 判断当前filename不为路径
        if bexists:  # 若存在则进行加载
            state = checkpoint_utils.load_checkpoint_to_cpu(filename)

            # load model parameters
            try:
                self.get_model().load_state_dict(
                    state["model"], strict=True, args=self.args
                )
                if utils.has_parameters(self.get_criterion()):
                    self.get_criterion().load_state_dict(
                        state["criterion"], strict=True
                    )
            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )

            extra_state = state["extra_state"]
            self._optim_history = state["optimizer_history"]
            last_optim_state = state.get("last_optimizer_state", None)

        if last_optim_state is not None and not reset_optimizer: # 若last_optim_state不为None,且选择从检查点中加载优化器状态:
            # rebuild optimizer after loading model, since params may have changed 加载模型后重建优化器，因为参数可能已更改
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert (
                last_optim["criterion_name"] == self.get_criterion().__class__.__name__
            ), "Criterion does not match; please reset the optimizer (--reset-optimizer)."
            assert (
                last_optim["optimizer_name"] == self.optimizer.__class__.__name__
            ), "Optimizer does not match; please reset the optimizer (--reset-optimizer)."

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self.set_num_updates(last_optim["num_updates"])

        if extra_state is not None:  # 若extra_state不为None:
            epoch = extra_state["train_iterator"]["epoch"]
            logger.info(
                "loaded checkpoint {} (epoch {} @ {} updates)".format(
                    filename, epoch, self.get_num_updates()
                )
            )

            self.lr_step(epoch)

            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()
        else:
            logger.info("no existing checkpoint found {}".format(filename)) # 打印当前filename下无已存在的检查点文件;默认情况

        return extra_state

    def get_train_iterator(
        self,
        epoch,
        combine=True,
        load_dataset=True,
        data_selector=None,
        shard_batch_itr=True,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch. 对于一个给定轮上在训练集上返回EpochBatchIterator"""
        if load_dataset:  # 若加载数据集
            logger.info("loading train data for epoch {}".format(epoch))  # 打印为当前轮加载训练数据
            self.task.load_dataset(
                self.args.train_subset,  # 给定的数据分片名称('train')
                epoch=epoch,             # 给定轮数
                combine=combine,         # 给定是否合并当前路径下的所有数据分片,如"train","train1"
                data_selector=data_selector,
            )  # 返回源和目标训练集的句子对数据集
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.args.train_subset),  #　返回加载的数据集split类,<fairseq.data.language_pair_dataset.LanguagePairDataset>
            max_tokens=self.args.max_tokens,  # 批次大小(token-level,默认)
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),  # 返回task允许的最大句子长度,即(self.args.max_source_positions, self.args.max_target_positions)
                self.model.max_positions(), # 返回model允许的最大句子长度,即(self.args.max_source_positions, self.args.max_target_positions)
                self.args.max_tokens,
            ), # 通过resolve_max_positions解决来自多个来源的排名限制,返回(self.args.max_source_positions, self.args.max_target_positions)
            ignore_invalid_inputs=True,  # 不为太长的句子引发异常的标记
            required_batch_size_multiple=self.args.required_batch_size_multiple,  # 需要一个批次是这个数的整数
            seed=self.args.seed,  # 给定随机数种子
            num_shards=self.args.distributed_world_size if shard_batch_itr else 1,  # 若shard_batch_itr为true,则将数据集分成distributed_world_size片
            shard_id=self.args.distributed_rank if shard_batch_itr else 0,  # 若shard_batch_itr为true,则将数据集的所有分片带上序号
            num_workers=self.args.num_workers,  # 多少个子进程用于数据加载,0表示将在主进程中加载数据
            epoch=epoch,  # 给定当前轮数
        ) # 返回一个基于torch.utils.data.Dataset上的多epoch数据迭代器

    @metrics.aggregate("train")
    def train_step(self, samples, raise_oom=False):
        """Do forward, backward and parameter update. """
        if self._dummy_batch == "DUMMY": # 若_dummy_batch为"DUMMY",取出samples[0]
            self._dummy_batch = samples[0]

        self._set_seed()    # 根据args.seed和更新批次数设置种子,以便从检查点恢复时可得到可复现的结果
        self.model.train()  # 设置所建立的model和criterion为train模式
        self.criterion.train()
        self.zero_grad()    # 清除所有优化器参数的梯度

        metrics.log_start_time("train_wall", priority=800, round=0)  # 记录某些事件的持续时间（以秒为单位）

        # forward and backward pass
        logging_outputs, sample_size, ooms = [], 0, 0  # 存储当前批次的输出记录元组的列表,当前遍历批次的所有目标token数总和,以及ooms
        for i, sample in enumerate(samples):  # 读取当前批次
            # id': 当前遍历的batch中全部句子对的ID的Tensor,按照源句子降序的order重新排列每行;'nsentences':句子对数;'ntokens':该批次所有目标句子的tokens总数;
            # 'net_input':
            #    'src_tokens':当前遍历的batch中全部源句子的tensor所转换的二维填充向量,按照源句子降序的order重新排列每行;
            #    'src_lengths':将当前遍历的batch中全部源句子的长度的Tensor;
            #    'prev_output_tokens':创建目标的移位版本(即全部目标句子的eos移到开头)
            # 'target': 当前遍历的batch中全部目标句子的tensor所转换的二维填充向量,按照源句子降序的order重新排列每行;
            sample = self._prepare_sample(sample)  # 将sample中的tensor全部移到cuda中

            if sample is None:  # 若sample不为空,则is_dummy_batch = False
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients 当样本为None时, 在虚拟批次上运行forward/backward, 并忽略最终的梯度
                sample = self._prepare_sample(self._dummy_batch)
                is_dummy_batch = True
            else:
                is_dummy_batch = False

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                每当*samples*包含多个mini-batch时,我们希望在本地累积梯度,并且仅在最后的backwards中调用all-reduce
                """
                if (
                    self.args.distributed_world_size > 1
                    and hasattr(self.model, "no_sync")
                    and i < len(samples) - 1
                ):  # 若存在多个可用gpu,且self.model存在非同步参数,以及当前批次未到全部批次的最后一个
                    return self.model.no_sync() # 将self.model切换为异步模式
                else:  # 默认情况,创建一个虚拟上下文管理器
                    return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync(): # 在所创建的虚拟上下文管理器下:
                    # forward and backward
                    loss, sample_size_i, logging_output = self.task.train_step(
                        sample=sample,
                        model=self.model,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        update_num=self.get_num_updates(),
                        ignore_grad=is_dummy_batch,
                    )  # 返回当前批次所有目标词对应的lable_smoothing loss,当前批次目标tokens数,以及输出记录元组
                    del loss  # 删除当前批次所有目标词对应的lable_smoothing loss

                # logging_output = {
                #     'loss': loss.data,  # 当前批次所有目标词对应的lable_smoothing loss
                #     'nll_loss': nll_loss.data,  # 当前批次所有目标词对应的nll_loss(y_hot损失)
                #     'ntokens': sample['ntokens'],  # 当前批次目标tokens数
                #     'nsentences': sample['target'].size(0),  # 当前批次目标句子数
                #     'sample_size': sample_size,  # 若sentence_avg标记为true,则sample大小为句子数,否则为tokens数
                # }
                logging_outputs.append(logging_output)  # 将当前批次的输出记录元组,存储当前批次的输出记录元组的列表
                sample_size += sample_size_i   # 记录当前遍历批次的所有目标token数总和

                # emptying the CUDA cache after the first step can
                # reduce the chance of OOM　第一个更新批次步骤清空CUDA缓存可以减少OOM的机会
                if self.cuda and self.get_num_updates() == 0: # 在第一个更新批次步骤清空CUDA缓存:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if raise_oom:
                        raise e
                    logger.warning(
                        "attempting to recover from OOM in forward/backward pass"
                    )
                    ooms += 1
                    self.zero_grad()
                else:
                    raise e

        if torch.is_tensor(sample_size):  # float化当前遍历批次的所有目标token数总和
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        if is_dummy_batch: # 若当前批次为虚拟批次,则忽略当前遍历批次的所有目标token数总和
            sample_size *= 0.  # multiply by 0 to preserve device

        # gather logging outputs from all replicas
        if self._sync_stats():  # 收集所有副本的输出记录,适用于分布式训练,非默认情况
            logging_outputs, (sample_size, ooms) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, ooms, ignore=is_dummy_batch,
            )

        try:  # 适用于分布式训练,非默认情况
            # multiply gradients by (# GPUs / sample_size) since DDP
            # already normalizes by the number of GPUs. Thus we get
            # (sum_of_gradients / sample_size).
            if not self.args.use_bmuf:
                self.optimizer.multiply_grads(
                    self.args.distributed_world_size / sample_size
                )
            elif sample_size > 0:  # BMUF needs to check sample size
                num = self.args.distributed_world_size if self._sync_stats() else 1
                self.optimizer.multiply_grads(num / sample_size)

            # clip grads　# 根据梯度修剪阈值来对当前优化器进行梯度修剪,
            grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)  # 返回所有训练参数的梯度副本总L2范数||g^||

            # check that grad norms are consistent across workers 检查所有workers的梯度范数是否一致
            if not self.args.use_bmuf:  # 适用于分布式训练,非默认情况
                self._check_grad_norms(grad_norm)

            # take an optimization step　运行一个优化步骤
            self.optimizer.step()  # 取出并遍历模型的所有训练参数,float()后,按照该参数的梯度由adam优化器更新其参数
            self.set_num_updates(self.get_num_updates() + 1)  # 将当前更新批次步骤+1,并设置更新后的num_updates;
            # 且在每一个更新批次步骤后更新学习率,并给优化器更新学习率以及更新MetersDict中lr和num_updates值

            # log stats # 记录存储当前批次的输出记录元组的列表,当前遍历批次的所有目标token数总和,所有训练参数的梯度副本总L2范数||g^||
            logging_output = self._reduce_and_log_stats(
                logging_outputs, sample_size, grad_norm,
            ) # 返回有序字典logging_output,其中键值对有,｛loss":当前批次平均目标词对应的－lable_smoothing loss/log(2);
              # "nll_loss":当前批次平均目标词对应的－nll_loss(y_hot损失)/log(2);'sample_size':当前批次目标tokens数｝

            # clear CUDA cache to reduce memory fragmentation 清除CUDA缓存以减少内存碎片,非默认情况
            if (
                self.args.empty_cache_freq > 0
                and (
                    (self.get_num_updates() + self.args.empty_cache_freq - 1)
                    % self.args.empty_cache_freq
                ) == 0
                and torch.cuda.is_available()
                and not self.args.cpu
            ):  # 非默认情况
                torch.cuda.empty_cache()
        except FloatingPointError:
            # re-run the forward and backward pass with hooks attached to print out where it fails
            with NanDetector(self.model):
                self.task.train_step(
                    sample, self.model, self.criterion, self.optimizer, self.get_num_updates(),
                    ignore_grad=False
                )
            raise
        except OverflowError as e:
            logger.info("NOTE: overflow detected, " + str(e))
            self.zero_grad()
            logging_output = None
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._log_oom(e)
                logger.error("OOM during optimization, irrecoverable")
            raise e

        if self.args.fp16:  # 若模型所采用的数据类型为Fp16:
            metrics.log_scalar("loss_scale", self.optimizer.scaler.loss_scale, priority=700, round=0)

        metrics.log_stop_time("train_wall")  # 记录某些事件的持续时间s,从log_stop_time-log_start_time

        return logging_output # 返回有序字典logging_output,其中键值对有,｛loss":当前批次平均目标词对应的－lable_smoothing loss/log(2);
              # "nll_loss":当前批次平均目标词对应的－nll_loss(y_hot损失)/log(2);'sample_size':当前批次目标tokens数｝

    @metrics.aggregate("valid")
    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        if self._dummy_batch == "DUMMY": # 若_dummy_batch为"DUMMY",定义虚假批次为samples
            self._dummy_batch = sample

        with torch.no_grad():  # 在验证过程下不修改模型参数,仅对当前模型参数的效果评估
            self.model.eval()  # 设置所建立的model和criterion为eval模式
            self.criterion.eval()

            sample = self._prepare_sample(sample) # 将sample中的tensor全部移到cuda中
            if sample is None:  # 若sample不为空,则is_dummy_batch = False
                sample = self._prepare_sample(self._dummy_batch)
                is_dummy_batch = True
            else:
                is_dummy_batch = False

            try:
                _loss, sample_size, logging_output = self.task.valid_step( # 与train的区别,不需要优化器更新模型参数,设置当前更新批次步骤数和忽略计算grad标记
                    sample, self.model, self.criterion
                ) # 返回当前验证批次所有目标词对应的lable_smoothing loss,当前批次目标tokens数,以及输出记录元组
                # logging_output = {
                #     'loss': loss.data,  # 当前验证批次所有目标词对应的lable_smoothing loss
                #     'nll_loss': nll_loss.data,  # 当前验证批次所有目标词对应的nll_loss(y_hot损失)
                #     'ntokens': sample['ntokens'],  # 当前验证批次目标tokens数
                #     'nsentences': sample['target'].size(0),  # 当前验证批次目标句子数
                #     'sample_size': sample_size,  # 若sentence_avg标记为true,则sample大小为句子数,否则为tokens数
                # }
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if not raise_oom:
                        logger.warning(
                            "ran out of memory in validation step, retrying batch"
                        )
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None  # free some memory
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, raise_oom=True)
                raise e

            logging_outputs = [logging_output] # 将当前验证批次的输出记录元组,存储当前批次的输出记录元组的列表
            if is_dummy_batch:  # 若当前验证批次为虚拟批次,则忽略当前遍历批次的所有目标token数总和
                sample_size *= 0  # multiply by 0 to preserve device

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1: # 收集所有副本的输出记录,适用于分布式训练,非默认情况
            logging_outputs, (sample_size, ) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, ignore=is_dummy_batch,
            )

        # log validation stats # 记录存储当前验证批次的输出记录元组的列表,当前遍历批次的所有目标token数总和
        logging_output = self._reduce_and_log_stats(logging_outputs, sample_size) # 返回有序字典logging_output,
        # 其中键值对有,｛loss":当前验证批次平均目标词对应的－lable_smoothing loss/log(2);
        # "nll_loss":当前验证批次平均目标词对应的－nll_loss(y_hot损失)/log(2);'sample_size':当前验证批次目标tokens数｝

        return logging_output # 返回有序字典logging_output,其中键值对有,｛loss":当前验证批次平均目标词对应的－lable_smoothing loss/log(2);
        # "nll_loss":当前验证批次平均目标词对应的－nll_loss(y_hot损失)/log(2);'sample_size':当前验证批次目标tokens数｝

    def zero_grad(self):  # 清除所有优化器参数的梯度
        self.optimizer.zero_grad()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch. 在每一轮epoch结束时更新学习率"""
        self.lr_scheduler.step(epoch, val_loss)  # 根据当前轮数以及验证集loss来返回所初始化的学习率调度器(InverseSquareRootSchedule类);并完成了初始化学习率的设置,以及后续更新批次步下学习率的计算
        # prefer updating the LR based on the number of steps 希望根据updates步骤数来更新LR
        return self.lr_step_update()  # 在该轮下反复调用每一次更新批次步骤下更新学习率,并给优化器更新学习率以及更新MetersDict中lr值

    def lr_step_update(self):
        """Update the learning rate after each update. 在每一个更新批次步骤后更新学习率,并给优化器更新学习率以及更新MetersDict中lr值"""
        new_lr = self.lr_scheduler.step_update(self.get_num_updates())  # get_num_updates()用于得到当前更新批次步骤,调用lr_scheduler类的当前更新批次步骤下学习率的计算函数
        metrics.log_scalar("lr", new_lr, weight=0, priority=300)  # 将"lr":当前更新批次步骤对应的学习率,weight,以及优先级,由log_scalar添加到MetersDict中实时记录并更新
        return new_lr  # 返回当前更新批次步骤下的学习率

    def get_lr(self):
        """Get the current learning rate. 获取当前学习率"""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance. 获取（非包装的）模型实例"""
        return self._model

    def get_criterion(self):
        """Get the (non-wrapped) criterion instance.获取（非包装的）criterion实例"""
        return self._criterion

    def get_meter(self, name):
        """[deprecated] Get a specific meter by name."""
        from fairseq import meters

        if 'get_meter' not in self._warn_once:
            self._warn_once.add('get_meter')
            utils.deprecation_warning(
                'Trainer.get_meter is deprecated. Please use fairseq.metrics instead.'
            )

        train_meters = metrics.get_meters("train")
        if train_meters is None:
            train_meters = {}

        if name == "train_loss" and "loss" in train_meters:
            return train_meters["loss"]
        elif name == "train_nll_loss":
            # support for legacy train.py, which assumed this meter is
            # always initialized
            m = train_meters.get("nll_loss", None)
            return m or meters.AverageMeter()
        elif name == "wall":
            # support for legacy train.py, which assumed this meter is
            # always initialized
            m = metrics.get_meter("default", "wall")
            return m or meters.TimeMeter()
        elif name == "wps":
            m = metrics.get_meter("train", "wps")
            return m or meters.TimeMeter()
        elif name in {"valid_loss", "valid_nll_loss"}:
            # support for legacy train.py, which assumed these meters
            # are always initialized
            k = name[len("valid_"):]
            m = metrics.get_meter("valid", k)
            return m or meters.AverageMeter()
        elif name == "oom":
            return meters.AverageMeter()
        elif name in train_meters:
            return train_meters[name]
        return None

    def get_num_updates(self):
        """Get the number of parameters updates. 得到当前更新批次步骤"""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates. 更新所传入的当前更新批次步骤(即+1)和学习率,并给优化器更新学习率以及更新MetersDict中lr和num_updates值"""
        self._num_updates = num_updates  # 更新所传入的当前更新批次步骤(即+1)
        self.lr_step_update() # 在每一个更新批次步骤后更新学习率,并给优化器更新学习率以及更新MetersDict中lr值
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200) # 将"num_updates":当前更新批次步骤数,weight,以及优先级,由log_scalar添加到MetersDict中实时记录并更新

    def _prepare_sample(self, sample):
        if sample == "DUMMY":
            raise Exception(
                "Trying to use an uninitialized 'dummy' batch. This usually indicates "
                "that the total number of batches is smaller than the number of "
                "participating GPUs. Try reducing the batch size or using fewer GPUs."
            )

        if sample is None or len(sample) == 0:  # 若当前batch为空:
            return None

        if self.cuda:  # 若存在可用gpu,则将sample中的tensor全部移到cuda中
            sample = utils.move_to_cuda(sample)

        def apply_half(t):
            if t.dtype is torch.float32:
                return t.half()
            return t

        if self.args.fp16: # 若模型所采用的数据类型为fp16,则将全部float32tensor全部转成float16
            sample = utils.apply_to_sample(apply_half, sample)

        return sample # 返回将tensor全部移到cuda中的sample(若设置fp16,则修改数据类型)

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints # 根据args.seed和更新批次数设置种子,以便从检查点恢复时可得到可复现的结果
        seed = self.args.seed + self.get_num_updates()  # 得到当前随机数生成器种子
        torch.manual_seed(seed)  # 在cpu和gpu下均固定种子数
        if self.cuda:
            torch.cuda.manual_seed(seed)

    def _sync_stats(self):
        # Return True if it's using multiple GPUs and DDP or multiple GPUs with
        # BMUF and it's a bmuf sync with warmup iterations completed before.
        return self.args.distributed_world_size > 1 and (
            (not self.args.use_bmuf)
            or (
                self.args.use_bmuf
                and (self.get_num_updates() + 1) % self.args.global_sync_iter == 0
                and (self.get_num_updates() + 1) > self.args.warmup_iterations
            )
        )

    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        logger.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()

    def _aggregate_logging_outputs(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        if self.task.__class__.logging_outputs_can_be_summed(self.get_criterion()):
            return self._fast_stat_sync_sum(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )
        else:
            return self._all_gather_list_sync(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )

    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        if ignore:
            logging_outputs = []
        results = list(zip(
            *distributed_utils.all_gather_list(
                [logging_outputs] + list(extra_stats_to_sum),
                max_size=getattr(self.args, 'all_gather_list_size', 16384),
            )
        ))
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _fast_stat_sync_sum(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. fast_stat_sync_sum is
        faster than all_gather_list_sync, but is only suitable when
        logging outputs are scalars and can be summed. Note that
        *logging_outputs* cannot contain any nested dicts/lists.
        """
        data = {}
        for i, stat in enumerate(extra_stats_to_sum):
            data['extra_stats_' + str(i)] = stat
        if len(logging_outputs) > 0:
            log_keys = list(logging_outputs[0].keys())
            for k in log_keys:
                if not ignore:
                    v = sum(log[k] for log in logging_outputs if k in log)
                else:
                    v = logging_outputs[0][k]
                    v = torch.zeros_like(v) if torch.is_tensor(v) else 0
                data['logging_outputs_' + k] = v
        else:
            log_keys = None

        data = distributed_utils.all_reduce_dict(
            data,
            device=self.device,
        )

        extra_stats_to_sum = [
            data['extra_stats_' + str(i)] for i in range(len(extra_stats_to_sum))
        ]
        if log_keys is not None:
            logging_outputs = [{k: data['logging_outputs_' + k] for k in log_keys}]
        else:
            logging_outputs = []
        return logging_outputs, extra_stats_to_sum

    def _check_grad_norms(self, grad_norm):
        """Check that grad norms are consistent across workers."""
        if self._grad_norm_buf is not None: # 适用于分布式训练,非默认情况
            self._grad_norm_buf.zero_()
            self._grad_norm_buf[self.args.distributed_rank] = grad_norm
            distributed_utils.all_reduce(self._grad_norm_buf)
            if not (self._grad_norm_buf == self._grad_norm_buf[0]).all():
                raise RuntimeError(
                    "Fatal error: gradients are inconsistent between workers. "
                    "Try --ddp-backend=no_c10d."
                )

    def _reduce_and_log_stats(self, logging_outputs, sample_size, grad_norm=None): # 记录存储当前批次的输出记录元组的列表,当前遍历批次的所有目标token数总和,当前批次所有训练参数的梯度副本总L2范数||g^||
        if grad_norm is not None:  # 若当前批次所有训练参数的梯度副本总L2范数||g^||不为None:
            metrics.log_speed("ups", 1., priority=100, round=2)
            metrics.log_scalar("gnorm", grad_norm, priority=400, round=3) #　将"gnorm":当前批次所有训练参数的梯度副本总L2范数||g^||,以及优先级,由log_scalar添加到MetersDict中实时记录并更新
            if self.args.clip_norm > 0:  # 若梯度修剪阈值>0:
                metrics.log_scalar(
                    "clip",
                    torch.where(
                        grad_norm > self.args.clip_norm,
                        grad_norm.new_tensor(100),
                        grad_norm.new_tensor(0),
                    ),
                    priority=500,
                    round=1,
                ) #　将"clip":经过梯度修剪的当前批次所有训练参数的梯度副本总L2范数||g^||,以及优先级,由log_scalar添加到MetersDict中实时记录并更新

        with metrics.aggregate() as agg:     # 在给定名称下整合指标的上下文管理器(一个MetersDict())
            if logging_outputs is not None:  # 若存储当前批次的输出记录元组的列表不为None:
                # 通过存储当前批次的输出记录元组的列表,和当前LabelSmoothedCrossEntropyCriterion类记录
                self.task.reduce_metrics(logging_outputs, self.get_criterion())
                # 调用父类FairseqTask方法,将如下输出值以及优先级,由log_scalar添加到MetersDict中实时记录并更新
                # "wpb":当前遍历批次的所有目标token数总和; "bsz":当前遍历批次的所有目标句子数;"loss":当前批次平均目标词对应的－lable_smoothing loss/log(2);
                # "nll_loss":当前批次平均目标词对应的－nll_loss(y_hot损失)/log(2);"ppl":2^(－nll_loss)模型的ppl

            # support legacy interface # 支持旧版接口
            logging_output = agg.get_smoothed_values() # 返回MetersDict中非"_"开头的key,以及其smoothed_value所组成的有序字典logging_output
            logging_output["sample_size"] = sample_size # 如:OrderedDict([('loss', 13.988), ('nll_loss', 13.982), ('ppl', 16175.65),
                                                        # ('wps', 0.0), ('wpb', 2931.0), ('bsz', 192.0), ('sample_size', 2931.0)])

            for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:  # 去除logging_output中的一些键值对
                if key_to_delete in logging_output:
                    del logging_output[key_to_delete]
            return logging_output
            # 返回有序字典logging_output,其中键值对有,｛loss":当前批次平均目标词对应的－lable_smoothing loss/log(2);
            # "nll_loss":当前批次平均目标词对应的－nll_loss(y_hot损失)/log(2);'sample_size':当前批次目标tokens数｝
