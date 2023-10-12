# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper around various loggers and progress bars (e.g., tqdm).
"""

import atexit
import json
import logging
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager
from numbers import Number
from typing import Optional

import torch

from .meters import AverageMeter, StopwatchMeter, TimeMeter


logger = logging.getLogger(__name__)


def progress_bar(
    iterator,
    log_format: Optional[str] = None,
    log_interval: int = 100,
    epoch: Optional[int] = None,
    prefix: Optional[str] = None,
    tensorboard_logdir: Optional[str] = None,
    default_log_format: str = 'tqdm',
):
    if log_format is None:  # 若未设置记录格式,则采用默认进度打印格式'tqdm'
        log_format = default_log_format
    if log_format == 'tqdm' and not sys.stderr.isatty(): # 测试时,由于采用'tqdm'且直接打印到窗口,所以进度打印格式为'simple'
        log_format = 'simple'

    # 根据记录格式,创建不同的bas类
    if log_format == 'json':
        # 训练/验证时,返回以JSON格式记录输出JsonProgressBar类,加载分好块的迭代器GroupedIterator类,log_interval,当前epoch编号,offset等
        bar = JsonProgressBar(iterator, epoch, prefix, log_interval)
    elif log_format == 'none':
        bar = NoopProgressBar(iterator, epoch, prefix)
    elif log_format == 'simple':
        # 测试时,返回以Simple格式记录输出SimpleProgressBar类,加载self._cur_epoch_itr可记录迭代数的迭代器,log_interval,epoch和prefix=None等
        bar = SimpleProgressBar(iterator, epoch, prefix, log_interval)
    elif log_format == 'tqdm':
        bar = TqdmProgressBar(iterator, epoch, prefix)
    else:
        raise ValueError('Unknown log format: {}'.format(log_format))

    if tensorboard_logdir: # 若用tensorboard保存日志的路径不为空;非默认情况
        try:
            # [FB only] custom wrapper for TensorBoard
            import palaas  # noqa
            from .fb_tbmf_wrapper import FbTbmfWrapper
            bar = FbTbmfWrapper(bar, log_interval)
        except ImportError:
            bar = TensorboardProgressBarWrapper(bar, tensorboard_logdir)

    return bar  # 训练/验证时,返回以JSON格式记录输出JsonProgressBar类,加载分好块的迭代器GroupedIterator类,log_interval,当前epoch编号,offset等
                # 测试时,返回以Simple格式记录输出SimpleProgressBar类,加载self._cur_epoch_itr可记录迭代数的迭代器,log_interval,epoch和prefix=None,offset等

def build_progress_bar(
    args,
    iterator,
    epoch: Optional[int] = None,
    prefix: Optional[str] = None,
    default: str = 'tqdm',
    no_progress_bar: str = 'none',
):
    """Legacy wrapper that takes an argparse.Namespace."""
    if getattr(args, 'no_progress_bar', False):
        default = no_progress_bar
    if getattr(args, 'distributed_rank', 0) == 0:
        tensorboard_logdir = getattr(args, 'tensorboard_logdir', None)
    else:
        tensorboard_logdir = None
    return progress_bar(
        iterator,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch,
        prefix=prefix,
        tensorboard_logdir=tensorboard_logdir,
        default_log_format=default,
    )


def format_stat(stat):  # 根据stat自身格式,将stat字符串化
    if isinstance(stat, Number):
        stat = '{:g}'.format(stat)
    elif isinstance(stat, AverageMeter):
        stat = '{:.3f}'.format(stat.avg)
    elif isinstance(stat, TimeMeter):
        stat = '{:g}'.format(round(stat.avg))
    elif isinstance(stat, StopwatchMeter):
        stat = '{:g}'.format(round(stat.sum))
    elif torch.is_tensor(stat):
        stat = stat.tolist()
    return stat


class BaseProgressBar(object):
    """Abstract class for progress bars. 进度条的抽象类"""
    def __init__(self, iterable, epoch=None, prefix=None):
        self.iterable = iterable  # 加载分好块的迭代器GroupedIterator类,基于self._cur_epoch_itr
        self.offset = getattr(iterable, 'offset', 0)  # 从迭代器中取出offset
        self.epoch = epoch  # 获取当前epoch编号
        self.prefix = ''    # 定义进度条epoch的前缀
        if epoch is not None:  # 若当前epoch编号不为None,则定义前缀如'epoch 001'等
            self.prefix += 'epoch {:03d}'.format(epoch)
        if prefix is not None: # 若给定有进度条epoch的前缀,则使用该前缀,如 ' | perfix epoch 001'
            self.prefix += ' | {}'.format(prefix)

    def __len__(self):
        return len(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        raise NotImplementedError

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        raise NotImplementedError

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        raise NotImplementedError

    def _str_commas(self, stats):
        return ', '.join(key + '=' + stats[key].strip()
                         for key in stats.keys())

    def _str_pipes(self, stats):
        return ' | '.join(key + ' ' + stats[key].strip()
                          for key in stats.keys())

    def _format_stats(self, stats):
        postfix = OrderedDict(stats)
        # Preprocess stats according to datatype
        for key in postfix.keys():
            postfix[key] = str(format_stat(postfix[key]))
        return postfix


@contextmanager
def rename_logger(logger, new_name):
    old_name = logger.name
    if new_name is not None:
        logger.name = new_name
    yield logger
    logger.name = old_name


class JsonProgressBar(BaseProgressBar):
    """Log output in JSON format. 以JSON格式记录输出"""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000):
        super().__init__(iterable, epoch, prefix) # 调用父类BaseProgressBar,加载好迭代器,offset,当前epoch编号,进度条epoch的前缀
        self.log_interval = log_interval  # 获取记录间隔,类似disp_freq
        self.i = None  # 初始化遍历迭代器的当前索引号
        self.size = None  # 初始化迭代器长度(即数据集句子对总数)

    def __iter__(self):  # 在progress.bar被调用后,自动调用该函数
        self.size = len(self.iterable)  # 获取迭代器长度
        for i, obj in enumerate(self.iterable, start=self.offset):  # 遍历迭代器中所有元素(下标/索引号从offset开始)
            self.i = i  # 更新遍历迭代器的当前索引号
            yield obj  # 不断地返回迭代器中的元素(即数据集中的句子idx编号)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval. 根据log_interval记录中间状态"""
        step = step or self.i or 0  # step为传入的当前更新批次步骤数,self.i为遍历到全部数据批次的第几个(self.i=step)
        if (
            step > 0  # 当前更新批次步骤数,以及进度打印频率不为空,且当前更新批次步骤可以整除进度打印频率:
            and self.log_interval is not None
            and step % self.log_interval == 0
        ):
            update = (
                self.epoch - 1 + (self.i + 1) / float(self.size)
                if self.epoch is not None
                else None
            )  # 若当前轮数不为None,(i + 1) / float(size)为完成了全部数据批次的百分之多少,update为所完成第几轮的百分之几批次(四舍五入到小数点后3位)
            stats = self._format_stats(stats, epoch=self.epoch, update=update)
            # 将'train_inner'下所汇聚的平滑值,以及'epoch':当前epoch和'update':所完成第几轮的百分之几批次添加stats中(OrderDict()类型)
            with rename_logger(logger, tag):  # 打印出stats中全部键值对
                logger.info(json.dumps(stats))

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats. 打印一轮完后的统计(tag:训练,验证)"""
        self.stats = stats
        if tag is not None: # 若当前为训练,验证的标签:
            self.stats = OrderedDict([(tag + '_' + k, v) for k, v in self.stats.items()]) # 给传入的stats,即smoothed_value所组成的有序字典中全部键加上tag
        stats = self._format_stats(self.stats, epoch=self.epoch)
        # 将'train(训练完)/root(验证完)'下所汇聚的平滑值,以及'epoch':当前epoch, (训练完特有)'update':所完成第几轮的百分之几批次添加进一个OrderDict()
        with rename_logger(logger, tag): # 打印出stats中全部键值对
            logger.info(json.dumps(stats))

    def _format_stats(self, stats, epoch=None, update=None):
        postfix = OrderedDict()
        if epoch is not None:   # 将当前epoch和所完成第几轮的百分之几批次添加进有序字典中
            postfix['epoch'] = epoch
        if update is not None:
            postfix['update'] = round(update, 3)
        # Preprocess stats according to datatype　# format_stat:根据stat自身格式,将stat字符串化
        for key in stats.keys():
            postfix[key] = format_stat(stats[key])  # 将'train_inner'下所汇聚的平滑值,以及'epoch':当前epoch和'update':所完成第几轮的百分之几批次添加进一个OrderDict()
        return postfix


class NoopProgressBar(BaseProgressBar):
    """No logging."""

    def __init__(self, iterable, epoch=None, prefix=None):
        super().__init__(iterable, epoch, prefix)

    def __iter__(self):
        for obj in self.iterable:
            yield obj

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        pass

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        pass


class SimpleProgressBar(BaseProgressBar):
    """A minimal logger for non-TTY environments. 非TTY环境的最小记录器"""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000):
        super().__init__(iterable, epoch, prefix) # 调用父类BaseProgressBar,加载好迭代器,offset,当前epoch编号,进度条epoch的前缀
        self.log_interval = log_interval # 获取记录间隔,类似disp_freq
        self.i = None     # 初始化遍历迭代器的当前索引号
        self.size = None  # 初始化迭代器长度(即数据集句子对总数)

    def __iter__(self): # 在progress.bar被调用后,自动调用该函数
        self.size = len(self.iterable) # 获取迭代器长度
        for i, obj in enumerate(self.iterable, start=self.offset):  # 遍历迭代器中所有元素(下标/索引号从offset开始)
            self.i = i  # 更新遍历迭代器的当前索引号
            yield obj  # 不断地返回迭代器中的元素(即数据集中的句子idx编号)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        step = step or self.i or 0
        if (
            step > 0
            and self.log_interval is not None
            and step % self.log_interval == 0
        ):
            stats = self._format_stats(stats)
            postfix = self._str_commas(stats)
            with rename_logger(logger, tag):
                logger.info(
                    '{}:  {:5d} / {:d} {}'
                    .format(self.prefix, self.i + 1, self.size, postfix)
                )

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        with rename_logger(logger, tag):
            logger.info('{} | {}'.format(self.prefix, postfix))


class TqdmProgressBar(BaseProgressBar):
    """Log to tqdm."""

    def __init__(self, iterable, epoch=None, prefix=None):
        super().__init__(iterable, epoch, prefix)
        from tqdm import tqdm
        self.tqdm = tqdm(iterable, self.prefix, leave=False)

    def __iter__(self):
        return iter(self.tqdm)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats according to log_interval."""
        self.tqdm.set_postfix(self._format_stats(stats), refresh=False)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        self.tqdm.write('{} | {}'.format(self.tqdm.desc, postfix))


try:
    _tensorboard_writers = {}
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def _close_writers():
    for w in _tensorboard_writers.values():
        w.close()


atexit.register(_close_writers)


class TensorboardProgressBarWrapper(BaseProgressBar):
    """Log to tensorboard."""

    def __init__(self, wrapped_bar, tensorboard_logdir):
        self.wrapped_bar = wrapped_bar
        self.tensorboard_logdir = tensorboard_logdir

        if SummaryWriter is None:
            logger.warning(
                "tensorboard or required dependencies not found, please see README "
                "for using tensorboard. (e.g. pip install tensorboardX)"
            )

    def _writer(self, key):
        if SummaryWriter is None:
            return None
        _writers = _tensorboard_writers
        if key not in _writers:
            _writers[key] = SummaryWriter(os.path.join(self.tensorboard_logdir, key))
            _writers[key].add_text('sys.argv', " ".join(sys.argv))
        return _writers[key]

    def __iter__(self):
        return iter(self.wrapped_bar)

    def log(self, stats, tag=None, step=None):
        """Log intermediate stats to tensorboard."""
        self._log_to_tensorboard(stats, tag, step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag=None, step=None):
        """Print end-of-epoch stats."""
        self._log_to_tensorboard(stats, tag, step)
        self.wrapped_bar.print(stats, tag=tag, step=step)

    def _log_to_tensorboard(self, stats, tag=None, step=None):
        writer = self._writer(tag or '')
        if writer is None:
            return
        if step is None:
            step = stats['num_updates']
        for key in stats.keys() - {'num_updates'}:
            if isinstance(stats[key], AverageMeter):
                writer.add_scalar(key, stats[key].val, step)
            elif isinstance(stats[key], Number):
                writer.add_scalar(key, stats[key], step)
        writer.flush()
