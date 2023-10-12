# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .. import FairseqOptimizer


class FairseqLRScheduler(object):

    def __init__(self, args, optimizer):
        super().__init__()
        if not isinstance(optimizer, FairseqOptimizer): # 所传入的optimizer类型必须为FairseqOptimizer
            raise ValueError('optimizer must be an instance of FairseqOptimizer')
        self.args = args   # 加载args和所传入的optimizer,以及self.best
        self.optimizer = optimizer
        self.best = None

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        pass

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {'best': self.best}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.best = state_dict['best']

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch. 在给定一轮结束时更新学习率"""
        if val_loss is not None: # 若传入的验证集loss不为空,而学习率调度器中无best,则返回当前传入的验证集loss
            if self.best is None:
                self.best = val_loss
            else:
                self.best = min(self.best, val_loss)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.optimizer.get_lr()
