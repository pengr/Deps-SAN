# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq import utils


class FairseqOptimizer(object):

    def __init__(self, args):  # 加载所有的args
        super().__init__()
        self.args = args

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        pass

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, '_optimizer'):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError('_optimizer must be an instance of torch.optim.Optimizer')
        return self._optimizer

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        raise NotImplementedError

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                yield p

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def get_lr(self):
        """Return the current learning rate. 返回当前的学习率"""
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        """Set the learning rate. 给优化器设置学习率(若采用预热,则初始学习率为args.warmup_init_lr,而不是args.lr)"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.optimizer.param_groups:
                group.update(optimizer_overrides)

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves. """
        # 调用torch.autograd.backward,根据链式法则自动计算出计算图叶子节点,给定张量w.r.t.的梯度总和
        loss.backward()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm.　根据max_norm(梯度修剪阈值)来对全部参数进行梯度修剪"""
        return utils.clip_grad_norm_(self.params, max_norm)

    def step(self, closure=None):
        """Performs a single optimization step. 运行一个优化步骤"""
        self.optimizer.step(closure)  # 取出并遍历模型的所有训练参数,float()后,按照该参数的梯度由adam优化器更新其参数

    def zero_grad(self):
        """Clears the gradients of all optimized parameters. 清除所有优化器参数的梯度"""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()

    @property
    def supports_memory_efficient_fp16(self):
        if hasattr(self.optimizer, 'supports_memory_efficient_fp16'):
            return self.optimizer.supports_memory_efficient_fp16
        return False

    @property
    def supports_flat_params(self):
        """
        Whether the optimizer supports collapsing of the model
        parameters/gradients into a single contiguous Tensor.
        """
        if hasattr(self.optimizer, 'supports_flat_params'):
            return self.optimizer.supports_flat_params
        return False

    def average_params(self):
        pass
