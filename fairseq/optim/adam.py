# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import types

import torch
import torch.optim
import torch.distributed as dist

from fairseq.optim import FairseqOptimizer, register_optimizer
from fairseq.optim.fused_adam import get_fused_adam_class

logger = logging.getLogger(__name__)


@register_optimizer('adam')
class FairseqAdam(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.　
    该优化器的weight decay行为与Adam的“AdamW”变体相对应,因此它与torch.optim.AdamW非常相似
    """

    def __init__(self, args, params):
        super().__init__(args)
        fused_adam_cls = get_fused_adam_class()  # 与fused_adam相关的参数
        use_fused_adam = (
            not getattr(args, 'use_old_adam', False)
            and fused_adam_cls is not None
            and torch.cuda.is_available()
        )
        if use_fused_adam:
            logger.info('using FusedAdam')
            self._optimizer = fused_adam_cls(params, **self.optimizer_config)
        else:  # 默认情况,self.optimizer_config为(lr,betas,eps,weight_decay)的字典,帮助初始化adam类
            self._optimizer = Adam(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')    # Adam优化器的beta1,beta2
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')  # Adam优化器的小数值e
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')  # 权重衰减值
        # Maintain backward compatibility with old checkpoints that have stored
        # optimizer state as fairseq.optim.adam.Adam. # 与旧检查点保持后向兼容性,这些检查点将优化器状态存储为fairseq.optim.adam.Adam
        parser.add_argument(
            "--use-old-adam",
            action='store_true',
            default=False,
            help="Use fairseq.optim.adam.Adam",
        )
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        返回一个kwarg字典,其用于覆盖存储在检查点中的优化器的args
        这使我们能够加载检查点并使用一组不同的优化器参数（例如,具有不同的学习率）来恢复训练
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
        }

    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)


class Adam(torch.optim.Optimizer):
    """Implements Adam algorithm.
　　　
    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)　
    # 该Adam algorithm的实现是根据以下内容从torch.optim.Adam修改的： `Fixed Weight Decay Regularization in Adam`

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups　　# 用于优化的可迭代的参数,这里为所有model,criterion中所有需要计算梯度的参数
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0) # 权重衰减(即L2正则惩罚项)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_ # 是否使用AMSGrad变体

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step. 运行一个优化步骤

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. 重新评估模型并返回损失的闭包
        """
        loss = None  # loss不起作用,
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:  # 取出并遍历模型的所有训练参数,float()后,按照该参数的梯度由adam优化器更新其参数
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                # TODO: remove check once pyTorch avoids a copy for this case
                if p.data_ptr() != p_data_fp32.data_ptr():
                    p.data.copy_(p_data_fp32)

        return loss
