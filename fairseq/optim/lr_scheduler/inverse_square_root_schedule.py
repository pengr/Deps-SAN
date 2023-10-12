# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('inverse_sqrt')
class InverseSquareRootSchedule(FairseqLRScheduler):
    """Decay the LR based on the inverse square root of the update number.　# 根据更新批次数的平方根倒数来衰减LR

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    # 预热阶段中,学习率从某个初始学习率（``--warmup-init-lr''）线性增加到所指定的学习率（``--lr''）;
    # 预热阶段后,按更新批次数呈正比衰减(更新批次数越大,lr越小),并且衰减因子设置为与所指定的学习率一致

    During warmup::  # 预热阶段中

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates) # 按线性将[warmup_init_lr,lr]区间平分warmup_updates次
      lr = lrs[update_num]  # 根据当前更新批次数来选择当前的lr

    After warmup::   # 预热阶段后

      decay_factor = args.lr * sqrt(args.warmup_updates)  # 衰减因子 -> 所指定的学习率*预热步数的开方)
      lr = decay_factor / sqrt(update_num)                # 当前更新批次数的lr -> 衰减因子/当前更新批次数的开方
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)  # 调用父类加载args和所传入的optimizer,以及self.best
        if len(args.lr) > 1:  # 若args.lr为固定的学习率列表,如[0.001,0.002,...]:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        warmup_end_lr = args.lr[0] # 取出所指定的args.lr,也即预热阶段最后的lr
        if args.warmup_init_lr < 0: # 若预热阶段初始lr<0,则若存在预热步骤,则将其定义为0
            args.warmup_init_lr = 0 if args.warmup_updates > 0 else warmup_end_lr

        # linearly warmup for the first args.warmup_updates 平分[warmup_init_lr,lr]warmup_updates次,每个区间的lr步长,用于前warmup_step的线性预热(预热阶段中)
        self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates

        # then, decay prop. to the inverse square root of the update number  衰减因子,用于按更新批次步骤衰减lr(预热阶段后)
        self.decay_factor = warmup_end_lr * args.warmup_updates**0.5

        # initial learning rate 设置初始学习率warmup_init_lr(预热阶段中)
        self.lr = args.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')  # 在前N个updates中线性预热学习率,等价于warmup_steps
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr') # 预热阶段的初始学习率;默认为args.lr
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch. 在给定一轮结束时更新学习率"""
        super().step(epoch, val_loss) # 若学习率调度器中无best,什么也不做
        # we don't change the learning rate at epoch boundaries 我们不会在时代边界改变学习率
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update. 每次update后更新学习率,并给优化器更新学习率"""
        if num_updates < self.args.warmup_updates:  # 若当前更新批次步骤未达到所设定的预热步骤数(预热阶段中)
            self.lr = self.args.warmup_init_lr + num_updates*self.lr_step  # 由warmup的初始学习率+当前更新批次步骤*每个区间的lr步长
        else:                                       # 若当前更新批次步骤达到所设定的预热步骤数(预热阶段后)
            self.lr = self.decay_factor * num_updates**-0.5   # 按更新批次步骤衰减lr,即衰减因子/(当前更新批次步骤的开方)
        self.optimizer.set_lr(self.lr)  # 每个更新批次步骤给优化器设置学习率
        return self.lr     # 返回当前更新批次步骤给优化器设置的学习率
