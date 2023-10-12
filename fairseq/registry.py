# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse


REGISTRIES = {}


def setup_registry(
    registry_name: str,
    base_class=None,
    default=None,
):
    assert registry_name.startswith('--')
    registry_name = registry_name[2:].replace('-', '_')

    REGISTRY = {}
    REGISTRY_CLASS_NAMES = set()

    # maintain a registry of all registries
    if registry_name in REGISTRIES:
        return  # registry already exists
    REGISTRIES[registry_name] = {
        'registry': REGISTRY,
        'default': default,
    }

    def build_x(args, *extra_args, **extra_kwargs):
        choice = getattr(args, registry_name, None)  #　根据当前registry_name('criterion','optimizer','lr_scheduler'),从args中取出对应的criterion->'label_smoothed_cross_entropy',optimizer->'adam',lr_scheduler->'inverse_sqrt'
        if choice is None:
            return None
        cls = REGISTRY[choice]  # 根据当前choice(即label_smoothed_cross_entropy,adam,inverse_sqrt),从REGISTRY中选出对应的类<LabelSmoothedCrossEntropyCriterion>,<FairseqAdam>,<InverseSquareRootSchedule>
        if hasattr(cls, 'build_' + registry_name): # 若所选出的类LabelSmoothedCrossEntropyCriterion,存在build_criterion属性:
            builder = getattr(cls, 'build_' + registry_name)  # 则从该类中取出该属性赋给builder
        else:
            builder = cls                                     # 否则直接定义builder为该类,如<FairseqAdam>,<InverseSquareRootSchedule>
        set_defaults(args, cls)  # 基于*add_args*设置默认参数的helper,帮助当前类的参数添加到args中(若args无该参数)
        return builder(args, *extra_args, **extra_kwargs) # 当前类无build_criterion(),调用父类FairseqCriterion的函数,由criterion对应类的init_args完成对criterion对应类的初始化
                                                          # 由于optimizer对应的builder为FairseqAdam类本身,由FairseqAdam类本身的init_args完成对FairseqAdam类本身的初始化
                                                          # 由于lr_scheduler对应的builder为InverseSquareRootSchedule类本身,由InverseSquareRootSchedule类本身的init_args完成对InverseSquareRootSchedule类本身的初始化
    def register_x(name):

        def register_x_cls(cls):
            if name in REGISTRY:
                raise ValueError('Cannot register duplicate {} ({})'.format(registry_name, name))
            if cls.__name__ in REGISTRY_CLASS_NAMES:
                raise ValueError(
                    'Cannot register {} with duplicate class name ({})'.format(
                        registry_name, cls.__name__,
                    )
                )
            if base_class is not None and not issubclass(cls, base_class):
                raise ValueError('{} must extend {}'.format(cls.__name__, base_class.__name__))
            REGISTRY[name] = cls
            REGISTRY_CLASS_NAMES.add(cls.__name__)
            return cls

        return register_x_cls

    return build_x, register_x, REGISTRY


def set_defaults(args, cls):
    """Helper to set default arguments based on *add_args*. 基于*add_args*设置默认参数的helper"""
    if not hasattr(cls, 'add_args'):  # 若当前类LabelSmoothedCrossEntropyCriterion不存在add_args参数,则直接返回
        return
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, allow_abbrev=False)
    cls.add_args(parser)  # 将当前类的参数添加到仅在此函数定义的parser内
    # copied from argparse.py: 从argparse.py复制,下面操作将当前类的参数'label_smoothing'加入到args中
    defaults = argparse.Namespace()
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            if not hasattr(defaults, action.dest):
                if action.default is not argparse.SUPPRESS:
                    setattr(defaults, action.dest, action.default)
    for key, default_value in vars(defaults).items():
        if not hasattr(args, key):  # 若args不存在当前类的参数'label_smoothing',则将该参数以及其默认值一起添加进args内
            setattr(args, key, default_value)
