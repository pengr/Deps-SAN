# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from typing import Callable, List, Optional

import torch

from fairseq import utils
from fairseq.data.indexed_dataset import get_available_dataset_impl


def get_preprocessing_parser(default_task="translation"):
    parser = get_parser("Preprocessing", default_task)  # choose different parser by default_task
    add_preprocess_args(parser)  # actual preprocessing parameters
    return parser


def get_training_parser(default_task="translation"):
    parser = get_parser("Trainer", default_task)  # choose different parser by default_task,与preprocess一样的通用设置
    add_dataset_args(parser, train=True)   # actual dataset and data loading parameters,根据训练标记(train)/测试(gen)标记决定独有部分
    add_distributed_training_args(parser)  # actual distributed_training parameters,未采用多GPU则全部为默认
    add_model_args(parser)                 # actual model parameters, 返回带有所有models类的字典
    add_optimization_args(parser)          # actual optimization parameters
    add_checkpoint_args(parser)
    return parser


def get_generation_parser(interactive=False, default_task="translation"):
    parser = get_parser("Generation", default_task) # choose different parser by default_task,与train一样的通用设置
    add_dataset_args(parser, gen=True) # actual dataset and data loading parameters,根据训练标记(train)/测试(gen)标记决定独有部分
    add_generation_args(parser) # actual generation parameters
    if interactive: # 若调用interactive.py则为True,额外添加actual interactive parameters
        add_interactive_args(parser)
    return parser


def get_interactive_generation_parser(default_task="translation"):
    return get_generation_parser(interactive=True, default_task=default_task)


def get_eval_lm_parser(default_task="language_modeling"):
    parser = get_parser("Evaluate Language Model", default_task)
    add_dataset_args(parser, gen=True)
    add_eval_lm_args(parser)
    return parser


def get_validation_parser(default_task=None):
    parser = get_parser("Validation", default_task)
    add_dataset_args(parser, train=True)
    group = parser.add_argument_group("Evaluation")
    add_common_eval_args(group)
    return parser


def eval_str_list(x, type=float):  # 若x为str,去除字符回归其原有类型,并将其本身和类型组成list
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def eval_bool(x, default=False): # 用于将非None的x,转换成bool类型;None类型的x,则返回default
    if x is None:
        return default
    try:
        return bool(eval(x))
    except TypeError:
        return default


def parse_args_and_arch(
    parser: argparse.ArgumentParser,
    input_args: List[str] = None,
    parse_known: bool = False,
    suppress_defaults: bool = False,
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None,
):
    """
    Args:
        parser (ArgumentParser): the parser                             # 传入的parser
        input_args (List[str]): strings to parse, defaults to sys.argv  # parse字符串;初始为[],默认设置为sys.argv
        parse_known (bool): only parse known arguments, similar to
            `ArgumentParser.parse_known_args`                           # 仅解析已知参数;默认为False
        suppress_defaults (bool): parse while ignoring all default      # parses时忽略所有默认值;默认为False
        modify_parser (Optional[Callable[[ArgumentParser], None]]):
            function to modify the parser, e.g., to set default values  # 修改parser的函数,例如设置默认值;默认为None
    """
    if suppress_defaults:  # parses时忽略所有默认值;默认为False
        # Parse args without any default values. This requires us to parse
        # twice, once to identify all the necessary task/model args, and a second
        # time with all defaults set to None.
        # 不带任何默认值来解析args. 这要求我们解析两次,第一次识别所有必要的task/model参数,第二次将所有默认值设置为None
        args = parse_args_and_arch( # 第一次识别所有必要的task/model参数
            parser,
            input_args=input_args,
            parse_known=parse_known,
            suppress_defaults=False,
        )
        # 第二次将所有默认值设置为None
        suppressed_parser = argparse.ArgumentParser(add_help=False, parents=[parser])
        suppressed_parser.set_defaults(**{k: None for k, v in vars(args).items()})
        args = suppressed_parser.parse_args(input_args)
        return argparse.Namespace(
            **{k: v for k, v in vars(args).items() if v is not None} # 返回不带任何默认值所解析的args
        )

    # 这里重新导入ARCH_MODEL_REGISTRY作为一个字典,自动import在(models)/ directory下的所有Python文件,
    # default:　fconv, 其他作为可选项,<我们方法>,需要关注<class 'fairseq.models.transformer.TransformerModel'>
    # 并且ARCH_CONFIG_REGISTRY与ARCH_MODEL_REGISTRY一模一样
    from fairseq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY

    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    # 在创建真正的解析器之前,我们需要导入可选的用户模块, 以导入自定义tasks, optimizers, architectures等
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)     # use to specify a custom location for additional modules(not parameters)
    usr_args, _ = usr_parser.parse_known_args(input_args)
    utils.import_user_module(usr_args)   # load additional custom modules if "--user-dir" isn't None

    if modify_parser is not None:  # 修改parser的函数,例如设置默认值;默认为None
        modify_parser(parser)

    # The parser doesn't know about model/criterion/optimizer-specific args, so
    # we parse twice. First we parse the model/criterion/optimizer, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    # 该parser不知道关于model/criterion/optimizer-specific args,所以需要解析两次.
    # 第一次我们解析model/criterion/optimizer,添加*-specific参数后解析第二次;
    # 若给定有input_args,则用其代替sys.argv;否则将全部的arguments设置添加进args(Namespace类型)
    args, _ = parser.parse_known_args(input_args)

    # Add model-specific args to parser. 添加model-specific参数到传入的parser
    if hasattr(args, "arch"): # 训练时时,args存在"arch"参数;测试时,args无"arch"参数,故为False
        model_specific_group = parser.add_argument_group(  # 注意,model_specific_group是在传入的parser中的一个group参数组
            "Model-specific configuration",
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values. # 取出仅包括显式给定为命令行参数或具有默认值的属性,实际上所传入的parser一样
            argument_default=argparse.SUPPRESS,
        )
        ARCH_MODEL_REGISTRY[args.arch].add_args(model_specific_group) # 将arg.arch指定模型的特定参数添加到model_specific_group(parser的参数组)内

    # Add *-specific args to parser. 添加特定参数到传入的parser
    # REGISTRIES is a dictionary, automatically import any Python files in the (criterion, tokenizer, bpe, optim, lr_scheduler)/ directory
    # tokenizer, bpe belong data/encoders, default:　criterion->cross_entropy, optimizer->nag,lr_scheduler->fixed
    from fairseq.registry import REGISTRIES

    # 训练时:主要添加criterion/optimizer/lr-scheduler-specific三个模块的默认类特定参数(args.tokenizer/bpe=None);
    # 测试时:主要添加optimizer/lr-scheduler-specific两个模块的默认类特定参数(args.tokenizer/bpe=None,而criterion默认类的add_args为空);
    for registry_name, REGISTRY in REGISTRIES.items():  # 遍历REGISTRIES的5个模块,'criterion','tokenizer','bpe','optimizer','lr_scheduler'
        choice = getattr(args, registry_name, None)  # 通过args('criterion')取出当前模块所要设置的类标记'cross_entropy'
        if choice is not None:  # 若当前模块所要设置的类标记不为None:
            cls = REGISTRY["registry"][choice]  # 通过REGISTRY["registry"][choice]定义该类<'CrossEntropyCriterion'>
            if hasattr(cls, "add_args"):  # 若该类中存在"add_args"属性:
                cls.add_args(parser)  # 将该类特有参数添加到所传入的parser中
    if hasattr(args, "task"):  # 训练/测试时,主要是添加task-specific模块的特定参数;
        from fairseq.tasks import TASK_REGISTRY
        # 通过TASK_REGISTRY[args.task]定义当前模块所要设置的类<'TranslationTask'>
        TASK_REGISTRY[args.task].add_args(parser) # 将该类特有参数添加到所传入的parser中
    if getattr(args, "use_bmuf", False):  # 适用于block distributed data parallelism;非默认情况
        # hack to support extra args for block distributed data parallelism
        from fairseq.optim.bmuf import FairseqBMUF

        FairseqBMUF.add_args(parser)

    # Modify the parser a second time, since defaults may have been reset 第二次修改parser, 因为默认值可能被重置;
    if modify_parser is not None: # 默认modify_parser为None
        modify_parser(parser)

    # Parse a second time. 第二次解析
    if parse_known: # 默认为None:
        args, extra = parser.parse_known_args(input_args)
    else: # 执行parse_args,对修改的默认值,以及所添加的特定参数加入到args中,定义extra=None
        args = parser.parse_args(input_args)
        extra = None

    # Post-process args. 后处理args
    # 训练时: 设置验证时的"max_sentences_valid"和"max_tokens_valid"与训练一致;测试时什么都不做
    if hasattr(args, "max_sentences_valid") and args.max_sentences_valid is None: #　若最大验证集批次(sent-level)为None,则设置与最大训练集批次(--max-sentences)一致;
        args.max_sentences_valid = args.max_sentences
    if hasattr(args, "max_tokens_valid") and args.max_tokens_valid is None:  # 若最大验证集批次(token-level)为None,则设置与最大训练集批次(--max_tokens)一致
        args.max_tokens_valid = args.max_tokens
    if getattr(args, "memory_efficient_fp16", False):  # 获取memory_efficient_fp16的值进行判断,若不存在则为False
        args.fp16 = True  # 鼓励使用内存效率更高的FP16训练版本,将fp16设置为True

    # Apply architecture configuration.　采用架构设置
    if hasattr(args, "arch"):  # 训练时:进入args.arch(即模型),将参数应用到模型内;测试时,args无"arch"参数,故为False
        ARCH_CONFIG_REGISTRY[args.arch](args)  # 具体地,针对args提供进来的参数,已存在的不变,未设置的则按base_architecture设置

    if parse_known:  # 默认为None:
        return args, extra
    else:            # 将全部的arguments设置添加进args(Namespace类型)
        return args


def get_parser(desc, default_task="translation"):
    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None)  # use to specify a custom location for additional modules(not parameters)
    usr_args, _ = usr_parser.parse_known_args()
    utils.import_user_module(usr_args)  # load additional custom modules if "--user-dir" isn't None

    parser = argparse.ArgumentParser(allow_abbrev=False)  # basic ArgumentParser for common configs
    # fmt: off
    parser.add_argument('--no-progress-bar', action='store_true', help='disable progress bar')  # don't report progress information
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='log progress every N batches (when progress bar is disabled)')  # equal to <disp_freq>
    parser.add_argument('--log-format', default=None, help='log format to use',
                        choices=['json', 'none', 'simple', 'tqdm'])  # log format to use
    parser.add_argument('--tensorboard-logdir', metavar='DIR', default='',
                        help='path to save logs for tensorboard, should match --logdir '
                             'of running tensorboard (default: no tensorboard logging)')  # path to save logs for tensorboard, default: no
    parser.add_argument('--seed', default=1, type=int, metavar='N',
                        help='pseudo random number generator seed')
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    # Parameters related to fp16
    parser.add_argument('--fp16', action='store_true', help='use FP16')
    parser.add_argument('--memory-efficient-fp16', action='store_true',
                        help='use a memory-efficient version of FP16 training; implies --fp16')  # 使用内存效率更高的FP16训练版本;指示--fp16
    parser.add_argument('--fp16-no-flatten-grads', action='store_true',
                        help='don\'t flatten FP16 grads tensor')
    parser.add_argument('--fp16-init-scale', default=2 ** 7, type=int,
                        help='default FP16 loss scale')
    parser.add_argument('--fp16-scale-window', type=int,
                        help='number of updates before increasing loss scale')
    parser.add_argument('--fp16-scale-tolerance', default=0.0, type=float,
                        help='pct of updates that can overflow before decreasing the loss scale')
    parser.add_argument('--min-loss-scale', default=1e-4, type=float, metavar='D',
                        help='minimum FP16 loss scale, after which training is stopped')
    parser.add_argument('--threshold-loss-scale', type=float,
                        help='threshold FP16 loss scale from below')
    ####################################################################
    parser.add_argument('--user-dir', default=None,
                        help='path to a python module containing custom extensions (tasks and/or architectures)')
    parser.add_argument('--empty-cache-freq', default=0, type=int,
                        help='how often to clear the PyTorch CUDA cache (0 to disable)')  # 多久清理一次Cuda缓存
    parser.add_argument('--all-gather-list-size', default=16384, type=int,
                        help='number of bytes reserved for gathering stats from workers')

    # REGISTRIES is a dictionary, automatically import any Python files in the (criterion, tokenizer, bpe, optim, lr_scheduler)/ directory
    # tokenizer, bpe belong data/encoders, default:　criterion->cross_entropy, optimizer->nag,lr_scheduler->fixed
    from fairseq.registry import REGISTRIES
    # traverse the parameters of the five modules of REGISTRIES, set the default parameters and optional parameters
    for registry_name, REGISTRY in REGISTRIES.items():
        parser.add_argument(
            '--' + registry_name.replace('_', '-'),
            default=REGISTRY['default'],
            choices=REGISTRY['registry'].keys(),
        )

    # Task definitions can be found under fairseq/tasks/ and fairseq/benchmark/
    # TASK_REGISTRY is a dictionary, automatically import any Python files in the (tasks)/ directory
    # where fairseq_task is the basic python file, default->default_task and set all tasks as optional parameters
    from fairseq.tasks import TASK_REGISTRY
    parser.add_argument('--task', metavar='TASK', default=default_task,
                        choices=TASK_REGISTRY.keys(),
                        help='task')
    # fmt: on
    return parser


def add_preprocess_args(parser):  # actual preprocessing parameters
    group = parser.add_argument_group("Preprocessing")
    # fmt: off
    # source and target languages
    group.add_argument("-s", "--source-lang", default=None, metavar="SRC",
                       help="source language")
    group.add_argument("-t", "--target-lang", default=None, metavar="TARGET",
                       help="target language")
    # the prefixes of train, valid and test files
    group.add_argument("--trainpref", metavar="FP", default=None,
                       help="train file prefix")
    group.add_argument("--validpref", metavar="FP", default=None,
                       help="comma separated, valid file prefixes")
    group.add_argument("--testpref", metavar="FP", default=None,
                       help="comma separated, test file prefixes")
    group.add_argument("--align-suffix", metavar="FP", default=None,
                       help="alignment file suffix")  # the alignment file suffix
    # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
    group.add_argument("--matrix-suffix", metavar="FP", default=None,
                        help="extra matrix file suffix")  # the matirx file suffix
    group.add_argument("--destdir", metavar="DIR", default="data-bin",
                       help="destination dir")  # the directory of final output file
    # <我们方法>,, map source and target words appearing less than threshold times to unknown
    group.add_argument("--thresholdtgt", metavar="N", default=0, type=int,
                       help="map words appearing less than threshold times to unknown")
    group.add_argument("--thresholdsrc", metavar="N", default=0, type=int,
                       help="map words appearing less than threshold times to unknown")
    # source and target dictionaries, reuse the given one
    group.add_argument("--tgtdict", metavar="FP",
                       help="reuse given target dictionary")
    group.add_argument("--srcdict", metavar="FP",
                       help="reuse given source dictionary")
    # source and target vocabulary sizes
    group.add_argument("--nwordstgt", metavar="N", default=-1, type=int,
                       help="number of target words to retain")
    group.add_argument("--nwordssrc", metavar="N", default=-1, type=int,
                       help="number of source words to retain")
    group.add_argument("--alignfile", metavar="ALIGN", default=None,
                       help="an alignment file (optional)")  # the alignment file
    parser.add_argument('--dataset-impl', metavar='FORMAT', default='mmap',
                        choices=get_available_dataset_impl(),
                        help='output dataset implementation') # the format of output dataset, 'backup/lazy/cached/mmap'-> different data processing methods
    group.add_argument("--joined-dictionary", action="store_true",
                       help="Generate joined dictionary")  # the shared vocabulary for src and tgt lang
    group.add_argument("--only-source", action="store_true",
                       help="Only process the source language")  # Only process the source language
    group.add_argument("--padding-factor", metavar="N", default=8, type=int,
                       help="Pad dictionary size to be multiple of N")  # Pad dictionary size to be multiple of N, which is important on some hardware(Nvidia Tensor Cores)
    group.add_argument("--workers", metavar="N", default=1, type=int,
                       help="number of parallel workers")  # number of parallel workers(processes)
    # fmt: on
    return parser


def add_dataset_args(parser, train=False, gen=False):  # actual dataset and data loading parameters,根据训练标记(train)/测试(gen)标记决定独有部分
    group = parser.add_argument_group("Dataset and data loading")
    # fmt: off
    # 该部分训练/验证集共用
    group.add_argument('--num-workers', default=1, type=int, metavar='N',
                       help='how many subprocesses to use for data loading')  # 用于数据加载的子进程数
    group.add_argument('--skip-invalid-size-inputs-valid-test', action='store_true',
                       help='ignore too long or too short lines in valid and test set')  # 忽略验证/测试集中过长/短的句子
    group.add_argument('--max-tokens', type=int, metavar='N',
                       help='maximum number of tokens in a batch')  # 一个训练/测试批次中有多少个token,即batch size (token-level),默认用于训练
    group.add_argument('--max-sentences', '--batch-size', type=int, metavar='N',
                       help='maximum number of sentences in a batch')  # 一个训练/测试批次中有多少个句子,即batch size (sents-level),默认用于测试
    group.add_argument('--required-batch-size-multiple', default=8, type=int, metavar='N',
                       help='batch size will be a multiplier of this value')  # 一个训练/验证/测试批次是这个数的整数
    parser.add_argument('--dataset-impl', metavar='FORMAT',
                        choices=get_available_dataset_impl(),
                        help='output dataset implementation')  # 输出数据集的格式
    if train:  # 该部分训练独有
        group.add_argument('--train-subset', default='train', metavar='SPLIT',
                           help='data subset to use for training (e.g. train, valid, test)')  # 用于训练的数据子集(split)名称,默认为'train'
        group.add_argument('--valid-subset', default='valid', metavar='SPLIT',
                           help='comma separated list of data subsets to use for validation'
                                ' (e.g. train, valid, test)')  # 用于验证的数据子集(split)名称,默认为'valid',逗号分割
        group.add_argument('--validate-interval', type=int, default=1, metavar='N',
                           help='validate every N epochs')  # 每N轮验证一次,等价于vaild_freq;默认每一轮验证一次(标准)
        group.add_argument('--fixed-validation-seed', default=None, type=int, metavar='N',
                           help='specified random seed for validation')  # 指定用于验证的随机数种子
        group.add_argument('--disable-validation', action='store_true',
                           help='disable validation')         # 禁止验证
        group.add_argument('--max-tokens-valid', type=int, metavar='N',
                           help='maximum number of tokens in a validation batch'
                                ' (defaults to --max-tokens)')  # 一个验证批次中有多少个token,即valid batch size (token-level),默认与--max-tokens一致
        group.add_argument('--max-sentences-valid', type=int, metavar='N',
                           help='maximum number of sentences in a validation batch'
                                ' (defaults to --max-sentences)')  # 一个验证批次中有多少个句子,即valid batch size (sent-level),默认与--max-sentences一致
        group.add_argument('--curriculum', default=0, type=int, metavar='N',
                           help='don\'t shuffle batches for first N epochs')  # 在前N轮内不打乱数据集
    if gen:  # 该部分测试独有
        group.add_argument('--gen-subset', default='test', metavar='SPLIT',
                           help='data subset to generate (train, valid, test)')  # 用于测试的数据子集(split)名称,默认为'test'
        group.add_argument('--num-shards', default=1, type=int, metavar='N',
                           help='shard generation over N shards')  # 将测试数据集分成N个分片
        group.add_argument('--shard-id', default=0, type=int, metavar='ID',
                           help='id of the shard to generate (id < num_shards)')  # 每个测试集分片的ID（ID <num_shards）
    # fmt: on
    return group


def add_distributed_training_args(parser):  # actual distributed training parameters
    group = parser.add_argument_group("Distributed training")
    # fmt: off
    group.add_argument('--distributed-world-size', type=int, metavar='N',
                       default=max(1, torch.cuda.device_count()),
                       help='total number of GPUs across all nodes (default: all visible GPUs)')
    group.add_argument('--distributed-rank', default=0, type=int,
                       help='rank of the current worker')
    group.add_argument('--distributed-backend', default='nccl', type=str,
                       help='distributed backend')
    group.add_argument('--distributed-init-method', default=None, type=str,
                       help='typically tcp://hostname:port that will be used to '
                            'establish initial connetion')
    group.add_argument('--distributed-port', default=-1, type=int,
                       help='port number (not required if using --distributed-init-method)')
    group.add_argument('--device-id', '--local_rank', default=0, type=int,
                       help='which GPU to use (usually configured automatically)')
    group.add_argument('--distributed-no-spawn', action='store_true',
                       help='do not spawn multiple processes even if multiple GPUs are visible')
    # "c10d" is PyTorch's DDP implementation and provides the fastest
    # training. "no_c10d" is a more robust, but slightly slower DDP
    # implementation. Try this if you get warning messages about
    # inconsistent gradients between workers, or if some of your model
    # parameters are not always used.
    group.add_argument('--ddp-backend', default='c10d', type=str,
                       choices=['c10d', 'no_c10d'],
                       help='DistributedDataParallel backend')
    group.add_argument('--bucket-cap-mb', default=25, type=int, metavar='MB',
                       help='bucket size for reduction')
    group.add_argument('--fix-batches-to-gpus', action='store_true',
                       help='don\'t shuffle batches between GPUs; this reduces overall '
                            'randomness and may affect precision but avoids the cost of '
                            're-reading the data')  # 不要在GPU之间混批处理；这会降低整体随机性,并可能影响精度,但避免了重新读取数据的成本
    group.add_argument('--find-unused-parameters', default=False, action='store_true',
                       help='disable unused parameter detection (not applicable to '
                       'no_c10d ddp-backend')
    group.add_argument('--fast-stat-sync', default=False, action='store_true',
                       help='[deprecated] this is now defined per Criterion')
    group.add_argument('--broadcast-buffers', default=False, action='store_true',
                       help='Copy non-trainable parameters between GPUs, such as '
                      'batchnorm population statistics')
    # fmt: on
    return group


def add_optimization_args(parser):  # actual optimization parameters
    group = parser.add_argument_group("Optimization")
    # fmt: off
    group.add_argument('--max-epoch', '--me', default=0, type=int, metavar='N',
                       help='force stop training at specified epoch')   # 在指定epoch内,强制停止训练
    group.add_argument('--max-update', '--mu', default=0, type=int, metavar='N',
                       help='force stop training at specified update')  # 在指定update(训练批次)内,强制停止训练
    group.add_argument('--clip-norm', default=25, type=float, metavar='NORM',
                       help='clip threshold of gradients')  # 梯度修剪阈值
    group.add_argument('--sentence-avg', action='store_true',
                       help='normalize gradients by the number of sentences in a batch'
                            ' (default is to normalize by number of tokens)')  # 通过批处理中句子的数量对梯度进行归一化(默认为tokens数)
    group.add_argument('--update-freq', default='1', metavar='N1,N2,...,N_K',
                       type=lambda uf: eval_str_list(uf, type=int),
                       help='update parameters every N_i batches, when in epoch i')  # 在第i个时期, 每N_i个批次更新一次参数,类似梯度累积更新
    group.add_argument('--lr', '--learning-rate', default='0.25', type=eval_str_list,
                       metavar='LR_1,LR_2,...,LR_N',
                       help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)') # 前N轮的学习率,超过N的全部轮使用LR_N,具体取决于--lr-scheduler
    group.add_argument('--min-lr', default=-1, type=float, metavar='LR',
                       help='stop training when the learning rate reaches this minimum')  # 当学习率达到最低要求时停止训练
    group.add_argument('--use-bmuf', default=False, action='store_true',
                       help='specify global optimizer for syncing models on different GPUs/shards') # 不同的GPU/数据分片上为了同步模型指定全局的优化器
    # fmt: on
    return group


def add_checkpoint_args(parser):  # actual Checkpointing parameters
    group = parser.add_argument_group("Checkpointing")
    # fmt: off
    group.add_argument('--save-dir', metavar='DIR', default='checkpoints',
                       help='path to save checkpoints')  # 存储检查点文件的路径
    group.add_argument('--restore-file', default='checkpoint_last.pt',
                       help='filename from which to load checkpoint '
                            '(default: <save-dir>/checkpoint_last.pt')  # 加载检查点的文件名,默认为<save-dir>/checkpoint_last.pt
    # 如果设置如下四种reset(dataloader,lr-scheduler,meters,optimizer),则不从检查点文件中加载该四种模块
    group.add_argument('--reset-dataloader', action='store_true',
                       help='if set, does not reload dataloader state from the checkpoint')
    group.add_argument('--reset-lr-scheduler', action='store_true',
                       help='if set, does not load lr scheduler state from the checkpoint')
    group.add_argument('--reset-meters', action='store_true',
                       help='if set, does not load meters from the checkpoint')
    group.add_argument('--reset-optimizer', action='store_true',
                       help='if set, does not load optimizer state from the checkpoint')
    group.add_argument('--optimizer-overrides', default="{}", type=str, metavar='DICT',
                       help='a dictionary used to override optimizer args when loading a checkpoint') # 加载检查点时用于覆盖优化器参数的字典
    group.add_argument('--save-interval', type=int, default=1, metavar='N',
                       help='save a checkpoint every N epochs')    # 每多少轮存储一个检查点文件,等价于save_freq
    group.add_argument('--save-interval-updates', type=int, default=0, metavar='N',
                       help='save a checkpoint (and validate) every N updates')  # 每多少updates(训练批次)存储一个检查点文件,等价于save_freq
    group.add_argument('--keep-interval-updates', type=int, default=-1, metavar='N',
                       help='keep the last N checkpoints saved with --save-interval-updates')  # 保留最后N个检查点(间隔为--save-interval-updates updates,vaswani论文方法)
    group.add_argument('--keep-last-epochs', type=int, default=-1, metavar='N',
                       help='keep last N epoch checkpoints')   # 保留最后N轮检查点(间隔为--save-interval epoch,非vaswani论文方法)
    group.add_argument('--keep-best-checkpoints', type=int, default=-1, metavar='N',
                       help='keep best N checkpoints based on scores')  # 基于scores保留最佳的N个检查点(并非scores决定和最佳模型,非vaswani论文方法)
    group.add_argument('--no-save', action='store_true',
                       help='don\'t save models or checkpoints')  # 不存储任何模型和检查点文件(非vaswani论文方法)
    group.add_argument('--no-epoch-checkpoints', action='store_true',
                       help='only store last and best checkpoints')  # 只存储最后和最佳的检查点文件(只要最后的.非vaswani论文方法)
    group.add_argument('--no-last-checkpoints', action='store_true',
                       help='don\'t store last checkpoints')     # 不存储最后的检查点文件(非vaswani论文方法)
    group.add_argument('--no-save-optimizer-state', action='store_true',
                       help='don\'t save optimizer-state as part of checkpoint')  # 检查点文件不存储优化器部分
    group.add_argument('--best-checkpoint-metric', type=str, default='loss',
                       help='metric to use for saving "best" checkpoints')  # 用于存储"best"检查点文件的指标
    group.add_argument('--maximize-best-checkpoint-metric', action='store_true',
                       help='select the largest metric value for saving "best" checkpoints')  # 选择最大的度量值以保存“最佳”检查点
    group.add_argument('--patience', type=int, default=-1, metavar='N',
                       help=('early stop training if valid performance doesn\'t '
                             'improve for N consecutive validation runs; note '
                             'that this is influenced by --validate-interval')) # early stopping的patience,多少个valid_freq验证集性能未提升即提早停止训练
    # fmt: on
    return group


def add_common_eval_args(group):
    # fmt: off
    group.add_argument('--path', metavar='FILE',
                       help='path(s) to model file(s), colon separated')  # 用于评估的模型文件的路径,冒号分隔
    group.add_argument('--remove-bpe', nargs='?', const='@@ ', default=None,
                       help='remove BPE tokens before scoring (can be set to sentencepiece)')  # 在评分之前删除BPE标识符,若设置则保持常量为"@@ "(可设置成sentencepiece)
    group.add_argument('--quiet', action='store_true',
                       help='only print final scores')  # 仅打印最终分数
    group.add_argument('--model-overrides', default="{}", type=str, metavar='DICT',
                       help='a dictionary used to override model args at generation '
                            'that were used during model training')  #  一个字典用于覆盖在测试时使用的经过模型训练得到的模型args
    group.add_argument('--results-path', metavar='RESDIR', type=str, default=None,
                       help='path to save eval results (optional)"')  # 保存评估结果的路径（可选）
    # fmt: on


def add_eval_lm_args(parser):
    group = parser.add_argument_group("LM Evaluation")
    add_common_eval_args(group)
    # fmt: off
    group.add_argument('--output-word-probs', action='store_true',
                       help='if set, outputs words and their predicted log probabilities to standard output')
    group.add_argument('--output-word-stats', action='store_true',
                       help='if set, outputs word statistics such as word count, average probability, etc')
    group.add_argument('--context-window', default=0, type=int, metavar='N',
                       help='ensures that every evaluated token has access to a context of at least this size,'
                            ' if possible')
    group.add_argument('--softmax-batch', default=sys.maxsize, type=int, metavar='N',
                       help='if BxT is more than this, will batch the softmax over vocab to this amount of tokens'
                            ' in order to fit into GPU memory')
    # fmt: on


def add_generation_args(parser):
    group = parser.add_argument_group("Generation") # actual Generation parameters
    add_common_eval_args(group) # 添加通用的eval参数,适用于不同任务(如translation,lm)
    # fmt: off
    group.add_argument('--beam', default=5, type=int, metavar='N',
                       help='beam size')  # 波束搜索的波束大小
    group.add_argument('--nbest', default=1, type=int, metavar='N',
                       help='number of hypotheses to output')  # 输出假设(所生成的预测翻译)的数量
    group.add_argument('--max-len-a', default=0, type=float, metavar='N',
                       help=('generate sequences of maximum length ax + b, '
                             'where x is the source length'))  # 生成最大长度为ax + b的序列, 其中x为源句子长度(a默认为0)
    group.add_argument('--max-len-b', default=200, type=int, metavar='N',
                       help=('generate sequences of maximum length ax + b, '
                             'where x is the source length'))  # 生成最大长度为ax + b的序列, 其中x为源句子长度(b默认为200)
    group.add_argument('--min-len', default=1, type=float, metavar='N',
                       help=('minimum generation length'))  # 最小生成序列长度
    group.add_argument('--match-source-len', default=False, action='store_true',
                       help=('generations should match the source length'))  # 生成的预测翻译序列应该匹配的最大源长度
    group.add_argument('--no-early-stop', action='store_true',
                       help='deprecated')  # 遗弃方法
    group.add_argument('--unnormalized', action='store_true',
                       help='compare unnormalized hypothesis scores')  # 比较未归一化的输出假设的得分
    group.add_argument('--no-beamable-mm', action='store_true',
                       help='don\'t use BeamableMM in attention layers')  # 不在注意力层中使用BeamableMM
    group.add_argument('--lenpen', default=1, type=float,
                       help='length penalty: <1.0 favors shorter, >1.0 favors longer sentences') # 长度惩罚α：<1.0有利于较短的句子,>1.0有利于较长的句子
    group.add_argument('--unkpen', default=0, type=float,
                       help='unknown word penalty: <0 produces more unks, >0 produces fewer') # 未知词惩罚：<0产生更多unks，>0产生更少unks
    group.add_argument('--replace-unk', nargs='?', const=True, default=None,
                       help='perform unknown replacement (optionally with alignment dictionary)')  # 执行未知词替换技术(带对齐词典则为可选项)
    group.add_argument('--sacrebleu', action='store_true',
                       help='score with sacrebleu')  # 由sacrebleu进行打分
    group.add_argument('--score-reference', action='store_true',
                       help='just score the reference translation')  # 只需给参考翻译打分
    group.add_argument('--prefix-size', default=0, type=int, metavar='PS',
                       help='initialize generation by target prefix of given length') # 通过给定长度的目标前缀来初始化生成的预测翻译序列
    group.add_argument('--no-repeat-ngram-size', default=0, type=int, metavar='N',
                       help='ngram blocking such that this size ngram cannot be repeated in the generation') # ngram块,使得这样大小的ngram不会在生成的预测翻译序列中被重复
    group.add_argument('--sampling', action='store_true',
                       help='sample hypotheses instead of using beam search')  # 采用随机搜索假设来替代beam search
    group.add_argument('--sampling-topk', default=-1, type=int, metavar='PS',
                       help='sample from top K likely next words instead of all words')  # 从前K个最可能的词中随机搜索,而不是所有单词
    group.add_argument('--sampling-topp', default=-1.0, type=float, metavar='PS',
                       help='sample from the smallest set whose cumulative probability mass exceeds p for next words') # 从最小集合中随机搜索,其中对下一个单词的累积概率值超过p
    group.add_argument('--temperature', default=1., type=float, metavar='N',
                       help='temperature for generation')  # 用于预测过程的温度
    group.add_argument('--diverse-beam-groups', default=-1, type=int, metavar='N',
                       help='number of groups for Diverse Beam Search') # 用于Diverse Beam Search的组数
    group.add_argument('--diverse-beam-strength', default=0.5, type=float, metavar='N',
                       help='strength of diversity penalty for Diverse Beam Search') # 用于Diverse Beam Search的多样性惩罚强度
    group.add_argument('--diversity-rate', default=-1.0, type=float, metavar='N',
                       help='strength of diversity penalty for Diverse Siblings Search') # 用于Diverse Siblings Search的多样性惩罚强度
    group.add_argument('--print-alignment', action='store_true',
                       help='if set, uses attention feedback to compute and print alignment to source tokens') # 若已设置,则使用注意力反馈来计算和打印对源tokens的对齐(对齐/注意力权重)
    group.add_argument('--print-step', action='store_true')  # 打印每一个生成步骤

    # arguments for iterative refinement generator 用于迭代强化生成器的参数
    group.add_argument('--iter-decode-eos-penalty', default=0.0, type=float, metavar='N',
                       help='if > 0.0, it penalized early-stopping in decoding.')  # 如果>0.0,则其在解码时会惩罚提前停止
    group.add_argument('--iter-decode-max-iter', default=10, type=int, metavar='N',
                       help='maximum iterations for iterative refinement.')  # 用于迭代强化的最大迭代次数
    group.add_argument('--iter-decode-force-max-iter', action='store_true',
                       help='if set, run exact the maximum number of iterations without early stop')  # 若设置,精确运行最大迭代次数而无需提前停止
    group.add_argument('--iter-decode-with-beam', default=1, type=int, metavar='N',
                       help='if > 1, model will generate translations varying by the lengths.')  # 若>1, 则模型将生成随长度变化的翻译
    group.add_argument('--iter-decode-with-external-reranker', action='store_true',
                       help='if set, the last checkpoint are assumed to be a reranker to rescore the translations'),  # 若设置，最后一个检查点将被认作重新为翻译打分的重排器
    group.add_argument('--retain-iter-history', action='store_true',
                       help='if set, decoding returns the whole history of iterative refinement') # 若设置,则解码返回迭代强化的全部历史记录

    # special decoding format for advanced decoding.　用于高级解码的特殊解码格式
    group.add_argument('--decoding-format', default=None, type=str, choices=['unigram', 'ensemble', 'vote', 'dp', 'bs']) # 设置高级解码的格式,选项有'unigram', 'ensemble', 'vote', 'dp', 'bs'
    # fmt: on
    return group


def add_interactive_args(parser):
    group = parser.add_argument_group("Interactive")
    # fmt: off
    group.add_argument('--buffer-size', default=0, type=int, metavar='N',
                       help='read this many sentences into a buffer before processing them')
    group.add_argument('--input', default='-', type=str, metavar='FILE',
                       help='file to read from; use - for stdin')
    # fmt: on


def add_model_args(parser):  # actual model parameters
    group = parser.add_argument_group("Model configuration")
    # fmt: off

    # Model definitions can be found under fairseq/models/
    #
    # The model architecture can be specified in several ways.
    # In increasing order of priority:
    # 1) model defaults (lowest priority) 模型默认参数,优先级最低
    # 2) --arch argument  arch参数,优先级中等
    # 3) --encoder/decoder-* arguments (highest priority) encoder/decoder参数,优先级最高

    # REGISTRIES is a dictionary, automatically import any Python files in the (models)/ directory
    # default:　fconv, 其他作为可选项,<我们方法>,需要关注<class 'fairseq.models.transformer.TransformerModel'>, 具体为
    # 其中有transformer,transformer_wmt_en_de,transformer_iwslt_de_en等,主要关注transformer_wmt_en_de
    from fairseq.models import ARCH_MODEL_REGISTRY
    # traverse the parameters of the models modules of REGISTRIES, set the default parameters and optional parameters
    group.add_argument('--arch', '-a', default='fconv', metavar='ARCH',
                       choices=ARCH_MODEL_REGISTRY.keys(),
                       help='Model Architecture')
    # fmt: on
    return group
