# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
import os
import re
import traceback
from collections import OrderedDict
from typing import Union

import torch
from fairseq.file_io import PathManager
from fairseq.models import FairseqDecoder, FairseqEncoder
from torch.serialization import default_restore_location


logger = logging.getLogger(__name__)


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    from fairseq import distributed_utils, meters

    prev_best = getattr(save_checkpoint, "best", val_loss) # 若save_checkpoint无'best'参数,则给定为当前验证集平均目标词对应的－lable_smoothing loss/log(2)
    if val_loss is not None: # 若当前验证集平均目标词对应的－lable_smoothing loss/log(2)不为空:
        best_function = max if args.maximize_best_checkpoint_metric else min # 默认最佳检查点指标越小越好
        save_checkpoint.best = best_function(val_loss, prev_best) # 比较当前/以往最佳验证集平均目标词对应的－lable_smoothing loss/log(2),存储最优的val_loss

    if args.no_save or not distributed_utils.is_master(args):  # 若不存储任何模型:
        return

    def is_better(a, b):
        return a >= b if args.maximize_best_checkpoint_metric else a <= b

    write_timer = meters.StopwatchMeter() # 开启一个计算某事件的总和/平均持续时间(s)的计时器
    write_timer.start()  # 记录代码所持续的时间(两次调用之间的时间)

    epoch = epoch_itr.epoch   # 获取当前轮数,是否轮数迭代器已到最后,以及当前更新批次步骤数
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict() # 创建检查点字典,其中键值对为{"checkpoint轮数,更新批次步骤数,最优检查点.pt":True,False}
    checkpoint_conds["checkpoint{}.pt".format(epoch)] = (
        end_of_epoch  # 若轮数迭代器已到最后,则此项为True
        and not args.no_epoch_checkpoints
        and epoch % args.save_interval == 0
    )
    checkpoint_conds["checkpoint_{}_{}.pt".format(epoch, updates)] = (
        not end_of_epoch
        and args.save_interval_updates > 0
        and updates % args.save_interval_updates == 0  # 若当前更新批次步骤数可以整除存储频率,则该项为True
    )
    checkpoint_conds["checkpoint_best.pt"] = val_loss is not None and (
        not hasattr(save_checkpoint, "best")
        or is_better(val_loss, save_checkpoint.best)  # 且若当前优于以往最佳验证集平均目标词对应的－lable_smoothing loss/log(2),则该项为True
    )
    if val_loss is not None and args.keep_best_checkpoints > 0:  # 非默认情况,不基于scores(如'val_loss')保留最佳的N个检查点
        checkpoint_conds["checkpoint.best_{}_{:.2f}.pt".format(
            args.best_checkpoint_metric, val_loss)] = (
            not hasattr(save_checkpoint, "best")
            or is_better(val_loss, save_checkpoint.best)
        )
    checkpoint_conds["checkpoint_last.pt"] = not args.no_last_checkpoints  # 若选择存储最后的检查点文件,则该项为True

    extra_state = {"train_iterator": epoch_itr.state_dict(), "val_loss": val_loss} # extra_state: {'train_iterator': {'epoch': 1, 'iterations_in_epoch': 1, 'shuffle': True}, 'val_loss': 13.211}
    if hasattr(save_checkpoint, "best"):  # 若save_checkpoint存在'best'参数,则将"best": 最优的val_loss添加进字典
        extra_state.update({"best": save_checkpoint.best})
    # extra_state: {'train_iterator': {'epoch': 1, 'iterations_in_epoch': 1, 'shuffle': True}, 'val_loss': 13.211, 'best': 13.211}
    checkpoints = [
        os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ] # 取出checkpoint_conds字典中值为True的checkpoint名称,给每个checkpoint名称加上checkpoint存储路径
    if len(checkpoints) > 0: # 若checkpoint检查点文件列表不为空:
        trainer.save_checkpoint(checkpoints[0], extra_state)  # 将所有训练状态保存在检查点文件checkpoint_epoch_updates.pt中
        for cp in checkpoints[1:]: # 并将所有训练状态同样保存在检查点文件:checkpoint_best.pt,checkpoint_last.pt下;
            PathManager.copy(checkpoints[0], cp, overwrite=True)

        write_timer.stop() # 记录代码所持续的时间(两次调用之间的时间)
        logger.info(
            "saved checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                checkpoints[0], epoch, updates, val_loss, write_timer.sum
            )
        )  # 打印:存储了检查点文件checkpoint_epoch_updates.pt的路径,(第几轮,第几个更新批次步骤数,val_loss值)花费xx时间

    if not end_of_epoch and args.keep_interval_updates > 0:  # 若轮数迭代器未到最后,且保留最后N个检查点(间隔为--save-interval-updates):
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            args.save_dir, pattern=r"checkpoint_\d+_(\d+)\.pt"  # 遍历检查点存储目录下所有检查点文件
        )
        for old_chk in checkpoints[args.keep_interval_updates :]: # 删除旧的检查站,始终保持keep_interval_updates个检查点,且检查点以降序排列
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_last_epochs > 0: # 保留最后N轮检查点(间隔为--save-interval epoch),非默认情况
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(args.save_dir, pattern=r"checkpoint(\d+)\.pt")
        for old_chk in checkpoints[args.keep_last_epochs :]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_best_checkpoints > 0: # 若基于scores(如'val_loss')保留最佳的N个检查点,非默认情况
        # only keep the best N checkpoints according to validation metric
        checkpoints = checkpoint_paths(
            args.save_dir, pattern=r"checkpoint\.best_{}_(\d+\.?\d*)\.pt".format(args.best_checkpoint_metric))
        if not args.maximize_best_checkpoint_metric:
            checkpoints = checkpoints[::-1]
        for old_chk in checkpoints[args.keep_best_checkpoints:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)


def load_checkpoint(args, trainer, **passthrough_args):
    """
    Load a checkpoint and restore the training iterator. 加载一个检查点文件并恢复其训练迭代器

    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """
    # only one worker should attempt to create the required dir 只有一块gpu应尝试创建所需的目录
    if args.distributed_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)  # 若模型输出目录不存在,则创建它

    if args.restore_file == "checkpoint_last.pt":  # 若所设置restore_file为"checkpoint_last.pt",则创建检查点文件路径为args.save_dir+"checkpoint_last.pt"
        checkpoint_path = os.path.join(args.save_dir, "checkpoint_last.pt")
    else:                                          # 否则创建检查点文件路径为所设置的restore_file,等价于"train_from"
        checkpoint_path = args.restore_file

    extra_state = trainer.load_checkpoint(  # 遍历所创建的检查点文件路径,查找是否已存在检查点并进行加载,默认extra_state为None
        checkpoint_path,
        args.reset_optimizer,            # 如果设置,则不从检查点加载optimizer state
        args.reset_lr_scheduler,         # 如果设置,则不从检查点加载lr scheduler state
        eval(args.optimizer_overrides),  # 加载检查点时用于覆盖优化器参数的字典
        reset_meters=args.reset_meters,  # 如果设置,则不从检查点加载meters
    )

    if (
        extra_state is not None
        and "best" in extra_state
        and not args.reset_optimizer
        and not args.reset_meters
    ):  # 若extra_state不为None且存在"best",并且选择从检查点中加载优化器和meters:
        save_checkpoint.best = extra_state["best"]

    if extra_state is not None and not args.reset_dataloader:  # 若若extra_state不为None:,且选择从检查点中加载数据状态:
        # restore iterator from checkpoint
        itr_state = extra_state["train_iterator"]
        epoch_itr = trainer.get_train_iterator(
            epoch=itr_state["epoch"], load_dataset=True, **passthrough_args
        )
        epoch_itr.load_state_dict(itr_state)
    else:  # 默认情况,在此处给定epoch数,是否加载数据标记来加载训练集数据
        epoch_itr = trainer.get_train_iterator(
            epoch=1, load_dataset=True, **passthrough_args
        )  # 返回一个基于torch.utils.data.Dataset上的多epoch数据迭代器

    trainer.lr_step(epoch_itr.epoch) # 在该轮下反复调用每一次更新批次步骤下更新学习率

    return extra_state, epoch_itr  # 返回extra_state 和 一个基于torch.utils.data.Dataset上的多epoch数据迭代器


def load_checkpoint_to_cpu(path, arg_overrides=None):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility).将检查点加载到CPU（进行升级以实现向后兼容性）"""
    with PathManager.open(path, "rb") as f:  # 基于当前用于评估的模型文件的路径,从cpu中加载出当前用于评估的模型对象state,如checkpoint_best模型
        state = torch.load(
            f, map_location=lambda s, l: default_restore_location(s, "cpu")
        )
    # state:{'args':当前用于评估的模型所有超参设置,
    #     'model':当前用于评估的模型所有训练参数值(即所训练的模型)
    # 'optimizer_history': 包含'criterion_name','optimizer_name','lr_scheduler_state'{best:val_loss),'num_updates'
    # 'extra_state':包含'train_iterator','val_loss','best'以及包含各种指标的'metrics'(其中有'default','train','train_inner','valid'四个上下文管理器),
    #               指标由'loss','val_loss','wps','ups','wpb','bsz','num_updates','lr','gnorm','wall','train_wall'
    # 'last_optimizer_state':包含state(优化器部分的所有训练参数值),'param_groups':优化器部分的所有超参设置}

    args = state["args"]  # 从当前用于评估的模型对象state取出当前用于评估的模型所有超参设置
    if arg_overrides is not None:  # 这里arg_overrides为{}不为None,但不会覆盖任何模型训练期间使用的模型超参
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)
    state = _upgrade_state_dict(state)  # 全部操作都不影响结果,返回传入的state
    return state


def load_model_ensemble(filenames, arg_overrides=None, task=None, strict=True):
    """Loads an ensemble of models. 加载一组模型

    Args:
        filenames (List[str]): checkpoint files to load  # 要加载的检查点文件,即用于评估的模型文件的路径(列表)
        arg_overrides (Dict[str,Any], optional): override model args that
            were used during model training  # 覆盖模型训练期间使用的模型超参(字典)
        task (fairseq.tasks.FairseqTask, optional): task to use for loading # 用于加载模型的task类
    """
    ensemble, args, _task = load_model_ensemble_and_task(
        filenames, arg_overrides, task, strict
    ) # 返回存储用于评估的模型(TransformerModel类,模型全部训练参数和缓冲区从模型文件中加载)的列表,用于评估的模型所有超参设置,用于加载模型的task类
    return ensemble, args # 返回存储用于评估的模型(TransformerModel类,模型全部训练参数和缓冲区从模型文件中加载)的列表,用于评估的模型所有超参设置


def load_model_ensemble_and_task(filenames, arg_overrides=None, task=None, strict=True):  # 加载一组模型和task(若为None)
    from fairseq import tasks

    ensemble = [] # 用于存储用于评估的模型(TransformerModel类,模型全部训练参数和缓冲区从模型文件中加载)的列表
    for filename in filenames:  # 遍历用于评估的模型文件的路径,默认仅有一个检查点文件
        if not PathManager.exists(filename):  # 检查当前用于评估的模型文件的路径是否存在:
            raise IOError("Model file not found: {}".format(filename))
        state = load_checkpoint_to_cpu(filename, arg_overrides)  # 根据当前用于评估的模型文件的路径,以及覆盖模型训练期间使用的模型超参(字典),将检查点加载到CPU（进行升级以实现向后兼容性）
        # state:{'args':当前用于评估的模型所有超参设置,
        #     'model':当前用于评估的模型所有训练参数值(即所训练的模型)
        # 'optimizer_history': 包含'criterion_name','optimizer_name','lr_scheduler_state'{best:val_loss),'num_updates'
        # 'extra_state':包含'train_iterator','val_loss','best'以及包含各种指标的'metrics'(其中有'default','train','train_inner','valid'四个上下文管理器),
        #               指标由'loss','val_loss','wps','ups','wpb','bsz','num_updates','lr','gnorm','wall','train_wall'
        # 'last_optimizer_state':包含state(优化器部分的所有训练参数值),'param_groups':优化器部分的所有超参设置}

        args = state["args"] # 从当前用于评估的模型对象state取出当前用于评估的模型所有超参设置
        if task is None:  # 若task为None,非默认情况
            task = tasks.setup_task(args)

        # build model for ensemble  # 根据全部参数args来建立模型和criterion(损失)
        model = task.build_model(args)  # 返回一个TransformerModel类,由与原始论文一致的Transfromer Encoder和Decoder组成
        model.load_state_dict(state["model"], strict=strict, args=args) # 将state_dict中模型全部训练参数和缓冲区加载self(即当前模型)中
        ensemble.append(model) # 存储当前用于评估的模型(TransformerModel类,模型全部训练参数和缓冲区从模型文件中加载)
    return ensemble, args, task # 返回存储用于评估的模型(TransformerModel类,模型全部训练参数和缓冲区从模型文件中加载)的列表,用于评估的模型所有超参设置,用于加载模型的task类


def checkpoint_paths(path, pattern=r"checkpoint(\d+)\.pt"):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = os.listdir(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = float(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logger.error(traceback.format_exc())


def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict


def save_state(
    filename,
    args,
    model_state_dict,
    criterion,
    optimizer,
    lr_scheduler,
    num_updates,
    optim_history=None,
    extra_state=None,
):  # 将所有训练状态保存在检查点文件filename中
    from fairseq import utils

    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}
    state_dict = {
        "args": args,
        "model": model_state_dict if model_state_dict else {},
        "optimizer_history": optim_history
        + [
            {
                "criterion_name": criterion.__class__.__name__,
                "optimizer_name": optimizer.__class__.__name__,
                "lr_scheduler_state": lr_scheduler.state_dict(),
                "num_updates": num_updates,
            }
        ],
        "extra_state": extra_state,
    }
    if utils.has_parameters(criterion):
        state_dict["criterion"] = criterion.state_dict()
    if not args.no_save_optimizer_state:
        state_dict["last_optimizer_state"] = convert_state_dict_type(
            optimizer.state_dict()
        )

    with PathManager.open(filename, "wb") as f:
        torch_persistent_save(state_dict, f)


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints. 升级旧模型检查点的辅助函数,全部操作都不影响结果 """
    from fairseq import models, registry, tasks

    # add optimizer_history
    if "optimizer_history" not in state:
        state["optimizer_history"] = [
            {"criterion_name": "CrossEntropyCriterion", "best_loss": state["best_loss"]}
        ]
        state["last_optimizer_state"] = state["optimizer"]
        del state["optimizer"]
        del state["best_loss"]
    # move extra_state into sub-dictionary
    if "epoch" in state and "extra_state" not in state:
        state["extra_state"] = {
            "epoch": state["epoch"],
            "batch_offset": state["batch_offset"],
            "val_loss": state["val_loss"],
        }
        del state["epoch"]
        del state["batch_offset"]
        del state["val_loss"]
    # reduce optimizer history's memory usage (only keep the last state)
    if "optimizer" in state["optimizer_history"][-1]:
        state["last_optimizer_state"] = state["optimizer_history"][-1]["optimizer"]
        for optim_hist in state["optimizer_history"]:
            del optim_hist["optimizer"]
    # record the optimizer class name
    if "optimizer_name" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["optimizer_name"] = "FairseqNAG"
    # move best_loss into lr_scheduler_state
    if "lr_scheduler_state" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["lr_scheduler_state"] = {
            "best": state["optimizer_history"][-1]["best_loss"]
        }
        del state["optimizer_history"][-1]["best_loss"]
    # keep track of number of updates
    if "num_updates" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["num_updates"] = 0
    # old model checkpoints may not have separate source/target positions
    if hasattr(state["args"], "max_positions") and not hasattr(
        state["args"], "max_source_positions"
    ):
        state["args"].max_source_positions = state["args"].max_positions
        state["args"].max_target_positions = state["args"].max_positions
    # use stateful training data iterator
    if "train_iterator" not in state["extra_state"]:
        state["extra_state"]["train_iterator"] = {
            "epoch": state["extra_state"]["epoch"],
            "iterations_in_epoch": state["extra_state"].get("batch_offset", 0),
        }
    # default to translation task
    if not hasattr(state["args"], "task"):
        state["args"].task = "translation"
    # --backup-text and --lazy-load are deprecated
    if getattr(state["args"], "raw_text", False):
        state["args"].dataset_impl = "backup"
    elif getattr(state["args"], "lazy_load", False):
        state["args"].dataset_impl = "lazy"
    # epochs start at 1
    if state["extra_state"]["train_iterator"] is not None:
        state["extra_state"]["train_iterator"]["epoch"] = max(
            state["extra_state"]["train_iterator"].get("epoch", 1),
            1,
        )

    # set any missing default values in the task, model or other registries 在task,model或其他注册器中设置所有缺少的默认值
    registry.set_defaults(state["args"], tasks.TASK_REGISTRY[state["args"].task])
    registry.set_defaults(state["args"], models.ARCH_MODEL_REGISTRY[state["args"].arch])
    for registry_name, REGISTRY in registry.REGISTRIES.items():
        choice = getattr(state["args"], registry_name, None)
        if choice is not None:
            cls = REGISTRY["registry"][choice]
            registry.set_defaults(state["args"], cls)

    return state  # 全部操作都不影响结果,返回传入的state


def prune_state_dict(state_dict, args):
    """Prune the given state_dict if desired for LayerDrop # 如果需要LayerDrop,则修剪给定的state_dict
    (https://arxiv.org/abs/1909.11556).

    Training with LayerDrop allows models to be robust to pruning at inference
    time. This function prunes state_dict to allow smaller models to be loaded
    from a larger model and re-maps the existing state_dict for this to occur.

    It's called by functions that load models from checkpoints and does not
    need to be called directly.
    """
    if not args or args.arch == "ptt_transformer":
        # args should not be none, but don't crash if it is.
        return state_dict

    encoder_layers_to_keep = (
        args.encoder_layers_to_keep if "encoder_layers_to_keep" in vars(args) else None
    )
    decoder_layers_to_keep = (
        args.decoder_layers_to_keep if "decoder_layers_to_keep" in vars(args) else None
    )

    if not encoder_layers_to_keep and not decoder_layers_to_keep:
        return state_dict

    # apply pruning
    logger.info(
        "Pruning model to specified layer configuration - this works best if the model was trained with LayerDrop"
    )

    def create_pruning_pass(layers_to_keep, layer_name):
        keep_layers = sorted(
            [int(layer_string) for layer_string in layers_to_keep.split(",")]
        )
        mapping_dict = {}
        for i in range(len(keep_layers)):
            mapping_dict[str(keep_layers[i])] = str(i)

        regex = re.compile("^{layer}.*\.layers\.(\d+)".format(layer=layer_name))
        return {"substitution_regex": regex, "mapping_dict": mapping_dict}

    pruning_passes = []
    if encoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(encoder_layers_to_keep, "encoder"))
    if decoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(decoder_layers_to_keep, "decoder"))

    new_state_dict = {}
    for layer_name in state_dict.keys():
        match = re.search("\.layers\.(\d+)\.", layer_name)
        # if layer has no number in it, it is a supporting layer, such as an
        # embedding
        if not match:
            new_state_dict[layer_name] = state_dict[layer_name]
            continue

        # otherwise, layer should be pruned.
        original_layer_number = match.group(1)
        # figure out which mapping dict to replace from
        for pruning_pass in pruning_passes:
            if original_layer_number in pruning_pass["mapping_dict"] and pruning_pass[
                "substitution_regex"
            ].search(layer_name):
                new_layer_number = pruning_pass["mapping_dict"][original_layer_number]
                substitution_match = pruning_pass["substitution_regex"].search(
                    layer_name
                )
                new_state_key = (
                    layer_name[: substitution_match.start(1)]
                    + new_layer_number
                    + layer_name[substitution_match.end(1) :]
                )
                new_state_dict[new_state_key] = state_dict[layer_name]

    # Since layers are now pruned, *_layers_to_keep are no longer needed.
    # This is more of "It would make it work fix" rather than a proper fix.
    if "encoder_layers_to_keep" in vars(args):
        args.encoder_layers_to_keep = None
    if "decoder_layers_to_keep" in vars(args):
        args.decoder_layers_to_keep = None

    return new_state_dict


def load_pretrained_component_from_model(
    component: Union[FairseqEncoder, FairseqDecoder], checkpoint: str
):
    """
    Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
    provided `component` object. If state_dict fails to load, there may be a
    mismatch in the architecture of the corresponding `component` found in the
    `checkpoint` file.
    """
    if not PathManager.exists(checkpoint):
        raise IOError("Model file not found: {}".format(checkpoint))
    state = load_checkpoint_to_cpu(checkpoint)
    if isinstance(component, FairseqEncoder):
        component_type = "encoder"
    elif isinstance(component, FairseqDecoder):
        component_type = "decoder"
    else:
        raise ValueError(
            "component to load must be either a FairseqEncoder or "
            "FairseqDecoder. Loading other component types are not supported."
        )
    component_state_dict = OrderedDict()
    for key in state["model"].keys():
        if key.startswith(component_type):
            # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
            component_subkey = key[len(component_type) + 1 :]
            component_state_dict[component_subkey] = state["model"][key]
    component.load_state_dict(component_state_dict, strict=True)
    return component


def verify_checkpoint_directory(save_dir: str) -> None:  # 检查save_dir是否存在,由一个dummy文件做写入操作进行确认,正常则删除
    if not os.path.exists(save_dir):  # 检查save_dir是否存在,即'checkpoints/ftrans'
        os.makedirs(save_dir, exist_ok=True)
    temp_file_path = os.path.join(save_dir, "dummy")  # 在save_dir目录创建dummy路径,并进行写入操作确认是否正常,确认正常后则删除
    try:
        with open(temp_file_path, "w"):
            pass
    except OSError as e:
        logger.warning("Unable to access checkpoint save directory: {}".format(save_dir))
        raise e
    else:
        os.remove(temp_file_path)
