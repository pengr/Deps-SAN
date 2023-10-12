#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import logging
import math
import os
import sys

import torch

from fairseq import bleu, checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.data import encoders


def main(args):
    assert args.path is not None, '--path required for generation!' # 测试时需提供用于评估的模型文件的路径
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'  # 采用beam search所生成的输出假设数要和波束大小一致
    assert args.replace_unk is None or args.dataset_impl == 'backup', \
        '--replace-unk requires a backup text dataset (--dataset-impl=backup)' # 采用unk替换技术需要在一个原始文本数据集上

    if args.results_path is not None: # 若保存评估结果的路径不为None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
        with open(output_path, 'w', buffering=1) as h:
            return _main(args, h)
    else:  # 选择将评估结果输出到窗口上
        return _main(args, sys.stdout)


def _main(args, output_file):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=output_file,
    )
    logger = logging.getLogger('fairseq_cli.generate') # 定义一个logger日志记录器

    utils.import_user_module(args)  # load additional custom modules if "--user-dir" isn't None

    if args.max_tokens is None and args.max_sentences is None: # 若没有设置批次大小; 脚本设置了batch size(即max_sentences);
        args.max_tokens = 12000
    logger.info(args) # 打印全部args

    use_cuda = torch.cuda.is_available() and not args.cpu # 进行cuda计算(单GPU)的标记

    # Load dataset splits # 由task(如translation等)加载数据集splits
    task = tasks.setup_task(args) # 返回初始化且加载源/目标字典(两个Dictionary类)后的TranslationTask类
    task.load_dataset(args.gen_subset) # 返回源和目标的句子对数据集LanguagePairDataset类

    # Set dictionaries 设置字典
    try:
        src_dict = getattr(task, 'source_dictionary', None)  # 取出task中已加载好的载源字典(Dictionary类)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary  # 取出task中已加载好的载目标字典(Dictionary类)

    # Load ensemble 加载集合模型
    logger.info('loading model(s) from {}'.format(args.path))  # 打印用于评估的模型文件的路径
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )  # 返回存储用于评估的模型(TransformerModel类,模型全部训练参数和缓冲区从模型文件中加载)的列表,用于评估的模型所有超参设置

    # Optimize ensemble for generation
    for model in models:  # 遍历用于评估的模型(TransformerModel类,模型全部训练参数和缓冲区从模型文件中加载)的列表,默认就一个检查点文件
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam, # 默认为波束搜索的波束大小
            need_attn=args.print_alignment, # 使用注意力反馈来计算和打印对源tokens的对齐(对齐/注意力权重),默认为False
        ) # 优化模型以加快生成速度,删除整个模型中所有模块module的权重归一化WeightNorm,将module.need_attn设置为True,并将整个模型设置为eval模式
        if args.fp16:  # 若模型的数据类型选择为fp16;非默认情况
            model.half()
        if use_cuda:   # 若存在可用gpu,则将模型移到cuda()中:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk) # 加载用于未知词替换的对齐字典(若没有未知词替换则对齐字典为None)

    # Load dataset (possibly sharded)
    # 初始化测试集的数据迭代器,与训练集基本一致,区别:默认批次大小为max_sentences(sents-level),未给定随机数生成器种子
    # ignore_invalid_inputs=False(仅当max_positions小于测试集中句子长度时生效),shuffle=False(测试集不打乱)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset), # 根据args.gen_subset返回加载的数据集split类,<LanguagePairDataset>
        max_tokens=args.max_tokens,  # 批次大小(token-level,测试时为None)
        max_sentences=args.max_sentences, # 默认批次大小(sents-level)
        max_positions=utils.resolve_max_positions(
            task.max_positions(),  # 返回task允许的最大句子长度,即(self.args.max_source_positions, self.args.max_target_positions)
            *[model.max_positions() for model in models] # 返回model允许的最大句子长度,即(self.args.max_source_positions, self.args.max_target_positions)
        ),  # 通过resolve_max_positions解决来自多个来源的排名限制,返回(self.args.max_source_positions, self.args.max_target_positions)
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,  # 为False,为太长的句子引发异常
        required_batch_size_multiple=args.required_batch_size_multiple,  # 需要一个批次是这个数的整数(sents-level)
        num_shards=args.num_shards,   # 将测试集分成distributed_world_size片
        shard_id=args.shard_id,       # 将测试集的所有分片带上序号
        num_workers=args.num_workers, # 多少个子进程用于数据加载,0表示将在主进程中加载数据
    ).next_epoch_itr(shuffle=False)  # next_epoch_itr前,返回一个基于torch.utils.data.Dataset上的多epoch数据迭代器(测试集)
    # next_epoch_itr后, 返回self._cur_epoch_itr可记录迭代数的迭代器,基于torch.data.DataLoader在给定的数据集<LanguagePairDataset>类上创建的数据加载器(测试集)

    progress = progress_bar.progress_bar( # 与验证集基本一致,区别:无需给定当前epoch编号和用tensorboard保存日志的路径
        itr,
        log_format=args.log_format,      # 进度打印所用格式
        log_interval=args.log_interval,  # 每多少个batches打印进度,等价于disp_freq
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'), # 若设置不报告进度信息,则默认进度打印格式为tqdm
    ) # 测试时,返回以Simple格式记录输出SimpleProgressBar类,加载self._cur_epoch_itr可记录迭代数的迭代器,log_interval,epoch和prefix=None,offset等

    # Initialize generator # 初始化生成器
    gen_timer = StopwatchMeter()  # 返回一个计算某事件的总和/平均持续时间(s)的计时器
    generator = task.build_generator(args) # 根据args建立一个生成器,返回一个用于生成给定源句子的翻译类SequenceGenerator,详情见fairseq_task.build_generator()

    # Handle tokenization and BPE　# 处理标记化和BPE
    # 预处理时由mose/tokenizer/tokenizer.perl,和subword-nmt/subword_nmt/apply_bpe.py分别对文本进行了totokenizer,bpe;
    # 而测试完后处理则无法利用设置totokenizer,bpe类,只能同样利用mose/tokenizer/detokenizer.perl,'--remove-bpe' flag
    tokenizer = encoders.build_tokenizer(args) # 由于未设置totokenizer,bpe类,故什么也不做
    bpe = encoders.build_bpe(args)

    def decode_fn(x): # 调用所建立的tokenizer和bpe类的decode,对所生成的预测翻译进行后处理;由于未建立tokenizer和bpe类,非默认情况
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Generate and compute BLEU score  # 生成并计算BLEU分数
    if args.sacrebleu:  # 若由sacrebleu进行打分,非默认情况
        scorer = bleu.SacrebleuScorer()
    else: # 默认由目标字典的pad/unk/eos符号对应idx,初始化BLEU评分类
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
        # 返回一个BleuStat类,其中主要为ctype类型的结构体,记录计算BLEU分需要的reflen,predlen,match1~4,count1~4(全部被初始化为0)
    num_sentences = 0  # 初始化完成句子数
    has_target = True  # 初始化存在目标测试句子
    wps_meter = TimeMeter()  # 计算每秒某事件的平均发生率的计时器
    for sample in progress: # 遍历SimpleProgressBar类中self._cur_epoch_itr可记录迭代数的迭代器,其中通过了多个类的__iter__(),得到一个批次的示例
        # id': 当前遍历的batch中全部句子对的ID的Tensor,按照源句子长度降序的order重新排列每行;'nsentences':句子对数;'ntokens':该批次所有目标句子的tokens总数;
        # 'net_input':
        #    'src_tokens':当前遍历的batch中全部源句子的tensor所转换的二维填充向量,按照源句子降序的order重新排列每行;
        #    'src_lengths':将当前遍历的batch中全部源句子的长度的Tensor;
        #    'prev_output_tokens':创建目标的移位版本(即全部目标句子的eos移到开头)
        # 'target': 当前遍历的batch中全部目标句子的tensor所转换的二维填充向量,按照源句子降序的order重新排列每行;
        sample = utils.move_to_cuda(sample) if use_cuda else sample  # 将sample中的tensor全部移到cuda中
        if 'net_input' not in sample: # 若sample中不存在'net_input',则需得到新遍历的batch
            continue

        prefix_tokens = None  # 初始化目标前缀标识符(强制解码器从这些标识符开始)
        if args.prefix_size > 0:  # 若通过给定长度的目标前缀来初始化生成的预测翻译序列;非默认情况
            prefix_tokens = sample['target'][:, :args.prefix_size]

        gen_timer.start()  # 计算某事件的总和/平均持续时间(s)的计时器,开始计时
        # 根据用于生成给定源句子的翻译类SequenceGenerator,存储用于评估的模型(TransformerModel类,模型全部训练参数和缓冲区从模型文件中加载)的列表,
        # 一个批次的示例以及目标前缀标识符(强制解码器从这些标识符开始)得到该批次的最终翻译结果,其中每个句子对应的beam_size个翻译结果按'scores'从小到大排序过(越小越好)
        hypos = task.inference_step(generator, models, sample, prefix_tokens)
        # hypos为该批次的最终翻译结果:每个句子对应有beam_size个翻译结果(按'scores'从小到大排序过),每个如下:
        # {
        #     'tokens': tokens_clone[i],  # 当前解码时间步的待生成目标词idx序列的副本
        #     'score': score,  # 当前解码时间步的归一化的概率lprobs最大且为eos的值;
        #     'attention': hypo_attn,　# src_len x tgt_len # 返回当前解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重的副本
        #     'alignment': None,
        #     'positional_scores': pos_scores[i],  # 当前解码时间步的每个假设的分数(lprobs值)
        # }
        num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos) # 累计该批次的每个句子的最佳翻译结果(h[0])中的翻译生成序列(['tokens'])的长度;即该批次的总翻译tokens数
        gen_timer.stop(num_generated_tokens) # 计算某事件的总和/平均持续时间(s)的计时器,结束计时

        for i, sample_id in enumerate(sample['id'].tolist()): # 遍历当前遍历的batch中的每个句子对(按源句子长度降序排序),i为遍历了多少个句子对,而sample_id为句子对在该批中的编号
            has_target = sample['target'] is not None  # 是否存在目标句子的标记

            # Remove padding  # 移除填充
            # 从sample['net_input']['src_tokens'][i, :]中取出当前遍历的batch中当前源句子的tensor填充向量(1D张量,形状为1,src_len);
            # 返回该tensor填充向量中不等于pad_idx的序列(若存在元素等于pad_idx,则直接舍弃),形状不定
            src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad()) # 返回所遍历的源句子的原始token索引tensor(无填充,cuda类型)
            target_tokens = None
            if has_target: # 若存在目标句子:
                # 从sample['target'][i, :]中取出当前遍历的batch中当前目标句子的tensor填充向量(1D张量,形状为1,tgt_len);
                # 返回该tensor填充向量中不等于pad_idx的序列(若存在元素等于pad_idx,则直接舍弃),形状不定
                target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu() # 返回所遍历的目标句子的原始token索引tensor(无填充,cpu类型)

            # Either retrieve the original sentences or regenerate them from tokens. # 检索原始句子或从标记中重新生成它们
            if align_dict is not None:  # 加载用于未知词替换的对齐字典(若没有未知词替换则对齐字典为None)
                src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
            else:
                if src_dict is not None: # 若源字典不为空,则根据所遍历的源句子的原始token索引tensor(无填充,cuda类型),以及选择在评分之前删除BPE标识符(arg.remove_bpe为"@@ ")
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                    # 返回移除bpe符号的所遍历到的源句子的原始token索引tensor经过对应转换后的string序列
                else:
                    src_str = ""
                if has_target:  # 若存在目标句子:
                    # 根据所遍历的目标句子的原始token索引tensor(无填充,cpu类型),选择在评分之前删除BPE标识符(arg.remove_bpe为"@@ "),转义标记为True和额外要忽略的标识符索引eos_idx
                    target_str = tgt_dict.string(
                        target_tokens,
                        args.remove_bpe,
                        escape_unk=True,
                        extra_symbols_to_ignore={
                            generator.eos,
                        }
                    )  # 返回移除bpe符号的所遍历到的目标句子的原始token索引tensor经过对应转换后的string序列,与源端唯一区别在于由于转义符号为True,unk_idx返回"<<unk>>"

            src_str = decode_fn(src_str) # 由于未建立tokenizer和bpe类,故不进行操作
            if has_target: # 若存在目标句子:
                target_str = decode_fn(target_str) # 由于未建立tokenizer和bpe类,故不进行操作

            if not args.quiet:  # 若不是仅打印最终分数:
                if src_dict is not None:  # 若源和目标字典类(Dictionary)不为None,
                    # 打印S/T-句子对在该批中的编号 移除bpe符号的所遍历到的源/目标句子的原始token索引tensor经过对应转换后的string序列->到output_file(默认到窗口)
                    print('S-{}\t{}'.format(sample_id, src_str), file=output_file)
                if has_target:  # 若存在目标句子:
                    print('T-{}\t{}'.format(sample_id, target_str), file=output_file)

            # Process top predictions # 处理最可能的预测
            for j, hypo in enumerate(hypos[i][:args.nbest]): # 根据输出假设的数量,从该批次中当前遍历句子的beam_size个最终翻译结果中选择出1个输出结果(默认)
                # hypo:{
                #     'tokens': tokens_clone[i],  # 当前解码时间步的待生成目标词idx序列的副本
                #     'score': score,  # 当前解码时间步的归一化的概率lprobs最大且为eos的值;
                #     'attention': hypo_attn,　# src_len x tgt_len # 返回当前解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重的副本
                #     'alignment': None,
                #     'positional_scores': pos_scores[i],  # 当前解码时间步的每个假设的分数(lprobs值)
                # }
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(), # 当前解码时间步的待生成目标词idx序列的副本
                    src_str=src_str,   # 移除bpe符号的所遍历到的源句子的原始token索引tensor经过对应转换后的string序列
                    alignment=hypo['alignment'],  # 默认为None
                    align_dict=align_dict,   # 默认为None
                    tgt_dict=tgt_dict,     # 目标字典类(Dictionary)
                    remove_bpe=args.remove_bpe,  # 选择在评分之前删除BPE标识符(arg.remove_bpe为"@@ ")
                    extra_symbols_to_ignore={
                        generator.eos,       # 额外要忽略的标识符索引eos_idx
                    }
                )
                # 返回hypo_tokens->当前解码时间步的待生成目标词idx序列的副本;
                # hypo_str->移除bpe符号的当前解码时间步的待生成目标词idx序列的副本经过对应转换后的string序列,与目标端区别在于转义符号为False,和源端一样unk_idx返回"<unk>";None
                detok_hypo_str = decode_fn(hypo_str)  # 由于未建立tokenizer和bpe类,故不进行操作
                if not args.quiet:  # 若不是仅打印最终分数:
                    score = hypo['score'] / math.log(2)  # convert to base 2 # 从当前解码时间步的归一化的概率lprobs最大且为eos的值除以log2,返回基数为2的生成序列分数
                    # original hypothesis (after tokenization and BPE) # 原始假设（在标记化和BPE之后）
                    print('H-{}\t{}\t{}'.format(sample_id, score, hypo_str), file=output_file)
                    # 打印H-句子对在该批中的编号 基数为2的生成序列分数　移除bpe符号的当前解码时间步的待生成目标词idx序列的副本经过对应转换后的string序列->到output_file(默认到窗口)
                    # detokenized hypothesis　# 去标记化后的输出假设
                    print('D-{}\t{}\t{}'.format(sample_id, score, detok_hypo_str), file=output_file)
                    # 打印D-句子对在该批中的编号 基数为2的生成序列分数 (去标记化,默认无操作)移除bpe符号的当前解码时间步的待生成目标词idx序列的副本经过对应转换后的string序列->到output_file(默认到窗口)
                    print('P-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            # convert from base e to base 2
                            hypo['positional_scores'].div_(math.log(2)).tolist(),
                        ))
                    ), file=output_file)
                    # 打印P-句子对在该批中的编号 当前解码时间步的每个假设的分数(lprobs值)/除以log2 ->到output_file(默认到窗口)

                    if args.print_alignment:  # 若已设置,则使用注意力反馈来计算和打印对源tokens的对齐(对齐/注意力权重);非默认设置
                        print('A-{}\t{}'.format(
                            sample_id,
                            ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                        ), file=output_file)

                    if args.print_step:   # 打印每一个生成步骤;非默认设置
                        print('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                    if getattr(args, 'retain_iter_history', False): # 若设置,则解码返回迭代强化的全部历史记录;非默认设置
                        for step, h in enumerate(hypo['history']):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h['tokens'].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)

                # Score only the top hypothesis # 仅对最可能的假设打分
                if has_target and j == 0:  # 若存在目标句子,且从该批次中当前遍历句子的beam_size个最终翻译结果中选择出第一个输出结果(按'scores'从小到大排序过,越小越好)
                    if align_dict is not None or args.remove_bpe is not None: # 选择在评分之前删除BPE标识符(arg.remove_bpe为"@@ "不为None)
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE # 转换标识符,以使用unk替换或不使用BPE进行评估
                        target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        # 又将target_str->移除bpe符号的所遍历到的目标句子的原始token索引tensor经过对应转换后的string序列,
                        # 转换回当前遍历的batch中当前目标句子的tensor填充向量(1D张量,形状为1,tgt_len,移除unk_idx)
                    if hasattr(scorer, 'add_string'): # 创建的打分类中存在'add_string'属性
                        scorer.add_string(target_str, hypo_str)
                    else:   # 默认情况,将当前遍历的batch中当前目标句子的tensor填充向量(1D张量,形状为1,tgt_len,移除unk_idx)和当前解码时间步的待生成目标词idx序列的副本喂入,进行打分
                        scorer.add(target_tokens, hypo_tokens)

        wps_meter.update(num_generated_tokens)  # 统计该批次的总翻译tokens数,即self.n=20100
        progress.log({'wps': round(wps_meter.avg)}) # wps_meter.avg即self.n/self.elapsed_time,故'wps': 每翻译一个tokens所需要花费的时间s
        num_sentences += sample['nsentences']  # 更新到当前目标生成批次时所完成的句子数

    logger.info('NOTE: hypothesis and token scores are output in base 2') # 打印"注意：输出假设和tokens分数均以2为基数输出"
    logger.info('Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    # 打印"翻译所有目标生成批次时所需完成的句子数(总翻译tokens数)所用多少s(每s多少个句子,每s多少个tokens)"
    if has_target:  # 若存在目标句子:
        logger.info('Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
    # 打印"以beam=多少生成test,BLEU4 = 0.00, 0.0/0.0/0.0/0.0 (BP=1.000, ratio=5.833, syslen=18690, reflen=3204)"

    return scorer


def cli_main():
    parser = options.get_generation_parser()  # 根据特定的任务得到测试设置
    args = options.parse_args_and_arch(parser) # 将全部的arguments设置添加进args(Namespace类型)
    main(args)


if __name__ == '__main__':
    cli_main()
