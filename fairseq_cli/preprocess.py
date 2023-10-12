#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

from collections import Counter
from itertools import zip_longest
import logging
from multiprocessing import Pool
import os
import shutil
import sys

from fairseq import options, tasks, utils
from fairseq.data import indexed_dataset
from fairseq.binarizer import Binarizer

# define a logger
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.preprocess')


def main(args):
    utils.import_user_module(args)  # load additional custom modules if "--user-dir" isn't None

    os.makedirs(args.destdir, exist_ok=True)  # confirm that the output directory exists

    logger.addHandler(logging.FileHandler(
        filename=os.path.join(args.destdir, 'preprocess.log'),
    ))                 # store all arguments into destdir/preprocess.log
    logger.info(args)  # print all arguments

    task = tasks.get_task(args.task)  # take out the python file and class by args.task

    def train_path(lang):  # 源/目标语言的训练集存放路径
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang): # 源/目标语言的最终输出文件存放路径
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):  # 源/目标语言的字典存放路径
        return dest_path("dict", lang) + ".txt"

    # filenames is the train.src/tgt path, and src/tgt=True determines if the src-side or tgt-side is processed
    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt  # OR operator
        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )  # 得到最终的字典(nwords个单词),由特殊符号开头,去除词频阈值以下的单词,并填充到可被padding_factor整除

    target = not args.only_source  # not only process the source language

    # If the src/tgt dictionary path is not set, and the src/tgt dictionary text dict.lang.txt exists, report FileExistsError
    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))
    if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))

    # Generate joined dictionary
    if args.joined_dictionary:
        assert not args.srcdict or not args.tgtdict, \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"
        # 若事先给定好srcdict或tgtdict,则直接加载
        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:  # 若未事先给定src_dict或tgt_dict,则靠源和目标数据集进行生成
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                {train_path(lang) for lang in [args.source_lang, args.target_lang]}, src=True
            )  # joined字典与单个源/目标字典区别仅在,joined将源和目标文本以及单词放在一起处理
        tgt_dict = src_dict  # 源和目标字典都是joined dictionary
    else:
        # Generate source and target dictionary
        # if source and target dictionary are given, choose it
        # else build_dictionary from train.src and train.tgt
        # 得到最终的源/目标字典(nwords个单词),由特殊符号开头,去除词频阈值以下的单词,并填充到可被padding_factor整除
        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary([train_path(args.source_lang)], src=True)

        if target:
            if args.tgtdict:
                tgt_dict = task.load_dictionary(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary([train_path(args.target_lang)], tgt=True)
        else:
            tgt_dict = None

    src_dict.save(dict_path(args.source_lang))  # 存放源/目标语言的字典
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args.target_lang))

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers): # input_prefix为训练/验证/测试集共用的前缀,output_prefix为"train/vaild/test"
        logger.info("[{}] Dictionary: {} types".format(lang, len(vocab) - 1))  # 打印源字典的符号数-1
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):  # worker_result -> 字典{句子总数,索引为unk_index的单词数,单词总数,以及存储索引为unk_index的单词的计数器}
            replaced.update(worker_result["replaced"])  # 将存储索引为unk_index的单词的计数器同步到replaced计数器
            n_seq_tok[0] += worker_result["nseq"]  # 记录该数据集的句子总数和单词总数
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )  # 通过前缀遍历到需要处理的文本
        offsets = Binarizer.find_offsets(input_file, num_workers)  # 将总文本分成num_workers片交给每个进程数去处理
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                    callback=merge_result
                )
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"), # 根据impl(最终数据集格式),选择创建数据集创建器
                                          impl=args.dataset_impl, vocab_size=len(vocab))  # dataset_dest_file用于获得最终数据数据集的完整文件路径(全部句子的np数组)
        merge_result(
            Binarizer.binarize(
                input_file, vocab, lambda t: ds.add_item(t),
                offset=0, end=offsets[1]
            )
        )  # 先定义静态函数binarize(不要求强制传入参数),然后确定binarize中consumer参数,最后进行merge_result
           # 前两步已经完成了对原始数据集的数字化处理,并写入到输出数据集的路径下, 后一步将存储索引为unk_index的单词的计数器同步到replaced计数器,记录该数据集的句子总数和单词总数
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))  # dataset_dest_file用于获得最终数据数据集的文件完整路径(全部句子的)

        logger.info(
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )  # 打印当前语言,当前处理数据集的前缀,句子总数,单词总数,unk单词占单词总数的比例

    def make_binary_alignment_dataset(input_prefix, output_prefix, num_workers):
        nseq = [0]

        def merge_result(worker_result):
            nseq[0] += worker_result['nseq']

        input_file = input_prefix
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize_alignments,
                    (
                        args,
                        input_file,
                        utils.parse_alignment,
                        prefix,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                    callback=merge_result
                )
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, None, "bin"),
                                          impl=args.dataset_impl)

        merge_result(
            Binarizer.binarize_alignments(
                input_file, utils.parse_alignment, lambda t: ds.add_item(t),
                offset=0, end=offsets[1]
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, None)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))

        logger.info(
            "[alignments] {}: parsed {} alignments".format(
                input_file,
                nseq[0]
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args.dataset_impl == "backup":  # 若输出数据集格式为raw,则直接将preprocess.sh预处理的文本复制为输出数据集
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:  # 若输出数据集格式不为raw,则需要进行数字化处理得到输出数据集
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)

    def make_all(lang, vocab):  # 处理源/目标语言下的训练集,全部验证集,全部测试集
        if args.trainpref:  # 通过源字典和源训练集路径预处理得到训练集
            make_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers)
        if args.validpref: # 通过源字典和源验证集路径预处理得到全部源验证集
            for k, validpref in enumerate(args.validpref.split(",")):  # 遍历所有的验证集(即以相同前缀如'xxx/valid')
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix, lang, num_workers=args.workers)
        if args.testpref:  # 通过源字典和源测试集路径预处理得到全部源测试集
            for k, testpref in enumerate(args.testpref.split(",")):  # 遍历所有的测试集(即以相同前缀如'xxx/test')
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args.workers)

    def make_all_alignments():
        if args.trainpref and os.path.exists(args.trainpref + "." + args.align_suffix):
            make_binary_alignment_dataset(args.trainpref + "." + args.align_suffix, "train.align", num_workers=args.workers)
        if args.validpref and os.path.exists(args.validpref + "." + args.align_suffix):
            make_binary_alignment_dataset(args.validpref + "." + args.align_suffix, "valid.align", num_workers=args.workers)
        if args.testpref and os.path.exists(args.testpref + "." + args.align_suffix):
            make_binary_alignment_dataset(args.testpref + "." + args.align_suffix, "test.align", num_workers=args.workers)

    make_all(args.source_lang, src_dict)  # 处理源/目标语言下的训练集,全部验证集,全部测试集
    if target:
        make_all(args.target_lang, tgt_dict)
    if args.align_suffix:  # 若对齐文件的后缀不为空:
        make_all_alignments()

    logger.info("Wrote preprocessed data to {}".format(args.destdir))  # 打印将preprocess.py预处理后的数据写入destdir(目标输出目录)下

    if args.alignfile:  # 若对齐文件不为空:
        assert args.trainpref, "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args.source_lang)
        tgt_file_name = train_path(args.target_lang)
        freq_map = {}
        with open(args.alignfile, "r", encoding='utf-8') as align_file:
            with open(src_file_name, "r", encoding='utf-8') as src_file:
                with open(tgt_file_name, "r", encoding='utf-8') as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = src_dict.encode_line(s, add_if_not_exist=False)
                        ti = tgt_dict.encode_line(t, add_if_not_exist=False)
                        ai = list(map(lambda x: tuple(x.split("-")), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(freq_map[srcidx], key=freq_map[srcidx].get)

        with open(
                os.path.join(
                    args.destdir,
                    "alignment.{}-{}.txt".format(args.source_lang, args.target_lang),
                ),
                "w", encoding='utf-8'
        ) as f:
            for k, v in align_dict.items():
                print("{} {}".format(src_dict[k], tgt_dict[v]), file=f)


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                      impl=args.dataset_impl, vocab_size=len(vocab))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos,
                             offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def binarize_alignments(args, filename, parse_alignment, output_prefix, offset, end):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, None, "bin"),
                                      impl=args.dataset_impl, vocab_size=None)

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_alignments(filename, parse_alignment, consumer, offset=offset,
                                        end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, lang):  # 用于获得最终输出数据集的前缀
    base = "{}/{}".format(args.destdir, output_prefix)
    if lang is not None:
        lang_part = ".{}-{}.{}".format(args.source_lang, args.target_lang, lang)
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = ".{}-{}".format(args.source_lang, args.target_lang)

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):  # base为最终输出数据集的前缀,extension为最终输出数据集的文件扩展名,如"bin"
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def get_offsets(input_file, num_workers):
    return Binarizer.find_offsets(input_file, num_workers)


def cli_main():
    parser = options.get_preprocessing_parser()  # get preprocess configuration according to specified task
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
