# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter

from fairseq.tokenizer import tokenize_line
import torch


def safe_readline(f): # 稳定的readline,先获得当前文件读取指针位置,然后再进行readline()
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError: # 若出现读取错误,则返回并读取当前读取位置的上一行
            pos -= 1
            f.seek(pos)  # search where this character begins


class Binarizer:  # 携带静态函数的类可以不实例化,直接调用如Binarizer.binarize()
    @staticmethod
    def binarize(
        filename,
        dict,
        consumer,
        tokenize=tokenize_line,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
        already_numberized=False,
    ):
        nseq, ntok = 0, 0  # 用于记录总句子数和总token数
        replaced = Counter()

        def replaced_consumer(word, idx):  # 若单词不是<unk>,但对应的idx为unk_index(3),则由replaced计数器进行记录
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, "r", encoding="utf-8") as f: # 遍历preprocess.sh得到的原始源训练集
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:  # 若当前行不为空:
                if end > 0 and f.tell() > end:  # 若设置终止读取行且当前读取位置到达了终止行,则break
                    break
                if already_numberized:  # 若当前读取的原始训练集已数字化处理,则对每行的单词切片,数字化"3"等,添加<eos>标识符,最后转化成torch.IntTensor
                    id_strings = line.strip().split()
                    id_list = [int(id_string) for id_string in id_strings]
                    if reverse_order:
                        id_list.reverse()
                    if append_eos:
                        id_list.append(dict.eos())
                    ids = torch.IntTensor(id_list)
                else:  # 返回当前句子数字化处理后的IntTensor,其中数字为其字典的idx
                    ids = dict.encode_line(
                        line=line,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                    )
                nseq += 1
                ntok += len(ids)
                consumer(ids)  # consumer为MMapIndexedDatasetBuilder或者IndexedDatasetBuilder的add_item(),# 将导入的tensor(即由当前句子各单词的idx组成的IntTensor)转成np数据,写入输出数据集内,由self._sizes记录该数组长度(即句子长度)
                line = f.readline()  # 读取下一个句子
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }  # 返回字典{句子总数,索引为unk_index的单词数,单词总数,以及存储索引为unk_index的单词的计数器}

    # <<改写>>, 增加输入矩阵文本的后缀,便于添加sdsa和pascal矩阵;
    @staticmethod
    def binarize_matrixs(filename, matrix_parser, consumer, offset=0, end=-1):
        nseq = 0

        with open(filename, "r") as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = matrix_parser(line)
                nseq += 1
                consumer(ids)
                line = f.readline()
        return {"nseq": nseq}

    @staticmethod
    def binarize_alignments(filename, alignment_parser, consumer, offset=0, end=-1):
        nseq = 0

        with open(filename, "r") as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = alignment_parser(line)
                nseq += 1
                consumer(ids)
                line = f.readline()
        return {"nseq": nseq}

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets
