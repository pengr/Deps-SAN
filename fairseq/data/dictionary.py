# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter
from multiprocessing import Pool

import torch
from fairseq.binarizer import safe_readline
from fairseq.data import data_utils
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        bos="<s>",
        extra_special_symbols=None,
    ):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)  # 拿到四种特殊符号的索引idx
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:  # <我们方法>,若有额外的特殊符号,则在此添加,与前4种一样
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)  # 通过Dictionary初始化,确定出特殊符号总数

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(self, tensor, bpe_symbol=None, escape_unk=False, extra_symbols_to_ignore=None):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        # 用于将token索引的张量转换为字符串的辅助函数,可以选择性地移除BPE符号或避开<UNK>单词
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:  # 处理所遍历到的源/目标句子的原始token索引tensor(无填充,cuda/cpu类型),若其为2D张量;非默认情况
            return "\n".join(self.string(t, bpe_symbol, escape_unk) for t in tensor)

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])  # 额外要忽略的标识符索引集合,初始化为set{}
        extra_symbols_to_ignore.add(self.eos())  # 添加eos_idx到额外要忽略的标识符索引集合中

        def token_string(i):
            if i == self.unk(): # 若当前访问的token索引为unk_idx,返回"<<unk>>"(转义符号为True)或"<unk>"(转义符号为False)
                return self.unk_string(escape_unk)
            else:   # 若当前访问的token索引不为unk_idx,则调用当前源/目标字典的symbols列表,返回该token索引对应的token
                return self[i]

        if hasattr(self, "bos_index"): # 默认情况,若当前源/目标字典存在"bos_index"属性：
            extra_symbols_to_ignore.add(self.bos()) # 添加bos_idx到额外要忽略的标识符索引集合中
            sent = " ".join(
                token_string(i)
                for i in tensor if i.item() not in extra_symbols_to_ignore
            )  # 逐个访问所遍历到的源/目标句子的原始token索引tensor中的元素,若token索引不在额外要忽略的标识符索引集合{eos_idx(1),bos_idx(0)}中,则将该token索引转为对应的string
        else:
            sent = " ".join(token_string(i) for i in tensor if i.item() not in extra_symbols_to_ignore)
            # 逐个访问所遍历到的源/目标句子的原始token索引tensor中的元素,若token索引不在额外要忽略的标识符索引集合{eos_idx(1),bos_idx(0)}中,则将该token索引转为对应的string
        # 若设置--remove_bpe,则arg.remove_bpe即bpe_symbol为"@@ ",
        return data_utils.process_bpe_symbol(sent, bpe_symbol) # 返回移除bpe符号的所遍历到的源/目标句子的原始token索引tensor经过对应转换后的string序列

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>> # 返回unknown字符串,可选择选择转义为：<<unk>>"""
        if escape: # 若转义符号为True,则返回"<<unk>>"
            return "<{}>".format(self.unk_word)
        else:   # 若转义符号为False,则返回"<unk>"
            return self.unk_word

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary 添加一个单词以及其词频到字典中"""
        # indices为{word:idx}的字典,count用于统计当前字典已有单词数的词频,symbols用于存储当前字典已有单词
        # 第一个if: 若word在当前字典内且未被覆写,则取出该单词的idx,在count中增加一次词频
        # 第二个if: 若word不在当前字典内,则根据symbols推算出当前为第idx个单词,添加进{word:idx},symbols,counts
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:  # 若未设置词汇表大小,则定义其为训练集所出现的单词总数
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))  # 取出特殊符号和索引,建立一个新的字典
        new_symbols = self.symbols[: self.nspecial]  # 取出特殊符号组成的列表
        new_count = self.count[: self.nspecial]      # 取出特殊符号对应词频

        c = Counter(
            dict(
                sorted(zip(self.symbols[self.nspecial :], self.count[self.nspecial :]))
            )
        )  # 忽略特殊符号所得到的单词及其词频的计数器
        for symbol, count in c.most_common(nwords - self.nspecial): # 实际上即按单词词频来遍历c中的全部单词
            if count >= threshold:  # 若单词词频高于所设最低词频阈值
                new_indices[symbol] = len(new_symbols)  # 通过当前单词列表长度推算出当前单词对应的idx,将当前单词:idx添加到字典中,字典前几项仍为特殊符号
                new_symbols.append(symbol)  # 将当前单词添加进单词列表中,列表前几项仍为特殊符号
                new_count.append(count)     # 将当前单词对应词频添加进单词词频列表中,列表前几项仍为特殊符号的词频
            else:     # 若单词词频低于所设最低词频阈值,则不计入单词表(unknown token)
                break

        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count) # 将词频列表,单词列表,以及最终字典self化
        self.symbols = list(new_symbols)
        self.indices = new_indices

        self.pad_to_multiple_(padding_factor)

    def pad_to_multiple_(self, padding_factor):
        """Pad Dictionary size to be a multiple of *padding_factor*."""
        if padding_factor > 1:
            i = 0
            while len(self) % padding_factor != 0:  # 若词汇表大小不能整除padding_factor,填充{'madeupword000i':i}i个单词和索引,词频均为0
                symbol = "madeupword{:04d}".format(i)
                self.add_symbol(symbol, n=0)
                i += 1

    def bos(self):  # 返回bos符号对应idx
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):  # 返回pad符号对应idx
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):  # 返回eos符号对应idx
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):  # 返回unk符号对应idx
        """Helper to get index of unk symbol"""
        return self.unk_index

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format: 从以下格式的文本文件加载字典：<symbol0> <count0>等(即dict.en/de.txt)

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f) # 从文本文件(即dict.en/de.txt,文本格式为:<symbol0> <count0>)加载预先存在的字典,并将其符号添加到当前实例(Dictionary)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance. 从文本文件(即dict.en/de.txt,文本格式为:<symbol0> <count0>)加载预先存在的字典,并将其符号添加到当前实例(Dictionary)
        """
        if isinstance(f, str): # 若f为文件名:
            try:
                with PathManager.open(f, "r", encoding="utf-8") as fd:  # 由文件名为f创建文件读取类
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines() # 获得当前文件的所有行
        indices_start_line = self._load_meta(lines) # 返回0作为开始行号

        for line in lines[indices_start_line:]: # 遍历当前文件的每一行,如:", 2877"
            try:
                line, field = line.rstrip().rsplit(" ", 1) # 获得清除空白,以及空格分片的字符列表
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False  # 默认覆写标记为False
                count = int(field)  # 获得当前单词的字频
                word = line  # 获得当前单词
                if word in self and not overwrite:  # 若为重复的单词,非默认情况:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file."
                        .format(word)
                    )
                self.add_symbol(word, n=count, overwrite=overwrite) # 添加一个单词以及其词频到字典中
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )

    def _save(self, f, kv_iterator):
        if isinstance(f, str):
            PathManager.mkdirs(os.path.dirname(f))
            with PathManager.open(f, "w", encoding="utf-8") as fd:
                return self.save(fd)
        for k, v in kv_iterator:
            print("{} {}".format(k, v), file=f)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(
            f,
            zip(
                ex_keys + self.symbols[self.nspecial :],
                ex_vals + self.count[self.nspecial :],
            ),
        )

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t

    def encode_line(
        self,
        line,
        line_tokenizer=tokenize_line,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ):
        words = line_tokenizer(line) # 首先得到该句子的标记化token列表　
        if reverse_order:  # 若反向句子:
            words = list(reversed(words))
        nwords = len(words)  # 统计句子长度(不带eos)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)  # 若加eos,则创建一个维度为nwords+1的随机的IntTensor

        for i, word in enumerate(words): # 遍历标记化token列表
            if add_if_not_exist:  # 若该单词不在字典内,则添加其进入词汇表,词频为1,idx为全部词汇数+1
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)  # 若该单词在字典中,则取出其对应的idx
            if consumer is not None: # 即binarize().replaced_consumer()
                consumer(word, idx)
            ids[i] = idx  # 用当前单词对应的idx替换随机IntTensor其对应位置上的数
        if append_eos: # 将IntTensor最后一个数定义为eos_index
            ids[nwords] = self.eos_index
        return ids  # 返回当前句子数字化处理后的IntTensor,其中数字为其字典的idx

    @staticmethod
    def _add_file_to_dictionary_single_worker(
        filename, tokenize, eos_word, worker_id=0, num_workers=1
    ):  # 主要统计单词和词频的函数
        counter = Counter()  # 初始化一个计数器
        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:  # 获取相对路径
            size = os.fstat(f.fileno()).st_size  # 分配每一条进程所需处理的行数
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)      # 移动文件读取指针到当前进程所开始处理的行数
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:  # 由计数器记录数据集中所有非空句子的单词(每句均带eos符号)以及对应词频
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:  # 返回当前文件指针在文件中的位置,若超过当前进程所需处理的最后行则break
                    break
                line = f.readline()  # 循环读取
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(counter):
            # 将数据集中所有非空句子的单词(每句均带eos符号)以及对应词频,按单词的unicode码顺序遍历
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)  # 添加当前遍历单词和词频到源/目标字典中

        if num_workers > 1:  # 若采用多进程处理:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        Dictionary._add_file_to_dictionary_single_worker,
                        (filename, tokenize, dict.eos_word, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                Dictionary._add_file_to_dictionary_single_worker(
                    filename, tokenize, dict.eos_word
                )  # 返回一个计数器,记录了数据集中所有非空句子的单词(每句均带eos符号)以及对应词频
            )


class TruncatedDictionary(object):
    def __init__(self, wrapped_dict, length):
        self.__class__ = type(
            wrapped_dict.__class__.__name__,
            (self.__class__, wrapped_dict.__class__),
            {},
        )
        self.__dict__ = wrapped_dict.__dict__
        self.wrapped_dict = wrapped_dict
        self.length = min(len(self.wrapped_dict), length)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < self.length:
            return self.wrapped_dict[i]
        return self.wrapped_dict.unk()
