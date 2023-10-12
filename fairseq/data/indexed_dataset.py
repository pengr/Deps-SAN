# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
import os
import shutil
import struct

import numpy as np
import torch

from . import FairseqDataset


def __best_fitting_dtype(vocab_size=None): # 若词汇表大小<2^(16),则采用uint16进行数字化即可
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def get_available_dataset_impl():  # 数据集的格式选择
    return ['backup', 'lazy', 'cached', 'mmap']


def infer_dataset_impl(path):
    if IndexedRawTextDataset.exists(path):  # 由IndexedRawTextDataset.exists判断path是否存在,存在即为纯文本,如'data-bin/iwslt14.tokenized.de-en/valid.de-en.de'
        return 'backup'
    elif IndexedDataset.exists(path):  # 根据path路径判断其对应的.idx和.bin文件是否存在,如'data-bin/iwslt14.tokenized.de-en/valid.de-en.de.bin/.idx'
        with open(index_file_path(path), 'rb') as f:  # 读取path路径对应的.idx文件
            magic = f.read(8) # 根据f.read(8),也即_HDR_MAGIC判断该path路径所选择的dataset_impl
            if magic == IndexedDataset._HDR_MAGIC:
                return 'cached'
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return 'mmap'
            else:
                return None
    else:  # 否则返回None
        return None


def make_builder(out_file, impl, vocab_size=None):  # 根据impl(最终数据集格式),选择创建数据集创建器
    if impl == 'mmap':
        return MMapIndexedDatasetBuilder(out_file, dtype=__best_fitting_dtype(vocab_size))
    else:
        return IndexedDatasetBuilder(out_file)


def make_dataset(path, impl, fix_lua_indexing=False, dictionary=None):  # 根据impl类型返回对应的Dataset类
    if impl == 'backup' and IndexedRawTextDataset.exists(path):
        assert dictionary is not None
        return IndexedRawTextDataset(path, dictionary)
    elif impl == 'lazy' and IndexedDataset.exists(path):
        return IndexedDataset(path, fix_lua_indexing=fix_lua_indexing)
    elif impl == 'cached' and IndexedDataset.exists(path):
        return IndexedCachedDataset(path, fix_lua_indexing=fix_lua_indexing)
    elif impl == 'mmap' and MMapIndexedDataset.exists(path):  # 完成对<fairseq.data.indexed_dataset.MMapIndexedDataset>类的初始化
        return MMapIndexedDataset(path)  # 完成对path路径下.idx和.bin文件的二进制读取,采用np.memmap()方法小段读入
    return None


def dataset_exists(path, impl):  # 根据路径path和输出数据集的格式impl,找对应数据集的exists()判断path下.idx和.bin文件是否存在
    if impl == 'backup':
        return IndexedRawTextDataset.exists(path)
    elif impl == 'mmap':
        return MMapIndexedDataset.exists(path)
    else:
        return IndexedDataset.exists(path)


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
    8: np.uint16
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):  # 根据路径的前缀得到.idx文件路径
    return prefix_path + '.idx'


def data_file_path(prefix_path):  # 根据路径的前缀得到.bin文件路径
    return prefix_path + '.bin'


class IndexedDataset(FairseqDataset):
    """Loader for TorchNet IndexedDataset"""
    _HDR_MAGIC = b'TNTIDX\x00\x00'

    def __init__(self, path, fix_lua_indexing=False):
        super().__init__()
        self.path = path
        self.fix_lua_indexing = fix_lua_indexing
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                'Index file doesn\'t match expected format. '
                'Make sure that --dataset-impl is configured properly.'
            )
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1,)
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code]
            self._len, self.s = struct.unpack('<QQ', f.read(16))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        if not self.data_file:
            self.read_data(self.path)
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        self.data_file.seek(self.data_offsets[i] * self.element_size)
        self.data_file.readinto(a)
        item = torch.from_numpy(a).long()
        if self.fix_lua_indexing:
            item -= 1  # subtract 1 for 0-based indexing
        return item

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):  # 根据path路径判断其对应的.idx和.bin文件是否存在
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory


class IndexedCachedDataset(IndexedDataset):

    def __init__(self, path, fix_lua_indexing=False):
        super().__init__(path, fix_lua_indexing=fix_lua_indexing)
        self.cache = None
        self.cache_index = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        if all(i in self.cache_index for i in indices):
            return
        if not self.data_file:
            self.read_data(self.path)
        indices = sorted(set(indices))
        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]
        self.cache = np.empty(total_size, dtype=self.dtype)
        ptx = 0
        self.cache_index.clear()
        for i in indices:
            self.cache_index[i] = ptx
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            a = self.cache[ptx: ptx + size]
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            ptx += size
        if self.data_file:
            # close and delete data file after prefetch so we can pickle
            self.data_file.close()
            self.data_file = None

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        ptx = self.cache_index[i]
        np.copyto(a, self.cache[ptx: ptx + a.size])
        item = torch.from_numpy(a).long()
        if self.fix_lua_indexing:
            item -= 1  # subtract 1 for 0-based indexing
        return item


class IndexedRawTextDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.lines.append(line.strip('\n'))
                tokens = dictionary.encode_line(
                    line, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):  # 判断path路径是否存在
        return os.path.exists(path)


class IndexedDatasetBuilder(object):
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float: 4,
        np.double: 8
    }

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, 'wb')
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]

    def add_item(self, tensor):
        # +1 for Lua compatibility
        bytes = self.out_file.write(np.array(tensor.numpy() + 1, dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def merge_file_(self, another_file):
        index = IndexedDataset(another_file)
        assert index.dtype == self.dtype

        begin = self.data_offsets[-1]
        for offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + offset)
        self.sizes.extend(index.sizes)
        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)

        with open(data_file_path(another_file), 'rb') as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, 'wb')
        index.write(b'TNTIDX\x00\x00')
        index.write(struct.pack('<Q', 1))
        index.write(struct.pack('<QQ', code(self.dtype), self.element_size))
        index.write(struct.pack('<QQ', len(self.data_offsets) - 1, len(self.sizes)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        index.close()


def _warmup_mmap_file(path):  # 事先读取一下path文件,确认无误
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00' # 设置_HDR_MAGIC标记来判别dataset_impl(输出数据格式)

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, 'wb')

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack('<Q', 1))
                    self._file.write(struct.pack('<B', code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):  # 存放每个句子的指针位置,如第1句:0,第2句:13*dtype_size(unint16->2),１3为句子长度
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes):
                    pointers = self._get_pointers(sizes)  # 存放每个句子的指针位置,

                    self._file.write(struct.pack('<Q', len(sizes)))

                    sizes = np.array(sizes, dtype=np.int32)  # 将带所有句子长度的列表转成np数组
                    self._file.write(sizes.tobytes(order='C'))  # 将np句子长度的数组写入输出数据集文件(.idx)文件内
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)  # 将存放每个句子的指针位置的列表转成np数组
                    self._file.write(pointers.tobytes(order='C'))  # 将存放每个句子的指针位置的np数组写入输出数据集文件(.idx)文件内
                    del pointers

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path):
            with open(path, 'rb') as stream:  # 读取path路径(此时为.idx文件)
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )  # 确保_HDR_MAGIC码为当前--dataset-impl所设置的
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version  # 确保version为当前--dataset-impl所设置的

                dtype_code, = struct.unpack('<B', stream.read(1))  # 取出path路径(此时为.idx文件的dtype_code,比如为8 -> <class 'numpy.uint16'>
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize  # 得到当前_dtype的itemsize,比如unint16则为2

                self._len = struct.unpack('<Q', stream.read(8))[0]  # 得到self._len
                offset = stream.tell()  # 返回stream文件写入的当前读取指针位置

            _warmup_mmap_file(path)  # 事先读取一下path文件,确认无误

            # np.memmap():将大二进制数据文件当做内存中的数组进行处理,允许将大文件分成小段进行读写,而不一次性将整个数组读入内存
            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)  # memoryview返回给定参数的内存查看对象,这里返回为path中.idx文件写入类对象
            # np.frombuffer将缓冲区解释为一维数组,即self._bin_buffer[offset:offset+self._len]
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset) # 统计每个句子的idx长度,也即句子对应的IntTensor长度(+eos的句长)
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes) # 定义当前读取指针向量,self._bin_buffer[offset+ self._sizes.nbytes:offset+ self._sizes.nbytes+self._len]

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path):  # 完成对<fairseq.data.indexed_dataset.MMapIndexedDataset>类的初始化
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None
        self._do_init(path)  # self._do_init()进行真正初始化,完成对.idx和.bin文件的二进制读取,采用np.memmap()方法小段读入

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path):
        self._path = path   # 导入数据的路径
        self._index = self.Index(index_file_path(self._path))  # 用数据路径对应的.idx文件来初始化Index类(记录数据集中全部+eos的句长),采用"rb"二进制来读取

        _warmup_mmap_file(data_file_path(self._path))  # 事先读取一下数据路径对应的.bin文件,确认无误
        # np.memmap():将大二进制数据文件当做内存中的数组进行处理,允许将大文件分成小段进行读写,而不一次性将整个数组读入内存
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap) # memoryview返回给定参数的内存查看对象,这里返回为path中.bin文件写入类对象

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        ptr, size = self._index[i]
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
        if self._index.dtype != np.int64:
            np_array = np_array.astype(np.int64)

        return torch.from_numpy(np_array)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):  # 根据path路径判断其对应的.idx和.bin文件是否存在
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )


class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, 'wb') # 开始对输出数据集的写入操作
        self._dtype = dtype  # 数字化输出数据集采用的数据类型
        self._sizes = []     # 存储所有句子的句子长度

    def add_item(self, tensor):  # 将导入的tensor(即由当前句子各单词的idx组成的IntTensor)转成np数据,写入输出数据集内,由self._sizes记录该数组长度(即句子长度)
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()  # 关闭先前打开的输出数据集写入类(.bin)

        # 采用MMapIndexedDataset.Index.writer()._Writer类,对当前输出数据集(.idx)进行写入操作
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes)
