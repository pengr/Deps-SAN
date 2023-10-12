# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import math
import torch

try:
    from fairseq import libbleu
except ImportError as e:
    import sys
    sys.stderr.write('ERROR: missing libbleu.so. run `pip install --editable .`\n')
    raise e


C = ctypes.cdll.LoadLibrary(libbleu.__file__) # 在python里loadlibrary方式调用c,加载fairseq/libbleu.cpython-36m-x86_64-linux-gnu.so


class BleuStat(ctypes.Structure):
    _fields_ = [
        ('reflen', ctypes.c_size_t),
        ('predlen', ctypes.c_size_t),
        ('match1', ctypes.c_size_t),
        ('count1', ctypes.c_size_t),
        ('match2', ctypes.c_size_t),
        ('count2', ctypes.c_size_t),
        ('match3', ctypes.c_size_t),
        ('count3', ctypes.c_size_t),
        ('match4', ctypes.c_size_t),
        ('count4', ctypes.c_size_t),
    ]


class SacrebleuScorer(object):
    def __init__(self):
        import sacrebleu
        self.sacrebleu = sacrebleu
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            raise NotImplementedError
        self.ref = []
        self.sys = []

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.sys.append(pred)

    def score(self, order=4):
        return self.result_string(order).score

    def result_string(self, order=4):
        if order != 4:
            raise NotImplementedError
        return self.sacrebleu.corpus_bleu(self.sys, [self.ref])


class Scorer(object):
    def __init__(self, pad, eos, unk):
        self.stat = BleuStat() # 加载一个BleuStat类,其中主要为ctype类型的结构体,记录计算BLEU分需要的reflen,predlen,match1~4,count1~4
        self.pad = pad  # 加载目标字典的pad/unk/eos符号对应idx
        self.eos = eos
        self.unk = unk
        self.reset() # 默认ctypes.byref由C的指针定位到BleuStat类,将内部值初始化为0

    def reset(self, one_init=False):
        if one_init:
            C.bleu_one_init(ctypes.byref(self.stat))
        else: # 默认ctypes.byref由C的指针定位到BleuStat类,将内部值初始化为0
            C.bleu_zero_init(ctypes.byref(self.stat))

    def add(self, ref, pred): # 默认情况,将当前遍历的batch中当前目标句子的tensor填充向量(1D张量,形状为1,tgt_len,移除unk_idx)和当前解码时间步的待生成目标词idx序列的副本喂入,进行打分
        if not isinstance(ref, torch.IntTensor): # 喂入的target_tokens, hypo_tokens必须为IntTensor类型
            raise TypeError('ref must be a torch.IntTensor (got {})'
                            .format(type(ref)))
        if not isinstance(pred, torch.IntTensor):
            raise TypeError('pred must be a torch.IntTensor(got {})'
                            .format(type(pred)))

        # don't match unknown words
        rref = ref.clone()  # 将当前遍历的batch中当前目标句子的tensor填充向量(1D张量,形状为1,tgt_len,移除unk_idx)创建一个副本
        assert not rref.lt(0).any() # 确保当前遍历的batch中当前目标句子的tensor填充向量(1D张量,形状为1,tgt_len,移除unk_idx)副本的全部元素>=0
        rref[rref.eq(self.unk)] = -999  # 将当前遍历的batch中当前目标句子的tensor填充向量(1D张量,形状为1,tgt_len,移除unk_idx)副本中unk_idx设置为-999

        rref = rref.contiguous().view(-1) # 将当前遍历的batch中当前目标句子的tensor填充向量(1D张量,形状为1,tgt_len,移除unk_idx)展平
        pred = pred.contiguous().view(-1) # 当前解码时间步的待生成目标词idx序列的副本展平

        C.bleu_add(
            ctypes.byref(self.stat),
            ctypes.c_size_t(rref.size(0)),
            ctypes.c_void_p(rref.data_ptr()),
            ctypes.c_size_t(pred.size(0)),
            ctypes.c_void_p(pred.data_ptr()),
            ctypes.c_int(self.pad),
            ctypes.c_int(self.eos)) # 由C脚本计算bleu分,/fairseq/libbleu.cpython-36m-x86_64-linux-gnu.so

    def score(self, order=4):
        psum = sum(math.log(p) if p > 0 else float('-Inf')
                   for p in self.precision()[:order])
        return self.brevity() * math.exp(psum / order) * 100

    def precision(self):
        def ratio(a, b):
            return a / b if b > 0 else 0

        return [
            ratio(self.stat.match1, self.stat.count1),
            ratio(self.stat.match2, self.stat.count2),
            ratio(self.stat.match3, self.stat.count3),
            ratio(self.stat.match4, self.stat.count4),
        ]

    def brevity(self):
        r = self.stat.reflen / self.stat.predlen
        return min(1, math.exp(1 - r))

    def result_string(self, order=4):
        assert order <= 4, "BLEU scores for order > 4 aren't supported"
        fmt = 'BLEU{} = {:2.2f}, {:2.1f}'
        for _ in range(1, order):
            fmt += '/{:2.1f}'
        fmt += ' (BP={:.3f}, ratio={:.3f}, syslen={}, reflen={})'
        bleup = [p * 100 for p in self.precision()[:order]]
        return fmt.format(order, self.score(order=order), *bleup,
                          self.brevity(), self.stat.predlen/self.stat.reflen,
                          self.stat.predlen, self.stat.reflen)
