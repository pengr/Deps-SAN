# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:  # 若原始目标序列target的维数比log_softmax(logits)少1,则将其扩充1个维度
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)  # y_hot损失,即预测正确的目标词的loss->-log(pt)_i=t　
    # 原理:根据是target寻找-lprobs里面对应的索引位置,由于两者均为二维向量,且dim=-1(即1); 形状为[bacth*tgt_len,1]
    #     故nll_loss[i][j] = -lprobs[i][target[i][j]],其中i为[0,bacth*tgt_len],j为0;
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)  # 交叉熵损失,即全部目标词的loss(无论是否预测正确)->-log(pt)
    # 将-lprobs按照最后一个维度累加,并保留维数; 形状为[bacth*tgt_len,1]
    if ignore_index is not None:  # 若填充符号idx不为None:
        pad_mask = target.eq(ignore_index)   # 获取原始目标序列的填充符号掩码矩阵pad_mask
        nll_loss.masked_fill_(pad_mask, 0.)  # 给填充符号的nll_loss(y_hot损失)和smooth_loss(交叉熵损失)均设置为0
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:  # 若reduce标记为True,则将该批次所有单词对应的nll_loss(y_hot损失)和smooth_loss(交叉熵损失)进行累加
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)  # 完成公式ε/K(用于分类的类别总数,即目标词汇表大小)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss  # 该批次所有目标词对应的lable_smoothing loss,即-(1-ε)*log(pt)_i=t - (ε/K)*log(pt)
    return loss, nll_loss  # 返回该批次所有目标词对应的lable_smoothing loss,以及nll_loss(y_hot损失)


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):  # 由task,senetence_avg,laberl_smoothing完成对LabelSmoothedCrossEntropyCriterion类的初始化
        super().__init__(task)  # 调用父类FairseqCriterion,加载task和目标字典中的pad符号idx
        self.sentence_avg = sentence_avg  # 获取sentence_avg标记
        self.eps = label_smoothing        # 获取标签平滑值

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')  # 标签平滑值,0表示不进行标签平滑
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample. # 计算给定sample的损失

        Returns a tuple with three elements:
        1) the loss  # 损失
        2) the sample size, which is used as the denominator for the gradient  # sample大小,用作梯度的分母
        3) logging outputs to display while training  # 训练时用于显示的日志输出
        """
        net_output = model(**sample['net_input'])
        # 返回经过vaswani论文"Linear层"(即目标端词嵌入矩阵)的解码器输出,
        # 字典"attn"-> 最后一个解码器层enc-dec self-attn的输出attn_weights在全部注意力头上平均注意力权重->softmax((QK^T)/dk^(-0.5)),
        # "inner_states"-> 存储每一层解码器的输出列表

        # 通过模型,以及模型的输出(经过"Linear层"的解码器输出,最后一个解码器层enc-dec self-attn的输出attn_weights在全部注意力头上平均注意力权重,存储每一层解码器的输出列表)
        # 以及当前批次的数据元组得到计算当前批次sample的损失; # 返回当前批次所有目标词对应的lable_smoothing loss,以及nll_loss(y_hot损失)
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens'] # 若sentence_avg标记为true,则sample大小为句子数,否则为tokens数
        logging_output = {
            'loss': loss.data,          # 当前批次所有目标词对应的lable_smoothing loss
            'nll_loss': nll_loss.data,  # 当前批次所有目标词对应的nll_loss(y_hot损失)
            'ntokens': sample['ntokens'], # 当前批次目标tokens数
            'nsentences': sample['target'].size(0), # 当前批次目标句子数
            'sample_size': sample_size, # 若sentence_avg标记为true,则sample大小为句子数,否则为tokens数
        }
        return loss, sample_size, logging_output  # 返回当前批次所有目标词对应的lable_smoothing loss,当前批次目标tokens数,以及输出记录元组

    def compute_loss(self, model, net_output, sample, reduce=True):
        # 通过模型,以及模型的输出(经过"Linear层"的解码器输出,最后一个解码器层enc-dec self-attn的输出attn_weights在全部注意力头上平均注意力权重,存储每一层解码器的输出列表),当前批次的数据元组得到计算当前批次sample的损失
        lprobs = model.get_normalized_probs(net_output, log_probs=True)  # 从模型的输出中获取归一化的概率(或对数概率)
        # 若选择返回log概率, 则对经过"Linear层"的解码器输出logits进行log_softmax(经过vaswani论文的"Softmax"),否则对其进行softmax
        lprobs = lprobs.view(-1, lprobs.size(-1)) # 对log_softmax(logits),从[bacth,tgt_len,tgt_vocab_size](目标端词汇表大小,注与源端共用)->[bacth*tgt_len,tgt_vocab_size]
        target = model.get_targets(sample, net_output).view(-1, 1)  # 返回sample字典中"target"对应的目标序列,并从[bacth,tgt_len]->[bacth*tgt_len,1]的列向量
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        ) # 将log_softmax(logits)->形状[bacth*tgt_len,tgt_vocab_size],原始目标序列target->形状([bacth*tgt_len,1]),
          # 标签平滑值eps,以及填充符号idx,以及reduce标记喂入label_smoothed_nll_loss中,
        return loss, nll_loss  # 返回当前批次所有目标词对应的lable_smoothing loss,以及nll_loss(y_hot损失)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training. 汇总来自数据并行训练的输出记录"""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs) # 从logging_outputs中取出当前批次所有目标词对应的lable_smoothing loss
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs) # 从logging_outputs中取出当前批次所有目标词对应的nll_loss(y_hot损失)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs) # 从logging_outputs中取出当前批次目标tokens数
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs) # 从logging_outputs中取出当前批次目标tokens数

        # 将"loss":当前批次平均目标词对应的－lable_smoothing loss/log(2),以及优先级,由log_scalar添加到MetersDict中实时记录并更新
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        # 将"nll_loss":当前批次平均目标词对应的－nll_loss(y_hot损失)/log(2),以及优先级,由log_scalar添加到MetersDict中实时记录并更新
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        # 从meters字典取出当前批次平均目标词对应的－nll_loss(y_hot损失),返回2^(－nll_loss)作为模型的ppl
        # 将"ppl":2^(－nll_loss)模型的ppl,以及优先级,由log_scalar添加到MetersDict中实时记录并更新
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
