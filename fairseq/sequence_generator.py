# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder


class SequenceGenerator(object):
    def __init__(
        self,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        temperature=1.,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None
    ):
        """Generates translations of a given source sentence. 生成给定源句子的翻译

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary # 目标端字典Dictionary类
            beam_size (int, optional): beam width (default: 1)   # 波束搜索的波束大小
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length   # 生成最大长度为ax + b的序列, 其中x为源句子长度(a默认为0,b默认为200)
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)  # 最小生成序列长度
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)  # 通过输出假设的长度归一化分数
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)  # 长度惩罚α：<1.0有利于较短的句子,>1.0有利于较长的句子
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)  # 未知词惩罚：<0产生更多unks，>0产生更少unks
            retain_dropout (bool, optional): use dropout when generating　
                (default: False)        # 生成时仍使用dropout
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)  # 用于预测过程的温度,>1.0会产生更均匀的样本,<1.0会产生更极端的样本
            match_source_len (bool, optional): outputs should match the source
                length (default: False)  # 生成的预测翻译序列应该匹配源长度
        """
        self.pad = tgt_dict.pad() # 加载目标字典的pad/unk/eos符号对应idx,词汇表大小,初始化一个源句子长度的
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.vocab_size = len(tgt_dict) # 加载目标字典的词汇表大小
        self.beam_size = beam_size      # 获取波束搜索的波束大小
        # the max beam size is the dictionary size - 1, since we never select pad　最大波束大小是词汇表大小-1,因为我们从不选择pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a  # 生成最大长度为ax + b的序列, 其中x为源句子长度(a默认为0,b默认为200)
        self.max_len_b = max_len_b
        self.min_len = min_len     # 最小生成序列长度
        self.normalize_scores = normalize_scores # 通过输出假设的长度归一化分数
        self.len_penalty = len_penalty  # 长度惩罚α：<1.0有利于较短的句子,>1.0有利于较长的句子
        self.unk_penalty = unk_penalty  # 未知词惩罚：<0产生更多unks，>0产生更少unks
        self.retain_dropout = retain_dropout  # 生成时是否仍使用dropout
        self.temperature = temperature  # 用于预测过程的温度,>1.0会产生更均匀的样本,<1.0会产生更极端的样本
        self.match_source_len = match_source_len # 生成的预测翻译序列应该匹配源长度
        self.no_repeat_ngram_size = no_repeat_ngram_size  # ngram块,使得这样大小的ngram不会在生成的预测翻译序列中被重复
        assert temperature > 0, '--temperature must be greater than 0' # 用于预测过程的温度必须>0

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )  # 加载目标字典的pad/unk/eos符号对应idx,词汇表大小,初始化一个源句子长度的BeamSearch类


    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Generate a batch of translations. 生成一个批次的翻译

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            # 模型集合,即存储用于评估的模型(TransformerModel类,模型全部训练参数和缓冲区从模型文件中加载)的列表
            sample (dict): batch # 一个批次的示例
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens # 强制解码器从这些标识符开始,默认为None
            bos_token (int, optional): beginning of sentence token  # 句子开头的标记符,默认为eos
                (default: self.eos)
        """
        model = EnsembleModel(models) # 返回用于评估的模型集合(默认就1个模型)的EnsembleModel类,其还加载了incremental_states为{模块:{}}的字典
        return self._generate(model, sample, **kwargs) # 通过用于生成给定源句子的翻译类SequenceGenerator的genrate(),
        # 由用于评估的模型集合(默认就1个模型)的EnsembleModel类和一个批次的示例返回该批次的最终翻译结果,其中每个句子对应的beam_size个翻译结果按'scores'从小到大排序过

    @torch.no_grad()
    def _generate(
        self,
        model,  # 用于评估的模型集合(默认就1个模型)的EnsembleModel类,其还加载了incremental_states为{模块:{}}的字典
        sample, # 一个批次的示例
        prefix_tokens=None,
        bos_token=None,
        **kwargs
    ):
        if not self.retain_dropout: # 默认生成时不仍使用dropout,故将用于评估的模型集合设置为eval()模式,即trainging=False,dropout不起作用
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        # model.forward通常将prev_output_tokens分别输入解码器,但是SequenceGenerator直接调用model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }  # 从sample的'net_input'中取出'src_tokens'->当前遍历的batch中全部源句子的tensor所转换的二维填充向量,按照源句子降序的order重新排列每行;
          #                           'src_lengths'->将当前遍历的batch中全部源句子的长度的Tensor(带eos);

        src_tokens = encoder_input['src_tokens'] # 'src_tokens':当前遍历的batch中全部源句子的tensor所转换的二维填充向量,按照源句子降序的order重新排列每行;
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1) # 将当前遍历的batch中全部源句子的长度的Tensor(不带eos/pad);
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]  # 由src_tokens的形状获得bsz,src_len
        src_len = input_size[1]
        beam_size = self.beam_size  # 获得波束搜索的波束宽度

        if self.match_source_len:  # 若生成的预测翻译序列应该匹配源长度;非默认情况
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),  # 生成序列的最大长度ax+b(默认a=0,b=200,其中x为src_len),默认为200
                # exclude the EOS marker # 排除EOS标记
                model.max_decoder_positions() - 1,  # 用于评估的模型集合(默认就1个模型)的各模块中解码端最大token位置,默认为1024-1
            ) # 默认生成序列的最大长度为200
        assert self.min_len <= max_len, 'min_len cannot be larger than max_len, please adjust these!'

        # compute the encoder output for each beam  # 计算每个波束的编码器输出
        encoder_outs = model.forward_encoder(encoder_input)
        # 给定sample的'src_tokens'(全部源句子的tensor所转换的二维填充向量)和'src_lengths'(全部源句子的长度的Tensor)由模型的编码器计算当前批次的编码器输出
        # EncoderOut(
        #     encoder_out=x,  # T x B x C  # 返回最后一层编码器的输出layer_norm(dropout(max(0, x'W1+b1)W2+b2) + x')
        #     encoder_padding_mask=encoder_padding_mask,  # B x T # 计算源端填充元素(<pad>)的位置为源端填充掩码矩阵,形状为(batch, src_len)
        #     encoder_embedding=encoder_embedding,  # B x T x C   # 未进行PE操作的原始源端词嵌入向量Ex
        #     encoder_states=encoder_states,  # List[T x B x C]  # 返回存有每一层编码器的输出的列表,测试时为None
        # )
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1) # 将[[0,..,bsz]]按dim=1扩维beam size倍,然后展平,若bsz=2,beam size=3,则[0,0,0,1,1,1]
        new_order = new_order.to(src_tokens.device).long()  # 将当前批次的句子行号新顺序new_order与src_tokens保持同一类型;后续不发生改变
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)
        # 给定当前批次的编码器输出和当前批次的句子行号新顺序new_order,返回根据new_order重新排序过(对bsz所在维度重复扩展beam size次)的编码器输出encoder_outs

        # initialize buffers # 初始化缓冲区
        # 创建一个形状如下的float32类型的零矩阵,表示每个假设的累积分数(累加的lprobs)
        scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone() # 返回一个scores的副本(占新的内存,仍进行梯度计算)
        # 创建一个形状如下的int64类型的填充矩阵(全为1),表示待生成目标词idx序列矩阵,其中max_len不包括EOS(开头和结尾各有一个),所以+2
        tokens = src_tokens.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone() # 返回一个tokens的副本(占新的内存,仍进行梯度计算)
        # 将句子开头的标记符默认设置为eos_idx(2),且待生成目标词idx序列矩阵将pad符号替换为句子开头的标记符以进行第一个时间步的解码
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn, attn_buf = None, None  # 初始化attn和attn_buf为None

        # The blacklist indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the blacklist would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        # 黑名单表示应忽略的候选对象,例如,假设我们正在采样,并且已经完成2/5个采样.然后黑名单会将2个位置标记为被忽略,因此我们仅完成剩下的3个样本
        # 由src_tokens创建一个形状[bsz,beam_size]的Bool类型的False矩阵
        blacklist = src_tokens.new_zeros(bsz, beam_size).eq(-1)  # forward and backward-compatible False mask 前向和后向兼容的False Mask

        # list of completed sentences　# 完成句子的列表
        finalized = [[] for i in range(bsz)]    # 最终确定句子的列表,初始化为bsz个空列表 # 每个sent, 搜索到的EOS的hypo
        finished = [False for i in range(bsz)]  # 句子完成标记的列表,初始化为bsz个Fasle的列表 # 每个sent是否完成搜索
        num_remaining_sent = bsz   # 所剩未完成的句子数

        # number of candidate hypos per step # 每个解码时间步下候选假设的数目->2*beam size
        # 每个时刻在扩展的时候, 得到cand_size个候选, 把搜索到eos到放到finalized中去, 保证至少还剩beam个cand用于下次搜索
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS　# 2*波束大小(如果一半为EOS)

        # offset arrays for converting between different indexing schemes 　# 用于在不同索引方案之间进行转换的偏移数组
        # 后面需要把 bsz x cand_size 的矩阵转化成一维, 用于记录原先是哪个sent的哪个cand
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        # 定义batch偏移数组为[[0*beam size],[1*beam size],...,[bsz*beam size]],类型与tokens一致;若bsz为3,beam size为５,则为[[0],[5],[10]]
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)
        # 定义化候选偏移数组为[0,1,...,2*beam_size],类型与tokens一致;若beam size为５,则为[0,1,2,...10]

        # helper function for allocating buffers on the fly # 帮助函数,用于动态分配缓冲区
        buffers = {}

        def buffer(name, type_of=tokens):  # 默认type_of->tokens为待生成目标词idx序列矩阵
            if name not in buffers:  # 若name不在动态分配缓冲区buffers内,返回与
                buffers[name] = type_of.new()  # 将{name:与type_of类型一致的tensor([])}键值对添加到buffers中
            return buffers[name]  # 返回从buffers字典中取出的name对应的值,如buffers不存在name,则返回tensor([])

        def is_finished(sent, step, unfin_idx):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            # 通过比较最终假设中的最差得分与未完成假设中的最佳得分,检查我们是否已完成给定句子的生成
            # 如果一个句子的未完成假设的得分已经比当前完成假设的得分差, 就不用再搜索了.
            """
            assert len(finalized[sent]) <= beam_size # 在此解码时间步中完成的最终假设总数要<= beam_size;
            if len(finalized[sent]) == beam_size or step == max_len: # 若在此解码时间步中完成的最终假设总数= beam_size或者当前解码时间步到达最大生成序列长度
                return True                                          # 则确认已完成该句子的生成,停止搜索
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.
            # 在此解码时间步中完成给定假设的最终确定, 同时使每个句子中最终确定的假设总数<= beam_size;
            注意：输入必须以所需的最终确定顺序进行, 以便输入较早出现的假设优先于输入较晚出现的假设;
            # 对每个step, 到达eos状态的hypo, 放到相应的finalize中去, 并更新finished状态.
            Args:
                step: current time step  # 当前解码时间步
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize # 索引向量,范围为[0,bsz*beam_size),指示要最终确定的假设;即前beam_size列中eos标识符的最高候选假设的波束索引
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis # 一个与bbsz_idx大小相同的向量,其中包含每个假设的分数;即前beam_size列中归一化的概率lprobs最大且为eos的值;
            """
            assert bbsz_idx.numel() == eos_scores.numel()  # 确保两者的元素个数相等,均为bsz*beam_size个

            # clone relevant token and attention tensors  # 克隆相关标记和注意力张量
            # torch.index_select->从张量的某个维度的指定位置选取数据
            tokens_clone = tokens.index_select(0, bbsz_idx) # 从tokens的第0个维度(bsz*beam_size所在行),通过由bbsz_idx选择数据复制到tokens_clone中去
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS;跳过第一个索引,其为EOS标记符
            assert not tokens_clone.eq(self.eos).any()  # 确保待生成目标词idx序列矩阵中不存在eos_idx
            tokens_clone[:, step] = self.eos  # 给待生成目标词idx序列矩阵的最大生成序列长度时间步位置设置为eos_idx
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2] if attn is not None else None
            # 若attn不为None,则从attn的第0个维度(bsz*beam_size所在行),通过bbsz_idx选择数据复制到attn_clone中取,再跳过第一个索引,其为EOS标记符
            # attn[:, :step + 2]表示到下一个解码时间步的attn权重,即定义为当前解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重

            # compute scores per token position　# 计算每个token位置的分数
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            # 从scores的第0个维度(bsz*beam_size所在行),通过bbsz_idx选择数据复制,[:, :step+1]即包含了全部
            pos_scores[:, step] = eos_scores  # 给到当前解码时间步之前的每个假设的累积分数的最大生成序列长度时间步位置,设置为前beam_size列中归一化的概率lprobs最大且为eos的值
            # convert from cumulative to per-position scores  # 从累积得分转换为按位置得分,如([1., 2., 3., 4.])累加后([1.,3.,6.,10.]),经过下述操作得回([1., 2., 3., 4.])
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores # 归一化句子级别的分数
            if self.normalize_scores:  # 通过输出假设的长度归一化分数,即将eos_scores除以(输出假设的总长度^长度惩罚α)
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = [] # 累积每个sent前面有多少个完成的sent,这样得到该sent对应到最初batch中是哪个idx
            prev = 0
            for f in finished: # 遍历句子完成标记的列表:
                if f:  # 若句子完成了,则prev+1;若句子未完成,则prev不变
                    prev += 1
                else:  # 累积每个sent前面有多少个完成的sent的列表
                    cum_unfin.append(prev)

            sents_seen = set()
            # 同时遍历最终确定的假设,即前beam_size列中eos标识符的最高候选假设的波束索引,以及包含每个假设的分数;即前beam_size列中归一化的概率lprobs最大且为eos的值;
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size  # 获得未完成句子的波束索引
                sent = unfin_idx + cum_unfin[unfin_idx] # 每个没有完成的句子,前面有多少完成的句子,用于确定当前句子位于最初batch中哪个位置

                sents_seen.add((sent, unfin_idx)) # 将当前句子位于最初batch中哪个位置,未完成句子的波束索引组成的元组加入集合内

                if self.match_source_len and step > src_lengths[unfin_idx]:  # 若生成的预测翻译序列应该匹配源长度,且当前解码时间步大于未完成句子的长度;非默认情况
                    score = -math.inf

                def get_hypo():  # 得到翻译结果

                    if attn_clone is not None: # 若全部解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重的副本不为空
                        # remove padding tokens from attn scores　#　从attn分数中删除填充符号,即pad_idx对应的attn_clone为None,故hypo_attn也为None
                        hypo_attn = attn_clone[i]
                    else:
                        hypo_attn = None

                    return {
                        'tokens': tokens_clone[i], # 当前解码时间步的待生成目标词idx序列的副本
                        'score': score,           # 当前解码时间步的归一化的概率lprobs最大且为eos的值;
                        'attention': hypo_attn,  # src_len x tgt_len # 返回当前解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重的副本
                        'alignment': None,
                        'positional_scores': pos_scores[i],  # 当前解码时间步的每个假设的分数(lprobs值)
                    }

                if len(finalized[sent]) < beam_size: # 若在此解码时间步中完成的最终假设总数<= beam_size;
                    finalized[sent].append(get_hypo()) # 则最终完成的翻译结果添加进当前句子位于最初batch中哪个位置对应的finalized列表中

            newly_finished = []  # 新的完成句子的行号列表
            for sent, unfin_idx in sents_seen: # 遍历存有当前句子位于最初batch中哪个位置,未完成句子的波束索引组成的元组集合:
                # check termination conditions for this sentence  # 检查此句子的终止条件
                if not finished[sent] and is_finished(sent, step, unfin_idx): # 若位于最初batch中的位置的句子完成标记为False,且is_finished确认已完成该句子的生成
                    finished[sent] = True # 将位于最初batch中的位置的句子完成标记更新为True
                    newly_finished.append(unfin_idx) # 并将未完成句子的波束索引加到新的完成句子的行号列表
            return newly_finished  # 返回新的完成句子的行号列表

        reorder_state = None  # 初始化是否重新排序解码器的内部状态,即最高候选假设的波束索引cand_bbsz_idx的i行每(active_hypos[i][j])列映射而成
        batch_idxs = None     # 初始化batch_idxs
        for step in range(max_len + 1): # 遍历每一个解码时间步(直到生成序列的最大长度+1,一个额外步骤用于EOS标记) # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams # 根据波束的先前选择对解码器的内部状态进行重新排序
            if reorder_state is not None: # 初始时重新排序解码器的内部状态为None:
                if batch_idxs is not None: # 若batch_idxs不为None,只有当句子完成才不为None
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                model.reorder_incremental_state(reorder_state)
                # 返回根据*new_order*重新排序过的Transformer decoder layer中self-attn和enc-dec attn中的incremental_state,主要对bsz所在维度进行重复扩展beam size次
                encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)
                # 返回根据*new_order*重新排序过的Transformer 编码器输出encoder_outs,主要对bsz所在维度进行重复扩展beam size次;
                # 注! 直到整个最大序列生成长度都遍历完,除了encoder_outs一开始重复扩展beam size次,后续保持不变
            lprobs, avg_attn_scores = model.forward_decoder(
                tokens[:, :step + 1], encoder_outs, temperature=self.temperature,
            )  # 给定待生成目标词idx序列矩阵的前step个时间步目标词idx输出,根据new_order重新排序过(对bsz所在维度重复扩展beam size次)的编码器输出encoder_outs,以及用于预测过程的温度
            # 返回当前解码时步的模型输出中获取归一化的概率(或对数概率); 当前解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重
            lprobs[lprobs != lprobs] = -math.inf # 确保无误不做任何操作,lprobs->[bsz*beam_size, tgt_vocab_size]

            lprobs[:, self.pad] = -math.inf  # never select pad # 将lprobs中对应tgt_vocab中pad符号idx的概率置为-inf
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty # 将lprobs中对应tgt_vocab中unk符号idx的概率减去未知词惩罚;非默认操作

            # handle max length constraint # 处理最大长度限制
            if step >= max_len:  # 若当前解码时间步到达最大生成序列长度,则仅保留tgt_vocab中eos符号idx的概率不为-inf;确保百分百解码为eos
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths) # 处理前缀令牌(可能具有不同的长度)
            if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:  # 由于prefix_tokens为None,非默认情况
                prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
                prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_mask = prefix_toks.ne(self.pad)
                lprobs[prefix_mask] = -math.inf
                lprobs[prefix_mask] = lprobs[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
                )
                # if prefix includes eos, then we should make sure tokens and
                # scores are the same across all beams
                eos_mask = prefix_toks.eq(self.eos)
                if eos_mask.any():
                    # validate that the first beam matches the prefix
                    first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
                    eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
                    target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
                    assert (first_beam == target_prefix).all()

                    def replicate_first_beam(tensor, mask):
                        tensor = tensor.view(-1, beam_size, tensor.size(-1))
                        tensor[mask] = tensor[mask][:, :1, :]
                        return tensor.view(-1, tensor.size(-1))

                    # copy tokens, scores and lprobs from the first beam to all beams
                    tokens = replicate_first_beam(tokens, eos_mask_batch_dim)
                    scores = replicate_first_beam(scores, eos_mask_batch_dim)
                    lprobs = replicate_first_beam(lprobs, eos_mask_batch_dim)
            elif step < self.min_len: # 若当前解码时间步小于最小生成序列长度
                # minimum length constraint (does not apply if using prefix_tokens) 最小长度限制(如果使用prefix_tokens则不适用)
                lprobs[:, self.eos] = -math.inf  # 让tgt_vocab中eos符号idx的概率为-inf,确保当前解码时间步不生成eos标记

            if self.no_repeat_ngram_size > 0: # ngram块,使得这样大小的ngram不会在生成的预测翻译序列中被重复;非默认情况
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                cpu_tokens = tokens.cpu()
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = cpu_tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        if ngram[-1] != self.pad:
                            gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                                    gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores # 记录注意力分数
            if type(avg_attn_scores) is list: # 若当前解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重为列表,则取出来
                avg_attn_scores = avg_attn_scores[0]
            if avg_attn_scores is not None: # 若当前解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重不为空:
                if attn is None:  # 第一个解码时间步时,attn初始为None
                    attn = scores.new(bsz * beam_size, avg_attn_scores.size(1), max_len + 2) # 根据scores的类型,随机初始化一个[bsz * beam_size,src_len,max_len + 2]的tensor矩阵
                    attn_buf = attn.clone()  # 返回一个attn的副本(占新的内存,仍进行梯度计算)
                attn[:, :, step + 1].copy_(avg_attn_scores) # 将下一解码时间步attn权重,定义为当前解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重
                # 类似attn[bsz * beam_size,src_len,step + 1]=avg_attn_scores
            scores = scores.type_as(lprobs)  # 确保scores的类型与当前解码时步的模型输出中获取归一化的概率(或对数概率)lprobs一致
            scores_buf = scores_buf.type_as(lprobs) # 确保scores_buf的类型与当前解码时步的模型输出中获取归一化的概率(或对数概率)lprobs一致
            eos_bbsz_idx = buffer('eos_bbsz_idx') # # 返回从buffers字典中取出的eos_bbsz_idx对应的值,如buffers不存在eos_bbsz_idx,则返回tensor([])
            eos_scores = buffer('eos_scores', type_of=scores) # 返回从buffers字典中取出的eos_scores对应的值,如buffers不存在eos_scores,则返回tensor([])

            self.search.set_src_lengths(src_lengths) # 在BeamSearch类中设置self.src_lengths为当前遍历的batch中全部源句子的长度的Tensor(不带eos/pad);

            if self.no_repeat_ngram_size > 0: # ngram块,使得这样大小的ngram不会在生成的预测翻译序列中被重复;非默认情况
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(cpu_tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                    banned_tokens_per_sample = gen_ngrams[bbsz_idx].get(ngram_index, [])
                    banned_tokens_per_sample = [(bbsz_idx, t) for t in banned_tokens_per_sample]
                    return banned_tokens_per_sample

                banned_tokens = []
                if step + 2 - self.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    for bbsz_idx in range(bsz * beam_size):
                        banned_tokens.extend(calculate_banned_tokens(bbsz_idx))

                if banned_tokens:
                    banned_tokens = torch.LongTensor(banned_tokens)
                    lprobs.index_put_(tuple(banned_tokens.t()), lprobs.new_tensor([-math.inf] * len(banned_tokens)))

            cand_scores, cand_indices, cand_beams = self.search.step(
                step,  # 当前解码时间步
                lprobs.view(bsz, -1, self.vocab_size), # 当前解码时步的模型输出中获取归一化的概率(或对数概率)的维度变换[bsz,beam_size,tgt_vocab_size],将pad_idx位置掩码为-inf(step=0,则还需掩码eos_idx)
                scores.view(bsz, beam_size, -1)[:, :, :step],
                # 初始化为[bsz,beam_size, 0]的float32类型的空矩阵,之后为[bsz,beam_size, step]的当前解码时间步之前的scores矩阵
            ) # 返回从包含每个假设的累积分数(或仅使用第一个波束)的当前解码时步的模型输出中获取归一化的概率(或对数概率)lprobs,选取最后一个dim维度上最大的2xbeam_size个值,2xbeam_size个值在词汇表的索引,2xbeam_size个值所属于第几个beam上

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            # cand_bbsz_idx包含最高候选假设的波束索引,其值的范围为[0,bsz * beam_size),尺寸为[bsz,cand_size]
            # batch偏移数组bbsz_offsets->[[0*beam size],[1*beam size],...,[bsz*beam size]],类型与tokens一致;
            # cand_bbsz_idx等于cand_beams+bbsz_offsets,即归一化的概率lprobs最大的2xbeam_size个值所属于第几个beam+[[0*beam size],[1*beam size],...,[bsz*beam size]]
            # 当cand_beams为[[1],[2],[3]],bbsz_offsets->[[0],[5],[10]],故cand_bbsz_idx:[[1,2,3],[6,7,8],[11,12,13]]等
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos, except for blacklisted ones
            # or candidates with a score of -inf # 最终确定以eos结尾的假设(前beam个最大概率的),但列入黑名单或得分为-inf的候选假设除外
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            # 通过归一化的概率lprobs最大的2xbeam_size个值在词汇表的索引找出解码为eos的值的位置,同时该位置要未设置为-inf;
            eos_mask[:, :beam_size][blacklist] = 0  # 选择前beam_size列,且列入黑名单的单词位置均设置为False

            # only consider eos when it's among the top beam_size indices # 仅当eos在前beam_size个索引中时才考虑(前beam个最大概率的)
            torch.masked_select(
                cand_bbsz_idx[:, :beam_size],
                mask=eos_mask[:, :beam_size],
                out=eos_bbsz_idx,
            )  # 根据mask中不为Fasle的值,取cand_bbsz_idx[:, :beam_size]中同样位置的项,将取值返回到eos_bbsz_idx中(一个1D张量)
            # 即eos_bbsz_idx表示前beam_size列中eos标识符的最高候选假设的波束索引;第一个解码步为None

            finalized_sents = set()  # 最终完成句子的行号集合
            if eos_bbsz_idx.numel() > 0:  # 若前beam_size列的eos标识符的最高候选假设的波束索引不为空,即在前beam_size列有句子要解码eos(翻译完了)
                torch.masked_select(
                    cand_scores[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_scores,
                ) # 根据mask中不为Fasle的值,取cand_scores[:, :beam_size]中同样位置的项,将取值返回到eos_scores中(一个1D张量)
                 # 即eos_scores表示前beam_size列中归一化的概率lprobs最大且为eos的值;

                # 根据当前解码时间步,以及前beam_size列中eos标识符的最高候选假设的波束索引,前beam_size列中归一化的概率lprobs最大且为eos的值;
                #　将最终完成的翻译结果添加进当前句子位于最初batch中哪个位置对应的finalized列表中,并返回新的完成句子的行号列表
                finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores)
                num_remaining_sent -= len(finalized_sents)  # 重新确定所剩未完成的句子数

            assert num_remaining_sent >= 0  # 若所剩未完成的句子数>=0
            if num_remaining_sent == 0: # 若所剩未完成的句子数为0,则跳出生成序列的过程
                break
            assert step < max_len  # 当前解码时间步不能超过最大生成序列长度

            if len(finalized_sents) > 0: # 最终完成句子的行号集合不为空,即有句子完成;
                new_bsz = bsz - len(finalized_sents)  # 需要由完成的句子数来重新确定bsz

                # construct batch_idxs which holds indices of batches to keep for the next pass # 构造batch_idxs,其中保存批次索引以备下次pass
                batch_mask = cand_indices.new_ones(bsz)  # 创建一个
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)  # 将用于在不同索引方案之间进行转换的偏移数组bbsz_offsets,形状更新为[new_bsz,beam_size]
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                blacklist = blacklist[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:  # 若没有句子完成,则batch_idxs为None
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos or
            # blacklisted hypos and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos. # 设置active_mask,值> cand_size表示eos或列入黑名单的假设,值<cand_size表示候选存活的假设.在此之后,每行的最小值是最可能的存活候选假设
            active_mask = buffer('active_mask')  # 返回从buffers字典中取出的active_mask对应的值,如buffers不存在active_mask,则返回tensor([])
            eos_mask[:, :beam_size] |= blacklist
            # 通过归一化的概率lprobs最大的2xbeam_size个值在词汇表的索引找出的解码为eos值的位置(lprobs未设置为-inf,且未列入黑名单),
            # 对其前beam列的位置与黑名单位置进行异或,返回给eos_mask[:, :beam_size]
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size, # 先将eos_mask转成int64类型(0/1的掩码矩阵),再*(2*beam_size)变成(0/2*beam_size的掩码矩阵)
                cand_offsets[:eos_mask.size(1)], # 定义化候选偏移数组为[0,1,...,2*beam_size],eos_mask.size(1)也为cand_size;故即为cand_offsets的全部
                out=active_mask,
            )  # 将(0/2*beam_size的掩码矩阵)+候选偏移数组cand_offsets为[0,1,...,2*beam_size],并返回值给active_mask(一个1D张量)
            # 满足active_mask的设置,值> cand_size表示eos或列入黑名单的假设,值<cand_size表示存活候选假设.在此之后,每行的最小值是最可能的存活候选假设

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask  # 获取最高的beam_size个存活假设,这些假设只是active_mask中值最小的假设
            active_hypos, new_blacklist = buffer('active_hypos'), buffer('new_blacklist')
            # 返回从buffers字典中取出的active_hypos和new_blacklist对应的值,如buffers不存在active_hypos和new_blacklist,则返回tensor([])
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,  # active_mask-> bsz,cand_size; largest为False,表示按从小到大的顺序;
                out=(new_blacklist, active_hypos)
            )  # 从全部候选假设active_mask中,在第一个dim维度上选取最小的beam_size个值及其对应的indices,分别返回给new_blacklist和active_hypos

            # update blacklist to ignore any finalized hypos # 更新黑名单以忽略任何最终确定的假设
            blacklist = new_blacklist.ge(cand_size)[:, :beam_size] # new_blacklist形状为bsz,beam_size;故取出其全部
            # new_blacklist为全部候选假设active_mask中在第一个dim维度上选取的最小beam_size个值,选择将值>cand_size列入黑名单的假设
            assert (~blacklist).any(dim=1).all()  # 确保blacklist为全部False的矩阵

            active_bbsz_idx = buffer('active_bbsz_idx') # 返回从buffers字典中取出的active_bbsz_idx对应的值,如buffers不存在active_bbsz_idx,则返回tensor([])
            # torch.gather->对于out指定位置上的值, 根据index去寻找input里面对应的索引位置,若dim=1,则有如下:out[i][j] = input[i][index[i][j]]
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            ) # active_bbsz_idx[2][3]=cand_bbsz_idx[2][active_hypos[2][3]],即遍历从全部候选假设active_mask在第一个dim维度上选取最小的beam_size个值对应的indices(active_hypos)
            # 将最高候选假设的波束索引cand_bbsz_idx的i行每(active_hypos[i][j])列,映射到active_bbsz_idx中去
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),  # 取出当前解码时间步的每个假设的累积分数scores,并将[bsz*beam_size,1]->[bsz, beam_size]得到scores'
            )# scores'[2][3]=cand_scores[2][active_hypos[2][3]],即遍历从全部候选假设active_mask在第一个dim维度上选取最小的beam_size个值对应的indices(active_hypos)
            # 将归一化的概率lprobs最大的前beam_size个值的i行每(active_hypos[i][j])列,映射到scores'中去

            active_bbsz_idx = active_bbsz_idx.view(-1) # 将active_bbsz_idx和active_scores展平,从[bsz, beam_size]->[bsz*beam_size]
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses # 对于存活假设,复制tokens和scores
            # torch.index_select->从张量的某个维度的指定位置选取数据
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],  # tokens[:, :step + 1]表示到当前解码时间步的待生成目标词idx序列矩阵(包括当前解码时间步),tokens_buf为tokens的副本(新内存地址,保持梯度计算)
            )  # 从tokens[:, :step + 1]的第0个维度(bsz*beam_size所在行),通过由active_bbsz_idx选择数据复制到tokens_buf[:, :step + 1]中去
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1], # 从dim=0复制tokens的到当前解码时间步的待生成目标词idx序列矩阵(包括当前解码时间步),形状为(bsz, beam_size, max_len+2)
            ) # tokens_buf'[2][3]=cand_indices[2][active_hypos[2][3]],即遍历从全部候选假设active_mask在第一个dim维度上选取最小的beam_size个值对应的indices(active_hypos),
            # 将归一化的概率lprobs最大的前beam_size个值在词汇表的索引的i行每(active_hypos[i][j])列,映射到tokens_buf'中去
            if step > 0:  # 若不是第一个解码时间步:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],   # scores[:, :step + 1]表示当前解码时间步之前的每个假设的累积分数,scores_buf为scores的副本(新内存地址,保持梯度计算)
                )  # 从scores[:, :step]的第0个维度(bsz*beam_size所在行),通过active_bbsz_idx选择数据复制到scores_buf[:, :step]中去
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step], # 从dim=0复制scores的到当前解码时间步之前的每个假设的累积分数,形状为(bsz, beam_size, max_len+2)
            )# scores_buf'[2][3]=cand_scores[2][active_hypos[2][3]],即遍历从全部候选假设active_mask在第一个dim维度上选取最小的beam_size个值对应的indices(active_hypos),
            # 将归一化的概率lprobs最大的前beam_size个值的i行每(active_hypos[i][j])列,映射到scores_buf'中去

            # copy attention for active hypotheses  # 复制对于存活假设的注意力
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                    # attn[:, :step + 2]表示到下一个解码时间步的attn权重,即定义为当前解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重
                    # 类似attn[bsz * beam_size,src_len,step+1]=avg_attn_scores,attn_buf为attn的副本(新内存地址,保持梯度计算)
                )  # 从attn[:, :, :step + 2]的第0个维度(bsz*beam_size所在行),通过active_bbsz_idx选择数据复制到attn_buf[:, :, :step + 2]中去

            # swap buffers # 交换缓冲区,将存有新的值tokens_buf/scores_buf/attn_buf复制给tokens/scores/attn,将旧的tokens值存回tokens_buf/scores_buf/attn_buf内
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder # 在解码器中对增量状态进行重新排序
            reorder_state = active_bbsz_idx # active_bbsz_idx为最高候选假设的波束索引cand_bbsz_idx的i行每(active_hypos[i][j])列映射而成

        # sort by score descending　# 按分数降序排序
        # finalized为该批次的最终翻译结果:每个句子对应有beam_size个翻译结果,每个如下:
        # {
        #     'tokens': tokens_clone[i],  # 当前解码时间步的待生成目标词idx序列的副本
        #     'score': score,  # 当前解码时间步的归一化的概率lprobs最大且为eos的值;
        #     'attention': hypo_attn,　# src_len x tgt_len # 返回当前解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重的副本
        #     'alignment': None,
        #     'positional_scores': pos_scores[i],  # 当前解码时间步的每个假设的分数(lprobs值)
        # }
        for sent in range(len(finalized)):  # 遍历该批次的最终翻译结果的每一个句子,对每一个句子的beam_size个结果按其'score'来从小到大排(越小越好)
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)  #
        return finalized  # 返回该批次的最终翻译结果,其中每个句子对应的beam_size个翻译结果按'scores'从小到大排序过


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models. 关于模型集合的包装器"""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models) # 将存储用于评估的模型(TransformerModel类,模型全部训练参数和缓冲区从模型文件中加载)的列表转成torch.nn.ModuleList类
        self.incremental_states = None # incremental_states
        if all(hasattr(m, 'decoder') and isinstance(m.decoder, FairseqIncrementalDecoder) for m in models): # 遍历用于评估的模型集合(默认就1个模型)中全部模块:
            self.incremental_states = {m: {} for m in models}  # 若该模块存在类型为FairseqIncrementalDecoder的decoder模块,定义incremental_states为{用于评估的模型TransformerModel类:{}}的字典

    def has_encoder(self): # 返回用于评估的模型集合(默认就1个模型)中是否'encoder'模块
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():  # 若用于评估的模型集合(默认就1个模型)不存在'encoder'模块
            return None
            # 给定sample的'src_tokens'(全部源句子的tensor所转换的二维填充向量)和'src_lengths'(全部源句子的长度的Tensor)由模型的编码器计算当前批次的编码器输出
        return [model.encoder(**encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1.):
        if len(self.models) == 1:  # 用于评估的模型集合(默认就1个模型)
            return self._decode_one(
                tokens,         # 待生成目标词idx序列矩阵的前step个时间步目标词idx输出
                self.models[0],  # 用于评估的模型集合(默认就1个模型)
                encoder_outs[0] if self.has_encoder() else None,  # 根据new_order重新排序过(对bsz所在维度重复扩展beam size次)的编码器输出
                self.incremental_states,  # 定义incremental_states为{用于评估的模型TransformerModel类:{}}的字典
                log_probs=True,
                temperature=temperature,  # 用于预测过程的温度
            )  # 当前解码时步的模型输出中获取归一化的概率(或对数概率); 当前解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重
        # 下面为非默认操作:
        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None: # 由于incremental_states为{模块:{}}的字典,默认情况
            decoder_out = list(model.forward_decoder(
                tokens, encoder_out=encoder_out, incremental_state=self.incremental_states[model],
            )) # 给定待生成目标词idx序列矩阵的前step个时间步目标词idx输出,根据new_order重新排序过(对bsz所在维度重复扩展beam size次)的编码器输出,
               # incremental_states为{用于评估的模型TransformerModel类:{}}的字典,由模型的解码器计算当前批次的解码器输出
        # 返回经过vaswani论文"Linear层"(即目标端词嵌入矩阵)的解码器输出Linear(layer_norm(dropout(max(0, x''W1+b1)W2+b2) + x''))(bsz,tgt_len,tgt_vocab),
        # 字典{"attn"-> 最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重->softmax((QK^T)/dk^(-0.5)), "inner_states"-> 存储每一层解码器的输出列表}
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        # 更新decoder_out中->经过vaswani论文"Linear层"(即目标端词嵌入矩阵)的解码器输出Linear(layer_norm(dropout(max(0, x''W1+b1)W2+b2) + x''))
        # 由于测试时采用逐个时间步解码,故只需要当前解码时步的经过vaswani论文"Linear层"解码器输出
        if temperature != 1.: # 若用于预测过程的温度不等于1.;非默认情况
            decoder_out[0].div_(temperature)
        attn = decoder_out[1] if len(decoder_out) > 1 else None # 取出decoder_out中字典{"attn":xxx,"inner_states":xxx}
        if type(attn) is dict: # 若attn为字典形式,则将attn更新->从decoder_out中字典中取出的"attn"的值
            attn = attn.get('attn', None)
        if type(attn) is list: # 若从decoder_out中字典中取出的"attn"的值为列表形式,则取出"attn"的值
            attn = attn[0]
        if attn is not None: # 若最后一个解码器层enc-dec self-attn的输出attn_weights在全部注意力头上平均注意力权重不为None:
            attn = attn[:, -1, :] # 由于测试时采用逐个时间步解码,故只需要当前解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs) # 从模型的输出中获取归一化的概率(或对数概率)
        # 若选择返回log概率, 则对经过"Linear层"的解码器输出logits进行log_softmax(经过vaswani论文的"Softmax"),否则对其进行softmax
        probs = probs[:, -1, :] # 由于测试时采用逐个时间步解码,故只需要当前解码时步的模型输出中获取归一化的概率(或对数概率)
        return probs, attn # 当前解码时间步的模型输出中获取归一化的概率(或对数概率); 当前解码时步的最后一个解码器层enc-dec attn的输出attn_weights在全部注意力头上平均注意力权重

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder(): # 若用于评估的模型集合(默认就1个模型)不存在'encoder'模块
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]  # 给定当前批次的编码器输出和当前批次的句子行号新顺序new_order,返回根据*new_order*重新排序过的编码器输出encoder_outs,主要对bsz所在维度进行重复扩展beam size次

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models: # 返回根据*new_order*重新排序过的Transformer layer中self-attn和enc-dec attn中的incremental_state,主要对bsz所在维度进行重复扩展beam size次
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)


class SequenceGeneratorWithAlignment(SequenceGenerator):

    def __init__(self, tgt_dict, left_pad_target=False, **kwargs):
        """Generates translations of a given source sentence.

        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        model = EnsembleModelWithAlignment(models)
        finalized = super()._generate(model, sample, **kwargs)

        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        src_tokens, src_lengths, prev_output_tokens, tgt_tokens = \
            self._prepare_batch_for_alignment(sample, finalized)
        if any(getattr(m, 'full_context_alignment', False) for m in model.models):
            attn = model.forward_align(src_tokens, src_lengths, prev_output_tokens)
        else:
            attn = [
                finalized[i // beam_size][i % beam_size]['attention'].transpose(1, 0)
                for i in range(bsz * beam_size)
            ]

        # Process the attn matrix to extract hard alignments.
        for i in range(bsz * beam_size):
            alignment = utils.extract_hard_alignment(attn[i], src_tokens[i], tgt_tokens[i], self.pad, self.eos)
            finalized[i // beam_size][i % beam_size]['alignment'] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        src_tokens = src_tokens[:, None, :].expand(-1, self.beam_size, -1).contiguous().view(bsz * self.beam_size, -1)
        src_lengths = sample['net_input']['src_lengths']
        src_lengths = src_lengths[:, None].expand(-1, self.beam_size).contiguous().view(bsz * self.beam_size)
        prev_output_tokens = data_utils.collate_tokens(
            [beam['tokens'] for example in hypothesis for beam in example],
            self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=True,
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam['tokens'] for example in hypothesis for beam in example],
            self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_align(self, src_tokens, src_lengths, prev_output_tokens):
        avg_attn = None
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens)
            attn = decoder_out[1]['attn']
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_attn.div_(len(self.models))
        return avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens,
                encoder_out=encoder_out,
                incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1] if len(decoder_out) > 1 else None
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if type(attn) is list:
            attn = attn[0]
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn
