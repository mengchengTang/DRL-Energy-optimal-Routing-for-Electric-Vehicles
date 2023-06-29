import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    主要是将静态元素和动态元素进行一次embedding，这里的sequence_len为1+Charging_num+custom_num)
    """
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output                                                          # (batch, hidden_size, seq_len)

class Attention(nn.Module):
    """将当前解码器输出与编码器做一次attention，这里注意."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))  #制定参数是可训练的，并用nn.parameter对模型进行初始化


        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),   #同上
                                          device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):

        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)         #先对decoder_hidden扩展一列，再把它的形状跟static_hidden对齐
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)        #1代表加列。竖着拼接

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)

        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)                                        #(batch, 1,seq_len)
        return attns                                                           #返回相应的概率向量

class Pointer(nn.Module):
    """将Attention求得的上下文向量跟编码器的隐藏状态做pointer"""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),     #这里注意是上下文向量，所以第三轴的维度为2*hidden_size
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)              #注意到这里使用了batch_first=true!!!!!
        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):

        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)                                            #将rnn_out的第二维去除（B，hidden_size)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh)

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))               #(B,1,sequence_len)*(B,sequence_len,hidden_size)=(B, 1, hidden_size)

        # Calculate the next output using Batch-matrix-multiply ops
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)                  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)   #把概况的第二层去掉，（batch，seq_len）

        return probs, last_hh                                                #prbs=(B,sequence_len),lasta_hh=(B,hidden_size)

class DRL4EVRP(nn.Module):
    """定义整个模型的更新过程：包括解码器、编码器、指针网络.

    Parameters
    ----------
    static_size: int
        定义静态元素的个数
        (e.g. 2 for (x, y) coordinates)
    dynamic_size:4
        分别是车辆的转载量、客户需求量、电动车SOC水平、
    hidden_size: int
        静态元素和动态元素编码后的隐藏层状态
    update_fn:
        选完下一个客户点时更新所有动态元素
    mask_fn:
        屏蔽策略的更新
    num_layers: int
        循环神经网络的层数
    dropout: float
        the dropout rate for the decoder
    """

    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        super(DRL4EVRP, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn
        self.mask_fn = mask_fn

        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)
        self.decode_type = None

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def forward(self, x, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            所有点的静态元素
        dynamic: Array of size (batch_size, feats, num_cities)
            所有点的动态元素，包括
        decoder_input: Array of size (batch_size, num_feats)
           解码器输入
        last_hh: Array of size (batch_size, num_hidden)
            当前隐藏层状态
        """
        static, dynamic, distances, slope = x

        static = static.float().to(device)
        dynamic = dynamic.float().to(device)
        distances = distances.float()
        slope = slope.float()
        decoder_input = static[:, :, 0:1]

        batch_size, input_size, sequence_size = static.size()


        now_idx = torch.zeros(batch_size,device= device)
        # Always use a mask - if no function is provided, we don't update it
        mask = torch.ones(batch_size, sequence_size, device=device)

        # Structures for holding the output sequences
        tour_idx, tour_logp, R = [], [],[]
        max_steps = sequence_size if self.mask_fn is None else 50

        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        for _ in range(max_steps):

            if not mask.byte().any():
                break

            # ... but compute a hidden rep for each element added to sequence
            decoder_hidden = self.decoder(decoder_input)

            probs, last_hh = self.pointer(static_hidden,
                                          dynamic_hidden,
                                          decoder_hidden, last_hh)
            probs = F.softmax(probs + mask.log(), dim=1)

            if self.decode_type == "sample":
                m = torch.distributions.Categorical(probs)
                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)

            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()
            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                dynamic,reward = self.update_fn(dynamic,distances,slope,now_idx,ptr.data)
                dynamic_hidden = self.dynamic_encoder(dynamic)

                # Since we compute the VRP in minibatches, some tours may have
                # number of stops. We force the vehicles to remain at the depot
                # in these cases, and logp := 0
                is_done = dynamic[:, 1].sum(1).eq(0).float()
                logp = logp * (1. - is_done)

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, distances, slope, ptr.data).detach()

            now_idx = ptr
            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))
            R.append(reward)
            decoder_input = torch.gather(static, 2,
                                         ptr.view(-1, 1, 1)
                                         .expand(-1, input_size, 1)).detach()

        tour_idx = torch.cat(tour_idx, dim=1)                                      # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)                                    # (batch_size, seq_len)
        R = torch.cat(R,dim=1)

        return tour_idx, tour_logp, R

    def sample_many(self, batch, batch_rep=1, iter_rep=1):
        """
        :param batch:数据
        :param batch_rep:把batch分为多少块
        :param iter_rep: 迭代的伦次
        :return:
        """
        static, dynamic, distances, slope = batch
        static = static[None, ...].expand(batch_rep, *static.size()).contiguous().view(-1, *static.size()[1:])
        dynamic = dynamic[None, ...].expand(batch_rep, *dynamic.size()).contiguous().view(-1, *dynamic.size()[1:])
        distances = distances[None, ...].expand(batch_rep, *distances.size()).contiguous().view(-1, *distances.size()[1:])
        slope = slope[None, ...].expand(batch_rep, *slope.size()).contiguous().view(-1,*slope.size()[1:])
        batch = static,dynamic,distances,slope

        costs = []
        pis = []
        for i in range(iter_rep):
            tour_idx, tour_logp, R = self.forward(batch)  # tour_idx的形状为[batch,sequence_len]

            cost = torch.sum(R, dim=1)  # [batch]

            costs.append(cost.view(batch_rep, -1).t()) # 每一个元素的形状为[batch,batch_rep]
            pis.append(tour_idx.view(batch_rep, -1, tour_idx.size(-1)).transpose(0, 1))
        costs = torch.cat(costs, 1)  # 这里是将迭代伦次的所有batch拼接起来，形状为 [batch,iter_rep]
        # 对不同伦次之间的索引进行拼接
        max_length = max(pi.size(-1) for pi in pis)
        pis = torch.cat(
            [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
            1)
        mincosts, argmincosts = costs.min(-1)  # [batch]
        minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]  # [batch_size,min_sequence]
        return mincosts, minpis  # [batch],这个就是采样出来最小的

    def beam_search(self,  x, beam_width, last_hh=None):
        # 拿出数据
        static, dynamic, distances, slope = x
        static = static.float().to(device)
        dynamic = dynamic.float().to(device)
        distances = distances.float().to(device)
        slope = slope.float().to(device)
        batch_size, input_size, sequence_size = static.size()

        # 编码一些数据
        mask = torch.ones(batch_size, sequence_size, device=device)
        tour_idx, tour_logp, R = [], [], []

        now_idx = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        # 对数据进行复制，为了beam_search
        static = static.repeat(beam_width, 1, 1)
        static_hidden = self.static_encoder(static)
        dynamic = dynamic.repeat(beam_width, 1, 1)
        dynamic_hidden = self.dynamic_encoder(dynamic)
        decoder_input = static[:, :, 0:1]
        mask = mask.repeat(beam_width, 1)  # (batch*beam_width, seq_len)
        batch_idx = torch.LongTensor([i for i in range(batch_size)]).to(device)
        batch_idx = batch_idx.repeat(beam_width)  # (batch*beam_width)
        now_idx = now_idx.repeat(beam_width, 1)
        distances = distances.repeat(beam_width, 1, 1)
        slope = slope.repeat(beam_width, 1, 1)

        max_steps = sequence_size if self.mask_fn is None else 50  # 定义迭代步长

        for _ in range(max_steps):

            if not mask.byte().any():  # 假如除仓库之外的任意一个点都不能去，且batch内实例车辆当前均在仓库
                break
            decoder_hidden = self.decoder(decoder_input)
            beam_probs, last_hh = self.pointer(static_hidden,
                                          dynamic_hidden,
                                          decoder_hidden, last_hh)
            beam_probs = F.softmax(beam_probs + mask.log(), dim=1)  # [batch*beam_width,sequence_len]


            if _ == 0:
                probs, idx = torch.topk(beam_probs[:batch_size], beam_width, dim=1)  # both: (batch, beam_width) 在一个batch中取出概率前beam_width个数据 其中torch.topk不降维
                idx = idx.transpose(1, 0).contiguous().view(-1, 1)  # (batch*beam_width, 1)
                probs = probs.transpose(1, 0).contiguous().view(-1, 1)  # ditto
                prob_log = torch.log(probs)  # (batch*beam_width, 1)
                tours = idx
                if self.update_fn is not None:
                    dynamic, reward = self.update_fn(dynamic, distances, slope, now_idx.squeeze(1), idx.squeeze(1))
                    costs = reward.view(-1,1)  # [batch]
                    dynamic_hidden = self.dynamic_encoder(dynamic)
                    is_done = dynamic[:, 1].sum(1).eq(0).float()
                    prob_log = (prob_log.squeeze(1) * (1. - is_done))[:,None]


                if self.mask_fn is not None:
                    mask = self.mask_fn(mask, dynamic, distances, slope, idx.squeeze(1)).detach()
                now_idx = idx

            else:
                prob_log_all = prob_log + torch.log(beam_probs)  # (batch*beam, seq_len)
                prob_log_all = torch.cat(prob_log_all.chunk(beam_width, dim=0), dim=1)  # (batch, seq_len*beam)
                prob_log, idx = torch.topk(prob_log_all, beam_width, dim=1)  # both: (batch, beam_width)
                prob_log = prob_log.transpose(1, 0).contiguous().view(-1, 1)  # (batch*beam_width, 1)

                hpt = (idx // sequence_size).transpose(1, 0).contiguous().view(-1)  # from which beam, (batch*beam) 来自于哪一个beam
                idx = idx % sequence_size  # which node
                idx = idx.transpose(1, 0).contiguous().view(-1, 1)  # (batch*beam_width, 1)
                bb_idx = batch_idx + hpt * batch_size   # 为了在beam_width中取数据  [0,1,2,3,...,batch_size+来自于上次的哪一个beam]
                tours = torch.cat((tours[bb_idx], idx), dim=1)
                prob_log = prob_log[bb_idx]
                now_idx = now_idx[bb_idx]
                distances = distances[bb_idx]
                slope = slope[bb_idx]
                mask = mask[bb_idx]
                dynamic = dynamic[bb_idx]

                if self.update_fn is not None:
                    dynamic, reward = self.update_fn(dynamic, distances, slope, now_idx.squeeze(1), idx.squeeze(1))
                    costs = torch.cat((costs[bb_idx], reward.view(-1,1)), dim=1)
                    dynamic_hidden = self.dynamic_encoder(dynamic)
                    is_done = dynamic[:, 1].sum(1).eq(0).float()
                    prob_log = (prob_log.squeeze(1) * (1. - is_done))[:,None]

                if self.mask_fn is not None:
                    mask = self.mask_fn(mask, dynamic, distances, slope, idx.squeeze(1)).detach()
                now_idx = idx

            decoder_input = torch.gather(static, 2,
                                         idx.squeeze(1).view(-1, 1, 1)
                                         .expand(-1, input_size, 1)).detach()
        costs = torch.cat(costs.sum(-1)[:,None].chunk(beam_width, dim=0), dim=1)
        min_costs, argmincosts = costs.min(-1)

        min_idx = torch.LongTensor([i for i in range(batch_size)]).to(device) + argmincosts* batch_size
        min_tours = tours[min_idx, :]

        return min_costs, min_tours,


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
