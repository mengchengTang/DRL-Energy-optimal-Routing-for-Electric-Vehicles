import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches
from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#decice = torch.device("cpu")

def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 args,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 update_dynamic = None,
                 update_mask = None
                 ):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0  # 解码温度
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.update_dynamic = update_dynamic
        self.update_mask = update_mask
        self.n_heads = n_heads
        step_context_dim = embedding_dim + 1  # 当前只考虑剩余容量、剩余soc、剩余时间
        # 图上的一些参数
        self.start_soc = args.Start_SOC
        self.t_limit = args.t_limit
        self.custom_num = args.num_nodes
        self.charging_num = args.charging_num
        # 对各个点进行编码的方式
        node_dim = 3  # x, y, 需求
        self.init_embed_depot_and_station = nn.Linear(2, embedding_dim)
        self.init_embed_station = nn.Linear(2, embedding_dim)
        self.init_embed = nn.Linear(node_dim, embedding_dim)  # 进入编码器之前进行一次编码
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input,):
        """
        :param input: 原始数据为一个元组 (static, dynamic, distances, slopes)
        :param return_pi: 这里选择返回所选序列，因为需要画图
        :return: tour_idx, tour_logp, R 保持与pointer的一致
        """
        # 处理输入数据，
        batch_size, _, num_node = input[0].shape
        lenth = len(input)
        if lenth == 4:
            static, dynamic, distances, slope = input
            static = static.float().to(device)
            dynamic = dynamic.float().to(device)
            distances = distances.float()
            slope = slope.float()
        else:
            static, dynamic, Elevations = input
            static = static.float().to(device)
            dynamic = dynamic.float().to(device)
            distances = torch.zeros(batch_size, num_node, num_node, device=device,)  # 计算距离矩阵
            for i in range(num_node):
                distances[:, i] = torch.sqrt(torch.sum(torch.pow(static[:, :, i:i + 1]- static[:, :, :], 2), dim=1))
            slope = torch.zeros(batch_size, num_node, num_node, device=device)
            for i in range(num_node):
                slope[:, i] = torch.clamp(torch.div((Elevations[:, i:i + 1] - Elevations[:, :]), distances[:, i] + 0.000001), min=-0.10,max=0.10)

        information = torch.cat((static, dynamic),dim=1).permute(0, 2, 1)
        embeddings, _ = self.embedder(self._init_embed(information[:, :, [0,1,3]]))

        _log_p,  pi,  cost = self._inner(information, distances, slope, embeddings)
        #  概率对数值
        ll = self._calc_log_likelihood(_log_p, pi)

        return pi , ll , cost

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p

    def _init_embed(self, input):
        # 对点的原始信息进行嵌入
        return torch.cat(
            (
                self.init_embed_depot_and_station(input[:, 0:1, 0:2] / 100),
                self.init_embed_station(input[:, 1 : self.charging_num + 1, 0:2] / 100),
                self.init_embed(torch.cat((input[:, self.charging_num + 1:, 0:2] / 100, input[:, self.charging_num + 1:,2:3]), dim= 2)),
            ),
            dim=1)


    def _inner(self, information, distances, slopes, embeddings):
        """
        :param information: 包含动态元素以及静态元素[batch_size, sequences_size, dim=6]
        :param distances:  距离矩阵 [batch_size, sequences_size, sequences_size]
        :param slopes:   坡度矩阵 [batch_size, sequences_size, sequences_size]
        :param embeddings: information经过编码得到的信息 [batch_size, sequences_size, embedding_dim]
        :return: tour_idx, tour_logp, R
        """
        outputs = []
        tour_idx = []
        R = []
        batch_size, sequences_size, _ = information.size()
        # static = information[:,:,0:2]  # 取出静态元素
        dynamic = information[:,:,2:].permute(0, 2, 1)   # 取出动态元素
        fixed = self._precompute(embeddings)  # NamedTuple,为编码器解码器传递信息
        i = 0
        max_steps =  100  # 最大步数
        mask = torch.ones(batch_size, sequences_size, device=device)  # 屏蔽数组
        now_idx = torch.zeros(batch_size, device=device)  # 当前的坐标

        for _ in range(max_steps):

            if not mask.byte().any():  # 当mask任意一个都为0，表示所有点都不能去
                break
            # 选择下一访问的节点 log_p:[batch_size, 1, num_node] mask:[batch_size, num_node]
            log_p = self._get_log_p(fixed, dynamic, now_idx, mask)
            selected = self._select_node(log_p.exp()[:, 0, :], mask)
            # 更新动态信息
            dynamic, reward = self.update_dynamic(dynamic, distances, slopes, now_idx, selected)
            # 更新屏蔽信息
            mask = self.update_mask(dynamic, distances, slopes, selected.data).detach()
            now_idx = selected
            # 添加当前选择结果
            outputs.append(log_p[:, 0, :])
            tour_idx.append(selected)
            R.append(reward)
            i += 1

        R = torch.cat(R, dim=1)  # (batch_size, seq_len)

        return torch.stack(outputs, 1), torch.stack(tour_idx, 1), R


    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param batch:数据
        :param batch_rep:指的是采样宽度
        :param iter_rep: 迭代的伦次
        :return:
        """
        static, dynamic, distances, slope = input
        static = static[None, ...].expand(batch_rep, *static.size()).contiguous().view(-1, *static.size()[1:])
        dynamic = dynamic[None, ...].expand(batch_rep, *dynamic.size()).contiguous().view(-1, *dynamic.size()[1:])
        distances = distances[None, ...].expand(batch_rep, *distances.size()).contiguous().view(-1, *distances.size()[1:])
        slope = slope[None, ...].expand(batch_rep, *slope.size()).contiguous().view(-1,*slope.size()[1:])
        batch = static, dynamic, distances, slope

        costs = []
        pis = []
        for i in range(iter_rep):
            tour_idx, tour_logp, R = self.forward(batch)  # tour_idx的形状为[batch,sequence_len]
            cost = torch.sum(R, dim=1)  # [batch]
            costs.append(cost.view(batch_rep, -1).t())  # 每一个元素的形状为[batch,batch_rep]
            pis.append(tour_idx.view(batch_rep, -1, tour_idx.size(-1)).transpose(0, 1))
        costs = torch.cat(costs, 1)  # 这里是将迭代伦次的所有batch拼接起来，形状为 [batch,iter_rep]

        # 对不同伦次之间的索引进行拼接
        max_length = max(pi.size(-1) for pi in pis)
        pis = torch.cat(
            [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
            1)
        mincosts, argmincosts = costs.min(-1)  # [batch]

        minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]  # [batch_size,min_sequence]

        return mincosts, minpis   # [batch],这个就是采样出来最小的


    def _select_node(self, probs, mask):
        '''
        用已知的概率和屏蔽策略选出下一需要访问的点
        :param probs: [batch, num_node]
        :param mask:  [batch, num_node]
        :return: [batch_size]
        '''
        mask = mask.eq(0).type(torch.int)
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sample":
            selected = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected


    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)


    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )


    def _get_log_p(self, fixed, dynamic, now_idx, mask, normalize=True):

        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, dynamic, now_idx))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p

    def _get_parallel_step_context(self, embeddings, dynamic, now_idx):
        """
        负责返回每个时间步车辆的当前点信息和当前的转载量
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param now_idx: (batch_size, 1)
        :param dynamic: 动态信息
        :return: (batch_size, 1, embedding_dim + 1)
        """
        current_node = now_idx[:,None].type(torch.int64)
        batch_size, num_steps = current_node.size()
        dynamic = dynamic.permute(0, 2, 1)
        return torch.cat(
            (
                torch.gather(
                    embeddings,
                    1,
                    current_node.contiguous()
                    .view(batch_size, num_steps, 1)
                    .expand(batch_size, num_steps, embeddings.size(-1))
                ).view(batch_size, num_steps, embeddings.size(-1)),
                dynamic[:, 0:1, 0:1],                   # [batch_size, 1, 1]
                # dynamic[:, 2:3, 0:1] / self.start_soc,  # 对特征进行归一化至 0~1 这个区间
                # dynamic[:, 3:4, 0:1] / self.t_limit
            ),
            -1
        )


    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        """
        用于返回每个节点被选中的概率(log_p)
        :param query: [batch_size, 1, embedding_dim]
        :param glimpse_K:
        :param glimpse_V:
        :param logit_K:
        :param mask: [batch, num_node]
        :return: logits [batch_size, 1, num_node]
        """

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads
        mask = mask[:,None,:].eq(0)  # 等于0的就设为负无穷不准去

        # 将glimpse所需的Q变形为：(n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
