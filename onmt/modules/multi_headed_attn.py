""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn

from onmt.utils.misc import generate_relative_positions_matrix,\
                            relative_matmul
# from onmt.utils.misc import aeq


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    # TODO:edge_type, fusion, add in opts.py
    def __init__(self, head_count, model_dim, dropout=0.1,
                 max_relative_positions=0, edge_type=0,
                 fusion="cat", gate=False):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.max_relative_positions = max_relative_positions

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

        if edge_type > 0:
            self.edge_type = edge_type
            self.linear_attention = nn.ModuleList(
                [nn.Linear(self.dim_per_head, 1)
                 for _ in range(2 * (edge_type+1))] # there edge-aware sub graph and one the whole graph
            )
            self.fusion = fusion
            self.sub_final_linear = nn.Linear(model_dim+model_dim*(edge_type), model_dim)

        self.gate = gate
        if gate:
            self.gate_linear = nn.ModuleList(
                [nn.Linear(model_dim, model_dim)
                 for _ in range(edge_type + 2)]
            )

    def forward(self, key, value, query, grh=None, mask=None,
                layer_cache=None, attn_type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0 and attn_type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        if self.max_relative_positions > 0 and attn_type == "self":
            context = unshape(context_original
                              + relative_matmul(drop_attn,
                                                relations_values,
                                                False))
        else:
            context = unshape(context_original)
        if self.gate:
            gate = torch.sigmoid(self.gate_linear[0](context))
            gate_context = gate * context
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        ## above is fully-connected graph
        ## Multi-View self-attention
        if grh is not None:
            assert query_len == key_len
            #assert key_len-1 != grh[0][-1][0], "the num of nodes is not consistent"
            views = []
            index = [(0,1), (2,3), (4,5), (6,7)]

            # whole sub graph
            h_i = self.linear_attention[index[-1][0]](value)
            h_j = self.linear_attention[index[-1][1]](value)
            e = nn.functional.leaky_relu(h_i + h_j.transpose(2, 3))  # default alpha=0.01, but =0.2 in tf

            grh_mask = torch.ones_like(grh)
            adj = (grh_mask < grh).unsqueeze(1).expand(-1, self.head_count, -1, -1)
            zero = torch.ones_like(e) * (-9e15)

            e_shape = e.shape
            attention = self.softmax(e.where(adj > 0, zero))  # 17 8 56 56

            whole_sub_view = torch.matmul(attention, value)
            if self.gate:
                gate = torch.sigmoid(self.gate_linear[-1](unshape(whole_sub_view)))
                views.append(gate * unshape(whole_sub_view))
            else:
                views.append(unshape(whole_sub_view))

            # edge-aware sub graph
            for i in range(self.edge_type-1):
                h_i = self.linear_attention[index[i][0]](value)
                h_j = self.linear_attention[index[i][1]](value)
                e = nn.functional.leaky_relu(h_i + h_j.transpose(2,3)) # default alpha=0.01, but =0.2 in tf

                label_id = i+2 # +2 because the followed is ones_like, so 1 can't be the edge type
                grh_mask = torch.ones_like(grh) * label_id
                eye = (torch.eye(grh.size(-1), dtype=torch.int64) * (4-label_id)).cuda() # here doesnt support multi-gpu
                grh_mask = grh_mask + eye
                adj = (grh_mask == grh).unsqueeze(1).expand(-1, self.head_count, -1, -1)
                zero = torch.ones_like(e) * (-9e15)

                e_shape = e.shape
                attention = self.softmax(e.where(adj>0, zero)) # 17 8 56 56

                sub_view = torch.matmul(attention, value)
                if self.gate:
                    gate = torch.sigmoid(self.gate_linear[i+1](unshape(sub_view)))
                    views.append(gate * unshape(sub_view))
                else:
                    views.append(unshape(sub_view))

            if self.fusion == "cat":
                ## TODO: MAX_P LSTM
                if self.gate:
                    Views = [gate_context] + views
                else:
                    Views = [context] + views
                output = torch.cat(Views, dim=-1)

                return self.sub_final_linear(output), top_attn

        output = self.final_linear(context)

        return output, top_attn

    def update_dropout(self, dropout):
        self.dropout.p = dropout
