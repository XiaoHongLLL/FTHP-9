import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import transformer.Modules as Modules


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = Modules.ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)  # 保持 LayerNorm 实例

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # --- 【Pre-LN】 ---
        # 1. 先进行 LayerNorm
        q_norm = self.layer_norm(q)
        k_norm = self.layer_norm(k)
        v_norm = self.layer_norm(v)
        # --- 【Pre-LN 结束】 ---

        # 2. 使用归一化后的 q, k, v
        q = self.w_qs(q_norm).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k_norm).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v_norm).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) if mask is not None else None  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))

        # 3. 直接进行残差连接 (去掉了外层的 LayerNorm)
        output = output + residual

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)  # 保持 LayerNorm 实例
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        # --- 【Pre-LN】 ---
        # 1. 先进行 LayerNorm
        x_norm = self.layer_norm(x)
        # --- 【Pre-LN 结束】 ---

        # 2. 使用归一化后的 x
        output = x_norm.transpose(1, 2)
        output = self.w_2(self.relu(self.w_1(output)))
        output = output.transpose(1, 2)

        output = self.dropout(output)

        # 3. 直接进行残差连接 (去掉了外层的 LayerNorm)
        output = output + residual

        return output