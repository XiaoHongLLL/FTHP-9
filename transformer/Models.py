import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint  # 【新增】用于计算 Exact NLL

import transformer.Constants as Constants
from transformer.Layers import Encoder, get_non_pad_mask
from flow_matching.path import ConditionalFlowMatcher
from flow_matching.loss import FlowMatchingLoss


# --- 1. GMM 分类头 (保持不变) ---
class GMMClassifierHead(nn.Module):
    def __init__(self, d_model, num_types):
        super().__init__()
        self.d_model = d_model
        self.num_types = num_types
        self.centers = nn.Parameter(torch.randn(num_types, d_model))

    def forward(self, z):
        B, L, D = z.shape
        z_flat = z.reshape(-1, D)
        dists = torch.cdist(z_flat, self.centers, p=2).pow(2)
        logits = -0.5 * dists
        return logits.view(B, L, self.num_types)


# --- 2. 门控融合层 (保持不变) ---
class GatedFusion(nn.Module):
    """
    使用门控机制融合 History Context (h) 和 Target Type Embedding (e).
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear_h = nn.Linear(d_model, d_model)
        self.linear_e = nn.Linear(d_model, d_model)
        
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, type_emb):
        concat = torch.cat([h, type_emb], dim=-1)
        g = self.gate_net(concat) 
        fused = g * self.linear_h(h) + (1 - g) * self.linear_e(type_emb)
        out = self.out_proj(fused)
        out = self.dropout(out)
        out = self.norm(out + h)
        return out


# --- 3. AdaINBlock (保持不变) ---
class AdaINBlock(nn.Module):
    def __init__(self, d_hid, d_cond):
        super().__init__()
        self.norm = nn.GroupNorm(8, d_hid)
        self.linear1 = nn.Linear(d_hid, d_hid)
        self.linear2 = nn.Linear(d_hid, d_hid)
        self.act = nn.GELU()
        self.scale_shift_gen = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_cond, 2 * d_hid)
        )

    def forward(self, x, c):
        residual = x
        h = self.norm(x)
        c_perm = c.permute(0, 2, 1)
        scale_shift = self.scale_shift_gen(c_perm.transpose(1, 2)).transpose(1, 2)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        h = h * (1 + scale) + shift
        h = h.permute(0, 2, 1)
        h = self.act(self.linear1(h))
        h = self.linear2(h)
        h = h.permute(0, 2, 1)
        return residual + h


# --- 4. VectorField (保持不变) ---
class VectorField_Optimized(nn.Module):
    def __init__(self, d_in, d_cond, d_hid, d_out, num_blocks=4):
        super().__init__()
        self.t_embed_layer = nn.Sequential(
            nn.Linear(1, d_hid),
            nn.GELU(),
            nn.Linear(d_hid, d_hid)
        )
        self.initial_mapping = nn.Linear(d_in + d_hid, d_hid)
        self.blocks = nn.ModuleList([
            AdaINBlock(d_hid, d_cond) for _ in range(num_blocks)
        ])
        self.final_output = nn.Linear(d_hid, d_out)

    def forward(self, x, t, c):
        t_embed = self.t_embed_layer(t)
        x_input = torch.cat([x, t_embed], dim=-1)
        h = self.initial_mapping(x_input)
        h = h.permute(0, 2, 1)
        for block in self.blocks:
            h = block(h, c)
        h = h.permute(0, 2, 1)
        out = self.final_output(h)
        return out


# --- 5. FlowMatchingTHP (核心修改) ---
class FlowMatchingTHP(nn.Module):

    def __init__(self, num_types, config):
        super().__init__()
        self.config = config
        self.num_types = num_types
        self.normalize = config.normalize

        self.encoder = Encoder(
            num_types=num_types,
            d_model=config.d_model,
            d_inner=config.d_inner_hid,
            n_layers=config.n_layers,
            n_head=config.n_head,
            d_k=config.d_k,
            d_v=config.d_v,
            dropout=config.dropout,
        )

        self.type_predictor = GMMClassifierHead(config.d_model, num_types)
        self.type_loss_func = nn.CrossEntropyLoss(ignore_index=-1)

        self.type_fusion = GatedFusion(config.d_model, dropout=config.dropout)

        self.x_dim = 1
        self.v_field = VectorField_Optimized(
            d_in=self.x_dim,
            d_cond=config.d_model,
            d_hid=config.d_inner_hid,
            d_out=self.x_dim,
            num_blocks=4
        )

        self.flow_matcher = ConditionalFlowMatcher(sigma=config.fm_sigma)
        self.fm_loss_func = FlowMatchingLoss()

        self.mean_log_data = 0.0
        self.var_log_data = 1.0
        self.mean_data = 1.0

    def forward(self, event_type, event_time, time_gap_norm):
        non_pad_mask = get_non_pad_mask(event_type)

        # 1. Encode History
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        c = enc_output[:, :-1, :]

        # 2. Predict Type
        type_logits = self.type_predictor(c)

        # 3. Prepare Flow Condition
        target_type_idx = event_type[:, 1:]
        target_type_emb = self.encoder.event_type_emb(target_type_idx)

        c_cond = self.type_fusion(c, target_type_emb)

        # 4. Flow Matching
        x_1 = time_gap_norm.unsqueeze(-1)
        x_0 = torch.randn_like(x_1) * self.config.fm_sigma 
        
        t = torch.rand(x_1.shape[0], x_1.shape[1], 1, device=x_1.device)
        x_t, u_t = self.flow_matcher.sample_conditional_path(x_0, x_1, t)

        v_pred = self.v_field(x_t, t, c_cond)

        prediction = {
            'v_pred': v_pred,
            'u_t': u_t,
            'type_logits': type_logits
        }

        return enc_output, prediction

    def compute_loss_diagnostic(self, prediction, event_type):
        v_pred = prediction['v_pred']
        u_t = prediction['u_t']
        type_logits = prediction['type_logits']

        target_type = event_type[:, 1:]
        mask = (target_type != Constants.PAD)

        mask_fm = mask.unsqueeze(-1).float()
        fm_loss = self.fm_loss_func(v_pred, u_t, mask_fm)

        target_labels = target_type - 1
        target_labels[~mask] = -1

        type_loss = self.type_loss_func(
            type_logits.reshape(-1, self.num_types),
            target_labels.reshape(-1)
        )

        total_loss = fm_loss + self.config.loss_lambda * type_loss

        return total_loss, fm_loss.item(), type_loss.item()
    
    # 【新增】计算 Exact NLL 的核心方法
    def get_exact_log_likelihood(self, event_type, event_time, time_gap_norm):
        """
        计算 Flow Matching 模型的真实 Log-Likelihood。
        会自动将 NLL 从归一化空间还原到原始 Log-Time 空间，以便与 THP 对比。
        """
        # 1. 准备 Condition
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        c = enc_output[:, :-1, :] # History Context
        
        target_type_idx = event_type[:, 1:]
        target_type_emb = self.encoder.event_type_emb(target_type_idx)
        c_cond = self.type_fusion(c, target_type_emb) 

        # 2. 准备数据 x1 (Data)
        valid_mask = non_pad_mask[:, 1:].squeeze(-1).bool() # (B, L)
        
        x1 = time_gap_norm.unsqueeze(-1) # (B, L, 1)
        
        # 筛选有效数据 (N_events, 1)
        x1_flat = x1[valid_mask] 
        c_cond_flat = c_cond[valid_mask] 
        
        if x1_flat.shape[0] == 0:
            return torch.tensor(0.0, device=x1.device), 0

        # 3. 定义带散度的 ODE 函数 (这里必须完整定义，不能省略)
        def ode_func(t, states):
            x = states[0]
            with torch.enable_grad():
                x.requires_grad_(True)
                
                # 适配 VectorField_Optimized 的 3D 输入需求: (N, 1, D)
                x_in = x.unsqueeze(1) # (N, 1, 1)
                t_in = t * torch.ones(x.shape[0], 1, 1, device=x.device)
                c_in = c_cond_flat.unsqueeze(1) # (N, 1, D)
                
                v_out = self.v_field(x_in, t_in, c_in) # (N, 1, 1)
                v = v_out.squeeze(1) # (N, 1)
                
                # 计算散度 (Divergence) = dv/dx
                grad_v = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
                divergence = grad_v.view(-1, 1)
                
            return v, divergence

        # 4. 逆向积分 (t=1 -> t=0)
        z_t0 = x1_flat
        delta_logp_t0 = torch.zeros(x1_flat.shape[0], 1, device=x1.device)
        
        # 使用高精度求解器计算 NLL
        times = torch.tensor([1.0, 0.0], device=x1.device)
        state_t = odeint(
            ode_func,
            (z_t0, delta_logp_t0),
            times,
            atol=1e-5,
            rtol=1e-5,
            method='dopri5'
        )
        
        z_0 = state_t[0][-1]      
        delta_logp = state_t[1][-1] 
        
        # 5. 计算 归一化空间 的 NLL
        sigma_fm = self.config.fm_sigma
        log_p_z0 = -0.5 * math.log(2 * math.pi) - math.log(sigma_fm) - 0.5 * (z_0 / sigma_fm)**2
        log_prob_norm = log_p_z0 + delta_logp
        nll_norm = -log_prob_norm.sum()
        
        # 6. 还原到原始空间 (Change of Variables)
        # NLL_orig = NLL_norm + sum( log(t) + log(data_sigma) )
        
        data_mu = self.mean_log_data
        data_sigma = self.var_log_data
        
        # 恢复 log(t)
        log_t = x1_flat * data_sigma + data_mu
        
        # 雅可比校正项
        jacobian_term = log_t + math.log(data_sigma)
        
        # 总 NLL
        nll_orig = nll_norm + jacobian_term.sum()
        
        return nll_orig, x1_flat.shape[0]

    def denormalize_time(self, time_norm):
        if self.normalize == 'log':
            return torch.exp(time_norm * self.var_log_data + self.mean_log_data)
        elif self.normalize == 'normal':
            return time_norm * self.mean_data
        else:
            return time_norm