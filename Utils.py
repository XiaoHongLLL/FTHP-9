# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformer.Layers import get_non_pad_mask

def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))

def log_likelihood(model, event_time, time_gap, event_type):
    """ Log-likelihood of sequence. (用于训练阶段计算 Loss) """
    non_pad_mask = get_non_pad_mask(event_type)
    return 0, 0 

def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """
    truth = types[:, 1:] - 1
    # 预测形状可能是 (B, L, C) 或 (B, L, N_samples, C)
    if prediction.ndim == 4:
        truth = truth.unsqueeze(2).expand(-1, -1, prediction.size(2))
        loss = loss_func(prediction.reshape(-1, prediction.size(-1)), truth.reshape(-1))
        loss = loss.view(truth.size())
        loss = loss.mean(2) # Average over samples
    else:
        loss = loss_func(prediction.reshape(-1, prediction.size(-1)), truth.reshape(-1))
        loss = loss.view(truth.size())
    
    return torch.sum(loss)

def evaluate_samples(t_sample, gt_t, type_sample, event_type, opt):
    """
    根据 SMURF-THP 论文计算指标: CS, CER, IL, CRPS, Accuracy
    【新增】: RMSE (通过返回 SSE Sum 实现)
    【关键修正】: Dequantization Mean Shift Correction (-0.5)
    """

    # ==========================================================
    # 1. 物理截断 (Physical Clamping)
    # ==========================================================
    if opt.normalize == 'log':
        t_sample = torch.clamp(t_sample, min=-10.0, max=10.0)

    # ==========================================================
    # 2. 反归一化 (Denormalize)
    # ==========================================================
    if opt.normalize == 'log':
        # 还原到 log 空间: x * std + mean
        # 然后 exp 还原到真实时间空间
        gt_t_real = torch.exp(gt_t * opt.var_log_data + opt.mean_log_data)
        t_sample_real = torch.exp(t_sample * opt.var_log_data + opt.mean_log_data)

    elif opt.normalize == 'normal':
        gt_t_real = gt_t * opt.mean_data
        t_sample_real = t_sample * opt.mean_data
    else:
        gt_t_real = gt_t
        t_sample_real = t_sample

    # ==========================================================
    # 3. [关键修正] Dequantization Mean Shift Correction
    # ==========================================================
    # 训练时我们加了 U[0, 1]，期望偏移了 +0.5。
    # 生成时，我们需要减去这个 0.5 才能对齐到原始整数网格。
    # 此外，时间间隔不能小于 0。
    t_sample_real = torch.clamp(t_sample_real - 0.5, min=0.0)
    
    # 对应的，为了计算精确的 RMSE，这里的 gt_t_real 应该是原始的整数数据
    # 但 Dataset 传进来的 gt_t 是加了噪声的。
    # 我们近似认为：gt_t_real (含噪) - 0.5 ≈ 原始整数 (在统计意义上)
    gt_t_real = torch.clamp(gt_t_real - 0.5, min=0.0)

    # ==========================================================
    # 4. 极值保护
    # ==========================================================
    max_val = getattr(opt, 'time_max', 1e7) * 5.0
    t_sample_real = torch.clamp(t_sample_real, max=max_val)

    # ==========================================================
    # 5. 准备 Mask
    # ==========================================================
    non_pad_mask = get_non_pad_mask(event_type)  # (B, L, 1)
    valid_mask = non_pad_mask[:, 1:].squeeze(2)  # (B, L-1)

    # ==========================================================
    # 6. 计算指标
    # ==========================================================

    # --- A. 计算覆盖率 (CS) ---
    target_quantiles = opt.eval_quantile # [0.05, ..., 0.95]
    t_pred_quantiles = torch.quantile(t_sample_real, target_quantiles, dim=-1) # (n_q, B, L-1)

    # 计算每个分位数是否覆盖了真实值
    hits_all = (gt_t_real.unsqueeze(0) <= t_pred_quantiles) * valid_mask.unsqueeze(0)
    batch_hit_counts = hits_all.sum(dim=(1, 2))

    # --- B. Interval Length (IL) ---
    t_median = torch.quantile(t_sample_real, 0.5, dim=-1)
    batch_il_sum = (t_median * valid_mask).sum().item()

    # --- C. CRPS ---
    num_samples = t_sample_real.size(-1)
    term1 = torch.abs(t_sample_real - gt_t_real.unsqueeze(-1)).mean(dim=-1)
    
    if num_samples > 100:
        # 近似计算以节省显存
        t_mean = t_sample_real.mean(dim=-1, keepdim=True)
        term2 = torch.abs(t_sample_real - t_mean).mean(dim=-1) * 2 
    else:
        # Sample Pairwise distance
        t_sample_perm = t_sample_real[:, :, torch.randperm(num_samples)]
        term2 = torch.abs(t_sample_real - t_sample_perm).mean(dim=-1)

    crps_map = term1 - 0.5 * term2
    batch_crps_sum = (crps_map * valid_mask).sum().item()

    # --- D. Type Accuracy ---
    truth = event_type[:, 1:] - 1
    # 兼容混合采样或 Argmax
    if isinstance(type_sample, tuple) or (isinstance(type_sample, torch.Tensor) and type_sample.ndim > 2):
        type_pred = type_sample if isinstance(type_sample, torch.Tensor) else type_sample[0]
        if type_pred.ndim > 2: 
             type_pred = type_pred.mode(dim=-1).values
    else:
        type_pred = type_sample
    
    batch_correct_type = (type_pred.eq(truth) * valid_mask).sum().item()

    # --- E. SSE (for RMSE) ---
    # 使用中位数作为点预测 (Robust prediction)
    se = (t_median - gt_t_real) ** 2     
    batch_sse_sum = (se * valid_mask).sum().item()

    # --- F. 总事件数 ---
    batch_total_events = valid_mask.sum().item()

    return {
        'hit_counts': batch_hit_counts,  # Tensor
        'il_sum': batch_il_sum,          # Float
        'crps_sum': batch_crps_sum,      # Float
        'correct_type': batch_correct_type, # Float
        'total_events': batch_total_events,  # Int
        'sse_sum': batch_sse_sum         # [新增] Float
    }

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 <= label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """
        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)

        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss.sum()