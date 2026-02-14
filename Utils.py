import math
import torch
import torch.nn as nn
import numpy as np
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
    """

    # ==========================================================
    # 1. 物理截断 (Physical Clamping)
    # ==========================================================
    if opt.normalize == 'log':
        t_sample = torch.clamp(t_sample, min=-10.0, max=8.0)

    # ==========================================================
    # 2. 反归一化 (Denormalize)
    # ==========================================================
    if opt.normalize == 'log':
        # 仅还原 Log 变换，保持在 (x / mean) 的尺度 (SMURF 逻辑)
        gt_t = torch.exp(gt_t * opt.var_log_data + opt.mean_log_data)
        t_sample = torch.exp(t_sample * opt.var_log_data + opt.mean_log_data)

    elif opt.normalize == 'normal':
        pass

    # ==========================================================
    # 3. 极值保护
    # ==========================================================
    max_val = getattr(opt, 'time_max', 1e7) * 2.0
    t_sample = torch.clamp(t_sample, max=max_val)

    # ==========================================================
    # 4. 准备 Mask
    # ==========================================================
    non_pad_mask = get_non_pad_mask(event_type)  # (B, L, 1)
    valid_mask = non_pad_mask[:, 1:].squeeze(2)  # (B, L-1)

    # ==========================================================
    # 5. 计算指标
    # ==========================================================

    # --- A. 计算覆盖率 (CS) ---
    target_quantiles = opt.eval_quantile # [0.05, ..., 0.95]
    t_pred_quantiles = torch.quantile(t_sample, target_quantiles, dim=-1) # (n_q, B, L-1)

    # 计算每个分位数是否覆盖了真实值
    hits_all = (gt_t.unsqueeze(0) <= t_pred_quantiles) * valid_mask.unsqueeze(0)
    batch_hit_counts = hits_all.sum(dim=(1, 2))

    # --- B. Interval Length (IL) - 【SMURF 逻辑】 ---
    t_median = torch.quantile(t_sample, 0.5, dim=-1)
    batch_il_sum = (t_median * valid_mask).sum().item()

    # --- C. CRPS ---
    num_samples = t_sample.size(-1)
    term1 = torch.abs(t_sample - gt_t.unsqueeze(-1)).mean(dim=-1)
    
    if num_samples > 100:
        # 近似计算以节省显存
        t_mean = t_sample.mean(dim=-1, keepdim=True)
        term2 = torch.abs(t_sample - t_mean).mean(dim=-1) * 2 
    else:
        t_sample_perm = t_sample[:, :, torch.randperm(num_samples)]
        term2 = torch.abs(t_sample - t_sample_perm).mean(dim=-1)

    crps_map = term1 - 0.5 * term2
    batch_crps_sum = (crps_map * valid_mask).sum().item()

    # --- D. Type Accuracy ---
    truth = event_type[:, 1:] - 1
    if isinstance(type_sample, tuple):
        type_sample = type_sample[0].argmax(dim=-1) if type_sample[0].ndim > 2 else type_sample[0]
    
    batch_correct_type = (type_sample.eq(truth) * valid_mask).sum().item()

    # --- [新增] E. RMSE 准备 (SSE Sum) ---
    # t_sample: (B, L, N_samples) -> 在外面被permute过，通常是 (B, L, N) 或者 (B, N, L) 需要确认
    # 注意：在 main.py 调用前 t_sample 已经是 (B, L, N) 或 (B, N, L)
    # 让我们检查上游调用：t_sample_norm = x1_norm.view(B, N, L).squeeze(-1).permute(0, 2, 1) -> (B, L, N)
    # 所以 dim=-1 是 sample 维度
    t_pred_median = t_sample.median(dim=-1).values # 取样本均值作为点预测 (B, L)
    se = (t_pred_median - gt_t) ** 2     # Squared Error
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