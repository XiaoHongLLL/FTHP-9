# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.utils.data
import math
from transformer import Constants
import functools

class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data, opt, split):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        # 1. 基础数据加载
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]
        self.length = len(data)

        # 2. Dequantization (去量子化) - 核心修改
        # StackOverflow 数据是离散的，这对连续 Flow 模型是灾难性的。
        # 我们必须加上 U[0, 1] 噪声使其连续化。
        raw_gaps = [[elem['time_since_last_event'] for elem in inst[1:]] for inst in data]
        
        self.time_gap = []
        for seq in raw_gaps:
            # 对训练集和测试集都加噪声是标准做法，保证分布一致性
            # 加 1e-5 防止 0 值导致的 log 错误，再加随机噪声
            noisy_seq = [x + np.random.uniform(0, 1) for x in seq]
            self.time_gap.append(noisy_seq)

        # 3. 基于去量子化后的 Gap 重构 Cumulative Time
        # 保持 t_i = t_{i-1} + gap_i 的一致性
        self.time = []
        for i, inst in enumerate(data):
            # 获取起始时间（通常为0）
            current_t = inst[0]['time_since_start']
            seq_time = [current_t]
            for gap in self.time_gap[i]:
                current_t += gap
                seq_time.append(current_t)
            self.time.append(seq_time)

        # 4. 计算统计量 (使用去量子化后的数据)
        time_flat = [t for seq in self.time_gap for t in seq]
        time_flat = np.array(time_flat)

        if split == 'train':
            if opt.normalize == 'normal':
                mean_data = time_flat.mean().item()
                opt.mean_data = mean_data
                # 归一化
                self.time_gap = [[elem / mean_data for elem in inst] for inst in self.time_gap]
                self.time = [[elem / mean_data for elem in inst] for inst in self.time]
            
            if opt.normalize == 'log':
                mean_data = time_flat.mean().item()
                # 计算 log 统计量
                # 注意：此时 time_flat 已经是连续的，且 > 0
                log_time = np.log(time_flat + 1e-9)
                mean_log_data = log_time.mean().item()
                var_log_data = log_time.std().item()
                
                print(f'[Dataset] Log Stats: Mean={mean_log_data:.4f}, Std={var_log_data:.4f}')
                
                opt.mean_data = mean_data
                opt.mean_log_data = mean_log_data
                opt.var_log_data = var_log_data

                # 应用 Log 归一化
                self.time_gap = [
                    [(math.log(elem + 1e-9) - opt.mean_log_data) / opt.var_log_data for elem in inst]
                    for inst in self.time_gap]
            
            # 记录分位数用于评估
            opt.time_min = np.min(time_flat)
            opt.time_max = np.max(time_flat)
            opt.time_mean = np.mean(time_flat)
            opt.time_std = np.std(time_flat)
            # 避免 quantile 报错
            if len(time_flat) > 0:
                opt.eval_quantile = torch.arange(opt.eval_quantile_step, 1.0, opt.eval_quantile_step).to(opt.device)
            
        else:
            # Test/Dev 使用训练集的统计量
            if opt.normalize == 'normal':
                self.time_gap = [[elem / opt.mean_data for elem in inst] for inst in self.time_gap]
                self.time = [[elem / opt.mean_data for elem in inst] for inst in self.time]
            if opt.normalize == 'log':
                self.time_gap = [
                    [(math.log(elem + 1e-9) - opt.mean_log_data) / opt.var_log_data for elem in inst]
                    for inst in self.time_gap]

        self.max_len = max([len(inst) for inst in data])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.time[idx], self.time_gap[idx], self.event_type[idx]

def pad_time(insts):
    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])
    return torch.tensor(batch_seq, dtype=torch.float32)

def pad_type(insts):
    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])
    return torch.tensor(batch_seq, dtype=torch.long)

def collate_fn(insts):
    time, time_gap, event_type = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    return time, time_gap, event_type

def get_dataloader(data, opt, shuffle=True, split='train'):
    ds = EventData(data, opt, split)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    opt.max_len = max(opt.max_len, ds.max_len)
    return dl