# test.py
import argparse
import numpy as np
import pickle
import torch
import os
from tqdm import tqdm
import pandas as pd # 引入 pandas 以便保存搜索结果

import transformer.Constants as Constants
import Utils
from preprocess.Dataset import get_dataloader
from transformer.Layers import get_non_pad_mask
from transformer.Models import FlowMatchingTHP
from flow_matching.solver import ODESolver


# --- Velocity Wrapper for torchdiffeq ---
class VelocityWrapper:
    def __init__(self, v_field, c_cond):
        self.v_field = v_field
        self.c_cond = c_cond

    def __call__(self, t, x):
        t_tensor = t * torch.ones_like(x)
        return self.v_field(x, t_tensor, self.c_cond)


def load_data_and_stats(opt):
    def load_pkl(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    # 1. 加载训练集 (为了计算 Mean/Std)
    train_data, num_types = load_pkl(opt.data + 'train.pkl', 'train')
    # 2. 加载测试集
    test_data, _ = load_pkl(opt.data + 'test.pkl', 'test')

    # 3. 重新计算统计量
    time_flat = []
    for i in train_data:
        time_flat += [elem['time_since_last_event'] for elem in i if elem['time_since_last_event'] > 0]
    time_flat = np.array(time_flat)

    opt.mean_data = time_flat.mean()

    if opt.normalize == 'log':
        time_flat_div = time_flat / opt.mean_data
        log_time = np.log(time_flat_div + 1e-9)
        opt.mean_log_data = log_time.mean()
        opt.var_log_data = log_time.std()

    opt.max_len = 0
    # 获取 Test Loader
    testloader = get_dataloader(test_data, opt, shuffle=False, split='test')
    return testloader, num_types


def evaluate_generation(model, testloader, opt):
    model.eval()
    solver = ODESolver(None)

    n_quantiles = len(opt.eval_quantile)
    total_hits = torch.zeros(n_quantiles, device=opt.device)
    total_il = 0
    total_crps = 0
    total_acc = 0
    total_events = 0
    
    # [Info] 打印当前的推断参数，方便调试
    print(f'[Config] Solver: {opt.solver_method}, Step: {opt.solver_step_size}, Clamp: {opt.clamp_threshold}, Mixture: {opt.mixture_sampling}')

    with torch.no_grad():
        for batch in tqdm(testloader, desc='  - (Testing)   ', leave=False):
            event_time, time_gap_norm, event_type = map(lambda x: x.to(opt.device), batch)

            non_pad_mask = get_non_pad_mask(event_type)
            enc_output = model.encoder(event_type, event_time, non_pad_mask)
            c = enc_output[:, :-1, :]  # (B, L-1, D)

            B, L, D = c.shape
            N = opt.n_samples

            # --- 混合采样 (Mixture Sampling) ---
            type_logits = model.type_predictor(c)
            
            if opt.mixture_sampling:
                # 使用多项式采样混合多个类型
                type_probs = torch.softmax(type_logits, dim=-1)
                flat_probs = type_probs.reshape(-1, type_probs.size(-1))
                sampled_type_indices = torch.multinomial(flat_probs, num_samples=N, replacement=True)
                sampled_type_indices = sampled_type_indices.view(B, L, N).permute(0, 2, 1)
                flat_sampled_types = sampled_type_indices.reshape(B * N, L)
                pred_types_input = flat_sampled_types + 1 
            else:
                # 传统的 Argmax (如果想做对比实验的话可以关掉 mixture)
                pred_types = type_logits.argmax(dim=-1)
                pred_types_input = (pred_types + 1).unsqueeze(1).repeat(1, N, 1).view(-1, L)

            pred_type_emb = model.encoder.event_type_emb(pred_types_input)
            c_expanded = c.unsqueeze(1).repeat(1, N, 1, 1).view(B * N, L, D)
            c_cond = model.type_fusion(torch.cat([c_expanded, pred_type_emb], dim=-1))

            # --- Flow Matching ---
            x0 = torch.randn(B * N, L, 1, device=opt.device)
            
            # [超参] 噪声缩放 (Temperature)
            if opt.fm_sigma_scale != 1.0:
                x0 = x0 * opt.fm_sigma_scale

            wrapper = VelocityWrapper(model.v_field, c_cond)
            solver.velocity_model = wrapper

            # [超参] 求解器设置完全由参数决定
            x1_norm = solver.sample(
                x0,
                torch.tensor([0.0, 1.0]),
                step_size=opt.solver_step_size,
                method=opt.solver_method
            )

            # [超参] 截断阈值
            if opt.normalize == 'log' and opt.clamp_threshold > 0:
                x1_norm = torch.clamp(x1_norm, min=-opt.clamp_threshold, max=opt.clamp_threshold)

            t_sample_norm = x1_norm.view(B, N, L).squeeze(-1).permute(0, 2, 1)
            pred_types_argmax = type_logits.argmax(dim=-1)

            metrics = Utils.evaluate_samples(
                t_sample_norm,
                time_gap_norm,
                pred_types_argmax,
                event_type,
                opt
            )

            if metrics['total_events'] > 0:
                total_hits += metrics['hit_counts']
                total_il += metrics['il_sum']
                total_crps += metrics['crps_sum']
                total_acc += metrics['correct_type']
                total_events += metrics['total_events']

    if total_events == 0: return None

    final_acc = total_acc / total_events
    final_crps = total_crps / total_events
    final_il = total_il / total_events

    actual_coverage = total_hits / total_events
    target_coverage = opt.eval_quantile
    mse = ((actual_coverage - target_coverage) ** 2).mean()
    final_cs = torch.sqrt(mse).item()

    idx_05 = int(0.5 / opt.eval_quantile_step) - 1
    if 0 <= idx_05 < len(actual_coverage):
        cov_05 = actual_coverage[idx_05].item()
        final_cer = abs(cov_05 - 0.5)
    else:
        final_cer = 0.0

    return {
        'Acc': final_acc,
        'CRPS': final_crps,
        'CS': final_cs,
        'CER': final_cer,
        'IL': final_il
    }


def main():
    parser = argparse.ArgumentParser()

    # Data & Model
    parser.add_argument('-data', required=True)
    parser.add_argument('-model_path', required=True)
    parser.add_argument('-normalize', type=str, default='log')
    
    # Model Architecture (Keep consistent with training)
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)
    parser.add_argument('-fm_sigma', type=float, default=0.1)

    # --- [关键修改] 推断超参数 ---
    parser.add_argument('-solver_method', type=str, default='euler', choices=['euler', 'rk4'])
    parser.add_argument('-solver_step_size', type=float, default=0.05, help='Step size for ODE solver')
    parser.add_argument('-clamp_threshold', type=float, default=-1.0, help='Clamp threshold for log-space time. -1 to disable.')
    parser.add_argument('-mixture_sampling', action='store_true', help='Enable mixture sampling for types')
    parser.add_argument('-fm_sigma_scale', type=float, default=1.0, help='Scale initial noise x0 variance')
    parser.add_argument('-n_samples', type=int, default=100)

    # Eval
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-eval_quantile_step', type=float, default=0.05)
    parser.add_argument('-seed', type=int, default=2023)
    parser.add_argument('-csv_out', type=str, default=None, help='Append results to a CSV file')

    opt = parser.parse_args()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.eval_quantile = torch.arange(opt.eval_quantile_step, 1.0, opt.eval_quantile_step, device=opt.device)

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    testloader, num_types = load_data_and_stats(opt)
    model = FlowMatchingTHP(num_types, opt)
    
    # Inject Stats
    model.mean_log_data = getattr(opt, 'mean_log_data', 0)
    model.var_log_data = getattr(opt, 'var_log_data', 1)
    model.mean_data = getattr(opt, 'mean_data', 1)
    model.to(opt.device)

    if os.path.exists(opt.model_path):
        checkpoint = torch.load(opt.model_path, map_location=opt.device)
        model.load_state_dict(checkpoint)
    else:
        print(f'[Error] Model path not found: {opt.model_path}')
        return

    # Run Eval
    results = evaluate_generation(model, testloader, opt)

    # Print
    if results:
        print(f"Result: CS={results['CS']:.4f} | CRPS={results['CRPS']:.4f} | CER={results['CER']:.4f} | IL={results['IL']:.4f}")
        
        # 保存到 CSV 方便脚本分析
        if opt.csv_out:
            import csv
            file_exists = os.path.isfile(opt.csv_out)
            with open(opt.csv_out, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    # Header
                    writer.writerow(['Method', 'Step', 'Clamp', 'Mixture', 'CS', 'CRPS', 'CER', 'IL'])
                writer.writerow([
                    opt.solver_method, opt.solver_step_size, opt.clamp_threshold, 
                    opt.mixture_sampling, 
                    f"{results['CS']:.4f}", f"{results['CRPS']:.4f}", f"{results['CER']:.4f}", f"{results['IL']:.4f}"
                ])

if __name__ == '__main__':
    main()