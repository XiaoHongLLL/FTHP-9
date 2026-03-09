# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pickle
import time
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import pandas as pd
import torch.nn.functional as F

import transformer.Constants as Constants
import Utils
from preprocess.Dataset import get_dataloader
from transformer.Layers import get_non_pad_mask
from transformer.Models import FlowMatchingTHP
from flow_matching.solver import ODESolver

torch.autograd.set_detect_anomaly(True)

def prepare_dataloader(opt):
    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    opt.max_len = 0
    trainloader = get_dataloader(train_data, opt, shuffle=True, split='train')
    devloader = get_dataloader(dev_data, opt, shuffle=False, split='dev')
    testloader = get_dataloader(test_data, opt, shuffle=False, split='test')

    return trainloader, devloader, testloader, num_types

def train_epoch(model, training_data, optimizer, opt):
    model.train()
    total_loss = 0
    total_fm = 0
    total_type = 0
    total_events = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        event_time, time_gap_norm, event_type = map(lambda x: x.to(opt.device), batch)
        optimizer.zero_grad()
        
        enc_out, prediction = model(event_type, event_time, time_gap_norm)
        
        loss, fm_val, type_val = model.compute_loss_diagnostic(prediction, event_type)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        num_event = (event_type[:, 1:] != Constants.PAD).sum().item()
        total_loss += loss.item() * num_event
        total_fm += fm_val * num_event
        total_type += type_val * num_event
        total_events += num_event

    return total_loss / total_events, total_fm / total_events, total_type / total_events

def eval_epoch(model, validation_data, eval_generation, opt):
    model.eval()

    if not eval_generation:
        total_loss = 0
        total_correct = 0
        total_events = 0
        with torch.no_grad():
            for batch in validation_data:
                event_time, time_gap_norm, event_type = map(lambda x: x.to(opt.device), batch)
                _, prediction = model(event_type, event_time, time_gap_norm)
                loss, _, _ = model.compute_loss_diagnostic(prediction, event_type)
                
                pred_type = prediction['type_logits'].argmax(dim=-1)
                truth = event_type[:, 1:] - 1
                mask = (truth != -1)
                correct = (pred_type[mask] == truth[mask]).sum().item()
                num_event = mask.sum().item()
                
                total_loss += loss.item() * num_event
                total_correct += correct
                total_events += num_event
        
        avg_loss = total_loss / total_events if total_events > 0 else 0
        avg_acc = total_correct / total_events if total_events > 0 else 0
        return avg_loss, avg_acc

    else:
        # Top-K Mixture Generation Logic
        class VelocityWrapper:
            def __init__(self, v_field, c_cond):
                self.v_field = v_field
                self.c_cond = c_cond
            def __call__(self, t, x):
                t_tensor = t * torch.ones_like(x)
                return self.v_field(x, t_tensor, self.c_cond)

        solver = ODESolver(None)

        n_quantiles = len(opt.eval_quantile)
        total_hits = torch.zeros(n_quantiles, device=opt.device)
        total_il = 0
        total_crps = 0
        total_acc = 0
        total_events = 0
        
        total_time_nll = 0.0
        total_type_nll = 0.0
        total_nll_events = 0
        total_sse = 0.0

        with torch.no_grad():
            for batch in tqdm(validation_data, desc='  - (Sampling)   ', leave=False):
                event_time, time_gap_norm, event_type = map(lambda x: x.to(opt.device), batch)

                # --- 1. NLL Calculation ---
                if hasattr(model, 'get_exact_log_likelihood'):
                    batch_time_nll_sum, batch_n_events = model.get_exact_log_likelihood(event_type, event_time, time_gap_norm)
                    
                    non_pad_mask = get_non_pad_mask(event_type)
                    enc_output = model.encoder(event_type, event_time, non_pad_mask, time_gap_norm)
                    c = enc_output[:, :-1, :]
                    
                    c_type = model.type_projector(c)
                    type_logits = model.type_predictor(c_type)
                    
                    target_labels = event_type[:, 1:] - 1
                    mask = (target_labels != -1)
                    
                    type_nll_loss = F.cross_entropy(
                        type_logits.view(-1, model.num_types), 
                        target_labels.view(-1), 
                        reduction='none',
                        ignore_index=-1
                    )
                    type_nll_sum = type_nll_loss.sum()
                    
                    total_time_nll += batch_time_nll_sum.item()
                    total_type_nll += type_nll_sum.item()
                    total_nll_events += batch_n_events

                # --- 2. Generation with Top-K Mixture ---
                non_pad_mask = get_non_pad_mask(event_type)
                enc_output = model.encoder(event_type, event_time, non_pad_mask, time_gap_norm)
                c = enc_output[:, :-1, :]
                
                # Decouple
                c_type = model.type_projector(c)
                c_time = model.time_projector(c)

                B, L, D = c.shape
                K = 3  # Top-3 Mixture

                # 2.1 获取 Top-K 概率
                type_logits = model.type_predictor(c_type)
                type_probs = torch.softmax(type_logits, dim=-1) # (B, L, NumTypes)
                topk_probs, topk_indices = torch.topk(type_probs, K, dim=-1) # (B, L, K)
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True) # Normalize

                # 2.2 准备 Mixture Condition
                c_time_expanded = c_time.unsqueeze(2).expand(-1, -1, K, -1)
                topk_type_emb = model.encoder.event_type_emb(topk_indices + 1)
                
                # Fusion
                c_cond_k = model.type_fusion(c_time_expanded, topk_type_emb)
                c_cond_flat = c_cond_k.view(B * L * K, D)

                # 2.3 ODE 求解 (Single Path per K, Total K paths)
                x0 = torch.randn(B * L * K, 1, 1, device=opt.device) * opt.fm_sigma
                wrapper = VelocityWrapper(model.v_field, c_cond_flat.unsqueeze(1))
                solver.velocity_model = wrapper

                x1_flat = solver.sample(
                    x0,
                    torch.tensor([0.0, 1.0]),
                    method=opt.solver_method,
                    step_size=opt.solver_step_size
                )
                
                if opt.normalize == 'log':
                    x1_flat = torch.clamp(x1_flat, min=-10.0, max=10.0)

                # 2.4 计算 Weighted Mean (用于 RMSE)
                t_pred_k_norm = x1_flat.view(B, L, K)
                t_pred_k_real = model.denormalize_time(t_pred_k_norm)
                
                # Expectation: Sum(Prob_k * Time_k)
                t_pred_mean = (t_pred_k_real * topk_probs).sum(dim=-1) # (B, L)

                # 2.5 计算 RMSE (使用 Top-K 期望值)
                if opt.normalize == 'log':
                    gt_t_real = torch.exp(time_gap_norm * opt.var_log_data + opt.mean_log_data)
                elif opt.normalize == 'normal':
                    gt_t_real = time_gap_norm * opt.mean_data
                else:
                    gt_t_real = time_gap_norm
                
                mask_loss = (event_type[:, 1:] != Constants.PAD)
                
                # [RMSE 修正逻辑]
                t_pred_integer = torch.clamp(t_pred_mean - 0.5, min=0.0)
                gt_integer = torch.floor(gt_t_real)
                
                diff = (t_pred_integer - gt_integer)[mask_loss]
                squared_error = (diff ** 2).sum().item()
                total_sse += squared_error
                total_events += mask_loss.sum().item()

                # --- 3. 用于 Utils 指标计算的 Argmax 采样 ---
                # [修复] 修正了 repeat 的维度错误
                N = opt.n_samples
                best_type = topk_indices[:, :, 0] # Top-1
                pred_types_input = best_type + 1
                pred_type_emb_best = model.encoder.event_type_emb(pred_types_input)
                c_cond_best = model.type_fusion(c_time, pred_type_emb_best)
                
                # [Fix] 使用 repeat_interleave 替代 unsqueeze+repeat，避免维度报错
                # (B, L, D) -> (B*N, L, D)
                c_cond_expanded = c_cond_best.repeat_interleave(N, dim=0)
                
                x0_samples = torch.randn(B * N, L, 1, device=opt.device) * opt.fm_sigma
                wrapper_samples = VelocityWrapper(model.v_field, c_cond_expanded)
                solver.velocity_model = wrapper_samples
                
                x1_samples = solver.sample(x0_samples, torch.tensor([0.0, 1.0]), method=opt.solver_method, step_size=opt.solver_step_size)
                t_sample_norm = x1_samples.view(B, N, L).squeeze(-1).permute(0, 2, 1)

                metrics = Utils.evaluate_samples(
                    t_sample_norm,
                    time_gap_norm,
                    best_type, # Use Top-1 prediction for accuracy check
                    event_type,
                    opt
                )

                if metrics['total_events'] > 0:
                    total_hits += metrics['hit_counts']
                    total_il += metrics['il_sum']
                    total_crps += metrics['crps_sum']
                    total_acc += metrics['correct_type']
                    # total_sse += metrics['sse_sum'] # [Skip] Use Top-K RMSE instead

        if total_events == 0: return {}

        final_acc = total_acc / total_events
        final_rmse = np.sqrt(total_sse / total_events) if total_events > 0 else 0.0
        
        if total_nll_events > 0:
            avg_time_nll = total_time_nll / total_nll_events
            avg_type_nll = total_type_nll / total_nll_events
            avg_joint_nll = avg_time_nll + avg_type_nll
        else:
            avg_time_nll, avg_type_nll, avg_joint_nll = 0, 0, 0

        actual_coverage = total_hits / total_events
        target_coverage = opt.eval_quantile 
        mse = ((actual_coverage - target_coverage) ** 2).mean()
        final_cs = torch.sqrt(mse).item()

        print(f'\n  - (Test Summary)')
        print(f'    Acc:       {final_acc:.4f}')
        print(f'    RMSE:      {final_rmse:.4f}')
        print(f'    Time NLL:  {avg_time_nll:.4f}')
        print(f'    Type NLL:  {avg_type_nll:.4f}')
        print(f'    Joint NLL: {avg_joint_nll:.4f}')
        print(f'    CS:        {final_cs:.4f}')

        return {
            'Acc': final_acc,
            'NLL': avg_joint_nll,
            'Time_NLL': avg_time_nll,
            'Type_NLL': avg_type_nll,
            'RMSE': final_rmse,
            'CS': final_cs
        }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-normalize', type=str, default='log', choices=['normal', 'log'])

    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-d_inner_hid', type=int, default=256)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-d_k', type=int, default=32)
    parser.add_argument('-d_v', type=int, default=32)

    # [Sigma=0.5] Balanced Variance
    parser.add_argument('-fm_sigma', type=float, default=0.5)
    
    parser.add_argument('-solver_method', type=str, default='euler')
    parser.add_argument('-solver_step_size', type=float, default=0.05)
    parser.add_argument('-n_samples', type=int, default=100)
    
    parser.add_argument('-d_latent', type=int, default=16)

    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-epoch', type=int, default=60)
    parser.add_argument('-lr', type=float, default=1e-4)
    
    # [Lambda=5.0] High weight for Type Accuracy
    parser.add_argument('-loss_lambda', type=float, default=5.0)

    parser.add_argument('-eval_epoch', type=int, default=5) 

    parser.add_argument('-save_path', default='./checkpoint.pth')
    parser.add_argument('-save_name', default='model')
    parser.add_argument('-eval_quantile_step', type=float, default=0.05) 
    parser.add_argument('-seed', type=int, default=2023)
    
    parser.add_argument('-just_eval', action='store_true')
    parser.add_argument('-load_path_name', type=str, default=None)
    parser.add_argument('-save_result', type=str, default=None)
    
    opt = parser.parse_args()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    opt.eval_quantile = torch.arange(opt.eval_quantile_step, 1.0, opt.eval_quantile_step, device=opt.device)

    trainloader, devloader, testloader, num_types = prepare_dataloader(opt)

    model = FlowMatchingTHP(num_types, opt)

    model.mean_log_data = getattr(opt, 'mean_log_data', 0)
    model.var_log_data = getattr(opt, 'var_log_data', 1)
    model.mean_data = getattr(opt, 'mean_data', 1)
    model.to(opt.device)

    if opt.just_eval:
        if opt.load_path_name is not None:
            print(f'[Info] Loading model from {opt.load_path_name} ...')
            checkpoint = torch.load(opt.load_path_name, map_location=opt.device)
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            print("[Error] Provide -load_path_name for evaluation.")
            return

        print(f'[Info] Start Evaluation...')
        results = eval_epoch(model, testloader, True, opt)
        if opt.save_result and results:
            df = pd.DataFrame([results])
            df.to_csv(f"{opt.save_result}_results.csv", index=False)
        return

    print(f'[Info] Model Parameters: {sum(p.numel() for p in model.parameters())}')
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    best_loss = float('inf')

    for epoch in range(1, opt.epoch + 1):
        print(f'[ Epoch {epoch} ]')
        t_loss, t_fm, t_type = train_epoch(model, trainloader, optimizer, opt)
        print(f'  - (Train) Loss: {t_loss:.4f} | FM: {t_fm:.4f} | Type: {t_type:.4f}')

        v_loss, v_acc = eval_epoch(model, devloader, False, opt)
        print(f'  - (Valid) Loss: {v_loss:.4f} | Acc: {v_acc:.4f}')

        if v_loss < best_loss:
            best_loss = v_loss
            print(f'    -> Best Loss updated: {best_loss:.4f}, Saving model...')
            torch.save(model.state_dict(), opt.save_path)

        if epoch % opt.eval_epoch == 0:
            print("  - (Running Generation Test...)")
            eval_epoch(model, testloader, True, opt)

        scheduler.step()

if __name__ == '__main__':
    main()