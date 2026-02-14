import argparse
import numpy as np
import pickle
import time
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import pandas as pd

import transformer.Constants as Constants
import Utils
from preprocess.Dataset import get_dataloader
from transformer.Layers import get_non_pad_mask
from transformer.Models import FlowMatchingTHP
from flow_matching.solver import ODESolver
import torch
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

    time_flat = []
    for i in train_data:
        time_flat += [elem['time_since_last_event'] for elem in i if elem['time_since_last_event'] > 0]
    time_flat = np.array(time_flat)

    opt.time_max = np.max(time_flat)
    opt.mean_data = time_flat.mean()

    if opt.normalize == 'log':
        time_flat_div = time_flat / opt.mean_data
        log_time = np.log(time_flat_div + 1e-9)
        opt.mean_log_data = log_time.mean()
        opt.var_log_data = log_time.std()
        print(f'[Info] Log Stats: Mean={opt.mean_log_data:.4f}, Std={opt.var_log_data:.4f}, Max={opt.time_max:.2f}')

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
        
        total_nll = 0.0
        total_nll_events = 0
        total_se_real = 0.0 # Squared Error in Real Time
        total_se_events = 0

        with torch.no_grad():
            for batch in tqdm(validation_data, desc='  - (Sampling)   ', leave=False):
                event_time, time_gap_norm, event_type = map(lambda x: x.to(opt.device), batch)

                # NLL Calculation (if supported by model)
                if hasattr(model, 'get_exact_log_likelihood'):
                    batch_nll_sum, batch_n_events = model.get_exact_log_likelihood(event_type, event_time, time_gap_norm)
                    total_nll += batch_nll_sum.item()
                    total_nll_events += batch_n_events

                non_pad_mask = get_non_pad_mask(event_type)
                enc_output = model.encoder(event_type, event_time, non_pad_mask)
                c = enc_output[:, :-1, :]
                B, L, D = c.shape
                N = opt.n_samples

                type_logits = model.type_predictor(c)
                pred_types = type_logits.argmax(dim=-1)

                pred_types_input = pred_types + 1
                pred_type_emb = model.encoder.event_type_emb(pred_types_input)
                c_cond = model.type_fusion(c, pred_type_emb)

                c_cond_expanded = c_cond.repeat_interleave(N, dim=0)
                
                x0 = torch.randn(B * N, L, 1, device=opt.device) * opt.fm_sigma

                wrapper = VelocityWrapper(model.v_field, c_cond_expanded)
                solver.velocity_model = wrapper

                use_method = opt.solver_method
                
                solve_kwargs = {}
                if use_method != 'dopri5':
                    solve_kwargs['step_size'] = opt.solver_step_size

                x1_norm = solver.sample(
                    x0,
                    torch.tensor([0.0, 1.0]),
                    method=use_method,
                    **solve_kwargs
                )
                
                if opt.normalize == 'log':
                    threshold = 8.0
                    x1_norm = torch.clamp(x1_norm, max=threshold, min=-threshold)

                # Denormalize Samples to Real Time
                t_sample_norm = x1_norm.squeeze(-1).reshape(B, N, L).permute(0, 2, 1) 
                t_sample_real = model.denormalize_time(t_sample_norm) # (B, L, N)
                
                t_pred_mean = t_sample_real.mean(dim=2) # (B, L)

                # ==============================================================
                # Fix: Add definition of t_gt_real before using it
                # ==============================================================
                if opt.normalize == 'log':
                    # Log transform inverse: exp(x * std + mean)
                    t_gt_real = torch.exp(time_gap_norm * opt.var_log_data + opt.mean_log_data)
                elif opt.normalize == 'normal':
                    t_gt_real = time_gap_norm * opt.mean_data
                else:
                    t_gt_real = time_gap_norm
                # ==============================================================

                mask_loss = (event_type[:, 1:] != Constants.PAD)
                diff = (t_pred_mean - t_gt_real)[mask_loss]
                squared_error = (diff ** 2).sum().item()
                
                total_se_real += squared_error
                total_se_events += mask_loss.sum().item()

                metrics = Utils.evaluate_samples(
                    t_sample_norm,
                    time_gap_norm,
                    pred_types,
                    event_type,
                    opt
                )

                if metrics['total_events'] > 0:
                    total_hits += metrics['hit_counts']
                    total_il += metrics['il_sum']
                    total_crps += metrics['crps_sum']
                    total_acc += metrics['correct_type']
                    total_events += metrics['total_events']

        if total_events == 0: return {}

        final_acc = total_acc / total_events
        final_crps = total_crps / total_events
        final_il = total_il / total_events
        
        final_rmse = np.sqrt(total_se_real / total_se_events) if total_se_events > 0 else 0.0
        final_nll = total_nll / total_nll_events if total_nll_events > 0 else 0.0

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

        print(f'  - (Test) Acc: {final_acc:.4f} | NLL: {final_nll:.4f} | RMSE: {final_rmse:.4f} | CRPS: {final_crps:.4f} | CS: {final_cs:.4f} | IL: {final_il:.4f}')

        return {
            'Acc': final_acc,
            'NLL': final_nll,
            'RMSE': final_rmse,
            'CRPS': final_crps,
            'CS': final_cs,
            'CER': final_cer,
            'IL': final_il
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-normalize', type=str, default='log', choices=['normal', 'log'])

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-fm_sigma', type=float, default=0.1)
    parser.add_argument('-solver_method', type=str, default='euler')
    parser.add_argument('-solver_step_size', type=float, default=0.05)
    parser.add_argument('-n_samples', type=int, default=50)
    
    parser.add_argument('-d_latent', type=int, default=16)

    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-loss_lambda', type=float, default=1.0)
    parser.add_argument('-eval_epoch', type=int, default=5) 

    parser.add_argument('-save_path', default='./checkpoint.pth')
    parser.add_argument('-save_name', default='model')
    parser.add_argument('-eval_quantile_step', type=float, default=0.05) 
    parser.add_argument('-seed', type=int, default=2023)
    
    parser.add_argument('-just_eval', action='store_true')
    parser.add_argument('-load_path_name', type=str, default=None)
    parser.add_argument('-save_result', type=str, default=None)
    
    parser.add_argument('-d_rnn', type=int, default=64)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-scheduler', type=str, default='cosLR')
    parser.add_argument('-eval_quantile', type=int, default=-1)

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
            try:
                state_dict = torch.load(opt.load_path_name, map_location=opt.device)
                model.load_state_dict(state_dict)
            except:
                checkpoint = torch.load(opt.load_path_name, map_location=opt.device)
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint, strict=False)
        else:
            print("[Error] In just_eval mode, please provide -load_path_name!")
            return

        print(f'[Info] Start Evaluation...')
        results = eval_epoch(model, testloader, True, opt)
        
        if opt.save_result:
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
            print("  - (Running Intermediate Generation Test...)")
            eval_epoch(model, testloader, True, opt)

        scheduler.step()

if __name__ == '__main__':
    main()