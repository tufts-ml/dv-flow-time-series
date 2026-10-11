import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "README.md"],
    pythonpath=True,
    dotenv=True,
)

import matplotlib.pyplot as plt
import os
import sys
import math
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from gluonts.evaluation import MultivariateEvaluator
from gluonts.model.forecast import SampleForecast
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from src.models.runner_ts import TSCFMLitModule
from src.models.conditional_flow_matching import ConditionalFlowMatcher
from src.datamodules import time_series_datamodule
from src.models.itransformer import iTransformerModel
from src.models.itransformer_deter import iTransformerModelDeter
from src.ode.sampler import Sampler

from itertools import combinations


def gp_kernel(x1, x2, length_scale=1.0, variance=1.0, kernel='ou'):
    if kernel=='ou':
        return variance * (np.exp(-np.abs(np.subtract.outer(x1, x2)) / length_scale)  +  np.eye(len(x1)))
    elif kernel == 'se':
        return variance * (np.exp(-0.5 * (np.subtract.outer(x1, x2) ** 2) / length_scale**2) +  np.eye(len(x1)))
    elif kernel == 'pe':
        return variance * (np.exp(-2 * np.sin(np.subtract.outer(x1, x2)) ** 2 / length_scale**2) +  np.eye(len(x1)))


def calc_seasonal(data, seasonal_mean, data_start, seasonal_start):
    seasonal_window = seasonal_mean.shape[1]
    num_seasons = int(data.shape[2]/seasonal_window)
    offset_len = data.shape[2] % seasonal_window
    shift_len = (data_start-seasonal_start)%seasonal_window

    # Create column indices for reorganization based on seasonal_start
    cols = torch.arange(seasonal_window).unsqueeze(0)  # Shape: (1, seasonal_window)
    cols = (cols + shift_len[:, None]) % seasonal_window       # Shape: (len(shift_len), seasonal_window), broadcast b to match columns
    seasonal_mean_expanded = seasonal_mean.unsqueeze(0).expand(len(shift_len), -1, -1)  # Shape: (len(shift_len), dim, seasonal_window)
    seasonal_mean_reorganized = seasonal_mean_expanded.gather(2, cols.unsqueeze(1).expand(-1, seasonal_mean.shape[0], -1))
    if num_seasons > 0:
        seasonal_mean_final = torch.cat([torch.cat(num_seasons*[seasonal_mean_reorganized],2), seasonal_mean_reorganized[:,:,:offset_len]],2)
    else:
        seasonal_mean_final = seasonal_mean_reorganized[:,:,:offset_len]
    return seasonal_mean_final

def get_torch_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        torch.cuda.current_device()
    except RuntimeError as err:
        print(f'Warning: CUDA is visible but failed to initialize ({err}); falling back to CPU.')
        return torch.device('cpu')
    return torch.device('cuda')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, help="name of the trained model folder")
    p.add_argument("--epochs", type=str, default='0021')
    p.add_argument("--data_name", type=str, default='exchange_rate_nips')
    p.add_argument("--num_samples", type=int, default=100)
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    model_name = args.model_name
    epochs=args.epochs
    ckpt_path = f'logs/train/runs/{model_name}/checkpoints/epoch_{epochs}.ckpt'
    device = get_torch_device()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    data_name = args.data_name
    num_batches_train_test_split=[100,1]
    num_samples = args.num_samples
    timefeats=True
    context_len = 24
    prediction_len = 24
    if data_name == 'exchange_rate_nips':
        context_len = 30
        prediction_len = 30
        


    selection_window = context_len+prediction_len
    # dim = dataset_train[0]['target'].shape[0]
    
    datamodule=time_series_datamodule.GluontsDataModule(data_name=data_name, 
    context_len=context_len,
    prediction_len=prediction_len,
    num_batches_train_test_split=num_batches_train_test_split,
    batch_size=32,
    )

    model = TSCFMLitModule.load_from_checkpoint(ckpt_path,
                net=iTransformerModel,
                deter_net = iTransformerModelDeter,
                datamodule=datamodule,
                ema_weight=0.9999,
                optimizer=None,
                flow_matcher = ConditionalFlowMatcher(sigma=0.1),
                timefeats=timefeats,
                map_location='cpu')
    model = model.to(device)
    model.net.eval()
    if model.deter_net is not None:
        model.deter_net.eval()
    
    
    datamodule.get_test_dataset()
    test_dataloader = datamodule.test_dataloader()

    train_var_prediction = model.datamodule.train_var_prediction.transpose(0,1).to(model.device)
    train_var_prediction = train_var_prediction.unsqueeze(0).expand(num_samples, *train_var_prediction.shape)


    trajectories = []
    predictions = []
    scaled_targets = []
    targets = []
    preds_deters = []
    sampling_time_sec = 0.0
    sampling_forward_batches = 0
    sampling_windows = 0


    sampler = Sampler()
    sample_fn = sampler.sample_ode(
                num_steps=150,
                atol=1e-6,
                rtol=1e-3,
                reverse=False
            )
    saving_path = ckpt_path.split('.')[0]

    cached_predictions_len = None

    for j,batch in enumerate(test_dataloader):
        x_time = np.arange(prediction_len)  
        mu = np.zeros(prediction_len)       
        cov_matrix = gp_kernel(x1=x_time, x2=x_time, kernel='ou')
        x0s = np.random.multivariate_normal(mu, cov_matrix, size=(num_samples, datamodule.dim))
        x0s = torch.tensor(x0s).to(model.device).to(torch.float32).transpose(1,2)


        xc = batch[0].to(model.device)
        x1 = batch[1].to(model.device)
        test_mu_g = batch[2].to(model.device)
        selected_test_timefeature_context = batch[3].to(model.device)
        selected_test_timefeature_prediction = batch[4].to(model.device)   
        tf = torch.cat([selected_test_timefeature_context, selected_test_timefeature_prediction],1)   
                
        seasonal_adj_test_context = batch[5].to(model.device)
        selected_adj_test_prediction = batch[6].to(model.device)
        lags = batch[7].to(model.device)
        lags_timefeats = batch[8].to(model.device)
 
        context_all = torch.cat([lags, xc],1)
        context_tf_all = torch.cat([lags_timefeats, selected_test_timefeature_context],1)
        

        period_index = pd.period_range(start=datamodule.dataset_test[j]['start'].start_time, 
        periods=datamodule.dataset_test[j]['target'].shape[1], 
        freq=datamodule.freq)
                                
        
        scaled_target = torch.cat([xc,x1],dim=1)
        target = pd.DataFrame(datamodule.dataset_test[j]['original_target'].T)

        target.index=period_index
        targets.append(target)
        scaled = pd.DataFrame(scaled_target.squeeze().cpu().numpy())
        scaled.index=period_index[-selection_window:]
        scaled_targets.append(scaled)
        

        if model.deter_net is not None:
            if timefeats:
                scaled_preds_deter = model.deter_net(xc, selected_test_timefeature_context)
            else:
                scaled_preds_deter = model.deter_net(xc, None)
            scaled_preds_deter = scaled_preds_deter.expand(num_samples, scaled_preds_deter.shape[1],scaled_preds_deter.shape[2])
        else:
            scaled_preds_deter = None   
            
        
        if datamodule.dim>500:
            scaled_preds = []
            for b in range(math.ceil(num_samples/50)):
                with torch.no_grad():
                    if scaled_preds_deter is None:
                        if timefeats:
                            model_kwargs = dict(xc=xc.expand(num_samples, *xc.shape[1:])[b*50:(b+1)*50], 
                            timefeats_c=selected_test_timefeature_context.expand(num_samples, *selected_test_timefeature_context.shape[1:])[b*50:(b+1)*50], 
                            timefeats_p=selected_test_timefeature_prediction.expand(num_samples, *selected_test_timefeature_prediction.shape[1:])[b*50:(b+1)*50],
                            x_deter=None)
                        else:
                            model_kwargs = dict(xc=xc.expand(num_samples, *xc.shape[1:])[b*50:(b+1)*50], 
                                timefeats_c=None, 
                                timefeats_p=None,
                                x_deter=None)
                    else:
                        if timefeats:
                            model_kwargs = dict(xc=xc.expand(num_samples, *xc.shape[1:])[b*50:(b+1)*50], 
                            timefeats_c=selected_test_timefeature_context.expand(num_samples, *selected_test_timefeature_context.shape[1:])[b*50:(b+1)*50], 
                            timefeats_p=selected_test_timefeature_prediction.expand(num_samples, *selected_test_timefeature_prediction.shape[1:])[b*50:(b+1)*50],
                            x_deter=scaled_preds_deter[b*50:(b+1)*50])
                        else:
                            model_kwargs = dict(xc=xc.expand(num_samples, *xc.shape[1:])[b*50:(b+1)*50], 
                            timefeats_c=None, 
                            timefeats_p=None,
                            x_deter=scaled_preds_deter[b*50:(b+1)*50])
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(model.device)
                    sampled_traj = sample_fn(x0s[b*50:(b+1)*50], model.net, **model_kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(model.device)
                    scaled_preds.append(sampled_traj[-1])
            scaled_preds = torch.cat(scaled_preds)
        else:
            with torch.no_grad():
                if timefeats:
                    model_kwargs = dict(xc=xc.expand(num_samples, *xc.shape[1:]), 
                    timefeats_c=selected_test_timefeature_context.expand(num_samples, *selected_test_timefeature_context.shape[1:]),
                    timefeats_p=selected_test_timefeature_prediction.expand(num_samples, *selected_test_timefeature_prediction.shape[1:]),
                    x_deter=scaled_preds_deter)
                else:
                    model_kwargs = dict(xc=xc.expand(num_samples, *xc.shape[1:]), 
                    timefeats_c=None,
                    timefeats_p=None,
                    x_deter=scaled_preds_deter)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(model.device)
                sampled_traj = sample_fn(x0s, model.net, **model_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(model.device)
                scaled_preds = sampled_traj[-1]
    

        preds = scaled_preds*torch.sqrt(train_var_prediction) + test_mu_g
        preds = preds + selected_adj_test_prediction

        trajectories.append(scaled_preds)
        predictions.append(preds)
        print(j, 'finished')
            

    torch.save(predictions, f'{saving_path}_preds.pt')
    torch.save(trajectories, f'{saving_path}_scaled_preds.pt')
    os.mkdir(f'logs/train/runs/{model_name}/figs')

    def plot(target, forecast, context_length, prediction_length, prediction_intervals=(50.0, 90.0), color='g', fname=None):
        label_prefix = ""
        rows = 4
        cols = 4
        fig, axs = plt.subplots(rows, cols, figsize=(24, 24))
        axx = axs.ravel()
        seq_len, target_dim = target.shape
        
        ps = [50.0] + [
                50.0 + f * c / 2.0 for c in prediction_intervals for f in [-1.0, +1.0]
            ]
            
        percentiles_sorted = sorted(set(ps))
        
        def alpha_for_percentile(p):
            return (p / 100.0) ** 0.3
            
        for dim in range(0, min(rows * cols, target_dim)):
            ax = axx[dim]

            target[-(context_length+prediction_length):][dim].plot(ax=ax)
            
            ps_data = [forecast.quantile(p / 100.0, dim=0)[:,dim] for p in percentiles_sorted]
            i_p50 = len(percentiles_sorted) // 2
            
            p50_data = ps_data[i_p50]
            p50_series = pd.Series(data=p50_data, index=target.index[-prediction_length:])
            p50_series.plot(color=color, ls="-", label=f"{label_prefix}median", ax=ax)
            
            for i in range(len(percentiles_sorted) // 2):
                ptile = percentiles_sorted[i]
                alpha = alpha_for_percentile(ptile)
                ax.fill_between(
                    target.index[-prediction_length:],
                    ps_data[i],
                    ps_data[-i - 1],
                    facecolor=color,
                    alpha=alpha,
                    interpolate=True,
                )
                # Hack to create labels for the error intervals.
                # Doesn't actually plot anything, because we only pass a single data point
                pd.Series(data=p50_data[:1], index=target.index[-prediction_length:][:1]).plot(
                    color=color,
                    alpha=alpha,
                    linewidth=10,
                    label=f"{label_prefix}{100 - ptile * 2}%",
                    ax=ax,
                )

        legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]    
        axx[0].legend(legend, loc="upper left")
        
        if fname is not None:
            plt.savefig(fname, bbox_inches='tight', pad_inches=0.05)


    plot(
        target=scaled_targets[0],
        forecast=trajectories[0].detach().cpu(),
        context_length=context_len,
        prediction_length=prediction_len,
        fname=f'logs/train/runs/{model_name}/figs/scaled.png'
    )
            

    plot(
        target=targets[0],
        forecast=predictions[0].detach().cpu(),
        context_length=context_len,
        prediction_length=prediction_len,
        fname=f'logs/train/runs/{model_name}/figs/original.png'
    )

    evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:], 
                                    target_agg_funcs={'sum': np.sum})

    def variable_corr_over_time(series, eps=1e-8):
        centered = series - series.mean(dim=-2, keepdim=True)
        cov = centered.transpose(-1, -2) @ centered
        norms = torch.linalg.vector_norm(centered, dim=-2)
        denom = norms.unsqueeze(-1) * norms.unsqueeze(-2)
        return cov / denom.clamp_min(eps)


    def build_corr_evaluation_data(targets, predictions, prediction_len):
        dim = predictions[0].shape[-1]
        if dim < 2:
            return [], []

        pair_mask = torch.triu(torch.ones(dim, dim, dtype=torch.bool), diagonal=1)
        num_pairs = int(pair_mask.sum().item())
        pair_columns = list(range(num_pairs))

        corr_targets = []
        corr_forecasts = []
        for target, prediction in zip(targets, predictions):
            target_window = torch.tensor(
                target.iloc[-prediction_len:].values,
                dtype=torch.float32,
            )
            prediction = prediction.detach().cpu().to(torch.float32)

            target_corr_vec = variable_corr_over_time(target_window)[pair_mask].numpy()
            prediction_corr_vec = variable_corr_over_time(prediction)[:, pair_mask].numpy()

            start_date = target.index[-prediction_len]
            corr_targets.append(
                pd.DataFrame(
                    [target_corr_vec],
                    index=target.index[-prediction_len:][:1],
                    columns=pair_columns,
                )
            )
            corr_forecasts.append(
                SampleForecast(
                    samples=prediction_corr_vec[:, None, :],
                    start_date=start_date,
                )
            )

        return corr_targets, corr_forecasts, num_pairs

    eval_count = min(len(targets), len(predictions))
    if eval_count == 0:
        raise RuntimeError(
            f"No aligned targets/predictions available for evaluation: "
            f"len(targets)={len(targets)}, len(predictions)={len(predictions)}"
        )
    if len(targets) != len(predictions):
        print(
            f"Warning: evaluating first {eval_count} aligned target/prediction pairs "
            f"(len(targets)={len(targets)}, len(predictions)={len(predictions)}, "
            f"len(dataset_test)={len(datamodule.dataset_test)}). If this is unexpected, "
            f"remove {saving_path}_preds.pt and rerun to regenerate predictions."
        )

    eval_targets = targets[:eval_count]
    eval_predictions = predictions[:eval_count]
    eval_preds_deters = preds_deters[:eval_count]

    corr_targets, corr_forecasts, num_corr_pairs = build_corr_evaluation_data(
        eval_targets,
        eval_predictions,
        prediction_len,
    )
    corr_agg_metric = {}
    if num_corr_pairs > 0:
        corr_agg_metric, corr_item_metrics = evaluator(
            corr_targets,
            corr_forecasts,
            num_series=num_corr_pairs,
        )

    forecasts = []
    for target, prediction in zip(eval_targets, eval_predictions):
        forecast = SampleForecast(samples=prediction.detach().cpu().numpy(),start_date=target.index[-prediction_len])
        forecasts.append(forecast)

    agg_metric, item_metrics = evaluator(eval_targets, forecasts, num_series=eval_count)

    metrics_dict = {'CRPS': agg_metric["m_sum_mean_wQuantileLoss"],
                    'ND': agg_metric["m_sum_ND"],
                    'NRMSE': agg_metric["m_sum_NRMSE"],
                    'sMAPE': agg_metric["m_sum_sMAPE"],
                    'corr_CRPS_mean': corr_agg_metric.get("mean_wQuantileLoss", np.nan)}
    

    def calc_quantile_risk(q, targets, predictions):
        qls = 0
        for target, prediction in zip(targets, predictions):
            pred_q = prediction.quantile(q, dim=0)
            ql = q*torch.max(torch.zeros_like(target), target-pred_q) + (1-q)*torch.max(torch.zeros_like(target), pred_q-target)
            ql = ql.sum()
            qls += ql
        return 2*qls/torch.abs(targets).sum()
        

    targets_test = torch.tensor(
        np.stack([t.iloc[-prediction_len:].values for t in eval_targets]),
        dtype=torch.float32,
    )
    predictions=[p.detach().cpu().to(torch.float32) for p in eval_predictions]


    metrics_dict['p90'] = calc_quantile_risk(0.9, targets_test, predictions).numpy()

    metrics_df = pd.DataFrame(metrics_dict, index=[0])
    metrics_df.to_csv(f'logs/train/runs/{model_name}/metrics.csv', index=False)
