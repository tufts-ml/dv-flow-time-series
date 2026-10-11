import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "README.md"],
    pythonpath=True,
    dotenv=True,
)

import math
import pandas as pd
from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import random
import torch
import os
from torch.utils.data import  TensorDataset
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature
from gluonts.dataset.repository.datasets import  get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    ValidationSplitSampler,
    TestSplitSampler,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)

from src.utils.pylogger import get_pylogger
from src.utils.utils import get_module_forward_input_names
from src.datamodules.components.base import BaseLightningDataModule
from src.datamodules.feature import fourier_time_features_from_frequency, lags_for_fourier_time_features_from_frequency


log = get_pylogger(__name__)


def remove_seasonal(data, seasonal_mean, data_start, seasonal_start):
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
    return data - seasonal_mean_final


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


def get_seasonal_start(start, freq):
    freq = freq.lower()
    if freq in ("h", "1h"):
        return start.hour
    elif freq == "b":
        return start.day
    elif freq in ("30min", "30t"):
        return start.hour * 2 + int(start.minute > 0)
    raise ValueError(f"Unsupported frequency for seasonal adjustment: {freq}")


class GluontsDataModule(BaseLightningDataModule):
    pass_to_model = True
    IS_TRAJECTORY = False

    def __init__(
        self,
        data_name: str,
        context_len: int,
        prediction_len: int,
        num_batches_train_test_split: Tuple[int, int] = (100, 1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        cardinality: Optional[List[int]] = None,
        seed=None,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.data_name = data_name
        
        self.context_len = context_len
        self.prediction_len = prediction_len
        
        dataset = get_dataset(self.data_name, regenerate=False, path=Path(f'{root}/data'))
        self.freq = dataset.metadata.freq
        self.seasonal_window = dataset.metadata.prediction_length
        train_grouper = MultivariateGrouper(max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))

        test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                        max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))
        if self.data_name == 'kdd_cup_2018_without_missing':
            dataset_train1 = list(dataset.train)
            s = set([x['start'] for x in dataset_train1])
            b = []
            for i in range(len(dataset.train)):
                t = dataset_train1[i]
                train_period_range = pd.period_range(start=t['start'], periods=t['target'].shape[0], 
                                            freq=dataset.metadata.freq)
                b.append(train_period_range[-1])
            
            dataset_train2 = []
            for i in range(len(dataset.train)):
                t = dataset_train1[i]
                train_period_range = pd.period_range(start=t['start'], periods=t['target'].shape[0], 
                                            freq=dataset.metadata.freq)
                start_idx = train_period_range.get_loc(max(s))
                end_idx = train_period_range.get_loc(min(b))
                dataset_train2.append({'target': t['target'][start_idx:end_idx+1-1000],
                                        'start': max(s),
                                        'item_id': t['item_id'],
                                        'feat_static_cat': t['feat_static_cat']})
            
            dataset_test1 = list(dataset.test)
            s = set([x['start'] for x in dataset_test1])
            dataset_test2 = []
            for i in range(len(dataset.test)):
                t = dataset_test1[i]
                train_period_range = pd.period_range(start=t['start'], periods=t['target'].shape[0], 
                                            freq=dataset.metadata.freq)
                start_idx = train_period_range.get_loc(max(s))
                end_idx = train_period_range.get_loc(min(b))
                dataset_test2.append({'target': t['target'][start_idx:end_idx+1],
                'start': max(s),
                'item_id': t['item_id'],
                'feat_static_cat': t['feat_static_cat']})
            dataset_train = train_grouper(dataset_train2)
            dataset_test = test_grouper(dataset_test2)
        else:
            dataset_train = train_grouper(dataset.train)
            dataset_test = test_grouper(dataset.test)
            

        self.dim = dataset_train[0]['target'].shape[0]

        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else lags_for_fourier_time_features_from_frequency(freq_str=self.freq)
        )
        self.time_features = (
            time_features
            if time_features is not None
            else fourier_time_features_from_frequency(self.freq)
        )
        
        self.history_len = self.context_len + max(self.lags_seq)
        selection_window = self.history_len+prediction_len
        self.selection_window = selection_window
        

        transformation = self.create_transformation()
        
        dataset_valid = []
        for t in dataset_train:
            train_period_range = pd.period_range(start=t['start'], periods=t['target'].shape[1], 
                                    freq=t['start'].freqstr)
            for i in range(100):
                data = {'target':t['target'][:,-(5*(100-i)+selection_window):-(5*(100-i))],
                        'start': train_period_range[-(5*(100-i)+selection_window)],
                        'feat_static_cat': t['feat_static_cat']}
                dataset_valid.append(data)


        self.dataset_valid = list(transformation(dataset_valid, is_train=False))
        self.dataset_test = list(transformation(dataset_test, is_train=False))

        self.train_size = num_batches_train_test_split[0] * batch_size
        self.eval_size = num_batches_train_test_split[1] * batch_size


        for i in range(len(dataset_train)):
            dataset_train[i]['target'] = dataset_train[i]['target'][:,:-(500+selection_window)]

        self.dataset_train = list(transformation(dataset_train, is_train=True))
        

        num_seasons = int(self.dataset_train[0]['target'].shape[1]/self.seasonal_window)
        whole_seasonal_train = self.dataset_train[0]['target'][:,:num_seasons*self.seasonal_window].reshape(self.dataset_train[0]['target'].shape[0], -1, self.seasonal_window)
        self.seasonal_mean = torch.tensor(whole_seasonal_train.mean(1))
        self.seasonal_start = get_seasonal_start(self.dataset_train[0]['start'], self.freq)
            
        for i in range(len(dataset_train)):
            seasonal_start = get_seasonal_start(self.dataset_train[i]['start'], self.freq)
            self.dataset_train[i]['target'] = remove_seasonal(torch.tensor(dataset_train[i]['target']).unsqueeze(0), self.seasonal_mean, torch.tensor(seasonal_start).unsqueeze(0), self.seasonal_start).squeeze(0).numpy()
            

        for i in range(len(dataset_valid)):
            seasonal_start = get_seasonal_start(dataset_valid[i]['start'], self.freq)
                                
            self.dataset_valid[i]['target'] = remove_seasonal(torch.tensor(dataset_valid[i]['target']).unsqueeze(0), self.seasonal_mean, torch.tensor(seasonal_start).unsqueeze(0), self.seasonal_start).squeeze(0).numpy()
                
        for i in range(len(dataset_test)):
            seasonal_start = get_seasonal_start(dataset_test[i]['start'], self.freq)
            
            self.dataset_test[i]['seasonal'] = calc_seasonal(torch.tensor(dataset_test[i]['target']).unsqueeze(0), self.seasonal_mean, torch.tensor(seasonal_start).unsqueeze(0), self.seasonal_start).squeeze(0)
            self.dataset_test[i]['original_target'] = dataset_test[i]['target']
            self.dataset_test[i]['target'] = remove_seasonal(torch.tensor(dataset_test[i]['target']).unsqueeze(0), self.seasonal_mean, torch.tensor(seasonal_start).unsqueeze(0), self.seasonal_start).squeeze(0).numpy()
                
        self.get_modeling_dataset()

    def create_transformation(self) -> Transformation:
        """时序数据的转换

        Returns:
            Transformation: 转换链
        """
        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
                ),
                # maps the target to (1, T)
                # if the target data is uni dimensional
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                ),  # expand the target to (1, T)
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_len,
                ),  # add time features
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),  # vstack time features
                SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0]),  # set static cat
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT,
                             expected_ndim=1),  # static cat
            ]
        )
        
    def get_lagged_subsequences(
        self,
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length
            length of sequence in the T (time) dimension (axis = 1).
        indices
            list of lag indices to be used.
            eg: [1,24,168]
        subsequences_length
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices),
            containing lagged subsequences.
            Specifically, lagged[i, :, j, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: history_length + begin_index >= 0
        # that is: history_length - lag_index - sequence_length >= 0
        # hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            # shape: (batch_size, 1, sub_seq_len, C)
            lagged_values.append(
                sequence[:, begin_index:end_index, ...].unsqueeze(1))
        # shape: (batch_size, sub_seq_len, I, C) I = len(indices)=3
        return torch.cat(lagged_values, dim=1).permute(0, 2, 1, 3)


    def select_timewindow(self,dataset,size):
    
        selected_targets = []
        selected_timefeats = []
        selected_seasonals = []
        # random.shuffle(dataset)
        for i in range(len(dataset)):
            period_range = pd.period_range(start=dataset[i]['start'], periods=dataset[i]['target'].shape[1], 
                               freq=self.freq)
            total_timesteps = dataset[i]['target'].shape[1]-self.selection_window
            # sample context window start time 
            weights = torch.ones(total_timesteps)
            indices = torch.multinomial(weights, size, replacement=True)

            indices_expanded = indices.unsqueeze(-1) + torch.arange(self.selection_window).view(1,self.selection_window)
            indices_expanded_target = indices_expanded.unsqueeze(0).expand(dataset[i]['target'].shape[0], -1, -1)
            indices_expanded_timefeats = indices_expanded.unsqueeze(0).expand(dataset[i]['time_feat'].shape[0], -1, -1)
            selected_target = torch.gather(torch.tensor(dataset[i]['target']).unsqueeze(-1).expand(-1,-1,self.selection_window), 1, indices_expanded_target).permute(1, 0, 2)
            selected_targets.append(selected_target)
            selected_timefeat = torch.gather(torch.tensor(dataset[i]['time_feat']).unsqueeze(-1).expand(-1,-1,self.selection_window), 1, indices_expanded_timefeats).permute(1, 0, 2)
            selected_timefeats.append(selected_timefeat)


        selected_targets = torch.cat(selected_targets)
        selected_timefeats = torch.cat(selected_timefeats)
        return selected_targets, selected_timefeats

    def get_modeling_dataset(self):
        
        np.random.seed(self.hparams.seed)
                

        selected_train_target, selected_train_timefeats  = self.select_timewindow(self.dataset_train, self.train_size)

        selected_eval_target = []
        selected_eval_timefeats = []
        for i in range(len(self.dataset_valid)):
            selected_eval_target.append(torch.tensor(self.dataset_valid[i]['target'][:,-(self.selection_window):]))
            selected_eval_timefeats.append(torch.tensor(self.dataset_valid[i]['time_feat'][:,-(self.selection_window):]))

        selected_eval_target = torch.stack(selected_eval_target)
        selected_eval_timefeats = torch.stack(selected_eval_timefeats)

        assert selected_train_target.shape[1] == self.dim, 'dataset dimension does not match cardinality'
        
        selected_train_target_context = selected_train_target[:,:,:self.history_len]
        selected_train_target_prediction = selected_train_target[:,:,self.history_len:]
        lags_train = self.get_lagged_subsequences(
            sequence=selected_train_target_context.transpose(1,2),
            sequence_length=self.history_len,
            indices=self.lags_seq,
            subsequences_length=self.context_len)
        selected_train_target_context = selected_train_target_context[:,:,-self.context_len:]
        
        
        selected_train_timefeat_context = selected_train_timefeats[:,:,:self.history_len]
        selected_train_timefeat_prediction = selected_train_timefeats[:,:,self.history_len:]
        
        lags_train_timefeats = self.get_lagged_subsequences(
            sequence=selected_train_timefeat_context.transpose(1,2),
            sequence_length=self.history_len,
            indices=self.lags_seq,
            subsequences_length=self.context_len)
        lags_train_timefeats = lags_train_timefeats.reshape(lags_train_timefeats.shape[0], lags_train_timefeats.shape[1]*lags_train_timefeats.shape[2], lags_train_timefeats.shape[3])
        selected_train_timefeat_context = selected_train_timefeat_context[:,:,-self.context_len:]
                        

        selected_eval_target_context = selected_eval_target[:,:,:self.history_len]
        selected_eval_target_prediction = selected_eval_target[:,:,self.history_len:]
        lags_eval = self.get_lagged_subsequences(
            sequence=selected_eval_target_context.transpose(1,2),
            sequence_length=self.history_len,
            indices=self.lags_seq,
            subsequences_length=self.context_len)
        selected_eval_target_context = selected_eval_target_context[:,:,-self.context_len:]
        
        selected_eval_timefeat_context = selected_eval_timefeats[:,:,:self.history_len]
        selected_eval_timefeat_prediction = selected_eval_timefeats[:,:,self.history_len:]
        lags_eval_timefeats = self.get_lagged_subsequences(
            sequence=selected_eval_timefeat_context.transpose(1,2),
            sequence_length=self.history_len,
            indices=self.lags_seq,
            subsequences_length=self.context_len)
        lags_eval_timefeats = lags_eval_timefeats.reshape(lags_eval_timefeats.shape[0], lags_eval_timefeats.shape[1]*lags_eval_timefeats.shape[2], lags_eval_timefeats.shape[3])
        selected_eval_timefeat_context = selected_eval_timefeat_context[:,:,-self.context_len:]

        # calculate empirical mu and sigma
        
        train_mu_g = selected_train_target_context[:,:,-1]
        train_mu_g_prediction = torch.stack([train_mu_g]*self.prediction_len,2) 
        train_mu_g_context = torch.stack([train_mu_g]*self.context_len,2) 
        train_var_last = torch.mean((selected_train_target_prediction[:,:,-1]-train_mu_g)**2,0)/self.prediction_len
        train_var_first = torch.mean((selected_train_target_prediction[:,:,0]-train_mu_g)**2,0)/self.prediction_len
        

        var_increment_prediction = ((train_var_last-train_var_first)/(self.prediction_len-1)).unsqueeze(-1).expand(-1,self.prediction_len)
        num_increments_prediction = torch.arange(self.prediction_len).unsqueeze(0).expand(self.dim,-1)
        self.train_var_prediction = var_increment_prediction * num_increments_prediction+train_var_first.unsqueeze(-1).expand(-1,self.prediction_len)
        self.train_var_prediction = self.train_var_prediction+0.00001
        scaled_train_prediction = (selected_train_target_prediction-train_mu_g_prediction)/torch.sqrt(self.train_var_prediction)
        

        train_var_context_first = torch.mean((selected_train_target_context[:,:,0]-train_mu_g)**2,0)/self.context_len
        var_increment_context = (train_var_context_first/(self.context_len-1)).unsqueeze(-1).expand(-1,self.context_len)
        num_increments_context = torch.arange(self.context_len-1, -1, -1).unsqueeze(0).expand(self.dim,-1)
        self.train_var_context = var_increment_context * num_increments_context
        self.train_var_context[:, -1] = 0
        
        self.train_var_context = self.train_var_context+0.00001
        scaled_train_context = (selected_train_target_context-train_mu_g_context)/torch.sqrt(self.train_var_context)

        #(batch_size, sub_seq_len, I, C)
        train_mu_g_history = torch.stack([lags_train[:,-1,:,:]]*self.context_len,1)
        train_var_history_first = torch.mean((lags_train[:,0,:,:]-lags_train[:,-1,:,:])**2,0)/self.context_len
        var_increment_history = (train_var_history_first/(self.context_len-1)).unsqueeze(0).expand(self.context_len,*train_var_history_first.shape)
        num_increments_history = torch.arange(self.context_len-1, -1, -1).unsqueeze(-1).unsqueeze(-1).expand(*var_increment_history.shape)
        self.train_var_history = var_increment_history * num_increments_history
        self.train_var_history[-1] = 0
        self.train_var_history = self.train_var_history+0.00001
        scaled_train_lags = (lags_train-train_mu_g_history)/torch.sqrt(self.train_var_history)
        
        self.data_train = TensorDataset(scaled_train_context.transpose(1,2), scaled_train_prediction.transpose(1,2), 
                                        train_mu_g.unsqueeze(-1).expand(selected_train_target_context.shape[0],self.dim,self.prediction_len).transpose(1,2),
                                        selected_train_timefeat_context.transpose(1,2), selected_train_timefeat_prediction.transpose(1,2),
                                        scaled_train_lags.reshape(-1, self.context_len * len(self.lags_seq), self.dim),
                                        lags_train_timefeats
                                        )

        
        eval_mu_g = selected_eval_target_context[:,:,-1]

        scaled_eval_prediction = (selected_eval_target_prediction-torch.stack([eval_mu_g]*self.prediction_len,2))/torch.sqrt(self.train_var_prediction)
        scaled_eval_context = (selected_eval_target_context-torch.stack([eval_mu_g]*self.context_len,2))/torch.sqrt(self.train_var_context)
        scaled_eval_lags = (lags_eval-torch.stack([lags_eval[:,-1,:,:]]*self.context_len,1))/torch.sqrt(self.train_var_history)
        
        self.data_val = TensorDataset(scaled_eval_context.transpose(1,2), scaled_eval_prediction.transpose(1,2), 
                                        eval_mu_g.unsqueeze(-1).expand(selected_eval_target_context.shape[0],self.dim,self.prediction_len).transpose(1,2),
                                        selected_eval_timefeat_context.transpose(1,2), selected_eval_timefeat_prediction.transpose(1,2),
                                        scaled_eval_lags.reshape(-1, self.context_len * len(self.lags_seq), self.dim),
                                        lags_eval_timefeats
                                        )

        


    def get_test_dataset(self):
        test_target = []
        test_timefeats = []
        test_seasonal = []
        for i in range(len(self.dataset_test)):
            test_target.append(torch.tensor(self.dataset_test[i]['target'][:,-(self.selection_window):]))
            test_timefeats.append(torch.tensor(self.dataset_test[i]['time_feat'][:,-(self.selection_window):]))
            test_seasonal.append(torch.tensor(self.dataset_test[i]['seasonal'][:,-(self.selection_window):]))
        test_target = torch.stack(test_target)
        test_timefeats = torch.stack(test_timefeats)
        test_target_context = test_target[:,:,:self.history_len]
        test_target_prediction = test_target[:,:,self.history_len:]
        lags_test = self.get_lagged_subsequences(
            sequence=test_target_context.transpose(1,2),
            sequence_length=self.history_len,
            indices=self.lags_seq,
            subsequences_length=self.context_len)
        test_target_context = test_target_context[:,:,-self.context_len:]
        
        test_timefeat_context = test_timefeats[:,:,:self.history_len]
        test_timefeat_prediction = test_timefeats[:,:,self.history_len:]
        lags_test_timefeats = self.get_lagged_subsequences(
            sequence=test_timefeat_context.transpose(1,2),
            sequence_length=self.history_len,
            indices=self.lags_seq,
            subsequences_length=self.context_len)
        lags_test_timefeats = lags_test_timefeats.reshape(lags_test_timefeats.shape[0], lags_test_timefeats.shape[1]*lags_test_timefeats.shape[2], lags_test_timefeats.shape[3])
        test_timefeat_context = test_timefeat_context[:,:,-self.context_len:]
        
        test_seasonal = torch.stack(test_seasonal)
        test_seasonal_context = test_seasonal[:,:,:self.history_len]
        test_seasonal_context = test_seasonal[:,:,-self.context_len:]
        test_seasonal_prediction = test_seasonal[:,:,self.history_len:]
            
        
        test_mu_g = test_target_context[:,:,-1]
        test_mu_g_prediction = torch.stack([test_mu_g]*self.prediction_len,2) 
        test_mu_g_context = torch.stack([test_mu_g]*self.context_len,2) 

        scaled_test_prediction = (test_target_prediction-test_mu_g_prediction)/torch.sqrt(self.train_var_prediction)
        
        scaled_test_context = (test_target_context-test_mu_g_context)/torch.sqrt(self.train_var_context)

        scaled_test_lags = (lags_test-torch.stack([lags_test[:,-1,:,:]]*self.context_len,1))/torch.sqrt(self.train_var_history)
        
        self.data_test = TensorDataset(scaled_test_context.transpose(1,2), scaled_test_prediction.transpose(1,2), 
                                    test_mu_g.unsqueeze(-1).expand(test_target_context.shape[0],self.dim,self.prediction_len).transpose(1,2),
                                    test_timefeat_context.transpose(1,2), test_timefeat_prediction.transpose(1,2),
                                    test_seasonal_context.transpose(1,2), test_seasonal_prediction.transpose(1,2),
                                    scaled_test_lags.reshape(-1, self.context_len * len(self.lags_seq), self.dim),
                                    lags_test_timefeats
                                    )

                
        

if __name__ == "__main__":
    datamodule = GluontsDataModule(
        data_name= 'electricity_nips',
        context_len=24,
        prediction_len=24,
        num_batches_train_test_split=[100,1],
        batch_size=32,
        )
    datamodule.get_test_dataset()
    dataloader = datamodule.test_dataloader()
    print('len dataloader', len(dataloader))
    batch = next(iter(dataloader))
    xc = batch[0]
    x1 = batch[1]
    mu = batch[2]
    tf_c = batch[3]
    tf_p = batch[4]
    seasonal_c = batch[5]
    seasonal_p = batch[6]
    lags = batch[7]
    lags_tf = batch[8]
    print('xc', xc.shape)
    print('x1', x1.shape)
    print('mu', mu.shape)
    print('tf_c', tf_c.shape)
    print('tf_p', tf_p.shape)
    print('seasonal_c', seasonal_c.shape)
    print('seasonal_p', seasonal_p.shape)
    print('lags', lags.shape)
    print('lags timefeats', lags_tf.shape)
