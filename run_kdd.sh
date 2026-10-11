#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"    # repo root if this script lives there
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

ARGS=(
datamodule=time_series 
logger=csv 
model=ts_cfm 

datamodule._target_=src.datamodules.time_series_datamodule.GluontsDataModule 
datamodule.batch_size=128
datamodule.data_name="kdd_cup_2018_without_missing" 
datamodule.context_len=24 
datamodule.prediction_len=24 
'datamodule.num_batches_train_test_split=[100, 1]' 


model._target_=src.models.runner_ts.TSCFMLitModule 
model.ema_weight=0.999
model.lambda_reg=0.3
model.timefeats=true

# model.net config
model.net._target_=src.models.itransformer.iTransformerModel
model.net._partial_=true
model.net.e_layers=4
model.net.hidden_size=1024
model.net.n_heads=8
model.net.dropout=0.1


# model.deter_net config
model.deter_net._target_=src.models.itransformer_deter.iTransformerModelDeter 
model.deter_net._partial_=true 
model.deter_net.num_layers=4 
model.deter_net.hidden_size=1024 
model.deter_net.n_heads=8
)

python -u src/train.py "${ARGS[@]}"

