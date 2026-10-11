# Deterministic Prediction as Global Velocity: Flow Matching for Probabilistic Time-Series Forecast 


## Setup

Use Python 3.8 to match the environment this project was developed with.

```bash
conda create -n dv-flow-time-series python=3.8.19 pip -y
conda activate dv-flow-time-series
pip install -r requirements.txt
```

If you run on a GPU node, make sure the installed PyTorch build is compatible with the node CUDA driver.

## Data

The datamodule uses GluonTS `get_dataset(..., path=Path(f"{root}/data"))`, so datasets are read from or downloaded into:

```text
data/
```

The provided scripts use:

- `exchange_rate_nips`
- `kdd_cup_2018_without_missing`


## Training

Run from the repo root.

```bash
bash run_exchange.sh
```

or:

```bash
bash run_kdd.sh
```

The scripts set `PYTHONPATH` and launch:

```bash
python -u src/train.py ...
```

Outputs are written under:

```text
logs/train/runs/<date_time>/
```

Useful Hydra overrides:

```bash
python -u src/train.py trainer=cpu datamodule.data_name=exchange_rate_nips
python -u src/train.py trainer=gpu trainer.devices=1 datamodule.data_name=exchange_rate_nips
```

## Evaluation

Evaluate a checkpoint with:

```bash
python -u src/test.py \
  --model_name <run_folder_name> \
  --epochs <epoch_number> \
  --data_name exchange_rate_nips \
```

Example:

```bash
python -u src/test.py --model_name 2026-10-10_14-42-15 --epochs 0021 --data_name exchange_rate_nips
```

The script expects the checkpoint at:

```text
logs/train/runs/<model_name>/checkpoints/epoch_<epochs>.ckpt
```

It saves:

- `<checkpoint_prefix>_preds.pt`
- `<checkpoint_prefix>_scaled_preds.pt`
- `logs/train/runs/<model_name>/figs/`
- `logs/train/runs/<model_name>/metrics.csv`



## Current Modeling Assumptions

This simplified code path assumes:

- scale method is SATN
- source noise is an OU Gaussian process
- the deterministic network is enabled in the training/evaluation path
