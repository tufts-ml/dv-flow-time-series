from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader

from .conditional_flow_matching import ConditionalFlowMatcher

# from .utils import get_wandb_logger
from ..utils.pylogger import get_pylogger
from .ema import ExponentialMovingAverage




log = get_pylogger(__name__)

def gp_kernel(x1, x2, length_scale=1.0, variance=1.0, kernel='ou'):
    if kernel=='ou':
        return variance * (np.exp(-np.abs(np.subtract.outer(x1, x2)) / length_scale)  +  np.eye(len(x1)))
    elif kernel == 'se':
        return variance * (np.exp(-0.5 * (np.subtract.outer(x1, x2) ** 2) / length_scale**2) +  np.eye(len(x1)))
    elif kernel == 'pe':
        return variance * (np.exp(-2 * np.sin(np.subtract.outer(x1, x2)) ** 2 / length_scale**2) +  np.eye(len(x1)))


def grad_norm(parameters, device=None) -> torch.Tensor:
    norms = []
    for p in parameters:
        if p.grad is not None:
            norms.append(p.grad.detach().norm(2))
    if not norms:
        return torch.tensor(0.0, device=device)
    return torch.norm(torch.stack(norms), 2)

class TSCFMLitModule(LightningModule):
    def __init__(
        self,
        net: Any,
        deter_net: Any,
        optimizer: Any,
        datamodule: LightningDataModule,
        flow_matcher: ConditionalFlowMatcher,
        scheduler: Optional[Any] = None,
        ema_weight: float = None,
        num_samples: int = 1,
        lambda_reg: float = 0.0,
        timefeats: bool=True,
        plot = None,
        sep_train=False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "deter_net",
                "optimizer",
                "scheduler",
                "datamodule",
                "flow_matcher",
            ],
            logger=False,
        )
        self.datamodule = datamodule
        self.is_trajectory = False
        self.ema_weight = ema_weight
        self.lambda_reg = lambda_reg
        self.timefeats = timefeats
        self.sep_train = sep_train
        if hasattr(datamodule, "IS_TRAJECTORY"):
            self.is_trajectory = datamodule.IS_TRAJECTORY
        # dims is either an integer or a tuple. This helps us to decide whether to process things as
        # a vector or as an image.
        if hasattr(datamodule, "dim"):
            self.dim = datamodule.dim
        else:
            raise NotImplementedError("Datamodule must have dim")
        
        model_input_dim = self.dim
        self.net = net(horizon=self.datamodule.hparams.prediction_len,input_dim=model_input_dim, data_dim=self.dim,
                       context_len=self.datamodule.hparams.context_len)
        if deter_net is not None:
            self.deter_net = deter_net(horizon=self.datamodule.hparams.prediction_len,input_dim=model_input_dim, data_dim=self.dim,
                        context_len=self.datamodule.hparams.context_len)
        else:
            self.deter_net = None

        self.optimizer = optimizer
        self.flow_matcher = flow_matcher
        self.scheduler = scheduler
        self.instantiate_ema(ema_weight)
        self.num_samples = num_samples
        


    def instantiate_ema(self, ema_weight):
        self.ema = ExponentialMovingAverage(
        self.net.parameters(), decay=ema_weight)


    def unpack_batch(self, batch):
        """Unpacks a batch of data to a single tensor."""
        xc = batch[0]
        x1 = batch[1]
        timefeats_c = batch[3]
        timefeats_p = batch[4]
        lags = batch[5]
        lags_timefeats = batch[6]
        
        return xc, x1, timefeats_c, timefeats_p, lags, lags_timefeats

    def preprocess_batch(self, batch, training=False):
        """Converts a batch of data into matched a random pair of (x0, x1)"""
        xc, x1, timefeats_c, timefeats_p, lags, lags_timefeats = self.unpack_batch(batch)
        x0 = []
        for i in range(self.num_samples):
            x_time = np.arange(x1.shape[1])      
            mu = np.zeros_like(x_time)       
            cov_matrix = gp_kernel(x1=x_time, x2=x_time, kernel='ou')
            samples = np.random.multivariate_normal(mu, cov_matrix, size=(x1.shape[0], x1.shape[2]))
            samples = torch.tensor(samples).to(x1.device).to(torch.float32)
            x0.append(samples.transpose(1,2))

        return x0, x1, xc, timefeats_c, timefeats_p, lags, lags_timefeats
    


    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        x0, x1, xc, timefeats_c, timefeats_p, lags, lags_timefeats = self.preprocess_batch(batch, training)
            
        loss=0
        logging_loss=0
        
        if self.deter_net is not None:
            if self.timefeats:
                f = self.deter_net(xc, timefeats_c)
            else:
                f = self.deter_net(xc, None)
                    

        for i in range(self.num_samples):
            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0[i], x1)
            
            if self.timefeats:
                gt = self.net(xt, t, xc, timefeats_c, timefeats_p)
            else:
                gt = self.net(xt, t, xc, None, None)
                
            if self.deter_net is not None:
                vt = gt+f
            else:
                vt = gt
            fm_loss = torch.nn.functional.mse_loss(vt, ut)
            logging_loss += fm_loss
            
            if self.deter_net is not None:
                mae_loss=torch.nn.functional.l1_loss(f, x1)
                loss += fm_loss + self.lambda_reg * mae_loss
            else:
                loss += fm_loss            

        return loss/self.num_samples, logging_loss/self.num_samples

    def training_step(self, batch: Any, batch_idx: int):
        loss, logging_loss = self.step(batch, training=True)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/fm_loss", logging_loss, on_step=True, prog_bar=False)
        return loss

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        loss, logging_loss = self.step(batch, training=False)
        self.log(f"{prefix}/loss", loss)
        self.log(f"{prefix}/fm_loss", logging_loss)
        return {"loss": loss, "x": batch, "fm_loss": logging_loss}
    
    
    def on_train_epoch_end(self):
        """Called at the end of the training epoch. This can be used for instance to update the train dataset
        with new data.

        """
        # Only update if not in the first epoch
        # If last epoch, we don't need to update since we will not use the dataset anymore
        if self.current_epoch < self.trainer.max_epochs - 1:
            log.info("Sampling training dataset for next epoch...")
            self.datamodule.get_modeling_dataset()

    def preprocess_epoch_end(self, outputs: List[Any], prefix: str):
        """Preprocess the outputs of the epoch end function."""
        v = {k: torch.cat([d[k] for d in outputs]) for k in ["x"]}
        x = v["x"]

        # Sample some random points for the plotting function
        rand = torch.randn_like(x)
        x = torch.stack([rand, x], dim=1)
        ts = x.shape[1]
        x0 = x[:, 0]
        x_rest = x[:, 1:]
        return ts, x, x0, x_rest


    def validation_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "val")
    
    def test_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "test")

    def train_dataloader(self):
        print('generating new training dataloader')
        return DataLoader(
            dataset=self.datamodule.data_train,
            batch_size=self.datamodule.hparams.batch_size,
            num_workers=self.datamodule.hparams.num_workers,
            pin_memory=self.datamodule.hparams.pin_memory,
            shuffle=True,
        )
    

    def val_dataloader(self):
        return DataLoader(
            dataset=self.datamodule.data_val,
            batch_size=self.datamodule.hparams.batch_size,
            num_workers=self.datamodule.hparams.num_workers,
            pin_memory=self.datamodule.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.datamodule.data_test,
            batch_size=self.datamodule.hparams.batch_size,
            num_workers=self.datamodule.hparams.num_workers,
            pin_memory=self.datamodule.hparams.pin_memory,
            shuffle=False,
        )
    

    def configure_optimizers(self):
        """Pass model parameters to optimizer."""
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is None:
            return optimizer

        scheduler = self.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)

    
    def on_fit_start(self):
        """
        Called at the very beginning of fit.
        If on DDP it is called on every process
        """
        if self.ema is not None:
            print(self.ema)
            self.ema.store(self.net.parameters())

    def on_fit_end(self):
        """
        Called at the very end of fit.
        If on DDP it is called on every process
        """
        if self.ema is not None:
            self.ema.copy_to(self.net.parameters())


    def on_pre_performance_check(self) -> None:
        """
        Called at the very beginning of the validation loop.
        """
        if self.ema is not None:
            self.ema.copy_to(self.net.parameters())


    def on_post_performance_check(self) -> None:
        """
        Called at the very end of the validation loop.
        """
        if self.ema is not None:
            self.ema.restore(self.net.parameters())

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        if self.ema is not None:
            self.ema.update(self.net.parameters())