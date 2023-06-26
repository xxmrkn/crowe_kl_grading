import argparse
from dataclasses import dataclass
from typing import ClassVar
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.configuration_helper import ConfigurationHelper

@dataclass
class WandbHelper(object):

    @staticmethod
    def wandb_config(cfg, model, fold):

        wandb.login(key=ConfigurationHelper.key)
        run = wandb.init(mode = cfg.models.general.wandb_mode,
                         project = f'{cfg.models.sign}{cfg.models.model.num_classes}class_crowe_kl-regression', 
                         config = {"model_name": cfg.models.model.name,
                                   "learning_rate": cfg.models.optimizer.lr,
                                   "datalist": cfg.models.general.datalist,
                                   "fold": cfg.models.general.fold,
                                   "epochs": cfg.models.train.epoch,
                                   "image_size": cfg.models.general.image_size,
                                   "batch_size": cfg.models.general.batch_size,
                                   "num_workers": cfg.models.general.num_workers,
                                   "num_classes": cfg.models.model.num_classes,
                                   "optimizer": cfg.models.optimizer.name,
                                   "loss": cfg.models.criterion.name,
                                   "sign": cfg.models.sign},
                         entity = "xxmrkn",
                         name = f"{cfg.models.sign}|datalist{cfg.models.general.datalist}|{cfg.models.model.num_classes}class|"
                                f"{cfg.models.model.name}|{cfg.models.general.fold}fold"
                                f"|fold-{fold+1}|batch-{cfg.models.general.batch_size}|lr-{cfg.models.optimizer.lr}")

        wandb.watch(model, log_freq=100)

        return run