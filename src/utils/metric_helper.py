from dataclasses import dataclass
from typing import ClassVar

import hydra
from omegaconf import OmegaConf, DictConfig

import torch 
import  torch.nn as nn
from torch.optim import lr_scheduler
from evaluation.evaluation_helper import FocalLoss

@dataclass
class MetricHelper(object):

    @staticmethod
    def select_criterion(cfg):
        if cfg.models.criterion.name == "MAE Loss":
            criterion = nn.L1Loss()

        elif cfg.models.criterion.name == 'Focal Loss':
            criterion = FocalLoss()

        else:
            print(f"{cfg.models.criterion.name} Is Not Undefined !!")

        return criterion

    @staticmethod
    def select_optimizer(cfg, param):
        if cfg.models.optimizer.name == "Adam":
            optimizer = torch.optim.Adam(param,
                                         lr = cfg.models.optimizer.lr,
                                         weight_decay = cfg.models.optimizer.wd)
        else:
            print(f"{cfg.models.optimizer.name} Is Not Undefined !!")

        return optimizer
    
    @staticmethod
    def select_scheduler(cfg, optimizer):
        if cfg.models.scheduler.name == "CosineAnnealingLR":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max = cfg.models.scheduler.t_max,
                                                       eta_min = cfg.models.scheduler.min_lr)
        else:
            print(f"{cfg.models.scheduler.name} Is Not Undefined !!")

        return scheduler
