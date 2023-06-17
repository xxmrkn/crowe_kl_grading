import torch 
import  torch.nn as nn
from torch.optim import lr_scheduler
from utils.argparser import get_args

opt = get_args()

def select_criterion():
    if opt.criterion == "MAE Loss":
        criterion = nn.L1Loss()
    return criterion

def select_optimizer(param):
    if opt.optimizer == "Adam":
        optimizer = torch.optim.Adam(param,
                                     lr = opt.lr,
                                     weight_decay = opt.wd)
    return optimizer

def select_scheduler(optimizer):
    if opt.scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max = opt.t_max,
                                                   eta_min = opt.min_lr)
    return scheduler
