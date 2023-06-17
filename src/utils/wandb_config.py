import argparse
import wandb
from utils.configuration import Configuration
from utils.argparser import get_args

def wandb_config(model,fold):
    
    opt = get_args()

    wandb.login(key=Configuration.key)
    run = wandb.init(project = f'{opt.sign}{opt.num_classes}class_crowe_kl-regression', 
                        config = {"model_name": opt.model,
                                  "learning_rate": opt.lr,
                                  "datalist": opt.datalist,
                                  "fold": opt.fold,
                                  "epochs": opt.epoch,
                                  "image_size": opt.image_size,
                                  "batch_size": opt.batch_size,
                                  "num_workers": opt.num_workers,
                                  "num_classes": opt.num_classes,
                                  "optimizer": opt.optimizer,
                                  "loss": opt.criterion,
                                  "sign": opt.sign},
                        entity = "",
                        name = f"{opt.sign}|datalist{opt.datalist}|{opt.num_classes}class|{opt.model}|{opt.fold}fold"
                               f"|fold-{fold+1}|batch-{opt.batch_size}|lr-{opt.lr}")

    wandb.watch(model, log_freq=100)

    return run