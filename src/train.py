import copy
import csv
import os
import time

import logging

import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

from utils.parser import get_args
from utils.Configuration import CFG
from utils.wandb_config import wandb_config
from datasets.dataset import TrainDataset,TestDataset,get_transforms
from model.select_model import choose_model
from function.load_datalist import load_train_data, load_valid_data
from function.prepare_dataframe import prepare_train_df, prepare_valid_df
from trainval_one_epoch import train_one_epoch, valid_one_epoch

import wandb


def main():

    #preparation training
    opt = get_args()
    CFG.set_seed(opt.seed)

    if torch.cuda.is_available():
        print(f"cuda: {torch.cuda.get_device_name}")
    
    #original dataframe for training
    data_df = pd.read_csv(opt.df_path)

    #load train/valid file
    train_file = load_train_data()
    valid_file = load_valid_data()

    #run training each fold
    for fold in [i for i in range(int(opt.fold))]:
        print(f'### Fold: {fold+1}')

        train_df = prepare_train_df(train_file,
                                    fold,
                                    data_df)
        valid_df = prepare_valid_df(valid_file,
                                    fold,
                                    data_df)

        train_dataset = TrainDataset(train_df,
                                     transform = get_transforms('train'))
        valid_dataset = TestDataset(valid_df,
                                    transform = get_transforms('valid'))

        train_loader = DataLoader(train_dataset,
                                  batch_size = opt.batch_size, 
                                  num_workers = opt.num_workers,
                                  shuffle = True,
                                  pin_memory=True,
                                  drop_last = False,)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size = opt.valid_batch_size, 
                                  num_workers = opt.num_workers,
                                  pin_memory=True,
                                  shuffle = False,
                                  drop_last=False,)

        #create model
        model = choose_model(opt.model)

        #Metrics
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr = opt.lr,
                                     weight_decay = opt.wd)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max = opt.t_max,
                                                   eta_min = opt.min_lr)
        #wandb
        run = wandb_config(model,fold)        

        #Training 
        since = time.time()
        best_loss = 100

        for epoch in range(opt.epoch):
            print(f'Epoch {epoch+1}/{opt.epoch}')
            print('-' * 10)

            train_loss = train_one_epoch(model, 
                                         optimizer, 
                                         scheduler, 
                                         criterion, 
                                         train_loader,)

            valid_loss = valid_one_epoch(model,
                                         optimizer,
                                         criterion,
                                         valid_loader)

            wandb.log({"Train Loss": train_loss, 
                       "Valid Loss": valid_loss,
                       "LR": scheduler.get_last_lr()[0],})

            #display results
            print(f'######## now training...  fold : {fold+1}')
            print(f"Train Loss: {train_loss}")
            print(f"Valid Loss: {valid_loss}")
            print('#'*50)

            if valid_loss < best_loss:
                best_loss = valid_loss
                tgt = f'{opt.result_path}/{opt.sign}/weights/{opt.model}'
                os.makedirs(tgt, exist_ok=True)
                path = os.path.join(tgt, f'{opt.sign}_datalist{opt.datalist}_fold{opt.fold}{fold+1}_{opt.epoch}epoch_weights.pth')

                torch.save(model.state_dict(),path)
                print('Save Model !')

        run.finish()

        #display wandb webpage link
        print(f"wandb website ------> {run.url}")

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed//3600}h {time_elapsed//60}m {time_elapsed%60:.2f}s')

    return model


if __name__ == '__main__':
    main()
