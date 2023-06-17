import time

import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from torch.utils.data import DataLoader

from utils.argparser import get_args
from utils.configuration import Configuration
from utils.wandb_config import wandb_config
from utils.display import Display
from utils.save_model import SaveModel
from utils.select_metric import select_criterion, select_optimizer, select_scheduler
from datasets.dataset import TrainDataset, TestDataset, get_transforms
from models.select_model import select_model
from utils.load_datalist import LoadData
from utils.prepare_dataframe import PrepareDataframe
from trainval_one_epoch import Training

import wandb


def main():

    # Preparation for training
    opt = get_args()
    Configuration.set_seed(opt.seed)

    # Display CUDA availability
    Display.display_availability()
    
    # Original dataframe for training
    data_df = pd.read_csv(opt.df_path)

    # load train/valid file
    train_file = LoadData.load_train_data()
    valid_file = LoadData.load_valid_data()

    # Run training for each fold
    for fold in [i for i in range(int(opt.fold))]:
        Display(fold).display_fold()

        # Prepare dataframe
        train_df = PrepareDataframe(fold, data_df).prepare_train_df(train_file)
        valid_df = PrepareDataframe(fold, data_df).prepare_valid_df(valid_file)

        # Dataset
        train_dataset = TrainDataset(train_df,
                                     transform = get_transforms('train'))
        valid_dataset = TestDataset(valid_df,
                                    transform = get_transforms('valid'))
        
        # Data Loader
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

        # Create model
        model = select_model(opt.model)

        # Metrics
        criterion = select_criterion()
        optimizer = select_optimizer(model.parameters())
        scheduler = select_scheduler(optimizer)
        
        # Wandb
        run = wandb_config(model,fold)        

        since = time.time()
        best_loss = 100

        # Training loop 
        for epoch in range(opt.epoch):
            Display.display_epoch(epoch)

            # Training loop
            train_loss = Training(model,
                                  optimizer,
                                  scheduler,
                                  criterion,
                                  train_loader).train_one_epoch()

            # Validation loop
            valid_loss = Training(model,
                                  optimizer,
                                  scheduler,
                                  criterion,
                                  valid_loader).valid_one_epoch()
            
            # Logging
            wandb.log({"Train Loss": train_loss, 
                       "Valid Loss": valid_loss,
                       "LR": scheduler.get_last_lr()[0],})

            # Display results
            Display(fold).display_progress(train_loss,
                                           valid_loss)

            # Save best model
            best_loss = SaveModel.save_model(valid_loss,
                                             best_loss,
                                             model, 
                                             fold)            

        run.finish()

        Display.display_end(run.url, since)

    return model


if __name__ == '__main__':
    main()
