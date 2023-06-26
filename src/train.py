import time

import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from torch.utils.data import DataLoader
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.configuration_helper import ConfigurationHelper
from utils.wandb_helper import WandbHelper
from utils.display_helper import DisplayHelper
from utils.saving_helper import SavingHelper
from utils.metric_helper import MetricHelper
from utils.loading_helper import LoadingDataHelper
from utils.dataframe_helper import DataframeHelper
from datasets.dataset import TrainDataset, ValidDataset, Transform
from models.select_model import select_model
from train_validation_one_epoch import Training

@DisplayHelper.display_time_spend
@hydra.main(version_base=None,
            config_path="configs/",
            config_name="config")
def main(cfg: DictConfig) -> None:

    # Display Config Setting
    DisplayHelper.display_config(cfg)

    # Preparation for training
    ConfigurationHelper.set_seed(cfg.models.general.seed)

    # Display CUDA availability
    DisplayHelper.display_availability()
    
    # Original dataframe for training
    data_df = pd.read_csv(cfg.models.path.df)

    # load train/valid file
    train_file = LoadingDataHelper.load_data(cfg, 'train')
    valid_file = LoadingDataHelper.load_data(cfg, 'valid')

    # Run training for each fold
    for fold in [i for i in range(cfg.models.general.fold)]:
        DisplayHelper(fold).display_fold()

        # Prepare dataframe
        train_df = DataframeHelper(data_df, fold).prepare_df(train_file)
        valid_df = DataframeHelper(data_df, fold).prepare_df(valid_file)

        # Dataset
        train_dataset = TrainDataset(cfg,
                                     train_df,
                                     transform = Transform.get_transforms(cfg, 'train'))
        valid_dataset = ValidDataset(cfg,
                                     valid_df,
                                     transform = Transform.get_transforms(cfg, 'valid'))
        
        # Data Loader
        train_loader = DataLoader(train_dataset,
                                  batch_size = cfg.models.general.batch_size, 
                                  num_workers = cfg.models.general.num_workers,
                                  shuffle = True,
                                  pin_memory = True,
                                  drop_last = False,)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size = cfg.models.general.valid_batch_size, 
                                  num_workers = cfg.models.general.num_workers,
                                  pin_memory = True,
                                  shuffle = False,
                                  drop_last = False,)

        # Create model
        model = select_model(cfg)

        # Metrics
        criterion = MetricHelper.select_criterion(cfg)
        optimizer = MetricHelper.select_optimizer(cfg, model.parameters())
        scheduler = MetricHelper.select_scheduler(cfg, optimizer)
        
        # Wandb
        run = WandbHelper.wandb_config(cfg, model, fold)        

        since = time.time()
        best_loss = 10**9

        # Training loop 
        for epoch in range(cfg.models.train.epoch):
            DisplayHelper.display_epoch(cfg, epoch)

            # Training loop
            train_loss = Training(model,
                                  optimizer,
                                  scheduler,
                                  criterion,
                                  train_loader).train_one_epoch(cfg)

            # Validation loop
            valid_loss = Training(model,
                                  optimizer,
                                  scheduler,
                                  criterion,
                                  valid_loader).valid_one_epoch(cfg)
            
            # Display results
            DisplayHelper(fold).display_progress(train_loss,
                                                 valid_loss) 

            # Save best model
            best_loss = SavingHelper.save_model(valid_loss,
                                                best_loss,
                                                model, 
                                                fold,
                                                cfg)     
                
            # Logging
            wandb.log({"Train Loss": train_loss, 
                       "Valid Loss": valid_loss,
                       "LR": scheduler.get_last_lr()[0]})

        run.finish()

        #DisplayHelper.display_end(run.url, since)
        ConfigurationHelper.init_list()
        LoadingDataHelper.init_list()


if __name__ == '__main__':
    main()