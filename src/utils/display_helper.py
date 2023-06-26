import datetime 
from dataclasses import dataclass
import time
from typing import ClassVar
import torch
import hydra
from omegaconf import DictConfig, OmegaConf


@dataclass
class DisplayHelper(object):

    fold: int
    
    # Display fold
    def display_fold(self) -> None:
        print(f'### FOLD : {self.fold + 1}')
    
    # Display results
    def display_progress(self,
                         train_loss: float,
                         valid_loss: float) -> None:
        print(f'-'*10 + f'  Now training...  Fold : {self.fold+1}')
        print(f"Train Loss : {train_loss}")
        print(f"Valid Loss : {valid_loss}")
        print('-'*40)
    
    @staticmethod
    def display_config(cfg) -> None:
        print()
        print("#"*10 + " Configuration " + "#"*10)
        print()
        print(OmegaConf.to_yaml(cfg))
        print("#"*33)
        print()

    # Display epoch
    @staticmethod
    def display_epoch(cfg,
                      epoch: int) -> None:
        print(f'Epoch {epoch+1}/{cfg.models.train.epoch}')

    # Display something when the training finished  
    @staticmethod 
    def display_end(run_url: str,
                    since: int) -> None:

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed//60}m {time_elapsed%60:.2f}s')
        print(f"wandb website ------> {run_url}")
        print()

    # Display CUDA availability
    @staticmethod
    def display_availability():
        if torch.cuda.is_available():
            print(f"CUDA is Available : {torch.cuda.get_device_name}")
            print()
        else:
            print(f"CUDA is not Available !!")
            print()

    # Display start & end & total time
    @staticmethod
    def display_time_spend(func):
        def print_time():

            t_delta = datetime.timedelta(hours=9)
            JST = datetime.timezone(t_delta, 'JST')
            now = datetime.datetime.now(JST)
            
            print('#'*50)
            print(f'### START : {now}')
            print('#'*50)

            func()

            print('#'*50)
            print(f'### START : {now}')
            print(f'### END : {datetime.datetime.now(JST)}')
            print(f'### TOTAL TIME : {datetime.datetime.now(JST) - now}')
            print('#'*50)

        return print_time