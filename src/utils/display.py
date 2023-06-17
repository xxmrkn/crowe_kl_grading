import time

from dataclasses import dataclass
import torch

from utils.argparser import get_args 

@dataclass
class Display(object):

    opt = get_args() 
    fold: int

    # Display fold
    def display_fold(self) -> None:
        print(f'### Fold : {self.fold + 1}')
    
    # Display results
    def display_progress(self,
                         train_loss: int,
                         valid_loss: int) -> None:
        print(f'#'*10 + f'  Now training...  Fold : {self.fold+1}')
        print(f"Train Loss : {train_loss}")
        print(f"Valid Loss : {valid_loss}")
        print('#'*30)

    # Display epoch
    @classmethod
    def display_epoch(cls,
                      epoch: int) -> None:
        print(f'Epoch {epoch+1}/{cls.opt.epoch}')
        print('-' * 10)

    # Display something when the training finished  
    @staticmethod  
    def display_end(run_url: str,
                    since: int) -> None:

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed//60}m {time_elapsed%60:.2f}s')
        print(f"wandb website ------> {run_url}")

    # Display CUDA availability
    @staticmethod
    def display_availability():
        if torch.cuda.is_available():
            print(f"CUDA Available : {torch.cuda.get_device_name}")
        else:
            print(f"CUDA is not Available !!")