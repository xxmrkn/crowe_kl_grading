from dataclasses import dataclass

from tqdm import tqdm
tqdm.pandas()
import torch

from utils.argparser import get_args
from utils.configuration import Configuration

@dataclass
class Training(object):

    model: None
    optimizer: None
    scheduler: None
    criterion: None
    dataloader: None 

    def train_one_epoch(self):

        self.model.train()
        grad_scaler = torch.cuda.amp.GradScaler()
        
        running_loss = 0.0
        
        pbar = tqdm(enumerate(self.dataloader, start=1),
                    total=len(self.dataloader),
                    desc='Train',
                    disable=True)

        for step, (inputs, labels) in pbar:         
            inputs = inputs.to(Configuration.device)
            labels  = labels.to(Configuration.device)
            labels = labels.unsqueeze(1)

            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs.to(torch.float32), 
                                      labels.to(torch.int64))

                running_loss += loss.item()
                train_loss = running_loss / step

            grad_scaler.scale(loss).backward()
            grad_scaler.step(self.optimizer)
            grad_scaler.update()

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix(train_loss = f'{train_loss:0.6f}',
                             lr = f'{current_lr:0.6f}',
                             gpu_mem = f'{mem:0.2f} GB')
            
        self.scheduler.step()     
        
        return train_loss


    def valid_one_epoch(self):

        #self.model.train()# turn ON dropout
        self.model.eval()# turn OFF dropout

        with torch.no_grad():
            
            dataset_size = 0
            running_loss = 0.0
            
            pbar = tqdm(enumerate(self.dataloader, start=1),
                        total=len(self.dataloader),
                        desc='Valid',
                        disable=True)

            for step, (inputs, labels, image_path, image_id) in pbar:        
                inputs  = inputs.to(Configuration.device)
                labels  = labels.to(Configuration.device)
                labels = labels.unsqueeze(1)

                batch_size = inputs.size(0)

                outputs = self.model(inputs)

                loss = self.criterion(outputs.to(torch.float32),
                                      labels.to(torch.int64))
                
                running_loss += loss.item()
                dataset_size += batch_size

                valid_loss = running_loss / step

                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix(valid_loss = f'{valid_loss:0.6f}',
                                 lr = f'{current_lr:0.6f}',
                                 gpu_memory = f'{mem:0.2f} GB')
            
        return valid_loss