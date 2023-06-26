import os
import torch
from dataclasses import dataclass
from typing import ClassVar

import hydra
from omegaconf import DictConfig, OmegaConf


@dataclass
class SavingHelper(object):

    @staticmethod
    def save_model(valid_loss,
                   best_loss,
                   model, 
                   fold,
                   cfg):
        
        if valid_loss < best_loss:
            best_loss = valid_loss

            if cfg.models.model.num_classes == 1:
                tgt = f'{cfg.models.path.result}/{cfg.models.sign}/Regression/weights/{cfg.models.model.name}'
                os.makedirs(tgt, exist_ok=True)

            else:
                tgt = f'{cfg.models.path.result}/{cfg.models.sign}/Classification/weights/{cfg.models.model.name}'
                os.makedirs(tgt, exist_ok=True)

            path = os.path.join(
                       tgt,
                       f'{cfg.models.sign}_datalist{cfg.models.general.datalist}_fold{cfg.models.general.fold}{fold+1}'
                       f'_{cfg.models.train.epoch}epoch_weights.pth'
                   )

            torch.save(model.state_dict(), path)
            print('Save Model !')

        return best_loss