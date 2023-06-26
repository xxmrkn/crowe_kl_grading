from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, Callable
import os
import pathlib

import hydra
from omegaconf import DictConfig, OmegaConf


def load_config_from_yaml(yaml_path: str) -> DictConfig:
    config = OmegaConf.load(yaml_path)
    return config


@dataclass
class LoadingWeightHelper(object):

    model_file: ClassVar[list[str]] = []
    
    @classmethod
    def get_weight_path(cls, cfg):

        if cfg.models.model.num_classes == 1:
            grading = 'Regression'
        else:
            grading = 'Classification'
        
        weight_path = pathlib.Path(
                          f'{cfg.models.path.result}/{cfg.models.sign}/{grading}/weights/{cfg.models.model.name}/'
                      ).glob(f'{cfg.models.sign}_datalist{cfg.models.general.datalist}*.pth')
        
        for i in weight_path:
            cls.model_file.append(
                f'{cfg.models.path.result}/{cfg.models.sign}/{grading}/weights/{cfg.models.model.name}/{i.name}'
            )
        if len(cls.model_file) != 0:
            print('- Loaded Pretrained Weight')
        else:
            print('- Model Path Read Failed !!')
        
        return cls.model_file
    
    @classmethod
    def init_list(cls):
        cls.model_file = []

@dataclass
class LoadingDataHelper(object):
    
    file_list: ClassVar[defaultdict[list]] = defaultdict(list)
    name_list: ClassVar[defaultdict[list]] = defaultdict(list)
    
    @classmethod
    def load_data(cls, cfg, mode) -> list[str]:

        p = pathlib.Path(
                f'{cfg.models.path.base}/datalist/datalist{cfg.models.general.datalist}'
                f'/k{cfg.models.general.fold}'
            ).glob(f'{mode}*.txt')
                
        for i in p:
            cls.file_list[mode].append(
                os.path.join(f'k{str(cfg.models.general.fold)}', i.name)
            )

        for name in cls.file_list[mode]:
            cls.name_list[mode].append(
                os.path.join(cfg.models.path.datalist,
                f'datalist{cfg.models.general.datalist}',
                name)
            )

        if len(cls.name_list[mode]) != cfg.models.general.fold:
            raise RuntimeError(f'Read fail. There are no text files.')

        return cls.name_list[mode]
    
    @classmethod
    def init_list(cls):
        cls.file_list = defaultdict(list)
        cls.name_list = defaultdict(list)
