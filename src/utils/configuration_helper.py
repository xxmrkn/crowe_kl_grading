import os
import random
from dataclasses import dataclass, field
from typing import ClassVar, Callable

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import numpy as np

@dataclass
class ConfigurationHelper(object):

    key: ClassVar[str] = '' 

    difference: ClassVar[list] = []
    original_outputs: ClassVar[list] = []
    path_list: ClassVar[list] = []
    fold_id: ClassVar[list] = []
    ground_truth: ClassVar[list] = []
    prediction_variance: ClassVar[list] = []
    prediction_mean: ClassVar[list] = []
    total_predict: ClassVar[list] = []
    total_correct: ClassVar[list] = []

    predicts_array: ClassVar[list] = []
    predicts_array_float: ClassVar[list] = []

    device: ClassVar[Callable] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    labels_dict: ClassVar[dict] = {}

    labels: ClassVar[list] = [0,1,2,3,4,5,6]

    labels_name: ClassVar[list] = []

    labels_index: ClassVar[list] = []

    # def constract_array(self):
    #     predicts_array = [[] for _ in range(self.cfg.inference.num_sampling)]
    #     predicts_array_float = [[] for _ in range(self.cfg.inference.num_sampling)]

    #     return predicts_array, predicts_array_float
    
    @classmethod 
    def init_list(cls):
        cls.difference = []
        cls.original_outputs = []
        cls.path_list = []
        cls.fold_id = []
        cls.ground_truth = []
        cls.prediction_variance = []
        cls.prediction_mean = []
        cls.total_predict = []
        cls.total_correct = []
    
    @classmethod 
    def init_predict_list(cls, cfg):
        cls.predicts_array = [[] for _ in range(cfg.models.inference.num_sampling)]
        cls.predicts_array_float = [[] for _ in range(cfg.models.inference.num_sampling)]

    @staticmethod 
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu()

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        print('> SEEDING DONE')
