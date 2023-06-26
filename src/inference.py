import argparse
import os
import pickle
import pathlib
import time
import itertools

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torchvision import models
from torch.utils.data import DataLoader

from tqdm import tqdm
tqdm.pandas()

from train_validation_one_epoch import Test
from utils.configuration_helper import ConfigurationHelper
from datasets.dataset import TestDataset
from datasets.dataset import Transform
from utils.display_helper import DisplayHelper
from utils.loading_helper import LoadingDataHelper
from utils.loading_helper import LoadingWeightHelper
from utils.dataframe_helper import DataframeHelper
from utils.summarize_helper import SummarizeHelper
from models.select_model import select_model
from evaluation.evaluation_helper import EvaluationHelper
from visualization.visualization_helper import VisualizeHelper

@DisplayHelper.display_time_spend
@hydra.main(version_base=None,
            config_path="configs/",
            config_name="config")
def main(cfg: DictConfig) -> None:
    
    # Display Config Setting
    DisplayHelper.display_config(cfg)
    
    # Data (Weight) Preparation 
    original_df = pd.read_csv(cfg.models.path.df)
    test_file = LoadingDataHelper.load_data(cfg, 'test')
    model_file = LoadingWeightHelper.get_weight_path(cfg)

    # Initiallize Array
    ConfigurationHelper.predicts_array = [[] for _ in range(cfg.models.inference.num_sampling)]
    ConfigurationHelper.predicts_array_float = [[] for _ in range(cfg.models.inference.num_sampling)]

    # Main Loop
    for fold, weight_path in enumerate(model_file):

        model = select_model(cfg)
        model.load_state_dict(torch.load(weight_path))
        
        test_df = DataframeHelper(original_df,
                                  fold).prepare_df(test_file)

        test_dataset = TestDataset(cfg,
                                   test_df,
                                   transform = Transform.get_transforms(cfg, 'valid'))                                                           

        test_loader = DataLoader(test_dataset,
                                 batch_size = cfg.models.general.valid_batch_size, 
                                 num_workers = cfg.models.general.num_workers,
                                 shuffle = False,
                                 pin_memory = True)
        
        
        # Monte-Carlo Dropout
        for num_sampling in range(cfg.models.inference.num_sampling):
            if num_sampling != 0:
                print(f'          ITER : {num_sampling+1}')
            else:
                print(f'FOLD : {fold+1}, ITER : {num_sampling+1}')
            
            # Inferenece
            Test(model,
                 fold,
                 num_sampling,
                 test_loader).test_one_epoch(cfg)
    
    # Calculate Values
    calculated_value = SummarizeHelper.calculate_value(cfg)
    
    # Helper function to summarize results
    VisualizeHelper.create_result_csv(cfg, *calculated_value)

    # Initialize Class-variables
    ConfigurationHelper.init_list()
    ConfigurationHelper.init_predict_list(cfg) 
    LoadingWeightHelper.init_list()
    LoadingDataHelper.init_list()
    
if __name__ == '__main__':
    main()