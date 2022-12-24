import argparse
import os
import pickle
import pathlib
import time
import itertools

import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torchvision import models
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from datasets.dataset import TestDataset
tqdm.pandas()

from utils.parser import get_args
from utils.Configuration import CFG
import datasets.dataset as dataset
from function.load_datalist import load_test_data
from function.prepare_dataframe import prepare_test_df
from model.select_model import choose_model
from evaluation.EvaluationHelper import EvaluationHelper
from visualization.VisualizeHelper import plot_uncertainty_mcdropout


def main():
    N = [str(i) for i in range(8,23)] #datalist8-datalist22 total=15
    print(N)
    opt = get_args()
    
    for k in N:
        
        since = time.time()
        #original dataframe for training
        data_df = pd.read_csv(opt.df_path)
        #load train/valid file
        test_file = load_test_data() 

        model_file = []
        p_w = pathlib.Path(f'{opt.result_path}/{opt.sign}/weights/{opt.model}/').glob(f'{opt.sign}_datalist{k}*.pth')
        for i in p_w:
            model_file.append(f'{opt.result_path}/{opt.sign}/weights/{opt.model}/{i.name}')
        print(model_file)

        total_predicts = []
        predicts = [[] for _ in range(opt.num_sampling)]
        predicts_float = [[] for _ in range(opt.num_sampling)]
        total_confusion_matrix = 0

        for fold,i in enumerate(model_file): #fold毎の.pthのループ
            print(f'datalist{k}, fold{fold}, modelfile{i}')

            model = choose_model(opt.model)
            model.load_state_dict(torch.load(i))

            for j in range(opt.num_sampling): #各foldをnum_inferenceの回数ループ(MCdropout)

                test_df = prepare_test_df(test_file, fold, data_df)

                test_dataset = TestDataset(test_df,
                                           transform=dataset.get_transforms('valid'))                                                           

                test_loader = DataLoader(test_dataset,
                                         batch_size=opt.batch_size, 
                                         num_workers=opt.num_workers,
                                         shuffle=False,
                                         pin_memory=True,
                                         drop_last=False)

                print(f'fold {fold+1}, iter {j+1}')

                #inferenece
                model.train()# turn ON/OFF dropout
                #model.eval()
                with torch.no_grad():
                        
                    pbar = tqdm(enumerate(test_loader, start=1),
                                total=len(test_loader), 
                                desc='Test',
                                disable=True)
                    for step, (inputs, labels, image_path, image_id) in pbar:                              
                        inputs = inputs.to(CFG.device)
                        labels = labels.tolist()

                        outputs = model(inputs)
                        outputs2 = outputs.tolist()

                        outputs2 = list(itertools.chain.from_iterable(outputs2))
                        outputs = EvaluationHelper.threshold_config(outputs)
                        outputs = list(itertools.chain.from_iterable(outputs))

                        acc_score = EvaluationHelper.acc(outputs,labels)
                        total_confusion_matrix += EvaluationHelper.conf_mtrx(outputs,labels)
                        
                        predicts[j].extend(outputs)
                        predicts_float[j].extend(outputs2)

                        torch.cuda.empty_cache()

                        if j == opt.num_sampling-1:
                            CFG.true.extend(labels)
                            CFG.path_list.extend(image_id)
                            CFG.fold_id.extend([fold+1]*len(labels))
        
        
        total = np.array(predicts_float)

        pre_avg = np.mean(total,axis=0)
        pre_var = np.var(total,axis=0)

        out = EvaluationHelper.threshold_config_for_inf(pre_avg)

        difference = np.array(out) - np.array(CFG.true)
        #print('diff',difference)

        plot_uncertainty_mcdropout(pre_var,
                                   difference,
                                   CFG.path_list,
                                   CFG.fold_id,
                                   opt.model,
                                   opt.num_sampling,
                                   pre_avg,
                                   out,
                                   CFG.true,
                                   k)

        time_elapsed = time.time() - since
        print(f'Inference complete in {time_elapsed//3600}h {time_elapsed//60}m {time_elapsed%60:.2f}s')

        CFG.difference = []
        CFG.probability = []
        CFG.path_list = []
        CFG.fold_id = []
        CFG.true = []
        CFG.var = []
        CFG.mean = []
        CFG.total_predict = []
        CFG.total_correct = []

if __name__ == '__main__':
    main()