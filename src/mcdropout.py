import argparse
import os
import pickle
import pathlib
import time

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
from src.dataset.dataset import TestDataset
tqdm.pandas()

import src.dataset.dataset as dataset
from src.model.select_model import choose_model
from utils.Configuration import CFG
from src.visualization.VisualizeHelper import plot_uncertainty, plot_uncertainty_mcdropout


def main():
    N = [str(i) for i in range(8,23)] #datalist8-datalist23 total=15
    print(N)
    
    for k in N:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--sampling', type=int, default=50, help="Number of Sampling")
        parser.add_argument('--model', type=str, default='VisionTransformer_Base16', help="Model name")
        parser.add_argument('--dir', type=str, default='', help="Directory name")
        
        opt = parser.parse_args()

        since = time.time()
        print(f"datalist = {k}")
        print(f'num_sampling = {opt.num_sampling}')
        print(f'model name = {opt.models}')
        print(f'dir = {opt.dir}')

        data_df = pd.read_csv(CFG.df_path)

        file_names = []

        p = pathlib.Path(f'../datalist{k}/k{CFG.n_fold}').glob('test*.txt')
        for i in p:
            file_names.append(f'k{CFG.n_fold}/'+i.name)

        name = []
        for j in range(len(file_names)):
            for i in range(CFG.n_fold):
                if str(i) in file_names[j]:
                    name.append(os.path.join(CFG.fold_path+f"datalist{k}/", file_names[j]))

        model_file = []
        p_w = pathlib.Path(f'{CFG.model_path}/{opt.models}/').glob(f'0908_train{int(k)-7}7*.pth')
        for i in p_w:
            model_file.append(f'{CFG.model_path}/{opt.models}/{i.name}')

        total_predicts = []
        predicts = [[] for _ in range(opt.num_sampling)]
        for fold,i in enumerate(model_file): #fold毎の.pthのループ

            model = choose_model(opt.models)
            model.load_state_dict(torch.load(i))

            for j in range(opt.num_sampling): #各foldをnum_inferenceの回数ループ(MCdropout)

                with open(name[fold]) as f:
                    line = f.read().splitlines()

                valid_df = data_df[data_df['UID'].isin(line)]

                valid_dataset = TestDataset(valid_df,
                                            transform=dataset.get_transforms('valid'))                                                           

                valid_loader = DataLoader(valid_dataset,
                                        batch_size=CFG.batch_size, 
                                        num_workers=CFG.num_workers,
                                        shuffle=False,
                                        pin_memory=True)

                print(f'fold {fold+1}, iter {j+1}')
                print(line)#foldに含まれるK番号の一覧
                print(len(line))

                #inferenece
                model.train()# turn ON/OFF dropout
                #model.eval()
                with torch.no_grad():
                        
                    pbar = tqdm(enumerate(valid_loader, start=1), total=len(valid_loader), desc='Valid ')
                    for step, (inputs, labels, image_path) in pbar:                              
                        inputs  = inputs.to(CFG.device)
                        labels  = labels.to(CFG.device)

                        outputs  = model(inputs)

                        n = nn.Softmax(dim=1)
                        outputs = n(outputs)
                        
                        predicts[j].extend(outputs.tolist())

                        torch.cuda.empty_cache()

                        if j == opt.num_sampling-1:
                            CFG.true.extend(CFG.to_numpy(labels).numpy())
                            CFG.path_list.extend(image_path)
                            CFG.fold_id.extend([fold+1]*len(labels))

        total = np.array(predicts)
        total

        pre_avg = np.mean(total,axis=0)
        pre_var = np.var(total,axis=0)

        for i in range(pre_avg.shape[0]):
            pred = np.argmax(pre_avg[i])
            CFG.var.append(pre_var[i][pred])
            CFG.mean.append(pre_avg[i][pred])
            CFG.difference.append(pred - CFG.true[i])
            CFG.total_predict.append(pred)

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