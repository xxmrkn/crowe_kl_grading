#from types import new_class
import cv2
import csv
import os
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.parser import get_args
from utils.Configuration import CFG
#from csv2pptx.csv2pptx import csv2pptx
#import seaborn as sns

def visualize_total_image(path,
                          id,
                          labels1,
                          labels2, 
                          num,
                          normal_acc, 
                          neighbor_acc,
                          flag):

    opt = get_args()

    if flag==1:
        new = [[0]*3 for _ in range(len(id))]

        for i in range(len(id)):
            new[i][0],new[i][1],new[i][2] = id[i],labels1[i],labels2[i]

        tgt = f"{CFG.results_path}/csv2pptx/{opt.model}"
        os.makedirs(tgt, exist_ok=True)

        with open(tgt + f"/{opt.sign}{opt.num_classes}class_{opt.fold}fold_{opt.epoch}epoch_outlier.csv", 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['ID','Actual','Pred'])
            writer.writerows(new)

        #csv2pptx(tmp_path,pptx_path)
        print('--> Saved Total Outlier csv')

    else:
        new2 = [[0]*3 for _ in range(len(id))]

        for i in range(len(path)):
            new2[i][0],new2[i][1],new2[i][2] = id[i],labels1[i],labels2[i]

        tgt2 = f"{CFG.results_path}/csv2pptx/{opt.model}"
        os.makedirs(tgt2, exist_ok=True)

        with open(tgt2 + f"/{opt.sign}{opt.num_classes}class_{opt.fold}fold_{opt.epoch}epoch_outlier2.csv", 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['ID','Actual','Pred'])
            writer.writerows(new2)
        
        #csv2pptx(tmp_path2,pptx_path2)
        print('--> Saved Total Outlier2 csv')


def plot_uncertainty_mcdropout(plob,
                               indicator,
                               path,
                               fold_id,
                               model_name,
                               iteration,
                               mean,
                               total_pred,
                               true,
                               datalist):
    opt = get_args()

    #plob : tensor([0.1873, 0.1862, 0.2046, 0.1839], device='cuda:0')
    #indicator : tensor([ 1, -1,  2, -1], device='cuda:0')
    # indicatorが0なら正解したサンプルの不確実性としてプロット
    # indicatorが-1,1なら1クラス外したサンプルの不確実性としてプロット
    # indicatorが-2より小さい、もしくは2より大きいのであれば、2クラス以上外したサンプルの不確実性としてプロット
    total_uncertainty = [[],[],[],[],[],[],[],[]]

    cnt = [0,0,0]

    for i in range(len(plob)):
        if indicator[i]==0:
            total_uncertainty[0].append(path[i])
            total_uncertainty[1].append(model_name)
            total_uncertainty[2].append('Exact')
            total_uncertainty[3].append(fold_id[i])
            total_uncertainty[4].append(plob[i])
            total_uncertainty[5].append(mean[i])
            total_uncertainty[6].append(true[i])
            total_uncertainty[7].append(total_pred[i])
            cnt[0]+=1

        elif indicator[i]==-1 or indicator[i]==1:
            total_uncertainty[0].append(path[i])
            total_uncertainty[1].append(model_name)
            total_uncertainty[2].append('1 Neighbor')
            total_uncertainty[3].append(fold_id[i])
            total_uncertainty[4].append(plob[i])
            total_uncertainty[5].append(mean[i])
            total_uncertainty[6].append(true[i])
            total_uncertainty[7].append(total_pred[i])
            cnt[1]+=1

        elif indicator[i]>=2 or indicator[i]<=-2:
            total_uncertainty[0].append(path[i])
            total_uncertainty[1].append(model_name)
            total_uncertainty[2].append('Others')
            total_uncertainty[3].append(fold_id[i])
            total_uncertainty[4].append(plob[i])
            total_uncertainty[5].append(mean[i])
            total_uncertainty[6].append(true[i])
            total_uncertainty[7].append(total_pred[i])
            cnt[2]+=1

    list_row = pd.DataFrame(total_uncertainty)
    list_row = list_row.transpose()
    list_row.columns = ['Path','Model','Uncertainty','Fold','Variance','Probability','True Label','Pred Label']
    print(list_row)

    tgt = f'{opt.result_path}/{opt.sign}/uncertainty/{opt.model}/'
    os.makedirs(tgt, exist_ok=True)

    list_row.to_csv(tgt + f'{opt.sign}_datalist{datalist}_uncertainty_{opt.epoch}epoch_iter{opt.num_sampling}.csv', index=False)







