import cv2
import csv
import os
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.Parser import get_args
from utils.Configuration import CFG

def visualize_confusion_matrix(matrix, rowlabels, columnlabels):

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(matrix.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(matrix.shape[1]) + 0.5, minor=False)
    plt.xlabel('pred')
    plt.ylabel('true')

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(rowlabels, minor=False)
    ax.set_yticklabels(columnlabels, minor=False)
    plt.savefig("")


#last epoch and fold
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

        with open(tgt + f"", 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['ID','Actual','Pred'])
            writer.writerows(new)

        print('--> Saved Total Outlier csv')

    else:
        new2 = [[0]*3 for _ in range(len(id))]

        for i in range(len(path)):
            new2[i][0],new2[i][1],new2[i][2] = id[i],labels1[i],labels2[i]

        tgt2 = f"{CFG.results_path}/csv2pptx/{opt.model}"
        os.makedirs(tgt2, exist_ok=True)

        with open(tgt2 + f"", 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['ID','Actual','Pred'])
            writer.writerows(new2)
        
        print('--> Saved Total Outlier2 csv')

def plot_uncertainty(plob, indicator, path, fold_id, model_name, iteration, dir):
    total_uncertainty = [[],[],[],[],[]]

    cnt = [0,0,0]

    for i in range(len(plob)):
        if indicator[i]==0:
            total_uncertainty[0].append(path[i])
            total_uncertainty[1].append(model_name)
            total_uncertainty[2].append('Correct')
            total_uncertainty[3].append(plob[i])
            total_uncertainty[4].append(fold_id[i])
            cnt[0]+=1

        elif indicator[i]==-1 or indicator[i]==1:
            total_uncertainty[0].append(path[i])
            total_uncertainty[1].append(model_name)
            total_uncertainty[2].append('1 Neighbor')
            total_uncertainty[3].append(plob[i])
            total_uncertainty[4].append(fold_id[i])
            cnt[1]+=1

        elif indicator[i]>=2 or indicator[i]<=-2:
            total_uncertainty[0].append(path[i])
            total_uncertainty[1].append(model_name)
            total_uncertainty[2].append('Others')
            total_uncertainty[3].append(plob[i])
            total_uncertainty[4].append(fold_id[i])
            cnt[2]+=1

    list_row = pd.DataFrame(total_uncertainty)
    list_row = list_row.transpose()
    list_row.columns = ['Path','Model','Uncertainty','Probability','Fold']

    list_row.to_csv(f'')



def plot_uncertainty_mcdropout(plob,
                               indicator,
                               path,
                               fold_id,
                               model_name,
                               iteration,
                               dir,
                               mean,
                               total_pred,
                               true,
                               datalist):
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

    #list_row.to_csv(f'{CFG.results_path}/outputs/{CFG.model_name}/{CFG.sign}{CFG.num_classes}class_uncertainty_boxplot_{CFG.n_fold}fold_{CFG.epochs}epoch_list.csv')
    list_row.to_csv(f'')



def plot_uncertainty_mcdropout2(plob, indicator, path, model_name, iteration, mean, total_pred, true):
    total_uncertainty = [[],[],[],[],[],[],[]]

    cnt = [0,0,0]

    for i in range(len(plob)):
        if indicator[i]==0:
            total_uncertainty[0].append(path[i])
            total_uncertainty[1].append(model_name)
            total_uncertainty[2].append('Correct')
            total_uncertainty[3].append(plob[i])
            total_uncertainty[4].append(mean[i])
            total_uncertainty[5].append(total_pred[i])
            total_uncertainty[6].append(true[i])
            cnt[0]+=1

        elif indicator[i]==-1 or indicator[i]==1:
            total_uncertainty[0].append(path[i])
            total_uncertainty[1].append(model_name)
            total_uncertainty[2].append('1 Neighbor')
            total_uncertainty[3].append(plob[i])
            total_uncertainty[4].append(mean[i])
            cnt[1]+=1

        elif indicator[i]>=2 or indicator[i]<=-2:
            total_uncertainty[0].append(path[i])
            total_uncertainty[1].append(model_name)
            total_uncertainty[2].append('Others')
            total_uncertainty[3].append(plob[i])
            total_uncertainty[4].append(mean[i])
            cnt[2]+=1

    list_row = pd.DataFrame(total_uncertainty)
    list_row = list_row.transpose()
    list_row.columns = ['Path','Model','Uncertainty','Variance','Probability']
    list_row.to_csv(f'')