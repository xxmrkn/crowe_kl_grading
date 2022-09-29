import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import f1_score,accuracy_score,classification_report
from sklearn.metrics import confusion_matrix

from utils.Configuration import CFG
from utils.Parser import get_args

class EvaluationHelper:

    def f_measure(y_pred, y_true):
        return f1_score(y_true,y_pred,average="macro")
        
    
    def accuracy(y_pred, y_true):
        return accuracy_score(y_true,y_pred)

    
    def conf_mtrx(y_pred, y_true):
        return confusion_matrix(y_true,y_pred,labels=CFG.labels)


    def one_mistake_acc(matrix,dataset_size):
        taikaku1 = sum(np.diag(matrix))
        taikaku2 = sum(np.diag(matrix, k=1)) + sum(np.diag(matrix, k=-1))
        other1 = dataset_size-taikaku1#normal
        other2 = dataset_size-taikaku1-taikaku2 #1 neighbor

        return (taikaku1+taikaku2)/dataset_size,other1,other2


    def total_acc(matrix,dataset_size):
        opt = get_args()
        taikaku1 = sum(np.diag(matrix)) #対角成分
        taikaku2 = sum(np.diag(matrix, k=1)) + sum(np.diag(matrix, k=-1)) #対角成分の両サイド
        other1 = dataset_size-taikaku1#normal
        other2 = dataset_size-taikaku1-taikaku2 #1 neighbor


        tgt = f'{CFG.results_path}/confusion_matrix/{opt.model}'
        os.makedirs(tgt, exist_ok=True)

        np.savetxt(tgt + f"/{opt.sign}_{opt.num_classes}class_"
                   f"{opt.fold}fold_{opt.epoch}epoch_confusion_matrix.txt", matrix, fmt="%.0f")
        
        # with open(tgt + f"/{opt.sign}_{opt.num_classes}class_"
        #           f"{opt.fold}fold_{opt.epoch}epoch_confusion_matrix.txt","w") as f:
        #     f.write(matrix)
        # print('dataset_size', dataset_size)
        # print('taikaku1, taikaku2, other1, other2',taikaku1, taikaku2, other1, other2)
        # print('--> Saved Total Confusion Matrix txt')

        #print(f"normal_acc:{taikaku1/dataset_size}, 1neighbor_acc:{(taikaku1+taikaku2)/dataset_size}")
        #バグの温床
        return taikaku1/dataset_size, (taikaku1+taikaku2)/dataset_size,\
               other1, other2


    def total_report(y_pred, y_true):
        opt = get_args()
        cls_repo = classification_report(y_true,y_pred)

        tgt2 = f'{CFG.results_path}/outputs/{opt.model}'
        os.makedirs(tgt2, exist_ok=True)

        # np.savetxt(tgt2 + f"/{opt.sign}_{opt.num_classes}class_"
        #            f"{opt.fold}fold_{opt.epoch}epoch_class_report.txt", cls_repo)

        with open(tgt2 + f"/{opt.sign}_{opt.num_classes}class_"
                  f"{opt.fold}fold_{opt.epoch}epoch_class_report.txt","wb") as f:
            pickle.dump(cls_repo, f)

        print('--> Saved Classification Report')

        return cls_repo


    def index_multi(pred_list, num):
        return [i for i, _num in enumerate(pred_list) if _num == num]


    def criterion(y_pred, y_true):
        return nn.CrossEntropyLoss(y_pred,y_true)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
