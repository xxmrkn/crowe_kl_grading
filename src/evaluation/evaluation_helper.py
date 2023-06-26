import os
import pickle
import numpy as np
from typing import ClassVar, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import hydra
from omegaconf import OmegaConf, DictConfig

from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

from utils.configuration_helper import ConfigurationHelper

@dataclass
class EvaluationHelper(object):

    y_true: List[float] = field(default_factory=list)
    y_pred: List[float] = field(default_factory=list)

    def f_measure(self):
        return f1_score(self.y_true, self.y_pred, average="macro")
    
    def acc(self):
        return accuracy_score(self.y_true, self.y_pred)
         
    def ans(self):
        return np.mean(np.abs(self.y_pred - self.y_true) < 0.3)
    
    def conf_mtrx(self):
        return confusion_matrix(self.y_true,
                                self.y_pred,
                                labels = ConfigurationHelper.labels)

    # @staticmethod
    # def one_mistake_acc(matrix, dataset_size):
    #     taikaku1 = sum(np.diag(matrix))
    #     taikaku2 = sum(np.diag(matrix, k=1)) + sum(np.diag(matrix, k=-1))
    #     other1 = dataset_size - taikaku1 # Exact Class
    #     other2 = dataset_size - taikaku1-taikaku2 # One-Neighbor

    #     return (taikaku1 + taikaku2) / dataset_size, other1, other2

    # @classmethod
    # def total_acc(cfg, matrix, dataset_size):
    #     taikaku1 = sum(np.diag(matrix))
    #     taikaku2 = sum(np.diag(matrix, k=1)) + sum(np.diag(matrix, k=-1))
    #     other1 = dataset_size-taikaku1 # Exact Class
    #     other2 = dataset_size-taikaku1-taikaku2 # One-Neighbor 

    #     tgt = f'{cfg.models.path.result}/{cfg.models.sign}/confusion_matrix/{cfg.models.model.name}'
    #     os.makedirs(tgt, exist_ok=True)

    #     np.savetxt(tgt + f"/{cfg.models.sign}_{cfg.models.model.num_classes}class_"
    #                f"{cfg.models.general.fold}fold_{cfg.models.train.epoch}epoch_confusion_matrix.txt", matrix, fmt="%.0f")
        
    #     return taikaku1/dataset_size, (taikaku1 + taikaku2) / dataset_size,

    # def total_report(self):
    #     cls_repo = classification_report(self.y_true, self.y_pred)

    #     tgt2 = f'{ConfigurationHelper.results_path}/outputs/{EvaluationHelper.cfg.model.name}'
    #     os.makedirs(tgt2, exist_ok=True)

    #     with open(tgt2 + f"/{EvaluationHelper.cfg.sign}_{EvaluationHelper.cfg.model.num_classes}class_"
    #               f"{EvaluationHelper.cfg.general.fold}fold_{EvaluationHelper.cfg.train.epoch}epoch_class_report.txt","wb") as f:
    #         pickle.dump(cls_repo, f)

    #     print('--> Saved Classification Report')

    #     return cls_repo

    # @staticmethod
    # def index_multi(pred_list, num):
    #     return [i for i, _num in enumerate(pred_list) if _num == num]

    # @staticmethod
    # def threshold_config(pred_value):
    #     pred_value = pred_value.tolist()
        
    #     """
    #     pred_value.shape : (Batchsize, 1)
        
    #     """
       
    #     for i in range(len(pred_value)):
      
    #         """
    #         class label 0 : pred_value<=0.5
    #         class label 1 : pred_value>0.5 pred_value<=1.5
    #         class label 2 : pred_value>1.5 pred_value<=2.5
    #         class label 3 : pred_value>2.5 pred_value<=3.5
    #         class label 4 : pred_value>3.5 pred_value<=4.5
    #         class label 5 : pred_value>4.5 pred_value<=5.5
    #         class label 6 : pred_value>5.5 

    #         """
    #         if pred_value[i][0]<=0.5:
    #             pred_value[i][0] = 0

    #         elif pred_value[i][0]>0.5 and pred_value[i][0]<=1.5:
    #             pred_value[i][0] = 1

    #         elif pred_value[i][0]>1.5 and pred_value[i][0]<=2.5:
    #             pred_value[i][0] = 2

    #         elif pred_value[i][0]>2.5 and pred_value[i][0]<=3.5:
    #             pred_value[i][0] = 3

    #         elif pred_value[i][0]>3.5 and pred_value[i][0]<=4.5:
    #             pred_value[i][0] = 4

    #         elif pred_value[i][0]>4.5 and pred_value[i][0]<=5.5:
    #             pred_value[i][0] = 5

    #         else:
    #             pred_value[i][0] = 6           

    #     return pred_value

    @staticmethod
    def threshold_config_for_inf(pred_value):
        pred_value = pred_value.tolist()
        
        """
        pred_value.shape : (Batchsize, 1)
        
        """
      
        for i in range(len(pred_value)):
            """
            class label 0 : pred_value<=0.5
            class label 1 : pred_value>0.5 pred_value<=1.5
            class label 2 : pred_value>1.5 pred_value<=2.5
            class label 3 : pred_value>2.5 pred_value<=3.5
            class label 4 : pred_value>3.5 pred_value<=4.5
            class label 5 : pred_value>4.5 pred_value<=5.5
            class label 6 : pred_value>5.5 

            """
            if pred_value[i]<=0.5:
                pred_value[i] = 0

            elif pred_value[i]>0.5 and pred_value[i]<=1.5:
                pred_value[i] = 1

            elif pred_value[i]>1.5 and pred_value[i]<=2.5:
                pred_value[i] = 2

            elif pred_value[i]>2.5 and pred_value[i]<=3.5:
                pred_value[i] = 3

            elif pred_value[i]>3.5 and pred_value[i]<=4.5:
                pred_value[i] = 4

            elif pred_value[i]>4.5 and pred_value[i]<=5.5:
                pred_value[i] = 5

            else:
                pred_value[i] = 6           

        return pred_value

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
            input = input.view(input.size(0),input.size(1),-1) # N,C,H,W => N,C,H*W
            input = input.transpose(1,2) # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2)) # N,H*W,C => N*H*W,C
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
 