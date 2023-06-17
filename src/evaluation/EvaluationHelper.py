import os
import pickle
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import f1_score,accuracy_score,classification_report
from sklearn.metrics import confusion_matrix

from utils.configuration import Configuration
from utils.argparser import get_args

class EvaluationHelper(object):

    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def f_measure(self):
        return f1_score(self.y_true,
                        self.y_pred,
                        average="macro")
    
    def acc(self):
        return accuracy_score(self.y_true, self.y_pred)
         
    def ans(self):
        return np.mean(np.abs(self.y_pred - self.y_true) < 0.3)
    
    def conf_mtrx(self):
        return confusion_matrix(self.y_true,
                                self.y_pred,
                                labels = Configuration.labels)

    @staticmethod
    def one_mistake_acc(matrix, dataset_size):
        taikaku1 = sum(np.diag(matrix))
        taikaku2 = sum(np.diag(matrix, k=1)) + sum(np.diag(matrix, k=-1))
        other1 = dataset_size - taikaku1 # Exact Class
        other2 = dataset_size - taikaku1-taikaku2 # One-Neighbor

        return (taikaku1 + taikaku2) / dataset_size, other1, other2

    @staticmethod
    def total_acc(matrix, dataset_size):
        opt = get_args()
        taikaku1 = sum(np.diag(matrix))
        taikaku2 = sum(np.diag(matrix, k=1)) + sum(np.diag(matrix, k=-1))
        other1 = dataset_size-taikaku1 # Exact Class
        other2 = dataset_size-taikaku1-taikaku2 # One-Neighbor 

        tgt = f'{opt.result_path}/{opt.sign}/confusion_matrix/{opt.model}'
        os.makedirs(tgt, exist_ok=True)

        np.savetxt(tgt + f"/{opt.sign}_{opt.num_classes}class_"
                   f"{opt.fold}fold_{opt.epoch}epoch_confusion_matrix.txt", matrix, fmt="%.0f")
        
        return taikaku1/dataset_size, (taikaku1 + taikaku2) / dataset_size,

    def total_report(self):
        opt = get_args()
        cls_repo = classification_report(self.y_true, self.y_pred)

        tgt2 = f'{Configuration.results_path}/outputs/{opt.model}'
        os.makedirs(tgt2, exist_ok=True)

        with open(tgt2 + f"/{opt.sign}_{opt.num_classes}class_"
                  f"{opt.fold}fold_{opt.epoch}epoch_class_report.txt","wb") as f:
            pickle.dump(cls_repo, f)

        print('--> Saved Classification Report')

        return cls_repo

    @staticmethod
    def index_multi(pred_list, num):
        return [i for i, _num in enumerate(pred_list) if _num == num]

    @staticmethod
    def threshold_config(pred_value):
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
            if pred_value[i][0]<=0.5:
                pred_value[i][0] = 0

            elif pred_value[i][0]>0.5 and pred_value[i][0]<=1.5:
                pred_value[i][0] = 1

            elif pred_value[i][0]>1.5 and pred_value[i][0]<=2.5:
                pred_value[i][0] = 2

            elif pred_value[i][0]>2.5 and pred_value[i][0]<=3.5:
                pred_value[i][0] = 3

            elif pred_value[i][0]>3.5 and pred_value[i][0]<=4.5:
                pred_value[i][0] = 4

            elif pred_value[i][0]>4.5 and pred_value[i][0]<=5.5:
                pred_value[i][0] = 5

            else:
                pred_value[i][0] = 6           

        return pred_value

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