import os
import pandas as pd
from typing import ClassVar
from dataclasses import dataclass, field

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.configuration_helper import ConfigurationHelper

@dataclass
class VisualizeHelper(object):

    result: ClassVar[list] = [[] for _ in range(8)]

    @classmethod
    def create_result_csv(cls, cfg, *args):
        if cfg.models.model.num_classes == 1:
            for i in range(len(ConfigurationHelper.ground_truth)):
                if args[2][i]==0:
                    cls.result[0].append(ConfigurationHelper.path_list[i])
                    cls.result[1].append(cfg.models.model.name)
                    cls.result[2].append('Exact')
                    cls.result[3].append(ConfigurationHelper.fold_id[i])
                    cls.result[4].append(args[3][i])
                    cls.result[5].append(args[0][i])
                    cls.result[6].append(args[1][i])
                    cls.result[7].append(ConfigurationHelper.ground_truth[i])
            
                elif abs(args[2][i])==1:
                    cls.result[0].append(ConfigurationHelper.path_list[i])
                    cls.result[1].append(cfg.models.model.name)
                    cls.result[2].append('One-Neighbor')
                    cls.result[3].append(ConfigurationHelper.fold_id[i])
                    cls.result[4].append(args[3][i])
                    cls.result[5].append(args[0][i])
                    cls.result[6].append(args[1][i])
                    cls.result[7].append(ConfigurationHelper.ground_truth[i])
            
                elif abs(args[2][i])>=2:
                    cls.result[0].append(ConfigurationHelper.path_list[i])
                    cls.result[1].append(cfg.models.model.name)
                    cls.result[2].append('Otherwise')
                    cls.result[3].append(ConfigurationHelper.fold_id[i])
                    cls.result[4].append(args[3][i])
                    cls.result[5].append(args[0][i])
                    cls.result[6].append(args[1][i])
                    cls.result[7].append(ConfigurationHelper.ground_truth[i])

            list_row = pd.DataFrame(cls.result)
            list_row = list_row.transpose()
            list_row.columns = ['PATH',
                                "MODEL",
                                "ACCURACY",
                                "FOLD",
                                "OUTPUTS",
                                "MEAN",
                                "VARIANCE",
                                "GROUND_TRUTH"]
            print(list_row)

        else:
            for i in range(len(ConfigurationHelper.ground_truth)):
                if ConfigurationHelper.difference[i]==0:
                    cls.result[0].append(ConfigurationHelper.path_list[i])
                    cls.result[1].append(cfg.models.model.name)
                    cls.result[2].append('Exact')
                    cls.result[3].append(ConfigurationHelper.fold_id[i])
                    cls.result[4].append(ConfigurationHelper.prediction_mean[i])
                    cls.result[5].append(ConfigurationHelper.prediction_variance[i])
                    cls.result[6].append(ConfigurationHelper.ground_truth[i])
                    cls.result[7].append(ConfigurationHelper.total_predict[i])
            
                elif abs(ConfigurationHelper.difference[i])==1:
                    cls.result[0].append(ConfigurationHelper.path_list[i])
                    cls.result[1].append(cfg.models.model.name)
                    cls.result[2].append('One-Neighbor')
                    cls.result[3].append(ConfigurationHelper.fold_id[i])
                    cls.result[4].append(ConfigurationHelper.prediction_mean[i])
                    cls.result[5].append(ConfigurationHelper.prediction_variance[i])
                    cls.result[6].append(ConfigurationHelper.ground_truth[i])
                    cls.result[7].append(ConfigurationHelper.total_predict[i])
            
                elif abs(ConfigurationHelper.difference[i])>=2:
                    cls.result[0].append(ConfigurationHelper.path_list[i])
                    cls.result[1].append(cfg.models.model.name)
                    cls.result[2].append('Otherwise')
                    cls.result[3].append(ConfigurationHelper.fold_id[i])
                    cls.result[4].append(ConfigurationHelper.prediction_mean[i])
                    cls.result[5].append(ConfigurationHelper.prediction_variance[i])
                    cls.result[6].append(ConfigurationHelper.ground_truth[i])
                    cls.result[7].append(ConfigurationHelper.total_predict[i])

            list_row = pd.DataFrame(cls.result)
            list_row = list_row.transpose()
            list_row.columns = ['PATH',
                                "MODEL",
                                "ACCURACY",
                                "FOLD",
                                "MEAN",
                                "VARIANCE",
                                "GROUND_TRUTH",
                                "OUTPUTS"]
            print(list_row)

        # Regression        
        if cfg.models.model.num_classes == 1:
            tgt = f'{cfg.models.path.result}/{cfg.models.sign}/Regression/csv/{cfg.models.model.name}'
            os.makedirs(tgt, exist_ok=True)

        #Classification
        else:
            tgt = f'{cfg.models.path.result}/{cfg.models.sign}/Classification/csv/{cfg.models.model.name}'
            os.makedirs(tgt, exist_ok=True)

        list_row.to_csv(
            tgt + 
            f'/{cfg.models.sign}_datalist{cfg.models.general.datalist}_'
            f'{cfg.models.train.epoch}epoch_iter{cfg.models.inference.num_sampling}.csv',
            index=False
        )