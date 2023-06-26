from dataclasses import dataclass
from typing import ClassVar
import numpy as np

import hydra
from omegaconf import OmegaConf, DictConfig

from utils.configuration_helper import ConfigurationHelper
from evaluation.evaluation_helper import EvaluationHelper

@dataclass
class SummarizeHelper(object):

    @classmethod
    def calculate_value(cls, cfg):

        if cfg.models.model.num_classes == 1:
            # Summarize predicition
            prediction_summary = np.array(ConfigurationHelper.predicts_array_float)

            # Mean, Variance of predicition
            prediction_mean = np.mean(prediction_summary, axis=0)
            prediction_variance = np.var(prediction_summary, axis=0)

            # Rounded predicition
            rounded_prediction = EvaluationHelper.threshold_config_for_inf(prediction_mean)
            
            # Difference between Rounded prediction and Ground-truth
            difference = np.array(rounded_prediction) - np.array(ConfigurationHelper.ground_truth)
            
            return [prediction_mean, prediction_variance, difference, rounded_prediction]

        else:
            # Summarize predicition
            prediction_summary = np.array(ConfigurationHelper.predicts_array)

            # Mean, Variance of predicition
            prediction_mean = np.mean(prediction_summary, axis=0)
            prediction_variance = np.var(prediction_summary, axis=0)

            for i in range(prediction_mean.shape[0]):

                prediction = np.argmax(prediction_mean[i])
                ConfigurationHelper.prediction_variance.append(prediction_variance[i][prediction])
                ConfigurationHelper.prediction_mean.append(prediction_mean[i][prediction])
                
                # Difference between Rounded prediction and Ground-truth
                ConfigurationHelper.difference.append(prediction - ConfigurationHelper.ground_truth[i])
                ConfigurationHelper.total_predict.append(prediction)

            return [0]