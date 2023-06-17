import os
import random

import torch
import numpy as np

class Configuration(object):

    key = ''

    difference = []
    probability = []
    path_list = []
    fold_id = []
    true = []
    var = []
    mean = []
    total_predict = []
    total_correct = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    labels_dict = {}

    labels = []

    labels_name = []

    labels_index = []

    @staticmethod 
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu()

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        print('> SEEDING DONE')
