import os
import random

import torch
import numpy as np

from utils.Parser import get_args

class CFG:
    opt = get_args()
    
    base_path = ''
    dataset_path2 = ''
    dataset_path = ''
    weights_path = f''
    datalist_path = f''
    image_path = dataset_path2 + ''
    csv_path = dataset_path + '' 
    df_path = dataset_path + ''
    results_path = f''

    key = ''

    scheduler = 'CosineAnnealinglr'

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


    if opt.num_classes == 7:
        labels_dict = {}

        labels = [0,1,2,3,4,5,6]

        labels_name = []

        labels_index = []
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu()

    def set_seed(seed = opt.seed):
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
