import os
import re

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
from torchvision import models 
from utils.configuration_helper import ConfigurationHelper

# For DenseNet161
def _load_state_dict(model, weights):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = weights
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    model.load_state_dict(state_dict)

# Model
def select_model(cfg):

    if cfg.models.model.name == 'VisionTransformer_Base16':
        model = models.vit_b_16(weights=None)
        
        # Load pretrained weight
        if cfg.models.model.pretrained:
            try:
                model.load_state_dict(
                    torch.load(
                        f'{cfg.models.path.base}/src/models/weight/vit_b_16-c867db91.pth'
                    )
                )
      
            except Exception as e:
                print(e, "- Weight file Not Found !!")
        else:
            print("- No Pretrained Weight (Training Mode or Inference Mode)")
        
        # Set the ratio of Dropout
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = cfg.models.model.dropout_ratio
        
        model.heads = nn.Sequential(nn.Linear(in_features=768,
                                              out_features=384,
                                              bias=True),
                                    nn.Linear(in_features=384,
                                              out_features=cfg.models.model.num_classes))
        
        model = model.to(ConfigurationHelper.device)

    elif cfg.models.model.name == 'VisionTransformer_Base32':
        model = models.vit_b_32()
        model.heads = nn.Sequential(nn.Linear(in_features=768,
                                              out_features=384,
                                              bias=True),
                                    nn.Linear(in_features=384,
                                              out_features=cfg.models.model.num_classes))
        model = model.to(ConfigurationHelper.device)

    elif cfg.models.model.name == 'VisionTransformer_Large16':
        model = models.vit_l_16()
        model.heads = nn.Sequential(nn.Linear(in_features=1024,
                                              out_features=512,
                                              bias=True),
                                    nn.Linear(in_features=512,
                                              out_features=256,
                                              bias=True),
                                    nn.Linear(in_features=256,
                                              out_features=cfg.models.model.num_classes))
        model = model.to(ConfigurationHelper.device)

    elif cfg.models.model.name == 'VisionTransformer_Large32':
        model = models.vit_l_32()
        model.heads = nn.Sequential(nn.Linear(in_features=1024,
                                              out_features=512,
                                              bias=True),
                                    nn.Linear(in_features=512,
                                              out_features=256,
                                              bias=True),
                                    nn.Linear(in_features=256,
                                              out_features=cfg.models.model.num_classes))
        model = model.to(ConfigurationHelper.device)
        
        

    elif cfg.models.model.name == 'DenseNet161':
        model = models.densenet161(weights=None)
        
        # Load pretrained weight
        if cfg.models.model.pretrained:
            try:
                _load_state_dict(
                    model = model,
                    weights = torch.load(f'{cfg.models.path.base}/src/models/weight/densenet161-8d451a50.pth')
                )
            except Exception as e:
                print(e, "- Weight file Not Found !!")
        else:
            print("- No Pretrained Weight (Training Mode or Inference Mode)")

    # Add Dropout 
        for i in range(12):
            if i>=4 and i<=10:
                a = len(list(model.children())[0][i])
                
                if a == 4:
                    list(model.children())[0][i].add_module('Dropout',
                                                            nn.Dropout(p=cfg.models.model.dropout_ratio))

        model.classifier = nn.Sequential(nn.Linear(in_features=2208,
                                                   out_features=1024,
                                                   bias=True),
                                         nn.Linear(in_features=1024,
                                                   out_features=512,
                                                   bias=True),
                                         nn.Linear(in_features=512,
                                                   out_features=256,
                                                   bias=True),
                                         nn.Linear(in_features=256,
                                                   out_features=cfg.models.model.num_classes))
        model = model.to(ConfigurationHelper.device)

    elif cfg.models.model.name == 'VGG16':
        model = models.vgg16(weights=None)
        
        # Load pretrained weight
        if cfg.models.model.pretrained:
            try:
                model.load_state_dict(
                    torch.load(
                        f'{cfg.models.path.base}/src/models/weight/vgg16-397923af.pth'
                    )
                )
            except:
                print("- Weight file Not Found !!")
        else:
            print("- No Pretrained Weight (Training Mode or Inference Mode)")

        modules = []
        for i,m in enumerate(list(model.children())[0]):
            if i in (4,9,16,30):
                modules.append(nn.Dropout(p=cfg.models.model.dropout_ratio, inplace=False))
            modules.append(m)

        model.features = nn.Sequential(*modules)
        model.classifier[6] = nn.Sequential(nn.Linear(in_features=4096,
                                                      out_features=2048,
                                                      bias=True),
                                            nn.Linear(in_features=2048,
                                                      out_features=1024,
                                                      bias=True),
                                            nn.Linear(in_features=1024,
                                                      out_features=512,
                                                      bias=True),
                                            nn.Linear(in_features=512,
                                                      out_features=256,
                                                      bias=True),
                                            nn.Linear(in_features=256,
                                                      out_features=cfg.models.model.num_classes))
        for m in model.classifier:
            if isinstance(m, nn.Dropout):
                m.p = cfg.models.model.dropout_ratio2

        model = model.to(ConfigurationHelper.device)

    return model