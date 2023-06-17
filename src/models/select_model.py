import torch
import torch.nn as nn
from torchvision import models
from utils.argparser import get_args
from utils.configuration import Configuration

# Model
def select_model(model_name: str):
    opt = get_args()

    if model_name == 'VisionTransformer_Base16':
        model = models.vit_b_16()
        
        # Load pretrained weight
        if opt.pretrained:
            try:
                model.load_state_dict(
                    torch.load(
                        f'{opt.base_path}/src/models/weight/vit_b_16-c867db91.pth'
                    )
                )
            except:
                print("Weight file Not Found !!")
        
        # Set ratio of Dropout
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.1
        
        model.heads = nn.Sequential(nn.Linear(in_features=768,
                                              out_features=384,
                                              bias=True),
                                    nn.Linear(in_features=384,
                                              out_features=opt.num_classes))
        
        model = model.to(Configuration.device)


    elif model_name == 'VisionTransformer_Base32':
        model = models.vit_b_32()
        model.heads = nn.Sequential(nn.Linear(in_features=768,
                                              out_features=384,
                                              bias=True),
                                    nn.Linear(in_features=384,
                                              out_features=opt.num_classes))
        model = model.to(Configuration.device)

    elif model_name == 'VisionTransformer_Large16':
        model = models.vit_l_16()
        model.heads = nn.Sequential(nn.Linear(in_features=1024,
                                              out_features=512,
                                              bias=True),
                                    nn.Linear(in_features=512,
                                              out_features=256,
                                              bias=True),
                                    nn.Linear(in_features=256,
                                              out_features=opt.num_classes))
        model = model.to(Configuration.device)

    elif model_name == 'VisionTransformer_Large32':
        model = models.vit_l_32()
        model.heads = nn.Sequential(nn.Linear(in_features=1024,
                                              out_features=512,
                                              bias=True),
                                    nn.Linear(in_features=512,
                                              out_features=256,
                                              bias=True),
                                    nn.Linear(in_features=256,
                                              out_features=opt.num_classes))
        model = model.to(Configuration.device)
        
        

    elif model_name == 'DenseNet161':
        model = models.densenet161(pretrained=False)
        
        # Load pretrained weight
        if opt.pretrained:
            try:
                model.load_state_dict(
                    torch.load(
                        f'{opt.base_path}/src/models/weight/densenet161-8d451a50.pth'
                    )
                )
            except:
                print("Weight file Not Found !!")

    # Add Dropout 
        for i in range(12):
            if i>=4 and i<=10:
                a = len(list(model.children())[0][i])
                
                for j in range(1,a+1):
                    if a == 4:
                        list(model.children())[0][i].add_module('Dropout', nn.Dropout(p=0.2))

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
                                                   out_features=opt.num_classes))
        model = model.to(Configuration.device)


    elif model_name == 'VGG16':
        model = models.vgg16(pretrained=False)
        
        # Load pretrained weight
        if opt.pretrained:
            try:
                model.load_state_dict(
                    torch.load(
                        f'{opt.base_path}/src/models/weight/vgg16-397923af.pth'
                    )
                )
            except:
                print("Weight file Not Found !!")

        modules = []
        for i,m in enumerate(list(model.children())[0]):
            if i in (4,9,16,30):
                modules.append(nn.Dropout(p=0.1, inplace=False))
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
                                                      out_features=opt.num_classes))
        for m in model.classifier:
            if isinstance(m, nn.Dropout):
                m.p = 0.1

        model = model.to(Configuration.device)

    return model