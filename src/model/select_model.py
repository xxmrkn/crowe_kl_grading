import torch
import torch.nn as nn
from torchvision import models
from model.coatnet import coatnet_0,coatnet_1,coatnet_2,coatnet_3,coatnet_4
from utils.Parser import get_args
from utils.Configuration import CFG

#model
def choose_model(model_name):
    opt = get_args()

    if model_name == 'CoAtNet_0':
        model = coatnet_0()
        model = model.to(CFG.device)

    elif model_name == 'CoAtNet_1':
        model = coatnet_1()
        model = model.to(CFG.device)

    elif model_name == 'CoAtNet_2':
        model = coatnet_2()
        model = model.to(CFG.device)

    elif model_name == 'CoAtNet_3':
        model = coatnet_3()
        model = model.to(CFG.device)

    # elif model_name == 'CoAtNet_4':
    #     model = coatnet_4()
    #     model = model.to(CFG.device)
    
    # Use Pretrained Weight
    elif model_name == 'VisionTransformer_Base16':
        model = models.vit_b_16(pretrained=True)
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.1

        model.heads = nn.Sequential(nn.Linear(in_features=768,
                                              out_features=opt.num_classes))
        
        model = model.to(CFG.device)

    elif model_name == 'VisionTransformer_Base32':
        model = models.vit_b_32(pretrained=True)
        model.heads = nn.Sequential(nn.Linear(in_features=768,
                                              out_features=CFG.num_classes))
        model = model.to(CFG.device)

    elif model_name == 'VisionTransformer_Large16':
        model = models.vit_l_16(pretrained=True)
        model.heads = nn.Sequential(nn.Linear(in_features=1024,
                                              out_features=CFG.num_classes))
        model = model.to(CFG.device)

    elif model_name == 'VisionTransformer_Large32':
        model = models.vit_l_32(pretrained=True)
        model.heads = nn.Sequential(nn.Linear(in_features=1024,
                                              out_features=CFG.num_classes))
        model = model.to(CFG.device)

    elif model_name == 'DenseNet161':
        model = models.densenet161(pretrained=True)

    # Add Dropout 
        for i in range(12):
            if i>=4 and i<=10:
                a = len(list(model.children())[0][i])
                
                for j in range(1,a+1):
                    if a == 4:
                        list(model.children())[0][i].add_module('Dropout', nn.Dropout(p=0.2))

        model.classifier = nn.Sequential(nn.Linear(in_features=2208,
                                                   out_features=opt.num_classes,
                                                   bias=True))
        model = model.to(CFG.device)


    elif model_name == 'VGG16':
        model = models.vgg16(pretrained=True)

        modules = []
        for i,m in enumerate(list(model.children())[0]):
            #print(i,m)
            if i in (4,9,16,30):
                modules.append(nn.Dropout(p=0.3, inplace=False))
            modules.append(m)

        model.features = nn.Sequential(*modules)
        model.classifier[6] = nn.Linear(in_features=4096,
                                        out_features=opt.num_classes)
        for m in model.classifier:
            if isinstance(m, nn.Dropout):
                m.p = 0.3

        model = model.to(CFG.device)

    elif model_name == 'Inceptionv3':
        model = models.inception_v3(pretrained=True)
        model.fc = nn.Linear(in_features=2048,
                             out_features=CFG.num_classes)
        model = model.to(CFG.device)

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier = nn.Linear(in_features=1280,
                                     out_features=CFG.num_classes)
        model = model.to(CFG.device)

    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(pretrained=True)
        model.classifier = nn.Linear(in_features=1536,
                                     out_features=CFG.num_classes)
        model = model.to(CFG.device)

    elif model_name == 'efficientnet_b7':
        model = models.efficientnet_b7(pretrained=True)
        model.classifier = nn.Linear(in_features=2560,
                                     out_features=CFG.num_classes)
        model = model.to(CFG.device)
    return model