import gc
from tqdm import tqdm
tqdm.pandas()

import torch
import torch.nn as nn

from utils.Parser import get_args
from utils.Configuration import CFG
from evaluation.EvaluationHelper import EvaluationHelper

def train_one_epoch(model,
                    optimizer,
                    scheduler,
                    criterion,
                    dataloader):

    opt = get_args()

    model.train()
    grad_scaler = torch.cuda.amp.GradScaler()
    
    running_loss = 0.0
    running_acc = 0.0
    running_f1 = 0.0
    
    pbar = tqdm(enumerate(dataloader, start=1),
                total=len(dataloader),
                desc='Train')
    for step, (inputs, labels) in pbar:         
        inputs = inputs.to(CFG.device)
        labels  = labels.to(CFG.device)
        
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            acc = EvaluationHelper.accuracy(CFG.to_numpy(preds), CFG.to_numpy(labels))
            f_m = EvaluationHelper.f_measure(CFG.to_numpy(preds), CFG.to_numpy(labels))
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_acc += acc
            running_f1 += f_m

            train_loss = running_loss / step
            train_acc = running_acc / step
            train_f1 = running_f1 / step

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        scheduler.step()

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{train_loss:0.6f}',
                         train_acc=f'{train_acc:0.6f}',
                         train_f1=f'{train_f1:0.6f}',
                         lr=f'{current_lr:0.6f}',
                         gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
            
        gc.collect()
    
    return train_loss, train_acc, train_f1


def valid_one_epoch(model,
                    optimizer,
                    criterion,
                    dataloader, 
                    id_list,
                    id_list2,
                    tmp_acc,
                    epoch,
                    clsrepo_labels,
                    clsrepo_preds,
                    id_for_cam):

    opt = get_args()

    #model.train()# turn ON dropout
    model.eval()# turn OFF dropout

    with torch.no_grad():
        
        dataset_size = 0
        running_loss = 0.0
        running_acc = 0.0
        running_f1 = 0.0

        c_mat = 0
        
        pbar = tqdm(enumerate(dataloader, start=1),
                    total=len(dataloader),
                    desc='Valid')
        for step, (inputs, labels, image_path) in pbar:        
            inputs  = inputs.to(CFG.device)
            labels  = labels.to(CFG.device)

            batch_size = inputs.size(0)

            outputs2  = model(inputs)

            n = nn.Softmax(dim=1)
            outputs = n(outputs2)
            _, preds = torch.max(outputs, 1)

            acc = EvaluationHelper.accuracy(CFG.to_numpy(preds),
                                            CFG.to_numpy(labels))
            f_m = EvaluationHelper.f_measure(CFG.to_numpy(preds),
                                             CFG.to_numpy(labels))
            cm  = EvaluationHelper.conf_mtrx(CFG.to_numpy(preds),
                                             CFG.to_numpy(labels))
            loss = criterion(outputs2,labels)
            
            #log miss classes
            diff = preds - labels #calculate diff

            if epoch == opt.epoch:
                id_for_cam[0].extend(image_path)
                id_for_cam[1].extend(CFG.to_numpy(labels))
                id_for_cam[2].extend(CFG.to_numpy(preds))

                clsrepo_labels += labels.tolist()
                clsrepo_preds += preds.tolist()

                CFG.probability.extend(CFG.to_numpy(_).numpy())
                CFG.difference.extend(CFG.to_numpy(diff).numpy())
                CFG.path_list.extend(image_path)

            running_loss += loss.item()
            running_acc += acc
            running_f1 += f_m
            dataset_size += batch_size
            c_mat += cm

            valid_loss = running_loss / step
            valid_acc = running_acc / step
            valid_f1 = running_f1 / step

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(valid_loss=f'{valid_loss:0.6f}',
                             valid_acc=f'{valid_acc:0.6f}',
                             valid_f1=f'{valid_f1:0.6f}',
                             lr=f'{current_lr:0.6f}',
                             gpu_memory=f'{mem:0.2f} GB')
            torch.cuda.empty_cache()
            
            #for visualize outliers
            for i in range(-opt.num_classes+1, opt.num_classes):
                if i in [0]: #remove nomal
                    pass
                else:
                    tmp = EvaluationHelper.index_multi(diff, i)

                    for j in range(len(tmp)):
                        id_list[i+opt.num_classes-1].append(image_path[tmp[j]])

            for i in range(-opt.num_classes+1,opt.num_classes):
                if i in [-1,0,1]: #remove nomal, 1 neighbor
                    pass
                else:
                    tmp = EvaluationHelper.index_multi(diff, i)

                    for j in range(len(tmp)):
                        id_list2[i+opt.num_classes-1].append(image_path[tmp[j]])

            gc.collect()
        
    return valid_loss, valid_acc, valid_f1, c_mat, dataset_size,\
           id_list, id_list2, tmp_acc, clsrepo_preds, clsrepo_labels, id_for_cam