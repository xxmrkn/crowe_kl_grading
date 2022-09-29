import copy
import csv
import gc
from glob import glob
import os
import pathlib
import time

import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from utils.Parser import get_args
from utils.Configuration import CFG
from visualization.VisualizeHelper import visualize_total_image
from dataset.dataset import TrainDataset,TestDataset,get_transforms
from model.select_model import choose_model
from evaluation.EvaluationHelper import EvaluationHelper, FocalLoss
from function.compare_acc import compare, compare2
from trainval_one_epoch import train_one_epoch, valid_one_epoch

import wandb


def main():

    #preparation training
    opt = get_args()
    CFG.set_seed(opt.seed)

    if torch.cuda.is_available():
        print(f"cuda: {torch.cuda.get_device_name}")
    
    data_df = pd.read_csv(CFG.df_path)

    #manage filename
    file_names = []

    p = pathlib.Path(f'../datalist{opt.datalist}/k{opt.fold}').glob('train*.txt')
    for i in p:
        file_names.append(f'k{opt.fold}/'+i.name)

    p = pathlib.Path(f'../datalist{opt.datalist}/k{opt.fold}').glob('valid*.txt')
    for i in p:
        file_names.append(f'k{opt.fold}/'+i.name)
    print(file_names)

    name = []
    for j in range(len(file_names)):
        for i in range(int(opt.fold)):
            if str(i) in file_names[j]:
                name.append(os.path.join(CFG.datalist_path+f'/datalist{opt.datalist}', file_names[j]))

    #run training each fold
    total_confusion_matrix = 0
    total_path = []
    total_id = []
    total_actual = []
    total_pred = []

    total_path2 = []
    total_id2 = []
    total_actual2 = []
    total_pred2 = []

    class_report_labels = []
    class_report_preds = []

    for fold in [i for i in range(int(opt.fold))]:
        print(f'#'*15)
        print(f'### Fold: {fold+1}')
        print(f'#'*15)

        #prepare dataframe for each fold
        with open(name[fold]) as f:
            line = f.read().splitlines()
        with open(name[fold+int(opt.fold)]) as f:
            line2 = f.read().splitlines()

        train_df = data_df[data_df['UID'].isin(line)]
        valid_df = data_df[data_df['UID'].isin(line2)]

        train_dataset = TrainDataset(train_df,
                                     transform = get_transforms('train'))
        valid_dataset = TestDataset(valid_df,
                                    transform = get_transforms('valid'))

        train_loader = DataLoader(train_dataset,
                                  batch_size = opt.batch_size, 
                                  num_workers = opt.num_workers,
                                  shuffle = True,
                                  pin_memory = True,
                                  drop_last = False)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size = opt.valid_batch_size, 
                                  num_workers = opt.num_workers,
                                  shuffle = False,
                                  pin_memory = True)

        #create model
        model = choose_model(opt.model)

        #Metrics
        criterion = FocalLoss(gamma=2)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr = opt.lr,
                                     weight_decay = opt.wd)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max = opt.t_max,
                                                   eta_min = opt.min_lr)

        #wandb
        wandb.login(key=CFG.key)
        run = wandb.init(project = f'', 
                         config = {"model_name": opt.model,
                                   "learning_rate": opt.lr,
                                   "fold": opt.fold,
                                   "epochs": opt.epoch,
                                   "image_size": opt.image_size,
                                   "batch_size": opt.batch_size,
                                   "num_workers": opt.num_workers,
                                   "num_classes": opt.num_classes,
                                   "optimizer": opt.optimizer,
                                   "loss": opt.criterion,
                                   "sign": opt.sign},
                         entity = "xxmrkn",
                         name = f"{opt.sign}|{opt.num_classes}class|{opt.model}|{opt.fold}fold"
                                f"|fold-{fold+1}|dim-{opt.image_size}**2|batch-{opt.batch_size}|lr-{opt.lr}")

        wandb.watch(model, log_freq=100)        


        #Training 
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())

        best_acc = 0.0
        best_acc2 = 0.0
        tmp_acc = 0.0
        best_epoch = -1
        best_epoch2 = -1

        id_for_cam = [[],[],[]]

        for epoch in range(opt.epoch):
            gc.collect()
            print(f'Epoch {epoch+1}/{opt.epoch}')
            print('-' * 10)

            id_list = [[] for _ in range(opt.num_classes*2-1)]
            id_list2 = [[] for _ in range(opt.num_classes*2-1)]

            path = []
            lab = []
            ac = []
            pre = []
            path2 = []
            lab2 = []
            ac2 = []
            pre2 = []

            train_loss, train_acc, train_f1 = train_one_epoch(model,
                                                              optimizer,
                                                              scheduler,
                                                              criterion,
                                                              train_loader)
            valid_loss, valid_acc, valid_f1,\
            cmatrix, dataset_size, id_list,\
            id_list2, tmp_acc, clsrepo_preds,\
            clsrepo_labels, id_for_cams = valid_one_epoch(model,
                                                          optimizer,
                                                          criterion,
                                                          valid_loader,
                                                          id_list,
                                                          id_list2,
                                                          tmp_acc, 
                                                          epoch+1,
                                                          class_report_preds,
                                                          class_report_labels,
                                                          id_for_cam)
            

            valid_acc2, others, others2 = EvaluationHelper.one_mistake_acc(cmatrix,
                                                                           dataset_size)

            cnt = [i  for i in range(-opt.num_classes+1, opt.num_classes)]


            # Log the metrics
            wandb.log({"Train Loss": train_loss, 
                       "Valid Loss": valid_loss,
                       "Train_Accuracy": train_acc,
                       "Valid Accuracy": valid_acc,
                       "Valid Accuracy2": valid_acc2,
                       "Train F-measure": train_f1,
                       "Valid F-measure": valid_f1,
                       "LR": scheduler.get_last_lr()[0]})
    

            #display results
            print('#'*50)
            print(f'######## now training...  fold : {fold+1}')
            print('#'*50)

            print(f"Train Loss: {train_loss}, Train Acc: {train_acc}, Train f1: {train_f1}")
            print(f"Valid Loss: {valid_loss}, Valid Acc: {valid_acc}, Valid Acc2: {valid_acc2}, Valid f1: {valid_f1}")

            print('#'*50)

            #If the score improved
            if valid_acc > best_acc:
                print(f"Valid Accuracy Improved ({best_acc:0.4f} ---> {valid_acc:0.4f})")

                best_acc = valid_acc
                best_epoch = epoch+1
                best_model_wts = copy.deepcopy(model.state_dict())

                print('save best epoch models!')

                run.summary["Best Accuracy"] = best_acc
                run.summary["Best Epoch"]   = best_epoch

            if valid_acc2 > best_acc2:
                print(f"Valid Accuracy2 Improved ({best_acc2:0.4f} ---> {valid_acc2:0.4f})")

                best_acc2 = valid_acc2
                best_epoch2 = epoch+1

                run.summary["Best Accuracy2"] = best_acc2
                run.summary["Best Epoch2"]   = best_epoch2
            
            path, lab, ac, pre, total_path,\
            total_id, total_actual, total_pred = compare(epoch,
                                                         id_list,
                                                         cnt,
                                                         data_df,
                                                         path,
                                                         lab,
                                                         ac,
                                                         pre,
                                                         total_path,
                                                         total_id,
                                                         total_actual,
                                                         total_pred,
                                                         others)

            path2, lab2, ac2, pre2, total_path2,\
            total_id2, total_actual2, total_pred2 = compare2(epoch,
                                                             id_list2,
                                                             cnt,
                                                             data_df,
                                                             path2,
                                                             lab2,
                                                             ac2,
                                                             pre2,
                                                             total_path2,
                                                             total_id2,
                                                             total_actual2,
                                                             total_pred2,
                                                             others2)

            #manage confusion matrix
            if epoch+1 == opt.epoch:
                total_confusion_matrix += cmatrix

            #calculate total accuracy
            if epoch+1 == opt.epoch and fold+1==int(opt.fold):
                total_dataset_size = len(valid_dataset)
                print(f'total_dataset_size {total_dataset_size}')
                print(f'#'*30)
                print(f'### Results of this experiment: {fold+1}')
                print(f'#'*30)

                normal_acc, neighbor_acc,\
                remain, remain2 = EvaluationHelper.total_acc(total_confusion_matrix,
                                                             total_dataset_size)
                
                #Visualize outliers
                flag=1
                visualize_total_image(total_path,
                                      total_id,
                                      total_actual,
                                      total_pred,
                                      remain,
                                      normal_acc,
                                      neighbor_acc,
                                      flag)
                #Visualize outliers2
                flag=2
                visualize_total_image(total_path2,
                                      total_id2,
                                      total_actual2,
                                      total_pred2,
                                      remain2,
                                      normal_acc,
                                      neighbor_acc,
                                      flag)
                
                #output classification report
                class_report = EvaluationHelper.total_report(clsrepo_preds, clsrepo_labels)

        #visualize gradCAM images
        if opt.model =='VGG16' or opt.model == 'DenseNet161':
            tgt = f'{CFG.results_path}/weights/{opt.model}'
            os.makedirs(tgt, exist_ok=True)

            torch.save(model.state_dict(),
                       tgt + f''
                       f'')
            
            id_for_cams = [list(x) for x in zip(*id_for_cams)]

            tgt2 = f''
            os.makedirs(tgt2, exist_ok=True)

            with open(tgt2 + f""
                      f"",'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(['ID','Actual','Pred'])
                writer.writerows(id_for_cams)

        else:
            tgt3 = f''
            os.makedirs(tgt3, exist_ok=True)

            torch.save(best_model_wts,
                       tgt3 + f''
                       f'')

        run.finish()

        #display wandb webpage link
        print(f"wandb website ------> {run.url}")

        #remove wandb files
        print(os.path.isdir('wandb'))
        #shutil.rmtree("wandb")

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed//3600}h {time_elapsed//60}m {time_elapsed%60:.2f}s')
        print(f'Best Epoch {best_epoch}, Best val Accuracy:) {best_acc:.4f},' 
              f'Best Epoch {best_epoch2}, Best val Accuracy2: {best_acc2:.4f}')

    return model


if __name__ == '__main__':
    main()
