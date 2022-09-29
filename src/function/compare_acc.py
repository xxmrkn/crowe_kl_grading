import re
from utils.Parser import get_args
from utils.Configuration import CFG

def compare(epoch,
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
            others):

    opt = get_args()

    if epoch+1==opt.epoch:
        for i,c in zip(id_list,cnt):
            flag = 1 
            if i in [0]:
                pass
            else:
                for j in range(len(i)):

                    label = re.findall('AP/(.*)',i[j])#extract ID

                    actual = data_df[data_df['ID'].str.contains(*label)]['target']
                    pred = actual+c

                    actual_label = CFG.labels_dict[actual.item()]
                    pred_label = CFG.labels_dict[pred.item()]
                        
                    path.append(i[j])
                    lab.append(*label)
                    ac.append(actual_label)
                    pre.append(pred_label)

                    if epoch+1==opt.epoch:
                        total_path.append(i[j])
                        total_id.append(*label)
                        total_actual.append(actual_label)
                        total_pred.append(pred_label)

        #visualize_image(path,lab,ac,pre,others,flag,fold+1,epoch+1)
    return path, lab, ac, pre, total_path, total_id, total_actual, total_pred

def compare2(epoch,
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
             others2):

    opt = get_args()

    if epoch+1==opt.epoch: 
        flag = 2
        #extract and visualize outliers
        for i,c in zip(id_list2,cnt):
            if i in [-1,0,1]:
                pass
            else:
                for j in range(len(i)):
                    label = re.findall('AP/(.*)',i[j])#extract ID

                    actual = data_df[data_df['ID'].str.contains(*label)]['target']
                    pred = actual+c

                    actual_label = CFG.labels_dict[actual.item()]
                    pred_label = CFG.labels_dict[pred.item()]

                    path2.append(i[j])
                    lab2.append(*label)
                    ac2.append(actual_label)
                    pre2.append(pred_label)

                    if epoch+1==opt.epoch:
                        total_path2.append(i[j])
                        total_id2.append(*label)
                        total_actual2.append(actual_label)
                        total_pred2.append(pred_label)

        #visualize_image(path2,lab2,ac2,pre2,others2,flag,fold+1,epoch+1)
    return path2, lab2, ac2, pre2, total_path2, total_id2, total_actual2, total_pred2