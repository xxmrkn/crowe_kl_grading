import os
import torch
from utils.argparser import get_args

class SaveModel(object):
    
    opt = get_args()

    @classmethod
    def save_model(cls,
                   valid_loss,
                   best_loss,
                   model, 
                   fold):
        
        if valid_loss < best_loss:
            best_loss = valid_loss

            if cls.opt.num_classes == 1:
                tgt = f'{cls.opt.result_path}/{cls.opt.sign}/Regression/weights/{cls.opt.model}'
                os.makedirs(tgt, exist_ok=True)

            else:
                tgt = f'{cls.opt.result_path}/{cls.opt.sign}/Classification/weights/{cls.opt.model}'
                os.makedirs(tgt, exist_ok=True)

            path = os.path.join(
                       tgt,
                       f'{cls.opt.sign}_datalist{cls.opt.datalist}_fold{cls.opt.fold}{fold+1}_{cls.opt.epoch}epoch_weights.pth'
                   )

            torch.save(model.state_dict(), path)
            print('Save Model !')

        return best_loss