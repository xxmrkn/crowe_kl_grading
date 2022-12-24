# crowe_kl_classification
Automatic Grading of Crowe and Kl

# Project
深層学習を用いたDigitally Reconstructed Radiographsに基づく変形性股関節症の多指標自動回帰・分類

# Aim 
CTから作成されたDRRを用いて、Crowe分類(脱臼度)・KL(OA重症度)の自動回帰・分類を行う。

# Configuration
* image_size : 224x224
* training_batchsize : 32
* validation_batchsize : 8
* test_batchsize : 8
* num_epochs : 200 or 300
* min_lr : 5e-4
* max_lr : 5e-5
* lr_scheduler : CosineAnnealingLR
* T_max : 300
* num_folds : 4
* num_classes : 1 or 7


# Model
* VisionTransoformer_Base16 
  [Reference : https://arxiv.org/abs/2010.11929]
* VGG16 
  [Reference : https://arxiv.org/abs/1409.1556]
* DenseNet161 
  [Reference : https://arxiv.org/abs/1608.06993]

All models were pretrained by ImageNet.

# Repository Composition
```
.  
├── 📁 bin  
│   ├── 📄 train.sh
│   ├── 📄 train2.sh
│   ├── 📄 inference.sh
│   └── 📄 inference2.sh
│  
└── 📁 src
    ├── 📁 dataset
    │   └── 📄 dataset.py
    ├── 📁 evaluation
    │   └── 📄 evaluationhelper.py  
    ├── 📁 function
    │   ├── 📄 load_datalist.py
    │   └── 📄 prepare_dataframe.py
    ├── 📁 model
    │   └── 📄 select_model.py  
    ├── 📁 utils
    │   ├── 📄 configuration.py
    │   ├── 📄 wandb_config.py
    │   └── 📄 parser.py
    ├── 📁 visualization
    │   └── 📄 visualizehelper.py
    ├── 📄 train.py
    ├── 📄 inference.py
    └── 📄 trainval_one_epoch.py
```