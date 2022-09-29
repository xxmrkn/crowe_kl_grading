# crowe_kl_classification
Automatic Classification Crowe and Kl

# Project
深層学習を用いたDigitally Reconstructed Radiographsに基づく変形性股関節症の多指標自動分類

# Aim 
CTから作成されたDRRを用いて、Crowe分類(脱臼度)・KL(OA重症度)の自動分類を行う。

# Configuration
* image_size : 224x224
* training_batchsize : 32
* validation_batchsize : 32
* num_epochs : 200
* min_lr : 1e-5
* max_lr : 3e-5
* lr_scheduler : CosineAnnealingLR
* T_max : 1800
* num_folds : 4
* num_classes : 7


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
│   └── 📄 train.sh
├── 📁 dataset, docs, notebook
│   └── 📄 No description
└── 📁 src
    ├── 📁 dataset
    │   └── 📄 dataset.py
    ├── 📁 evaluation
    │   └── 📄 EvaluationHelper.py  
    ├── 📁 function
    │   ├── 📄 compare_acc.py
    │   └── 📄 extract.py
    ├── 📁 model
    │   ├── 📄 coatnet.py
    │   └── 📄 select_model.py  
    ├── 📁 utils
    │   ├── 📄 Configuration.py
    │   └── 📄 Parser.py
    ├── 📁 visualization
    │   ├── 📄 visualize_bplot.py
    │   ├── 📄 visualize_lineplot.py
    │   └── 📄 VisualizeHelper.py
    ├── 📄 mcdropout.py
    ├── 📄 train.py
    └── 📄 trainval_one_epoch.py
```