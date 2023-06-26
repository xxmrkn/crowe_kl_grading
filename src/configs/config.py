from dataclasses import dataclass

@dataclass
class Sign:
    sing: str

@dataclass
class Path:
    base: str
    df:  str
    datalist: str
    image: str
    result: str

@dataclass
class General:
    datalist: int
    fold: int
    batch_size: int
    valid_batch_size: int
    num_workers: int
    wandb_mode: str
    seed: int

@dataclass
class Train:
    epoch: int

@dataclass
class Inference:
    num_sampling: int

@dataclass
class Model:
    name: str
    pretrained: str
    mcdropout: str
    dropout_ratio: float
    dropout_ratio2: float
    num_classes: int

@dataclass
class Scheduler:
    name: str
    t_max: int
    t_0: int
    min_lr: float

@dataclass
class Optimizer:
    name: str
    lr: float
    wd: float

@dataclass
class Criterion:
    name: str

@dataclass
class MyConfig:
    sign_cfg: Sign = Sign()
    path_cfg: Path = Path()
    general_cfg: General = General()
    train_cfg: Train = Train()
    inference_cfg: Inference = Inference()
    model_cfg: Model = Model()
    scheduler_cfg: Scheduler = Scheduler()
    optimizer_cfg: Optimizer = Optimizer()
    criterion_cfg: Criterion = Criterion()

