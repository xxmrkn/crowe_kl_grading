import argparse

def get_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sign', type=str, default='train', help="Unique Sign")
    parser.add_argument('--datalist', type=str, default='5', help="Datalist ID")
    parser.add_argument('--model', type=str, default='VisionTransformer_Base16', help="Model Name")
    parser.add_argument('--epoch', type=int, default=200, help="Number of Epoch")
    parser.add_argument('--fold', type=str, default='4', help="Number of Fold")
    parser.add_argument('--image_size', type=int, default=224, help="Image size")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
    parser.add_argument('--valid_batch_size', type=int, default=8, help="Validation Batch Size")
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer Name')
    parser.add_argument('--criterion', type=str, default='Focal Loss', help='Loss Function')
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning Rate")
    parser.add_argument('--min_lr', type=float, default=3e-5, help="Min learning Rate")
    parser.add_argument('--t_max', type=int, default=1800, help="No Description")
    parser.add_argument('--t_0', type=int, default=25, help="No Description")
    parser.add_argument('--wd', type=float, default=1e-4, help="Weight Decay")
    parser.add_argument('--num_classes', type=int, default=1, help="Number of Classes")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of Workers")
    parser.add_argument('--seed', type=int, default=42, help="Number of Seed")

    return parser.parse_args()