import pickle
import argparse

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

from Configuration import CFG

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, default='new_mcdropout', help="Choose directory")
    
    opt = parser.parse_args()
    print(f'dir = {opt.dir}')

    total_uncertainty2 = pd.read_csv(f'')
    total_uncertainty1 = pd.read_csv(f'')

    plt.figure(figsize=(12, 7))
    sns.set_style('whitegrid')
    sns.set_palette("Set1",3)    
    sns.pointplot(x="Number of Sampling",
                    y="Accuracy (Normal / 1Neighbor)",
                    hue="Model",
                    data=total_uncertainty1)
    sns.set_palette("Set1",3)
    sns.pointplot(x="Number of Sampling",
                    y="Accuracy (Normal / 1Neighbor)",
                    hue="Model",
                    data=total_uncertainty2,
                    linestyles='--',
                    markers=',')

    plt.savefig(f'')

if __name__ == '__main__':
    main()