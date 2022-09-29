import pickle
import argparse

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

from Configuration import CFG

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sampling', type=int, default=1, help="Number of sampling")
    parser.add_argument('--mode', type=str, default='Variance', help="Choose mode Probability or Variance")
    
    opt = parser.parse_args()
    print(f'sampling = {opt.sampling}')
    print(f'mode = {opt.mode}')
    
    total_uncertainty = pd.read_csv(f'')

    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')
    
    if opt.mode == 'Probability':
        plt.figure(figsize=(40, 40))
        g = sns.catplot(x="Model",
                        y="Probability",
                        hue="Uncertainty",
                        data=total_uncertainty,
                        kind="box",
                        legend_out=False,
                    )
        g.set(ylim=(0.0, 1.0))

    elif opt.mode == 'Variance':
        plt.figure(figsize=(40, 40))
        g = sns.catplot(x="Model",
                        y="Variance",
                        hue="Uncertainty",
                        data=total_uncertainty,
                        kind="box",
                        legend_out=False,
                    )
        g.set(ylim=(0.0, 0.02))

    else:
        raise('No Exist !')
    g.fig.set_figheight(10)
    g.fig.set_figwidth(8)

    new_labels = [f"Exact",f"1 Neighbor",f"Others"]
    
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)

    g.savefig(f'')


if __name__ == '__main__':
    main()