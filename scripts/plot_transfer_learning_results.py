import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / 'results/'

FIG_LABELS = {'LF ➔ HF': ['a)', 'b)'],
                 'LF ➔ UHF': ['c)', 'd)'],
                 'HF ➔ UHF': ['e)', 'f)']}
BLUE, RED = '#000082', '#FE0000'
SCATTER_DICT = {'color': BLUE, 'alpha': .4, 'marker': 'x',
                'label': 'observation', 's': 60}
MEANS_DICT = {'color': RED, 'marker': '*', 's': 120, 'label': 'mean'}

def main(args):
    df = pd.read_csv(PROJECT_DIR / 'data/transfer_learning.csv')
    df = df.loc[df['Dimension'] == args.dimension]  # Use only 2D or 4D
    tol = args.tolerance
    df = df.loc[:, ['Experiment name', 'Dimension', 'Initial points source 0',
                    'Initial points source 1', f'Iterations [{tol} kcal/mol]',
                    f'Totaltime [{tol} kcal/mol]']]
    df = df.rename(columns={f'Iterations [{tol} kcal/mol]': 'Iterations',
                       f'Totaltime [{tol} kcal/mol]': 'Totaltime'})
    tmp_df = df.loc[df['Initial points source 1'] != 0, ['Iterations']]
    tmp_df.to_csv('tmp.csv')
    plot_tl_results(df)


def plot_tl_results(df):
    fig, axs = plt.subplot_mosaic([['a)', 'c)', 'e)'], ['b)', 'd)', 'f)']],
                                  figsize=(8, 5), sharex=True)
    for label, ax in axs.items():
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(-25/72, -7/72,
                                              fig.dpi_scale_trans)
        ax.text(1.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize=18, verticalalignment='top')
    means = {}
    for setup in FIG_LABELS:
        for measure, ax_label in zip(
            ['Iterations', 'Totaltime'], FIG_LABELS[setup]):
            means[setup] = {measure: {'x': [], 'y': []}}
            baseline_setup = setup.split(" ")[-1]
            x_tl = np.array(df.loc[
                df['Experiment name'] == setup]['Initial points source 1'])
            y_bl = df.loc[df['Experiment name'] == baseline_setup][measure]
            y_tl = df.loc[df['Experiment name'] == setup][measure]
            if measure == 'Totaltime':
                y_bl, y_tl = y_bl/3600, y_tl/3600
            x_bl = np.zeros_like(y_bl)
            for x, y in zip([x_bl, x_tl], [y_bl, y_tl]):
                tmp_df = pd.DataFrame({'x': x, 'y': y})
                x_means = np.array(tmp_df['x'].unique())
                y_means = np.array([tmp_df.loc[tmp_df['x'] == x_mean, ['y']].mean()
                                    for x_mean in x_means])
                means[setup][measure]['x'].append(x_means.squeeze())
                means[setup][measure]['y'].append(y_means.squeeze())
                axs[ax_label].scatter(x, y, **SCATTER_DICT)
                axs[ax_label].scatter(x_means, y_means, **MEANS_DICT)
                print(ax_label, x_means, y_means)
            if measure == 'Totaltime':
                axs[ax_label].set_ylabel('CPU time [h]')
                axs[ax_label].set_xlabel('Number of DFT samples')
            elif measure == 'Iterations':
                axs[ax_label].set_ylabel('BOSS iterations')
    plt.tight_layout()
    plt.savefig('test.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--show_plots',
                        action='store_true',
                        dest='show_plots',
                        help="Show (and don't save) plots.")
    parser.add_argument('-d', '--dimension',
                        type=int,
                        default='2',
                        help="Chose between 2 or 4 (2D or 4D).")
    parser.add_argument('-t', '--tolerance',
                        type=float,
                        default=0.1,
                        help='Tolerance level to plot convergence for.')
    args = parser.parse_args()
    main(args)
