import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtransforms
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / 'results/'

# 'LF ➔ HF': ['a)', 'b)'],
FIG_LABELS = {'LF ➔ UHF': ['a)', 'b)'],
              'HF ➔ UHF': ['c)', 'd)']}
BLUE, RED = '#000082', '#FE0000'
SCATTER_DICT = {'color': BLUE, 'alpha': .4, 'marker': 'x',
                'label': 'CCSD(T) run', 's': 60}
MEANS_DICT = {'color': RED, 'marker': '*', 's': 120, 'label': 'mean'}
SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE = 14, 18, 22
mpl.rc('xtick', labelsize=SMALL_SIZE)
mpl.rc('ytick', labelsize=SMALL_SIZE)


def main(args):
    df = pd.read_csv(PROJECT_DIR / 'data/transfer_learning.csv')
    df = df.loc[df['Dimension'] == args.dimension]  # Use only 2D or 4D
    tol = args.tolerance
    df = df.loc[:, ['Experiment name', 'Dimension', 'Initial points source 0',
                    'Initial points source 1', f'Iterations [{tol} kcal/mol]',
                    f'Totaltime [{tol} kcal/mol]']]
    df = df.rename(columns={f'Iterations [{tol} kcal/mol]': 'Iterations',
                   f'Totaltime [{tol} kcal/mol]': 'Totaltime'})
    plot_tl_results(df)


def plot_tl_results(df):
    fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'd)']],
                                  figsize=(5, 5), sharex=True)
    means = {}
    for setup in FIG_LABELS:
        means[setup] = {}
        for measure, ax_label in zip(
                ['Iterations', 'Totaltime'], FIG_LABELS[setup]):
            means[setup][measure] = []
            baseline_setup = setup.split(" ")[-1]
            x_tl = np.array(df.loc[
                df['Experiment name'] == setup]['Initial points source 1'])
            y_bl = df.loc[df['Experiment name'] == baseline_setup][measure]
            y_tl = df.loc[df['Experiment name'] == setup][measure]
            x_bl = np.zeros_like(y_bl)
            for x, y in zip([x_bl, x_tl], [y_bl, y_tl]):
                tmp_df = pd.DataFrame({'x': x, 'y': y})
                x_means = np.array(tmp_df['x'].unique())
                y_means = np.array(
                    [tmp_df.loc[tmp_df['x'] == x_mean, ['y']].mean()
                     for x_mean in x_means])
                for x_mean, y_mean in zip(x_means, y_means):
                    x_mean = x_mean.squeeze()
                    y_mean = y_mean.squeeze()
                    means[setup][measure].append([int(x_mean), float(y_mean)])
                axs[ax_label].scatter(x, y, **SCATTER_DICT)
                axs[ax_label].scatter(x_means, y_means, **MEANS_DICT)
    for setup in means:
        for measure, ax_label in zip(means[setup], FIG_LABELS[setup]):
            tmp = means[setup][measure]
            tmp = np.array(tmp)
            tmp = tmp[tmp[:, 0].argsort()]
            axs[ax_label].plot(tmp[:, 0], tmp[:, 1], zorder=-10, c=RED,
                               ls='dashed')

    axs['a)'].set_title('LF ➔ UHF', fontsize=SMALL_SIZE)
    axs['a)'].set_ylabel('BOSS iterations', fontsize=SMALL_SIZE)
    axs['b)'].set_ylabel('CPU time [h]', fontsize=SMALL_SIZE)
    axs['c)'].set_title('HF ➔ UHF', fontsize=SMALL_SIZE)
    axs['c)'].set_yticks([])
    axs['d)'].set_yticks([])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs['d)'].legend(by_label.values(), by_label.keys(), fontsize=SMALL_SIZE,
                     loc='lower left')
    fig.supxlabel('Number of Lower-Fi Samples', fontsize=SMALL_SIZE)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    plt.savefig('results/tran_learn_gain.pdf')


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
