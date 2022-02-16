import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / 'results/'

FIG_LABELS = {'LF ➔ HF': ['a)', 'b)'],
                 'LF ➔ UHF': ['c)', 'd)'],
                 'HF ➔ UHF': ['e)', 'f)']}
ORANGE, GREEN = '#FF6347', '#32CD32'
PLOT_COLORS = {2: ORANGE, 4: GREEN}
SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE = 14, 18, 22
mpl.rc('xtick', labelsize=SMALL_SIZE)
mpl.rc('ytick', labelsize=SMALL_SIZE)

def main(args):
    df = pd.read_csv(PROJECT_DIR / 'data/cpu_time_reductions.csv')
    if args.dimension in [2, 4]:
        df = df.loc[df['Dimension'] == args.dimension]   # Use only 2D or 4D
    mask = ['VHF' not in entry for entry in df.loc[:, 'Fidelities']]
    df = df.loc[mask]
    plot_transfer_learning_overview(df)


def plot_transfer_learning_overview(df):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    initpts = df.loc[:, 'Secondary initial points']
    cpu_reduction = df.loc[:, 'CPU time relative to baseline']
    correlation = df.loc[:, 'Correlation']
    dimension = df.loc[:, 'Dimension'].to_list()
    colors = [PLOT_COLORS[dim] for dim in dimension]
    labels = np.array([f'{dim}D' for dim in dimension])
    for x, x_c, y, c, l in zip(initpts, correlation, cpu_reduction, colors,
                               labels):
        axs[0].scatter(x, 100*(1-y), c=c, label=l, alpha=.7)
        axs[1].scatter(x_c, 100*(1-y), c=c, label=l, alpha=.7)
    axs[0].set_ylabel('CPU time\nsavings in %', fontsize=SMALL_SIZE)
    axs[0].set_xlabel('Number of\nLow-Fi Data', fontsize=SMALL_SIZE)
    axs[1].set_xlabel('Correlation\nbetween fidelities', fontsize=SMALL_SIZE)
    axs[0].set_yticks([25, 50, 75])
    axs[0].set_xticks([50, 100, 150, 200])
    axs[1].set_xticks([.95, .975, 1.0])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[1].legend(by_label.values(), by_label.keys(), fontsize=SMALL_SIZE,
               loc='lower right')
    #fig.suptitle('CPU time savings', fontsize=MEDIUM_SIZE)
    fig.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/tran_learn_overview.png')


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
