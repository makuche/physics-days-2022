import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from pathlib import Path
from matplotlib.legend_handler import HandlerTuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / 'results/'

FIG_LABELS = {'LF ➔ HF': ['a)', 'b)'],
                 'LF ➔ UHF': ['c)', 'd)'],
                 'HF ➔ UHF': ['e)', 'f)']}
BLUE, RED, GREEN, ORANGE = '#000082', '#FE0000', '#32CD32', '#FF8C00'
PLOT_COLORS = {2: BLUE, 4: GREEN}
SYMBOLS = {2: 'o', 4: 's'}
SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE = 18, 20, 22
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
    fig = plt.figure(figsize=(5, 4))
    cm = plt.cm.get_cmap('PiYG', 2)
    initpts = df.loc[:, 'Secondary initial points']
    cpu_reduction = df.loc[:, 'CPU time relative to baseline']
    correlation = df.loc[:, 'Correlation']
    dimension = df.loc[:, 'Dimension'].to_list()
    markers = [SYMBOLS[dim] for dim in dimension]
    labels = np.array([f'{dim}D' for dim in dimension])
    corr_min, corr_max = correlation.min(), correlation.max()
    for x, y, c, l, marker in zip(initpts, cpu_reduction, correlation,
                                labels, markers):
        sc = plt.scatter(x, 100*(1-y), c=c, label=l, marker=marker, s=300,
                        vmin=corr_min, vmax=corr_max, cmap=cm)
    cb = plt.colorbar(sc)
    cb.set_label('Pearson correlation', fontsize=SMALL_SIZE)
    plt.ylabel('CPU time savings in %', fontsize=SMALL_SIZE)
    plt.xlabel('Number of Low-Fi Data', fontsize=SMALL_SIZE)
    plt.yticks([25, 50, 75])
    plt.xticks([50, 100, 150, 200])
    handles, _ = plt.gca().get_legend_handles_labels()
    handles = np.delete(np.asarray(handles), [0,3])
    plt.legend([(handles[0], handles[1]), (handles[2], handles[3])],
               ['2D', '4D'], fontsize=SMALL_SIZE, loc='center left',
               handler_map={tuple: HandlerTuple(ndivide=None, pad=.7)})
    plt.xlim(20, 215)
    plt.ylim(15, 80)
    fig.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/tran_learn_overview.pdf')


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
