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
GREEN, VIOLET = '#24641c', '#8c0454'
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
    plt.figure(figsize=(4, 4))
    initpts = df.loc[:, 'Secondary initial points']
    cpu_reduction = df.loc[:, 'CPU time relative to baseline']
    correlation = df.loc[:, 'Correlation']
    anchors = [0, 1, 2, 4, 5, 6]
    savings = 100 * (1 - np.array(cpu_reduction))
    savings[0], savings[1] = savings[1], savings[0]
    correlations = np.array(correlation)
    correlations[0], correlations[1] = correlations[1], correlations[0]
    lf_points = np.array(initpts)
    lf_points[0], lf_points[1] = lf_points[1], lf_points[0]
    colors = [VIOLET if c < 0.95 else GREEN for c in correlations]
    plt.barh(anchors, savings, tick_label=lf_points, color=colors)
    plt.axhline(3, ls='dashed', c='gray')
    plt.ylabel('Number of Low-Fi Data', fontsize=SMALL_SIZE)
    plt.xlabel('CPU time savings in %', fontsize=SMALL_SIZE)
    plt.xticks([0, 25, 50, 75])
    plt.xlim(0, 80)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/tran_learn_overview_bar.pdf')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--show_plots',
                        action='store_true',
                        dest='show_plots',
                        help="Show (and don't save) plots.")
    parser.add_argument('-d', '--dimension',
                        type=int,
                        default='10',  # Default !={2,4} to plot all points
                        help="Chose between 2 or 4 (2D or 4D).")
    parser.add_argument('-t', '--tolerance',
                        type=float,
                        default=0.1,
                        help='Tolerance level to plot convergence for.')
    args = parser.parse_args()
    main(args)
