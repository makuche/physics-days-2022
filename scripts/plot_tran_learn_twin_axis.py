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
setups = ['LF ➔ UHF', 'HF ➔ UHF']
BLUE, RED, GREEN, ORANGE = '#000082', '#FE0000', '#32CD32', '#FF8C00'
SCATTER_DICT = {'color': BLUE, 'alpha': .7, 'marker': 'x',
                'label': 'CCSD(T) run', 's': 45, 'zorder': 2.01}
MEANS_DICT = {'color': BLUE, 'marker': '*', 's': 350, 'label': 'mean',
              'zorder': 2.05}
SCATTER_DICT_TWIN = {'color': GREEN, 'alpha': .7, 'marker': 'x',
                'label': 'CCSD(T) run', 's': 45, 'zorder': 2.01}
MEANS_DICT_TWIN = {'color': GREEN, 'marker': '*', 's': 350, 'label': 'mean'}
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
    tl_results, tl_means = dataframe_to_dict(df, setups=setups)
    plot_tl_results(tl_results, tl_means)


def dataframe_to_dict(df, setups):
    tl_results, tl_means = {}, {}
    for setup in setups:
        tl_results[setup], tl_means[setup] = {}, {}
        for measure in ['Totaltime', 'Iterations']:
            baseline_setup = setup.split(" ")[-1]
            y_tl = df.loc[df['Experiment name'] == setup][measure]
            y_bl = df.loc[df['Experiment name'] == baseline_setup][measure]
            x_tl = df.loc[df['Experiment name'] ==
                          setup]['Initial points source 1']
            x_bl = np.zeros_like(y_bl)
            for coords in [y_tl, y_bl, x_tl, x_bl]:
                coords = np.array(coords)
            x, y = np.hstack((x_bl, x_tl)), np.hstack((y_bl, y_tl))
            coords = np.vstack((x, y)).T
            tmp_df = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1]})
            x_means = np.array(tmp_df['x'].unique())
            y_means = np.array([tmp_df.loc[tmp_df['x'] == x_mean, ['y']].mean()
                               for x_mean in x_means])
            means = np.vstack((x_means, y_means.squeeze())).T
            means = means[means[:, 0].argsort()]
            coords = coords[coords[:, 0].argsort()]
            tl_results[setup][measure], tl_means[setup][measure] = coords, means
    return tl_results, tl_means


def plot_tl_results(tl_results, tl_means):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    for setup, (ax_idx, ax) in zip(setups, enumerate(axs)):
        points, means = tl_results[setup], tl_means[setup]
        ax2 = ax.twinx()

        if ax_idx == 1:
            ax2.set_ylabel('CPU time [h]', fontsize=MEDIUM_SIZE, color=BLUE)
        ax2.tick_params(axis='y', labelcolor=BLUE)
        ax2.set_title(setup, fontsize=MEDIUM_SIZE, color='k')
        ax2.scatter(points['Totaltime'][:, 0], points['Totaltime'][:, 1],
                   **SCATTER_DICT)
        ax2.scatter(means['Totaltime'][:, 0], means['Totaltime'][:, 1],
                   **MEANS_DICT)
        ax2.plot(means['Totaltime'][:, 0], means['Totaltime'][:, 1],
                 ls='dashed', zorder=2)
        if ax_idx == 0:
            ax.set_yticks([])
            ax2.set_yticks([])
        else:
            ax.set_yticks([0, 50, 100])
        ax.scatter(points['Iterations'][:, 0], points['Iterations'][:, 1],
                    **SCATTER_DICT_TWIN)
        ax.scatter(means['Iterations'][:, 0], means['Iterations'][:, 1],
                    **MEANS_DICT_TWIN)
        ax.plot(means['Iterations'][:, 0], means['Iterations'][:, 1],
                 ls='dashed', c=GREEN, zorder=2)
        if ax_idx == 0:
            ax.set_ylabel('BO iterations', fontsize=MEDIUM_SIZE, color=GREEN)
        ax.tick_params(axis='y', labelcolor=GREEN)
    fig.supxlabel('Number of Lower-Fi Samples', fontsize=SMALL_SIZE)
    fig.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/tran_learn_twin_axis.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

