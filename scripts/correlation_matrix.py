import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data/'


def main():
    data = pd.read_csv(f'{DATA_DIR}/fidelity_correlations.csv',
                       index_col='Fidelity')
    mask = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    with sn.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sn.heatmap(data, annot=True, fmt='.3g', mask=mask,
                            cbar_kws={
                                'label': 'Pearson Correlation Coefficient'})
        ax.yaxis.label.set_size(20)
        fig = ax.get_figure()
        fig.savefig("test.png")

if __name__ == '__main__':
    main()
