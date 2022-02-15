import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.patheffects import Stroke, Normal
from scipy.ndimage import gaussian_filter

from src.read_write import load_yaml, load_json, save_json


FOLDER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR  = FOLDER_DIR / 'data/pot_energy_surface'
RESULTS_DIR = FOLDER_DIR / 'results'
pes_files = ['low_fi', 'high_fi', 'ultra_high_fi']

FILE_PATHS = [DATA_DIR / f'{pes_file}.json' for pes_file in pes_files]
NAMES = ['Low', 'High', 'Ultra high']
TRUEMINS_2D = {'Low':         17.4815,
               'High': -203012.37364,
               'Ultra high': -202861.33811}



class DummySettings:
    def __init__(self, data):
        self.dim = data['settings']['dim']
        self.pp_m_slice = data['settings']['pp_m_slice']
        self.bounds = np.array(data['settings']['bounds'])


def calculate_drop_shadow(z, l_0, l_1, sigma=5, alpha=.5):
    """Compute shadow for contour between l_0 and l_1.
    Renders the contour in black, then blues the contour using
    Gaussian filter. Finally sets the alpha channel.
    """
    fig = plt.figure(figsize=(5, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.contourf(z, vmin=z.min(), vmax=z.max(), levels=[l_0, l_1],
                origin='lower', colors='black', extent=[-1, 1, -1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    canvas.draw()
    #A =

def model(data):
    fig, axs = plt.subplots(1, 3, figsize=(21, 5), sharex=True, sharey=True)

    for ax, fidelity_data in zip(axs, data):
        settings, model_data = data['STS'], data['model_data']
        dim = settings.dim
        coords, mu, nu = model_data[:, :dim], model_data[:, -2], model_data[:, -1]
        slice_x, slice_y = settings.pp_m_slice[0], settings.pp_m_slice[1]
        num_slices = settings.pp_m_slice[2]
        num_levels = 20
        x, y = coords[:, slice_x], coords[:, slice_y]
        z = mu.reshape(num_slices, num_slices)
        dz = (z.max() - z.min()) / num_levels
        levels = np.linspace(z.min(), z.max(), num_levels, endpoint=True)
        # cmap = plt.get_cmap("RdBu")
        # plt.contour(x[:num_slices], y[::num_slices],
        #             mu.reshape(num_slices, num_slices), levels, colors='k',
        #             zorder=1)
        # plt.imshow(z)
        ax.contourf(x[:num_slices], y[::num_slices], z, levels=num_levels,
                    zorder=0, cmap='viridis')

    plt.show()


def main():
    data_dicts = [load_json(path, '') for path in FILE_PATHS]
    for data_dict in data_dicts:
        data_dict['STS'] = DummySettings(data_dict)
        data_dict['model_data'] = np.array(data_dict['model_data'])
        data_dict['xhat'] = np.array(data_dict['xhat'])
        data_dict['acqs'] = np.array(data_dict['acqs'])
    model(data_dicts)
#    model(data_dicts)

# def model(data):
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.0, 5.0), sharex=True,
#                             sharey=True)
#     for pes_idx, data in enumerate(data):
#         STS, model_data = data['STS'], data['model_data']
#         data['model_data'][:, -2] -= TRUEMINS_2D[NAMES[pes_idx]]
#         xhat, acqs = data['xhat'], data['acqs']
#         coords = model_data[:, :STS.dim]
#         mu, nu = model_data[:, -2], model_data[:, -1]
#         npts = STS.pp_m_slice[2]
#         x1, x2 = coords[:, STS.pp_m_slice[0]], coords[:, STS.pp_m_slice[1]]
#         axs[pes_idx].contour(x1[:npts], x2[::npts],
#                              mu.reshape(npts, npts), 8, colors='k', zorder=1)
#         im = axs[pes_idx].contourf(x1[:npts], x2[::npts],
#                                    mu.reshape(npts, npts), 150, cmap='viridis',
#                                    zorder=0, antialiased=True)
#         if pes_idx == 0:
#             axs[pes_idx].scatter(*xhat, c='red', marker='*', s=150,
#                                  label='GMP', zorder=2)
#         else:
#             axs[pes_idx].scatter(*xhat, c='red', marker='*', s=150, zorder=2)
#     fig.colorbar(im, ax=axs, shrink=0.8, location='bottom')
#     axs[0].legend(fontsize=15)
#     for i, label in enumerate(['(a) LF', '(b) HF', '(c) UHF']):
#         txt = axs[i].text(-42, 285, label, c='black', weight='bold',
#                           fontsize=17)
#         txt.set_bbox(BOX_STYLE)
#     fig.subplots_adjust(left=0.03, right=0.99, top=0.98,
#                         bottom=0.28, wspace=0.05)
#     plt.savefig(RESULTS_DIR / 'pes.png')
#     plt.show()
#     plt.close()


if __name__ == '__main__':
    main()
