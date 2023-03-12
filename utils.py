import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from scipy.stats import spearmanr

ROOT = r"G:\My Drive\MA\Thesis"

def scatter(data_dict, ax=None, alpha=None, yerr=None, spearman_label=False):
    (x_label, x), (y_label, y) = data_dict.items()
    x, y = np.array(x), np.array(y)

    if ax is None:
        ax = plt.gca()

    no_nan_idx = np.isfinite(x * y)
    x, y = x[no_nan_idx], y[no_nan_idx]

    if yerr is None:
        ax.scatter(x, y, alpha=alpha)
    else:
        ax.errorbar(x, y, yerr, ecolor="k", capsize=7, fmt="o")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if spearman_label:
        add_spearman_label(x, y, ax)


def add_spearman_label(x, y, ax):
    rho, p = spearmanr(x, y)
    if np.isfinite(rho * p):
        rho_label = fr"$\rho={round(rho, 2)}$"
        p_round_order = 3
        if p == 0:
            p_label = "$p=0$"
        elif p >= 10 ** -p_round_order:
            p_label = f"$p={round(p, p_round_order)}$"
        else:
            p_label = f"$p<10^{{{int(np.log10(p))}}}$"

        # placement solution from https://stackoverflow.com/a/59109053:
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="w", ec="w", lw=0, alpha=0)] * 2
        labels = [rho_label, p_label]
        ax.legend(handles, labels, title="Spearman", handlelength=0, handletextpad=0,
                  loc="best", fontsize="small", fancybox=True, framealpha=0.7)
