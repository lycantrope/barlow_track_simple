from __future__ import annotations

from typing import Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import numpy.typing as npt

mpl.rcParams["svg.fonttype"] = "none"
# mpl.rcParams["font.family"] = "Verdana"


def plot_loss(
    loss: Sequence[Union[float, int]],
    ax: Optional[maxes.Axes] = None,
) -> maxes.Axes:
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.plot(loss)
    ax.set_xlabel("epoch")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.set_ylabel("loss")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_matrices(matrices: npt.ArrayLike, ax: Optional[maxes.Axes] = None, title=""):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    im = ax.imshow(matrices, cmap="magma", vmin=-0.1, vmax=1.0)

    ax.figure.colorbar(im, ax=ax)
    off_diag_val = (np.sum(matrices) - np.trace(matrices)) / (
        np.size(matrices) - np.shape(matrices)[0]
    )
    ax.set_title(f"{title}\nAvg Off-Diag: {off_diag_val:.4f}")
