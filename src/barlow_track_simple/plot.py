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


def init_ax() -> maxes.Axes:
    fig = plt.figure(layout="constrained", frameon=False)
    return fig.add_subplot(111)


def plot_loss(
    loss: Sequence[Union[float, int]],
    ax: Optional[maxes.Axes] = None,
) -> maxes.Axes:
    if ax is None:
        ax = init_ax()

    ax.plot(loss)
    ax.set_xlabel("epoch")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.set_ylabel("loss")
    ax.patch.set_alpha(0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_matrices(
    matrices: npt.ArrayLike,
    ax: Optional[maxes.Axes] = None,
    title="",
    show_colorbar: bool = False,
):
    if ax is None:
        ax = init_ax()

    pcm = ax.imshow(matrices, cmap="magma", vmin=-0.1, vmax=1.0)
    if show_colorbar:
        cax = ax.inset_axes((1.04, 0.05, 0.05, 0.9))
        fig = ax.get_figure(root=True)
        assert fig is not None, "Cannot retrieve root figure"
        fig.colorbar(pcm, cax=cax)

    off_diag_val = (np.sum(matrices) - np.trace(matrices)) / (
        np.size(matrices) - np.shape(matrices)[0]
    )
    ax.patch.set_alpha(0.0)
    ax.set_title(f"{title}\nAvg Off-Diag: {off_diag_val:.4f}")
