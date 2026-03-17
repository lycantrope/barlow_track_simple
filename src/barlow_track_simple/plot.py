from typing import Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
