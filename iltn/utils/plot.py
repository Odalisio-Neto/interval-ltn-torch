import matplotlib.axes
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

from iltn.events.event import Event


def plot_events(x_time: ArrayLike, events: list[Event], ax: matplotlib.axes.Axes, c: str = None,
                zorders: list[int] = None, **mf_kwargs):    
    if zorders is None:
        zorders = [1]*len(events)
    for (i,event) in enumerate(events):
        ax.plot(x_time, event.mf(x_time, **mf_kwargs), c=c, label=event.label, zorder=zorders[i])
        

def set_size(
    width: float | str,
    fraction: float = 1.0,
    subplots: tuple[int, int] = (1, 1),
    height_margin: float = 0.0,
) -> tuple[float, float]:
    """Excerpt from: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type in "thesis", "beamer", "notebook".
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    height_margin: float, optional
            Margin as a ratio of the plot size, which you wish to leave below/above
            the plots, for example, for a title or a legend.

    Returns
    -------
    fig_dim: tuple[float,float]
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    elif width == "notebook":
        width_pt = 570.0
    else:
        width_pt = width
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    fig_height_in = (1 + height_margin) * fig_height_in
    return (fig_width_in, fig_height_in)


def set_tex_style() -> None:
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        #"font.family": "times",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
    plt.rcParams.update(tex_fonts)