from matplotlib.axes import Axes
from numpy.random import rand


def stripplot(ax: Axes, datas: tuple, jitter: float = 0.1, x=None, colors=None, **kwargs):

    for i, data in enumerate(datas):
        if x is not None:
            offset = x[i]
        else:
            offset = i

        offsets = rand(len(data)) * jitter + offset + 1 - jitter/2

        if colors is not None:
            ax.scatter(offsets, data, c=colors[i], **kwargs)
        else:
            ax.scatter(offsets, data, **kwargs)