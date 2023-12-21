import numpy as np
import warnings
import pathlib
import matplotlib.pyplot as plt

## plotting functions ###
def plot_scat(
    x, y, L = None,
    plot_title: str = 'Scatterplot',
    fig_size: tuple = (16, 8),
    dot_size: float = 3,
    out_path: str = str(pathlib.Path().resolve()) + '/temp.png'
):# -> None:
    """Plots and saves a scatterplot.

    Args:
        x (list | numpy.array | pandas.Series): x axis values.
        y (list | numpy.array | pandas.Series): y axis values.
        L (list | numpy.array | pandas.Series | None, optional): Category label for
            each value. If set to `None`, there will be only one category.
            Defaults to None.
        plot_title (str, optional): Plot title. Defaults to 'Scatterplot'.
        fig_size (tuple, optional): Plot width and height. Defaults to (16, 8).
        dot_size (float, optional): Scatterplot point/dot size. Defaults to 3.
        out_path (str, optional): Path to save plot.
            Defaults to str(pathlib.Path().resolve())+'/temp.png'.

    Returns:
        None: Function does not return anything, it saves plot to file.
    """
    # prepare arguments
    if len(x) != len(y):
        raise ValueError('x is not the same length as y: ', len(x), ' vs ', len(y))
    if len(x) == 0:
        warnings.warn('x contains no values, ignoring plot.')
        return None
    if L is None or len(x) != len(x):
        warnings.warn('plot colour categories maldefined, using single colour.')
        L = np.zeros((len(x)))

    x = np.array(x)
    y = np.array(y)
    L = np.array(L)

    # plot
    _, ax = plt.subplots(figsize=fig_size)
    Ls = np.unique(L)
    for l in Ls:
        mask = (L == l)
        ax.scatter(x[mask], y[mask], label=l, s=dot_size)
    ax.set_title(plot_title)
    if len(Ls) > 1:
        ax.legend()

    # save plot
    plt.savefig(out_path)
    plt.close()

