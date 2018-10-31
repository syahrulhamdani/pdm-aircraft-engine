import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os


def cycle_plot(data, engine_id, colname, rowname=None):
    """helper function to plot ``col`` and indices (cycle) of certain engine.

    parameters
    ----------
    data: Dataset
    engine_id (int): Engine-ID
    colname (str): feature name as y-axis
    rownmae (str): default ``None``. feature name as x-axis

    returns
    -------
    None
    """
    if rowname is None:
        x = 'Cycle'
    else:
        x = rowname
    dataengine = data.loc[data['EngineID'] == engine_id]
    ax = sns.lmplot(x=x, y=colname, data=dataengine, height=10, fit_reg=False)
    plt.title(f'{x} vs {colname} in Engine-{engine_id}')
    plt.show()

    return ax


def save_plot(figure, name=None, pathname='src/visualization'):
    """Save figure into ``pathname``."""
    if name is None:
        filename = os.path.join(pathname, figure.ax.title.get_text())
    else:
        filename = os.path.join(pathname, name)
    figure.savefig(filename)
    print('[INFO] figure is saved')
