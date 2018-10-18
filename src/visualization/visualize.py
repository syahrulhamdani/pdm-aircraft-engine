import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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
    sns.lmplot(x=x, y=colname, data=data, scatter_kws={'s': 10}, size=7)
    plt.title(f'{x} vs {colname} in Engine-{engine_id}')
    plt.show()