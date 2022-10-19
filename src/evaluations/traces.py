import enum
from typing import List, Tuple
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np

def plot_trace(trace:np.array, col_names: List):
    if trace.shape[0] != len(col_names):
        raise ValueError(f"Traces have {trace.shape[0]} leads, but only {len(col_names)} lead indentifiers specified ")
    n_leads = trace.shape[0]
    row_height=5
    col_len = 20
    fig, axs = plt.subplots(n_leads, figsize=(col_len, n_leads * row_height), sharex=True)
    axs = axs.flat
    for i, col_name in enumerate(col_names):
        axs[i].plot(trace[i], linewidth=0.5)
        axs[i].set_title(col_name)
    for ax in axs:
        ax.set_xticks(np.arange(0, 4096, 500))
        ax.set_axisbelow(True) # set grid to below
        ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, axs

