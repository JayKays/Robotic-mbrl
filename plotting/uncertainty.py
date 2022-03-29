
from util import load_model, load_uncertainty_log
from mbrl.models.util import estimate_uncertainty

from matplotlib import pyplot as plt
import numpy as np


def model_uncertainty_on_dataset(model_dir, data_dir):
    model = load_model(model_dir)
    uncertainty = estimate_uncertainty(model, data_dir)
    return uncertainty


def plot_uncertainty(dirs, idx=0, plot_epsilon=False, show=True):
    
    legends = []
    for dir in dirs.keys():
        uncertainty = load_uncertainty_log(dirs[dir])
        if plot_epsilon:
            plt.plot(uncertainty["ext"][idx:]/np.max(uncertainty["ext"]))
        else:
            plt.plot(uncertainty["ext"][idx:])
        legends.append(dir)

    plt.legend(legends)

    if plot_epsilon: plt.title("Epsilon Value through training")
    else: plt.title("Model Uncertainties")

    if show:
        plt.show()