
import numpy as np

from matplotlib import pyplot as plt
from util import load_results
from scipy.ndimage import uniform_filter1d


def largest_so_far(array: np.ndarray):

    largest = np.min(array)
    output = np.ones_like(array) * largest
    for i in range(array.shape[0]):
        if array[i] > largest:
            largest = array[i]
        
        output[i] = largest

    return output


def plot_reward(dirs, idx=0, largest=False, filter = False, show=True):
    
    legends = []
    for dir in dirs.keys():
        res = load_results(dirs[dir])
        if largest:
            plt.plot(res["env_step"][idx:], largest_so_far(res["episode_reward"])[idx:])
        elif filter:
            plt.plot(res["env_step"][idx:], uniform_filter1d(res["episode_reward"], size=5)[idx:])
        else:
            plt.plot(res["env_step"][idx:], res["episode_reward"][idx:])
        legends.append(dir)
    
    plt.legend(legends)
    plt.title("Environment rewards")
    if show:
        plt.show()