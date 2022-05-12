


import numpy as np
from matplotlib import pyplot as plt
import scipy

from scipy.ndimage import gaussian_filter1d, uniform_filter1d




def largest_so_far(array: np.ndarray):

    largest = np.min(array, axis=1, keepdims=True)
    # output = np.ones_like(array) * largest
    output = np.repeat(largest, array.shape[1], axis=1)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] > largest[i]:
                largest[i] = array[i,j]
            
            output[i,j] = largest[i]

    return output

def plot_results(
        largest = False, 
        filter = False, 
        smooth=True,
        sigma = 3,
        std_scale = 2, 
        idx=0, 
        filname="data.npz"
    ):

    data = np.load(filname)
    
    legends = []
    for key in data.keys():
        res = data[key][1:,:]
        if largest:
            mean = res.mean(axis=0)
            mean = largest_so_far(np.expand_dims(mean,0))[0,:]
            std = res.std(axis=0)
        elif filter:
            mean = res.mean(axis=0)
            mean = uniform_filter1d(mean, size=5)
            std = res.std(axis=0)
        else:
            mean = np.mean(res, axis=0)
            std = np.std(res, axis=0)

        upper = mean + std
        lower = mean - std

        if smooth:
            mean = gaussian_filter1d(mean, sigma=sigma)
            upper = gaussian_filter1d(upper, sigma=sigma)
            lower = gaussian_filter1d(lower, sigma=sigma)
            
        x = data[key][0,:]
        plt.plot(x, mean)
        plt.fill_between(x, lower, upper, alpha=0.4)

        legends.append(key)
    
    plt.legend(legends)
    plt.xlabel("Environment steps")
    plt.ylabel("Environent reward")
    # plt.show()