


import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

# from ..plot_utils import plot_results
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
        filename="data.npz"
    ):

    data = np.load(filename)
    plt.figure(figsize=(8,4))


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
        plt.fill_between(x, lower, upper, alpha=0.2)

        legends.append(key)
    
    plt.legend(legends)
    plt.xlabel("Training steps")
    plt.ylabel("Environent reward")

    plt.rc('axes', axisbelow=True)
    plt.grid(True, axis='y', zorder=-1)

def run(show = True,filename = "cheetah_training/data.npz"):
    plot_results(sigma=3, filename=filename)

    save_name = filename.split("/")[0] + "/cheetah_training_rewards.pdf"

    plt.title("Cheetah training rewards")
    plt.xlim([0,200000])
    plt.savefig(save_name, format = "pdf")
    plt.savefig("all_figures/" + save_name.split('/')[-1], format="pdf")
    if show:
        plt.show()


if __name__ == "main":
    run(filname="/data.npz")