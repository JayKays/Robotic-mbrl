



from matplotlib import pyplot as plt
import numpy as np



def plot_epsilon(data):

    plt.figure(figsize=(8,4))

    for key in data.keys():
        plt.plot(data[key], label=key)

    plt.legend()
    plt.show()

def run():

    data = np.load("cheetah_epsilon/data.npz")

    plot_epsilon(data)