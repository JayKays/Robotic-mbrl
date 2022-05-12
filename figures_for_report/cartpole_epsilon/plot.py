



from turtle import color
from matplotlib import pyplot as plt
import numpy as np



def plot_epsilon(data):

    plt.figure(figsize=(8,4))
    plt.plot(data["random"], label="random", color='tab:orange')
    plt.plot(data["policy"], label= "policy", color='tab:green')


    plt.legend()
    plt.show()

def run():

    data = np.load("cartpole_epsilon/data.npz")

    plot_epsilon(data)