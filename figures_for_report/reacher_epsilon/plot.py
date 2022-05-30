



from matplotlib import pyplot as plt
import numpy as np


def plot_epsilon(filename):

    data = np.load(filename)

    plt.figure(figsize=(8,4))

    plt.plot(data["random"], label="exp_random", color='C1')
    plt.plot(data["policy"], label= "exp_policy", color='C2')

    plt.legend(loc="upper right")

    plt.xlabel("Number of model updates")
    plt.ylabel("Epsilon value")


def run(show = True, filename = "reacher_epsilon/data.npz"):
    plot_epsilon(filename)

    save_name = filename.split("/")[0] + "/reacher_epsilon.pdf"

    plt.title("Reacher exploration weighting")
    plt.savefig(save_name, format="pdf")
    plt.savefig("all_figures/" + save_name.split('/')[-1], format="pdf")
    if show:
        plt.show()