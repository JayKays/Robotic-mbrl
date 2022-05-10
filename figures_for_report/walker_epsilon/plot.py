



from matplotlib import pyplot as plt
import numpy as np



def plot_epsilon(filename):

    data = np.load(filename)

    plt.figure(figsize=(8,4))

    plt.plot(data["random"], label="random", color='tab:orange')
    plt.plot(data["policy"], label= "policy", color='tab:green')

    plt.legend()

    plt.xlabel("Number of model updates")
    plt.ylabel("Epsilon value")


def run(show = True, filename = "walker_epsilon/data.npz"):
    plot_epsilon(filename)

    save_name = filename.split("/")[0] + "/walker_epsilon.pdf"

    plt.title("Epsilon value through trainng")
    plt.savefig(save_name, format="pdf")
    plt.savefig("all_figures/" + save_name.split('/')[-1], format="pdf")
    plt.show()