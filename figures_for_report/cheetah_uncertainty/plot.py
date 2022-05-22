



from matplotlib import pyplot as plt
import numpy as np



def plot_epsilon(filename):

    data = np.load(filename)

    plt.figure(figsize=(8,4))

    plt.plot(data["pets"], label = "pets", color="tab:blue")
    plt.plot(data["exp_random"], label="exp_random", color='tab:orange')
    plt.plot(data["exp_policy"], label="exp_policy", color='tab:green')

    plt.legend(loc="upper right")

    plt.xlabel("Training steps (x 10 000)")
    plt.ylabel("Uncertainty estimate")


def run(show = True, filename = "cheetah_uncertainty/data.npz"):
    plot_epsilon(filename)

    save_name = filename.split("/")[0] + "/cheetah_uncertainty.pdf"

    plt.title("Halfcheetah model uncertainty")
    plt.ylim([0, 0.2])
    plt.savefig(save_name, format="pdf")
    plt.savefig("all_figures/" + save_name.split('/')[-1], format="pdf")
    if show:
        plt.show()