

import numpy as np
from matplotlib import pyplot as plt

def reward_bar_chart(filename, std_error = True):
    """
    assumes a dictonary of the form {modeltype: reward_array, ...}
    """



    reward_dict = np.load(filename)
    means = np.zeros(len(reward_dict))
    stds = np.zeros(len(reward_dict))
    mins = np.zeros(len(reward_dict))
    maxs = np.zeros(len(reward_dict))
    tags = []    

    for i, key in enumerate(reward_dict.keys()):
        rewards = np.squeeze(reward_dict[key])

        means[i] = rewards.mean()
        stds[i] = rewards.std()
        mins[i] = np.min(rewards)
        maxs[i] = np.max(rewards)
        tags.append(key)

    if std_error:
        yerr = stds
    else:
        yerr = np.vstack((mins, maxs))
        print(yerr)
    

    colors = {
        "pets": "tab:blue",
        "random_uncertainty": "tab:orange",
        "policy_uncertainty": "tab:green"
    }

    x = np.arange(len(tags))
    plt.figure(figsize=(8,4))
    plt.rc('axes', axisbelow=True)
    plt.bar(x, means, 0.4, yerr=yerr, capsize=10, color = [colors[tags[i]] for i in x])
    plt.grid(True, axis='y', zorder=-1)
    plt.xticks(x, tags)
    plt.ylabel("Environment reward")


def run(show = True, filename = "walker_bars/data.npz"):
    reward_bar_chart(filename=filename)

    save_name = filename.split("/")[0] + "/walker_reward_bars.pdf"

    plt.title("Walker rewards")
    plt.savefig(save_name, format="pdf")
    plt.savefig("all_figures/" + save_name.split('/')[-1], format="pdf")
    if show:
        plt.show()

if __name__ == "main":
    run(filname="data.npz")