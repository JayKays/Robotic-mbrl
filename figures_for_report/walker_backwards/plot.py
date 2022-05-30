

import numpy as np
from matplotlib import pyplot as plt

def reward_bar_chart(filename, std_error = False):
    """
    assumes a dictonary of the form {modeltype: reward_array, ...}
    """
    reward_dict = np.load(filename)
    means = np.zeros(len(reward_dict)-1)
    stds = np.zeros(len(reward_dict)-1)
    mins = np.zeros(len(reward_dict)-1)
    maxs = np.zeros(len(reward_dict)-1)
    tags = []    

    colors = {
        "pets": "C0",
        "exp_random": "C1",
        "exp_policy": "C2"
    }
    labels = {
        "pets": "pets",
        "random_uncertainty": "exp_random",
        "policy_uncertainty": "exp_policy"
    }

    for i, key in enumerate(reward_dict.keys()):
        if key == "policy_uncertainty": continue
        rewards = np.squeeze(reward_dict[key])

        means[i] = rewards.mean()
        stds[i] = rewards.std()
        mins[i] = np.min(rewards)
        maxs[i] = np.max(rewards)
        tags.append(labels[key])

    if std_error:
        yerr = stds
    else:
        yerr = None
    
    

    x = np.arange(len(tags))
    width = 0.4
    margin = (1 - width) + width / 2


    plt.figure(figsize=(8,4))
    plt.rc('axes', axisbelow=True)
    plt.bar(x, means, width, align = "center", color = [colors[tags[i]] for i in x])
    plt.xlim(-margin, len(x) - 1 + margin)
    plt.grid(True, axis='y', zorder=-1)
    plt.xticks(x, tags)
    plt.ylabel("Environment reward")
    plt.xlabel("Agent type")


def run(show = True, filename = "walker_backwards/data.npz"):
    reward_bar_chart(filename=filename)

    save_name = filename.split("/")[0] + "/walker_backwards.pdf"

    plt.title("Performance walking backwards in walker environment")

    plt.ylim([0,100])
    plt.savefig(save_name, format="pdf")
    plt.savefig("all_figures/" + save_name.split('/')[-1], format="pdf")
    if show:
        plt.show()


# if __name__ == "__main__":
    # run(filname="data.npz")

    # result_bars("checkpoints_cheetah.npz")