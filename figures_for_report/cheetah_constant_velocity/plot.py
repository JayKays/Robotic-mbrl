

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

def result_bars(filename, plot_max = False, plot_mean = True, plot_std = False, discard_outliers=False):

    data_dict = np.load(filename, allow_pickle=True)
    plt.figure()
    
    plt.rc('axes', axisbelow=True)
    plt.grid(True, axis='y', zorder=-1)

    pos = {
        "pets": -0.11,
        "random_uncertainty": 0.11,
        "policy_uncertainty": 0.22
    }

    colors = {
        "pets": "C0",
        "random_uncertainty": "C1",
        "policy_uncertainty": "C2"
    }

    labels = {
        "pets": "pets",
        "random_uncertainty": "exp_random",
        "policy_uncertainty": "exp_policy"
    }

    

    for exp in data_dict.keys():
        if exp == "policy_uncertainty": continue

        result_dict = data_dict[exp].item()

        sorted_keys = [int(key) for key in list(result_dict.keys()) if key != "final"]
        sorted_keys.sort()
        checkpoints= [str(key) for key in sorted_keys]

        
        x = np.arange(len(checkpoints))
        values = np.zeros(len(checkpoints))
        stds = np.zeros(len(checkpoints))
        for idx, cp in enumerate(checkpoints):
            
            res = result_dict[cp]

            if discard_outliers:
                res = np.sort(res)[0,1:-1]
            if plot_max:
                y = np.max(res)
            elif plot_mean:
                y = np.mean(res)

            values[idx] = y
            stds[idx ]= res.std()
        
        if plot_std:
            plt.bar(x + pos[exp], values, width=0.2, yerr=stds, color=colors[exp], label=labels[exp])
        else:
            plt.bar(x + pos[exp], values, width=0.2, color=colors[exp], label=labels[exp])
    
    plt.ylabel("Recieved reward")
    plt.xlabel("Desired velocity (m/s)")
    plt.xticks(np.arange(len(checkpoints)),[str(cp) for cp in checkpoints ])
    plt.legend(loc = "upper right")
    # plt.show()


def run(show = True, filename = "cheetah_constant_velocity/fixed_data.npz"):
    result_bars(filename=filename)

    save_name = filename.split("/")[0] + "/cheetah_desired_vel.pdf"

    plt.title("Performance tracking a desired velocity in halfcheetah")
    plt.savefig(save_name, format="pdf")
    plt.savefig("all_figures/" + save_name.split('/')[-1], format="pdf")
    if show:
        plt.show()


if __name__ == "__main__":
    # run(filname="data.npz")

    result_bars("checkpoints_cheetah.npz")