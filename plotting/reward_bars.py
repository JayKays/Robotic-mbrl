


from matplotlib import pyplot as plt
import numpy as np

def reward_bar_chart(reward_dict):
    """
    assumes a dictonary of the form {modeltype: reward_array, ...}
    """

    means = np.zeros(len(reward_dict))
    stds = np.zeros(len(reward_dict))
    legends = []    

    for i, key in enumerate(reward_dict.keys()):

        means[i] = reward_dict[key].mean()
        stds[i] = reward_dict[key].std()
        legends.append(key)

    plt.bar(legends, means, yerr=stds)
    if np.all(means < 0):
        plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":

    reward = np.load("model_testing/reacher.npz")
    reward = np.load("model_testing/cheetah.npz")

    reward_bar_chart(reward)