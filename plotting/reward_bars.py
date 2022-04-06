


from matplotlib import pyplot as plt
import numpy as np

from util import load_results

def reward_bar_chart(reward_dict, training_dirs = None, remove_outliers = True):
    """
    assumes a dictonary of the form {modeltype: reward_array, ...}
    """

    means = np.zeros(len(reward_dict))
    stds = np.zeros(len(reward_dict))
    training_max = np.zeros(len(reward_dict))
    tags = []    

    for i, key in enumerate(reward_dict.keys()):
        if training_dirs is not None:
            training_max[i] = np.mean(load_results(training_dirs[key])["episode_reward"][-20:])
            training_max[i] = np.median(np.sort(load_results(training_dirs[key])["episode_reward"])[-20:])
        rewards = np.squeeze(reward_dict[key])
        if remove_outliers:
            rewards = np.sort(rewards)[1:-1]

        means[i] = rewards.mean()
        stds[i] = rewards.std()
        tags.append(key)

    x = np.arange(len(tags))
    plt.rc('axes', axisbelow=True)
    plt.bar(x-0.2, means, 0.4, yerr=stds,capsize=10, label="Loaded model result")
    if training_dirs is not None:
        plt.bar(x+0.2, training_max, 0.4, label="Max during training")
    plt.grid(True, axis='y', zorder=-1)
    plt.xticks(x, tags)
    plt.legend()

if __name__ == "__main__":


    reacher_dirs = {
        "pets": "/home/jaykay/Robotic-mbrl/remote_copy/EE/pets_comparison/pets_reacher/2022.03.28/103539",
        "random": "/home/jaykay/Robotic-mbrl/remote_copy/EE/reacher_random_unc/pets_reacher/2022.03.25/172453",
        "policy": "/home/jaykay/Robotic-mbrl/remote_copy/EE/reacher_policy_unc/pets_reacher/2022.03.28/103733"
    }
    cheetah_dirs = {
        "pets": "/home/jaykay/Robotic-mbrl/remote_copy/EE/pets_comparison/pets_halfcheetah/2022.03.18/165809",
        "random": "/home/jaykay/Robotic-mbrl/remote_copy/EE/cheetah_random_uncertainty/pets_halfcheetah/2022.03.22/174906",
        "policy": "/home/jaykay/Robotic-mbrl/remote_copy/EE/cheetah_policy_unc/pets_halfcheetah/2022.03.22/175027"
    }

    env ="cheetah"

    if env == "cheetah": 
        dirs = cheetah_dirs
        dirs = None
        title = "Cheetah Rewards"
        reward = np.load("model_testing/cheetah_desired_vel.npz")
    elif env == "reacher":
        dirs = reacher_dirs
        title= "Reacher Rewards" 
        reward = np.load("model_testing/reacher.npz")

    reward_bar_chart(reward, dirs)
    plt.title(title)
    plt.show()