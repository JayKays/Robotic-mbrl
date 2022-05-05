
from unicodedata import unidata_version
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# from figures_for_report.plot_utils import largest_so_far

def largest_so_far(array: np.ndarray):

    largest = np.min(array, axis=1, keepdims=True)
    # output = np.ones_like(array) * largest
    output = np.repeat(largest, array.shape[1], axis=1)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] > largest[i]:
                largest[i] = array[i,j]
            
            output[i,j] = largest[i]

    return output

def load_save_training_rewards(load_dir, save_dir, num_seeds=3, experiments = None):

    dirs = {
    "pets" : load_dir + "/pets",
    "random" : load_dir + "/random_uncertainty",
    "policy" : load_dir + "/policy_uncertainty",
    }

    if experiments is None:
        exps = dirs.keys()
    else: exps = experiments
    # rewards = {"pets": pd.DataFrame(), "random": pd.DataFrame(), "policy": pd.DataFrame()}
    rewards = {}
    for key in exps:
        df = pd.DataFrame()
        for i in range(num_seeds):
            reward_file = dirs[key] + f"/seed_{i}/results.csv"
            results = pd.read_csv(reward_file)
            steps = results["env_step"].values
            reward = results["episode_reward"]
            new_steps = df.index.union(steps)
            df = df.reindex(new_steps)
            df[f"{i}"] = pd.Series(reward.values, index=steps)

        #interpolate NaN values
        df = df.fillna(method="ffill").fillna(method="backfill")

        rewards[key] = np.vstack([df.index.to_numpy(), df.to_numpy().T]).astype("float64")
        # plt.plot(df.index.values, df.values)
        # plt.show()

    test = rewards["pets"]
    print(test.shape)


    np.savez(save_dir + "/data.npz", **rewards)

def load_save_epsilon(load_dir, save_dir, num_seeds=3):
    dirs = {
    # "pets" : load_dir + "/pets",
    "random" : load_dir + "/random_uncertainty",
    "policy" : load_dir + "/policy_uncertainty",
    }

    epsilons = {}
    for key in dirs.keys():
        eps = 0
        length = 0
        for i in [2]:
            uncertainty = np.load(dirs[key] + f"/seed_{i}/uncertainty_log.npz")["ext"]
            # length = max(uncertainty.shape[0], length)
            if eps == 0: eps += uncertainty*0
            if uncertainty.shape[0] > length:
                length = uncertainty.shape[0]
                eps = np.pad(eps, (0,eps.shape[0] - length),"edge")
            elif uncertainty.shape[0] < length:
                uncertainty = np.pad(uncertainty, (0,length - uncertainty.shape[0]), "edge")
            
            eps += uncertainty / largest_so_far(np.expand_dims(uncertainty, 0))[0,:]
            
        epsilons[key] = eps/num_seeds

    np.savez(save_dir + "/data.npz", **epsilons)

if __name__ == "__main__":
    
    results_dir = "/home/jaykay/Robotic-mbrl/test_results/"
    save_dir = "/home/jaykay/Robotic-mbrl/figures_for_report/"

    exps = ["cartpole"]#,"reacher","walker", "cheetah"]

    for exp in exps:

        load_save_training_rewards(results_dir + exp, save_dir + f"{exp}_training", experiments=["pets", "random", "policy"])
        load_save_epsilon(results_dir + exp, save_dir + f"{exp}_epsilon")