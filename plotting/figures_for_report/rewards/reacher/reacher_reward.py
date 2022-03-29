
import sys; sys.path.insert(0, "../../..")

from matplotlib import pyplot as plt
from plotting import plot_reward





if __name__ == "__main__":

    dirs = {
        "pets": "/home/jaykay/Robotic-mbrl/remote_copy/EE/pets_comparison/pets_reacher/2022.03.28/103539",
        "random": "/home/jaykay/Robotic-mbrl/remote_copy/EE/reacher_random_unc/pets_reacher/2022.03.25/172453",
        "policy": "/home/jaykay/Robotic-mbrl/remote_copy/EE/reacher_policy_unc/pets_reacher/2022.03.28/103733"
    }
    
    plot_reward(dirs,idx=2, largest=True, show=False)
    plt.title("New title")
    plt.show()
    plot_reward(dirs,idx=2, filter = True, show=False)
    plt.show()