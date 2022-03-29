

import sys; sys.path.insert(0, "../../..")

from matplotlib import pyplot as plt
from plotting import plot_reward





if __name__ == "__main__":

    cheetah_dirs = {
        "pets": "/home/jaykay/Robotic-mbrl/remote_copy/EE/pets_comparison/pets_halfcheetah/2022.03.18/165809",
        "random": "/home/jaykay/Robotic-mbrl/remote_copy/EE/cheetah_random_uncertainty/pets_halfcheetah/2022.03.22/174906",
        "policy": "/home/jaykay/Robotic-mbrl/remote_copy/EE/cheetah_policy_unc/pets_halfcheetah/2022.03.22/175027"
    }
    
    plot_reward(cheetah_dirs, largest=True, show=False)
    plt.show()
    plot_reward(cheetah_dirs, filter = True, show=False)
    plt.show()