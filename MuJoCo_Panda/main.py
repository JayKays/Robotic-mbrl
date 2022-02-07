from matplotlib import pyplot as plt
import numpy as np

from env.Panda_VIC_env import VIC_Env
from mujoco_panda.utils.viewer_utils import render_frame

import time

def main():
    VIC_env = VIC_Env()
    curr_ee, curr_ori = VIC_env.robot.ee_pose()
    print(VIC_env.robot.ee_pose()[1])
    # --------------------------------------

    VIC_env.controller.set_active(True)
    now_r = time.time()
    i = 0
    count = 0
    Data = []
    VIC_env.reset()
    while True:
        # get current robot end-effector pose
        timestep = VIC_env.controller.timestep / 1
        #print("helloooo",i, timestep)
        robot_pos, robot_ori = VIC_env.robot.ee_pose()
        render_frame(VIC_env.robot.viewer, robot_pos, robot_ori)
        render_frame(VIC_env.robot.viewer, VIC_env.x_d[:, i], VIC_env.goal_ori, alpha=0.2)
        if timestep >= i:
            elapsed_r = time.time() - now_r
            # render controller target and current ee pose using frames
            action = np.array([0.0001, 0.00001])  # np.random.uniform(1.e-6, 0.01, 2)
            s = VIC_env.step(action)
            #print(s)
            i += 1
            if i == 4999:
                break
        VIC_env.robot.render()  # render the visualisation

    plt.plot(VIC_env.Fz_history)
    # plt.plot(VIC_env.controller.x_history[0,:])
    # plt.plot(VIC_env.controller.x_history[1,:])
    plt.show()
    # input("Trajectory complete. Hit Enter to deactivate controller")
    VIC_env.controller.set_active(False)
    VIC_env.controller.stop_controller_cleanly()

if __name__ == "__main__":
    main()