import os
import time
import mujoco_py
import numpy as np
import gym
from gym import spaces
# sys.path.append("/home/akhil/PhD/RoL/mujoco_panda-master")
from mujoco_panda import PandaArm
from mujoco_panda.utils.viewer_utils import render_frame
from mujoco_panda.utils.debug_utils import ParallelPythonCmd
from mujoco_panda.controllers.torque_based_controllers import VIC
from mujoco_panda.utils import VIC_func as func
from mujoco_panda.controllers.torque_based_controllers import VIC_config as cfg
import time
import random
import quaternion
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt


class VIC_Env(gym.Env):

    def __init__(self):

        MODEL_PATH = os.environ['MJ_PANDA_PATH'] + '/mujoco_panda/models/'
        self.robot = PandaArm(model_path=MODEL_PATH + 'panda_block_table.xml',
                              render=True, compensate_gravity=False, smooth_ft_sensor=True)
        if mujoco_py.functions.mj_isPyramidal(self.robot.model):
            print("Type of friction cone is pyramidal")
        else:
            print("Type of friction cone is eliptical")
        # self.init_jpos = np.array([-0.0242187,-0.26637015,-0.23036408,-1.99276073,-0.05414588,1.72812007, 0.52766157])
        self.init_jpos = np.array([0., -0.7, 0, -2.356, 0, 1.656, 0.785])
        self.robot.hard_set_joint_positions(self.init_jpos)

        self.robot.sim_step()
        time.sleep(1.0)

        self.action_space = spaces.Box(low=np.array([cfg.GAMMA_B_LOWER, cfg.GAMMA_K_LOWER]), \
                                     high=np.array([cfg.GAMMA_B_UPPER, cfg.GAMMA_K_UPPER]))
        # print("space: ", self.action_space.low[0])
        self.observation_space = spaces.Box(
            low=np.array([cfg.LOWER_Fz, cfg.LOWER_Z_ERROR, cfg.LOWER_Vz, cfg.LOWER_X_ERROR, \
                          cfg.LOWER_Y_ERROR]), \
            high=np.array([cfg.UPPER_Fz, cfg.UPPER_Z_ERROR, cfg.UPPER_Vz, cfg.UPPER_X_ERROR, \
                           cfg.UPPER_Y_ERROR]))
        self.max_num_it = cfg.MAX_NUM_IT
        self.M = cfg.M
        self.B = cfg.B
        self.K = cfg.K
        self.gamma = np.identity(18)
        self.gamma[8, 8] = cfg.GAMMA_B_INIT
        self.gamma[14, 14] = cfg.GAMMA_K_INIT
        self.init_action = np.array([cfg.GAMMA_B_INIT, cfg.GAMMA_B_INIT])
        self.Fz_offset = 0

        self.lam = np.zeros(18)
        # set desired pose/force trajectory
        self.f_d = func.generate_Fd_steep(self.max_num_it, cfg.Fd, cfg.T)
        self.f_d[2, :] = -cfg.Fd

        self.lam = np.zeros(18)
        # set desired pose/force trajectory
        self.f_d = func.generate_Fd_steep(self.max_num_it, cfg.Fd, cfg.T)
        self.f_d[2, :] = -cfg.Fd
        self.goal_ori = np.asarray(self.robot.ee_pose()[1])
        self.x_d_ddot, self.x_d_dot, self.x_d = func.generate_desired_trajectory_tc(self.robot, self.max_num_it, cfg.T,
                                                                                    move_in_x=True)
        plt.plot(self.x_d[0, :]) 
        plt.plot(self.x_d[1, :])
        plt.show()

        ctrl_config = {
            'fc': cfg.PUBLISH_RATE,
            'M': cfg.M,
            'K': cfg.K,
            'B': cfg.B,
            'gamma_K': cfg.GAMMA_K_INIT,
            'gamma_B': cfg.GAMMA_B_INIT,
            'max_num_it': cfg.MAX_NUM_IT * 10,
            'K_v': cfg.K_v,
            'P': cfg.P,
            'B_hat_lower': cfg.B_hat_lower,
            'B_hat_upper': cfg.B_hat_upper,
            'K_hat_lower': cfg.K_hat_lower,
            'K_hat_upper': cfg.K_hat_upper,
            'use_null_space_control': True,
            'ft_dir': [0, 0, 0, 0, 0, 0],
            # newton meter
            'null_kp': [5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0],
            'null_kd': 0,
            'null_ctrl_wt': 2.5,
            'use_orientation_ctrl': True,
            'linear_error_thr': 0.025,
            'angular_error_thr': 0.01,
        }
        self.Fz_history = np.zeros(self.max_num_it)
        self.i = 0
        # print("mass: ", p.mass_matrix())
        self.controller = VIC(self.robot, config=ctrl_config)

    # activate controller (simulation step and controller thread now running)

    def get_obs(self):
        obs_dict = self.controller.state_dict
        return np.array(
            [obs_dict["FT"][2] - self.f_d[2, self.i] - self.Fz_offset, \
             obs_dict["pose"][2] - self.x_d[2, self.i], \
             obs_dict["vel"][2] - self.x_d_dot[2, self.i]])

    def step(self, action):
        self.controller.set_goal(action, self.x_d[:, self.i], self.goal_ori, self.x_d_dot[:, self.i], \
                                 self.x_d_ddot[:, self.i], goal_force=self.f_d[:, self.i])
        self.controller._send_cmd()
        self.state = self.get_obs()
        self.Fz_history[self.i] = self.state[0]
        self.i += 1
        self.controller.get_robot_states()
        self.state = self.get_obs()
        #print("torque ", self.controller._cmd)
        #print("pose ", self.x_d[:, self.i-1] - self.robot.ee_pose()[0])
        #print("smoothed FT reading: ", self.robot.get_ft_reading(pr=True)[0])
        return self.state

    def reset(self):
        print("resetting the controller")
        self.controller.reset()
        print("resetting env")
        index = 0  # np.random.randint(0, (0.9 * self.max_num_it))
        self.iteration = index
        self.lam = np.zeros(18)
        self.robot.hard_set_joint_positions(self.init_jpos)
        self.time_per_iteration = np.zeros(self.max_num_it)
        self.x_history = np.zeros((6, self.max_num_it))
        self.x_dot_history = np.zeros((6, self.max_num_it))
        self.p_hist = np.zeros((3, self.max_num_it))
        self.Fz_history = np.zeros(self.max_num_it)
        self.h_e_hist = np.zeros((6, self.max_num_it))
        self.gamma_B_hist = np.zeros(self.max_num_it)
        self.gamma_K_hist = np.zeros(self.max_num_it)
        self.Kp_pos_hist = np.zeros(self.max_num_it)
        self.Kp_z_hist = np.zeros(self.max_num_it)
        self.Kd_z_hist = np.zeros(self.max_num_it)

        # array with data meant for plotting
        self.data_for_plotting = np.zeros((17, self.max_num_it))
        self.steps_beyond_done = None
        return self.get_obs()


if __name__ == "__main__":
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
    plt.show()
    # input("Trajectory complete. Hit Enter to deactivate controller")
    VIC_env.controller.set_active(False)
    VIC_env.controller.stop_controller_cleanly()