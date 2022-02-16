import os
import time
import pathlib
import mujoco_py
import numpy as np
import gym
from gym import logger, spaces
# sys.path.append("/home/akhil/PhD/RoL/mujoco_panda-master")
from mujoco_panda import PandaArm
from mujoco_panda.utils.viewer_utils import render_frame
from mujoco_panda.controllers.torque_based_controllers import HuangVIC
from mujoco_panda.controllers.torque_based_controllers import VIC
from mujoco_panda.utils import VIC_func as func
from mujoco_panda.controllers.torque_based_controllers import VIC_config as cfg
import time
import random
import quaternion
import matplotlib.pyplot as plt


class UltrasoundEnv(gym.Env):

    def __init__(self, position_as_action = False, controller = "VIC_Huang", control_rate = None, log_dir = None, max_num_it = cfg.MAX_NUM_IT, render=True):

        MODEL_PATH = os.environ['MJ_PANDA_PATH'] + '/mujoco_panda/models/'
        #self.robot = PandaArm(model_path=MODEL_PATH + 'panda_block_table.xml',
                             #render=True, compensate_gravity=False, smooth_ft_sensor=True)
        self.robot = PandaArm(model_path=MODEL_PATH + 'panda_ultrasound.xml',
                             render=render, compensate_gravity=False, smooth_ft_sensor=True)
        if mujoco_py.functions.mj_isPyramidal(self.robot.model):
            print("Type of friction cone is pyramidal")
        else:
            print("Type of friction cone is eliptical")
        # self.init_jpos = np.array([-0.0242187,-0.26637015,-0.23036408,-1.99276073,-0.05414588,1.72812007, 0.52766157])
        #self.init_jpos = np.array([0., -0.7, 0, -2.356, 0, 1.656, 0.785])
        self.init_jpos = np.array([-2.03e-03, -8.42e-01, 1.09e-03, -2.26e+00, 5.80e-04, 1.41e+00, 8.24e-01])
        self.robot.hard_set_joint_positions(self.init_jpos)
        self.robot.sim_step()
        time.sleep(1.0)
        self.position_as_action  = position_as_action

        ''''
        Choose a controller type from VIC_Huang and VIC.
        VIC: the standard variable impedance controller where the delta K values are fed as actions
        VIC_Huang: VIC controller with an adaptive law, gamma parameter of the adaptive law is fed as action
        '''

        if controller == "VIC_Huang":
            self.controller = HuangVIC(self.robot, max_num_it=max_num_it, control_rate = control_rate, )
            if position_as_action:
                self.action_space = spaces.Box(low=np.array([cfg.GAMMA_K_LOWER, cfg.DELTA_Z_LOWER]), \
                                           high=np.array([cfg.GAMMA_K_UPPER, cfg.DELTA_Z_UPPER]))
            else:
                self.action_space = spaces.Box(low=np.array([cfg.GAMMA_K_LOWER]), \
                                     high=np.array([cfg.GAMMA_K_UPPER]))
        elif controller == "VIC":
            print("VIC controller")
            self.controller = VIC(self.robot,max_num_it=max_num_it, )
            if position_as_action:
                self.action_space = spaces.Box(low=np.array([cfg.DELTA_K_LOWER, cfg.DELTA_Z_LOWER]), \
                                           high=np.array([cfg.DELTA_K_UPPER, cfg.DELTA_Z_UPPER]))
            else:
                self.action_space = spaces.Box(low=np.array([cfg.DELTA_K_LOWER]), \
                                     high=np.array([cfg.DELTA_K_UPPER]))
        else:
            raise ValueError("Invalid contorller type")
            
        self.observation_space = spaces.Box(
            low=np.array([cfg.LOWER_Fz, cfg.LOWER_Z_ERROR, cfg.LOWER_Kz, cfg.LOWER_Fz]), \
            high=np.array([cfg.UPPER_Fz, cfg.UPPER_Z_ERROR, cfg.UPPER_Kz, cfg.UPPER_Fz]))

        self.done = False
        self.max_num_it = self.controller.max_num_it
        self.F_offset = np.zeros(6)

        # set desired pose/force trajectory
        self.f_d = func.generate_Fd_steep(self.max_num_it, cfg.Fd, cfg.T)
        self.f_d[2, :] = -cfg.Fd
        # set desired pose/force trajectory
        self.f_d = func.generate_Fd_steep(self.max_num_it, cfg.Fd, cfg.T)
        self.f_d[2, :] = cfg.Fd
        self.goal_ori = np.asarray(self.robot.ee_pose()[1])
        #self.goal_ori = np.array([0, -1, -3.82e-01,  0])
        self.x_d_ddot, self.x_d_dot, self.x_d = func.generate_desired_trajectory_tc(self.robot, self.max_num_it, cfg.T,
                                                                                    move_in_x=True)
        plt.plot(self.x_d[1, :])
        plt.show()
        self.i = 0
        self.log_dir = os.getcwd() if cfg.LOG and log_dir is None else log_dir 
        self.log_dict = {}

        self.torque_hist = np.zeros((self.max_num_it, 7))
        self.rawF_hist = np.zeros((self.max_num_it, 6))
        self.F_hist = np.zeros_like(self.rawF_hist)

        # print("mass: ", p.mass_matrix())

    # activate controller (simulation step and controller thread now running)

    #design teh
    def force_mean_filter(self, filter_type="mean", window_size= 5):
        if self.i < window_size:
            return (np.sum(self.F_history, axis=0)/(self.i+1))
        else:
            return (np.sum(self.F_history[self.i-window_size+1:,:], axis=0)/window_size)


    def get_obs(self):
        obs_dict = self.controller.state_dict
        return np.array(
            [obs_dict["FT"][2] , \
             obs_dict["pose"][2] - self.x_d[2, self.i], \
             obs_dict["K"][2,2], \
             self.f_d[2, self.i]])

    def get_reward(self):
        obs_dict = self.controller.state_dict
        force_reward = np.exp(-np.square(3*(obs_dict["FT"][2] - self.f_d[2, self.i] - self.F_offset[2])))
        x_reward = np.exp(-np.square(300 * (obs_dict["pose"][0] - self.x_d[0, self.i])))
        y_reward = np.exp(-np.square(300 * (obs_dict["pose"][1] - self.x_d[1, self.i])))
        reward = 1*force_reward + 0*x_reward + 0*y_reward
        return reward


    def step(self, action):
        #if self.position_as_action:
        self.controller.set_goal(action, self.x_d[:, self.i], self.goal_ori, self.x_d_dot[:, self.i], \
                                 self.x_d_ddot[:, self.i], goal_force=self.f_d[:, self.i])
        self.controller._send_cmd()
        self.state = self.get_obs()

        self.i += 1
        # print(f"Env i: {self.i}, controller timestep: {self.controller.timestep}")
        self.controller.get_robot_states()
        self.state = self.get_obs()
        if (self.i>= self.max_num_it-1)  or (np.abs(self.state[0]) > 100):
            done = True#(self.iteration >= self.max_num_it)
        else:
            done = False
        reward = self.get_reward()
        info = {}
        # print(f"Done: {done}, i: {self.i}")
        
        self.torque_hist[self.i,:] = self.controller._cmd
        self.F_hist[self.i,:] = self.controller.state_dict["FT"]
        self.rawF_hist[self.i,:] = self.controller.state_dict["FT_raw"]

        if done and cfg.LOG:
            print("DONE, Saving state log")
            self.update_log()
            self.save_log()
        
        #print(self.robot.get_ft_reading())
        #print("torque ", self.controller._cmd)
        #print("pose ", self.x_d[:, self.i-1] - self.robot.ee_pose()[0])
        #print("smoothed FT reading: ", self.robot.get_ft_reading(pr=True)[0])
        return self.state, reward, done, info

    def reset(self):
        # if cfg.LOG:
        #     self.update_log()
        #     self.save_log()

        print("resetting the controller")
        self.controller.reset()
        print("resetting envs")
        index = 0  # np.random.randint(0, (0.9 * self.max_num_it))
        self.i = index
        self.robot.hard_set_joint_positions(self.init_jpos)
        return self.get_obs()

    def render(self, mode="human"):
        self.robot.render()

    def update_log(self):
        if not bool(self.log_dict): #Checks if log_dict is empty
            self.log_dict["F"] = self.F_hist
            self.log_dict["torque"] = self.torque_hist
            self.log_dict["F_raw"] = self.rawF_hist
            # self.log_dict["x"] = self.controller.x_history.T
            # self.log_dict["x_dot"] = self.controller.x_dot_history.T
            # self.log_dict["p"] = self.controller.p_hist.T
            # self.log_dict["Fz"] = self.controller.Fz_history
            # self.log_dict["h_e"] = self.controller.h_e_hist.T
            # self.log_dict["Kp"] = self.controller.Kp_pos_hist
            # self.log_dict["Kp_z"] = self.controller.Kp_z_hist
            # self.log_dict["Kd_z"] = self.controller.Kd_z_hist
        else:
            self.log_dict["F"] = np.append(self.log_dict["F"], self.F_hist[:self.i,:], axis = 0)
            self.log_dict["F_raw"] = np.append(self.log_dict["F_raw"], self.rawF_hist[:self.i,:], axis = 0)
            self.log_dict["torque"] = np.append(self.log_dict["torque"], self.torque_hist[:self.i,:], axis = 0)
            # self.log_dict["x"] = np.append(self.log_dict["x"], self.controller.x_history.T[:self.controller.timestep,:], axis = 0)
            # self.log_dict["x_dot"] = np.append(self.log_dict["x_dot"], self.controller.x_dot_history.T[:self.controller.timestep,:], axis = 0)
            # self.log_dict["p"] = np.append(self.log_dict["p"], self.controller.p_hist.T[:self.controller.timestep,:], axis = 0)
            # self.log_dict["Fz"] = np.append(self.log_dict["Fz"], self.controller.Fz_history[:self.controller.timestep], axis = 0)
            # self.log_dict["h_e"] = np.append(self.log_dict["h_e"], self.controller.h_e_hist.T[:self.controller.timestep,:], axis = 0)
            # self.log_dict["Kp"] = np.append(self.log_dict["Kp"], self.controller.Kp_pos_hist[:self.controller.timestep], axis = 0)
            # self.log_dict["Kp_z"] = np.append(self.log_dict["Kp_z"], self.controller.Kp_z_hist[:self.controller.timestep], axis = 0)
            # self.log_dict["Kd_z"] = np.append(self.log_dict["Kd_z"], self.controller.Kd_z_hist[:self.controller.timestep], axis = 0)

    def save_log(self):
        if self.log_dir is not None and cfg.LOG:
            path = pathlib.Path(self.log_dir) / "controller_state_dict.npz"
            np.savez(path, **self.log_dict)


def make_ultrasound_env(env_cfg):

    num_it = env_cfg.overrides.get("trial_length", cfg.MAX_NUM_IT)
    controller = env_cfg.overrides.get("controller", "VIC_Huang")
    control_rate = env_cfg.overrides.get("control_rate", cfg.PUBLISH_RATE)
    
    render = env_cfg.get("render", True)
    
    env = UltrasoundEnv(max_num_it=num_it, controller=controller, control_rate=control_rate, render = render)

    return env

if __name__ == "__main__":

    VIC_env = UltrasoundEnv(controller = "VIC_Huang") #
    #gym.make("gym_robotic_ultrasound:ultrasound-v0")
    curr_ee, curr_ori = VIC_env.robot.ee_pose()
    print(VIC_env.robot.ee_pose()[1])
    # --------------------------------------

    VIC_env.controller.set_active(True)
    now_r = time.time()
    i = 0
    count = 0
    Data = []
    VIC_env.reset()
    for i in range(1):
        VIC_env.robot.render()
        print(i)
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
            action = 0.00001# 10*(1-2*np.random.random())#0#.000001#np.array([0.0001, 0.00001])  # np.random.uniform(1.e-6, 0.01, 2)
            s,r,_,_ = VIC_env.step(action)
            print("reward: ", r)
            #print(s)
            i += 1
            if i == 4999:
                break
        VIC_env.robot.render()  # render the visualisation

    plt.plot(VIC_env.controller.F_history[:,2])
    plt.show()
    # input("Trajectory complete. Hit Enter to deactivate controller")
    VIC_env.controller.set_active(False)
    VIC_env.controller.stop_controller_cleanly()