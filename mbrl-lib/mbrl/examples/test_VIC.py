import gym
from franka_interface import ArmInterface
import numpy as np
import rospy
from gym import logger, spaces
from gym.utils import seeding
import sys
sys.path.append("/home/akhil/catkin_ws/src/Compliant_control/gym-panda/VIC")
from gym_panda.envs import VIC_func as func
# from gym_panda.envs.VIC_ObsSpace import ObservationSpace
from gym_panda.envs import VIC_config as cfg
import matplotlib.pyplot as plt
import time
import random

np.random.seed(0)
sim = True
rospy.init_node("VIC")
rate = rospy.Rate(cfg.PUBLISH_RATE)
robot = ArmInterface()
joint_names = robot.joint_names()
print(joint_names)
max_num_it = 100
f_d_ddot,f_d_dot, f_d= func.generate_Fd_constant(max_num_it)#func.generate_Fd_steep(max_num_it,cfg.T,cfg.Fd)
goal_ori = robot.endpoint_pose()['orientation'] # goal orientation = current (initial) orientation [remains the same the entire duration of the run]
r_d_ddot, r_d_dot, p_d  = func.generate_desired_trajectory(robot,max_num_it,cfg.T,sim,move_in_x=True)
#print(p_d)
x_hist = np.zeros((5,max_num_it))
iteration = 0
time_per_iteration = np.zeros(max_num_it)

F, h_e, ori, p, x, J, v, joint_v = robot.get_HFMC_states(x_hist,iteration,time_per_iteration,goal_ori,joint_names,sim)
F_offset = F
p_z_init = p[2]
print(v, joint_v)

#