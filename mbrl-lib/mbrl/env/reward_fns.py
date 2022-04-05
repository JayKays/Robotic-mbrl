# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from . import termination_fns


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)


def cartpole_pets(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    goal_pos = torch.tensor([0.0, 0.6]).to(next_obs.device)
    x0 = next_obs[:, :1]
    theta = next_obs[:, 1:2]
    ee_pos = torch.cat([x0 - 0.6 * theta.sin(), -0.6 * theta.cos()], dim=1)
    obs_cost = torch.exp(-torch.sum((ee_pos - goal_pos) ** 2, dim=1) / (0.6 ** 2))
    act_cost = -0.01 * torch.sum(act ** 2, dim=1)
    return (obs_cost + act_cost).view(-1, 1)


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    return (~termination_fns.inverted_pendulum(act, next_obs)).float().view(-1, 1)


def halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()
    return (reward_run + reward_ctrl).view(-1, 1)


def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)
    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]
    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act ** 2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1, 1)

def ultrasound(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    #for 5 states
    #assert len(next_obs.shape) == len(act.shape) == 2
    #reward_ctrl = -0.1 * act.square().sum(dim=1)
    Fd = torch.tensor([100]).to(next_obs.device)
    F_reward = (next_obs[:, 2] - Fd) ** 2
    #F_reward = torch.exp(-torch.sum(3*F_reward ** 2, dim=1))
    pose_reward = torch.sum((100 * next_obs[:, 3:5] ** 2), dim=1)
    act_cost = 0 * (act ** 2).sum(axis=1)
    reward = -(0*F_reward + 1*pose_reward + 0*act_cost)

    #print(reward.view(-1, 1))
    return (reward).view(-1, 1)


def HFMC1(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    Fd= torch.tensor([3]).to(next_obs.device)
    f_z = next_obs[:, 0]
    print("Fd = ", Fd, "fz = ", f_z)
    delta_f = (Fd - f_z).abs().sum(axis=1)
    sq_error = torch.square(delta_f)
    return -(sq_error).view(-1, 1)

def panda_traj_tracking(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    #for 5 states
    #assert len(next_obs.shape) == len(act.shape) == 2
    #reward_ctrl = -0.1 * act.square().sum(dim=1)
    obs_cost = torch.sum((100 * next_obs[:, 0:3] ** 2), dim=1)
    #obs_cost = 10*next_obs[:,0:3].abs().sum(axis=1)
    act_cost = 0 * (act ** 2).sum(axis=1)
    #reward = torch.exp(-torch.sum((10*next_obs[:,0:3] ** 2), dim=1))#np.exp(-np.square(next_obs[3]))
    reward  = -(1*obs_cost + 1*act_cost)
    return reward.view(-1, 1)

def panda_reacher_cartesian(act: torch.Tensor, next_obs: torch.Tensor, pre_obs: torch.Tensor,  pre_act: torch.Tensor ) -> torch.Tensor:
    #for 5 states
    #assert len(next_obs.shape) == len(act.shape) == 2
    #reward_ctrl = -0.1 * act.square().sum(dim=1)
    #print(act.size())
    #obs_cost = torch.sum((100 * next_obs[:, 0:3] ** 2), dim=1)
    obs_cost = 1 * torch.sum(torch.square(100*(act[:, 3:6] + pre_obs[:, 0:3] - next_obs[:, 0:3])), dim= 1)
    #print(obs_cost)
    #obs_cost = 10*next_obs[:,0:3].abs().sum(axis=1)
    #print(next_obs[:, 3:6], pre_obs)
    acc_cost = torch.sum((100*(next_obs[:, 3:6]-pre_obs[:,3:6])**2), dim= 1)
    #print(next_obs[:, 3:6]-pre_obs[:,3:6])
    #print("acc_cost ", acc_cost)
    #print("obs_cost ", obs_cost)
    #print(pre_obs.size())
    if pre_act is not None:
        smooth_cost = torch.sum((1*(act[:, 0:3]-pre_act[:,0:3])**2), dim= 1)
        #print(smooth_cost,obs_cost)
    else:
        smooth_cost = 0
    act_cost = 1 * (act[:,0:3] ** 2).sum(axis=1)
    reward  = -(1*obs_cost + 1*act_cost + 0 * acc_cost + 0*smooth_cost)
    #print("reward:", reward)
    return reward.view(-1, 1)

def panda_tray(act: torch.Tensor, next_obs: torch.Tensor, pre_obs: torch.Tensor,  pre_act: torch.Tensor ) -> torch.Tensor:
    obs_cost = 1 * torch.sum(torch.square(1000*(act[:, 3:6] + pre_obs[:, 0:3] - next_obs[:, 0:3])), dim= 1)
    acc_cost = torch.sum((1000*(next_obs[:, 3:6]-pre_obs[:,3:6])**2), dim= 1)
    if pre_act is not None:
        smooth_cost = torch.sum((1*(act[:, 0:3]-pre_act[:,0:3])**2), dim= 1)
    else:
        smooth_cost = 0
    act_cost = 1 * (act[:,0:3] ** 2).sum(axis=1)
    reward  = -(1*obs_cost + 1*act_cost + 0 * acc_cost + 0*smooth_cost)
    #print("reward:", reward)
    return reward.view(-1, 1)

def panda_pusher(act: torch.Tensor, next_obs: torch.Tensor, pre_obs: torch.Tensor,  pre_act: torch.Tensor ) -> torch.Tensor:
    obs_cost = 1 * torch.sum(torch.square(1000*(act[:, 3:6] + pre_obs[:, 0:3] - next_obs[:, 0:3])), dim= 1)
    acc_cost = torch.sum((1000*(next_obs[:, 3:6]-pre_obs[:,3:6])**2), dim= 1)
    if pre_act is not None:
        smooth_cost = torch.sum((1*(act[:, 0:3]-pre_act[:,0:3])**2), dim= 1)
    else:
        smooth_cost = 0
    act_cost = 1 * (act[:,0:3] ** 2).sum(axis=1)
    reward  = -(1*obs_cost + 1*act_cost + 0 * acc_cost + 0*smooth_cost)
    #print("reward:", reward)
    return reward.view(-1, 1)