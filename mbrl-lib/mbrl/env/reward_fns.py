# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from . import termination_fns
from .util import get_reacher_EE_pos

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

def reacher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    goal_pos = next_obs[:,7:10]
    ee_pos = get_reacher_EE_pos(next_obs)

    obs_cost = ((goal_pos - ee_pos)**2).sum(axis=1)
    act_cost = 0.01 * (act ** 2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1,1)


def walker2d(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:,0]
    angle = next_obs[:,1]
    vel = next_obs[:,8]

    alive = (height > 0.8) * (height < 2.0) * (angle > -1.0) * (angle < 1.0)
    
    alive_reward = 1.0
    reward_run = vel
    act_cost = 1e-3 * torch.sum(act ** 2, axis = 1)

    reward = reward_run + alive_reward - act_cost

    return reward.view(-1,1)


def HFMC(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    #for 5 states
    #assert len(next_obs.shape) == len(act.shape) == 2
    #reward_ctrl = -0.1 * act.square().sum(dim=1)
    Fd = torch.tensor([3]).to(next_obs.device)
    F_reward = next_obs[:, :1] - Fd
    F_reward = torch.exp(-torch.sum(3*F_reward ** 2, dim=1))
    x_reward = torch.exp(-torch.sum((300*next_obs[:,3:4] ** 2), dim=1))#np.exp(-np.square(next_obs[3]))
    y_reward = torch.exp(-torch.sum((300*next_obs[:,4:5] ** 2), dim=1))#np.exp(-np.square(self.state[4]))
    #print(reward.size())
    reward = 0.5*F_reward + 0.4*x_reward + 0.1*y_reward

    #print(reward.view(-1, 1))
    return (reward).view(-1, 1)


def HFMC1(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    Fd= torch.tensor([3]).to(next_obs.device)
    f_z = next_obs[:, 0]
    print("Fd = ", Fd, "fz = ", f_z)
    delta_f = (Fd - f_z).abs().sum(axis=1)
    sq_error = torch.square(delta_f)
    return -(sq_error).view(-1, 1)