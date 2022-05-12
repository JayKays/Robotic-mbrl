# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, Tuple, cast

import gym
import hydra.utils
import numpy as np
import omegaconf
import torch
import pathlib
import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac as pytorch_sac
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
import mbrl.util.env
from mbrl.planning.sac_wrapper import SACAgent
import matplotlib.pyplot as plt

MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]

def plot(traj, goal, k, f_ext, f, acc ):
    #k =k.astype(int)
    #print(k)
    for i in range (15):
        plt.subplot(5,3,i+1)
        if i in (0,1,2):
            plt.plot(100*traj[:, i])
            plt.plot(100*goal[:, i])
        elif i  in (3,4,5):
            plt.plot(k[:, i-3])
        elif i in (6,7,8):
            plt.plot(f_ext[:, i-6])
        elif i in (9,10,11):
            plt.plot(f[:, i - 9])
        else:
            plt.plot(acc[:, i - 12])
    plt.show()

def test(agent_dir, model_dir,
    env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
    num_episodes: int,
    video_recorder: pytorch_sac.VideoRecorder,
    test_model_predictions: bool = False
) -> float:

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    ext_actions = cfg.overrides.get("uncontrolled_states", False)
    print("ext_act: ", ext_actions)
    if (ext_actions):
        ext_act_shape = np.shape(env.get_external_states())[0]
        act_shape = list(act_shape)
        act_shape[0] += ext_act_shape
        act_shape = tuple(act_shape)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    avg_episode_reward = 0
    agent_path = pathlib.Path(agent_dir)
    model_path = pathlib.Path(model_dir)
    agent = mbrl.planning.load_agent(agent_path, env)

    dynamics_model = mbrl.util.common.create_one_dim_tr_model(
        cfg,
        obs_shape,
        act_shape,
        model_dir=model_path,
    )
    model_env = mbrl.models.ModelEnv(
       env, dynamics_model, termination_fn, None, generator=torch_generator
    )


    env.set_render(True)
    for episode in range(num_episodes):
        obs = env.reset()
        model_state = model_env.reset(
            initial_obs_batch=cast(np.ndarray, np.tile(obs.reshape(1, -1), (5, 1))),  # obs.reshape(-1,1),#
            return_as_np=True,
        )
        video_recorder.init(enabled=(episode == 0))
        done = False
        episode_reward = 0
        steps_trial = 0
        goal = []
        stiffness = []
        observations = []
        acceleration = []
        cartesian_force = []
        ext_force = []
        total_error = 0

        while not done:
            action = agent.act(cast(np.ndarray, obs.reshape(1,-1)), None, None,batched= True)
            model_state = model_env.reset(
                initial_obs_batch=cast(np.ndarray, np.tile(obs.reshape(1, -1), (5, 1))),  # obs.reshape(-1,1),#
                return_as_np=True,
            )
            #print(action)
            if test_model_predictions:
                pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
                    torch.from_numpy(np.tile(action, (5,1))), model_state, sample=True
                )
            g, K = env.get_goal()
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            #print( np.sum(np.abs(pred_next_obs -obs)[:,:3] ))
            #print((pred_next_obs[0, :3] - obs[:3]))
            total_error += np.sum(np.abs((pred_next_obs[0, 6:9] - obs[6:9])))

            steps_trial += 1
            extra_obs = env.get_extra_obs()
            video_recorder.record(env)
            episode_reward += reward
            observations.append(extra_obs['pose'][0:3])
            goal.append(g)
            stiffness.append(np.diag(extra_obs['K'])[0:3])
            # print(np.diag(extra_obs['K'])[0:3])
            acceleration.append(extra_obs['acceleration'])
            cartesian_force.append(extra_obs['cartesian_force'][0:3])
            ext_force.append(extra_obs['FT'][0:3])
        avg_episode_reward += episode_reward
        if done:
            print(total_error)
            plot(np.array(observations), np.array(goal), np.array(stiffness), \
                 np.array(ext_force), np.array(cartesian_force), np.array(acceleration))
    return avg_episode_reward / num_episodes


@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    dir = "/home/akhil/PhD/RoL/Robotic-mbrl/mbrl-lib/exp/mbpo/default/panda_reacher_cartesian_env/2022.04.01/193537"
    agent_dir = dir
    model_dir = dir
    dataset_dir = dir
    results_dir = dir
    model_path = pathlib.Path(model_dir)
    ext_conf = False
    if not ext_conf:
        cfg = mbrl.util.common.load_hydra_cfg(model_path)
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    num_episodes = 100

    save_video = cfg.get("save_video", False)
    video_recorder = pytorch_sac.VideoRecorder(dir if save_video else None)

    test(agent_dir, model_dir, env, term_fn, reward_fn, cfg , num_episodes, video_recorder,  test_model_predictions=True)


if __name__ == "__main__":
    run()
