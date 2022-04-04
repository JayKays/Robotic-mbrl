# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional

import gym
import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math

import argparse
import pathlib
from typing import List
import hydra
import numpy as np
import omegaconf
import torch

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.algorithms.planet as planet
import mbrl.algorithms.bets as bets
import mbrl.algorithms.exploration_pets as EE
import mbrl.util.env
import matplotlib.pyplot as plt
EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT

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

def test( model_dir, dataset_dir, output_dir,
    env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------

    debug_mode = cfg.get("debug_mode", True)

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

    work_dir = work_dir or os.getcwd()
    print(f"Results will be saved at {work_dir}.")

    if silent:
        logger = None
    else:
        logger = mbrl.util.Logger(work_dir)
        logger.register_group(
            mbrl.constants.RESULTS_LOG_NAME, EVAL_LOG_FORMAT, color="green"
        )

    model_path = pathlib.Path(model_dir)
    output_path = pathlib.Path(output_dir)
    pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)

    #cfg = mbrl.util.common.load_hydra_cfg(model_path)
    handler = mbrl.util.create_handler(cfg)

    #env, term_fn, reward_fn = handler.make_env(cfg)
    #reward_fn = reward_fn

    dynamics_model = mbrl.util.common.create_one_dim_tr_model(
        cfg,
        obs_shape,
        act_shape,
        model_dir=model_path,
    )
    # -------- Create and populate initial env dataset --------
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )

    # ---------------------------------------------------------
    # ---------- Create model environment and agent -----------
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
    )
    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
    )
    #print(cfg.algorithm.agent)

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    current_trial = 0
    max_total_reward = -np.inf
    env.set_render(True)
    while env_steps < cfg.overrides.num_steps:
        obs = env.reset()
        agent.reset()
        done = False
        total_reward = 0.0
        steps_trial = 0
        goal = []
        stiffness = []
        observations = []
        acceleration = []
        cartesian_force = []
        ext_force = []
        #print(env_Steps)
        #print("hello")
        while not done:
            # --------------- Model Training -----------------
            '''if env_steps % cfg.algorithm.freq_train_model == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )'''
            g, K = env.get_goal()
            # --- Doing env step using the agent and adding to model dataset ---
            next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, ext_actions, agent, {}, replay_buffer
            )

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            env_steps += 1
            extra_obs = env.get_extra_obs()
            #print(done)
            if done:
                plot(np.array(observations), np.array(goal), np.array(stiffness), \
                     np.array(ext_force), np.array(cartesian_force), np.array(acceleration))
                observations = []
                goal = []
                stiffness = []
                acceleration = []
                cartesian_force = []
                ext_force = []
            else:
                observations.append(extra_obs['pose'][0:3])
                goal.append(g)
                stiffness.append(np.diag(extra_obs['K'])[0:3])
                # print(np.diag(extra_obs['K'])[0:3])
                acceleration.append(extra_obs['acceleration'])
                cartesian_force.append(extra_obs['cartesian_force'][0:3])
                ext_force.append(extra_obs['FT'][0:3])
            if debug_mode:
                print(f"Step {env_steps}: Reward {reward:.3f}.")

            #if cfg.render and hasattr(env, "render"):
                #env.render()

        if logger is not None:
            logger.log_data(
                mbrl.constants.RESULTS_LOG_NAME,
                {"env_step": env_steps, "episode_reward": total_reward},
            )
        current_trial += 1
        if debug_mode:
            print(f"Trial: {current_trial }, reward: {total_reward}.")

        max_total_reward = max(max_total_reward, total_reward)

    return np.float32(max_total_reward)

@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    dir = "/home/akhil/PhD/RoL/Robotic-mbrl/mbrl-lib/exp/pets/default/panda_reacher_cartesian_env/2022.04.04/171120"
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
    test(model_dir, dataset_dir, results_dir,env, term_fn, reward_fn, cfg)


if __name__ == "__main__":
    run()

