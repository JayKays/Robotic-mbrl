# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional

import gym
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math

EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT

import time

def train(
    env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    work_dir = work_dir or os.getcwd()
    ext_data_dir = work_dir + "/.."*6 + f"/datasets/{cfg.overrides.env}"
    print(f"Results will be saved at {work_dir}.")

    if silent:
        logger = None
    else:
        logger = mbrl.util.Logger(work_dir)
        logger.register_group(
            mbrl.constants.RESULTS_LOG_NAME, EVAL_LOG_FORMAT, color="green"
        )

    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32

    if not os.path.isfile(ext_data_dir + "/replay_buffer.npz"):
        os.makedirs(ext_data_dir, exist_ok=True)

        data_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
        )
        num_trajectories = cfg.overrides.get("num_ext_trajectories", 20)
        for _ in range(num_trajectories):
            mbrl.util.common.rollout_agent_trajectories(
                env,
                cfg.algorithm.initial_exploration_steps,
                mbrl.planning.RandomAgent(env),
                {},
                replay_buffer=data_buffer,
            )
            data_buffer.save(ext_data_dir)

        print(f"Saved {num_trajectories} random trajectories to {os.path.abspath(ext_data_dir)}")

    ext_data = np.load(ext_data_dir + "/replay_buffer.npz")
    uncertainty_record = {"full": np.array([]), "trial": np.array([]), "ext": np.array([])}

    # -------- Create and populate initial env dataset --------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env),
        {},
        replay_buffer=replay_buffer,
    )
    replay_buffer.save(work_dir)

    # ---------------------------------------------------------
    # ---------- Create model environment and agent -----------
    model_env = mbrl.models.ModelExpEnv(
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=logger,
    )

    agent = mbrl.planning.create_agent_for_model(cfg.overrides.agent_type,
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
    )

    # times = np.empty((cfg.overrides.num_steps,))

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    current_trial = 0
    max_total_reward = -np.inf
    while env_steps < cfg.overrides.num_steps:
        obs = env.reset()
        agent.reset()
        done = False
        total_reward = 0.0
        steps_trial = 0
        while not done:
            # --------------- Model Training -----------------
            if env_steps % cfg.algorithm.freq_train_model == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )
                
                full_uncertainty = mbrl.models.util.estimate_uncertainty(model_env.dynamics_model, work_dir)
                trial_uncertainty = mbrl.models.util.estimate_uncertainty(model_env.dynamics_model, work_dir, idx=-cfg.overrides.trial_length)
                ext_uncertainty = model_env.update_epsilon(ext_data["obs"], ext_data["action"])

                mbrl.models.util.log_uncertainty(
                    work_dir, 
                    uncertainty_record, 
                    full=full_uncertainty.cpu(), 
                    trial= trial_uncertainty.cpu(), 
                    ext=ext_uncertainty.cpu()
                )

            # start = time.time()
            # --- Doing env step using the agent and adding to model dataset ---
            next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            env_steps += 1

            if debug_mode:
                print(f"Step {env_steps}: Reward {reward:.3f}.")

            if cfg.render and hasattr(env, "render"):
                env.render()

            # times[(env_steps-1)%times.shape[0]] = time.time() - start

        if logger is not None:
            logger.log_data(
                mbrl.constants.RESULTS_LOG_NAME,
                {"env_step": env_steps, "episode_reward": total_reward},
            )
        current_trial += 1
        if debug_mode:
            print(f"Trial: {current_trial }, reward: {total_reward}.")

        max_total_reward = max(max_total_reward, total_reward)

    # print(f"Average step time: {times.mean()}")
    return np.float32(max_total_reward)
