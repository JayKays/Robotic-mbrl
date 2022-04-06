
import pathlib
from black import out

import numpy as np
import torch

import mbrl.util
import mbrl.util.common
import mbrl.planning



def test_model(model_dir, num_trials=1, output_dir=None, reward_fn_ext=None, term_fn_ext=None):
    """
    Run a set number of trials with a trained model
    """

    model_path = pathlib.Path(model_dir)
    if output_dir is not None:
        output_path = pathlib.Path(output_dir)
        pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)

    cfg = mbrl.util.common.load_hydra_cfg(model_path)
    handler = mbrl.util.create_handler(cfg)

    env, term_fn_env, reward_fn_env = handler.make_env(cfg)

    reward_fn = reward_fn_ext or reward_fn_env
    term_fn = term_fn_ext or term_fn_env

    use_external_reward = reward_fn_ext is not None

    dynamics_model = mbrl.util.common.create_one_dim_tr_model(
        cfg,
        env.observation_space.shape,
        env.action_space.shape,
        model_dir=model_path,
    )

    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        env.observation_space.shape,
        env.action_space.shape,
    )

    torch_generator = torch.Generator(device=cfg.device)

    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, term_fn, reward_fn, generator=torch_generator
    )
    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
    )

    rewards = np.empty(num_trials)
    for trial in range(num_trials):
        obs = env.reset()
        agent.reset()
        done = False
        total_reward = 0.0
        steps_trial = 0

        while not done:

            next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            if use_external_reward:
                
                act = torch.tensor(replay_buffer.get_all().astuple()[1][replay_buffer.cur_idx-1])
                reward = reward_fn(act.unsqueeze(0), torch.tensor(next_obs).unsqueeze(0))

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            
        print(f"Trial: {trial}, Reward: {total_reward}")
        rewards[trial] = total_reward
    
    return rewards

def halfcheetah_desired_velocity(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * torch.exp(act.square().sum(dim=1))
    reward_run = 10*torch.exp(-0.3*torch.abs((next_obs[:, 0] - 8)))
    return (reward_run + reward_ctrl).view(-1, 1)

if __name__ == "__main__":

    cheetah_dirs = {
        "pets": "/home/jaykay/Robotic-mbrl/remote_copy/EE/pets_comparison/pets_halfcheetah/2022.03.18/165809",
        "random": "/home/jaykay/Robotic-mbrl/remote_copy/EE/cheetah_random_uncertainty/pets_halfcheetah/2022.03.22/174906",
        "policy": "/home/jaykay/Robotic-mbrl/remote_copy/EE/cheetah_policy_unc/pets_halfcheetah/2022.03.22/175027"
    }
    
    reacher_dirs = {
        "pets": "/home/jaykay/Robotic-mbrl/remote_copy/EE/pets_comparison/pets_reacher/2022.03.28/103539",
        "random": "/home/jaykay/Robotic-mbrl/remote_copy/EE/reacher_random_unc/pets_reacher/2022.03.25/172453",
        "policy": "/home/jaykay/Robotic-mbrl/remote_copy/EE/reacher_policy_unc/pets_reacher/2022.03.28/103733"
    }

    dirs = {
        "reacher": reacher_dirs,
        "cheetah": cheetah_dirs
    }

    env_key = "cheetah"
    filename = f"{env_key}_desired_vel"
    num_trials = 5
    print(f"----- Running {env_key} env ------")


    rewards = {}
    outdir = "model_testing"
    for key in dirs[env_key].keys():
        print(f"Testing {key} model")
        reward = test_model(dirs[env_key][key], num_trials=num_trials, output_dir=outdir, reward_fn_ext=halfcheetah_desired_velocity)
        rewards[key] = reward
    
        np.savez(f"model_testing/{filename}.npz", **rewards)
