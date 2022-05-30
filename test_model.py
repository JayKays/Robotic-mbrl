

import pathlib

import numpy as np
import torch

import mbrl.util
import mbrl.util.common
import mbrl.planning
from gym import wrappers


def test_model(model_dir, num_trials=1, reward_fn_ext=None, term_fn_ext=None, using_checkpoint = False ,render = False):
    """
    Runs a given number of trials with a trained agent loaded from model_dir.
    
    Includes the option to use external reward and termination functions other than the
    ones from the environment in order to test an agents performance on a different task
    with the same dynamics.
    """
    
    if using_checkpoint:
        model_path = pathlib.Path("/".join(model_dir.split("/")[:-2]))
    else:
        model_path = pathlib.Path(model_dir)

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
    dynamics_model.load(model_dir)
    for trial in range(num_trials):
        obs = env.reset()
        agent.reset()
        done = False
        total_reward = 0.0
        steps_trial = 0
        if render: env.render()
        while not done:

            next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            if use_external_reward:
                
                act = torch.tensor(replay_buffer.get_all().astuple()[1][replay_buffer.cur_idx-1])
                reward = np.squeeze(reward_fn(act.unsqueeze(0), torch.tensor(next_obs).unsqueeze(0)).numpy())

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            if render: env.render()
            
        print(f"\t Trial: {trial}, Reward: {total_reward}")
        rewards[trial] = total_reward
    if render:
        input("Press Enter to stop simulation")
        env.close()
    return rewards

def test_multiple_seeds(result_dir, num_trials, seeds, rew_fn=None, term_fn=None):
    """
    Tests multiple seeds of the same model and environment. The different models
    must be found in "result_dir/seed_x" folders, where x corresponds to the input seeds.
    """

    results = np.zeros((len(seeds), num_trials))
    for it, seed in enumerate(seeds):
        print(f"Seed {seed}")
        model_dir = result_dir + f"seed_{seed}"

        results[it, :] = test_model(model_dir, num_trials, rew_fn, term_fn)

    return results

def test_check_points(result_dir, num_trials, seeds, checkpoints = None, rew_fn=None, term_fn=None):

    """
    Tests performance of model checkpoints and potentially multiple seeds. Models must be found in
    "results_dir/seed_x/check_points/
    """

    if checkpoints is None:
        print("No checkpoints given, testing final model")
        return test_multiple_seeds(result_dir, num_trials, seeds, rew_fn, term_fn)

    result_dict = {}
    for checkpoint in checkpoints:
        results = np.zeros((len(seeds), num_trials))
        print(f"Checkpoint {checkpoint}")
        for it, seed in enumerate(seeds):
            model_dir = result_dir + f"seed_{seed}/"
            
            if checkpoint != "final":
                model_dir += f"check_points/{checkpoint}"

            results[it, :] = test_model(model_dir, num_trials, rew_fn, term_fn, using_checkpoint=checkpoint!="final")

        result_dict[str(checkpoint)] = results

    return result_dict



def halfcheetah_desired_velocity(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * torch.exp(act.square().sum(dim=1))
    reward_run = 10*torch.exp(-0.3*torch.abs((next_obs[:, 0] - 8)))
    return (reward_run + reward_ctrl).view(-1, 1)

def halfcheetah_backwards(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = -1*next_obs[:, 0]
    return (reward_run + reward_ctrl).view(-1, 1)

def walker_backwards(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    vel = next_obs[:,8]
    
    alive_reward = 1.0
    reward_run = -vel
    act_cost = 1e-3 * torch.sum(act ** 2, axis = 1)

    reward = reward_run + alive_reward - act_cost

    return reward.view(-1,1)

if __name__ == "__main__":


    results_dir = "/test_results/"
    exps = ["pets","random_uncertainty", "policy_uncertainty"]
    env = "reacher"

    # filename = f"checkpoints_{env}"
    filename = f"{env}_test"
    num_trials = 1
    print(f"----- Running {env} env ------")

    rewards = {}
    outdir = "model_testing"
    for exp in exps:
        print(f"Testing {exp} model(s)")
        reward = test_model(results_dir + env + f"/{exp}/seed_0", 
            num_trials=num_trials,
            render=True,
            # reward_fn_ext=halfcheetah_backwards,
            # reward_fn_ext=walker_backwards,
            # reward_fn_ext=halfcheetah_desired_velocity
        )
        rewards[exp] = reward

        # np.savez(f"model_testing/{filename}.npz", **rewards)
