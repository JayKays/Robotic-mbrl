

from mbrl.models.model_env import ModelEnv
from .trajectory_opt import TrajectoryOptimizerAgent, create_trajectory_optim_agent_for_model
from .core import Agent, complete_agent_cfg
import omegaconf
import numpy as np
import mbrl
import hydra

class EEAgent(Agent):
    """
    Implements a exploration vs exploitation based agent. 
    By utilizing the underlying dynamics model's ability to quantify uncertainty, the agent aim to 
    explore parts of the state/action space were predictions are uncertain.
    """

    def __init__(self, 
        exploration_agent: TrajectoryOptimizerAgent, 
        exploitation_agent: TrajectoryOptimizerAgent,
        epsilon: float = 0.1
    ):

        self.explorer = exploration_agent
        self.exploiter = exploitation_agent

        self.epsilon = epsilon
        self.explore = True
    
    def should_explore(self):
        "returns True if agent should perform an exploration step, and False for exploitation"
        if not self.explore:
            return False
        
        if np.random.random() < self.epsilon:
            return True
        else:
            return False
    
    def act(self, agent_obs):

        if self.should_explore():
            return self.explorer.act(agent_obs)
        else:
            return self.exploiter.act(agent_obs)


    def plan(self):
        if self.should_explore():
            return self.explorer.plan()
        else:
            return self.exploiter.plan()
    
    def disable_exploration(self):
        self.explore = False
    
    def enable_exploration(self):
        self.explore = True

def create_exploration_agent_for_model(
    model_env: mbrl.models.ModelExpEnv,
    agent_cfg: omegaconf.DictConfig, 
    num_particles: int = 1
):

    complete_agent_cfg(model_env, agent_cfg)
    agent = hydra.utils.instantiate(agent_cfg)

    def trajectory_eval_fn(initial_state, action_sequences):
        return model_env.evaluate_action_sequences(
            action_sequences, initial_state=initial_state, num_particles=num_particles, disagreement = True
        )

    agent.set_trajectory_eval_fn(trajectory_eval_fn)

    return agent


def create_ee_agent_for_model(
    model_env: mbrl.models.ModelExpEnv,
    agent_cfg: omegaconf.DictConfig,
    num_particles: int = 1
) -> EEAgent:

    exploration_agent = create_exploration_agent_for_model(model_env, agent_cfg, num_particles)
    cem_agent = create_trajectory_optim_agent_for_model(model_env, agent_cfg, num_particles)

    eps = agent_cfg.get("epsilon", None)
    if eps is None: print("No epsilon in config, default value used for Exploration threshold")

    agent = EEAgent(exploration_agent=exploration_agent, exploitation_agent= cem_agent, epsilon=eps or 0.1)

    return agent

    