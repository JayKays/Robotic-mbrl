


from ast import withitem
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import torch

import mbrl.types
import mbrl.util.math
from . import Model, ModelEnv

class ModelExpEnv(ModelEnv):
    """
    Extends the ModelEnv class to allow for epxloration via disagreement,
    where the rewards are based on the disagreement between each ensemble model
    to encourage exploration of the state action space were the dynamics model
    is uncertain.

    Functionality is equivalent to the parent class ModelEnv when exploration = False

    espilon: weighting parameter between exploration and exploitation rewards
    """

    def __init__(
        self,
        env: gym.Env,
        model: Model,
        termination_fn: mbrl.types.TermFnType,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
        generator: Optional[torch.Generator] = None,
        exploration: bool = True,
        init_epsilon: float = .5,
        uncertainty_map: str = None,
        reward_map: str = None
    ):

        super().__init__(env, model, termination_fn, reward_fn, generator)
        self.exploration = exploration
        self.epsilon = init_epsilon
        self.max_uncertainty = -1

        if reward_map is not None:
            print(reward_map)
            self._reward_map = getattr(mbrl.util.math, reward_map, self._reward_map)
        
        if uncertainty_map is not None:
            self._uncertainty_map = getattr(mbrl.util.math, uncertainty_map, self._uncertainty_map)
        
    def step(
        self,
        actions: mbrl.types.TensorType,
        model_state: Dict[str, torch.Tensor],
        sample: bool = False,
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:

        with torch.no_grad():
            next_observs, exploit_rewards, dones, next_model_state = super().step(actions, model_state, sample)
            
            if self.exploration:
                model_in = self.dynamics_model._get_model_input(model_state["obs"], actions)
                means, _ = self.dynamics_model.forward(model_in, use_propagation=False)
                explore_rewards = torch.var(means, dim=0).mean(dim=1, keepdim=True)
                explore_rewards = self._uncertainty_map(explore_rewards)
                rewards = self.epsilon*explore_rewards + (1 - self.epsilon)*self._reward_map(exploit_rewards)

            else:
                rewards = exploit_rewards
        
        return next_observs, rewards, dones, next_model_state

    def evaluate_action_sequences(self, action_sequences: torch.Tensor, initial_state: np.ndarray, num_particles: int) -> torch.Tensor:

        value =  super().evaluate_action_sequences(action_sequences, initial_state, num_particles)
        
        return value

    def _toggle_exploration(self):
        self.exploration = not self.exploration
    
    def _uncertainty_map(self, value):
        # return torch.tanh(1e-3*value)
        return torch.log1p(value)
    
    def _reward_map(self, value):
        return value
    
    def set_exploration(self, exploration: bool):
        self.exploration = exploration
    
    def set_espilon(self, epsilon: float):
        
        #Ensures 0 <= epsilon <= 1        
        self.epsilon = min(max(epsilon,0),1)

        print(f"New epsilon: {self.epsilon}")
    
    def update_epsilon(self, obs, act):
        "Updates epsilon values based on uncertainty from given input"
        with torch.no_grad():
            model_in = self.dynamics_model._get_model_input(obs, act)
            means, _ = self.dynamics_model.forward(model_in, use_propagation=False)
            uncertainty = torch.var(means, dim=0).mean()
            
            if self.max_uncertainty < uncertainty:
                print(f"New max found: {uncertainty}")
                self.max_uncertainty = uncertainty


            print(f"Uncertainty estimate: {uncertainty}")
            self.set_espilon(uncertainty/self.max_uncertainty)
            return uncertainty




