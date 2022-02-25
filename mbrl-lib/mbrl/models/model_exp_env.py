


from typing import Dict, Optional, Tuple

import gym
import numpy as np
import torch

import mbrl.types
from . import Model, ModelEnv

class ModelExpEnv(ModelEnv):
    """
    Extends the ModelEnv class to allow for epxloration via disagreement,
    where the rewards are based on the disagreement between each ensemble model
    to encourage exploration of the state action space were the dynamics model
    is uncertain.

    Functionality is equivalent to the parent class ModelEnv when disagreement = False
    """

    def __init__(
        self,
        env: gym.Env,
        model: Model,
        termination_fn: mbrl.types.TermFnType,
        reward_fn: Optional[mbrl.types.RewardFnType] = None,
        generator: Optional[torch.Generator] = None,
        diagreement: bool = False
    ):

        super().__init__(env, model, termination_fn, reward_fn, generator)
        self.disagreement = diagreement
    
    def step(
        self,
        actions: mbrl.types.TensorType,
        model_state: Dict[str, torch.Tensor],
        sample: bool = False,
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:


        next_observs, rewards, dones, next_model_state = super().step(actions, model_state, sample)

        if self.disagreement:
            with torch.no_grad():
                model_in = self.dynamics_model._get_model_input(model_state["obs"], actions)
                means, _ = self.dynamics_model.forward(model_in, use_propagation=False)
                rewards = torch.var(means, dim=0).mean(dim=1, keepdim=True)
        
        return next_observs, rewards, dones, next_model_state

    def evaluate_action_sequences(self, action_sequences: torch.Tensor, initial_state: np.ndarray, num_particles: int, disagreement: bool = None) -> torch.Tensor:

        temp = self.disagreement
        self.disagreement = self.disagreement or disagreement

        value =  super().evaluate_action_sequences(action_sequences, initial_state, num_particles)

        self.disagreement = temp

        return value

    def set_disagreement(self, disagree):
        self.disagreement = disagree


