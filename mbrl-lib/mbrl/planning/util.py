

import string
import omegaconf
import mbrl
from . import (
    Agent,
    create_ee_agent_for_model,
    create_exploration_agent_for_model,
    create_trajectory_optim_agent_for_model
)


def create_agent_for_model(
    agent_type: string,
    model_env: mbrl.models.ModelExpEnv,
    agent_cfg: omegaconf.DictConfig,
    num_particles: int = 1
) -> Agent:
    
    if agent_type == "Exploration":
        return create_exploration_agent_for_model(model_env, agent_cfg, num_particles)
    elif agent_type == "Exploration_Exploitation":
        return create_ee_agent_for_model(model_env, agent_cfg, num_particles)
    elif agent_type == "pets":
        model_env.set_exploration(False)
        return create_trajectory_optim_agent_for_model(model_env, agent_cfg, num_particles)
    else:
        raise ValueError("Invalid agent type")
