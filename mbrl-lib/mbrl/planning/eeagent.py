
from typing import Sequence
from .trajectory_opt import TrajectoryOptimizerAgent
from .core import Agent, complete_agent_cfg
import omegaconf
from .trajectory_opt import TrajectoryOptimizer
import numpy as np
import mbrl

class EEAgent(Agent):
    """
    Implements a exploration vs exploitation based agent. 
    By utilizing the underlying dynamics model's ability to quantify uncertainty, the agent aim to 
    explore parts of the state/action space were predictions are uncertain.
    """

    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: Sequence[float],
        action_ub: Sequence[float],
        planning_horizon: int = 1,
        replan_freq: int = 1,
        verbose: bool = False,
        keep_last_solution: bool = True
    ):

        self.optimizer = TrajectoryOptimizer(
            optimizer_cfg,
            np.array(action_lb),
            np.array(action_ub),
            planning_horizon=planning_horizon,
            replan_freq=replan_freq,
            keep_last_solution=keep_last_solution
        )


        self.exploitation_eval_fn: mbrl.types.TrajectoryEvalFnType = None
        self.exploration_eval_fn: mbrl.types.TrajectoryEvalFnType = None
        self.planning_horizon = planning_horizon
        self.replan_freq = replan_freq
        self.verbose = verbose


    def set_exploration_eval_fn(
        self, eval_fn: mbrl.types.TrajectoryEvalFnType
    ):
        self.exploration_eval_fn = eval_fn

    def set_exploitation_eval_fn(
        self, eval_fn: mbrl.types.TrajectoryEvalFnType
    ):
        self.exploitation_eval_fn = eval_fn

    def plan(self): pass
    def act(self): pass


def create_ee_agent_for_model(
    model: mbrl.models.Ensemble,
    agent_cfg: omegaconf.DictConfig,
    num_particles: int = 1
) -> EEAgent:

    #TODO Figure out how to use model env with uncertainty based exploration
    pass


    


