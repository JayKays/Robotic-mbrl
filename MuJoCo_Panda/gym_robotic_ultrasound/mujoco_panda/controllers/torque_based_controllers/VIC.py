from mujoco_panda.utils.tf import quatdiff_in_euler
from .configs import BASIC_HYB_CONFIG
import numpy as np
import quaternion
import time
from mujoco_panda.controllers.torque_based_controllers import VIC_config as cfg
from mujoco_panda.controllers.torque_based_controllers.VIC_base_controller import BaseControllerVIC


class VIC(BaseControllerVIC):
    """
    Torque-based task-space hybrid force motion controller.
    Computes the joint torques required for achieving a desired
    end-effector pose and/or wrench. Goal values and directions
    are defined in cartesian coordinates.

    First computes cartesian force for achieving the goal using PD
    control law, then computes the corresponding joint torques using 
    :math:`\tau = J^T F`.
    
    """

    def __init__(self, robot_object, control_rate=None, *args, **kwargs):
        """
        contstructor

        :param robot_object: the :py:class:`PandaArm` object to be controlled
        :type robot_object: PandaArm
        :param config: dictionary of controller parameters, defaults to 
            BASIC_HYB_CONFIG (see config for reference)
        :type config: dict, optional
        """
        super().__init__(robot_object, control_rate=control_rate, max_num_it = kwargs.get("max_num_it"),)
        #super(VIC,self).__init__(robot_object, config)
        self.demo_data_dict = {}
        self.demo_data = []
        self.delta_K = np.zeros((6, 6))
        self.B_hat_lower = cfg.B_hat_lower
        self.B_hat_upper = cfg.B_hat_upper
        self.K_hat_lower = cfg.K_hat_lower
        self.K_hat_upper = cfg.K_hat_upper

    def _compute_cmd(self):
        """
        Actual computation of command given the desired goal states

        :return: computed joint torque values
        :rtype: np.ndarray (7,)
        """
        #self.gamma[8, 8] = self.action[0]  # gamma B
        self.delta_K[2,2] = self.action *10 # gamma K

        x, x_dot, delta_x, jacobian, robot_inertia, F_ext_2D = \
            self.fetch_states(self.timestep,  self.goal_pos)

        self.update_MBK(self.delta_K)
        self.perform_torque_Huang1992(self.M, self.B, self.K, self.goal_acc, self.goal_vel, x, \
                            x_dot, self.goal_pos,  F_ext_2D, jacobian, robot_inertia)


    def reset(self):
        super().reset()
        self.timestep = 0
        self.delta_K = np.zeros((6,6))

    def update_MBK(self, delta_K,):
        #M_hat = M  # + np.diagflat(lam[0:6]) M is chosen to be constant
        self.K = self.K + delta_K
        self.K = np.clip(self.K, self.K_hat_lower, self.K_hat_upper)
        self.B[2,2] = 2*np.sqrt(self.K[2,2])
        # ensure_limits(1,5000,M_hat)
        self.B = np.clip(self.B, self.B_hat_lower, self.B_hat_upper)

 #Not used in this impelenmnetatin , used for collecting demo data in the threading based version
    def set_demonstration_data(self):
        self.demo_data_dict["goal_acc"] = self.goal_acc
        self.demo_data_dict["goal_vel"] = self.goal_vel
        self.demo_data_dict["pose"] = self.get_x(self.goal_ori) # x
        self.demo_data_dict["x_dot"] = np.concatenate((self._robot.ee_velocity()[0], self._robot.ee_velocity()[1]))
        self.demo_data_dict["goal_pos"] = self.goal_pos
        self.demo_data_dict["F_ext_2D"] = np.array([0, 0, self._robot.get_ft_reading()[0][2], 0, 0, 0]).reshape([6, 1])
        self.demo_data_dict["J"] = self._robot.jacobian() # jacobian
        self.demo_data_dict["robot_inertia"] = self._robot.mass_matrix()
        self.demo_data_dict["FT"] = np.concatenate((self._robot.get_ft_reading()[0],self._robot.get_ft_reading()[1] ))
        self.demo_data_dict["goal_ori"] = self.goal_ori
        self.demo_data_dict["goal_force"] = self.goal_force