
import torch
import numpy as np


def get_reacher_EE_pos(states: torch.Tensor) -> torch.Tensor:
    theta1, theta2, theta3, theta4, theta5, theta6, _ = (
        states[:, :1],
        states[:, 1:2],
        states[:, 2:3],
        states[:, 3:4],
        states[:, 4:5],
        states[:, 5:6],
        states[:, 6:],
    )

    rot_axis = torch.cat(
        [
            torch.cos(theta2) * torch.cos(theta1),
            torch.cos(theta2) * torch.sin(theta1),
            -torch.sin(theta2),
        ],
        axis=1,
    ).to(states.device)
    rot_perp_axis = torch.cat(
        [-torch.sin(theta1), torch.cos(theta1), torch.zeros(theta1.shape, device=states.device)], axis=1
    ).to(states.device)
    cur_end = torch.cat(
        [
            0.1 * torch.cos(theta1) + 0.4 * torch.cos(theta1) * torch.cos(theta2),
            0.1 * torch.sin(theta1) + 0.4 * torch.sin(theta1) * torch.cos(theta2) - 0.188,
            -0.4 * torch.sin(theta2),
        ],
        axis=1,
    ).to(states.device)

    for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
        perp_all_axis = torch.cross(rot_axis, rot_perp_axis)
        x = torch.cos(hinge) * rot_axis
        y = torch.sin(hinge) * torch.sin(roll) * rot_perp_axis
        z = -torch.sin(hinge) * torch.cos(roll) * perp_all_axis
        new_rot_axis = x + y + z
        new_rot_perp_axis = torch.cross(new_rot_axis, rot_axis)
        new_rot_perp_axis[
            torch.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30
        ] = rot_perp_axis[torch.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
        new_rot_perp_axis /= torch.linalg.norm(
            new_rot_perp_axis, axis=1, keepdims=True
        )
        rot_axis, rot_perp_axis, cur_end = (
            new_rot_axis,
            new_rot_perp_axis,
            cur_end + length * new_rot_axis,
        )

    return cur_end

def get_EE_pos_np(states):
    theta1, theta2, theta3, theta4, theta5, theta6, _ = (
        states[:, :1],
        states[:, 1:2],
        states[:, 2:3],
        states[:, 3:4],
        states[:, 4:5],
        states[:, 5:6],
        states[:, 6:],
    )

    rot_axis = np.concatenate(
        [
            np.cos(theta2) * np.cos(theta1),
            np.cos(theta2) * np.sin(theta1),
            -np.sin(theta2),
        ],
        axis=1,
    )
    rot_perp_axis = np.concatenate(
        [-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1
    )
    cur_end = np.concatenate(
        [
            0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
            0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
            -0.4 * np.sin(theta2),
        ],
        axis=1,
    )

    for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
        perp_all_axis = np.cross(rot_axis, rot_perp_axis)
        x = np.cos(hinge) * rot_axis
        y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
        z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
        new_rot_axis = x + y + z
        new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
        new_rot_perp_axis[
            np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30
        ] = rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
        new_rot_perp_axis /= np.linalg.norm(
            new_rot_perp_axis, axis=1, keepdims=True
        )
        rot_axis, rot_perp_axis, cur_end = (
            new_rot_axis,
            new_rot_perp_axis,
            cur_end + length * new_rot_axis,
        )

    return cur_end