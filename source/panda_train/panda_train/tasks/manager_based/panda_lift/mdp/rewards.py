# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv


def finger_object_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # finger positions
    left_finger_idx  = robot.find_bodies("panda_leftfinger")[0][0]
    right_finger_idx = robot.find_bodies("panda_rightfinger")[0][0]
    left_pos  = robot.data.body_pos_w[:, left_finger_idx, :]
    right_pos = robot.data.body_pos_w[:, right_finger_idx, :]
    obj_pos   = obj.data.root_pos_w[:, :3]

    # distance reward
    left_dist  = torch.norm(left_pos - obj_pos, dim=-1)
    right_dist = torch.norm(right_pos - obj_pos, dim=-1)
    proximity  = torch.exp(-left_dist / std) + torch.exp(-right_dist / std)

    # gripper closure — finger joints: 0=closed, 0.04=open
    left_joint_idx  = robot.find_joints("panda_finger_joint1")[0][0]
    right_joint_idx = robot.find_joints("panda_finger_joint2")[0][0]
    left_joint  = robot.data.joint_pos[:, left_joint_idx]
    right_joint = robot.data.joint_pos[:, right_joint_idx]
    gripper_closed = 1.0 - ((left_joint + right_joint) / 2.0) / 0.04  # 0=open, 1=closed

    # only reward closing when fingers are near object
    near_object = (proximity > 1.5).float()  # both fingers within ~std of object
    closure_reward = near_object * gripper_closed

    return proximity + closure_reward
    
def ee_height_penalty(
    env: ManagerBasedRLEnv,
    min_height: float = 0.13,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies("panda_hand")[0][0]
    ee_height = robot.data.body_pos_w[:, ee_idx, 2]
    penalty = torch.clamp(min_height - ee_height, min=0.0)
    # print(f"EE height: min={ee_height.min().item():.4f} max={ee_height.max().item():.4f} mean={ee_height.mean().item():.4f}")
    return -penalty


def ee_orientation_upright_reward(env, lift_height: float = 0.1) -> torch.Tensor:
    """Reward upright gripper orientation only when cube is lifted."""
    # Check if cube is lifted
    object_pos = env.scene["object"].data.root_pos_w
    cube_lifted = (object_pos[:, 2] > lift_height).float()  # [N]
    
    # EE z-axis in world frame
    ee_quat = env.scene["ee_frame"].data.target_quat_w[:, 0]
    w, x, y, z = ee_quat[:, 0], ee_quat[:, 1], ee_quat[:, 2], ee_quat[:, 3]
    ee_z_world_z = 1 - 2*(x*x + y*y)  # z component of EE z-axis
    
    # Reward pointing down (-z) only when lifted
    reward = -ee_z_world_z * cube_lifted
    return reward