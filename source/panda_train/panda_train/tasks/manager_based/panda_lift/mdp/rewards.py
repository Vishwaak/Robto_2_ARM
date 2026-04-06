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