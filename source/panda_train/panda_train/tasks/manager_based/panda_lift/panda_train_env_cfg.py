# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.modifiers import ModifierCfg
import isaaclab.sim as sim_utils

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
# from isaaclab_assets.robots.franka import FRANKA_FR3_CFG  # isort: skip
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm

from panda_train.tasks.manager_based.panda_lift.mdp.events import randomize_camera_pose, randomize_object_scale

import torch
from collections import deque

import wandb
import os


from isaaclab.sensors import TiledCameraCfg  # isort: skip


from panda_train.tasks.manager_based.panda_lift.mdp.rewards import finger_object_distance
from panda_train.tasks.manager_based.panda_lift.mdp.rewards import ee_height_penalty, ee_orientation_upright_reward

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.envs import mdp as base_mdp

from panda_train.tasks.manager_based.panda_lift.network import DepthPositionPredictor, LatentProbe, CharbonnierLoss

from panda_train.tasks.manager_based.panda_lift.usd.franka_fr3_cfg import FRANKA_FR3_CFG

IMG_SIZE = 128

##
# Environment configuration
##



def save_depth_image(env, env_ids):
    step = env.common_step_counter
    if step % 50 != 0:
        return
    
    depth = env.scene["wrist_camera"].data.output["distance_to_image_plane"]
    
    # Check across ALL envs how often cube is visible
    # Cube should appear as low depth values (~0.1-0.5m)
    depth_flat = depth[..., 0] if depth.shape[-1] == 1 else depth[:, 0]
    
    # Pixels within cube distance range
    cube_range = (depth_flat > 0.05) & (depth_flat < 0.5)
    visibility = cube_range.float().mean(dim=(1,2))  # per env
    
    print(f"Step {step}:")
    print(f"  raw depth: min={depth_flat.min():.3f}, max={depth_flat.max():.3f}, mean={depth_flat.mean():.3f}")
    print(f"  Cube visibility (frac pixels in 0.05-0.5m): "
          f"mean={visibility.mean():.3f}, "
          f"min={visibility.min():.3f}, "
          f"envs with >1% visible: {(visibility > 0.01).sum()}/{len(visibility)}")
    
    # img = depth_flat[0].cpu().numpy()
    # save_dir = "/home/xerous/Desktop/project/logs/depth_imgs"
    # os.makedirs(save_dir, exist_ok=True)

    # plt.figure(figsize=(6, 6))
    # plt.imshow(img, cmap='gray', vmin=0.0, vmax=3.0)
    # plt.colorbar(label='depth (m)')
    # plt.title(f"Step {step} | vis={visibility[0]:.3f}")
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f"depth_step{step:06d}.png"))
    # plt.close()

@configclass
class CriticCfg(ObsGroup):
    # mirror all actor terms
    joint_pos    = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel    = ObsTerm(func=mdp.joint_vel_rel)
    actions      = ObsTerm(func=mdp.last_action)
    target_object_position = ObsTerm(func=mdp.generated_commands,
                                     params={"command_name": "object_pose"})

    # privileged — ground truth object state
    object_pos   = ObsTerm(
        func=mdp.object_position_in_robot_root_frame,
    )
    object_vel   = ObsTerm(
        func=base_mdp.root_lin_vel_w,
        params={"asset_cfg": SceneEntityCfg("object")},
    )
    object_quat  = ObsTerm(
        func=base_mdp.root_quat_w,
        params={"asset_cfg": SceneEntityCfg("object")},
    )

    def __post_init__(self):
        self.enable_corruption = False  # no noise for critic
        self.concatenate_terms = True

@configclass
class StudentObsCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    target_object_position = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "object_pose"},
    )
    actions = ObsTerm(func=mdp.last_action)


    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True  # keeps depth_image as separate tensor


@configclass
class TeacherObsCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    target_object_position = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "object_pose"},
    )
    actions = ObsTerm(func=mdp.last_action)
    object_pos = ObsTerm(func=mdp.object_position_in_robot_root_frame)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True

from isaaclab.utils.modifiers import ModifierCfg

# in your mdp or a utils file
def depth_to_channels_first(x: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 3, 1, 2).contiguous().clone()
    invalid = (x == 0.0) | ~torch.isfinite(x) | (x > 3.0)  # catch inf/nan
    x = torch.clamp(x, 0.1, 2.0)
    x = (x - 0.1) / 0.9
    x[invalid] = 1.0                    # invalid → far (1.0 normalized)
    return x

@configclass
class DepthObsCfg(ObsGroup):
    depth_image = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "distance_to_image_plane"},
        modifiers=[ModifierCfg(func=depth_to_channels_first)],
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class FrankaCubeLiftDepthEnvCfg(LiftEnvCfg):
    """
    Franka cube lift environment with wrist-mounted depth camera.

    Observation space adds 64D depth latent on top of whatever
    the base LiftEnvCfg policy group defines.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- sim parameters --- 

        # self.sim.render.rendering_mode  = "performance"
        self.sim.render.enable_reflections = False
        self.sim.render.enable_global_illumination = False
        self.sim.render.enable_shadows = False
        self.sim.render.enable_ambient_occlusion = False
        self.sim.render.enable_translucency = False

        # --- Robot ---
        self.scene.robot = FRANKA_FR3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=FRANKA_FR3_CFG.spawn.replace(
            activate_contact_sensors=True
                ),
            )

        # --- Actions ---
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["fr3_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["fr3_finger_joint1"],
            open_command_expr={"fr3_finger_joint1": 0.04},
            close_command_expr={"fr3_finger_joint1": 0.0},
        )

        # --- End effector ---
        self.commands.object_pose.body_name = "fr3_hand"

        # --- Object ---
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # --- Frame transformer ---
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fr3_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr3_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.103]),
                ),
            ],
        )

        
        # --- Wrist Camera ---
        # Attached to panda_hand so it moves with the arm
        self.scene.wrist_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/fr3_hand/wrist_camera",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(0.059, -0.069, 0.0),  # 50mm forward, 65mm down from drawing
                   rot=(1.0, 0.0, 0.0, 0.0),
                    convention="ros",
                ),
                data_types=["depth","distance_to_image_plane","rgba"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=1.93,
                    horizontal_aperture=3.896,
                    clipping_range=(0.1, 3.0),
                ),
                height=IMG_SIZE,
                width=IMG_SIZE,
            )

        # self.events.randomize_camera = EventTerm(
        #         func=randomize_camera_pose,
        #         mode="reset",
        #         params={
        #             "camera_cfg": SceneEntityCfg("wrist_camera"),
        #             # ±5mm position uncertainty
        #             "pos_range": {
        #                 "x": (-0.005, 0.005),
        #                 "y": (-0.005, 0.005),
        #                 "z": (-0.005, 0.005),
        #             },
        #             # ±3 degrees rotation uncertainty
        #             "rot_range": {
        #                 "roll":  (-0.052, 0.052),
        #                 "pitch": (-0.052, 0.052),
        #                 "yaw":   (-0.052, 0.052),
        #             },
        #         },
        #     )
        
        self.events.randomize_object_scale = EventTerm(
            func=randomize_object_scale,
            mode="reset",
            params={
                "object_cfg": SceneEntityCfg("object"),
                "scale_range": (0.7, 1.2),  # 70% to 100% of original size
                "object_type": "cube"
            },
        )

        self.events.randomize_table_scale = EventTerm(
            func=randomize_object_scale,
            mode="reset",
            params={
                "object_cfg": SceneEntityCfg("table"),
                "scale_range": (0.8, 1.5),  # wider/narrower table,
                "object_type": "table"
            },
        )

        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.2, 0.2),
                    "y": (-0.15, 0.15),
                    "z": (0.0, 0.0),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )

        self.events.save_depth = EventTerm(
            func=save_depth_image,
            mode="interval",
            interval_range_s=(0.5, 0.5),  # every 0.5 sim seconds
            params={},
        )
        self.observations.critic = CriticCfg()

        self.observations.student = StudentObsCfg() 

        self.observations.teacher = TeacherObsCfg()
        
        self.observations.depth = DepthObsCfg()
        
        self.scene.contact_sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                update_period=0.0,
                history_length=3,
                debug_vis=False,
            )

        # Terminate episode on table contact
        self.terminations.table_contact = DoneTerm(
            func=base_mdp.illegal_contact,
            params={
                "threshold": 10.0,
                "sensor_cfg": SceneEntityCfg(
                    "contact_sensor",
                    body_names=["fr3_hand"],
                ),
            },
)

        self.rewards.ee_height_penalty = RewTerm(
                func=ee_height_penalty,
                weight=3.0,
                params={"min_height": 0.02}, 
        )

        # self.rewards.gripper_close_near_object = RewTerm(
        #         func=mdp.gripper_close_near_object,
        #         weight=2.0,
        #         params={"std": 0.1,},
        # )
        self.rewards.grasp_and_lift_bonus = RewTerm(
                    func=mdp.grasp_and_lift_bonus,
                    weight=50.0,
                    params={"lift_height": 0.05},
        )

        # self.rewards.ee_orientation_upright_reward = RewTerm (
        #             func=ee_orientation_upright_reward,
        #             weight = 15.0,
        #             params={"lift_height": 0.02}
        # )
        self.curriculum.action_rate = CurrTerm(
            func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1.0e-1, "num_steps": 400000}
        )

        self.curriculum.joint_vel = CurrTerm(
            func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1.0e-1, "num_steps": 400000}
        )

        self.rewards.reaching_object.weight = 1.0        # was 1.0
        self.rewards.lifting_object.weight = 25.0        # was 15.0
        self.rewards.object_goal_tracking.weight = 15.0   # was default
    
@configclass
class FrankaCubeLiftDepthEnvCfg_PLAY(FrankaCubeLiftDepthEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.student.enable_corruption = False
        self.observations.depth.enable_corruption = False
        self.events.randomize_camera = None

        self.observations.policy = StudentObsCfg()

        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                    "z": (0.0, 0.0),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )

        self.events.randomize_object_scale = EventTerm(
            func=randomize_object_scale,
            mode="reset",
            params={
                "object_cfg": SceneEntityCfg("object"),
                "scale_range": (0.7, 1.0),  # 70% to 100% of original size
                "object_type": "cube"
            },
        )



