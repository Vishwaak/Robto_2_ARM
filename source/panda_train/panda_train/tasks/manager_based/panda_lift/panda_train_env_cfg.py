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
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm

from panda_train.tasks.manager_based.panda_lift.mdp.events import randomize_camera_pose, randomize_object_scale



from isaaclab.sensors import TiledCameraCfg  # isort: skip


from panda_train.tasks.manager_based.panda_lift.mdp.rewards import finger_object_distance
from panda_train.tasks.manager_based.panda_lift.mdp.rewards import ee_height_penalty

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import mdp as base_mdp

PHASE = 1
IMG_SIZE = 128  # for depth image obs
LATENT_DIM = 64
END_STEP = 200  # number of steps over which to anneal out the object position term in obs

##
# Depth Encoder
##
class DepthEncoderModifier:
    """CNN encoder: (N, 1, H, W) depth → (N, 64) latent."""

    def __init__(self, latent_dim: int = 64):
        self.latent_dim = latent_dim
        self._encoder = None

    def _build_encoder(self, device, sample_input: torch.Tensor):
        """Infer linear layer size from actual input shape."""
        conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ELU(),
            nn.Flatten(),
        ).to(device)
        with torch.no_grad():
            flat_dim = conv(sample_input[:1]).shape[-1]
        return nn.Sequential(
            conv,
            nn.Linear(flat_dim, self.latent_dim), nn.ELU(),
        ).to(device)

    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)               # (N, H, W) → (N, 1, H, W)
        depth = torch.clamp(depth, 0.01, 2.0) / 2.0  # normalize to [0, 1]
        if self._encoder is None:
            self._encoder = self._build_encoder(depth.device, depth)
            print(f"[DepthEncoder] Initialized on {depth.device}, input: {depth.shape}")
        return self._encoder(depth)                  # (N, 64)


# Module-level encoder instance shared across all envs
_lift_depth_encoder = DepthEncoderModifier(latent_dim=LATENT_DIM)


def lift_depth_encoded(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Custom obs term: fetches wrist depth image and encodes to 64D latent.
    Returns: (N, 64) — compatible with flat obs concatenation.
    """

    num_envs = env.num_envs
    device = env.device

    if PHASE == 1:
        latent = torch.zeros((num_envs, LATENT_DIM), device=device)  # zeros, object_pos active
    else:
        sensor = env.scene[sensor_cfg.name]
        depth = sensor.data.output["depth"]          # (N, H, W, 1)
        depth = depth.squeeze(-1).unsqueeze(1)       # (N, 1, H, W)
        latent = _lift_depth_encoder(depth)            # (N, 64)
    
    return latent

def object_pos_or_zero(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    real_pos = mdp.object_position_in_robot_root_frame(
            env, robot_cfg=robot_cfg, object_cfg=object_cfg)

    if PHASE == 2:
        current_iter = env.common_step_counter // (env.num_envs * 96)
        blend = max(0.0, 1.0 - current_iter / END_STEP)
        real_pos = mdp.object_position_in_robot_root_frame(
            env, robot_cfg=robot_cfg, object_cfg=object_cfg)

        if current_iter > END_STEP:
            print("[object_pos_or_zero] Blending complete, fully unmasked object position.")
            return torch.zeros((env.num_envs, 3), device=env.device)
        
        return blend * real_pos

    elif PHASE == 1:
        return real_pos

    else:
        return torch.zeros((env.num_envs, 3), device=env.device)

##
# Environment configuration
##
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
class PolicyCfg(ObsGroup):
    joint_pos    = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel    = ObsTerm(func=mdp.joint_vel_rel)
    object_pos = ObsTerm(
            func=object_pos_or_zero, 
            params={"robot_cfg": SceneEntityCfg("robot"),"object_cfg": SceneEntityCfg("object"),},)
    target_object_position = ObsTerm(                         # same name as original
        func=mdp.generated_commands,
        params={"command_name": "object_pose"},
    )
    actions      = ObsTerm(func=mdp.last_action)
    depth_image  = ObsTerm(func=lift_depth_encoded,
                           params={"sensor_cfg": SceneEntityCfg("wrist_camera")})

    def __post_init__(self):
        self.enable_corruption = True   # add noise to actor obs for robustness
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
        self.scene.robot = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=FRANKA_PANDA_CFG.spawn.replace(
            activate_contact_sensors=True
                ),
            )

        # --- Actions ---
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # --- End effector ---
        self.commands.object_pose.body_name = "panda_hand"

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
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )

        # --- Wrist Camera ---
        # Attached to panda_hand so it moves with the arm
        self.scene.wrist_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(0.05, 0.0, -0.065),  # 50mm forward, 65mm down from drawing
                    rot=(0.0, 0.0, 0.0, 1.0),
                    convention="ros",
                ),
                data_types=["depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=1.93,
                    horizontal_aperture=3.896,
                    clipping_range=(0.1, 3.0),
                ),
                height=IMG_SIZE,
                width=IMG_SIZE,
            )

        self.events.randomize_camera = EventTerm(
                func=randomize_camera_pose,
                mode="reset",
                params={
                    "camera_cfg": SceneEntityCfg("wrist_camera"),
                    # ±5mm position uncertainty
                    "pos_range": {
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.005, 0.005),
                    },
                    # ±3 degrees rotation uncertainty
                    "rot_range": {
                        "roll":  (-0.052, 0.052),
                        "pitch": (-0.052, 0.052),
                        "yaw":   (-0.052, 0.052),
                    },
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

        self.events.randomize_table_scale = EventTerm(
            func=randomize_object_scale,
            mode="reset",
            params={
                "object_cfg": SceneEntityCfg("table"),
                "scale_range": (0.8, 1.4),  # wider/narrower table,
                "object_type": "table"
            },
        )

        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.45, 0.45),
                    "y": (-0.4, 0.4),
                    "z": (0.0, 0.0),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )
        
        self.observations.critic = CriticCfg()

        self.observations.policy = PolicyCfg() 

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
                "threshold": 5.0,
                "sensor_cfg": SceneEntityCfg(
                    "contact_sensor",
                    body_names=["panda_hand"],
                ),
            },
)

        # --- Contact Sensor ---
        # self.scene.contact_sensor = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/.*",
        #     update_period=0.0,
        #     history_length=3,
        #     debug_vis=False,
        # )

        self.rewards.ee_height_penalty = RewTerm(
                func=ee_height_penalty,
                weight=1.0,
                params={"min_height": 0.13}, 
        )

        self.rewards.reaching_object.weight = 10.0        # was 1.0
        self.rewards.lifting_object.weight = 20.0        # was 15.0
        self.rewards.object_goal_tracking.weight = 10.0   # was default
    
@configclass
class FrankaCubeLiftDepthEnvCfg_PLAY(FrankaCubeLiftDepthEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.randomize_camera = None



