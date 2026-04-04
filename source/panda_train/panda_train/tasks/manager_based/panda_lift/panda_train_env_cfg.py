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

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab.managers import RewardTermCfg as RewTerm


from isaaclab.sensors import TiledCameraCfg  # isort: skip


from panda_train.tasks.manager_based.panda_lift.mdp.rewards import finger_object_distance



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
_lift_depth_encoder = DepthEncoderModifier(latent_dim=64)


def lift_depth_encoded(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Custom obs term: fetches wrist depth image and encodes to 64D latent.
    Returns: (N, 64) — compatible with flat obs concatenation.
    """
    sensor = env.scene[sensor_cfg.name]
    depth = sensor.data.output["depth"]          # (N, H, W, 1)
    depth = depth.squeeze(-1).unsqueeze(1)       # (N, 1, H, W)
    return _lift_depth_encoder(depth)            # (N, 64)


##
# Environment configuration
##

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

        self.sim.render.rendering_mode  = "performance"
        self.sim.render.enable_reflections = False
        self.sim.render.enable_global_illumination = False
        self.sim.render.enable_shadows = False
        self.sim.render.enable_ambient_occlusion = False
        self.sim.render.enable_translucency = False

        # --- Robot ---
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
                    pos=(0.08, 0.0, 0.04),
                    rot=(0.0, 0.0, 0.0, 1.0),
                    convention="ros",
                ),
                data_types=["depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.01, 2.0),
                ),
                height=128,
                width=128,
            )

        # --- Depth Observation ---
        # Encoded to 64D latent — compatible with flat obs concatenation
        self.observations.policy.depth_image = ObsTerm(
            func=lift_depth_encoded,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
        )
    
    # custom rewards
    # In __post_init__
        # self.rewards.finger_object_distance = RewTerm(
        #     func=finger_object_distance,
        #     weight=3.0,
        #     params={"std": 0.05},
        # )

@configclass
class FrankaCubeLiftDepthEnvCfg_PLAY(FrankaCubeLiftDepthEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False