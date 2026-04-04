import math


from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import (
    FrankaReachEnvCfg as BaseCfg,
)
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils


from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab.utils.modifiers import ModifierCfg

import torch
from torch import nn


 
class DepthEncoderModifier:
    """
    CNN encoder that compresses a (N, 1, 64, 64) depth image to (N, 64) latent.
    Used as an Isaac Lab observation modifier so RSL-RL sees a flat 96D vector.
 
    The encoder is trained end-to-end with PPO — gradients flow through it
    because Isaac Lab applies modifiers inside the obs computation graph.
    """
 
    def __init__(self, latent_dim: int = 64, image_height: int = 64, image_width: int = 64):
        self.latent_dim = latent_dim
        self.image_height = image_height
        self.image_width = image_width
        self._encoder = None  # lazy init on first call (device not known at cfg time)
 
    def _build_encoder(self, device: torch.device) -> nn.Module:
        """Build CNN encoder. Called once on first forward pass."""
        # For 64x64 input:
        # Conv1: (1,64,64) → (32,31,31)
        # Conv2: (32,31,31) → (64,15,15)
        # Conv3: (64,15,15) → (64,7,7)
        # Flatten: 64*7*7 = 3136
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0), nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, self.latent_dim), nn.ELU(),
        ).to(device)
 
    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: (N, H, W) or (N, 1, H, W) depth image tensor, values in meters.
        Returns:
            latent: (N, latent_dim) encoded representation.
        """
        # Lazy init encoder on first call
        if self._encoder is None:
            self._encoder = self._build_encoder(depth.device)
 
        # Ensure shape is (N, 1, H, W)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)  # (N, H, W) → (N, 1, H, W)
 
        # Normalize depth to [0, 1] using clipping range (0.01, 2.0)
        depth = torch.clamp(depth, 0.01, 2.0) / 2.0
 
        return self._encoder(depth)  # (N, latent_dim)
 
# Module-level encoder instance — shared across all uses
_depth_encoder_instance = DepthEncoderModifier(latent_dim=64, image_height=64, image_width=64)



def depth_encoded(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Custom obs term: fetches depth image and encodes it to a 64D latent.
    Returns: (N, 64) tensor — compatible with flat obs concatenation.
    """
    # Get camera sensor
    sensor = env.scene[sensor_cfg.name]
    # Get depth image: (N, H, W, 1)
    depth = sensor.data.output["depth"]
    # Reshape to (N, 1, H, W) for CNN
    depth = depth.squeeze(-1).unsqueeze(1)  # (N, 1, H, W)
    return _depth_encoder_instance(depth)   # (N, 64)

@configclass
class PandaTrainEnvCfg(BaseCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 4096

        wrist_camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera",  # attached to hand
                update_period=0.0,       # update_period not update_rate_hz
                height=64,
                width=64,
                data_types=["depth"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.01, 2.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.08, 0.0, 0.04),
                    rot=(0.0, 0.0, 0.0, 1.0),
                    convention="ros",
                ),
            )
        
        self.scene.wrist_camera = wrist_camera

        self.observations.policy.depth_image = ObsTerm(
            func=depth_encoded,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
            # no data_type, no normalize — depth_encoded handles everything internally
        )
      
        # switch to joint velocity controller
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.175, use_default_offset=True
        )


        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

       
        
        self.rewards.end_effector_position_tracking.weight = -2.0
        self.rewards.end_effector_position_tracking_fine_grained.weight = -0.5 
        self.rewards.end_effector_orientation_tracking.weight = -0.75

        self.rewards.action_rate = RewTerm(
            func=mdp.action_rate_l2,
            weight=-0.01,
            )
        self.rewards.reach_bonus = RewTerm(
            func=mdp.position_command_error_tanh,
            weight=0.75,
            params={
                "std": 0.1,
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "command_name": "ee_pose",
                },
        )
        self.rewards.joint_vel = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.0001,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )