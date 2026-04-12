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

import wandb
import os


from isaaclab.sensors import TiledCameraCfg  # isort: skip


from panda_train.tasks.manager_based.panda_lift.mdp.rewards import finger_object_distance
from panda_train.tasks.manager_based.panda_lift.mdp.rewards import ee_height_penalty

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import mdp as base_mdp

from panda_train.tasks.manager_based.panda_lift.network import DepthPositionPredictor


PHASE = 1
IMG_SIZE = 128  # for depth image obs
LATENT_DIM = 64
END_STEP = 250  # number of steps over which to anneal out the object position term in obs
_last_save_step = 0
SAVE_INTERVAL = 100*96


# Module-level instance
_lift_depth_predictor = DepthPositionPredictor(latent_dim=64)
_predictor_optimizer = torch.optim.Adam(_lift_depth_predictor.parameters(), lr=1e-3)




def _save_predictor(env, step: int) -> None:
    """
    Save predictor weights to the same log folder Isaac Lab uses.
    Isaac Lab stores logs in env.cfg.sim.device... actually the
    easiest way is to grab it from wandb.run.dir if available,
    otherwise fall back to a local path.
    """
    
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"depth_predictor_{step}.pt")
    if wandb.run is not None:
        wandb.define_metric("DepthPredictor/*", step_metric="predictor_step")
    

    print(path)
    torch.save({
        "model":     _lift_depth_predictor.state_dict(),
        "optimizer": _predictor_optimizer.state_dict(),
        "step":      step,
    }, path)

    # Also tell W&B to track the file as an artifact
    if wandb.run is not None:
        artifact = wandb.Artifact(
            name="depth_predictor",
            type="model",
            description=f"Depth predictor weights at step {step}",
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    print(f"[DepthPredictor] Saved → {path}")


def _get_predictor(device: torch.device):
    """Lazy init — creates predictor on correct device on first call."""
    global _lift_depth_predictor, _predictor_optimizer
    if _lift_depth_predictor is None:
        _lift_depth_predictor = DepthPositionPredictor(latent_dim=64).to(device)
        _predictor_optimizer = torch.optim.Adam(
            _lift_depth_predictor.parameters(), lr=1e-3
        )
        print(f"[DepthPredictor] Initialized on {device}")
        # Verify
        print(f"[DepthPredictor] is nn.Module: {isinstance(_lift_depth_predictor, nn.Module)}")
        for name, p in _lift_depth_predictor.named_parameters():
            print(f"  {name}: requires_grad={p.requires_grad} device={p.device}")
    return _lift_depth_predictor, _predictor_optimizer

def lift_depth_predict_pos(
    env,
    sensor_cfg: SceneEntityCfg,
    robot: SceneEntityCfg = SceneEntityCfg("robot"),
    object: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    global _last_save_step

    sensor = env.scene[sensor_cfg.name]
    depth = sensor.data.output["depth"].squeeze(-1).unsqueeze(1)
    depth = torch.clamp(depth, 0.1, 3.0) / 3.0

    # Move to correct device on first call
    if next(_lift_depth_predictor.parameters()).device != depth.device:
        _lift_depth_predictor.to(depth.device)

    if PHASE == 1:
        gt_pos = mdp.object_position_in_robot_root_frame(
            env, robot_cfg=robot, object_cfg=object
        )

        _lift_depth_predictor.train()
        depth = depth.detach().float()
        gt_pos = gt_pos.detach()
        pred = _lift_depth_predictor(depth)
        loss = nn.functional.mse_loss(pred, gt_pos)
        loss = torch.tensor(loss, requires_grad=True)
        _predictor_optimizer.zero_grad()
        loss.backward()
        _predictor_optimizer.step()

        step = env.common_step_counter// env.num_envs

        # ── Log to W&B ────────────────────────────────────────────────────
       
        with torch.no_grad():
            err = (pred - gt_pos).norm(dim=-1).mean().item()

            # W&B logs alongside RSL-RL's existing metrics
        if wandb.run is not None:
            wandb.log({
                "DepthPredictor/loss":     loss.item(),
                "DepthPredictor/mean_err_m": err,
                "DepthPredictor/predictor_step": step//96,
                
            }, commit=False)

        # ── Save weights to Isaac Lab log folder ──────────────────────────
        if env.common_step_counter > 0 and env.common_step_counter % (SAVE_INTERVAL * env.num_envs) == 0:
            _save_predictor(env, env.common_step_counter)

        return gt_pos.detach()

    else:
        _lift_depth_predictor.eval()
        with torch.no_grad():
            pred = _lift_depth_predictor(depth)

        step = env.common_step_counter
        if step % 500 == 0:
            gt_pos = mdp.object_position_in_robot_root_frame(
                env, robot_cfg=robot, object_cfg=object
            )
            err = (pred - gt_pos).norm(dim=-1).mean().item()

            if wandb.run is not None:
                wandb.log({
                    "DepthPredictor/phase2_err_m": err,
                }, step=step)

        return pred.detach()
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
    target_object_position = ObsTerm(                         # same name as original
        func=mdp.generated_commands,
        params={"command_name": "object_pose"},
    )
    actions      = ObsTerm(func=mdp.last_action)
    object_pos = ObsTerm(
            func=lift_depth_predict_pos, 
            params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"), 
            "robot": SceneEntityCfg("robot"),
            "object": SceneEntityCfg("object"), },)
    # depth_image  = ObsTerm(func=lift_depth_encoded,
    #                        params={"sensor_cfg": SceneEntityCfg("wrist_camera")})

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
                "scale_range": (0.8, 1.5),  # wider/narrower table,
                "object_type": "table"
            },
        )

        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.35, 0.35),
                    "y": (-0.3, 0.3),
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

        self.rewards.ee_height_penalty = RewTerm(
                func=ee_height_penalty,
                weight=3.0,
                params={"min_height": 0.13}, 
        )

        self.rewards.reaching_object.weight = 5.0        # was 1.0
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



