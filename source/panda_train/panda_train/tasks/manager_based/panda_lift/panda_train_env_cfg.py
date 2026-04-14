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

import torch
from collections import deque

import wandb
import os


from isaaclab.sensors import TiledCameraCfg  # isort: skip


from panda_train.tasks.manager_based.panda_lift.mdp.rewards import finger_object_distance
from panda_train.tasks.manager_based.panda_lift.mdp.rewards import ee_height_penalty

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import mdp as base_mdp

from panda_train.tasks.manager_based.panda_lift.network import DepthPositionPredictor, LatentProbe, CharbonnierLoss


PHASE = 1
IMG_SIZE = 128  # for depth image obs
LATENT_DIM = 64
END_STEP = 250  # number of steps over which to anneal out the object position term in obs
_last_save_step = 0
SAVE_INTERVAL = 100*96
BUFFER_SIZE = 128

# Module-level instance

depth_predictor_path = "/home/xerous/Desktop/project/logs/rsl_rl/franka_lift_depth/2026-04-14_00-19-30_training_lift_depth_pred_joint/depth_predictor_4200.pt"
_lift_depth_predictor = DepthPositionPredictor(latent_dim=64)
_predictor_optimizer = torch.optim.Adam(_lift_depth_predictor.parameters(), lr=3e-4)
_latent_probe = LatentProbe()

_depth_buffer     = deque(maxlen=BUFFER_SIZE)
_joint_buffer     = deque(maxlen=BUFFER_SIZE)
_gt_buffer        = deque(maxlen=BUFFER_SIZE)

_charbonnier          = CharbonnierLoss(eps=1e-3)

if os.path.exists(depth_predictor_path):
    checkpoint = torch.load(depth_predictor_path, map_location="cuda")
    _lift_depth_predictor.load_state_dict(checkpoint["model"])
    _lift_depth_predictor.to("cuda")
    _predictor_optimizer.load_state_dict(checkpoint["optimizer"])
    # move optimizer state to GPU
    for state in _predictor_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    print(f"[DepthPredictor] Loaded weights from {depth_predictor_path}")

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

def collect_depth_data(
    env,
    env_ids: torch.Tensor,
    robot:  SceneEntityCfg = SceneEntityCfg("robot"),
    object: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    sensor    = env.scene["wrist_camera"]
    raw_depth = sensor.data.output["depth"]          # (N, H, W, 1)
 
    depth = (
        raw_depth
        .squeeze(-1)                                  # (N, H, W)
        .unsqueeze(1)                                 # (N, 1, H, W)
        .float()
        .clamp(0.1, 3.0)
        .div(3.0)
        .cpu()
        .detach()
    )
 
    joint_pos = (
        mdp.joint_pos_rel(env)
        .float()
        .cpu()
        .detach()
    )
 
    gt_pos = (
        mdp.object_position_in_robot_root_frame(env, robot_cfg=robot, object_cfg=object)
        .float()
        .cpu()
        .detach()
    )
 
    _depth_buffer.append(depth)
    _joint_buffer.append(joint_pos)
    _gt_buffer.append(gt_pos)
 
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
    env_ids: torch.Tensor,
    robot: SceneEntityCfg = SceneEntityCfg("robot"),
    object: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    global _last_save_step

    sensor = env.scene["wrist_camera"]
    depth = sensor.data.output["depth"].squeeze(-1).unsqueeze(1)
    depth = torch.clamp(depth, 0.1, 3.0) / 3.0

    global _lift_depth_predictor
    # Move to correct device on first call
    print(f"Depth tensor device: {depth.device}")
    if next(_lift_depth_predictor.parameters()).device != depth.device:
        
        _lift_depth_predictor.to(depth.device)

    if PHASE == 1:
        gt_pos = mdp.object_position_in_robot_root_frame(
            env, robot_cfg=robot, object_cfg=object
        )
        
        charbonnier_loss = CharbonnierLoss()

        _lift_depth_predictor.train()
        torch.set_grad_enabled(True)
        with torch.enable_grad():
            depth = depth.float().detach().requires_grad_(False)
            joint_pos = mdp.joint_pos_rel(env).float().detach()
            gt_pos = gt_pos.detach().float()
            pred_pos = _lift_depth_predictor(depth, joint_pos)
            print(depth.requires_grad, joint_pos.requires_grad, gt_pos.requires_grad, pred_pos.requires_grad)
            print(depth.grad_fn, joint_pos.grad_fn, gt_pos.grad_fn, pred_pos.grad_fn)
            loss = charbonnier_loss(pred_pos, gt_pos)
            loss = torch.tensor(loss, requires_grad=True)  # ensure loss has grad
            # loss = torch.tensor(loss, requires_grad=True)
            _predictor_optimizer.zero_grad()
            loss.backward()
            for p in _lift_depth_predictor.parameters():
                print(p.grad)
            _predictor_optimizer.step()
        torch.set_grad_enabled(False)
        step = env.common_step_counter// env.num_envs
        # probe_loss, probe_error = _latent_probe.update(latent, gt_pos)
        # ── Log to W&B ────────────────────────────────────────────────────
       
        # with torch.no_grad():
        #     err = (pred - gt_pos).norm(dim=-1).mean().item()

            # W&B logs alongside RSL-RL's existing metrics
        if wandb.run is not None:
            wandb.log({
                "Probe/loss":     loss.item(),
                "Probe/mean_err_m": torch.norm(pred_pos - gt_pos, dim=-1).mean().item(),
                "Probe/predictor_step": step//96,
                
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

def train_depth_predictor(path: str) -> dict:
    if len(_depth_buffer) < BUFFER_SIZE:
        return {}

    depth     = torch.cat(list(_depth_buffer), dim=0)
    joint_pos = torch.cat(list(_joint_buffer), dim=0)
    gt_pos    = torch.cat(list(_gt_buffer),    dim=0)

    device = next(_lift_depth_predictor.parameters()).device
    batch_size = 2048
    indices = torch.randperm(len(depth))

    _lift_depth_predictor.train()
    total_loss = 0.0
    total_err  = 0.0
    n_batches  = 0

    for start in range(0, len(depth), batch_size):
        idx = indices[start:start + batch_size]
        d   = depth[idx].to(device)
        j   = joint_pos[idx].to(device)
        gt  = gt_pos[idx].to(device)

        pred = _lift_depth_predictor(d, j)
        loss = _charbonnier(pred, gt)

        _predictor_optimizer.zero_grad()
        loss.backward()
        _predictor_optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()
            total_err  += (pred - gt).norm(dim=-1).mean().item()
            n_batches  += 1

    # no clear — deque rolls naturally

    if path is not None:
        torch.save({
            "model":     _lift_depth_predictor.state_dict(),
            "optimizer": _predictor_optimizer.state_dict(),
        }, path)

    return {
        "DepthPredictor/loss":     total_loss / n_batches,
        "DepthPredictor/mean_err": total_err  / n_batches,
    }
def pred_obj_pos( env,
    robot: SceneEntityCfg = SceneEntityCfg("robot"),
    object: SceneEntityCfg = SceneEntityCfg("object"),) -> torch.Tensor:
    sensor = env.scene["wrist_camera"]
    depth = sensor.data.output["depth"].squeeze(-1).unsqueeze(1).float()
    depth = torch.nan_to_num(depth, nan=0.0, posinf=3.0, neginf=0.0)
    depth = torch.clamp(depth, 0.1, 3.0) / 3.0
    joint_pos = mdp.joint_pos_rel(env).float().detach()
    _lift_depth_predictor.to(depth.device)
    _lift_depth_predictor.eval()
    gt_pos = mdp.object_position_in_robot_root_frame(env, robot_cfg=robot, object_cfg=object).float().detach()

    with torch.no_grad():
        pred = _lift_depth_predictor(depth, joint_pos)
        error = torch.norm(pred - gt_pos, dim=-1).mean().item()

        if error > 0.08:
            alpha = 0.2
        elif error > 0.05:
            alpha = 0.5
        elif error > 0.03:
            alpha = 0.8
        else:
            alpha = 1.0
            
    value = alpha * pred.detach() + (1 - alpha) * gt_pos.detach()
    print("using pred pos with alpha {:.2f}, error {:.3f}".format(alpha, error))
    return value

def noisy_object_pos(env, robot, object, noise_std=0.05):
    gt_pos = mdp.object_position_in_robot_root_frame(env, robot_cfg=robot, object_cfg=object)
    noise = torch.randn_like(gt_pos) * noise_std
    return (gt_pos + noise).clamp_(-1.0, 1.0)
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
    # object_pos   = ObsTerm(
    #     func=mdp.object_position_in_robot_root_frame)
    # object_pos = ObsTerm(
    #     func=noisy_object_pos,
    #     params={
    #         "robot": SceneEntityCfg("robot"),
    #         "object": SceneEntityCfg("object"),
    #         "noise_std": 0.09
    #     }
    # )

    object_pos = ObsTerm(
        func=pred_obj_pos,
        params={
            "robot": SceneEntityCfg("robot"),
            "object": SceneEntityCfg("object"),
        }
    )
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
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                    "z": (0.0, 0.0),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )
        
        self.events.train_predictor = EventTerm(
            func=collect_depth_data,
            mode="interval",
            interval_range_s=(0.0,0.0),  # every step
                params={
                "robot": SceneEntityCfg("robot"),
                "object": SceneEntityCfg("object"),
            },
        )

        self.events.collect_at_reset = EventTerm(
            func=collect_depth_data,
            mode="reset",
            params={
                "robot": SceneEntityCfg("robot"),
                "object": SceneEntityCfg("object"),
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

        # self.rewards.gripper_close_near_object = RewTerm(
        #         func=mdp.gripper_close_near_object,
        #         weight=2.0,
        #         params={"std": 0.1,},
        # )
        self.rewards.grasp_and_lift_bonus = RewTerm(
                    func=mdp.grasp_and_lift_bonus,
                    weight=25.0,
                    params={"lift_height": 0.05},
        )

        self.rewards.reaching_object.weight = 1.0        # was 1.0
        self.rewards.lifting_object.weight = 20.0        # was 15.0
        self.rewards.object_goal_tracking.weight = 15.0   # was default
    
@configclass
class FrankaCubeLiftDepthEnvCfg_PLAY(FrankaCubeLiftDepthEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.randomize_camera = None



