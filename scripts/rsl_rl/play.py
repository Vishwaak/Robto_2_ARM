# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

import cv2
import numpy as np

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import panda_train.tasks  # noqa: F401


import csv
import os
import torch

def logging(env, log_path="/home/xerous/Desktop/project/logs/debug_log.csv"):
    step = env.common_step_counter
    
    robot = env.scene["robot"]
    ee = env.scene["ee_frame"]
    obj = env.scene["object"]
    contact = env.scene["contact_sensor"]

    # Get finger and hand body indices
    body_names = robot.body_names
    hand_idx = body_names.index("fr3_hand")
    left_finger_idx = body_names.index("fr3_leftfinger")
    right_finger_idx = body_names.index("fr3_rightfinger")
    tcp_idx = body_names.index("fr3_hand_tcp")

    hand_pos = robot.data.body_pos_w[0, hand_idx].cpu()
    left_finger_pos = robot.data.body_pos_w[0, left_finger_idx].cpu()
    right_finger_pos = robot.data.body_pos_w[0, right_finger_idx].cpu()
    tcp_pos = robot.data.body_pos_w[0, tcp_idx].cpu()

    # Initialize CSV on first call
    if not hasattr(logging, "_initialized"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "ee_x", "ee_y", "ee_z",
                "cube_x", "cube_y", "cube_z",
                "offset_x", "offset_y", "offset_z",
                "finger1", "finger2",
                "contact_leftfinger_x", "contact_leftfinger_y", "contact_leftfinger_z",
                "contact_rightfinger_x", "contact_rightfinger_y", "contact_rightfinger_z",
                "joint_pos_0", "joint_pos_1", "joint_pos_2", "joint_pos_3",
                "joint_pos_4", "joint_pos_5", "joint_pos_6",
                "hand_x", "hand_y", "hand_z",
                "left_finger_x", "left_finger_y", "left_finger_z",
                "right_finger_x", "right_finger_y", "right_finger_z",
                "tcp_x", "tcp_y", "tcp_z","ee_qw", "ee_qx", "ee_qy", "ee_qz",
            ])
        logging._initialized = True

    ee_pos = ee.data.target_pos_w[0, 0].cpu()
    cube_pos = obj.data.root_pos_w[0].cpu()
    offset = (ee_pos - cube_pos).cpu()
    finger_joints = robot.data.joint_pos[0, 7:].cpu()
    joint_pos = robot.data.joint_pos[0, :7].cpu()
    
    contact_left = contact.data.net_forces_w[0, 10].cpu()
    contact_right = contact.data.net_forces_w[0, 11].cpu()
    ee_quat = ee.data.target_quat_w[0, 0].cpu()

    tcp_idx = robot.body_names.index("fr3_hand_tcp")
    left_idx = robot.body_names.index("fr3_leftfinger")
    hand_idx = robot.body_names.index("fr3_hand")

    print("hand z:        ", robot.data.body_pos_w[0, hand_idx, 2].item())
    print("left_finger z: ", robot.data.body_pos_w[0, left_idx, 2].item())
    print("tcp z:         ", robot.data.body_pos_w[0, tcp_idx, 2].item())
    row = [
        step,
        *ee_pos.numpy().round(4),
        *cube_pos.numpy().round(4),
        *offset.numpy().round(4),
        *finger_joints.numpy().round(4),
        *contact_left.numpy().round(4),
        *contact_right.numpy().round(4),
        *joint_pos.numpy().round(4),
        *hand_pos.numpy().round(4),
        *left_finger_pos.numpy().round(4),
        *right_finger_pos.numpy().round(4),
        *tcp_pos.numpy().round(4),
        *ee_quat.numpy().round(4)
    ]


    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


    depth_output_dir = os.path.join(log_dir, "videos", "depth_cam")
    os.makedirs(depth_output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Replace the depth_writer variable and save_depth_frame function with this:

    depth_state = {"writer": None}
    rgb_state = {"writer": None}  

    def save_depth_frame(env, save_rgb: bool = True):
        # Unwrap through all wrappers to get Isaac env
        isaac_env = env.unwrapped
        while hasattr(isaac_env, 'env'):
            isaac_env = isaac_env.env
            if hasattr(isaac_env, 'scene'):
                break

        # ── Depth ────────────────────────────────────────────────────────────────
        depth = isaac_env.scene["wrist_camera"].data.output["distance_to_image_plane"]

        if depth.shape[-1] == 1:
            frame = depth[0, :, :, 0].cpu().numpy()
        else:
            frame = depth[0, 0].cpu().numpy()

        frame = np.where(np.isfinite(frame), frame, 3.0)
        frame = np.clip(frame, 0.1, 3.0)
        frame = ((frame - 0.1) / (3.0 - 0.1) * 255).astype(np.uint8)
        frame_colored = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        if depth_state["writer"] is None:
            h, w = frame_colored.shape[:2]
            depth_state["writer"] = cv2.VideoWriter(
                os.path.join(depth_output_dir, "depth_cam.avi"),
                fourcc, 30, (w, h)
            )
            print(f"[INFO] Depth writer opened: {depth_state['writer'].isOpened()}, size=({w},{h})")

        depth_state["writer"].write(frame_colored)

        # ── RGB ──────────────────────────────────────────────────────────────────
        if not save_rgb:
            return

        rgb_raw = isaac_env.scene["wrist_camera"].data.output.get("rgba")
        if rgb_raw is None:
            return  # camera not configured with RGB output — skip silently

        # Tensor shape: (N, H, W, 4) RGBA  or  (N, H, W, 3) RGB
        rgb_np = rgb_raw[0].cpu().numpy()          # (H, W, 3or4)  uint8
        if rgb_np.shape[-1] == 4:                  # drop alpha channel if present
            rgb_np = rgb_np[..., :3]

        # Isaac Lab outputs RGB; OpenCV expects BGR
        bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

        if rgb_state["writer"] is None:
            h, w = bgr.shape[:2]
            rgb_state["writer"] = cv2.VideoWriter(
                os.path.join(depth_output_dir, "rgb_cam.avi"),
                fourcc, 30, (w, h)
            )
            print(f"[INFO] RGB writer opened: {rgb_state['writer'].isOpened()}, size=({w},{h})")

        rgb_state["writer"].write(bgr)

    device = env_cfg.sim.device

   

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
   
    
    timestep = 0
    max_depth_frames = 1000  # save 1000 frames (~33 seconds at 30fps)
    depth_frame_count = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # Get the policy obs tensor from TensorDict
            obs_tensor = obs["policy"][0].cpu()  # first env, policy group

            actions = policy(obs)

            obs, _, dones, info = env.step(actions)
         
            # obs, _, dones, _ = env.step(actions)
            save_depth_frame(env)  # always save, no frame count check
            logging(env.unwrapped)
            # print(f"depth prediction: {pred_obj_pos(env)}")

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break
                   
            
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if depth_state["writer"] is not None:
        depth_state["writer"].release()
        print(f"[INFO] Depth video saved to: {depth_output_dir}/depth_cam.avi")
     
    if rgb_state["writer"] is not None:
        rgb_state["writer"].release()
        print(f"[Info] RGB video saved to : {depth_output_dir}/rgb_video.avi")
    
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
