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

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # Replace the depth_writer variable and save_depth_frame function with this:

    depth_state = {"writer": None}  

    def save_depth_frame(env):
        # Unwrap through all wrappers to get Isaac env
        isaac_env = env.unwrapped
        while hasattr(isaac_env, 'env'):
            isaac_env = isaac_env.env
            if hasattr(isaac_env, 'scene'):
                break

        depth = isaac_env.scene["wrist_camera"].data.output["distance_to_image_plane"]
        
        if depth.shape[-1] == 1:
            frame = depth[0, :, :, 0].cpu().numpy()
        else:
            frame = depth[0, 0].cpu().numpy()
        
        # Handle inf/nan
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




    # pred_path = "/home/xerous/Desktop/project/logs/rsl_rl/franka_lift_depth/2026-04-10_17-15-24_training_lift_async_depth_PHASE_2_domain_randomization_increase/model_1300_encoder.pt"
    # IMG_SIZE = 128
    device = env_cfg.sim.device

    # if os.path.exists(encoder_path):
    #     print(f"[INFO] Loading encoder from: {encoder_path}")
    #     _lift_depth_encoder._encoder = _lift_depth_encoder._build_encoder(
    #         device,
    #         torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=device)
    #     )
    #     _lift_depth_encoder._encoder.load_state_dict(
    #         torch.load(encoder_path, map_location=device)
    #     )
    #     _lift_depth_encoder._encoder.eval()  # eval mode for inference
    #     print("[INFO] Encoder loaded.")
    # else:
    #     print(f"[WARNING] Encoder not found: {encoder_path}")

    # predictor_path = "/home/xerous/Desktop/project/logs/rsl_rl/franka_lift_depth/2026-04-14_00-19-30_training_lift_depth_pred_joint/depth_predictor_4000.pt"
    # import panda_train.tasks.manager_based.panda_lift.panda_train_env_cfg as env_module

    # print("[INFO] Depth predictor loaded.")
    # if os.path.exists(predictor_path):
    #     checkpoint = torch.load(predictor_path, map_location=device)
    #     env_module._lift_depth_predictor.load_state_dict(checkpoint["model"])
    #     env_module._lift_depth_predictor.to(device)
    #     env_module._lift_depth_predictor.eval()
    #     print("[INFO] Depth predictor loaded.")
    # else:
    #     print(f"[WARNING] Depth predictor not found: {predictor_path}")


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

            print(f"[obs] joint_pos:  {obs_tensor[0:9].numpy().round(3)}")
            print(f"[obs] joint_vel:  {obs_tensor[9:18].numpy().round(3)}")
            print(f"[obs] object_pos: {obs_tensor[18:21].numpy().round(3)}")
            print(f"[obs] goal_pose:  {obs_tensor[21:28].numpy().round(3)}")
            print(f"[obs] actions:    {obs_tensor[28:35].numpy().round(3)}")
            print(f"[obs] depth_mean: {obs_tensor[35:99].mean().item():.4f}")
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            save_depth_frame(env)  # always save, no frame count check
            # print(f"depth prediction: {pred_obj_pos(env)}")

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    if depth_state["writer"] is not None:
                        depth_state["writer"].release()
                        print(f"[INFO] Depth video saved to: {depth_output_dir}/depth_cam.avi")
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
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
