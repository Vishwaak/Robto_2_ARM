# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class FrankaLiftDepthPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 2000
    save_interval = 50
    experiment_name = "franka_lift_depth"
    run_name = "training_lift_async_depth_PHASE_2_domain_randomization"  # used for logging and checkpointing
    log_root_path = "/home/xerous/Desktop/project/logs/"
    logger = "wandb"
    wandb_project = "isaac-panda"
    resume=True
    load_run = "2026-04-08_21-01-51_training_lift_async_depth_PHASE_1_domain_randomization"  # exact run name
    load_checkpoint = "model_950.pt"  # exact filename

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
