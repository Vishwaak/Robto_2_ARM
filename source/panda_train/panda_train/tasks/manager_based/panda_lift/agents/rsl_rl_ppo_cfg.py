# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlDistillationRunnerCfg, RslRlDistillationStudentTeacherCfg, RslRlDistillationAlgorithmCfg, RslRlMLPModelCfg

from panda_train.modules.custom_runner import DistillationWithAux, PPOWithAux
from panda_train.modules.network import CNNModelWithAux

# @configclass
# class FrankaLiftDepthPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 96
#     max_iterations = 5000
#     save_interval = 100
#     experiment_name = "franka_lift_depth"
#     run_name = "pick_lift_policy"  # used for logging and checkpointing
#     log_root_path = "/home/xerous/Desktop/project/logs/"
#     # logger = "wandb"
#     wandb_project = "isaac-panda"

#     obs_groups = {
#     "policy": ["teacher"],
#     }

#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_obs_normalization=False,
#         critic_obs_normalization=False,
#         actor_hidden_dims=[256, 128, 64],
#         critic_hidden_dims=[256, 128, 64],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.001,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-4,
#         schedule="adaptive",
#         gamma=0.98,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )

@configclass
class FrankaLiftDepthPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 5000
    save_interval = 100
    experiment_name = "franka_lift_depth"
    run_name = "depth_ppo_cnn_finetunning_cuda_0"
    log_root_path = "/home/xerous/Desktop/project/logs/"
    logger = "wandb"
    wandb_project = "isaac-panda"
    load_run ="2026-04-18_12-57-47_depth_ppo_cnn"
    load_checkpoint='model_1200.pt'

    obs_groups = {
        "actor": ["depth", "student"],  # actor: depth CNN + proprio
        "critic": ["critic"],            # privileged critic
    }

    actor = RslRlCNNModelCfg(
        class_name="panda_train.modules.network.CNNModelWithAux",
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=1.0,
        cnn_cfg={
            "depth": RslRlCNNModelCfg.CNNCfg(
                output_channels=[32, 64, 128],
                kernel_size=3,
                stride=2,
                norm='none',
                activation="elu",
                global_pool="max",
            )
        },
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        stochastic=False,
        init_noise_std=1.0,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="panda_train.modules.custom_runner.PPOWithAux",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class DistillationRunnerCfg(RslRlDistillationRunnerCfg):

  
    load_run = "2026-04-16_20-43-27_pick_lift_policy"

    num_steps_per_env = 96
    max_iterations = 5000
    save_interval = 100
    empirical_normalization = False
    experiment_name = "franka_distillation_depth"
    run_name = "dist_train"
    log_root_path = "/home/xerous/Desktop/project/logs/"
    # logger = "wandb"
    wandb_project = "isaac-panda"

    student: RslRlCNNModelCfg = RslRlCNNModelCfg(
        class_name="panda_train.modules.network.CNNModelWithAux",
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=1.0,
        cnn_cfg={
            "depth": RslRlCNNModelCfg.CNNCfg(
                output_channels=[32,64, 3],
                kernel_size=3,
                stride=2,
                norm='none',
                activation="elu",
                global_pool="max",  # flattens spatial dims to vector
            )
        },
    )

    teacher: RslRlMLPModelCfg = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=1.0,
    )

    algorithm: RslRlDistillationAlgorithmCfg = RslRlDistillationAlgorithmCfg(
        class_name = "panda_train.modules.custom_runner.DistillationWithAux",
        num_learning_epochs=8,
        learning_rate=5e-4,
        gradient_length=36,
        max_grad_norm=1.0,
        loss_type="huber",
        optimizer="adam",
    )

    obs_groups = {
    "student": ["depth","student"],
    "teacher": ["teacher"],
    }

    # Teacher PPO checkpoint
   