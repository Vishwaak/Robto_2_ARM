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
    experiment_name = "fr3_lift_depth_PPO_obj_pos"
    run_name = "lift_depth_ppo_128_img_new_fov"
    log_root_path = "/home/xerous/Desktop/project/logs/"
    logger = "wandb"
    wandb_project = "isaac-fr3"
    # load_run = "2026-04-24_12-18-53_lift_reach_extended"
    # model = 'model_500.pt'
    # load_run ="2026-04-23_20-54-35_lift_reach"
    # load_checkpoint='model_2400.pt'

    obs_groups = {
        "actor": ["depth", "student"],  # actor: depth CNN + proprio
        "critic": ["critic"],            # privileged critic
    }

    # obs_groups = {
    #     "actor": ["teacher"],  # actor: depth CNN + proprio
    #     "critic": ["critic"],            # privileged critic
    # }

    
    

    actor = RslRlCNNModelCfg(
        class_name="panda_train.modules.network.CNNModelWithAux",
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=1.0,
        cnn_cfg={
            "depth": RslRlCNNModelCfg.CNNCfg(
                output_channels=[32, 64], #[32,64,128]
                kernel_size=5, #3
                stride=3, #2
                norm='none',
                activation="elu",
                global_pool="max",
            )
        },
    )

    # actor = RslRlMLPModelCfg(
    #     hidden_dims=[256, 128, 64],
    #     activation="elu",
    #     obs_normalization=False,
    #     stochastic=True,
    #     init_noise_std=1.0,

    # )

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
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class DistillationRunnerCfg(RslRlDistillationRunnerCfg):

    load_run = "2026-04-24_15-36-31_lift_reach_extended"
    num_steps_per_env = 96
    max_iterations = 5000
    save_interval = 100
    empirical_normalization = False
    experiment_name = "fr3_lift_depth_PPO_obj_pos"
    run_name = "franka_lift_object_pos_distil"
    log_root_path = "/home/xerous/Desktop/project/logs/"
    logger = "wandb"
    wandb_project = "isaac-fr3"

    # student: RslRlCNNModelCfg = RslRlCNNModelCfg(
    #     class_name="panda_train.modules.network.CNNModelWithAux",
    #     hidden_dims=[256, 128, 64],
    #     activation="elu",
    #     obs_normalization=False,
    #     stochastic=True,
    #     init_noise_std=1.0,
    #     cnn_cfg={
    #         "depth": RslRlCNNModelCfg.CNNCfg(
    #             output_channels=[32,64],
    #             kernel_size=3,
    #             stride=2,
    #             norm='none',
    #             activation="elu",
    #             global_pool="max",  
    #         )
    #     },
    # )

    student: RslRlMLPModelCfg = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=1.0,

    )

    load_checkpoint = 'model_900.pt'
    teacher: RslRlMLPModelCfg = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=1.0,
    )

    algorithm: RslRlDistillationAlgorithmCfg = RslRlDistillationAlgorithmCfg(
        # class_name = "panda_train.modules.custom_runner.DistillationWithAux",
        num_learning_epochs=8,
        learning_rate=5e-4,
        gradient_length=36,
        max_grad_norm=1.0,
        loss_type="huber",
        optimizer="adam",
    )

    obs_groups = {
    "student": ["teacher"],
    "teacher": ["teacher"],
    }

    # Teacher PPO checkpoint
   