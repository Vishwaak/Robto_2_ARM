# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from . import agents

from .panda_train_env_cfg import FrankaCubeLiftDepthEnvCfg  # correct class name

gym.register(
    id="Isaac-Panda-Lift-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaCubeLiftDepthEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaLiftDepthPPORunnerCfg",
    },
)