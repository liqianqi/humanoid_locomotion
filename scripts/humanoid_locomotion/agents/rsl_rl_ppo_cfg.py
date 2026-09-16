# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


def _rsx_policy(hidden_dims: list[int]) -> RslRlPpoActorCriticCfg:
    # @configclass does not expose a typed __init__; set fields after construction.
    cfg = RslRlPpoActorCriticCfg()
    cfg.init_noise_std = 1.0
    cfg.actor_obs_normalization = False
    cfg.critic_obs_normalization = False
    cfg.actor_hidden_dims = hidden_dims
    cfg.critic_hidden_dims = hidden_dims
    cfg.activation = "elu"
    return cfg


def _rsx_algorithm() -> RslRlPpoAlgorithmCfg:
    cfg = RslRlPpoAlgorithmCfg()
    cfg.value_loss_coef = 1.0
    cfg.use_clipped_value_loss = True
    cfg.clip_param = 0.2
    cfg.entropy_coef = 0.008
    cfg.num_learning_epochs = 5
    cfg.num_mini_batches = 4
    cfg.learning_rate = 1.0e-3
    cfg.schedule = "adaptive"
    cfg.gamma = 0.99
    cfg.lam = 0.95
    cfg.desired_kl = 0.01
    cfg.max_grad_norm = 1.0
    return cfg


@configclass
class RsxRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "rsx_rough"
    policy = _rsx_policy([512, 256, 128])
    algorithm = _rsx_algorithm()


@configclass
class RsxFlatPPORunnerCfg(RsxRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()  # type: ignore[misc]

        self.max_iterations = 1500
        self.experiment_name = "rsx_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
