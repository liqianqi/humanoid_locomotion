# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the armed RSX (Berkeley RS-X recipe, with mirror symmetry)."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from humanoid_locomotion.symmetry import symmetry_data_augmentation_function


def _algorithm() -> RslRlPpoAlgorithmCfg:
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
    # Symmetry OFF. Confirmed 3x that mirror symmetry (even with the clone-swap bug fixed and a
    # clock-free reward set) collapses training here -> ~92% falls, no stepping. Instead the L/R
    # asymmetry (limp) is fixed with a phase-clock gait reward that forces BOTH feet to alternate.
    cfg.symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=False,
        use_mirror_loss=False,
        data_augmentation_func=symmetry_data_augmentation_function,
        mirror_loss_coeff=0.1,
    )
    return cfg


def _policy() -> RslRlPpoActorCriticCfg:
    cfg = RslRlPpoActorCriticCfg()
    cfg.init_noise_std = 1.0
    cfg.actor_hidden_dims = [256, 128, 128]
    cfg.critic_hidden_dims = [256, 128, 128]
    cfg.activation = "elu"
    return cfg


@configclass
class RsxFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000  # converges (episode length 1000) by ~1100; 3000 is plenty and faster
    save_interval = 200
    experiment_name = "rsx_flat"
    empirical_normalization = False  # Berkeley sets this explicitly; keep obs un-normalized to match
    policy = _policy()
    algorithm = _algorithm()


@configclass
class RsxRoughPPORunnerCfg(RsxFlatPPORunnerCfg):
    experiment_name = "rsx_rough"
