# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity locomotion env for the RSX biped, adapted from the G1 flat/rough configs."""

from copy import deepcopy
from typing import Any, cast

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from . import mdp as rsx_mdp
from .rsx import RSX_CFG

_FOOT_BODY = "leg_[lr]5_Link"
_ANKLE_JOINT = "leg_[lr]5_joint"
_FOOT_LR = ["leg_l5_Link", "leg_r5_Link"]
_HIP_AUX_JOINTS = ["leg_[lr]2_joint", "leg_[lr]3_joint"]
_LEG_JOINTS = ["leg_[lr][1-4]_joint"]
_TOTAL_JOINTS = ["leg_[lr][1-5]_joint"]

@configclass
class RsxRewards(RewardsCfg):
    """Reward terms for the RSX biped."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # std=0.25 (G1 uses 0.5 for 0-1 m/s): with commands <= 0.5 m/s a looser kernel still pays
    # ~0.7 for standing still at 0.3 m/s, which biases the policy toward not walking.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.25}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_BODY),
            "threshold": 0.30,
        },
    )
    # Dense gait-clock reward: alternating single-support schedule (period 0.7 s). This is the main
    # driver toward a real cross-step gait; it gives a smooth gradient the shuffle optimum lacks.
    # Paired with the gait_phase observation so the policy can time its stride.
    gait_contact = RewTerm(
        func=cast(Any, rsx_mdp.feet_gait_contact),
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LR, preserve_order=True),
            "period": 0.7,
            "offset": 0.5,
            "stance_ratio": 0.55,
        },
    )
    # Only prolonged double-support (shuffle). Brief contact after a real step is allowed.
    double_stance = RewTerm(
        func=cast(Any, rsx_mdp.double_stance),
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LR, preserve_order=True),
            "hold_time": 0.20,
        },
    )
    # Crossing step: landing foot must be ahead of the stance foot (~14 cm).
    landing_overstep = RewTerm(
        func=cast(Any, rsx_mdp.landing_overstep),
        weight=6.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LR, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LR, preserve_order=True),
            "target_overstep": 0.14,
            "min_air_time": 0.18,
            "min_travel": 0.10,
        },
    )
    swing_foot_clearance = RewTerm(
        func=rsx_mdp.swing_foot_clearance,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LR, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LR, preserve_order=True),
            "target_height": 0.04,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_BODY),
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_BODY),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[_ANKLE_JOINT])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_HIP_AUX_JOINTS)},
    )


@configclass
class RsxRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: RewardsCfg = RsxRewards()

    def __post_init__(self):
        super().__post_init__()
        robot = deepcopy(RSX_CFG)
        robot.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.robot = robot
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        self.events.push_robot = None  # type: ignore[assignment]
        self.events.add_base_mass = None  # type: ignore[assignment]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None  # type: ignore[assignment]

        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None  # type: ignore[assignment]
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.dof_acc_l2.weight = -1.25e-6
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=_TOTAL_JOINTS)
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["leg_[lr][1-5]_joint"]
        )

        self.actions.joint_pos.scale = 0.4
        # Stage 1 curriculum: forward walking only. Keep PLAY / WASD commands inside these ranges.
        # 20% standing envs (was 50%) so that most envs receive the feet_air_time gait reward.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.2
        # A real cross-step stride is only *required* above ~0.4 m/s; at 0.15-0.35 m/s the velocity
        # reward is fully satisfied by a shuffle, which was the local optimum the policy fell into.
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.9)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Gait-phase clock so the policy can synchronise its stride with the feet_gait_contact reward.
        self.observations.policy.gait_phase = ObsTerm(  # type: ignore[attr-defined]
            func=rsx_mdp.gait_phase, params={"period": 0.7, "offset": 0.5}
        )

        # Self-collisions are disabled in RSX_CFG, so any base_link contact is a real fall.
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"
        self.terminations.base_contact.params["threshold"] = 1.0


@configclass
class RsxRoughEnvCfg_PLAY(RsxRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Fixed forward command inside the training range; no random standing envs.
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None  # type: ignore[assignment]
        self.events.push_robot = None  # type: ignore[assignment]


@configclass
class RsxFlatEnvCfg(RsxRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None  # type: ignore[assignment]
        self.observations.policy.height_scan = None  # type: ignore[assignment]
        self.curriculum.terrain_levels = None  # type: ignore[assignment]

        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.dof_acc_l2.weight = -1.0e-6
        self.rewards.feet_air_time.weight = 1.25
        self.rewards.feet_air_time.params["threshold"] = 0.30
        self.rewards.gait_contact.weight = 1.5  # type: ignore[attr-defined]
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=_LEG_JOINTS)

        # Command ranges are inherited from RsxRoughEnvCfg (slow walk 0.15-0.35 m/s).


@configclass
class RsxFlatEnvCfg_PLAY(RsxFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Fixed forward command inside the training range; no random standing envs.
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None  # type: ignore[assignment]
        self.events.push_robot = None  # type: ignore[assignment]
