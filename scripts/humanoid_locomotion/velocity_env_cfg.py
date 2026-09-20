# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity locomotion env for the ARMED RSX biped.

Rewritten on 2026-09-19 to follow the UC Berkeley RS-X recipe: armed whole-body robot, a *simple,
clock-free* reward set, heading-command tracking, and left/right mirror symmetry (data augmentation
+ mirror loss). The elaborate gait machinery of the legs-only project (phase clock, gait_contact,
landing_overstep, forward_progress, ...) is gone -- symmetry + arms + feet_air_time produce the gait.
Structure mirrors the Berkeley env_cfg so the two can be read side by side.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from typing import Any, cast

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from . import mdp as rsx_mdp
from .rsx import RSX_CFG, RSX_JOINTS
from .symmetry import SymmetryCfg


##
# Scene
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Flat terrain scene with the armed RSX robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = RSX_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")  # pyright: ignore[reportAttributeAccessIssue]
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP
##
@configclass
class RSXObservations:
    """Observation specifications. No gait-phase clock (it broke left/right symmetry)."""

    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RSX_JOINTS, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RSX_JOINTS, preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action)
        # gait-phase clock (sin,cos) so the policy can time the alternating step schedule.
        gait_phase = ObsTerm(func=cast(Any, rsx_mdp.gait_phase), params={"period": 0.7, "offset": 0.5})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RSXCommands:
    """Heading-tracked velocity command (holds a commanded heading -> low net drift)."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # Forward + turn only (no backward, no strafe): a focused, learnable task that yields a clean
        # cross-step gait; backward command was producing the sit-back/lean. heading_command holds
        # the commanded heading -> low net yaw drift.
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.9),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class RSXActions:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RSX_JOINTS,
        scale=0.25,
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class RSXRewards:
    """Simple, clock-free reward set (Berkeley RS-X)."""

    # -40 (was -100): a big fall penalty made the policy too afraid of single-support to ever step
    # (it froze/shuffled). Lower it so committing to a real step is worth the risk.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-40.0)

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25)
    # Reduce body sway: penalize torso roll/pitch angular velocity harder (this is the rocking rate,
    # from the base IMU gyro) plus a firmer upright (tilt) penalty.
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    # Keep the torso tall (~0.36 m) -> no crouch/sit-back.
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 0.36, "asset_cfg": SceneEntityCfg("robot", body_names="torso")},
    )

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=4.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # weight 4 (was 8): 8 over-did it into a high-knee/kicking march. 4 gives real steps at normal height.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=4.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_.*"),
            "threshold": 0.25,
        },
    )
    # Small swing clearance (3 cm, was 8 cm): just enough to clear the ground / step over, not a high kick.
    swing_foot_clearance = RewTerm(
        func=cast(Any, rsx_mdp.swing_foot_clearance),
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_.*"),
            "target_height": 0.03,
        },
    )
    # Step LENGTH (horizontal), independent of knee height: reward the swing foot LANDING ahead of the
    # stance foot by ~one foot-length -> the trailing foot overtakes to become the new lead, a real
    # stride, not just catching up. Gated (must be airborne, travel forward, start behind) so it can't
    # be farmed by a planted split stance.
    # Phase-clock gait: forces BOTH feet to alternate single-support on schedule (period 0.7 s). This
    # is what fixes the limp (one leg striding, the other only following) -- both legs must swing on
    # their scheduled beat. Paired with the gait_phase observation so the policy can time it.
    gait_contact = RewTerm(
        func=cast(Any, rsx_mdp.feet_gait_contact),
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_leg_ankle_pitch", "right_leg_ankle_pitch"], preserve_order=True),
            "period": 0.7,
            "offset": 0.5,
            "stance_ratio": 0.6,
        },
    )
    landing_overstep = RewTerm(
        func=cast(Any, rsx_mdp.landing_overstep),
        weight=8.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_leg_ankle_pitch", "right_leg_ankle_pitch"], preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_leg_ankle_pitch", "right_leg_ankle_pitch"], preserve_order=True),
            "target_overstep": 0.16,  # bigger stride: swing foot lands ~a full foot-length past the stance foot
            "min_air_time": 0.15,
            "min_travel": 0.10,
        },
    )
    # Standing-still upright posture: when no command, gait rewards gate off and the torso would
    # otherwise settle into a leaned-back brace. Pull the legs back to the default (upright) stance.
    stand_still_posture = RewTerm(
        func=cast(Any, rsx_mdp.stand_still_joint_deviation_l1),
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"]),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_.*"),
        },
    )

    # -0.75 (was -0.2): at -0.2 the hips could roll/yaw far, giving a "stepover/circling" swing (leg
    # rotates inward then swings out) instead of a straight sagittal step. Strong penalty keeps the
    # legs in the sagittal plane for a stable Feishu-like stride.
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.75,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    # Lock the arms' LATERAL freedom (shoulder roll/yaw) hard -> arms stay in the sagittal plane and
    # cannot swing sideways into the body.
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"])},
    )
    # Arms held STILL and straight (no swing needed, per spec). Keeping shoulder-pitch + elbow near
    # default makes the arms hang straight and quiet; with the roll/yaw lock above they never swing
    # sideways into the body. (Active arm swing dropped: reward-shaped versions were gamed into static
    # arms or regressed the gait to a shuffle -- not worth risking the walk.)
    joint_deviation_arms_pitch = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint", ".*_elbow_pitch_joint"])},
    )


@configclass
class RSXTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["torso", ".*_shoulder_.*", ".*_elbow_.*", ".*_hip_.*", ".*_knee_.*"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class RSXEvents:
    physics_material = EventTerm(
        func=cast(Any, mdp.randomize_rigid_body_material),
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RSXCurriculum:
    pass


##
# Symmetry (left/right mirror map, indexed into RSX_JOINTS)
##
def _idx(name: str) -> int:
    return RSX_JOINTS.index(name)


# pitch joints (sagittal) -> SWAP
_SYM_SWAP = [
    (_idx("arm_left_shoulder_pitch_joint"), _idx("arm_right_shoulder_pitch_joint")),
    (_idx("arm_left_elbow_pitch_joint"), _idx("arm_right_elbow_pitch_joint")),
    (_idx("leg_left_hip_pitch_joint"), _idx("leg_right_hip_pitch_joint")),
    (_idx("leg_left_knee_pitch_joint"), _idx("leg_right_knee_pitch_joint")),
    (_idx("leg_left_ankle_pitch_joint"), _idx("leg_right_ankle_pitch_joint")),
]
# roll / yaw joints (lateral/transverse) -> SWAP + NEGATE
_SYM_SWAP_NEGATE = [
    (_idx("arm_left_shoulder_roll_joint"), _idx("arm_right_shoulder_roll_joint")),
    (_idx("arm_left_shoulder_yaw_joint"), _idx("arm_right_shoulder_yaw_joint")),
    (_idx("leg_left_hip_roll_joint"), _idx("leg_right_hip_roll_joint")),
    (_idx("leg_left_hip_yaw_joint"), _idx("leg_right_hip_yaw_joint")),
]


@configclass
class RsxSymmetry(SymmetryCfg):
    action_swap_terms = {"joint_pos": _SYM_SWAP}
    action_swap_negate_terms = {"joint_pos": _SYM_SWAP_NEGATE}
    action_negate_terms = {"joint_pos": []}

    observation_swap_terms = {"joint_pos": _SYM_SWAP, "joint_vel": _SYM_SWAP, "actions": _SYM_SWAP}
    observation_swap_negate_terms = {
        "joint_pos": _SYM_SWAP_NEGATE,
        "joint_vel": _SYM_SWAP_NEGATE,
        "actions": _SYM_SWAP_NEGATE,
    }
    observation_negate_terms = {
        "velocity_commands": [1, 2],   # vy, wz
        "base_lin_vel": [1],           # vy (critic obs)
        "base_ang_vel": [0, 2],        # roll rate, yaw rate
        "projected_gravity": [1],      # gy
    }


##
# Env assembly
##
@configclass
class RsxEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: RSXObservations = RSXObservations()
    commands: RSXCommands = RSXCommands()
    actions: RSXActions = RSXActions()
    rewards: RSXRewards = RSXRewards()
    terminations: RSXTerminations = RSXTerminations()
    events: RSXEvents = RSXEvents()
    curriculum: RSXCurriculum = RSXCurriculum()
    symmetry: SymmetryCfg = RsxSymmetry()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True  # pyright: ignore[reportAttributeAccessIssue]
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


# Flat is the real config; Rough is an alias (we only train on plane) so the registrations stay valid.
@configclass
class RsxFlatEnvCfg(RsxEnvCfg):
    pass


@configclass
class RsxRoughEnvCfg(RsxEnvCfg):
    pass


@configclass
class RsxFlatEnvCfg_PLAY(RsxFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # keyboard / fixed playback owns the command
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.events.base_external_force_torque = None  # pyright: ignore[reportAttributeAccessIssue]


@configclass
class RsxRoughEnvCfg_PLAY(RsxFlatEnvCfg_PLAY):
    pass
