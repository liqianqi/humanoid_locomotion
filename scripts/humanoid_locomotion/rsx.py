# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the RSX biped -- ARMED whole-body model (UC Berkeley RS-X recipe).

Faithful port of the Berkeley robstride.py: same 18-DOF layout, init pose, and actuator gains
(these are the vendor/Berkeley-tuned values and are trusted). The ONE necessary difference is that
we load the self-contained URDF (their 210 MB USD is an unfetched Git-LFS stub) and convert it here.

CRITICAL: collider_type="convex_decomposition". The converter default "convex_hull" turns each
curved foot mesh into a single convex hull with a rounded sole -> the robot rocks and cannot stand
flat, which was the real reason training collapsed (same boat-sole trap as the legs-only model on
2026-09-16). Convex decomposition keeps the flat sole.
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_RSX_URDF_PATH = str(Path(__file__).resolve().parents[2] / "assets" / "rsx_arm" / "rsx.urdf")


def _rpm_to_rad_s(rpm: float) -> float:
    return rpm * (2.0 * math.pi / 60.0)


# RobStride datasheet (armature kg m^2, rated_torque N*m, rated_load_speed rpm).
_RS00_MINI = {"armature": 0.001, "torque": 5.0, "speed": 260.0}
_RS00 = {"armature": 0.001, "torque": 5.0, "speed": 260.0}
_RS02 = {"armature": 4.2e-3, "torque": 6.0, "speed": 360.0}
_RS03_MINI = {"armature": 0.015, "torque": 20.0, "speed": 180.0}


def _act(names, spec, stiffness, damping):
    return ImplicitActuatorCfg(
        joint_names_expr=names,
        armature=spec["armature"],
        effort_limit_sim=spec["torque"],
        velocity_limit_sim=_rpm_to_rad_s(spec["speed"]),
        stiffness=stiffness,
        damping=damping,
    )


RSX_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_RSX_URDF_PATH,
        fix_base=False,
        activate_contact_sensors=True,
        collider_type="convex_decomposition",  # keep the flat foot sole (see module docstring)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            "arm_left_shoulder_pitch_joint": 0.0,
            "arm_left_shoulder_roll_joint": 0.0,
            "arm_left_shoulder_yaw_joint": 0.0,
            "arm_left_elbow_pitch_joint": 0.0,
            "arm_right_shoulder_pitch_joint": 0.0,
            "arm_right_shoulder_roll_joint": 0.0,
            "arm_right_shoulder_yaw_joint": 0.0,
            "arm_right_elbow_pitch_joint": 0.0,
            "leg_left_hip_pitch_joint": -0.2,
            "leg_left_hip_roll_joint": 0.0,
            "leg_left_hip_yaw_joint": 0.0,
            "leg_left_knee_pitch_joint": 0.4,
            "leg_left_ankle_pitch_joint": -0.2,
            "leg_right_hip_pitch_joint": -0.2,
            "leg_right_hip_roll_joint": 0.0,
            "leg_right_hip_yaw_joint": 0.0,
            "leg_right_knee_pitch_joint": 0.4,
            "leg_right_ankle_pitch_joint": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # Berkeley-tuned gains (trusted vendor values).
    actuators={
        "shoulder_pitch": _act([".*_shoulder_pitch_joint"], _RS02, 40, 2),
        "shoulder_roll": _act([".*_shoulder_roll_joint"], _RS00, 40, 2),
        "shoulder_yaw": _act([".*_shoulder_yaw_joint"], _RS00_MINI, 40, 2),
        "elbow": _act([".*_elbow_pitch_joint"], _RS00, 40, 2),
        "hip_pitch": _act([".*_hip_pitch_joint"], _RS03_MINI, 60, 2),
        "hip_roll_yaw": _act([".*_hip_roll_joint", ".*_hip_yaw_joint"], _RS02, 40, 2),
        "knee": _act([".*_knee_pitch_joint"], _RS03_MINI, 40, 2),
        "ankle": _act([".*_ankle_pitch_joint"], _RS00, 10, 2),
    },
)
"""RSX armed whole-body robot (RobStride actuators, Berkeley gains)."""

RSX_JOINTS = list(RSX_CFG.init_state.joint_pos.keys())
