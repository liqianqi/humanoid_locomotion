# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the RSX biped.

Motor map (left and right legs share the same models), from RobStride
RS series specs dated 2026-07-13:

* joint 1 (hip pitch): RS06
* joints 2-3 (hip roll, thigh yaw): RS02
* joint 4 (knee): RS06
* joint 5 (ankle): RS00
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

_RSX_URDF_PATH = str(Path(__file__).resolve().parents[2] / "assets" / "rsx" / "urdf" / "asm1.SLDASM.urdf")

# rpm -> rad/s (no-load speed from RobStride 2026-07-13 spec)
_RS00_VEL = 315.0 * 2.0 * math.pi / 60.0
_RS02_VEL = 410.0 * 2.0 * math.pi / 60.0
_RS06_VEL = 480.0 * 2.0 * math.pi / 60.0

# Reflected rotor inertia (J_rotor * gear_ratio^2) added to each joint, kg m^2. Not measured for RS series;
# 0.01 is the value Isaac Lab uses for the Unitree G1 legs/ankles and is the right order of magnitude for
# quasi-direct-drive actuators. Replace with datasheet values when available.
_ARMATURE = 0.01

RSX_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_RSX_URDF_PATH,
        fix_base=False,
        activate_contact_sensors=True,
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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Symmetric pre-squat (sign convention: hip +=flexion, knee +=flexion, ankle +=plantarflexion; requires the
        # knee axes to be consistent in the URDF, leg_r4_joint axis flipped on 2026-09-16). hip - knee - ankle = 0
        # keeps the torso level. Hip 0.13 / knee 0.40 / ankle -0.27 places the whole-body CoM ~1 cm ahead of the
        # ankle, inside the flat sole box; zero-action probe (2026-09-16): stands indefinitely, steady pitch ~0 rad
        # (0.10/-0.30 gave +0.065 rad forward lean, 0.15/-0.25 gave -0.05 rad backward lean).
        # Lowest sole point is ~0.3674 m below base_link in this pose -> 0.371 m leaves ~4 mm clearance.
        pos=(0.0, 0.0, 0.371),
        joint_pos={
            "leg_[lr]1_joint": 0.13,
            "leg_[lr]4_joint": 0.40,
            "leg_[lr]5_joint": -0.27,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # NOTE: DCMotorCfg is an *explicit* PD (torque applied every physics step, dt=5 ms). The RSX links are
    # very light (0.12-0.2 kg, I ~ 1e-4 kg m^2), so without reflected rotor inertia the discrete PD loop
    # is unstable (Kd*dt/I >> 2): joints reached 12-19 rad/s within 2 steps of touchdown. `armature`
    # adds the gearbox-reflected rotor inertia (same as G1's 0.01) and makes the loop stable.
    actuators={
        "rs06": DCMotorCfg(
            joint_names_expr=["leg_[lr][14]_joint"],
            effort_limit=11.0,
            saturation_effort=36.0,
            velocity_limit=_RS06_VEL,
            stiffness=60.0,
            damping=3.0,
            armature=_ARMATURE,
        ),
        "rs02": DCMotorCfg(
            joint_names_expr=["leg_[lr][23]_joint"],
            effort_limit=7.0,
            saturation_effort=17.0,
            velocity_limit=_RS02_VEL,
            stiffness=50.0,
            damping=2.5,
            armature=_ARMATURE,
        ),
        # Ankle stiffness must exceed the gravitational "negative stiffness" of the body pivoting about the
        # ankles, m*g*h_com/2 = 12.7*9.81*0.367/2 ~ 23 N m/rad, otherwise the zero-action stance topples.
        # 20 (G1 value) is below that for this robot; 40 gives a positive margin.
        "rs00": DCMotorCfg(
            joint_names_expr=["leg_[lr]5_joint"],
            effort_limit=5.0,
            saturation_effort=14.0,
            velocity_limit=_RS00_VEL,
            stiffness=40.0,
            damping=2.0,
            armature=_ARMATURE,
        ),
    },
)
"""RSX biped with RobStride RS06 / RS02 / RS00 DC-motor limits."""
