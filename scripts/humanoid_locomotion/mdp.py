"""RSX-specific reward terms."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def _cmd_moving(env: ManagerBasedRLEnv, command_name: str, cmd_threshold: float) -> torch.Tensor:
    return torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > cmd_threshold


def _contact_sensor(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> ContactSensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    if not isinstance(sensor, ContactSensor):
        raise TypeError(f"Expected ContactSensor at '{sensor_cfg.name}', got {type(sensor).__name__}")
    return sensor


def _need_time(value: torch.Tensor | None, name: str) -> torch.Tensor:
    if value is None:
        raise RuntimeError(f"ContactSensor.{name} is None; enable track_air_time on the sensor.")
    return value


def _feet_progress_along_cmd(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    body_ids: list[int] | slice,
    command_name: str,
) -> torch.Tensor:
    """Foot positions along the commanded heading, in the yaw-aligned base frame. Shape [N, n_feet]."""
    feet_w = asset.data.body_pos_w[:, body_ids, :3]
    n_env, n_feet, _ = feet_w.shape
    rel = feet_w - asset.data.root_pos_w[:, None, :3]
    q = yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(-1, n_feet, -1).reshape(-1, 4)
    feet_yaw = quat_apply_inverse(q, rel.reshape(-1, 3)).reshape(n_env, n_feet, 3)
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    cmd_n = cmd_xy / torch.clamp(torch.norm(cmd_xy, dim=1, keepdim=True), min=1.0e-6)
    return torch.sum(feet_yaw[..., :2] * cmd_n.unsqueeze(1), dim=-1)


def _leg_phase(env: ManagerBasedRLEnv, period: float, offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env gait phase in [0, 1) for the left and right leg.

    The clock is stateless: it is derived from ``episode_length_buf`` so it resets to 0 on every
    env reset and is identical between the observation term and the reward term (they must agree, or
    the policy is rewarded for matching a clock it cannot see). ``offset`` puts the right leg in
    anti-phase (0.5) so the two feet alternate.
    """
    t = (env.episode_length_buf.float() * env.step_dt) % period / period
    left = t
    right = (t + offset) % 1.0
    return left, right


def gait_phase(
    env: ManagerBasedRLEnv, period: float = 0.7, offset: float = 0.5
) -> torch.Tensor:
    """Observation: (sin, cos) of the left-leg gait phase so the policy can time its own gait."""
    left, _ = _leg_phase(env, period, offset)
    two_pi = 2.0 * math.pi
    return torch.stack([torch.sin(two_pi * left), torch.cos(two_pi * left)], dim=1)


def feet_gait_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    period: float = 0.7,
    offset: float = 0.5,
    stance_ratio: float = 0.55,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Dense gait-clock reward: +1 per foot whose contact state matches the scheduled stance/swing.

    Expected schedule (per leg): stance while its phase < ``stance_ratio`` (slight >0.5 gives a short
    double-support overlap), swing otherwise. Rewarding the match to an alternating single-support
    schedule gives a smooth gradient toward a real cross-step gait, unlike the all-or-nothing
    ``landing_overstep``. Off for near-zero commands so standing envs are not forced to march.
    Body order is [left, right] (sensor_cfg must use preserve_order=True).
    """
    contact = _contact_sensor(env, sensor_cfg)
    contact_time = _need_time(contact.data.current_contact_time, "current_contact_time")[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    left, right = _leg_phase(env, period, offset)
    left_stance = left < stance_ratio
    right_stance = right < stance_ratio
    match_left = (in_contact[:, 0] == left_stance).float()
    match_right = (in_contact[:, 1] == right_stance).float()
    reward = match_left + match_right
    return reward * _cmd_moving(env, command_name, cmd_threshold)


def forward_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Linear reward for speed ALONG the commanded direction, capped at the commanded speed.

    The exp velocity tracker is flat-zero far from target, so once the policy drifts into walking
    backward there is no gradient to climb back (the 2026-09-18 probe found 79% of samples moving
    backward at vx=-0.185 while commanded +0.6). This term keeps a constant +gradient for every
    velocity below command -- including negative -- so forward is always uphill. It is capped at the
    commanded speed (cannot be farmed by overspeeding), pays 0 for standing and a negative for moving
    backward. Off for near-zero commands.
    """
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    cmd_norm = torch.norm(cmd, dim=1)
    cmd_dir = cmd / cmd_norm.clamp(min=1.0e-6).unsqueeze(1)
    vel_xy = asset.data.root_lin_vel_b[:, :2]
    v_along = torch.sum(vel_xy * cmd_dir, dim=1)
    reward = torch.minimum(v_along, cmd_norm)
    return reward * _cmd_moving(env, command_name, cmd_threshold)


def _base_yaw(asset: Articulation) -> torch.Tensor:
    """World yaw of the base (rad) from its quaternion (w, x, y, z)."""
    q = asset.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class heading_hold(ManagerTermBase):
    """Hold the heading captured at reset, whenever no turn is commanded.

    The systematic heading drift is a *learned* left/right gait asymmetry (its sign varies run to
    run), which reward-strength tweaks cannot fix. This term instead penalizes NET heading deviation
    from a per-env reference directly -- the VIO/nav metric itself -- so the policy must actively
    correct whatever asymmetry it has. Non-saturating (quadratic) so far-off headings still pull back.
    While a yaw rate IS commanded the reference follows the current heading and the penalty is off, so
    commanded turns are not fought.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.ref_yaw = _base_yaw(env.scene["robot"]).clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        yaw = _base_yaw(self._env.scene["robot"])
        if env_ids is None:
            self.ref_yaw = yaw.clone()
        else:
            self.ref_yaw[env_ids] = yaw[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "base_velocity",
        cmd_yaw_threshold: float = 0.1,
    ) -> torch.Tensor:
        cur = _base_yaw(env.scene["robot"])
        cmd_yaw = env.command_manager.get_command(command_name)[:, 2]
        turning = cmd_yaw.abs() >= cmd_yaw_threshold
        # while turning, let the reference follow the heading so commanded turns are free
        self.ref_yaw = torch.where(turning, cur, self.ref_yaw)
        err = torch.atan2(torch.sin(cur - self.ref_yaw), torch.cos(cur - self.ref_yaw))
        return torch.square(err) * (~turning).float()


def lateral_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Quadratic penalty on velocity PERPENDICULAR to the commanded direction (cross-track).

    The exp tracker penalizes 2D velocity error but saturates; this keeps a sharp gradient on the
    sideways component that makes the velocity vector wobble ~16 deg off-command and the path weave.
    Straighter path -> steadier heading for VIO/nav. Off for near-zero commands.
    """
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    cmd_norm = torch.norm(cmd, dim=1)
    cmd_dir = cmd / cmd_norm.clamp(min=1.0e-6).unsqueeze(1)
    perp = torch.stack([-cmd_dir[:, 1], cmd_dir[:, 0]], dim=1)  # 90deg left of command
    vel_xy = asset.data.root_lin_vel_b[:, :2]
    v_perp = torch.sum(vel_xy * perp, dim=1)
    return torch.square(v_perp) * _cmd_moving(env, command_name, cmd_threshold)


def ang_vel_z_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Quadratic penalty on yaw-rate tracking error (body frame).

    The exp tracking reward saturates to a flat zero far from target (at the measured ~1.3 rad/s
    wobble its gradient vanishes), so the policy never learns to damp the violent per-step yaw twist.
    An L2 penalty keeps a strong gradient at every error magnitude. It penalizes deviation from the
    *commanded* yaw rate, so commanded turns are not punished.
    """
    asset: Articulation = env.scene["robot"]
    cmd_z = env.command_manager.get_command(command_name)[:, 2]
    yaw_rate = asset.data.root_ang_vel_b[:, 2]
    return torch.square(yaw_rate - cmd_z)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """L1 joint deviation from the default pose, active ONLY when the command is ~zero (standing).

    All gait-shaping rewards are gated off at zero command, so nothing otherwise constrains the feet
    when standing and they freeze in whatever (e.g. fore-aft crossed) pose the policy last held. This
    pulls the whole stance back to the symmetric default while standing. It is the mirror gate of
    ``_cmd_moving`` and must not overlap the walking regime, or it fights the gait.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(angle), dim=1)
    standing = ~_cmd_moving(env, command_name, cmd_threshold)
    return deviation * standing.float()


def swing_foot_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target_height: float = 0.06,
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward swing-foot height up to ``target_height`` (plane world z). Stance and zero-cmd are 0."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact = _contact_sensor(env, sensor_cfg)
    in_air = _need_time(contact.data.current_air_time, "current_air_time")[:, sensor_cfg.body_ids] > 0.0
    height = torch.clamp(asset.data.body_pos_w[:, asset_cfg.body_ids, 2], min=0.0, max=target_height)
    reward = torch.sum(height * in_air.float(), dim=1)
    return reward * _cmd_moving(env, command_name, cmd_threshold)


class double_stance(ManagerTermBase):
    """Penalize walking without a real single-stance for longer than ``hold_time``.

    Contact flicker used to reset ``current_contact_time`` and dodge a simple hold
    check. Chatter is treated as "down", so a split-shuffle still accumulates.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.both_down_time = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.both_down_time.zero_()
        else:
            self.both_down_time[env_ids] = 0.0

    def __call__(  # type: ignore[override]
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        hold_time: float = 0.20,
        cmd_threshold: float = 0.1,
        debounce: float = 0.05,
    ) -> torch.Tensor:
        contact = _contact_sensor(env, sensor_cfg)
        contact_time = _need_time(contact.data.current_contact_time, "current_contact_time")[:, sensor_cfg.body_ids]
        air_time = _need_time(contact.data.current_air_time, "current_air_time")[:, sensor_cfg.body_ids]
        down = contact_time > debounce
        up = air_time > debounce
        effective_down = down | ~(down | up)
        both_down = torch.all(effective_down, dim=1)
        self.both_down_time = torch.where(
            both_down, self.both_down_time + env.step_dt, torch.zeros_like(self.both_down_time)
        )
        return (self.both_down_time > hold_time).float() * _cmd_moving(env, command_name, cmd_threshold)


class landing_overstep(ManagerTermBase):
    """Pay only when the trailing foot swings forward and lands past the stance foot.

    The previous stateless version was farmed by keeping a split stance and
    chattering the already-leading foot (``first_contact`` + large overstep,
    but ``last_air_time`` ~ 0). A valid step must:

    * start behind the other foot
    * stay in the air at least ``min_air_time``
    * travel at least ``min_travel`` along the command
    * land ahead of the stance foot
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.takeoff_own = torch.zeros(env.num_envs, 2, device=env.device)
        self.takeoff_other = torch.zeros(env.num_envs, 2, device=env.device)
        self.was_air = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.takeoff_own.zero_()
            self.takeoff_other.zero_()
            self.was_air.zero_()
        else:
            self.takeoff_own[env_ids] = 0.0
            self.takeoff_other[env_ids] = 0.0
            self.was_air[env_ids] = False

    def __call__(  # type: ignore[override]
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        target_overstep: float = 0.14,
        min_air_time: float = 0.18,
        min_travel: float = 0.10,
        cmd_threshold: float = 0.1,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        contact = _contact_sensor(env, sensor_cfg)
        air_time = _need_time(contact.data.current_air_time, "current_air_time")[:, sensor_cfg.body_ids]
        in_air = air_time > 0.0
        progress = _feet_progress_along_cmd(env, asset, asset_cfg.body_ids, command_name)

        just_takeoff = in_air & ~self.was_air
        if torch.any(just_takeoff):
            self.takeoff_own = torch.where(just_takeoff, progress, self.takeoff_own)
            self.takeoff_other = torch.where(just_takeoff, progress.flip(dims=[1]), self.takeoff_other)

        first_contact = contact.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        last_air = _need_time(contact.data.last_air_time, "last_air_time")[:, sensor_cfg.body_ids]
        other_down = (
            _need_time(contact.data.current_contact_time, "current_contact_time")[:, sensor_cfg.body_ids].flip(
                dims=[1]
            )
            > 0.05
        )
        overstep = progress - progress.flip(dims=[1])
        travel = progress - self.takeoff_own
        started_behind = self.takeoff_own < self.takeoff_other + 0.02
        valid = (
            first_contact
            & other_down
            & started_behind
            & (last_air > min_air_time)
            & (travel > min_travel)
            & (overstep > 0.04)
        )
        score = torch.clamp(overstep / target_overstep, min=0.0, max=1.0)
        reward = torch.sum(score * valid.float(), dim=1)

        self.was_air = in_air
        return reward * _cmd_moving(env, command_name, cmd_threshold)
