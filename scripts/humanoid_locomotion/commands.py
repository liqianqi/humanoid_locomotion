"""Velocity command with a dedicated IN-PLACE-TURN mode.

Extends Isaac Lab's ``UniformVelocityCommand`` so that a fixed fraction of envs are commanded to
turn on the spot (vx=vy=0, sizable |yaw|). Straight walking and turn-in-place are thus trained as
SEPARATE regimes: the ordinary envs (heading-tracked or directly-sampled yaw) cover forward/curved
walking, and the in-place envs give the policy the coverage it needs to pivot at zero forward speed.

Why this is needed: with a plain UniformVelocityCommand, vx is sampled uniformly, so states with
"vx~=0 AND |yaw| large" are rare (~2%), and the policy never learns to turn on the spot (measured
in-place yaw 0.008 rad/s vs 0.5 commanded). Forcing a slice of envs into pure yaw fixes the coverage.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass


class InPlaceTurnVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand where a fraction of envs get a pure in-place yaw command."""

    cfg: InPlaceTurnVelocityCommandCfg

    def __init__(self, cfg: InPlaceTurnVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.is_inplace_turn_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        pick = torch.rand(ids.numel(), device=self.device) <= self.cfg.rel_inplace_turn_envs
        self.is_inplace_turn_env[ids] = pick
        turn_ids = ids[pick]
        if turn_ids.numel() == 0:
            return
        # pure yaw: zero the linear command, sample a sizable signed yaw rate
        self.vel_command_b[turn_ids, 0] = 0.0
        self.vel_command_b[turn_ids, 1] = 0.0
        lo, hi = self.cfg.inplace_turn_speed
        mag = torch.empty(turn_ids.numel(), device=self.device).uniform_(lo, hi)
        sign = torch.where(torch.rand(turn_ids.numel(), device=self.device) < 0.5, -1.0, 1.0)
        self.vel_command_b[turn_ids, 2] = mag * sign
        # keep the yaw we just set: don't let heading control overwrite it, and don't zero it as standing
        self.is_heading_env[turn_ids] = False
        self.is_standing_env[turn_ids] = False

    # No _update_command override needed: the parent only rewrites yaw for heading envs and zeros
    # standing envs, and in-place envs are excluded from both (flags cleared above), so their
    # [0, 0, yaw] command persists untouched.


@configclass
class InPlaceTurnVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = InPlaceTurnVelocityCommand

    # fraction of envs commanded to turn on the spot each resample
    rel_inplace_turn_envs: float = 0.18
    # magnitude range (rad/s) for the in-place yaw; kept above the reward "moving" yaw gate (0.3) and
    # within the ang_vel_z tracking range so track_ang_vel_z_exp can score it
    inplace_turn_speed: tuple[float, float] = (0.35, 0.5)
