"""Play a trained RSX policy with WASD velocity commands.

Standalone replacement for Isaac Lab's ``rsl_rl/play_wasd.py`` (only ``cli_args`` is reused from
there). The key sensitivities default to the *training* command range of the current RSX config
(forward 0.4-0.9 m/s, yaw +/-0.4 rad/s; strafe disabled -- unicycle model). Override to stay inside
distribution or to probe out-of-distribution:

    /home/ubuntu/IsaacLab/isaaclab.sh -p scripts/play_wasd.py --task Isaac-Velocity-Flat-RSX-Play-v0 \\
        --num_envs 1 --checkpoint <model.pt> --vx_max 0.6 --vy_max 0.0 --wz_max 0.4

W/S forward-back, A/D strafe, Q/E turn. Set any of --vx_max/--vy_max/--wz_max to 0 to disable that axis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

from isaaclab.app import AppLauncher

import humanoid_locomotion  # noqa: F401

_RSL_RL_DIRS = [
    Path("/home/ubuntu/IsaacLab/scripts/reinforcement_learning/rsl_rl"),
    Path(__file__).resolve().parents[2].parent / "IsaacLab/scripts/reinforcement_learning/rsl_rl",
]
for _d in _RSL_RL_DIRS:
    if (_d / "cli_args.py").is_file():
        sys.path.insert(0, str(_d))
        break
else:
    raise FileNotFoundError("Could not find Isaac Lab rsl_rl/cli_args.py")

import cli_args  # isort: skip  # noqa: E402

parser = argparse.ArgumentParser(description="Play a trained RSL-RL policy with WASD velocity control.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time if possible.")
parser.add_argument("--allow_backward", action="store_true", default=False, help="Allow negative vx (S).")
# Command magnitudes. Defaults match the training ranges in velocity_env_cfg.RsxRoughEnvCfg
# (lin_vel_x 0.4-0.9, lin_vel_y +/-0.15, ang_vel_z +/-0.4).
parser.add_argument("--vx_max", type=float, default=0.6, help="Forward speed sent by W (m/s).")
parser.add_argument("--vy_max", type=float, default=0.0, help="Strafe speed sent by A/D (m/s). 0 = disabled (unicycle model: forward+turn only).")
parser.add_argument("--wz_max", type=float, default=0.4, help="Yaw rate sent by Q/E (rad/s). 0 = disabled.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import class_to_dict  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

installed_version = metadata.version("rsl-rl-lib")


class Se2WASDKeyboard(Se2Keyboard):
    """SE(2) keyboard with WASD plus Q/E yaw."""

    def __str__(self) -> str:
        msg = f"Keyboard Controller for SE(2): {self.__class__.__name__}\n"
        msg += f"\tW / S : forward {self.v_x_sensitivity:.2f} m/s / backward (or stop if backward is disabled)\n"
        msg += f"\tA / D : strafe left / right {self.v_y_sensitivity:.2f} m/s\n"
        msg += f"\tQ / E : turn left / right {self.omega_z_sensitivity:.2f} rad/s\n"
        msg += "\tL     : stop (zero command)\n"
        msg += "\tArrow keys and numpad also work.\n"
        msg += "\tClick the Isaac Sim viewport first. Do not hold RMB (that is camera fly mode)."
        return msg

    def _create_key_bindings(self):
        fwd = np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity
        left = np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity
        ccw = np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity
        self._INPUT_KEY_MAPPING = {
            "W": fwd, "S": -fwd, "A": left, "D": -left, "Q": ccw, "E": -ccw,
            "NUMPAD_8": fwd, "UP": fwd, "NUMPAD_2": -fwd, "DOWN": -fwd,
            "NUMPAD_4": left, "LEFT": left, "NUMPAD_6": -left, "RIGHT": -left,
            "NUMPAD_7": ccw, "Z": ccw, "NUMPAD_9": -ccw, "X": -ccw,
        }


def _prepare_keyboard_commands(env_cfg: ManagerBasedRLEnvCfg):
    """Stop random resampling / heading override so the keyboard owns the command."""
    cmd_cfg = cast(CommandsCfg, env_cfg.commands).base_velocity
    cmd_cfg.heading_command = False
    cmd_cfg.rel_heading_envs = 0.0
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.ranges.lin_vel_x = (0.0, 0.0)
    cmd_cfg.ranges.lin_vel_y = (0.0, 0.0)
    cmd_cfg.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(cmd_cfg.ranges, "heading"):
        cmd_cfg.ranges.heading = (0.0, 0.0)


_dbg_counter = [0]


def _apply_keyboard_command(env, keyboard: Se2WASDKeyboard, allow_backward: bool):
    command = keyboard.advance().to(env.unwrapped.device)
    if not allow_backward:
        command[0] = torch.clamp(command[0], min=0.0)
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.cfg.heading_command = False
    term.is_heading_env[:] = False
    term.is_standing_env[:] = False
    term.time_left[:] = 1.0e6
    term.vel_command_b[:] = command
    # DEBUG: print the live command ~2x/sec so you can see whether Q/E actually change wz.
    _dbg_counter[0] += 1
    if _dbg_counter[0] % 25 == 0:
        c = command.tolist()
        print(f"[cmd] vx={c[0]:+.2f}  vy={c[1]:+.2f}  wz={c[2]:+.2f}", flush=True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    runner_cfg = cast(RslRlOnPolicyRunnerCfg, agent_cfg)
    env_cfg.seed = runner_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _prepare_keyboard_commands(env_cfg)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", runner_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, runner_cfg.load_run, runner_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    task_env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg))
    env = RslRlVecEnvWrapper(task_env, clip_actions=runner_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    print("[WARN] The env uses the *current* task config, not the env.yaml saved next to the checkpoint.")
    runner_kwargs = class_to_dict(runner_cfg)
    if runner_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, runner_kwargs, log_dir=None, device=runner_cfg.device)
    elif runner_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, runner_kwargs, log_dir=None, device=runner_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {runner_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    keyboard = Se2WASDKeyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=args_cli.vx_max, v_y_sensitivity=args_cli.vy_max, omega_z_sensitivity=args_cli.wz_max
        )
    )
    print(keyboard)
    if not args_cli.allow_backward:
        print("[INFO] S is treated as stop. Pass --allow_backward to send negative vx.")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            _apply_keyboard_command(env, keyboard, args_cli.allow_backward)
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    # keep relative "logs/..." lookups and Hydra's "outputs/" anchored to the project root, not to
    # whatever directory this was launched from
    os.chdir(Path(__file__).resolve().parents[1])
    cast(Any, main)()
    simulation_app.close()
