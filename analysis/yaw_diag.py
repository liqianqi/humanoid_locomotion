"""Headless yaw-drift / gait-symmetry probe for the RSX flat policy.

Runs the trained policy under a FIXED forward command (no yaw) and measures the uncommanded
yaw rate, to tell apart two very different heading-drift causes:

  * SYSTEMATIC BIAS  -> the robot always turns the same way  => a left/right asymmetry
                        (URDF / default pose / gait phase). Fix the asymmetry, not the reward.
  * RANDOM WEAKNESS  -> the robot twists both ways, heading random-walks => yaw control is too
                        weak. Fix on the reward side (yaw-rate penalty / tracking weight).

It also prints left/right contact-time and hip-pitch to expose gait asymmetry.

Usage (from the repo `scripts/` dir, so `humanoid_locomotion` imports):
    /home/ubuntu/isaacsim_venv/bin/python -u analysis/yaw_diag.py
Edit RUN_DIR to probe a specific run; by default it auto-picks the newest rsx_flat run.

Notes learned the hard way:
  * run with `-u` (unbuffered) or the result never reaches the log until the process exits;
  * `simulation_app.close()` hangs for minutes on this build -> we flush and `os._exit(0)` instead.
"""

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys

import torch
import gymnasium as gym
import importlib.metadata as _im
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg

import humanoid_locomotion  # noqa: F401  (registers tasks)
from humanoid_locomotion.agents.rsl_rl_ppo_cfg import RsxFlatPPORunnerCfg

TASK = "Isaac-Velocity-Flat-RSX-Play-v0"
LOG_ROOT = "/home/ubuntu/humanoid_locomotion/logs/rsl_rl/rsx_flat"
RUN_DIR = sorted(glob.glob(f"{LOG_ROOT}/2026-*"))[-1]  # newest run; override to pin one
CKPT = sorted(glob.glob(f"{RUN_DIR}/model_*.pt"), key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]
NUM = 64
DEVICE = "cuda:0"
WARMUP = 120   # steps to settle before measuring
MEASURE = 400  # measured steps
FWD = 0.6      # fixed forward command (m/s)

env_cfg = parse_env_cfg(TASK, device=DEVICE, num_envs=NUM)
c = env_cfg.commands.base_velocity
c.rel_standing_envs = 0.0
c.ranges.lin_vel_x = (FWD, FWD)
c.ranges.lin_vel_y = (0.0, 0.0)
c.ranges.ang_vel_z = (0.0, 0.0)
c.ranges.heading = (0.0, 0.0)

env = gym.make(TASK, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

agent_cfg = handle_deprecated_rsl_rl_cfg(RsxFlatPPORunnerCfg(), _im.version("rsl-rl-lib"))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=DEVICE)
runner.load(CKPT)
policy = runner.get_inference_policy(device=env.unwrapped.device)
print(f"[diag] loaded {CKPT}", flush=True)

robot = env.unwrapped.scene["robot"]
contact = env.unwrapped.scene.sensors["contact_forces"]
jn = robot.data.joint_names
li_hipP = jn.index("leg_l1_joint"); ri_hipP = jn.index("leg_r1_joint")
foot_ids, _ = contact.find_bodies(["leg_l5_Link", "leg_r5_Link"], preserve_order=True)

obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]

yaw_rate_sum = torch.zeros(NUM, device=DEVICE)
yaw_rate_cnt = torch.zeros(NUM, device=DEVICE)
all_yaw = []; fwd_speed = []
lc_time = []; rc_time = []; lhip = []; rhip = []

step = 0
with torch.inference_mode():
    while step < WARMUP + MEASURE:
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        dones = dones.bool().view(-1)
        if step >= WARMUP:
            yr = robot.data.root_ang_vel_w[:, 2]   # world yaw rate; cmd=0 -> all of it is drift
            live = ~dones
            yaw_rate_sum += torch.where(live, yr, torch.zeros_like(yr))
            yaw_rate_cnt += live.float()
            all_yaw.append(yr[live].detach().clone())
            fwd_speed.append(robot.data.root_lin_vel_b[live, 0].detach().clone())
            ct = contact.data.current_contact_time[:, foot_ids]
            lc_time.append(ct[live, 0].detach().clone()); rc_time.append(ct[live, 1].detach().clone())
            lhip.append(robot.data.joint_pos[live, li_hipP].detach().clone())
            rhip.append(robot.data.joint_pos[live, ri_hipP].detach().clone())
        if step % 100 == 0:
            print(f"[diag] step {step}/{WARMUP + MEASURE}", flush=True)
        if dones.any():
            yaw_rate_sum[dones] = 0.0
            yaw_rate_cnt[dones] = 0.0
        step += 1

allyaw = torch.cat(all_yaw); fwd = torch.cat(fwd_speed)
per_env = yaw_rate_sum / yaw_rate_cnt.clamp(min=1)
pe = per_env[yaw_rate_cnt > 20]

print("\n==================== YAW-DRIFT DIAGNOSTIC ====================")
print(f"run={os.path.basename(RUN_DIR)}  envs={NUM}  steps={MEASURE}  (cmd: fwd {FWD} m/s, yaw 0)")
print(f"forward speed mean = {fwd.mean():+.3f} (target {FWD})")
print("\n-- Uncommanded yaw rate (rad/s) --")
print(f"  mean (signed) = {allyaw.mean():+.4f}   [near 0 = no systematic bias]")
print(f"  mean |rate|   = {allyaw.abs().mean():.4f}   [drift magnitude]")
print(f"  std           = {allyaw.std():.4f}")
print(f"  per-env LEFT/RIGHT: {(pe>0).sum().item()}/{(pe<0).sum().item()}")
ratio = abs(allyaw.mean().item()) / max(allyaw.abs().mean().item(), 1e-6)
print(f"  bias/noise ratio = {ratio:.2f}   [>~0.5 SYSTEMATIC BIAS; <~0.3 RANDOM weakness]")
print("\n-- Left/Right symmetry --")
lct = torch.cat(lc_time).mean().item(); rct = torch.cat(rc_time).mean().item()
lh = torch.cat(lhip).mean().item(); rh = torch.cat(rhip).mean().item()
print(f"  contact-time  L={lct:.3f} R={rct:.3f}  diff={lct-rct:+.3f}")
print(f"  hip-pitch     L={lh:+.3f} R={rh:+.3f}  diff={lh-rh:+.3f}")
print("=============================================================\n", flush=True)

sys.stdout.flush()
os._exit(0)  # simulation_app.close() hangs for minutes on this build
