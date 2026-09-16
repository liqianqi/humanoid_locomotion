# humanoid_locomotion

RSX 10-DoF 双足在 Isaac Lab 里的速度跟踪训练配置。不改 Isaac Lab 源码，用本仓库的 `scripts/train.py` / `play.py` / `play_wasd.py` 注册任务后转调官方 RSL-RL 脚本。

先安装本包：`pip install -e .`（须在 Isaac Sim 的 Python 环境里）。

## 环境

| 组件 | 版本 |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.2（`isaaclab` 0.54.3） |
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 |
| rsl-rl-lib | 5.0.1 |
| gymnasium | 1.2.1 |

本机路径：Isaac Lab `/home/ubuntu/IsaacLab`，解释器 `/home/ubuntu/isaacsim_venv/bin/python`。

## 任务

- 训练：`Isaac-Velocity-Flat-RSX-v0` / `Isaac-Velocity-Rough-RSX-v0`
- 回放：对应的 `*-Play-v0`

日志在 `logs/rsl_rl/rsx_flat/` 或 `rsx_rough/`。

## 训练

```bash
/home/ubuntu/IsaacLab/isaaclab.sh -p scripts/train.py \
  --task Isaac-Velocity-Flat-RSX-v0 --num_envs 4096 --headless
```

录像加 `--video`。不要 GUI 训练（窗口会假死）。

## 回放

自动前进（Play 任务固定 `vx=0.3`）：

```bash
/home/ubuntu/IsaacLab/isaaclab.sh -p scripts/play.py \
  --task Isaac-Velocity-Flat-RSX-Play-v0 --num_envs 1 \
  --checkpoint logs/rsl_rl/rsx_flat/<run>/model_*.pt --real-time
```

WASD（先点视口；W 前进，S 停，默认无横移/转向）：

```bash
/home/ubuntu/IsaacLab/isaaclab.sh -p scripts/play_wasd.py \
  --task Isaac-Velocity-Flat-RSX-Play-v0 --num_envs 1 \
  --checkpoint logs/rsl_rl/rsx_flat/<run>/model_*.pt --real-time
```
