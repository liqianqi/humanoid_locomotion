# 诊断探针方法：怎么"量"出问题的根因

方法论（`tuning_methodology.md`）讲的是"看奖励曲线判断 A/B"。但很多现象光看 TensorBoard 分不清根因——比如"方向漂移"到底是**左右不对称的定向偏置**，还是**随机控制太弱**。这时要写一个**无头探针脚本**，用固定指令跑几百步，把关键物理量直接量出来。本文记录这套做法。配合 `diagnosis_2026-09-18.md` 阅读。

---

## 一、核心思想：让现象变成一个可判定的数

一个模糊的抱怨（"走得不稳/方向乱"）要先翻译成**一个能算出数、且不同根因会给出不同数值区间**的指标。

例：偏航漂移的根因判定用 **`bias/noise` 比值**：
- `bias` = 无指令偏航率的**带符号均值**（总往一边拐才会大）
- `noise` = 无指令偏航率的**绝对值均值**（每步乱扭就会大）
- `bias/noise > ~0.5` → **系统性偏置**（左右不对称，去查机体/URDF/默认姿态/步态相位）
- `bias/noise < ~0.3` → **随机控制弱**（去奖励侧加惩罚/加权重）

这一个比值就把"改机体"还是"改奖励"这条最贵的岔路分开了。

---

## 二、探针脚本的骨架（`analysis/yaw_diag.py` 为模板）

1. **无头启动** Isaac Sim：`AppLauncher(args)`，`args.headless=True`。
2. **固定指令**：用 `parse_env_cfg` 拿 PLAY 配置，把 `rel_standing_envs=0`、`lin_vel_x=(v,v)`、`lin_vel_y=0`、`ang_vel_z=0`。指令恒定，任何偏航都是"无指令漂移"，好归因。
3. **多环境求统计**：`num_envs=64`，一次跑出分布（左转/右转各多少），比单环境可靠。
4. **加载权重**：`OnPolicyRunner` + `runner.load(ckpt)`，注意先过 `handle_deprecated_rsl_rl_cfg`（新版 rsl_rl 的 agent cfg 结构变了，否则报 `KeyError: 'class_name'`）。
5. **热身再测**：先跑 `WARMUP≈120` 步让姿态稳定，再记录 `MEASURE≈400` 步。
6. **直接读物理量**（不是读奖励）：
   - `robot.data.root_ang_vel_w[:,2]` = 世界系偏航率
   - `robot.data.root_lin_vel_b[:,0]` = 机体前进速度（确认真在走）
   - `contact.data.current_contact_time[:, foot_ids]` = 左右脚接触时间
   - `robot.data.joint_pos[:, joint_idx]` = 关节角（查左右对称）
7. **处理重置**：环境会 done 重置；只统计存活环境（`live = ~dones`），done 的累积清零。

---

## 三、两个必踩的坑（已在脚本里规避）

1. **输出缓冲**：重定向到文件时 stdout 是块缓冲，中途看不到进度、结果也可能卡在缓冲里。
   → 用 `python -u`（无缓冲），关键 `print(..., flush=True)`。
2. **`simulation_app.close()` 在本机会卡几分钟**（GPU 占用不降、日志停更，像死机）。
   → 算完统计后 `sys.stdout.flush()` 然后 `os._exit(0)` 硬退出，不调用 close()。

（另外：跑脚本前确认 GPU 空闲、无训练在跑，避免抢显存。）

---

## 四、判定指标速查（可复用到别的症状）

| 症状 | 探针测什么 | 判据 |
|---|---|---|
| 方向漂移 | 无指令偏航率的 signed-mean vs abs-mean | `bias/noise` >0.5 偏置 / <0.3 随机 |
| 左右不对称 | 左右脚接触时间、左右同名关节均值之差 | diff 明显≠0 → 步态/机体不对称 |
| 是否真在走 | 机体前进速度均值 | 接近指令值才算在走（排除拖步/原地） |
| 站不稳/晃 | roll/pitch 角速度均值与 std | 数值大 → 姿态振荡 |
| 出生即倒 | 前几步机身/各 link 接触力、机身高度 | 见 `diagnosis_2026-09-16.md` 的零动作探针 |

> 通用套路：**固定输入 → 多环境跑 → 直接量物理量（别只看奖励）→ 设计一个能把不同根因分开的数**。

---

## 五、本项目已积累的探针

| 脚本 | 用途 | 结论所在 |
|---|---|---|
| `analysis/probe_contacts.py` | 零动作站立，量接触力/机身高度，查自碰撞与初始姿态 | `diagnosis_2026-09-16.md` |
| `analysis/yaw_diag.py` | 固定前进，量偏航率与左右对称，分"偏置 vs 随机" | `diagnosis_2026-09-18.md` §偏航诊断 |

运行示例（务必在 `scripts/` 目录下，让 `humanoid_locomotion` 能导入）：
```bash
cd /home/ubuntu/humanoid_locomotion/scripts
/home/ubuntu/isaacsim_venv/bin/python -u ../analysis/yaw_diag.py > /tmp/yaw.log 2>&1
```
默认自动选最新的 `rsx_flat` run 的最新 checkpoint；要固定某次训练就改脚本里的 `RUN_DIR`。
