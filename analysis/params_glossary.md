# RSX 参数词典

逐项解释 `velocity_env_cfg.py` / `rsx.py` / `agents/rsl_rl_ppo_cfg.py` 里的参数：**含义 + 调大调小的效果 + 当前值**。配合 `tuning_methodology.md`（怎么调）和 `diagnosis_2026-09-18.md`（实战案例）阅读。

> 通用约定：奖励项 = `func`（算什么）× `weight`（多重要，**正=鼓励，负=惩罚**）。

---

## A. 奖励项（`RsxRewards`）

### A1. 任务项（真正要它做的事）

| 项 | 含义 | 关键参数 |
|---|---|---|
| `track_lin_vel_xy_exp` (w=1.0) | 前进/侧移速度跟上指令。指数核 `exp(-误差²/std²)`：误差 0 得 1 分，误差 = std 时得 0.37 分 | **`std=0.25`**：核的"宽容度"。越大越宽松（差一点也给高分，易偏向不动），越小越严格 |
| `track_ang_vel_z_exp` (w=2.0) | 转向角速度跟上指令，同样指数核 | `std=0.25`。权重 2.0 比前进重，因为转向更难学 |

> `std` 最易误解：它是**指数/高斯核的"标准差"**，控制"差多少还算及格"，是奖励形状的旋钮，不是物理量。

### A2. 步态塑形项（逼出"正常走路"的样子）

| 项 | 含义 | 关键参数 |
|---|---|---|
| `feet_air_time` (w=1.0) | 单脚支撑期间奖励脚的腾空/触地时长，鼓励迈实步 | `threshold=0.30`：奖励**封顶值**≈期望单步时长 |
| `gait_contact` (w=1.0 / flat 1.5) | 密集步态时钟：脚的着地状态与"应着地/应摆动"的时间表对上就给分 | `period=0.7` / `offset=0.5` / `stance_ratio=0.55`（详见方法论） |
| `double_stance` (w=**-1.0**) | **惩罚**长时间双脚同时着地（拖步） | `hold_time=0.20`：双脚同时着地超过 0.2 s 才罚（短暂双支撑允许） |
| `landing_overstep` (w=6.0) | 奖励"后脚摆到前脚前方落地"的跨步 | `target_overstep=0.14`（跨过 14cm 满分）、`min_air_time=0.18`、`min_travel=0.10`（防作弊门槛） |
| `swing_foot_clearance` (w=1.5) | 摆动脚要抬到一定高度（别蹭地） | `target_height=0.04`：目标抬脚 4cm |

### A3. 平滑与安全惩罚（均为负权重）

| 项 | 含义 | 影响 |
|---|---|---|
| `termination_penalty` (w=-200) | 摔倒/非法接触时的大罚分 | 越负越"怕死"，太大导致保守不敢迈步 |
| `feet_slide` (w=-0.2) | 罚脚着地时打滑 | 鼓励踩稳 |
| `dof_pos_limits` (w=-1.0) | 罚关节顶到限位（这里只管踝） | 保护关节 |
| `joint_deviation_hip` (w=-0.3) | 罚髋 roll/yaw 偏离默认姿态太多 | 防劈叉/内外八；太负会限制重心侧移 |
| `flat_orientation_l2` (w=-1.0) | 罚身体倾斜（保持躯干竖直） | — |
| `action_rate_l2` (w=-0.015) | 罚**相邻两步动作变化大**（抖动） | 越负越平滑，太负不敢快速摆腿 |
| `dof_acc_l2` (w=-1.25e-6) | 罚关节**加速度**（更高阶平滑） | 抑制抽搐 |
| `dof_torques_l2` (w=-1.5e-7) | 罚**力矩**（省电/护电机） | 太负会软绵无力 |
| `lin_vel_z_l2` | 罚上下颠簸（竖直速度） | flat 里 -0.2 |

> 这些 `*_l2` 是 **L2 惩罚**（值的平方和）。系数极小（1e-6 级）是因为力矩/加速度数值本身很大，要乘小系数才不至于压过主任务。

---

## B. 控制与环境参数（`RsxRoughEnvCfg.__post_init__`）

| 参数 | 含义 | 当前值/说明 |
|---|---|---|
| `actions.joint_pos.scale=0.4` | 策略输出（动作）乘 0.4 再加到默认关节角上作为 PD 目标，相当于"动作幅度上限" | 越大迈得越猛也越易失稳；0.4 偏保守 |
| `commands...ranges.lin_vel_x=(0.4,0.9)` | 训练时随机采样的前进速度指令区间 | 见诊断报告 |
| `lin_vel_y=(-0.15,0.15)` / `ang_vel_z=(-0.4,0.4)` | 侧移 / 转向指令区间 | — |
| `rel_standing_envs=0.2` | 20% 并行环境指令为"站着不动"，学站立 | 太高会让多数环境学不到步态 |
| `heading_command=False` | 关掉"朝向控制"，直接给角速度指令 | 阶段一只练前进 |
| `events.reset_base.pose_range` | 每次重置随机化出生位置/朝向（x/y±0.5m, yaw±π） | 域随机化，增强鲁棒性 |
| `events.push_robot / add_base_mass / base_com = None` | 关掉推力扰动/随机配重/质心随机 | 阶段一先简化，稳定后再加回 |
| `terminations.base_contact` | 身体（base_link）接触地面判摔倒→重置 | `threshold=1.0` N（自碰撞已关，任何机身触地都是真摔） |

> 隐藏但关键：**`decimation=4` × 物理步 `dt=0.005s` → 控制步 `step_dt=0.02s`**（策略每 0.02 秒决策一次）。这是 `gait_contact` 里 period 换算成"多少步一个周期"的依据（0.7 / 0.02 ≈ 35 步/周期）。

---

## C. 机器人与执行器（`rsx.py`）

### C1. 初始姿态
| 参数 | 含义 |
|---|---|
| `init_state.pos=(0,0,0.371)` | 出生躯干高度 0.371 m（留 4mm 脚底间隙，见 09-16 报告） |
| `joint_pos` 髋0.13/膝0.40/踝-0.27 | 默认**预蹲姿态**，满足 `髋-膝-踝=0`（躯干水平），质心落在脚掌内 |

### C2. 执行器 PD（电机模型）
`DCMotorCfg` 是**显式 PD 控制**：`τ = Kp·(目标角-实际角) - Kd·角速度`，再受限幅。

| 参数 | 含义 | 直觉 |
|---|---|---|
| `stiffness` (Kp) | **位置刚度**：偏离目标角多少就出多大力 | 越大越"硬"、跟踪快但易抖/饱和。RS06=60, RS02=50, 踝=40 |
| `damping` (Kd) | **速度阻尼**：抑制振荡 | 越大越"黏"、稳但迟钝。3 / 2.5 / 2.0 |
| `effort_limit` | **实际力矩硬上限**（真正卡你的值） | 11 / 7 / 5 N·m |
| `saturation_effort` | DC 电机随转速衰减用的参考力矩 | 36 / 17 / 14；**实际上限由 effort_limit 决定**，不是这个 |
| `velocity_limit` | 电机最高转速（超过后可用力矩衰减） | 由 RobStride 空载转速换算 |
| `armature` | **反射转动惯量**（转子经减速比折算到关节的惯量） | =0.01。**很关键**：连杆太轻时没有它离散 PD 会发散抖振（见 09-16） |

### C3. 物理/求解器
| 参数 | 含义 |
|---|---|
| `enabled_self_collisions=False` | 关掉自碰撞（CAD 凸包内部穿插会产生假接触，见 09-16） |
| `solver_position_iteration_count=4` | 求解器每步迭代次数，越多越准越慢 |
| `soft_joint_pos_limit_factor=0.9` | 软限位在硬限位 90% 处开始，留缓冲 |

---

## D. PPO 超参（`agents/rsl_rl_ppo_cfg.py`）

| 参数 | 含义 | 直觉 |
|---|---|---|
| `gamma=0.99` | **折扣因子**：未来奖励打折率 | 越接近 1 越看长远（0.99 ≈ 看未来 ~100 步） |
| `lam=0.95` | GAE 的 λ，优势估计偏差/方差权衡 | 常用 0.9~0.97 |
| `clip_param=0.2` | PPO 策略更新的**裁剪范围**，防单步更新过猛 | 标准值 0.2 |
| `entropy_coef=0.008` | **探索系数**：鼓励动作随机性 | 越大越爱探索（不易过早收敛到拖步这种局部最优）；太大学不专 |
| `learning_rate=1e-3` + `schedule=adaptive` | 学习率，adaptive 按 `desired_kl` 自动增减 | — |
| `desired_kl=0.01` | 目标 KL 散度，自适应学习率的调节目标 | 更新步子的"体感油门" |
| `num_steps_per_env=24` | 每次更新前每个环境采集 24 步 | 越大数据越多越稳、越慢 |
| `num_learning_epochs=5` / `num_mini_batches=4` | 每批数据重复训练几遍、分几个 mini-batch | — |
| `init_noise_std=1.0` | 策略初始动作噪声，初期探索强度 | — |
| `actor/critic_hidden_dims=[256,128,128]` | 神经网络每层宽度 | 网络容量 |
| `max_iterations=1500` | 训练总迭代数 | — |

> 与"拖步局部最优"最相关的 PPO 旋钮是 **`entropy_coef`**（探索）——探索不足时策略容易早早钻进拖步出不来。但当前主要瓶颈是奖励设计，不是它。

---

## 附：观测项（`observations.policy`）

策略每步"看到"的输入。基础项来自 Isaac Lab（`base_lin_vel`、`base_ang_vel`、`projected_gravity`、`velocity_commands`、`joint_pos`、`joint_vel`、`actions`，flat 无 `height_scan`），本项目新增：

| 项 | 含义 |
|---|---|
| `gait_phase` | 步态相位时钟的 `(sin, cos)`，让策略知道"现在该迈哪只脚"，与 `gait_contact` 奖励配对（二者必须用同一时钟公式，否则策略在追一个看不见的节拍） |
