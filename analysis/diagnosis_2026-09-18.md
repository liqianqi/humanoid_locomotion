# RSX 小碎步（拖步）诊断与修改（2026-09-18）

分析对象：`logs/rsl_rl/rsx_flat/` 下三次 run 的 TensorBoard 明细、当前 `velocity_env_cfg.py` / `mdp.py` / `rsx.py`，以及本机 Isaac Lab 的 `feet_air_time_positive_biped` 实现。

**结论**：机器人走"小碎步"不是环境或 URDF 的 bug，而是**奖励设计把拖步（shuffle）变成了局部最优解**。日志明确支持这一判断。

---

## 一、决定性证据

三次 run 每回合的奖励明细（已含权重的累计值）：

| 项目 | 09-17_18-57（旧） | 09-18_11-28 | 09-18_13-09（最新） |
|---|---|---|---|
| track_lin_vel_xy | **0.93** | 0.35 | **0.90** |
| feet_air_time | 0.007 | 0.0006 | 0.031 |
| landing_overstep | （无此项） | **22.76** | **0.0** |
| swing_foot_clearance | （无） | 0.002 | 0.031 |
| action_rate_l2 | -0.01 | **-3.27** | -0.01 |
| mean_episode_length | 995 | 997 | 988 |

读出的关键事实：

1. **只有速度跟踪一直拿高分（~0.9），步态类奖励几乎为 0。** 机器人用拖步就能满足"跟上指令速度"这个目标，照样接近满分；不摔倒、速度也对——对策略而言这就是最优解。这就是小碎步的本质。

2. **`landing_overstep`（交叉迈步跨过奖励，weight=30）没起作用：**
   - 最新 run（13-09）里**恒为 0.0** → 门槛条件太严，一次都没触发，完全没有梯度，策略"看不到"它。
   - 11-28 run 里反而**高达 22.76、主导全部奖励（384）** → 但代价是速度跟踪暴跌到 0.35、action_rate 到 -3.27（动作剧烈抖动）。这不是走路，而是为了刷奖励往前猛扑（farming）。

3. **11-28 与 13-09 的唯一配置差别**：`landing_overstep` 加了 `min_air_time:0.18` / `min_travel:0.1`。门槛放松就暴发、收紧就恒为 0——**这是一个非 0 即 1 的稀疏奖励，对 PPO 是一道爬不上去的悬崖。**

---

## 二、为什么拖步会成为最优解（原因）

1. **指令速度太慢**（训练 0.15–0.35 m/s，PLAY 固定 0.25）。这个速度不需要真正的大步幅，小碎步就能达到，速度跟踪奖励因此能被拖步"刷满"。这是根本原因。

2. **`feet_air_time` 的 threshold=0.4 s 太高，且拖步几乎进不了单脚支撑。** 该奖励只在**单脚支撑**（`single_stance`）期间支付 mode-time（clamp 到 threshold）。拖步大部分时间是双脚支撑 → `single_stance=False` → 奖励≈0。所以"从小步幅平滑增大到大步幅"的梯度基本不存在。

3. **`landing_overstep` 是硬门槛稀疏奖励**：只有"不触发 or 暴发"两种状态，形不成平滑学习信号，weight=30 也过大。

4. **终止惩罚 -200 + 容易拿的速度奖励 → 策略规避风险。** 真正的大步有单脚支撑瞬间的摔倒风险；接近双脚支撑的拖步更安全，于是收敛到"安全又能拿分"的一侧。

5. **没有周期性步态相位（gait/phase clock）奖励。** 没有任何东西强制左右单脚支撑交替，"交叉迈步"这个结构没有被约束出来。action scale=0.4 加 action_rate/dof_acc 惩罚，进一步把大幅度髋/膝摆动变成高成本，压向小动作。

---

## 三、已实施的修改

### 1. `scripts/humanoid_locomotion/mdp.py`（新增相位步态机制）

| 新增 | 说明 |
|---|---|
| `_leg_phase()` | 无状态相位时钟：`phase = (episode_length_buf * step_dt) % period / period`，右腿反相 0.5。reset 时自动归零；观测与奖励共用同一公式，保证策略能"看到"它要对齐的时钟。 |
| `gait_phase()`（观测） | 输出左腿相位的 `(sin, cos)`，让策略给自己的步态计时。 |
| `feet_gait_contact()`（奖励） | 按"交替单脚支撑"时间表打分：每只脚实际接触状态与调度的支撑/摆动一致就 +1（每步最高 +2）。提供密集梯度，补上稀疏 `landing_overstep` 缺的那段。指令≈0 时关闭（不强迫站立踏步）。脚序为 `[左, 右]`（sensor 用 `preserve_order=True`）。 |

相位参数：`period=0.7 s`，`offset=0.5`，`stance_ratio=0.55`（略 >0.5，给一小段双支撑重叠）。

### 2. `scripts/humanoid_locomotion/velocity_env_cfg.py`

| 改动 | 原值 → 新值 | 原因 |
|---|---|---|
| 训练 `lin_vel_x` | (0.15, 0.35) → **(0.4, 0.9)** | 低速拖步就能满分；≥0.4 m/s 才"必须"迈真正的步 |
| 新增 `gait_contact` 奖励 | — → weight **1.0**（flat **1.5**） | 密集步态时钟信号，作为主驱动 |
| 新增 `gait_phase` 观测 | — | 与 `gait_contact` 配对，供策略计时 |
| `feet_air_time.threshold` | 0.40 → **0.30** | 更早给"抬脚时长"梯度 |
| `landing_overstep.weight` | 30 → **6** | 从"独霸/暴发"降为锦上添花的加分项 |
| Flat/Rough PLAY `lin_vel_x` | 0.25 → **0.6** | 落在新训练区间内 |

---

## 四、核心逻辑

> 之前"跟上速度"用拖步就能满足，而引导步态的奖励要么零梯度（`landing_overstep=0`）要么暴发（=22.76 却毁掉跟踪）。
> 现在 **提高速度让拖步够不到指令** + **相位时钟给出从拖步平滑走向交叉迈步的密集梯度**，两者一起把局部最优从"小碎步"移开。

---

## 五、验证情况

- ✅ `mdp.py` / `velocity_env_cfg.py` `py_compile` 通过。
- ✅ `env.episode_length_buf`（每步 +1、reset 归零）、`env.step_dt`（=0.005×4=0.02 s）确认存在，clock 在 obs/reward 间一致（每步相位 +0.0286，约 35 步一个步态周期）。
- ✅ 观测项以实例属性注入，`ObservationManager` 会收集（与现有 `height_scan=None` 同一机制）。
- ⚠️ 完整实例化配置需启动 Isaac Sim（`omni` 依赖），留到实跑时验证。

## 六、待办

1. 先跑 **flat**（`Isaac-Velocity-Flat-RSX-v0`，1500 iters）；建议先 ~30 iters 冒烟，确认 `gait_contact` 正常上涨、无报错，再放全量。
2. 训练后核对：`gait_contact` 是否稳定上升、`track_lin_vel_xy` 是否仍高、`feet_air_time`/`swing_foot_clearance` 是否明显 > 0、`action_rate` 是否未暴走。
3. 若步态成形，再逐步加入转向、侧移、随机扰动，并按需微调 `period` 与各权重。
4. 旧 checkpoint（含 13-09 之前）在新配置（多了 `gait_phase` 观测、改了速度区间）下不可复用，需从头训练。

---

## 七、启动训练时的报错与修复（`double_stance` / `landing_overstep` 签名）

### 报错

```
ValueError: The term 'double_stance' expects mandatory parameters: ['kwargs']
and optional parameters: [], but received: ['command_name', 'sensor_cfg', 'hold_time'].
```

### 原因

这是一个**与上面奖励改动无关的既有潜在 bug**。

Isaac Lab 在注册 `ManagerTermBase` 时会**静态检查** `__call__` 的签名（`manager_base.py:363-375`）。而 `double_stance` 和 `landing_overstep` 的 `__call__` 原来写成：

```python
def __call__(self, *args, **kwargs):   # ← 问题所在
```

这种写法下，检查器把 `**kwargs`（VAR_KEYWORD）当成一个**名为 `kwargs` 的必须参数**，它和 `params` 里的键（`command_name` / `sensor_cfg` / `hold_time`）对不上 → **必然报错**。也就是说这一版**根本无法加载**；过去 11-28/13-09 的 run 用的是另一版（显式参数版）。这次跑全新训练才踩到这颗雷。

### 修复（`mdp.py`）

把两个类的 `__call__` 改回 Isaac Lab 规范的**显式命名参数**（逻辑完全不变，只是把 `kwargs.get(...)` 挪到参数默认值）：

```python
# double_stance
def __call__(self, env, sensor_cfg, command_name="base_velocity",
             hold_time=0.20, cmd_threshold=0.1, debounce=0.05): ...

# landing_overstep
def __call__(self, env, sensor_cfg, asset_cfg, command_name="base_velocity",
             target_overstep=0.14, min_air_time=0.18, min_travel=0.10, cmd_threshold=0.1): ...
```

并删除因此不再使用的 `from typing import Any`。

### 验证

- ✅ `py_compile` 通过（两个文件）。
- ✅ 按 Isaac Lab 校验式 `set(args[min_argc:]) == set(term_params + args_with_defaults)` 手工核对全部自定义项均通过：
  `double_stance` ✓ / `landing_overstep` ✓ / `feet_gait_contact`（新）✓ / `swing_foot_clearance` ✓ / `gait_phase`（新观测）✓。

此报错已解决，可重新启动训练。
