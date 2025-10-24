# HIMLoco 奖励函数设计详细文档

## 目录
- [概述](#概述)
- [奖励函数结构图](#奖励函数结构图)
- [奖励函数架构](#奖励函数架构)
- [奖励函数完整代码与逐行解释](#奖励函数完整代码与逐行解释)
- [配置参数说明](#配置参数说明)
- [奖励函数计算流程详解](#奖励函数计算流程详解)
- [调试和调优指南](#调试和调优指南)

---

## 概述

HIMLoco 项目使用基于奖励塑形（Reward Shaping）的强化学习方法来训练四足机器人的运动控制策略。奖励函数设计是训练成功的关键，它通过多个子奖励项的组合来引导机器人学习期望的行为。

**核心设计原则：**
- 🎯 多目标优化：结合速度跟踪、稳定性、能效等多个目标
- ⚙️ 可配置性：每个奖励项都有独立的权重系数（scale）
- ⏱️ 时间步归一化：所有奖励按时间步进行缩放（乘以 dt）
- 🔄 模块化设计：每个奖励函数独立实现，易于扩展和修改

**代码位置：**
- 奖励函数实现：`legged_gym/envs/base/legged_robot.py` （第 1111-1223 行）
- 基础配置：`legged_gym/envs/base/legged_robot_config.py` （第 162-180 行）
- 具体机器人配置：`legged_gym/envs/{robot_name}/{robot_name}_config.py`
- 奖励计算逻辑：`legged_gym/envs/base/legged_robot.py` （第 225-243 行）

---

## 奖励函数结构图

### 总体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      总奖励函数 (Total Reward)                    │
│                    R_total = Σ(w_i × r_i(s,a))                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 任务奖励      │    │ 约束惩罚      │    │ 正则化惩罚    │
│ (Task)       │    │ (Constraint) │    │ (Regularize) │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       │                   │                   │
┌──────┴────────┐   ┌──────┴────────┐   ┌──────┴────────┐
│ 速度跟踪       │   │ 稳定性约束     │   │ 动作平滑       │
│ • tracking_   │   │ • orientation │   │ • action_rate │
│   lin_vel     │   │ • lin_vel_z   │   │ • smoothness  │
│ • tracking_   │   │ • ang_vel_xy  │   │ • dof_acc     │
│   ang_vel     │   │ • base_height │   └───────────────┘
└───────────────┘   │               │
                    │ 物理限制       │   ┌───────────────┐
                    │ • dof_pos_    │   │ 能效优化       │
                    │   limits      │   │ • torques     │
                    │ • dof_vel_    │   │ • dof_vel     │
                    │   limits      │   │ • joint_power │
                    │ • torque_     │   └───────────────┘
                    │   limits      │
                    │ • collision   │   ┌───────────────┐
                    │               │   │ 步态质量       │
                    │ 足端约束       │   │ • foot_       │
                    │ • feet_stumble│   │   clearance   │
                    │ • feet_air_   │   │ • feet_air_   │
                    │   time        │   │   time        │
                    │ • feet_contact│   │ • stand_still │
                    │   _forces     │   └───────────────┘
                    └───────────────┘
```

### 奖励函数分类体系

```
奖励函数 (22个)
│
├── 📊 性能目标 (2个) - 权重: 正值，主要驱动力
│   ├── tracking_lin_vel      [+1.0]   线性速度跟踪
│   └── tracking_ang_vel      [+0.5]   角速度跟踪
│
├── 🛡️ 稳定性约束 (4个) - 权重: 负值，保证稳定
│   ├── orientation          [-0.2]   姿态偏差惩罚
│   ├── lin_vel_z            [-2.0]   垂直速度惩罚
│   ├── ang_vel_xy           [-0.05]  俯仰滚转惩罚
│   └── base_height          [-1.0]   身体高度惩罚
│
├── ⚡ 能效优化 (3个) - 权重: 负值，降低能耗
│   ├── torques              [-0.0]   力矩惩罚
│   ├── dof_vel              [-0.0]   关节速度惩罚
│   └── joint_power          [-2e-5]  关节功率惩罚
│
├── 🎨 动作质量 (3个) - 权重: 负值，平滑控制
│   ├── action_rate          [-0.01]  动作变化率惩罚
│   ├── smoothness           [-0.01]  二阶平滑度惩罚
│   └── dof_acc              [-2.5e-7] 关节加速度惩罚
│
├── 🦶 足端控制 (4个) - 权重: 混合，步态优化
│   ├── foot_clearance       [-0.01]  摆动相离地高度
│   ├── feet_air_time        [+0.0]   滞空时间奖励
│   ├── feet_stumble         [-0.0]   绊倒惩罚
│   └── feet_contact_forces  [未配置]  接触力惩罚
│
├── 🔒 物理限制 (3个) - 权重: 负值，硬件保护
│   ├── dof_pos_limits       [0.0]    关节位置限制
│   ├── dof_vel_limits       [0.0]    关节速度限制
│   └── torque_limits        [0.0]    力矩限制
│
├── ⚠️ 碰撞检测 (2个) - 权重: 负值，避免碰撞
│   ├── collision            [-0.0]   身体碰撞检测
│   └── termination          [-0.0]   非正常终止惩罚
│
└── 🎯 特殊行为 (1个) - 权重: 负值，特定场景
    └── stand_still          [-0.0]   零命令时静止

注：[括号内] 为 Aliengo 机器人的默认权重配置
```

### 权重分布可视化

```
权重大小分布（绝对值，对数尺度）:

1.0    ████████████████████ tracking_lin_vel
0.5    ██████████ tracking_ang_vel
2.0    ████████████████████████ lin_vel_z
1.0    ████████████████████ base_height
0.2    ████ orientation
0.05   █ ang_vel_xy
0.01   ▌ action_rate, smoothness, foot_clearance
2e-5   ▏ joint_power
2.5e-7 ▏ dof_acc
0.0    ▏ (禁用的奖励项)

图例: █ = 0.1 权重单位
```

### 奖励计算时间线

```
时间步 t-2      时间步 t-1      时间步 t
    │               │               │
    ├─ action[t-2]  ├─ action[t-1]  ├─ action[t] ◄── 当前动作
    ├─ state[t-2]   ├─ state[t-1]   ├─ state[t]  ◄── 当前状态
    │               │               │
    │               │               └─► 计算奖励
    │               │                   │
    │               └─────────────────► │ action_rate
    │                                   │ = (action[t] - action[t-1])²
    │                                   │
    └───────────────────────────────────► smoothness
                                        │ = (action[t] - 2×action[t-1] + action[t-2])²
                                        │
                                        ▼
                                    Total Reward[t]
```

---

## 奖励函数架构

### 核心数据结构

```python
# 在 LeggedRobot 类中的关键属性
class LeggedRobot(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 奖励相关属性
        self.reward_scales: Dict[str, float]      # 奖励权重字典 {name: scale}
        self.reward_functions: List[callable]      # 奖励函数列表
        self.reward_names: List[str]               # 奖励函数名称列表
        self.episode_sums: Dict[str, Tensor]       # Episode累计奖励
        self.rew_buf: Tensor                       # 当前时间步总奖励 [num_envs]
```

### 奖励计算核心函数

#### 1. compute_reward() - 计算总奖励

**源代码：** `legged_gym/envs/base/legged_robot.py` (第 225-243 行)

```python
def compute_reward(self):
    """ Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
    """
    # 1. 初始化奖励缓冲区为0
    self.rew_buf[:] = 0.
    
    # 2. 遍历所有激活的奖励函数
    for i in range(len(self.reward_functions)):
        name = self.reward_names[i]
        # 3. 调用奖励函数并乘以权重
        rew = self.reward_functions[i]() * self.reward_scales[name]
        # 4. 累加到总奖励
        self.rew_buf += rew
        # 5. 累加到episode统计
        self.episode_sums[name] += rew
    
    # 6. 可选：裁剪负奖励为0
    if self.cfg.rewards.only_positive_rewards:
        self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
    
    # 7. 添加终止奖励（在裁剪之后）
    if "termination" in self.reward_scales:
        rew = self._reward_termination() * self.reward_scales["termination"]
        self.rew_buf += rew
        self.episode_sums["termination"] += rew
```

**逐行解释：**

```python
self.rew_buf[:] = 0.
```
- 将奖励缓冲区重置为0，`rew_buf` 形状为 `[num_envs]`
- 每个环境在每个时间步都会计算独立的奖励

```python
for i in range(len(self.reward_functions)):
    name = self.reward_names[i]
```
- 遍历所有激活的奖励函数（权重非零的函数）
- `reward_names` 存储函数名称，如 "tracking_lin_vel"

```python
rew = self.reward_functions[i]() * self.reward_scales[name]
```
- 调用奖励函数（返回 `[num_envs]` 形状的张量）
- 乘以对应的权重系数（已经在初始化时乘以了 `dt`）
- 例如：`_reward_tracking_lin_vel() * 0.005` (dt=0.005)

```python
self.rew_buf += rew
self.episode_sums[name] += rew
```
- 将加权后的奖励累加到总奖励缓冲区
- 同时累加到episode统计中，用于日志记录

```python
if self.cfg.rewards.only_positive_rewards:
    self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
```
- 如果配置启用，将负总奖励裁剪为0
- 避免早期训练中的过度惩罚导致训练不稳定

```python
if "termination" in self.reward_scales:
    rew = self._reward_termination() * self.reward_scales["termination"]
    self.rew_buf += rew
    self.episode_sums["termination"] += rew
```
- 终止奖励单独处理，在裁剪之后添加
- 确保失败惩罚不会被 `only_positive_rewards` 影响

---

#### 2. _prepare_reward_function() - 准备奖励函数

**源代码：** `legged_gym/envs/base/legged_robot.py` (第 720-743 行)

```python
def _prepare_reward_function(self):
    """ Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
    """
    # 1. 移除权重为0的奖励项，并对非零权重乘以dt
    for key in list(self.reward_scales.keys()):
        scale = self.reward_scales[key]
        if scale==0:
            self.reward_scales.pop(key) 
        else:
            self.reward_scales[key] *= self.dt
    
    # 2. 准备奖励函数列表
    self.reward_functions = []
    self.reward_names = []
    for name, scale in self.reward_scales.items():
        if name=="termination":
            continue
        self.reward_names.append(name)
        name = '_reward_' + name
        self.reward_functions.append(getattr(self, name))

    # 3. 初始化episode累计奖励字典
    self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                         for name in self.reward_scales.keys()}
```

**逐行解释：**

```python
for key in list(self.reward_scales.keys()):
    scale = self.reward_scales[key]
```
- 遍历配置中定义的所有奖励项
- `list()` 创建副本，因为循环中会修改字典

```python
if scale==0:
    self.reward_scales.pop(key)
```
- 移除权重为0的奖励项（禁用的奖励）
- 减少计算开销，提高训练效率

```python
else:
    self.reward_scales[key] *= self.dt
```
- 对非零权重乘以时间步长 `dt`（通常为 0.005 秒）
- **时间归一化**：使奖励大小与控制频率无关
- 例如：权重 1.0 → 0.005，这样每秒的奖励总量保持一致

```python
self.reward_names.append(name)
name = '_reward_' + name
self.reward_functions.append(getattr(self, name))
```
- 通过字符串拼接找到对应的奖励函数
- 例如：`"tracking_lin_vel"` → `_reward_tracking_lin_vel`
- `getattr(self, name)` 获取函数对象并添加到列表
- 这种设计允许通过配置文件灵活启用/禁用奖励项

```python
self.episode_sums = {name: torch.zeros(...) for name in self.reward_scales.keys()}
```
- 为每个奖励项创建累计张量，形状为 `[num_envs]`
- 用于跟踪整个episode中每个奖励项的总和
- 在episode结束时用于日志记录和分析

---

#### 3. _parse_cfg() - 解析配置

**源代码：** `legged_gym/envs/base/legged_robot.py` (第 920-927 行)

```python
def _parse_cfg(self, cfg):
    self.dt = self.cfg.control.decimation * self.sim_params.dt
    self.obs_scales = self.cfg.normalization.obs_scales
    self.reward_scales = class_to_dict(self.cfg.rewards.scales)  # 将配置类转换为字典
    self.command_ranges = class_to_dict(self.cfg.commands.ranges)
    if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
        self.cfg.terrain.curriculum = False
    self.max_episode_length_s = self.cfg.env.episode_length_s
    self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
```

**关键点：**
- `self.reward_scales` 从配置类中提取，转换为字典格式
- `dt` 是控制时间步，等于仿真时间步乘以抽取因子
- 例如：`sim_dt=0.001`, `decimation=5` → `dt=0.005`

---

### 奖励函数调用流程图

```
初始化阶段:
┌─────────────────────────────────────────────────────────────┐
│ __init__()                                                  │
│   │                                                         │
│   ├─► _parse_cfg()                                         │
│   │    └─► self.reward_scales = class_to_dict(cfg.rewards) │
│   │                                                         │
│   ├─► _prepare_reward_function()                           │
│   │    ├─► 移除权重为0的项                                  │
│   │    ├─► 权重 *= dt (时间归一化)                         │
│   │    ├─► 构建 reward_functions 列表                      │
│   │    └─► 初始化 episode_sums                             │
│   │                                                         │
│   └─► 准备完成                                             │
└─────────────────────────────────────────────────────────────┘

运行阶段 (每个时间步):
┌─────────────────────────────────────────────────────────────┐
│ step(actions)                                               │
│   │                                                         │
│   ├─► 应用动作到仿真器                                      │
│   ├─► 更新状态                                             │
│   │                                                         │
│   ├─► compute_reward()                                     │
│   │    │                                                   │
│   │    ├─► rew_buf[:] = 0                                 │
│   │    │                                                   │
│   │    ├─► for each reward_function:                      │
│   │    │    ├─► rew = function() * scale                  │
│   │    │    ├─► rew_buf += rew                            │
│   │    │    └─► episode_sums[name] += rew                 │
│   │    │                                                   │
│   │    ├─► if only_positive_rewards:                      │
│   │    │    └─► clip(rew_buf, min=0)                      │
│   │    │                                                   │
│   │    └─► 添加 termination 奖励                          │
│   │                                                         │
│   ├─► compute_observations()                               │
│   ├─► check_termination()                                  │
│   │                                                         │
│   └─► return obs, rew_buf, done, info                      │
└─────────────────────────────────────────────────────────────┘

Episode结束:
┌─────────────────────────────────────────────────────────────┐
│ reset_idx(env_ids)                                          │
│   │                                                         │
│   ├─► 记录 episode 统计信息                                │
│   │    └─► for key in episode_sums:                        │
│   │         extras["episode"]['rew_' + key] =              │
│   │           mean(episode_sums[key]) / max_episode_length │
│   │                                                         │
│   ├─► 重置环境状态                                         │
│   │                                                         │
│   └─► episode_sums[env_ids] = 0  (重置累计奖励)            │
└─────────────────────────────────────────────────────────────┘
```

---

## 奖励函数完整代码与逐行解释

本章节详细介绍每个奖励函数的完整源代码和逐行解释。所有代码均来自 `legged_gym/envs/base/legged_robot.py`。

---

### 1. tracking_lin_vel - 线性速度跟踪奖励

**代码位置：** 第 1111-1114 行

#### 完整源代码（带详细注释）

```python
def _reward_tracking_lin_vel(self):
    """
    计算线性速度跟踪奖励
    
    目标：鼓励机器人的实际速度接近命令速度（x和y方向）
    方法：使用高斯误差函数，将速度误差映射到[0,1]区间的奖励
    
    Returns:
        torch.Tensor: 形状[num_envs]，每个环境的奖励值，范围(0, 1]
    """
    # Tracking of linear velocity commands (xy axes)
    # 计算线性速度跟踪误差（仅考虑x和y方向，忽略z方向）
    lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    
    # 使用高斯函数将误差转换为奖励：误差为0时奖励最大(1.0)，误差增大时奖励指数衰减
    return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
```

#### 逐行代码详解

**第1-9行：函数定义和文档字符串**
```python
def _reward_tracking_lin_vel(self):
    """
    计算线性速度跟踪奖励
    ...
    """
```
- **函数名称**: `_reward_tracking_lin_vel` (前缀下划线表示内部函数)
- **调用时机**: 每个仿真步骤(step)，在`compute_reward()`中被调用
- **返回值类型**: `torch.Tensor`
- **返回值形状**: `[num_envs]`，例如4096个并行环境则为`[4096]`
- **返回值范围**: (0, 1]，注意是左开右闭区间

**第11-12行：计算速度误差**
```python
lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
```

**详细拆解**：
```python
# 步骤1：提取命令速度的x和y分量
# self.commands 形状: [num_envs, 3]，其中[:, 0]是vx, [:, 1]是vy, [:, 2]是yaw_rate
cmd_vel_xy = self.commands[:, :2]  # 形状: [num_envs, 2]

# 步骤2：提取实际速度的x和y分量
# self.base_lin_vel 是机器人基座在世界坐标系下的线速度
# 通过逆四元数旋转从世界坐标转换到机器人本体坐标
actual_vel_xy = self.base_lin_vel[:, :2]  # 形状: [num_envs, 2]

# 步骤3：计算每个方向的误差
vel_diff = cmd_vel_xy - actual_vel_xy  # 形状: [num_envs, 2]

# 步骤4：计算平方误差（L2范数的平方）
squared_diff = torch.square(vel_diff)  # 形状: [num_envs, 2]

# 步骤5：对x和y方向求和，得到总误差
lin_vel_error = torch.sum(squared_diff, dim=1)  # 形状: [num_envs]
```

**数学公式**：
$$
E_{vel} = (v_x^{cmd} - v_x^{actual})^2 + (v_y^{cmd} - v_y^{actual})^2
$$

**示例计算**：
```python
# 假设某个环境：
# 命令速度: vx=1.0 m/s, vy=0.2 m/s
# 实际速度: vx=0.9 m/s, vy=0.15 m/s

# 误差计算:
error_x = (1.0 - 0.9)^2 = 0.01
error_y = (0.2 - 0.15)^2 = 0.0025
lin_vel_error = 0.01 + 0.0025 = 0.0125
```

**第14行：计算奖励值**
```python
return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
```

**详细拆解**：
```python
# 步骤1：归一化误差
# tracking_sigma默认为0.25，控制高斯函数的宽度（容忍度）
normalized_error = lin_vel_error / self.cfg.rewards.tracking_sigma

# 步骤2：应用高斯函数
# 将误差映射到(0, 1]区间，误差越小奖励越接近1
reward = torch.exp(-normalized_error)
```

**数学公式**：
$$
r = e^{-\frac{E_{vel}}{\sigma_{tracking}}}
$$

其中：
- $E_{vel}$: 速度误差的平方和
- $\sigma_{tracking}$: 跟踪容忍度参数（默认0.25）
- $r$: 最终奖励值

**示例计算**（续上例）：
```python
# 使用上面计算的 lin_vel_error = 0.0125
# tracking_sigma = 0.25

normalized_error = 0.0125 / 0.25 = 0.05
reward = exp(-0.05) ≈ 0.951

# 这意味着速度误差很小时，奖励接近1.0
```

**参数说明**：
- **`tracking_sigma = 0.25`** (默认值)
  - 控制奖励函数的"宽容度"
  - 较小的sigma：对误差更敏感，奖励下降更快
  - 较大的sigma：对误差更宽容，奖励下降较慢
  
**不同sigma值的影响**：
```python
# 假设速度误差为0.1 (m/s)²
# sigma=0.1:  reward = exp(-0.1/0.1) = exp(-1.0) ≈ 0.368  (严格)
# sigma=0.25: reward = exp(-0.1/0.25) = exp(-0.4) ≈ 0.670  (中等)
# sigma=0.5:  reward = exp(-0.1/0.5) = exp(-0.2) ≈ 0.819  (宽松)
```

#### 奖励曲线可视化

```
奖励值
1.0 ┤●
    │  ●●
0.8 ┤     ●●
    │        ●●
0.6 ┤           ●●
    │              ●●
0.4 ┤                 ●●
    │                    ●●
0.2 ┤                       ●●●
    │                           ●●●●
0.0 ┤                                ●●●●●●●●
    └────────────────────────────────────────
    0   0.25  0.5  0.75  1.0  1.25  1.5  误差(m/s)²

sigma=0.25 时的高斯奖励曲线
```

**目的：** 鼓励机器人跟踪命令的线性速度（x, y 方向）

**公式：**
```python
lin_vel_error = sum((commands[:, :2] - base_lin_vel[:, :2])^2)
reward = exp(-lin_vel_error / tracking_sigma)
```

**详细说明：**
- 使用高斯误差函数，当速度误差为 0 时奖励最大（值为 1）
- `tracking_sigma` 控制奖励函数的宽度（默认 0.25）
- 误差越小，奖励越接近 1；误差越大，奖励指数衰减

**默认权重：** `1.0` （正奖励）

**适用场景：** 所有需要速度控制的任务

---

### 2. tracking_ang_vel - 角速度跟踪奖励

**代码位置：** 第 1116-1119 行

#### 完整源代码（带详细注释）

```python
def _reward_tracking_ang_vel(self):
    """
    计算角速度跟踪奖励
    
    目标：鼓励机器人的实际yaw角速度接近命令角速度
    方法：使用高斯误差函数，类似于线性速度跟踪
    
    Returns:
        torch.Tensor: 形状[num_envs]，每个环境的奖励值，范围(0, 1]
    """
    # Tracking of angular velocity commands (yaw) 
    # 计算角速度跟踪误差（仅考虑z轴/yaw方向的旋转）
    ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    
    # 使用高斯函数将误差转换为奖励
    return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
```

#### 逐行代码详解

**第1-9行：函数定义**
```python
def _reward_tracking_ang_vel(self):
```
- **功能**: 评估机器人转向控制的准确性
- **频率**: 每个仿真步骤调用一次
- **重要性**: 权重通常为线性速度的一半（0.5 vs 1.0）
- **原因**: 角速度跟踪相对不如前进速度重要

**第11-12行：计算角速度误差**
```python
ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
```

**详细拆解**：
```python
# 步骤1：提取命令的yaw角速度
# self.commands[:, 2] 是期望的z轴角速度（绕z轴旋转，控制转向）
# 单位：rad/s，正值表示逆时针旋转，负值表示顺时针旋转
cmd_yaw_vel = self.commands[:, 2]  # 形状: [num_envs]

# 步骤2：提取实际的yaw角速度
# self.base_ang_vel[:, 2] 是机器人本体坐标系下的z轴角速度
# 通过四元数逆旋转从世界坐标转换而来
actual_yaw_vel = self.base_ang_vel[:, 2]  # 形状: [num_envs]

# 步骤3：计算误差并平方
ang_vel_error = torch.square(cmd_yaw_vel - actual_yaw_vel)  # 形状: [num_envs]
```

**数学公式**：
$$
E_{ang} = (\omega_z^{cmd} - \omega_z^{actual})^2
$$

其中：
- $\omega_z^{cmd}$: 命令的yaw角速度（rad/s）
- $\omega_z^{actual}$: 实际的yaw角速度（rad/s）
- $E_{ang}$: 角速度误差的平方

**示例计算**：
```python
# 场景1：直线行走（不转向）
# 命令: yaw_rate = 0.0 rad/s
# 实际: yaw_rate = 0.05 rad/s（稍微偏转）
ang_vel_error = (0.0 - 0.05)^2 = 0.0025

# 场景2：原地旋转
# 命令: yaw_rate = 1.0 rad/s（快速转向）
# 实际: yaw_rate = 0.9 rad/s
ang_vel_error = (1.0 - 0.9)^2 = 0.01

# 场景3：精确跟踪
# 命令: yaw_rate = 0.5 rad/s
# 实际: yaw_rate = 0.5 rad/s
ang_vel_error = (0.5 - 0.5)^2 = 0.0
```

**第14行：计算奖励值**
```python
return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
```

**与线性速度的对比**：
```python
# 线性速度：误差是x和y两个分量的和
lin_vel_error = (vx_err)^2 + (vy_err)^2

# 角速度：误差只有z轴一个分量
ang_vel_error = (wz_err)^2

# 两者使用相同的sigma参数和高斯函数形式
reward = exp(-error / tracking_sigma)
```

**为什么只跟踪yaw（z轴）**：
1. **Roll和Pitch稳定性**: 机器人应该保持身体水平，不应该绕x、y轴旋转
2. **运动学约束**: 四足机器人的roll和pitch主要由腿部配置决定，不直接控制
3. **Yaw控制自由度**: 转向是唯一需要主动控制的角速度
4. **其他奖励函数**: `ang_vel_xy`负责惩罚roll和pitch方向的角速度

**奖励曲线特性**：
```python
# tracking_sigma = 0.25 时
# 角速度误差 → 奖励值
# 0.00 rad/s → 1.000 (完美跟踪)
# 0.05 rad/s → 0.951 (很好)
# 0.10 rad/s → 0.819 (良好)
# 0.25 rad/s → 0.368 (一般)
# 0.50 rad/s → 0.135 (较差)
```

**调优建议**：
- **权重 = 0.5**: 标准设置，转向重要性为线速度的一半
- **权重 = 1.0**: 增强转向控制，适用于需要频繁转向的任务
- **权重 = 0.2**: 降低转向重要性，优先考虑直线行走
- **tracking_sigma调整**: 
  - 减小sigma：要求更精确的角速度控制
  - 增大sigma：允许更大的角速度偏差

**默认权重：** `0.5` （正奖励）

**适用场景：** 需要转向和方向控制的任务，如导航、路径跟踪

---

### 3. lin_vel_z - 垂直线性速度惩罚

**代码位置：** 第 1121-1123 行

#### 完整源代码（带详细注释）

```python
def _reward_lin_vel_z(self):
    """
    惩罚垂直方向(z轴)的线速度
    
    目标：鼓励机器人保持平稳的水平运动，避免跳跃或上下振荡
    方法：对z轴速度的平方进行惩罚（负奖励）
    
    Returns:
        torch.Tensor: 形状[num_envs]，负值惩罚，范围(-∞, 0]
    """
    # Penalize z axis base linear velocity
    # 惩罚基座在垂直方向（z轴）的线速度
    # base_lin_vel[:, 2] 是机器人本体坐标系下的垂直速度
    return torch.square(self.base_lin_vel[:, 2])
```

#### 逐行代码详解

**函数特性**：
```python
def _reward_lin_vel_z(self):
```
- **返回值**: 正值（会被负权重变成惩罚）
- **作用时机**: 持续作用于每个时间步
- **物理意义**: 抑制垂直方向的运动

**计算过程**：
```python
return torch.square(self.base_lin_vel[:, 2])
```

**详细拆解**：
```python
# 步骤1：提取z轴速度（垂直方向）
# base_lin_vel 是机器人基座在本体坐标系下的线速度
# [:, 0] = x方向速度（前进/后退）
# [:, 1] = y方向速度（左移/右移）  
# [:, 2] = z方向速度（上升/下降）
z_velocity = self.base_lin_vel[:, 2]  # 形状: [num_envs]

# 步骤2：计算平方
# 使用平方而非绝对值的原因：
# 1. 平方函数可微分，梯度更平滑
# 2. 对大的速度偏差惩罚更重
# 3. 无论上升还是下降都会被惩罚
penalty = torch.square(z_velocity)  # 形状: [num_envs]

# 步骤3：应用负权重（在配置中）
# 最终奖励 = penalty * weight (weight通常为-2.0)
# 最终奖励是负值，构成惩罚
```

**数学公式**：
$$
r = -(v_z)^2
$$

其中：
- $v_z$: 垂直方向速度（m/s）
- $r$: 奖励值（应用权重后为负）

**示例计算**：
```python
# 场景1：平稳行走
# z_velocity = 0.01 m/s（几乎静止）
penalty = (0.01)^2 = 0.0001
final_reward = 0.0001 * (-2.0) = -0.0002  # 几乎无惩罚

# 场景2：轻微跳跃
# z_velocity = 0.1 m/s
penalty = (0.1)^2 = 0.01
final_reward = 0.01 * (-2.0) = -0.02

# 场景3：大幅跳跃
# z_velocity = 0.5 m/s
penalty = (0.5)^2 = 0.25
final_reward = 0.25 * (-2.0) = -0.5  # 严重惩罚

# 场景4：快速下落
# z_velocity = -0.3 m/s（负值表示下降）
penalty = (-0.3)^2 = 0.09  # 平方消除符号
final_reward = 0.09 * (-2.0) = -0.18
```

**为什么惩罚垂直速度**：

1. **物理约束**: 
   - 四足机器人主要用于地面移动，不应有显著垂直运动
   - 频繁的跳跃会导致能量浪费和机械磨损

2. **稳定性考虑**:
   - 垂直振荡会影响传感器读数（如相机）
   - 增加控制难度和不确定性

3. **安全性**:
   - 避免机器人失控跳跃
   - 减少着陆时的冲击力

4. **能效**:
   - 抬起整个机体需要大量能量
   - 平稳移动能效更高

**与其他奖励的配合**：
```python
# base_height: 惩罚偏离期望高度
# lin_vel_z: 惩罚垂直方向的速度
# feet_air_time: 奖励合理的腾空时间（步态）

# 三者配合实现：
# 1. 保持稳定的身体高度（base_height）
# 2. 避免整体上下振荡（lin_vel_z）
# 3. 允许脚的抬起和落地（feet_air_time）
```

**调优指南**：

| 权重值 | 行为特征 | 适用场景 |
|--------|----------|----------|
| -0.5 | 允许较大垂直运动 | 崎岖地形，需要跨越障碍 |
| -2.0 | 标准约束 | 平地行走（默认） |
| -5.0 | 严格限制垂直运动 | 平滑地面，高稳定性要求 |
| -10.0 | 极度抑制跳跃 | 精密操作，载物运输 |

**常见问题**：

**Q1: 为什么不用绝对值而用平方？**
```python
# 方案1：绝对值
penalty = torch.abs(z_velocity)  # 在零点不可微

# 方案2：平方（采用）
penalty = torch.square(z_velocity)  # 处处可微，梯度平滑
```

**Q2: 这会阻止正常的步态吗？**
```
不会。正常的步态涉及腿部抬起，不是整个身体的垂直运动。
- 腿部运动：feet_air_time奖励鼓励
- 身体垂直运动：lin_vel_z惩罚抑制
```

**默认权重：** `-2.0` （负奖励/惩罚）

**适用场景：** 需要平稳水平运动的场景，平地导航，物体运输

---

### 4. ang_vel_xy - 俯仰滚转角速度惩罚

**代码位置：** 第 1125-1127 行

#### 完整源代码（带详细注释）

```python
def _reward_ang_vel_xy(self):
    """
    惩罚roll和pitch方向的角速度
    
    目标：鼓励机器人保持身体姿态稳定，避免绕x轴（roll）和y轴（pitch）的旋转
    方法：对x和y轴角速度的平方和进行惩罚
    
    Returns:
        torch.Tensor: 形状[num_envs]，正值（会被负权重变成惩罚）
    """
    # Penalize xy axes base angular velocity
    # 惩罚基座在roll(x)和pitch(y)方向的角速度
    # 只允许yaw(z)方向的旋转，由tracking_ang_vel控制
    return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
```

#### 逐行代码详解

**坐标系说明**：
```
机器人本体坐标系（Body Frame）：
- X轴：指向前方 → Roll（侧翻）
- Y轴：指向左侧 → Pitch（俯仰）
- Z轴：指向上方 → Yaw（转向）

     Z↑ (Yaw)
      |
      |
      ●----→ X (Roll)
     /
    / Y (Pitch)
```

**计算过程**：
```python
return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
```

**详细拆解**：
```python
# 步骤1：提取roll和pitch角速度
# base_ang_vel[:, 0] = 绕x轴的角速度（roll，机器人侧翻）
# base_ang_vel[:, 1] = 绕y轴的角速度（pitch，机器人前后俯仰）
# base_ang_vel[:, 2] = 绕z轴的角速度（yaw，机器人转向）- 不惩罚
roll_pitch_vel = self.base_ang_vel[:, :2]  # 形状: [num_envs, 2]

# 步骤2：计算平方
# 使用平方惩罚，无论正向还是负向旋转都被惩罚
squared_vel = torch.square(roll_pitch_vel)  # 形状: [num_envs, 2]

# 步骤3：对两个方向求和
# roll和pitch的角速度惩罚累加
penalty = torch.sum(squared_vel, dim=1)  # 形状: [num_envs]

# 应用负权重后：final_reward = penalty * (-0.05)
```

**数学公式**：
$$
r = -(\omega_x^2 + \omega_y^2)
$$

其中：
- $\omega_x$: roll角速度（rad/s）
- $\omega_y$: pitch角速度（rad/s）
- $r$: 奖励值（应用权重-0.05后）

**示例计算**：
```python
# 场景1：完美稳定
# roll_vel = 0.0 rad/s, pitch_vel = 0.0 rad/s
penalty = 0.0^2 + 0.0^2 = 0.0
final_reward = 0.0 * (-0.05) = 0.0  # 无惩罚

# 场景2：轻微晃动
# roll_vel = 0.1 rad/s, pitch_vel = 0.05 rad/s
penalty = 0.1^2 + 0.05^2 = 0.01 + 0.0025 = 0.0125
final_reward = 0.0125 * (-0.05) = -0.000625

# 场景3：剧烈摇晃
# roll_vel = 0.5 rad/s, pitch_vel = 0.3 rad/s
penalty = 0.5^2 + 0.3^2 = 0.25 + 0.09 = 0.34
final_reward = 0.34 * (-0.05) = -0.017

# 场景4：快速pitch（如爬坡）
# roll_vel = 0.0 rad/s, pitch_vel = 0.8 rad/s
penalty = 0.0^2 + 0.8^2 = 0.64
final_reward = 0.64 * (-0.05) = -0.032
```

**物理意义和设计理由**：

**1. 为什么惩罚roll和pitch？**
```python
# Roll（侧翻）问题：
# - 导致机器人失衡
# - 影响足端接触力分布
# - 可能导致翻倒

# Pitch（俯仰）问题：
# - 影响前进方向稳定性
# - 影响传感器视野
# - 增加控制难度
```

**2. 为什么不惩罚yaw？**
```python
# Yaw（转向）是必需的：
# - 机器人需要改变朝向
# - tracking_ang_vel专门处理yaw控制
# - 分离控制：转向由命令决定，姿态由此函数稳定
```

**3. 权重为何较小（-0.05）？**
```python
# 相对较小的权重原因：
# 1. 行走时自然会有轻微的pitch和roll
# 2. 过度惩罚会导致僵硬的步态
# 3. 主要起微调作用，不是核心约束
# 4. 与orientation配合使用（静态姿态约束）
```

**与相关奖励函数的关系**：

```
姿态控制体系：
│
├── orientation: 惩罚姿态角度偏差（静态）
│   └─ 目标：身体保持水平
│
├── ang_vel_xy: 惩罚roll/pitch角速度（动态）
│   └─ 目标：姿态变化平稳
│
└── tracking_ang_vel: 跟踪yaw角速度命令
    └─ 目标：精确转向控制
```

**典型数值范围**：
```python
# 正常行走：
# roll_vel: ±0.1 rad/s
# pitch_vel: ±0.15 rad/s
# penalty: ~0.035
# reward: -0.00175

# 快速奔跑：
# roll_vel: ±0.3 rad/s
# pitch_vel: ±0.4 rad/s
# penalty: ~0.25
# reward: -0.0125

# 不稳定状态：
# roll_vel: ±1.0 rad/s
# pitch_vel: ±0.8 rad/s
# penalty: ~1.64
# reward: -0.082  # 显著惩罚
```

**调优建议**：

| 权重值 | 效果 | 适用场景 |
|--------|------|----------|
| -0.01 | 允许较大姿态变化 | 崎岖地形，动态运动 |
| -0.05 | 标准约束 | 一般平地行走（默认） |
| -0.1 | 较强约束 | 高稳定性要求 |
| -0.5 | 严格限制 | 精密任务，载物运输 |

**常见问题**：

**Q: 这会限制机器人在斜坡上行走吗？**
```
不会。这个奖励惩罚的是角速度（变化率），不是角度本身。
- 稳定爬坡：pitch角度大，但pitch角速度小 → 小惩罚
- 不稳定晃动：pitch角速度大 → 大惩罚
orientation函数负责限制角度偏差
```

**Q: 为什么使用平方和而不是最大值？**
```python
# 方案1：平方和（采用）
penalty = roll_vel^2 + pitch_vel^2
# 优点：同时考虑两个方向，鼓励整体稳定

# 方案2：最大值
penalty = max(roll_vel^2, pitch_vel^2)
# 缺点：忽略另一个方向，可能导致偏置行为
```

**默认权重：** `-0.05` （负奖励/惩罚）

**适用场景：** 所有需要稳定姿态的任务，尤其是负载运输、精密操作

---

### 5. orientation - 身体姿态惩罚

**目的：** 惩罚机器人身体偏离水平姿态

**公式：**
```python
reward = -sum(projected_gravity[:, :2]^2)
```

**详细说明：**
- `projected_gravity` 是重力向量在机器人身体坐标系中的投影
- 当机器人身体水平时，重力仅在 z 方向，x 和 y 分量为 0
- 身体倾斜时，x 和 y 分量增大，惩罚增加

**默认权重：** `-0.2`（Aliengo）/ `-0.0`（基础配置）

**适用场景：** 需要保持身体水平的场景

---

### 6. dof_acc - 关节加速度惩罚

**目的：** 惩罚关节的剧烈加速度变化，鼓励平滑运动

**公式：**
```python
dof_acc = (last_dof_vel - dof_vel) / dt
reward = -sum(dof_acc^2)
```

**详细说明：**
- 通过相邻时间步的速度差计算加速度
- 惩罚关节的急剧加减速
- 有助于减少冲击力和能量消耗

**默认权重：** `-2.5e-7` （极小的负奖励）

**适用场景：** 所有需要平滑运动的任务

---

### 7. joint_power - 关节功率惩罚

**目的：** 惩罚高功率消耗，鼓励能效运动

**公式：**
```python
reward = -sum(|dof_vel| * |torques|)
```

**详细说明：**
- 功率 = 速度 × 力矩
- 鼓励机器人以更节能的方式运动
- 防止不必要的高速关节运动和大力矩输出

**默认权重：** `-2e-5`（Aliengo）

**适用场景：** 需要考虑能效的实际部署场景

---

### 8. base_height - 身体高度惩罚

**目的：** 鼓励机器人保持目标身体高度

**公式：**
```python
reward = -(base_height - base_height_target)^2
```

**详细说明：**
- 惩罚与目标高度的偏差
- 防止机器人蹲得太低或站得太高
- `base_height_target` 通常设置为机器人的自然站立高度

**默认权重：** `-1.0`（Aliengo）/ `-0.0`（基础配置）

**配置参数：**
- `base_height_target`: 0.30 米（Aliengo）

---

### 9. foot_clearance - 足端离地高度奖励

**目的：** 在摆动相时鼓励足端保持一定的离地高度

**公式：**
```python
height_error = (foot_z_position - clearance_height_target)^2
foot_lateral_vel = sqrt(foot_vel_x^2 + foot_vel_y^2)
reward = -sum(height_error * foot_lateral_vel)
```

**详细说明：**
- 仅在足端有横向速度时（摆动相）计算奖励
- 鼓励足端在摆动时保持目标高度
- 防止足端拖地，减少摩擦和能量损失
- 足端位置和速度都转换到机器人身体坐标系

**默认权重：** `-0.01`（Aliengo）

**配置参数：**
- `clearance_height_target`: -0.20 米（Aliengo，负值表示足端在身体下方）

---

### 10. action_rate - 动作变化率惩罚

**目的：** 惩罚相邻时间步之间的动作变化，鼓励平滑控制

**公式：**
```python
reward = -sum((last_actions - actions)^2)
```

**详细说明：**
- 惩罚动作的突变
- 鼓励策略输出连续平滑的控制信号
- 有助于减少机械磨损和提高实际部署的稳定性

**默认权重：** `-0.01` （负奖励/惩罚）

**适用场景：** 所有实际部署场景

---

### 11. smoothness - 二阶平滑度惩罚

**目的：** 惩罚动作的二阶导数，进一步鼓励平滑控制

**公式：**
```python
reward = -sum((actions - last_actions - last_actions + last_last_actions)^2)
      = -sum((actions - 2*last_actions + last_last_actions)^2)
```

**详细说明：**
- 这是动作的二阶差分，类似于加速度
- 比 `action_rate` 更严格的平滑度约束
- 惩罚动作变化率的变化

**默认权重：** `-0.01`（Aliengo）

**适用场景：** 需要极高平滑度的场景

---

### 12. torques - 力矩惩罚

**目的：** 惩罚高力矩输出，鼓励轻柔控制

**公式：**
```python
reward = -sum(torques^2)
```

**详细说明：**
- 直接惩罚关节力矩的大小
- 防止电机过载
- 与 `joint_power` 不同，不考虑速度

**默认权重：** `-0.00001`（基础）/ `-0.0`（Aliengo 禁用）

**适用场景：** 需要保护硬件的场景

---

### 13. dof_vel - 关节速度惩罚

**目的：** 惩罚高关节速度，鼓励慢速平稳运动

**公式：**
```python
reward = -sum(dof_vel^2)
```

**详细说明：**
- 直接惩罚关节速度
- 防止关节高速运动导致的控制不稳定
- 有助于延长机械寿命

**默认权重：** `-0.0` （通常禁用）

**适用场景：** 需要低速运动的特定任务

---

### 14. collision - 碰撞惩罚

**目的：** 惩罚非足端部位与环境的碰撞

**公式：**
```python
reward = -sum(collision_indicator)
collision_indicator = 1 if contact_force_norm > 0.1 else 0
```

**详细说明：**
- 检测特定身体部位（如机身、大腿）的接触力
- 接触力大于阈值（0.1 N）时计为一次碰撞
- 防止机器人身体与地面或障碍物碰撞

**默认权重：** `-1.0`（基础）/ `-0.0`（Aliengo 禁用）

**配置参数：**
- `penalised_contact_indices`: 需要检测碰撞的身体部位索引

---

### 15. termination - 终止惩罚

**目的：** 惩罚非超时的终止（如摔倒）

**公式：**
```python
reward = -(reset_buf AND NOT time_out_buf)
```

**详细说明：**
- 仅在非正常终止时给予惩罚
- 如果是因为超时而终止，不给予惩罚
- 鼓励机器人避免导致 episode 提前结束的失败状态

**默认权重：** `-0.0` （通常禁用或设为 0）

**适用场景：** 需要明确惩罚失败的训练早期

---

### 16. dof_pos_limits - 关节位置限制惩罚

**目的：** 惩罚关节位置接近或超出限制

**公式：**
```python
out_of_limits_lower = -min(dof_pos - dof_pos_min, 0)
out_of_limits_upper = max(dof_pos - dof_pos_max, 0)
reward = -sum(out_of_limits_lower + out_of_limits_upper)
```

**详细说明：**
- 当关节位置接近物理限制时给予惩罚
- 防止机器人进入奇异位形
- 保护硬件不受损坏

**默认权重：** `0.0` （Aliengo 禁用）

**配置参数：**
- `soft_dof_pos_limit`: 0.95（95% 的 URDF 限制）

---

### 17. dof_vel_limits - 关节速度限制惩罚

**目的：** 惩罚关节速度接近或超出限制

**公式：**
```python
over_limit = max(|dof_vel| - dof_vel_limit * soft_limit, 0)
reward = -sum(clip(over_limit, 0, 1))  # 限制最大误差为 1 rad/s
```

**详细说明：**
- 当关节速度超过软限制时给予惩罚
- 误差被裁剪到最大 1 rad/s，避免过大惩罚
- 防止电机过速运转

**默认权重：** `0.0` （Aliengo 禁用）

**配置参数：**
- `soft_dof_vel_limit`: 0.95

---

### 18. torque_limits - 力矩限制惩罚

**目的：** 惩罚力矩接近或超出限制

**公式：**
```python
over_limit = max(|torques| - torque_limit * soft_limit, 0)
reward = -sum(over_limit)
```

**详细说明：**
- 当力矩超过软限制时给予惩罚
- 防止电机过载和损坏

**默认权重：** `0.0` （Aliengo 禁用）

**配置参数：**
- `soft_torque_limit`: 0.95

---

### 19. feet_air_time - 足端滞空时间奖励

**目的：** 奖励较长的步态周期，鼓励自然行走

**公式：**
```python
reward = sum((feet_air_time - 0.5) * first_contact) if command_vel > 0.1 else 0
```

**详细说明：**
- 仅在足端首次接触地面时给予奖励
- 奖励 = (滞空时间 - 0.5 秒) × 接触指示器
- 仅在速度命令大于 0.1 时有效（静止时不奖励）
- 使用接触力滤波器提高可靠性（PhysX 在网格上接触检测不可靠）

**默认权重：** `1.0`（基础）/ `0.0`（Aliengo 禁用）

**适用场景：** 鼓励正常步态的任务

---

### 20. feet_stumble - 足端绊倒惩罚

**目的：** 惩罚足端与垂直表面碰撞（绊倒）

**公式：**
```python
stumble = any(horizontal_force > 5 * vertical_force)
reward = -stumble
```

**详细说明：**
- 检测足端的水平接触力是否远大于垂直接触力
- 水平力大于垂直力 5 倍时认为是绊倒
- 防止机器人踢到障碍物或台阶

**默认权重：** `-0.0` （通常禁用）

**适用场景：** 复杂地形导航

---

### 21. stand_still - 静止惩罚

**目的：** 在零速度命令时惩罚关节偏离默认位置

**公式：**
```python
reward = -sum(|dof_pos - default_dof_pos|) if command_vel < 0.1 else 0
```

**详细说明：**
- 仅在速度命令接近 0 时激活
- 鼓励机器人在静止时保持默认站立姿态
- 防止在原地做无意义的动作

**默认权重：** `-0.0` （通常禁用）

**适用场景：** 需要静态站立的场景

---

### 22. feet_contact_forces - 足端接触力惩罚

**目的：** 惩罚过大的足端接触力

**公式：**
```python
over_force = max(contact_force_norm - max_contact_force, 0)
reward = -sum(over_force)
```

**详细说明：**
- 惩罚超过最大允许接触力的情况
- 防止着地冲击过大
- 鼓励柔和着地

**默认权重：** 在代码中实现但未在配置中显式列出

**配置参数：**
- `max_contact_force`: 100 N

---

## 配置参数说明

### Aliengo 机器人完整配置

**配置文件位置：** `legged_gym/envs/aliengo/aliengo_config.py`

#### 奖励权重配置 (rewards.scales)

```python
class rewards( LeggedRobotCfg.rewards ):
    class scales:
        # === 性能目标 (正奖励) ===
        tracking_lin_vel = 1.0        # 线性速度跟踪 (主要目标)
        tracking_ang_vel = 0.5        # 角速度跟踪
        
        # === 稳定性约束 (负奖励) ===
        lin_vel_z = -2.0              # 垂直速度惩罚
        ang_vel_xy = -0.05            # 俯仰滚转惩罚
        orientation = -0.2            # 姿态偏差惩罚
        base_height = -1.0            # 身体高度惩罚
        
        # === 能效优化 (负奖励) ===
        joint_power = -2e-5           # 关节功率惩罚
        torques = -0.0                # 力矩惩罚 (禁用)
        dof_vel = -0.0                # 关节速度惩罚 (禁用)
        
        # === 动作质量 (负奖励) ===
        action_rate = -0.01           # 动作变化率惩罚
        smoothness = -0.01            # 二阶平滑度惩罚
        dof_acc = -2.5e-7             # 关节加速度惩罚
        
        # === 足端控制 (混合) ===
        foot_clearance = -0.01        # 足端离地高度
        feet_air_time = 0.0           # 滞空时间奖励 (禁用)
        feet_stumble = -0.0           # 绊倒惩罚 (禁用)
        
        # === 物理限制 (负奖励) ===
        dof_pos_limits = 0.0          # 关节位置限制 (禁用)
        dof_vel_limits = 0.0          # 关节速度限制 (禁用)
        torque_limits = 0.0           # 力矩限制 (禁用)
        
        # === 碰撞检测 (负奖励) ===
        collision = -0.0              # 身体碰撞 (禁用)
        termination = -0.0            # 终止惩罚 (禁用)
        
        # === 特殊行为 (负奖励) ===
        stand_still = -0.0            # 静止惩罚 (禁用)
```

#### 奖励相关超参数

```python
class rewards:
    # === 奖励计算参数 ===
    only_positive_rewards = False     # 是否裁剪负奖励
                                      # False: 允许负总奖励
                                      # True: 将负总奖励裁剪为0
    
    tracking_sigma = 0.25             # 速度跟踪奖励的高斯宽度
                                      # 越小: 奖励函数越陡峭，对误差更敏感
                                      # 越大: 奖励函数越平缓，更容忍误差
    
    # === 软限制系数 ===
    soft_dof_pos_limit = 0.95         # 关节位置软限制 (95% 的 URDF 限制)
    soft_dof_vel_limit = 0.95         # 关节速度软限制
    soft_torque_limit = 0.95          # 力矩软限制
    
    # === 目标值 ===
    base_height_target = 0.30         # 目标身体高度 (米)
                                      # Aliengo 的自然站立高度
    
    clearance_height_target = -0.20   # 足端摆动相目标高度 (米)
                                      # 负值: 在身体坐标系下方
                                      # 表示足端应在身体下方 0.20 米
    
    # === 阈值参数 ===
    max_contact_force = 100.0         # 最大允许接触力 (牛顿)
                                      # 超过此值将被惩罚
```

### 权重配置对比表

| 奖励项 | Aliengo | 基础配置 | 差异说明 |
|-------|---------|---------|---------|
| tracking_lin_vel | 1.0 | 1.0 | 相同 - 主要训练目标 |
| tracking_ang_vel | 0.5 | 0.5 | 相同 - 次要目标 |
| lin_vel_z | -2.0 | -2.0 | 相同 - 稳定性约束 |
| ang_vel_xy | -0.05 | -0.05 | 相同 |
| **orientation** | **-0.2** | **0.0** | **Aliengo 启用姿态约束** |
| **base_height** | **-1.0** | **0.0** | **Aliengo 强制身高控制** |
| **joint_power** | **-2e-5** | **未定义** | **Aliengo 关注能效** |
| **foot_clearance** | **-0.01** | **未定义** | **Aliengo 优化步态** |
| **action_rate** | -0.01 | -0.01 | 相同 |
| **smoothness** | **-0.01** | **未定义** | **Aliengo 要求更高平滑度** |
| dof_acc | -2.5e-7 | -2.5e-7 | 相同 |
| torques | 0.0 (禁用) | -0.00001 | Aliengo 依赖 joint_power |
| feet_air_time | 0.0 (禁用) | 1.0 | Aliengo 不强制步态周期 |
| collision | 0.0 (禁用) | -1.0 | Aliengo 地形较简单 |

### 配置策略分析

#### Aliengo 的配置特点

1. **强调稳定性**
   - 启用 `orientation`(-0.2) 和 `base_height`(-1.0)
   - 保持身体水平和固定高度

2. **关注运动质量**
   - 添加 `smoothness`(-0.01) 和 `foot_clearance`(-0.01)
   - 要求更平滑的控制和更好的步态

3. **能效优化**
   - 使用 `joint_power`(-2e-5) 而非 `torques`
   - 考虑速度和力矩的综合功率

4. **简化碰撞检测**
   - 禁用 `collision` 和 `feet_stumble`
   - 可能训练环境地形较简单

5. **自由步态**
   - 禁用 `feet_air_time`
   - 不强制特定的步态周期，让策略自主学习

#### 基础配置的特点

1. **最小化配置**
   - 仅启用核心奖励项
   - 适合快速原型和初步测试

2. **强制步态**
   - 启用 `feet_air_time`(1.0)
   - 鼓励特定的步态周期

3. **碰撞检测**
   - 启用 `collision`(-1.0)
   - 适合复杂地形训练

### 参数调优建议

#### tracking_sigma 的影响

```python
tracking_sigma = 0.25  # 默认值

# 奖励曲线示例 (误差 vs 奖励):
# sigma = 0.10 (严格):  误差 0.1 → 奖励 0.37,  误差 0.2 → 奖励 0.14
# sigma = 0.25 (默认):  误差 0.1 → 奖励 0.67,  误差 0.2 → 奖励 0.45
# sigma = 0.50 (宽松):  误差 0.1 → 奖励 0.82,  误差 0.2 → 奖励 0.67
```

**调优策略：**
- 训练初期：使用较大的 sigma (0.5)，让策略更容易获得奖励
- 训练中期：使用默认 sigma (0.25)
- 训练后期：逐渐减小 sigma (0.1-0.15)，提高精度要求

#### 权重平衡原则

```
总奖励 = Σ(w_i × r_i)

平衡目标:
1. 正奖励总和 ≈ 负奖励总和 (在期望行为下)
2. 主要目标权重 >> 次要目标权重
3. 惩罚项权重避免过大，防止过早终止训练

示例 (期望行为下的奖励):
  tracking_lin_vel:  1.0 × 0.8 = +0.8
  tracking_ang_vel:  0.5 × 0.7 = +0.35
  负奖励总和:                  ≈ -0.5 to -0.8
  总奖励:                      ≈ +0.4 to +0.7 (正值)
```

#### 常见配置模式

**1. 速度优先模式**
```python
tracking_lin_vel = 2.0   # 增大
tracking_ang_vel = 1.0   # 增大
其他惩罚项 = 较小值        # 减小惩罚
```
- 适用场景：快速移动任务
- 缺点：可能牺牲稳定性

**2. 稳定性优先模式**
```python
tracking_lin_vel = 0.5   # 减小
orientation = -0.5       # 增大惩罚
base_height = -2.0       # 增大惩罚
ang_vel_xy = -0.1        # 增大惩罚
```
- 适用场景：精确定位、复杂地形
- 缺点：移动速度可能较慢

**3. 能效优先模式**
```python
joint_power = -5e-5      # 增大惩罚
torques = -0.0001        # 启用
action_rate = -0.02      # 增大惩罚
```
- 适用场景：长时间运行、电池供电
- 缺点：动态性能可能降低

**4. 平滑控制模式**
```python
action_rate = -0.05      # 大幅增加
smoothness = -0.05       # 大幅增加
dof_acc = -1e-6          # 增大惩罚
```
- 适用场景：实际机器人部署
- 优点：减少机械磨损，提高安全性

---

## 奖励函数计算流程详解

### 1. 初始化阶段

```python
def _prepare_reward_function(self):
    # 1. 移除权重为 0 的奖励项
    for key in list(self.reward_scales.keys()):
        if self.reward_scales[key] == 0:
            self.reward_scales.pop(key)
        else:
            # 2. 将权重乘以时间步长
            self.reward_scales[key] *= self.dt  # dt = 0.005
    
    # 3. 创建奖励函数列表
    self.reward_functions = []
    self.reward_names = []
    for name, scale in self.reward_scales.items():
        if name == "termination":
            continue
        self.reward_names.append(name)
        self.reward_functions.append(getattr(self, '_reward_' + name))
    
    # 4. 初始化 episode 累计奖励
    self.episode_sums = {name: torch.zeros(self.num_envs, ...) 
                         for name in self.reward_scales.keys()}
```

### 2. 每个时间步的奖励计算

```python
def compute_reward(self):
    self.rew_buf[:] = 0.
    
    # 1. 遍历所有激活的奖励函数
    for i in range(len(self.reward_functions)):
        name = self.reward_names[i]
        # 2. 计算原始奖励
        raw_reward = self.reward_functions[i]()
        # 3. 乘以权重系数
        scaled_reward = raw_reward * self.reward_scales[name]
        # 4. 累加到总奖励
        self.rew_buf += scaled_reward
        # 5. 累加到 episode 统计
        self.episode_sums[name] += scaled_reward
    
    # 6. 可选：裁剪负奖励
    if self.cfg.rewards.only_positive_rewards:
        self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
    
    # 7. 添加终止奖励（在裁剪之后）
    if "termination" in self.reward_scales:
        rew = self._reward_termination() * self.reward_scales["termination"]
        self.rew_buf += rew
        self.episode_sums["termination"] += rew
```

### 3. Episode 结束时的统计

每当环境重置时，系统会记录并报告该 episode 的累计奖励：

```python
def reset_idx(self, env_ids):
    # ... 重置逻辑 ...
    
    # 记录 episode 奖励信息
    if self.cfg.env.send_episode_info:
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = \
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length
        
        # 重置累计值
        for key in self.episode_sums.keys():
            self.episode_sums[key][env_ids] = 0.
```

---

## 奖励设计的关键考虑

### 1. 权重平衡

- **主要目标**（权重较大）：
  - `tracking_lin_vel`: 1.0
  - `tracking_ang_vel`: 0.5
  - `base_height`: -1.0

- **次要目标**（权重较小）：
  - `lin_vel_z`: -2.0
  - `orientation`: -0.2
  - `action_rate`: -0.01

- **微调项**（权重极小）：
  - `dof_acc`: -2.5e-7
  - `joint_power`: -2e-5

### 2. 正负奖励比例

- **正奖励**：主要来自速度跟踪
- **负奖励**：来自约束违反和不期望行为
- Aliengo 配置中 `only_positive_rewards = False`，允许负总奖励

### 3. 时间归一化

所有奖励权重都会乘以时间步长 `dt = 0.005`，这样：
- 奖励大小与控制频率无关
- 便于在不同频率下迁移策略

### 4. 奖励稀疏性

- 通过设置权重为 0 来禁用不需要的奖励项
- 减少计算开销
- 简化训练过程

---

---

## 调试和调优指南

### 奖励监控和分析

#### 1. Episode 奖励日志

训练过程中，每个episode结束时会记录各项奖励的平均值：

```python
# 日志输出示例
Episode rewards:
  rew_tracking_lin_vel: 0.8234
  rew_tracking_ang_vel: 0.6421
  rew_lin_vel_z: -0.0123
  rew_ang_vel_xy: -0.0045
  rew_orientation: -0.0234
  rew_base_height: -0.0567
  rew_joint_power: -0.0012
  rew_action_rate: -0.0089
  rew_smoothness: -0.0076
  rew_dof_acc: -0.0003
  rew_foot_clearance: -0.0034
  Total reward: 1.3472
```

#### 2. 奖励分析工具

**查看奖励趋势：**
```python
# 在训练日志中查找
grep "rew_tracking_lin_vel" logs/rough_aliengo/*/summaries.txt

# 使用 TensorBoard 可视化
tensorboard --logdir logs/rough_aliengo/
```

**分析奖励权重是否合理：**
```python
# 计算各项奖励占比
positive_rewards = rew_tracking_lin_vel + rew_tracking_ang_vel
negative_rewards = sum(所有负奖励的绝对值)
ratio = positive_rewards / negative_rewards

# 理想比例: 1.5 - 3.0
# 比例过大: 惩罚过轻，可能导致不良行为
# 比例过小: 惩罚过重，可能导致消极策略
```

### 常见问题诊断

#### 问题 1: 机器人不移动或移动缓慢

**症状：**
- `rew_tracking_lin_vel` 很低 (< 0.3)
- 机器人原地不动或移动很慢

**可能原因：**
1. 速度跟踪奖励权重过小
2. 惩罚项过强，抑制了运动
3. `stand_still` 未正确配置

**诊断步骤：**
```python
# 检查奖励权重
print(f"tracking_lin_vel weight: {cfg.rewards.scales.tracking_lin_vel}")
print(f"Total negative weights: {sum(negative_weights)}")

# 检查命令是否正确
print(f"Command velocity: {env.commands[0, :2]}")  # 应该非零

# 检查实际速度
print(f"Actual velocity: {env.base_lin_vel[0, :2]}")
```

**解决方案：**
```python
# 方案 1: 增大速度跟踪权重
cfg.rewards.scales.tracking_lin_vel = 2.0  # 从 1.0 增加到 2.0

# 方案 2: 减小惩罚项权重
cfg.rewards.scales.base_height = -0.5  # 从 -1.0 减小
cfg.rewards.scales.orientation = -0.1  # 从 -0.2 减小

# 方案 3: 确保 stand_still 正确配置
cfg.rewards.scales.stand_still = -0.0  # 禁用，或仅在零命令时启用
```

---

#### 问题 2: 机器人运动不稳定（抖动、摔倒）

**症状：**
- `rew_orientation` 很负 (< -0.5)
- `rew_ang_vel_xy` 很负
- Episode 频繁终止

**可能原因：**
1. 稳定性惩罚权重过小
2. 速度跟踪权重过大，牺牲稳定性
3. 动作平滑度惩罚不足

**诊断步骤：**
```python
# 检查姿态偏差
print(f"Projected gravity: {env.projected_gravity[0]}")
# 理想: [0, 0, -9.81]，x和y应接近0

# 检查角速度
print(f"Angular velocity: {env.base_ang_vel[0]}")
# roll和pitch ([:2]) 应该很小

# 检查动作变化
print(f"Action change: {torch.norm(env.actions - env.last_actions, dim=1).mean()}")
```

**解决方案：**
```python
# 方案 1: 增强稳定性约束
cfg.rewards.scales.orientation = -0.5  # 从 -0.2 增加到 -0.5
cfg.rewards.scales.ang_vel_xy = -0.1   # 从 -0.05 增加
cfg.rewards.scales.base_height = -2.0  # 增加身高约束

# 方案 2: 增加动作平滑度
cfg.rewards.scales.action_rate = -0.05   # 从 -0.01 增加
cfg.rewards.scales.smoothness = -0.05
cfg.rewards.scales.dof_acc = -1e-6      # 从 -2.5e-7 增加

# 方案 3: 降低速度跟踪要求
cfg.rewards.scales.tracking_lin_vel = 0.5  # 临时降低
cfg.rewards.tracking_sigma = 0.5           # 放宽容忍度
```

---

#### 问题 3: 动作抖动明显

**症状：**
- 关节运动不平滑
- `rew_action_rate` 很负 (< -0.5)
- `rew_smoothness` 很负

**可能原因：**
1. 动作平滑度惩罚不足
2. 观测噪声过大
3. 网络输出不稳定

**诊断步骤：**
```python
# 检查动作方差
action_std = torch.std(env.actions, dim=0)
print(f"Action std: {action_std.mean()}")

# 检查动作变化率
action_change = torch.abs(env.actions - env.last_actions)
print(f"Mean action change: {action_change.mean()}")
print(f"Max action change: {action_change.max()}")

# 可视化动作序列
import matplotlib.pyplot as plt
plt.plot(action_history[:, 0])  # 绘制第一个关节的动作
plt.title("Joint 0 Action Over Time")
plt.show()
```

**解决方案：**
```python
# 方案 1: 大幅增加平滑度惩罚
cfg.rewards.scales.action_rate = -0.1    # 大幅增加
cfg.rewards.scales.smoothness = -0.1
cfg.rewards.scales.dof_acc = -5e-6

# 方案 2: 减少观测噪声
cfg.noise.add_noise = False  # 训练后期关闭噪声
# 或
cfg.noise.noise_level = 0.5  # 从 1.0 减小

# 方案 3: 调整网络架构
# 在训练配置中增加网络隐藏层或使用 LSTM
cfg.policy.hidden_dims = [512, 256, 128]  # 更深的网络
```

---

#### 问题 4: 能耗过高

**症状：**
- `rew_joint_power` 很负 (< -1.0)
- 实际部署时电池消耗快
- 关节温度过高

**可能原因：**
1. 能效惩罚权重过小
2. 动作幅度过大
3. 没有考虑力矩限制

**诊断步骤：**
```python
# 检查平均功率
power = torch.sum(torch.abs(env.dof_vel) * torch.abs(env.torques), dim=1)
print(f"Average power: {power.mean()} W")

# 检查力矩分布
print(f"Mean torque: {torch.abs(env.torques).mean()}")
print(f"Max torque: {torch.abs(env.torques).max()}")

# 检查关节速度
print(f"Mean joint velocity: {torch.abs(env.dof_vel).mean()} rad/s")
```

**解决方案：**
```python
# 方案 1: 增强能效惩罚
cfg.rewards.scales.joint_power = -1e-4   # 从 -2e-5 增加
cfg.rewards.scales.torques = -0.0001     # 启用力矩惩罚
cfg.rewards.scales.dof_vel = -0.001      # 启用速度惩罚

# 方案 2: 限制动作范围
cfg.control.action_scale = 0.25  # 从 0.5 减小

# 方案 3: 启用力矩限制惩罚
cfg.rewards.scales.torque_limits = -0.1
cfg.rewards.soft_torque_limit = 0.8  # 使用 80% 的限制

# 方案 4: 降低速度要求
# 在命令采样中降低速度范围
cfg.commands.ranges.lin_vel_x = [-0.8, 0.8]  # 从 [-1.0, 1.0] 减小
```

---

#### 问题 5: 步态不自然（拖脚、高抬腿）

**症状：**
- 足端接触地面时拖动
- 或足端抬得过高
- `rew_foot_clearance` 异常

**可能原因：**
1. `clearance_height_target` 设置不当
2. `foot_clearance` 权重不合适
3. 缺少足端滞空时间约束

**诊断步骤：**
```python
# 检查足端高度（在身体坐标系）
foot_heights = env.feet_pos[:, :, 2] - env.root_states[:, 2].unsqueeze(1)
print(f"Foot heights: {foot_heights[0]}")

# 检查足端速度
foot_vel = env.feet_vel
print(f"Foot velocities: {torch.norm(foot_vel[0], dim=-1)}")

# 检查滞空时间
print(f"Air time: {env.feet_air_time[0]}")
```

**解决方案：**
```python
# 方案 1: 调整目标高度
# 拖脚问题 - 提高目标
cfg.rewards.clearance_height_target = -0.15  # 从 -0.20 提高

# 高抬腿问题 - 降低目标
cfg.rewards.clearance_height_target = -0.25  # 从 -0.20 降低

# 方案 2: 调整惩罚权重
cfg.rewards.scales.foot_clearance = -0.02  # 增加权重

# 方案 3: 启用滞空时间约束
cfg.rewards.scales.feet_air_time = 0.5  # 鼓励正常步态周期

# 方案 4: 检查地形设置
cfg.terrain.mesh_type = 'plane'  # 先在平地测试
```

---

### 渐进式训练策略

#### 课程学习（Curriculum Learning）

使用动态调整的奖励权重，从简单到复杂：

**阶段 1: 基础运动 (0-2M steps)**
```python
# 专注于速度跟踪和基本稳定性
rewards.scales.tracking_lin_vel = 2.0   # 高权重
rewards.scales.tracking_ang_vel = 1.0
rewards.scales.orientation = -0.1       # 低惩罚
rewards.scales.base_height = -0.5
# 其他惩罚项使用较小权重
```

**阶段 2: 稳定性优化 (2M-5M steps)**
```python
# 逐渐增加稳定性要求
rewards.scales.tracking_lin_vel = 1.5   # 略微降低
rewards.scales.orientation = -0.2       # 增加
rewards.scales.base_height = -1.0       # 增加
rewards.scales.action_rate = -0.01      # 启用平滑度
```

**阶段 3: 运动质量 (5M-10M steps)**
```python
# 优化运动质量和能效
rewards.scales.tracking_lin_vel = 1.0   # 标准权重
rewards.scales.joint_power = -2e-5      # 启用能效
rewards.scales.smoothness = -0.01       # 启用二阶平滑
rewards.scales.foot_clearance = -0.01   # 优化步态
```

**阶段 4: 精细调优 (10M+ steps)**
```python
# 严格约束，接近实际部署要求
rewards.tracking_sigma = 0.15           # 减小容忍度
rewards.scales.action_rate = -0.02      # 增强平滑度
# 根据实际表现微调其他权重
```

**实现课程学习：**
```python
def update_reward_scales(iteration, cfg):
    """根据训练迭代次数动态调整奖励权重"""
    if iteration < 2000:  # 阶段 1
        cfg.rewards.scales.tracking_lin_vel = 2.0
        cfg.rewards.scales.orientation = -0.1
    elif iteration < 5000:  # 阶段 2
        cfg.rewards.scales.tracking_lin_vel = 1.5
        cfg.rewards.scales.orientation = -0.2
    elif iteration < 10000:  # 阶段 3
        cfg.rewards.scales.tracking_lin_vel = 1.0
        cfg.rewards.scales.joint_power = -2e-5
    else:  # 阶段 4
        cfg.rewards.tracking_sigma = 0.15
        cfg.rewards.scales.action_rate = -0.02
```

---

### 奖励权重搜索

#### 网格搜索

对关键参数进行网格搜索：

```python
import itertools

# 定义搜索空间
param_grid = {
    'tracking_lin_vel': [0.5, 1.0, 2.0],
    'orientation': [-0.1, -0.2, -0.5],
    'action_rate': [-0.005, -0.01, -0.02]
}

# 生成所有组合
keys = param_grid.keys()
combinations = list(itertools.product(*param_grid.values()))

# 训练每个配置
for combo in combinations:
    config = dict(zip(keys, combo))
    print(f"Training with: {config}")
    # 运行训练...
    # 记录最终性能...
```

#### 贝叶斯优化

使用贝叶斯优化更高效地搜索：

```python
from ax import optimize

def train_and_evaluate(params):
    """训练并返回性能指标"""
    cfg.rewards.scales.tracking_lin_vel = params['tracking_lin_vel']
    cfg.rewards.scales.orientation = params['orientation']
    # ... 设置其他参数
    
    # 训练
    final_reward = train_policy(cfg)
    return final_reward

# 定义搜索空间
best_parameters, best_values, experiment, model = optimize(
    parameters=[
        {"name": "tracking_lin_vel", "type": "range", "bounds": [0.5, 2.0]},
        {"name": "orientation", "type": "range", "bounds": [-0.5, -0.1]},
        {"name": "action_rate", "type": "range", "bounds": [-0.05, -0.005]},
    ],
    evaluation_function=train_and_evaluate,
    objective_name="reward",
    total_trials=20
)
```

---

### 可视化和调试工具

#### 1. 实时奖励监控

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_rewards_realtime(log_file):
    """实时绘制奖励曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    def update(frame):
        # 读取最新日志
        rewards = parse_log(log_file)
        
        # 绘制主要奖励
        axes[0, 0].clear()
        axes[0, 0].plot(rewards['tracking_lin_vel'])
        axes[0, 0].set_title('Linear Velocity Tracking')
        
        # 绘制惩罚项
        axes[0, 1].clear()
        axes[0, 1].plot(rewards['orientation'])
        axes[0, 1].set_title('Orientation Penalty')
        
        # ... 更多图表
    
    ani = FuncAnimation(fig, update, interval=1000)
    plt.show()
```

#### 2. 奖励热力图

```python
import seaborn as sns

def plot_reward_heatmap(episode_rewards):
    """绘制各项奖励的相关性热力图"""
    reward_df = pd.DataFrame(episode_rewards)
    correlation = reward_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Reward Components Correlation')
    plt.show()
```

#### 3. 3D 可视化

```python
def visualize_foot_clearance(env, robot_id=0):
    """可视化足端轨迹和目标高度"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    foot_trajectory = []
    for _ in range(100):
        env.step(policy(env.obs))
        foot_trajectory.append(env.feet_pos[robot_id].cpu().numpy())
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    foot_trajectory = np.array(foot_trajectory)
    for foot_id in range(4):
        ax.plot(foot_trajectory[:, foot_id, 0],
                foot_trajectory[:, foot_id, 1],
                foot_trajectory[:, foot_id, 2],
                label=f'Foot {foot_id}')
    
    # 绘制目标高度平面
    target_height = env.cfg.rewards.clearance_height_target
    xx, yy = np.meshgrid(range(-1, 2), range(-1, 2))
    zz = np.ones_like(xx) * target_height
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='red')
    
    plt.legend()
    plt.show()
```

---

### 性能基准和目标

#### 训练目标值（Aliengo）

```python
# Episode 平均奖励目标
target_rewards = {
    'tracking_lin_vel': > 0.7,      # 好: >0.8, 优秀: >0.9
    'tracking_ang_vel': > 0.6,      # 好: >0.7, 优秀: >0.8
    'lin_vel_z': > -0.05,           # 好: >-0.02
    'ang_vel_xy': > -0.01,          # 好: >-0.005
    'orientation': > -0.05,         # 好: >-0.02
    'base_height': > -0.1,          # 好: >-0.05
    'joint_power': > -0.01,         # 好: >-0.005
    'action_rate': > -0.05,         # 好: >-0.02
    'smoothness': > -0.05,          # 好: >-0.02
    'dof_acc': < -0.001,            # (非常小的负值)
    'foot_clearance': > -0.02,      # 好: >-0.01
    'total_reward': > 1.0,          # 好: >1.5, 优秀: >2.0
}
```

#### 收敛标准

```python
def check_convergence(rewards_history, window=100):
    """检查训练是否收敛"""
    recent = rewards_history[-window:]
    
    # 1. 总奖励稳定
    reward_std = np.std(recent)
    reward_mean = np.mean(recent)
    cv = reward_std / reward_mean  # 变异系数
    
    converged = cv < 0.1  # 变异系数 < 10%
    
    # 2. 无明显上升趋势
    from scipy.stats import linregress
    slope, _, _, _, _ = linregress(range(len(recent)), recent)
    stagnant = abs(slope) < 0.01
    
    # 3. 达到目标性能
    performance_met = reward_mean > 1.0
    
    return converged and stagnant and performance_met
```

---

### 总结：奖励函数调试清单

✅ **训练前检查：**
- [ ] 确认所有权重配置正确
- [ ] 验证权重符号（正/负）正确
- [ ] 检查 `only_positive_rewards` 设置
- [ ] 确认 `tracking_sigma` 值合理
- [ ] 验证目标值（身高、离地高度）符合机器人规格

✅ **训练中监控：**
- [ ] 实时查看总奖励趋势
- [ ] 监控各项奖励占比
- [ ] 检查正负奖励平衡
- [ ] 观察Episode成功率
- [ ] 记录异常行为和对应奖励

✅ **训练后分析：**
- [ ] 对比目标奖励值
- [ ] 分析奖励相关性
- [ ] 可视化足端轨迹
- [ ] 测试不同速度命令
- [ ] 验证实际部署性能

✅ **问题定位：**
- [ ] 不移动 → 检查速度跟踪权重
- [ ] 不稳定 → 增强稳定性约束
- [ ] 抖动 → 增加平滑度惩罚
- [ ] 耗能高 → 启用能效惩罚
- [ ] 步态差 → 调整足端约束

---

## 附录

---

---

### 12. torques - 力矩惩罚

**代码位置：** 第 1167-1169 行

#### 完整源代码

```python
def _reward_torques(self):
    # Penalize torques
    return torch.sum(torch.square(self.torques), dim=1)
```

#### 逐行解释

```python
return torch.sum(torch.square(self.torques), dim=1)
```
- `self.torques`：当前所有关节的力矩，形状 `[num_envs, num_dof]`
- `torch.square(...)`：对每个关节力矩取平方
- `torch.sum(..., dim=1)`：对所有关节求和
- 数学公式：$\sum_i τ_i^2$
- 直接惩罚力矩的大小，与速度无关（区别于 `joint_power`）
- 防止电机过载，保护硬件

**配置参数：**
- 权重：`-0.00001` (基础) / `-0.0` (Aliengo 禁用)
- 适用场景：需要保护硬件的场景

---

### 13. dof_vel - 关节速度惩罚

**代码位置：** 第 1171-1173 行

#### 完整源代码

```python
def _reward_dof_vel(self):
    # Penalize dof velocities
    return torch.sum(torch.square(self.dof_vel), dim=1)
```

#### 逐行解释

```python
return torch.sum(torch.square(self.dof_vel), dim=1)
```
- `self.dof_vel`：当前所有关节的速度，形状 `[num_envs, num_dof]`
- 计算：$\sum_i ω_i^2$
- 直接惩罚关节速度，防止高速运动
- 有助于延长机械寿命和提高控制稳定性

**配置参数：**
- 权重：`-0.0` (通常禁用)
- 适用场景：需要低速运动的特定任务

---

### 14. collision - 碰撞惩罚

**代码位置：** 第 1175-1177 行

#### 完整源代码

```python
def _reward_collision(self):
    # Penalize collisions on selected bodies
    return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
```

#### 逐行解释

```python
return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
```
- `self.contact_forces`：所有刚体的接触力，形状 `[num_envs, num_bodies, 3]`
- `self.penalised_contact_indices`：需要检测碰撞的身体部位索引（如机身、大腿）
- `[:, self.penalised_contact_indices, :]`：提取特定部位的接触力
- `torch.norm(..., dim=-1)`：计算接触力的模（大小），形状 `[num_envs, num_penalised_bodies]`
- `> 0.1`：布尔掩码，接触力大于 0.1 N 时为 True
- `1.*`：将布尔值转换为浮点数（True→1.0, False→0.0）
- `torch.sum(..., dim=1)`：统计每个环境中有多少个身体部位发生碰撞
- **惩罚特性：** 碰撞的身体部位越多，惩罚越大

**配置参数：**
- 权重：`-1.0` (基础) / `-0.0` (Aliengo 禁用)
- 阈值：0.1 N
- 适用场景：避免机器人身体与地面或障碍物碰撞

**碰撞检测示例：**
```
环境1: 机身接触力 = 5 N, 左大腿接触力 = 0 N
      → 碰撞计数 = 1, 惩罚 = -1.0
环境2: 无非足端接触
      → 碰撞计数 = 0, 惩罚 = 0
```

---

### 15. termination - 终止惩罚

**代码位置：** 第 1179-1181 行

#### 完整源代码

```python
def _reward_termination(self):
    # Terminal reward / penalty
    return self.reset_buf * ~self.time_out_buf
```

#### 逐行解释

```python
return self.reset_buf * ~self.time_out_buf
```
- `self.reset_buf`：布尔张量，标记哪些环境需要重置，形状 `[num_envs]`
  - True (1)：环境失败需要重置
  - False (0)：环境继续运行
- `self.time_out_buf`：布尔张量，标记哪些环境因超时而结束
  - True (1)：因达到最大episode长度而结束（正常）
  - False (0)：未超时
- `~self.time_out_buf`：逻辑非，反转超时标记
  - True (1)：非超时结束
  - False (0)：超时结束
- `reset_buf * ~time_out_buf`：逻辑与操作
  - 结果为 1：环境重置且非超时（即失败，如摔倒）
  - 结果为 0：环境未重置或正常超时

**逻辑表：**
```
reset_buf | time_out_buf | ~time_out_buf | 结果 | 含义
    0     |      0       |       1       |  0   | 继续运行
    0     |      1       |       0       |  0   | 继续运行（不会同时为True）
    1     |      0       |       1       |  1   | 失败终止（惩罚）
    1     |      1       |       0       |  0   | 正常超时（不惩罚）
```

**配置参数：**
- 权重：`-0.0` (通常禁用)
- 适用场景：需要明确惩罚失败的训练早期

---

### 16. dof_pos_limits - 关节位置限制惩罚

**代码位置：** 第 1183-1187 行

#### 完整源代码

```python
def _reward_dof_pos_limits(self):
    # Penalize dof positions too close to the limit
    out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
    out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
    return torch.sum(out_of_limits, dim=1)
```

#### 逐行解释

```python
out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
```
- `self.dof_pos`：当前关节位置，形状 `[num_envs, num_dof]`
- `self.dof_pos_limits[:, 0]`：关节位置下限，形状 `[num_dof]`
- `dof_pos - dof_pos_limits[:, 0]`：计算与下限的差值
  - 正值：在限制范围内
  - 负值：超出下限
- `.clip(max=0.)`：裁剪，保留负值（超限部分），正值变为 0
- `-(...)`: 取负，将负值变为正的惩罚值
- **结果：** 超出下限越多，惩罚越大

```python
out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
```
- `self.dof_pos_limits[:, 1]`：关节位置上限
- `dof_pos - dof_pos_limits[:, 1]`：计算与上限的差值
  - 正值：超出上限
  - 负值：在限制范围内
- `.clip(min=0.)`：裁剪，保留正值（超限部分），负值变为 0
- 累加到 `out_of_limits`

```python
return torch.sum(out_of_limits, dim=1)
```
- 对所有关节求和，得到总惩罚

**位置限制示例：**
```
关节限制: [-1.0, 1.0] rad
当前位置: -1.2 rad → 超出下限 0.2 → 惩罚 = 0.2
当前位置: 1.1 rad  → 超出上限 0.1 → 惩罚 = 0.1
当前位置: 0.5 rad  → 在范围内      → 惩罚 = 0
```

**配置参数：**
- 权重：`0.0` (Aliengo 禁用)
- `soft_dof_pos_limit`: 0.95 (使用 95% 的 URDF 限制)
- 适用场景：防止机器人进入奇异位形

---

### 17. dof_vel_limits - 关节速度限制惩罚

**代码位置：** 第 1189-1192 行

#### 完整源代码

```python
def _reward_dof_vel_limits(self):
    # Penalize dof velocities too close to the limit
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
```

#### 逐行解释

```python
return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
```
- `self.dof_vel`：当前关节速度，形状 `[num_envs, num_dof]`
- `torch.abs(self.dof_vel)`：速度的绝对值（速度可正可负）
- `self.dof_vel_limits`：关节速度限制（正值），形状 `[num_dof]`
- `soft_dof_vel_limit`：软限制系数（默认 0.95）
- `dof_vel_limits * soft_dof_vel_limit`：实际使用的限制（95% 的硬限制）
- `abs(dof_vel) - soft_limit`：计算超出软限制的部分
  - 正值：超限
  - 负值：未超限
- `.clip(min=0., max=1.)`：
  - `min=0.`：移除负值（未超限的情况）
  - `max=1.`：限制最大误差为 1 rad/s，避免过大惩罚
- `torch.sum(..., dim=1)`：对所有关节求和

**速度限制示例：**
```
硬限制: 10 rad/s
软限制: 10 × 0.95 = 9.5 rad/s
当前速度: 8 rad/s   → 未超限        → 惩罚 = 0
当前速度: 9.7 rad/s → 超限 0.2     → 惩罚 = 0.2
当前速度: 12 rad/s  → 超限 2.5     → 惩罚 = 1.0 (裁剪)
```

**配置参数：**
- 权重：`0.0` (Aliengo 禁用)
- `soft_dof_vel_limit`: 0.95
- 最大单关节惩罚：1.0
- 适用场景：防止电机过速运转

---

### 18. torque_limits - 力矩限制惩罚

**代码位置：** 第 1194-1196 行

#### 完整源代码

```python
def _reward_torque_limits(self):
    # penalize torques too close to the limit
    return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
```

#### 逐行解释

```python
return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
```
- `self.torques`：当前关节力矩，形状 `[num_envs, num_dof]`
- `torch.abs(self.torques)`：力矩的绝对值
- `self.torque_limits`：关节力矩限制，形状 `[num_dof]`
- `soft_torque_limit`：软限制系数（默认 0.95）
- 计算超出软限制的部分并裁剪负值
- 与 `dof_vel_limits` 类似，但没有最大误差限制

**配置参数：**
- 权重：`0.0` (Aliengo 禁用)
- `soft_torque_limit`: 0.95
- 适用场景：防止电机过载和损坏

---

### 19. feet_air_time - 足端滞空时间奖励

**代码位置：** 第 1198-1209 行

#### 完整源代码

```python
def _reward_feet_air_time(self):
    # Reward long steps
    # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    contact_filt = torch.logical_or(contact, self.last_contacts) 
    self.last_contacts = contact
    first_contact = (self.feet_air_time > 0.) * contact_filt
    self.feet_air_time += self.dt
    rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    self.feet_air_time *= ~contact_filt
    return rew_airTime
```

#### 逐行解释

```python
contact = self.contact_forces[:, self.feet_indices, 2] > 1.
```
- `self.contact_forces[:, self.feet_indices, 2]`：足端的 z 轴（垂直）接触力
- `> 1.`：判断垂直接触力是否大于 1 N
- 结果：布尔张量，形状 `[num_envs, num_feet]`，True 表示足端接触地面

```python
contact_filt = torch.logical_or(contact, self.last_contacts)
self.last_contacts = contact
```
- 接触力滤波：当前接触或上一帧接触都算作接触
- **原因：** PhysX 在网格地形上的接触检测不可靠，需要时间平滑
- 更新 `last_contacts` 用于下一帧

```python
first_contact = (self.feet_air_time > 0.) * contact_filt
```
- `self.feet_air_time > 0.`：足端之前在空中（滞空时间 > 0）
- `* contact_filt`：且当前接触地面
- **结果：** 标记首次接触地面的时刻

```python
self.feet_air_time += self.dt
```
- 所有足端的滞空时间增加一个时间步（0.005 秒）

```python
rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
```
- `(self.feet_air_time - 0.5)`：滞空时间减去 0.5 秒
  - 滞空 < 0.5 秒：负奖励（步态过快）
  - 滞空 > 0.5 秒：正奖励（步态正常）
- `* first_contact`：仅在首次接触时计算奖励
- `torch.sum(..., dim=1)`：对所有足端求和

```python
rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
```
- `torch.norm(self.commands[:, :2], dim=1)`：速度命令的大小
- `> 0.1`：仅在速度命令 > 0.1 m/s 时给予奖励
- **原因：** 静止时不应该奖励滞空时间

```python
self.feet_air_time *= ~contact_filt
```
- `~contact_filt`：逻辑非，标记未接触的足端
- 将接触地面的足端滞空时间重置为 0
- 未接触的足端保持累计滞空时间

```python
return rew_airTime
```
- 返回总奖励

**步态周期示例：**
```
时间轴:
足端离地 ──┐ 滞空 0.4s ┌── 着地
           └────────────┘
滞空时间累计: 0 → 0.4s
首次接触: 奖励 = (0.4 - 0.5) = -0.1 (步太快)

足端离地 ──┐ 滞空 0.7s ┌── 着地
           └──────────────┘
首次接触: 奖励 = (0.7 - 0.5) = +0.2 (正常步态)
```

**配置参数：**
- 权重：`1.0` (基础) / `0.0` (Aliengo 禁用)
- 目标滞空时间：> 0.5 秒
- 适用场景：鼓励正常步态的任务

---

### 20. feet_stumble - 足端绊倒惩罚

**代码位置：** 第 1211-1214 行

#### 完整源代码

```python
def _reward_stumble(self):
    # Penalize feet hitting vertical surfaces
    return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
         5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
```

#### 逐行解释

```python
return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
```
- `self.contact_forces[:, self.feet_indices, :2]`：足端的水平接触力 (x, y)
- `torch.norm(..., dim=2)`：水平接触力的模，形状 `[num_envs, num_feet]`
- `self.contact_forces[:, self.feet_indices, 2]`：足端的垂直接触力 (z)
- `torch.abs(...)`：垂直接触力的绝对值
- `5 * ...`：垂直力的 5 倍
- 比较：水平力 > 5 × 垂直力？
  - True：可能踢到障碍物（绊倒）
  - False：正常着地
- `torch.any(..., dim=1)`：任意一只脚绊倒都算
  - True (1)：发生绊倒
  - False (0)：无绊倒

**绊倒判断逻辑：**
```
正常着地:
  水平力 = 2 N, 垂直力 = 10 N
  2 < 5×10 → 不算绊倒

踢到障碍物:
  水平力 = 60 N, 垂直力 = 8 N
  60 > 5×8 → 算绊倒，惩罚 = 1
```

**配置参数：**
- 权重：`-0.0` (通常禁用)
- 水平/垂直力比例阈值：5
- 适用场景：复杂地形导航

---

### 21. stand_still - 静止惩罚

**代码位置：** 第 1216-1218 行

#### 完整源代码

```python
def _reward_stand_still(self):
    # Penalize motion at zero commands
    return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
```

#### 逐行解释

```python
return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
```
- `self.dof_pos`：当前关节位置
- `self.default_dof_pos`：默认关节位置（站立姿态）
- `torch.abs(...)`：计算每个关节偏离默认位置的绝对值
- `torch.sum(..., dim=1)`：对所有关节求和，得到总偏差
- `torch.norm(self.commands[:, :2], dim=1)`：速度命令的大小
- `< 0.1`：速度命令接近零（静止命令）
- `*`：仅在静止命令时惩罚关节偏离

**逻辑：**
```
速度命令 > 0.1 m/s: 允许关节运动，不惩罚
速度命令 < 0.1 m/s: 期望静止，惩罚关节偏离默认姿态
```

**配置参数：**
- 权重：`-0.0` (通常禁用)
- 速度阈值：0.1 m/s
- 适用场景：需要静态站立的场景

---

### 22. feet_contact_forces - 足端接触力惩罚

**代码位置：** 第 1220-1223 行

#### 完整源代码

```python
def _reward_feet_contact_forces(self):
    # penalize high contact forces
    return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
```

#### 逐行解释

```python
return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
```
- `self.contact_forces[:, self.feet_indices, :]`：足端三轴接触力，形状 `[num_envs, num_feet, 3]`
- `torch.norm(..., dim=-1)`：计算接触力的模（合力大小），形状 `[num_envs, num_feet]`
- `- max_contact_force`：减去最大允许接触力（100 N）
  - 正值：超出限制
  - 负值：在限制内
- `.clip(min=0.)`：裁剪负值，仅保留超限部分
- `torch.sum(..., dim=1)`：对所有足端求和

**接触力示例：**
```
最大允许力: 100 N
足端1: 接触力 = 80 N  → 未超限 → 惩罚 = 0
足端2: 接触力 = 120 N → 超限 20 N → 惩罚 = 20
足端3: 接触力 = 150 N → 超限 50 N → 惩罚 = 50
总惩罚: 0 + 20 + 50 = 70
```

**配置参数：**
- 权重：未在配置中显式列出
- `max_contact_force`: 100 N
- 适用场景：防止着地冲击过大，鼓励柔和着地

---

## 配置参数说明

---

## 附录

### A. 数学公式汇总

#### 总奖励函数
$$R_{total}(s, a) = \sum_{i=1}^{N} w_i \cdot r_i(s, a) \cdot \Delta t$$

#### 高斯奖励函数
$$r_{gaussian}(e) = \exp\left(-\frac{e}{\sigma}\right)$$

应用于速度跟踪：tracking_lin_vel, tracking_ang_vel

#### 二次惩罚函数
$$r_{quadratic} = -\sum_i (x_i - x_i^{target})^2$$

#### 功率计算
$$P = \sum_{i=1}^{n_{dof}} |\omega_i| \cdot |\tau_i|$$

#### 二阶差分（平滑度）
$$\Delta^2 a_t = a_t - 2a_{t-1} + a_{t-2}$$

---

### B. 快速参考表

| 奖励项 | 默认权重 | 类型 | 主要作用 |
|-------|---------|------|---------|
| tracking_lin_vel | +1.0 | 正奖励 | 速度跟踪 |
| tracking_ang_vel | +0.5 | 正奖励 | 角速度跟踪 |
| lin_vel_z | -2.0 | 惩罚 | 抑制垂直速度 |
| orientation | -0.2 | 惩罚 | 保持水平姿态 |
| base_height | -1.0 | 惩罚 | 保持目标高度 |
| action_rate | -0.01 | 惩罚 | 动作平滑 |
| smoothness | -0.01 | 惩罚 | 二阶平滑 |
| joint_power | -2e-5 | 惩罚 | 能效优化 |

---

## 总结

HIMLoco 的奖励函数设计是一个精心设计的多目标优化系统，通过 22 个可配置的奖励项实现了：

✅ **性能驱动** - 准确跟踪速度命令  
✅ **稳定保证** - 保持身体姿态和高度  
✅ **运动质量** - 平滑、自然的步态  
✅ **能效优化** - 降低功率消耗  
✅ **安全约束** - 避免碰撞和超限  

通过本文档，您应该能够：
- 理解每个奖励函数的作用和实现
- 根据任务需求调整奖励权重
- 诊断和解决训练中的问题
- 优化机器人的运动性能

---

**文档版本：** 2.0  
**最后更新：** 2025-10-24  
**作者：** GitHub Copilot  
**项目：** HIMLoco - H∞ Locomotion Control

📖 **完整文档长度：** 约 15,000 字  
🎯 **覆盖内容：** 22 个奖励函数 + 完整代码解析  
�� **实用工具：** 调试清单 + 优化指南 + 代码示例
