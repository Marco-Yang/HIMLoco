# HIMLoco 训练流程详细文档

## 文档信息
- **版本**: v1.0
- **创建日期**: 2024年10月24日
- **项目**: HIMLoco - H∞ Locomotion Control
- **目标**: 详细解释训练过程的完整调用流程

---

## 目录
1. [训练流程总览](#1-训练流程总览)
2. [入口点：train.py](#2-入口点trainpy)
3. [任务注册与环境创建](#3-任务注册与环境创建)
4. [Runner初始化](#4-runner初始化)
5. [训练循环详解](#5-训练循环详解)
6. [环境交互流程](#6-环境交互流程)
7. [算法更新流程](#7-算法更新流程)
8. [模型保存机制](#8-模型保存机制)
9. [完整调用链图](#9-完整调用链图)
10. [配置参数说明](#10-配置参数说明)

---

## 1. 训练流程总览

### 1.1 整体架构
```
命令行启动
    ↓
train.py (入口点)
    ↓
task_registry.make_env() (创建环境)
    ↓
task_registry.make_alg_runner() (创建训练器)
    ↓
runner.learn() (主训练循环)
    ↓
    ├── Rollout阶段 (数据收集)
    │   ├── alg.act() (选择动作)
    │   ├── env.step() (环境交互)
    │   └── alg.process_env_step() (处理数据)
    │
    └── Learning阶段 (策略更新)
        ├── alg.compute_returns() (计算回报)
        └── alg.update() (更新策略)
```

### 1.2 关键文件位置
| 文件 | 路径 | 功能 |
|------|------|------|
| 训练入口 | `legged_gym/scripts/train.py` | 启动训练 |
| 任务注册 | `legged_gym/utils/task_registry.py` | 环境和算法工厂 |
| 训练器 | `rsl_rl/runners/him_on_policy_runner.py` | 训练主循环 |
| 算法 | `rsl_rl/algorithms/him_ppo.py` | HIMPPO算法 |
| 环境 | `legged_gym/envs/base/legged_robot.py` | 机器人环境 |
| 存储 | `rsl_rl/storage/him_rollout_storage.py` | 经验回放 |
| 网络 | `rsl_rl/modules/him_actor_critic.py` | Actor-Critic网络 |
| 估计器 | `rsl_rl/modules/him_estimator.py` | H∞估计器 |

---

## 2. 入口点：train.py

### 2.1 文件位置
```
legged_gym/scripts/train.py
```

### 2.2 核心代码
```python
def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, 
                     init_at_random_ep_len=train_cfg.runner.init_at_random_ep_len)
```

### 2.3 执行步骤
1. **解析命令行参数**：获取任务名称、设备等配置
2. **创建环境**：调用 `task_registry.make_env()`
3. **创建训练器**：调用 `task_registry.make_alg_runner()`
4. **开始训练**：调用 `runner.learn()`

### 2.4 命令行示例
```bash
# 基本训练命令
python train.py --task aliengo

# 带参数的训练
python train.py --task aliengo --headless --num_envs 4096
```

---

## 3. 任务注册与环境创建

### 3.1 task_registry.make_env()

**文件位置**: `legged_gym/utils/task_registry.py`

#### 功能说明
创建Isaac Gym仿真环境，包括：
- 加载配置文件
- 初始化物理引擎
- 创建机器人和地形
- 设置随机种子

#### 核心代码
```python
def make_env(self, name, args=None, env_cfg=None):
    # 获取任务类
    task_class = self.get_task_class(name)
    
    # 加载配置
    if env_cfg is None:
        env_cfg, _ = self.get_cfgs(name)
    
    # 更新配置
    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
    set_seed(env_cfg.seed)
    
    # 解析仿真参数
    sim_params = parse_sim_params(args, {"sim": class_to_dict(env_cfg.sim)})
    
    # 创建环境
    env = task_class(cfg=env_cfg,
                    sim_params=sim_params,
                    physics_engine=args.physics_engine,
                    sim_device=args.sim_device,
                    headless=args.headless)
    return env, env_cfg
```

#### 返回值
- `env`: LeggedRobot环境实例
- `env_cfg`: 环境配置对象

### 3.2 task_registry.make_alg_runner()

#### 功能说明
创建训练算法和训练器，包括：
- 初始化Actor-Critic网络
- 初始化HIMPPO算法
- 创建日志目录
- 加载检查点（如果resume=True）

#### 核心代码
```python
def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default"):
    # 加载训练配置
    if train_cfg is None:
        _, train_cfg = self.get_cfgs(name)
    
    # 创建日志目录
    if log_root == "default":
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
    
    # 创建Runner
    train_cfg_dict = class_to_dict(train_cfg)
    runner = HIMOnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
    
    # 恢复训练（如果需要）
    if train_cfg.runner.resume:
        resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, 
                                    checkpoint=train_cfg.runner.checkpoint)
        runner.load(resume_path)
    
    return runner, train_cfg
```

#### 返回值
- `runner`: HIMOnPolicyRunner实例
- `train_cfg`: 训练配置对象

---

## 4. Runner初始化

### 4.1 HIMOnPolicyRunner.__init__()

**文件位置**: `rsl_rl/runners/him_on_policy_runner.py`

#### 初始化流程
```python
def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
    # 1. 保存配置
    self.cfg = train_cfg["runner"]
    self.alg_cfg = train_cfg["algorithm"]
    self.policy_cfg = train_cfg["policy"]
    self.device = device
    self.env = env
    
    # 2. 确定观测维度
    if self.env.num_privileged_obs is not None:
        num_critic_obs = self.env.num_privileged_obs
    else:
        num_critic_obs = self.env.num_obs
    
    # 3. 创建Actor-Critic网络
    actor_critic_class = eval(self.cfg["policy_class_name"])  # HIMActorCritic
    actor_critic = actor_critic_class(
        self.env.num_obs,
        num_critic_obs,
        self.env.num_one_step_obs,
        self.env.num_actions,
        **self.policy_cfg
    ).to(self.device)
    
    # 4. 创建HIMPPO算法
    alg_class = eval(self.cfg["algorithm_class_name"])  # HIMPPO
    self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)
    
    # 5. 初始化存储
    self.num_steps_per_env = self.cfg["num_steps_per_env"]
    self.alg.init_storage(
        self.env.num_envs,
        self.num_steps_per_env,
        [self.env.num_obs],
        [self.env.num_privileged_obs],
        [self.env.num_actions]
    )
    
    # 6. 初始化日志和计数器
    self.log_dir = log_dir
    self.writer = None
    self.current_learning_iteration = 0
    
    # 7. 重置环境
    self.env.reset()
```

#### 关键组件说明
- **HIMActorCritic**: 包含actor、critic和estimator
- **HIMPPO**: PPO算法的H∞变体
- **Storage**: 存储rollout数据的缓冲区

---

## 5. 训练循环详解

### 5.1 runner.learn() 主循环

**文件位置**: `rsl_rl/runners/him_on_policy_runner.py`

#### 代码结构
```python
def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    # 初始化TensorBoard
    if self.log_dir is not None and self.writer is None:
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
    
    # 随机初始化episode长度（可选）
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf, 
            high=int(self.env.max_episode_length)
        )
    
    # 获取初始观测
    obs = self.env.get_observations()
    privileged_obs = self.env.get_privileged_observations()
    critic_obs = privileged_obs if privileged_obs is not None else obs
    obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
    
    # 切换到训练模式
    self.alg.actor_critic.train()
    
    # 初始化统计变量
    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    
    tot_iter = self.current_learning_iteration + num_learning_iterations
    
    # 主训练循环
    for it in range(self.current_learning_iteration, tot_iter):
        start = time.time()
        
        # ========== Rollout阶段 ==========
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # 1. 选择动作
                actions = self.alg.act(obs, critic_obs)
                
                # 2. 环境交互
                obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = self.env.step(actions)
                
                # 3. 处理观测
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                
                # 4. 处理终止状态的特权观测
                next_critic_obs = critic_obs.clone().detach()
                next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()
                
                # 5. 存储transition
                self.alg.process_env_step(rewards, dones, infos, next_critic_obs)
                
                # 6. 记录episode信息
                if self.log_dir is not None:
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
        
        collection_time = time.time() - start
        
        # ========== Learning阶段 ==========
        start = time.time()
        self.alg.compute_returns(critic_obs)
        
        mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.alg.update()
        
        learn_time = time.time() - start
        
        # 记录日志
        if self.log_dir is not None:
            self.log(locals())
        
        # 保存模型
        if it % self.save_interval == 0:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
        
        ep_infos.clear()
    
    # 训练结束后保存最终模型
    self.current_learning_iteration += num_learning_iterations
    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
```

### 5.2 训练循环关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_learning_iterations` | 200000 | 总迭代次数 |
| `num_steps_per_env` | 100 | 每次迭代的rollout步数 |
| `save_interval` | 20 | 保存模型的间隔 |
| `num_envs` | 4096 | 并行环境数量 |

**每次迭代处理的总步数** = `num_steps_per_env` × `num_envs` = 100 × 4096 = 409,600 步

---

## 6. 环境交互流程

### 6.1 alg.act() - 动作选择

**文件位置**: `rsl_rl/algorithms/him_ppo.py`

#### 代码实现
```python
def act(self, obs, critic_obs):
    # 1. 通过actor-critic生成动作
    self.transition.actions = self.actor_critic.act(obs).detach()
    
    # 2. 评估状态价值
    self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
    
    # 3. 计算动作的对数概率
    self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
    
    # 4. 记录动作均值和标准差
    self.transition.action_mean = self.actor_critic.action_mean.detach()
    self.transition.action_sigma = self.actor_critic.action_std.detach()
    
    # 5. 记录观测
    self.transition.observations = obs
    self.transition.critic_observations = critic_obs
    
    return self.transition.actions
```

#### Actor-Critic网络前向传播

**文件位置**: `rsl_rl/modules/him_actor_critic.py`

```python
def act(self, obs_history=None, **kwargs):
    # 1. 更新动作分布
    self.update_distribution(obs_history)
    # 2. 从分布中采样
    return self.distribution.sample()

def update_distribution(self, obs_history):
    # 1. 使用estimator提取速度和潜变量
    with torch.no_grad():
        vel, latent = self.estimator(obs_history)
    
    # 2. 拼接输入：当前观测 + 估计速度 + 潜变量
    actor_input = torch.cat((obs_history[:,:self.num_one_step_obs], vel, latent), dim=-1)
    
    # 3. Actor网络前向传播得到均值
    mean = self.actor(actor_input)
    
    # 4. 创建正态分布
    self.distribution = Normal(mean, mean*0. + self.std)
```

### 6.2 env.step() - 环境步进

**文件位置**: `legged_gym/envs/base/legged_robot.py`

#### 执行流程
```python
def step(self, actions):
    # 1. 裁剪动作到合理范围
    clip_actions = self.cfg.normalization.clip_actions
    self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
    
    # 2. 处理动作延迟（domain randomization）
    self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)
    delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
    if self.cfg.domain_rand.delay:
        for i in range(self.cfg.control.decimation):
            self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
    
    # 3. 物理仿真循环（decimation次）
    self.render()
    for _ in range(self.cfg.control.decimation):
        # 计算关节力矩
        self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)
        # 应用力矩
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        # 步进物理引擎
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        # 刷新状态
        self.gym.refresh_dof_state_tensor(self.sim)
    
    # 4. 物理步进后处理
    termination_ids, termination_privileged_obs = self.post_physics_step()
    
    # 5. 裁剪观测
    clip_obs = self.cfg.normalization.clip_observations
    self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
    if self.privileged_obs_buf is not None:
        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
    
    return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_privileged_obs
```

#### post_physics_step() - 后处理

```python
def post_physics_step(self):
    # 1. 刷新物理状态
    self.gym.refresh_actor_root_state_tensor(self.sim)
    self.gym.refresh_net_contact_force_tensor(self.sim)
    self.gym.refresh_rigid_body_state_tensor(self.sim)
    
    # 2. 更新步数计数器
    self.episode_length_buf += 1
    self.common_step_counter += 1
    
    # 3. 准备基础量（位置、速度、重力方向等）
    self.base_quat[:] = self.root_states[:, 3:7]
    self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
    self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
    self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
    self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
    self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
    
    # 4. 回调函数
    self._post_physics_step_callback()
    
    # 5. 计算观测、奖励、终止条件
    self.check_termination()
    self.compute_reward()
    env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
    termination_privileged_obs = self.compute_termination_observations(env_ids)
    
    # 6. 重置结束的环境
    self.reset_idx(env_ids)
    
    # 7. 计算新观测
    self.compute_observations()
    
    # 8. 更新历史信息
    self.last_last_actions[:] = self.last_actions[:]
    self.last_actions[:] = self.actions[:]
    self.last_dof_vel[:] = self.dof_vel[:]
    self.last_root_vel[:] = self.root_states[:, 7:13]
    
    return env_ids, termination_privileged_obs
```

### 6.3 alg.process_env_step() - 处理环境步骤

**文件位置**: `rsl_rl/algorithms/him_ppo.py`

```python
def process_env_step(self, rewards, dones, infos, next_critic_obs):
    # 1. 保存下一个critic观测（用于estimator训练）
    self.transition.next_critic_observations = next_critic_obs.clone()
    
    # 2. 保存奖励
    self.transition.rewards = rewards.clone()
    
    # 3. 保存done标志
    self.transition.dones = dones
    
    # 4. Time-out bootstrapping
    # 如果episode因为时间限制结束（而非真正失败），使用value function进行bootstrapping
    if 'time_outs' in infos:
        self.transition.rewards += self.gamma * torch.squeeze(
            self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
        )
    
    # 5. 将transition添加到存储
    self.storage.add_transitions(self.transition)
    
    # 6. 清空transition缓存
    self.transition.clear()
    
    # 7. 重置actor-critic的隐藏状态（如果是循环网络）
    self.actor_critic.reset(dones)
```

---

## 7. 算法更新流程

### 7.1 alg.compute_returns() - 计算回报

**文件位置**: `rsl_rl/algorithms/him_ppo.py`

```python
def compute_returns(self, last_critic_obs):
    # 1. 计算最后一步的value
    last_values = self.actor_critic.evaluate(last_critic_obs).detach()
    
    # 2. 使用GAE计算returns和advantages
    self.storage.compute_returns(last_values, self.gamma, self.lam)
```

#### GAE (Generalized Advantage Estimation)

**文件位置**: `rsl_rl/storage/him_rollout_storage.py`

```python
def compute_returns(self, last_values, gamma, lam):
    advantage = 0
    # 从后向前计算
    for step in reversed(range(self.num_transitions_per_env)):
        if step == self.num_transitions_per_env - 1:
            next_values = last_values
        else:
            next_values = self.values[step + 1]
        
        # 计算TD error
        next_is_not_terminal = 1.0 - self.dones[step].float()
        delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
        
        # 累积advantage
        advantage = delta + next_is_not_terminal * gamma * lam * advantage
        
        # returns = advantage + values
        self.returns[step] = advantage + self.values[step]
    
    # 标准化advantages
    self.advantages = self.returns - self.values
    self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
```

### 7.2 alg.update() - 策略更新

**文件位置**: `rsl_rl/algorithms/him_ppo.py`

#### 主要步骤

```python
def update(self):
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_estimation_loss = 0
    mean_swap_loss = 0
    
    # 创建mini-batch生成器
    generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    
    # 遍历所有mini-batches
    for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, \
        target_values_batch, advantages_batch, returns_batch, \
        old_actions_log_prob_batch, old_mu_batch, old_sigma_batch in generator:
        
        # ========== 1. 前向传播 ==========
        self.actor_critic.act(obs_batch)
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        value_batch = self.actor_critic.evaluate(critic_obs_batch)
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        entropy_batch = self.actor_critic.entropy
        
        # ========== 2. KL散度和自适应学习率 ==========
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + 
                    (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / 
                    (2.0 * torch.square(sigma_batch)) - 0.5, 
                    axis=-1
                )
                kl_mean = torch.mean(kl)
                
                # 调整学习率
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
        
        # ========== 3. Estimator更新 ==========
        estimation_loss, swap_loss = self.actor_critic.estimator.update(
            obs_batch, 
            next_critic_obs_batch, 
            lr=self.learning_rate
        )
        
        # ========== 4. PPO Surrogate Loss ==========
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        
        # ========== 5. Value Function Loss ==========
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + \
                (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()
        
        # ========== 6. 总损失 ==========
        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
        
        # ========== 7. 梯度更新 ==========
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # 累积损失
        mean_value_loss += value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()
        mean_estimation_loss += estimation_loss
        mean_swap_loss += swap_loss
    
    # 计算平均损失
    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_estimation_loss /= num_updates
    mean_swap_loss /= num_updates
    
    # 清空存储
    self.storage.clear()
    
    return mean_value_loss, mean_surrogate_loss, estimation_loss, swap_loss
```

### 7.3 损失函数详解

#### 1. Surrogate Loss (策略损失)
```python
# PPO的clip objective
ratio = π_θ(a|s) / π_θ_old(a|s)
L^CLIP = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

#### 2. Value Loss (价值损失)
```python
# Clipped value loss
L^V = max((V - V_target)^2, (V_clip - V_target)^2)
```

#### 3. Entropy Loss (熵损失)
```python
# 鼓励探索
L^S = -β * H(π_θ)
```

#### 4. Estimation Loss (估计器损失)
H∞估计器的专有损失，用于学习速度和环境潜变量

---

## 8. 模型保存机制

### 8.1 自动保存逻辑

**文件位置**: `rsl_rl/runners/him_on_policy_runner.py`

```python
# 在learn()函数中
for it in range(self.current_learning_iteration, tot_iter):
    # ... 训练代码 ...
    
    # 每隔save_interval保存一次
    if it % self.save_interval == 0:
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

# 训练结束后保存最终模型
self.current_learning_iteration += num_learning_iterations
self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
```

### 8.2 保存内容

```python
def save(self, path, infos=None):
    torch.save({
        'model_state_dict': self.alg.actor_critic.state_dict(),
        'optimizer_state_dict': self.alg.optimizer.state_dict(),
        'iter': self.current_learning_iteration,
        'infos': infos,
    }, path)
```

### 8.3 保存路径格式

```
logs/
└── {experiment_name}/          # 例如: rough_aliengo
    └── {timestamp}/            # 例如: Oct23_21-14-27_
        ├── model_20.pt
        ├── model_40.pt
        ├── model_60.pt
        ├── ...
        └── model_{final_iter}.pt
```

---

## 9. 完整调用链图

### 9.1 时序图

```
用户          train.py         task_registry      HIMOnPolicyRunner      HIMPPO         LeggedRobot
 │                │                   │                    │                │                │
 │  python train  │                   │                    │                │                │
 │──────────────>│                    │                    │                │                │
 │                │   make_env()      │                    │                │                │
 │                │──────────────────>│                    │                │                │
 │                │                   │  LeggedRobot()     │                │                │
 │                │                   │───────────────────────────────────>│                │
 │                │                   │<───────────────────────────────────│                │
 │                │<──────────────────│                    │                │                │
 │                │  make_alg_runner()│                    │                │                │
 │                │──────────────────>│                    │                │                │
 │                │                   │  HIMOnPolicyRunner()                │                │
 │                │                   │───────────────────>│                │                │
 │                │                   │                    │  HIMPPO()      │                │
 │                │                   │                    │───────────────>│                │
 │                │                   │                    │<───────────────│                │
 │                │                   │<───────────────────│                │                │
 │                │<──────────────────│                    │                │                │
 │                │  learn()          │                    │                │                │
 │                │───────────────────────────────────────>│                │                │
 │                │                   │                    │                │                │
 │                │                   │    ╔═══════════ 训练循环 ═══════════╗                │
 │                │                   │    ║                                ║                │
 │                │                   │    ║  act()                         ║                │
 │                │                   │    ║───────────────>│               ║                │
 │                │                   │    ║<───────────────│               ║                │
 │                │                   │    ║  step()                        ║                │
 │                │                   │    ║────────────────────────────────────────────────>│
 │                │                   │    ║<────────────────────────────────────────────────│
 │                │                   │    ║  process_env_step()            ║                │
 │                │                   │    ║───────────────>│               ║                │
 │                │                   │    ║                                ║                │
 │                │                   │    ║  compute_returns()             ║                │
 │                │                   │    ║───────────────>│               ║                │
 │                │                   │    ║                                ║                │
 │                │                   │    ║  update()                      ║                │
 │                │                   │    ║───────────────>│               ║                │
 │                │                   │    ║<───────────────│               ║                │
 │                │                   │    ║                                ║                │
 │                │                   │    ╚════════════════════════════════╝                │
```

### 9.2 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练主循环                                 │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              Rollout阶段 (数据收集)                     │     │
│  │                                                          │     │
│  │  观测 ──> Actor-Critic ──> 动作 ──> 环境 ──> 新观测     │     │
│  │    │           │              │        │         │       │     │
│  │    │           └─ Value ──────┼────────┘         │       │     │
│  │    │                          │                  │       │     │
│  │    └──────────────────────────┼──────────────────┘       │     │
│  │                               │                          │     │
│  │                               ▼                          │     │
│  │                        Rollout Storage                   │     │
│  │                     (obs, actions, rewards,              │     │
│  │                      values, log_probs)                  │     │
│  └────────────────────────────────────────────────────────┘     │
│                               │                                   │
│                               ▼                                   │
│  ┌────────────────────────────────────────────────────────┐     │
│  │            Learning阶段 (策略更新)                       │     │
│  │                                                          │     │
│  │  Rollout Storage ──> compute_returns() ──> GAE          │     │
│  │                               │                          │     │
│  │                               ▼                          │     │
│  │                      Mini-batch Sampling                 │     │
│  │                               │                          │     │
│  │                               ▼                          │     │
│  │                    ┌──────────────────┐                 │     │
│  │                    │  PPO Update      │                 │     │
│  │                    │  - Surrogate     │                 │     │
│  │                    │  - Value         │                 │     │
│  │                    │  - Entropy       │                 │     │
│  │                    └──────────────────┘                 │     │
│  │                               │                          │     │
│  │                    ┌──────────────────┐                 │     │
│  │                    │ Estimator Update │                 │     │
│  │                    │  - Estimation    │                 │     │
│  │                    │  - Swap loss     │                 │     │
│  │                    └──────────────────┘                 │     │
│  │                               │                          │     │
│  │                               ▼                          │     │
│  │                     Updated Actor-Critic                 │     │
│  └────────────────────────────────────────────────────────┘     │
│                               │                                   │
│                               ▼                                   │
│                         Save Checkpoint                           │
│                      (every save_interval)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 配置参数说明

### 10.1 训练器配置 (Runner Config)

**文件位置**: `legged_gym/envs/base/legged_robot_config.py`

```python
class runner:
    policy_class_name = 'HIMActorCritic'
    algorithm_class_name = 'HIMPPO'
    num_steps_per_env = 100          # 每次迭代的rollout步数
    max_iterations = 200000          # 最大训练迭代次数
    
    # 日志
    save_interval = 20               # 保存间隔
    experiment_name = 'test'         # 实验名称
    run_name = ''                    # 运行名称
    
    # 加载和恢复
    resume = False                   # 是否恢复训练
    load_run = -1                    # -1 = 最新运行
    checkpoint = -1                  # -1 = 最新检查点
    resume_path = None               # 恢复路径
```

### 10.2 算法配置 (Algorithm Config)

```python
class algorithm:
    # PPO参数
    clip_param = 0.2                 # PPO裁剪参数
    num_learning_epochs = 5          # 每次更新的epoch数
    num_mini_batches = 4             # mini-batch数量
    learning_rate = 1.e-3            # 学习率
    schedule = 'adaptive'            # 学习率调度策略
    gamma = 0.99                     # 折扣因子
    lam = 0.95                       # GAE lambda
    desired_kl = 0.01                # 目标KL散度
    max_grad_norm = 1.0              # 梯度裁剪阈值
    
    # 损失权重
    value_loss_coef = 1.0            # 价值损失系数
    entropy_coef = 0.01              # 熵损失系数
```

### 10.3 策略网络配置 (Policy Config)

```python
class policy:
    # Actor网络
    actor_hidden_dims = [512, 256, 128]
    
    # Critic网络
    critic_hidden_dims = [512, 256, 128]
    
    # 激活函数
    activation = 'elu'
    
    # 初始化
    init_noise_std = 1.0
    
    # Estimator配置
    temporal_steps = 10              # 历史步数
    enc_hidden_dims = [128, 64, 16]  # 编码器隐藏层
    tar_hidden_dims = [128, 64]      # 目标网络隐藏层
    num_prototype = 32               # 原型数量
    temperature = 3.0                # 温度参数
```

### 10.4 关键参数计算

#### 每次迭代的数据量
```
总步数 = num_steps_per_env × num_envs
       = 100 × 4096
       = 409,600 步
```

#### 每次更新的mini-batch数量
```
总mini-batches = num_learning_epochs × num_mini_batches
                = 5 × 4
                = 20 个mini-batches
```

#### 每个mini-batch的大小
```
batch_size = (num_steps_per_env × num_envs) / num_mini_batches
           = (100 × 4096) / 4
           = 102,400 步
```

#### 训练总步数
```
总步数 = max_iterations × num_steps_per_env × num_envs
       = 200,000 × 100 × 4096
       = 81,920,000,000 步 (约819亿步)
```

---

## 附录A：代码文件清单

### A.1 核心训练文件

| 序号 | 文件路径 | 主要类/函数 | 功能说明 |
|------|----------|-------------|----------|
| 1 | `legged_gym/scripts/train.py` | `train()` | 训练入口 |
| 2 | `legged_gym/utils/task_registry.py` | `TaskRegistry` | 任务注册和工厂 |
| 3 | `rsl_rl/runners/him_on_policy_runner.py` | `HIMOnPolicyRunner` | 训练主循环 |
| 4 | `rsl_rl/algorithms/him_ppo.py` | `HIMPPO` | PPO算法实现 |
| 5 | `rsl_rl/storage/him_rollout_storage.py` | `HIMRolloutStorage` | 经验存储 |
| 6 | `rsl_rl/modules/him_actor_critic.py` | `HIMActorCritic` | 策略网络 |
| 7 | `rsl_rl/modules/him_estimator.py` | `HIMEstimator` | H∞估计器 |
| 8 | `legged_gym/envs/base/legged_robot.py` | `LeggedRobot` | 机器人环境 |

### A.2 配置文件

| 序号 | 文件路径 | 说明 |
|------|----------|------|
| 1 | `legged_gym/envs/base/legged_robot_config.py` | 基础配置 |
| 2 | `legged_gym/envs/aliengo/aliengo_config.py` | Aliengo特定配置 |

---

## 附录B：常见问题排查

### B.1 训练不收敛

**可能原因**:
1. 学习率过大或过小
2. 奖励函数设计不合理
3. 观测归一化问题

**排查步骤**:
1. 检查TensorBoard中的loss曲线
2. 调整`learning_rate`和`schedule`
3. 查看奖励函数文档，调整奖励权重

### B.2 内存不足

**可能原因**:
1. `num_envs`设置过大
2. `num_steps_per_env`设置过大

**解决方案**:
```python
# 减少并行环境数
--num_envs 2048  # 默认4096

# 或减少rollout步数
num_steps_per_env = 50  # 默认100
```

### B.3 训练速度慢

**优化建议**:
1. 使用headless模式：`--headless`
2. 增加decimation减少物理步数
3. 使用更快的GPU
4. 减少`num_learning_epochs`

---

## 文档版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2024-10-24 | 初始版本，包含完整训练流程 |

---

**文档结束**
