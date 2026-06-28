# UAV City Delivery Path Planning based on DQN

基于深度强化学习(Double DQN + Dueling Network + PER)的无人机城市物流配送三维路径规划。

## 场景

500m×500m×200m的城市低空环境，无人机从配送站出发，穿越随机建筑群和禁飞区，在风场影响下飞抵目标快递柜。

## 算法

- **Double DQN** — 解耦动作选择与价值估计
- **Dueling Network** — 分离状态价值V(s)和动作优势A(s,a)
- **Prioritized Experience Replay** — 优先高TD-error样本学习

## 项目结构

```
RL/
├── env/
│   ├── config.py          # 超参数和环境配置
│   └── uav_env.py         # 3D城市配送环境(Gym风格)
├── agent/
│   ├── network.py         # Dueling Q-Network
│   ├── replay_buffer.py   # 优先经验回放(SumTree)
│   └── dqn_agent.py       # Double DQN Agent
├── utils/
│   └── visualize.py       # 可视化(matplotlib)
├── train.py               # 训练主程序
├── eval.py                # 评估脚本
├── gen_figures.py         # 生成可视化图表
└── results/               # 输出（模型、图表）
```

## 使用

```bash
# 创建环境
conda create -n rl_uav python=3.10 -y && conda activate rl_uav
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy matplotlib tqdm

# 训练
python train.py

# 生成可视化图表
python gen_figures.py
```

## 环境要求

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA)
- NumPy, Matplotlib, tqdm 
