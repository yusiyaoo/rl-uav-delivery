"""
UAV City Logistics Delivery — Configuration & Hyperparameters
All parameters are centralized here for easy tuning.
"""

import numpy as np

# ===================== Environment Settings =====================

# 3D space boundaries (meters)
SPACE_X = 500.0   # east-west
SPACE_Y = 500.0   # north-south
SPACE_Z = 200.0   # altitude

# Start and goal positions
START_POS = np.array([30.0, 30.0, 60.0], dtype=np.float32)
GOAL_POS = np.array([460.0, 460.0, 50.0], dtype=np.float32)
GOAL_RADIUS = 15.0   # threshold for "arrived"

# UAV physical constraints
MAX_SPEED = 20.0        # m/s, maximum velocity per axis
STEP_SIZE = 5.0         # m, displacement per action step
MAX_BATTERY = 1000.0    # energy units
HOVER_ENERGY = 0.8      # energy per hover step
MOVE_ENERGY = 1.0       # base energy per move step
CLIMB_ENERGY = 1.5      # energy multiplier for ascending
DESCEND_ENERGY = 0.7    # energy multiplier for descending
MIN_ALTITUDE = 10.0     # m, minimum flight altitude (above ground)
MAX_ALTITUDE = SPACE_Z  # maximum flight altitude

# Sensor and observation
SENSOR_RANGE = 150.0    # m, maximum distance the UAV can sense obstacles
SAFETY_MARGIN = 20.0    # m, distance below which risk penalty activates

# Number of buildings
N_BUILDINGS = 8         # randomly generated buildings
BUILDING_MIN_SIDE = 8.0   # m
BUILDING_MAX_SIDE = 35.0  # m
BUILDING_MIN_HEIGHT = 10.0  # m
BUILDING_MAX_HEIGHT = 90.0 # m

# No-fly zones
N_NOFLY_ZONES = 1
NOFLY_MIN_RADIUS = 20.0
NOFLY_MAX_RADIUS = 35.0

# Wind field parameters
WIND_PREVAILING = np.array([3.0, -1.0], dtype=np.float32)  # prevailing wind (x, y) m/s
WIND_VARIANCE = 2.0  # local wind variation amplitude

# ===================== Reward Settings =====================

REWARD_ARRIVE = 500.0
REWARD_COLLISION = -50.0
REWARD_STEP = -0.2          # time penalty per step
REWARD_PROGRESS_SCALE = 0.15 # bonus per meter closer to goal
REWARD_RISK_MAX = -0.5      # maximum risk penalty when right next to obstacle
REWARD_WIND_SCALE = 0.05    # scale for wind alignment reward

# ===================== Training Settings =====================

N_EPISODES = 800
MAX_STEPS_PER_EPISODE = 400
BATCH_SIZE = 64
GAMMA = 0.99          # discount factor
LR = 3e-4             # learning rate
TAU = 0.005           # soft update rate for target network
BUFFER_CAPACITY = 100_000
MIN_BUFFER_SIZE = 2000

# Epsilon-greedy exploration
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995       # per episode (faster decay)

# Prioritized Experience Replay
PER_ALPHA = 0.6        # prioritization exponent (0 = uniform, 1 = full priority)
PER_BETA_START = 0.4   # importance-sampling correction (anneals to 1)
PER_BETA_END = 1.0
PER_BETA_ANNEAL = 0.999  # per episode
PER_EPSILON = 1e-6     # small constant to avoid zero priority

# Network architecture
INPUT_DIM = 15         # state dimension
HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 128
N_ACTIONS = 7

# Evaluation
EVAL_FREQ = 50         # evaluate every N episodes
EVAL_EPISODES = 10

# Saving
SAVE_PATH = "results/model.pth"
LOG_PATH = "results/training_log.npz"

# Random seeds
SEED = 42
