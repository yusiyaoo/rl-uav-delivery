"""
UAV City Logistics Delivery Environment (Gym-style)

3D continuous space with buildings, no-fly zones, and wind field.
The UAV must navigate from start to goal while minimizing time, energy,
and safety risks.
"""

import numpy as np
from env.config import *


class Building:
    """A rectangular-prism building obstacle."""

    def __init__(self, rng: np.random.Generator):
        w = rng.uniform(BUILDING_MIN_SIDE, BUILDING_MAX_SIDE)
        d = rng.uniform(BUILDING_MIN_SIDE, BUILDING_MAX_SIDE)
        h = rng.uniform(BUILDING_MIN_HEIGHT, BUILDING_MAX_HEIGHT)
        # Position: ensure the building fits within the space
        cx = rng.uniform(w / 2 + 5, SPACE_X - w / 2 - 5)
        cy = rng.uniform(d / 2 + 5, SPACE_Y - d / 2 - 5)
        self.cx = cx
        self.cy = cy
        self.w = w       # x-extent
        self.d = d       # y-extent
        self.h = h       # z-extent (height)
        self.z_min = 0.0

    def contains(self, pos: np.ndarray) -> bool:
        """Check if a 3D point is inside this building."""
        x, y, z = pos
        if z < self.z_min or z > self.h:
            return False
        if abs(x - self.cx) > self.w / 2:
            return False
        if abs(y - self.cy) > self.d / 2:
            return False
        return True

    def distance_to(self, pos: np.ndarray) -> float:
        """Minimum distance from pos to this building surface (2D xy for simplicity)."""
        x, y, _ = pos
        # Closest point on the building footprint (in xy plane)
        cx_closest = np.clip(x, self.cx - self.w / 2, self.cx + self.w / 2)
        cy_closest = np.clip(y, self.cy - self.d / 2, self.cy + self.d / 2)
        return np.sqrt((x - cx_closest) ** 2 + (y - cy_closest) ** 2)

    def ray_distance(self, origin: np.ndarray, direction: np.ndarray) -> float:
        """
        Distance from origin along direction to intersection with this building.
        Returns SENSOR_RANGE if no intersection within range.
        Direction must be a unit vector.
        """
        t_min = 0.0
        t_max = SENSOR_RANGE

        # Check against each slab (x, y, z)
        # X slabs
        inv_d = 1.0 / direction[0] if abs(direction[0]) > 1e-8 else 0.0
        if inv_d != 0.0:
            t1 = (self.cx - self.w / 2 - origin[0]) * inv_d
            t2 = (self.cx + self.w / 2 - origin[0]) * inv_d
            t_near = min(t1, t2)
            t_far = max(t1, t2)
            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)
            if t_min > t_max:
                return SENSOR_RANGE

        # Y slabs
        inv_d = 1.0 / direction[1] if abs(direction[1]) > 1e-8 else 0.0
        if inv_d != 0.0:
            t1 = (self.cy - self.d / 2 - origin[1]) * inv_d
            t2 = (self.cy + self.d / 2 - origin[1]) * inv_d
            t_near = min(t1, t2)
            t_far = max(t1, t2)
            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)
            if t_min > t_max:
                return SENSOR_RANGE

        # Z slabs (0 to building height)
        inv_d = 1.0 / direction[2] if abs(direction[2]) > 1e-8 else 0.0
        if inv_d != 0.0:
            t1 = (0.0 - origin[2]) * inv_d
            t2 = (self.h - origin[2]) * inv_d
            t_near = min(t1, t2)
            t_far = max(t1, t2)
            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)
            if t_min > t_max:
                return SENSOR_RANGE

        if t_min > 0 and t_min < SENSOR_RANGE:
            return t_min
        return SENSOR_RANGE


class NoFlyZone:
    """A cylindrical no-fly zone."""

    def __init__(self, rng: np.random.Generator):
        self.cx = rng.uniform(100, SPACE_X - 100)
        self.cy = rng.uniform(100, SPACE_Y - 100)
        self.radius = rng.uniform(NOFLY_MIN_RADIUS, NOFLY_MAX_RADIUS)
        self.z_min = 0.0
        self.z_max = SPACE_Z

    def contains(self, pos: np.ndarray) -> bool:
        x, y, z = pos
        if z < self.z_min or z > self.z_max:
            return False
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.radius ** 2

    def distance_to(self, pos: np.ndarray) -> float:
        x, y, _ = pos
        return max(0.0, np.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2) - self.radius)


def _buildings_too_close(b1: Building, b2: Building, min_gap: float = 5.0) -> bool:
    """Check if two buildings are too close to each other."""
    dx = abs(b1.cx - b2.cx)
    dy = abs(b1.cy - b2.cy)
    return dx < (b1.w + b2.w) / 2 + min_gap and dy < (b1.d + b2.d) / 2 + min_gap


class UAVDeliveryEnv:
    """
    UAV City Logistics Delivery Environment.

    State space (15-dim):
        [dx_goal, dy_goal, dz_goal,   # relative position to goal (3)
         vx, vy, vz,                   # current velocity (3)
         battery_ratio,                # remaining energy (1)
         dist_front, dist_back, dist_left, dist_right, dist_up, dist_down,  # sensor (6)
         wind_x, wind_y]               # local wind (2)

    Action space (7 discrete):
        0: hover
        1: forward  (+x)
        2: backward (-x)
        3: left     (+y)
        4: right    (-y)
        5: up       (+z)
        6: down     (-z)
    """

    def __init__(self, seed: int = SEED):
        self.rng = np.random.default_rng(seed)
        self._action_to_delta = {
            0: np.array([0.0, 0.0, 0.0], dtype=np.float32),    # hover
            1: np.array([STEP_SIZE, 0.0, 0.0], dtype=np.float32),    # forward (+x)
            2: np.array([-STEP_SIZE, 0.0, 0.0], dtype=np.float32),   # backward (-x)
            3: np.array([0.0, STEP_SIZE, 0.0], dtype=np.float32),    # left (+y)
            4: np.array([0.0, -STEP_SIZE, 0.0], dtype=np.float32),   # right (-y)
            5: np.array([0.0, 0.0, STEP_SIZE], dtype=np.float32),    # up (+z)
            6: np.array([0.0, 0.0, -STEP_SIZE], dtype=np.float32),   # down (-z)
        }
        self._sensor_directions = {
            'front': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'back': np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            'left': np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'right': np.array([0.0, -1.0, 0.0], dtype=np.float32),
            'up': np.array([0.0, 0.0, 1.0], dtype=np.float32),
            'down': np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }
        self.reset()

    def _generate_obstacles(self):
        """Generate buildings and no-fly zones, avoiding start and goal areas."""
        self.buildings = []
        for _ in range(N_BUILDINGS * 3):  # try more, keep valid ones
            b = Building(self.rng)
            # Don't place buildings at start or goal
            if (np.linalg.norm(np.array([b.cx, b.cy]) - START_POS[:2]) < 60.0
                    or np.linalg.norm(np.array([b.cx, b.cy]) - GOAL_POS[:2]) < 60.0):
                continue
            # Don't overlap with existing buildings
            if any(_buildings_too_close(b, existing) for existing in self.buildings):
                continue
            self.buildings.append(b)
            if len(self.buildings) >= N_BUILDINGS:
                break

        self.nofly_zones = []
        for _ in range(N_NOFLY_ZONES * 2):
            nf = NoFlyZone(self.rng)
            if (np.linalg.norm(np.array([nf.cx, nf.cy]) - START_POS[:2]) < 60.0
                    or np.linalg.norm(np.array([nf.cx, nf.cy]) - GOAL_POS[:2]) < 60.0):
                continue
            self.nofly_zones.append(nf)
            if len(self.nofly_zones) >= N_NOFLY_ZONES:
                break

        # Collect all obstacles (buildings + no-fly zones) for collision checking
        self.obstacles = self.buildings + self.nofly_zones

    def _get_wind(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute local wind vector at position.
        Simple model: prevailing wind + sinusoidal spatial variations.
        """
        x, y, _ = pos
        local = np.array([
            WIND_VARIANCE * np.sin(2 * np.pi * y / SPACE_Y + np.pi / 4),
            WIND_VARIANCE * np.cos(2 * np.pi * x / SPACE_X),
        ], dtype=np.float32)
        return WIND_PREVAILING + local

    def _check_collision(self, pos: np.ndarray) -> bool:
        """Check if position is inside any obstacle."""
        for obs in self.obstacles:
            if obs.contains(pos):
                return True
        return False

    def _min_obstacle_distance(self, pos: np.ndarray) -> float:
        """Minimum distance from position to any obstacle."""
        min_dist = float('inf')
        for obs in self.obstacles:
            d = obs.distance_to(pos)
            if d < min_dist:
                min_dist = d
        return min_dist

    def _sensor_readings(self, pos: np.ndarray) -> np.ndarray:
        """Get 6-directional distance readings from position."""
        readings = []
        for _, direction in self._sensor_directions.items():
            min_t = SENSOR_RANGE
            for b in self.buildings:
                t = b.ray_distance(pos, direction)
                if t < min_t:
                    min_t = t
            # Also check no-fly zone boundaries (approximate with distance check)
            for nf in self.nofly_zones:
                # Simple radial check: if direction points toward no-fly zone center
                to_center = np.array([nf.cx - pos[0], nf.cy - pos[1], 0.0])
                d_center = np.linalg.norm(to_center)
                if d_center > 0:
                    to_center /= d_center
                    cos_angle = np.dot(direction[:2], to_center[:2])
                    if cos_angle > 0:
                        # Estimate intersection
                        # Distance to no-fly zone boundary along this direction
                        # (simplified: project center-to-pos onto direction)
                        # Line-circle intersection approximation
                        proj = np.dot(np.array([nf.cx - pos[0], nf.cy - pos[1], 0.0]), direction)
                        if proj > 0:
                            closest_approach = np.linalg.norm(
                                np.array([nf.cx - pos[0], nf.cy - pos[1]]) - proj * direction[:2]
                            )
                            if closest_approach < nf.radius:
                                t_nf = proj - np.sqrt(max(0, nf.radius ** 2 - closest_approach ** 2))
                                t_nf = max(0.0, t_nf)
                                if t_nf < min_t:
                                    min_t = t_nf
            readings.append(min_t / SENSOR_RANGE)  # normalize to [0, 1]
        return np.array(readings, dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reset the environment to start state."""
        self.pos = START_POS.copy()
        self.vel = np.zeros(3, dtype=np.float32)
        self.battery = MAX_BATTERY
        self.step_count = 0
        self._prev_distance = np.linalg.norm(self.pos - GOAL_POS)
        self._generate_obstacles()
        self._trajectory = [self.pos.copy()]
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector."""
        rel_goal = GOAL_POS - self.pos                     # 3
        vel_normalized = self.vel / MAX_SPEED              # 3
        battery_ratio = np.array([self.battery / MAX_BATTERY], dtype=np.float32)  # 1
        sensor = self._sensor_readings(self.pos)            # 6
        wind = self._get_wind(self.pos)                     # 2
        # Normalize wind by max wind speed
        wind_normalized = wind / (np.linalg.norm(WIND_PREVAILING) + WIND_VARIANCE + 0.1)
        # Normalize goal relative position
        rel_goal_normalized = rel_goal / np.array([SPACE_X, SPACE_Y, SPACE_Z])
        return np.concatenate([
            rel_goal_normalized,
            vel_normalized,
            battery_ratio,
            sensor,
            wind_normalized,
        ]).astype(np.float32)

    def step(self, action: int):
        """
        Execute one action step.

        Returns:
            obs (np.ndarray): new observation
            reward (float): reward for this step
            done (bool): episode termination
            info (dict): additional info
        """
        delta = self._action_to_delta[action]
        new_pos = self.pos + delta

        # Enforce boundaries
        new_pos[0] = np.clip(new_pos[0], 0.0, SPACE_X)
        new_pos[1] = np.clip(new_pos[1], 0.0, SPACE_Y)
        new_pos[2] = np.clip(new_pos[2], MIN_ALTITUDE, MAX_ALTITUDE)
        # If clamped, adjust delta to actual displacement
        actual_delta = new_pos - self.pos
        actual_distance = np.linalg.norm(actual_delta)

        # Update velocity (approximate)
        self.vel = actual_delta  # simplified: velocity equals displacement per step

        # --- Reward computation ---
        reward = 0.0

        # 1. Time penalty
        reward += REWARD_STEP

        # 2. Energy cost
        if action == 0:  # hover
            energy = HOVER_ENERGY
        elif action == 5:  # up
            energy = CLIMB_ENERGY * actual_distance / STEP_SIZE if STEP_SIZE > 0 else 1.0
        elif action == 6:  # down
            energy = DESCEND_ENERGY * actual_distance / STEP_SIZE if STEP_SIZE > 0 else 1.0
        else:  # horizontal
            energy = MOVE_ENERGY * actual_distance / STEP_SIZE if STEP_SIZE > 0 else 1.0

        # 3. Wind effect on energy
        wind = self._get_wind(self.pos)
        if actual_distance > 0 and action != 0:
            move_dir = actual_delta / actual_distance
            # Wind in horizontal plane only affects horizontal movement
            if action in [1, 2, 3, 4]:
                wind_component = np.dot(wind, move_dir[:2])
                # Positive = tailwind (saves energy), negative = headwind (costs more)
                wind_factor = 1.0 - wind_component * 0.1  # up to ±30% effect
            else:
                wind_factor = 1.0
            energy *= max(0.5, wind_factor)

        self.battery -= energy
        reward -= 0.05 * energy  # energy cost penalty
        reward += REWARD_WIND_SCALE * np.dot(wind, actual_delta[:2]) if actual_distance > 0 else 0.0

        # Move UAV
        self.pos = new_pos

        # 4. Collision check
        collided = self._check_collision(self.pos)
        if collided:
            reward += REWARD_COLLISION
            self._trajectory.append(self.pos.copy())
            return self._get_obs(), reward, True, {'collision': True, 'arrived': False}

        # 5. Risk penalty (proximity to obstacles)
        min_dist = self._min_obstacle_distance(self.pos)
        if min_dist < SAFETY_MARGIN:
            risk_penalty = REWARD_RISK_MAX * (1.0 - min_dist / SAFETY_MARGIN)
            reward += risk_penalty

        # 6. Progress reward (reduction in distance to goal)
        current_distance = np.linalg.norm(self.pos - GOAL_POS)
        progress = self._prev_distance - current_distance
        reward += REWARD_PROGRESS_SCALE * progress
        self._prev_distance = current_distance

        # 7. Arrival check
        arrived = current_distance < GOAL_RADIUS
        if arrived:
            reward += REWARD_ARRIVE
            self._trajectory.append(self.pos.copy())
            return self._get_obs(), reward, True, {'collision': False, 'arrived': True}

        # 8. Battery depletion
        if self.battery <= 0:
            self._trajectory.append(self.pos.copy())
            return self._get_obs(), reward, True, {'collision': False, 'arrived': False, 'battery_depleted': True}

        # 9. Step limit
        self.step_count += 1
        if self.step_count >= MAX_STEPS_PER_EPISODE:
            self._trajectory.append(self.pos.copy())
            return self._get_obs(), reward, True, {'collision': False, 'arrived': False, 'timeout': True}

        self._trajectory.append(self.pos.copy())
        return self._get_obs(), reward, False, {}

    def get_trajectory(self) -> np.ndarray:
        """Return the trajectory as a Nx3 array."""
        return np.array(self._trajectory)

    def seed(self, s: int):
        self.rng = np.random.default_rng(s)
