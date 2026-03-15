import os
import shutil
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback


ALGORITHMS = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}


class WaypointSearchEnv(gym.Env):
    """Waypoint search environment for training a trajectory planning policy."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        waypoint_array,
        max_steps,
        pedestrian_prob=0.35,
        distance_penalty=0.002,
        revisit_penalty=0.25,
        pedestrian_reward=1.0,
        seed=0,
    ):
        super().__init__()
        self.waypoint_array = waypoint_array
        self.n_waypoints = waypoint_array.shape[0]
        self.max_steps = max_steps
        self.pedestrian_prob = pedestrian_prob
        self.distance_penalty = distance_penalty
        self.revisit_penalty = revisit_penalty
        self.pedestrian_reward = pedestrian_reward
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(self.n_waypoints)
        obs_dim = 2 + self.n_waypoints * 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.current_idx = 0
        self.steps = 0
        self.visited = np.zeros(self.n_waypoints, dtype=np.float32)
        self.pedestrians = np.zeros(self.n_waypoints, dtype=np.float32)

    def _distance_norm(self, from_idx, to_idx):
        distance = np.linalg.norm(
            self.waypoint_array[from_idx] - self.waypoint_array[to_idx], ord=2
        )
        return float(distance / (distance + 1.0))

    def _obs(self):
        current = np.array(
            [self.current_idx / max(1, self.n_waypoints - 1)], dtype=np.float32
        )
        budget = np.array([self.steps / max(1, self.max_steps)], dtype=np.float32)
        return np.concatenate((current, budget, self.visited, self.pedestrians)).astype(
            np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_idx = 0
        self.steps = 0
        self.visited[:] = 0
        self.visited[0] = 1
        self.pedestrians = (
            self.rng.random(self.n_waypoints) < self.pedestrian_prob
        ).astype(np.float32)
        self.pedestrians[0] = 0
        return self._obs(), {}

    def step(self, action):
        reward = 0.0
        done = False
        truncated = False

        if self.visited[action] == 1:
            reward -= self.revisit_penalty
        else:
            reward += self.pedestrian_reward * float(self.pedestrians[action])
            reward -= self.distance_penalty * self._distance_norm(self.current_idx, action)
            self.visited[action] = 1

        self.current_idx = int(action)
        self.steps += 1

        if self.steps >= self.max_steps or np.all(self.visited == 1):
            done = True

        return self._obs(), float(reward), done, truncated, {}


def _build_model(algorithm, env, random_seed, kwargs):
    algo = algorithm.upper()
    if algo not in ALGORITHMS:
        raise ValueError(
            f"Unsupported rl_algorithm '{algorithm}'. Available: {list(ALGORITHMS)}"
        )

    model_cls = ALGORITHMS[algo]
    return model_cls("MlpPolicy", env, seed=random_seed, verbose=0, **kwargs)


def generate_candidate_waypoints(cfg):
    """Generate waypoint candidates (no path*.csv required)."""
    x0, y0, z0 = getattr(cfg, "rl_initial_position", (-320.34, -206.58, 128.0))
    step = float(getattr(cfg, "rl_waypoint_spacing", 40.0))
    rings = int(getattr(cfg, "rl_waypoint_rings", 2))

    candidates = [[float(x0), float(y0), float(z0)]]
    for r in range(1, rings + 1):
        radius = r * step
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            rad = np.deg2rad(angle)
            x = x0 + radius * np.cos(rad)
            y = y0 + radius * np.sin(rad)
            candidates.append([float(x), float(y), float(z0)])

    max_points = int(getattr(cfg, "rl_max_candidate_waypoints", len(candidates)))
    return candidates[:max_points]


def _find_existing_model_path(model_dir, model_name):
    best_path = os.path.join(model_dir, f"{model_name}_best.zip")
    final_path = os.path.join(model_dir, f"{model_name}_final.zip")
    base_path = os.path.join(model_dir, f"{model_name}.zip")

    if os.path.exists(best_path):
        return best_path
    if os.path.exists(final_path):
        return final_path
    if os.path.exists(base_path):
        return base_path
    return None


def train_or_load_model(waypoints, cfg):
    max_steps = min(
        len(waypoints) - 1,
        int(getattr(cfg, "rl_planner_max_steps", len(waypoints) - 1)),
    )
    env = WaypointSearchEnv(
        waypoint_array=np.asarray(waypoints, dtype=np.float32),
        max_steps=max_steps,
        pedestrian_prob=float(getattr(cfg, "rl_planner_pedestrian_probability", 0.35)),
        distance_penalty=float(getattr(cfg, "rl_planner_distance_penalty", 0.002)),
        revisit_penalty=float(getattr(cfg, "rl_planner_revisit_penalty", 0.25)),
        pedestrian_reward=float(getattr(cfg, "rl_planner_pedestrian_reward", 1.0)),
        seed=int(getattr(cfg, "random_seed", 1)),
    )

    model_dir = getattr(cfg, "rl_model_dir", "./trained_models")
    model_name = getattr(cfg, "rl_model_name", "trajectory_planner")
    force_retrain = bool(getattr(cfg, "rl_force_retrain", True))
    save_final = bool(getattr(cfg, "rl_save_final_model", True))
    total_timesteps = int(getattr(cfg, "rl_total_timesteps", 5000))
    eval_freq = int(getattr(cfg, "rl_eval_freq", 1000))
    n_eval_episodes = int(getattr(cfg, "rl_eval_episodes", 5))

    os.makedirs(model_dir, exist_ok=True)

    if not force_retrain:
        existing_path = _find_existing_model_path(model_dir, model_name)
        if existing_path is not None:
            model_cls = ALGORITHMS[getattr(cfg, "rl_algorithm", "PPO").upper()]
            return model_cls.load(existing_path, env=env), env, existing_path

    model = _build_model(
        getattr(cfg, "rl_algorithm", "PPO"),
        env,
        int(getattr(cfg, "random_seed", 1)),
        {
            "learning_rate": float(getattr(cfg, "rl_learning_rate", 3e-4)),
            "gamma": float(getattr(cfg, "rl_gamma", 0.99)),
        },
    )

    eval_env = WaypointSearchEnv(
        waypoint_array=np.asarray(waypoints, dtype=np.float32),
        max_steps=max_steps,
        pedestrian_prob=float(getattr(cfg, "rl_planner_pedestrian_probability", 0.35)),
        distance_penalty=float(getattr(cfg, "rl_planner_distance_penalty", 0.002)),
        revisit_penalty=float(getattr(cfg, "rl_planner_revisit_penalty", 0.25)),
        pedestrian_reward=float(getattr(cfg, "rl_planner_pedestrian_reward", 1.0)),
        seed=int(getattr(cfg, "random_seed", 1)) + 1000,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=max(1, eval_freq),
        n_eval_episodes=max(1, n_eval_episodes),
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    best_model_temp = os.path.join(model_dir, "best_model.zip")
    named_best_path = os.path.join(model_dir, f"{model_name}_best.zip")
    named_final_path = os.path.join(model_dir, f"{model_name}_final")

    if os.path.exists(best_model_temp):
        shutil.copyfile(best_model_temp, named_best_path)

    if save_final:
        model.save(named_final_path)

    model_path = _find_existing_model_path(model_dir, model_name)
    if model_path is None:
        model_path = named_final_path + ".zip"

    return model, env, model_path


def build_rl_trajectory(cfg):
    """Train (or load) an RL model and return a planned trajectory."""
    path_list = generate_candidate_waypoints(cfg)
    if len(path_list) <= 2:
        return path_list, None

    model, env, model_file = train_or_load_model(path_list, cfg)

    obs, _ = env.reset(seed=int(getattr(cfg, "random_seed", 1)))
    chosen_actions = []
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        if action == 0 or action in chosen_actions:
            unvisited = np.where(env.visited == 0)[0]
            unvisited = [
                int(x)
                for x in unvisited
                if int(x) != 0 and int(x) not in chosen_actions
            ]
            if not unvisited:
                break
            action = unvisited[0]

        obs, _, done, _, _ = env.step(action)
        if action > 0 and action not in chosen_actions:
            chosen_actions.append(action)
        if done:
            break

    remaining = [idx for idx in range(1, len(path_list)) if idx not in chosen_actions]
    ordered = [0] + chosen_actions + remaining
    final_path = [path_list[idx] for idx in ordered]

    return final_path, model_file
