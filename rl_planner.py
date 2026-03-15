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


class DirectionDistanceEnv(gym.Env):
    """RL env where actions choose flight direction + distance step."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        initial_position,
        max_steps,
        action_distances,
        n_directions=8,
        area_limit=400.0,
        revisit_penalty=0.25,
        step_distance_penalty=0.002,
        pedestrian_reward=1.0,
        pedestrian_prob=0.35,
        seed=0,
    ):
        super().__init__()
        self.initial_position = np.asarray(initial_position, dtype=np.float32)
        self.max_steps = int(max_steps)
        self.action_distances = np.asarray(action_distances, dtype=np.float32)
        self.n_directions = int(n_directions)
        self.area_limit = float(area_limit)
        self.revisit_penalty = float(revisit_penalty)
        self.step_distance_penalty = float(step_distance_penalty)
        self.pedestrian_reward = float(pedestrian_reward)
        self.pedestrian_prob = float(pedestrian_prob)
        self.rng = np.random.default_rng(seed)

        self.n_actions = self.n_directions * len(self.action_distances)
        self.action_space = spaces.Discrete(self.n_actions)
        # [x_norm, y_norm, z_norm, step_ratio]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.position = self.initial_position.copy()
        self.steps = 0
        self.visited_cells = set()
        self.hotspots = np.zeros((0, 2), dtype=np.float32)

    def _obs(self):
        delta = self.position - self.initial_position
        return np.array(
            [
                float(np.clip(delta[0] / self.area_limit, -1.0, 1.0)),
                float(np.clip(delta[1] / self.area_limit, -1.0, 1.0)),
                float(np.clip(delta[2] / max(1.0, self.area_limit), -1.0, 1.0)),
                float(self.steps / max(1, self.max_steps)),
            ],
            dtype=np.float32,
        )

    def _decode_action(self, action):
        direction_idx = int(action) % self.n_directions
        distance_idx = int(action) // self.n_directions
        theta = 2.0 * np.pi * direction_idx / self.n_directions
        step_len = float(self.action_distances[distance_idx])
        dx = step_len * np.cos(theta)
        dy = step_len * np.sin(theta)
        return dx, dy, step_len

    def _cell(self, pos):
        return (int(np.round(pos[0] / 10.0)), int(np.round(pos[1] / 10.0)))

    def _hotspot_reward(self, pos_xy):
        if len(self.hotspots) == 0:
            return 0.0
        dists = np.linalg.norm(self.hotspots - pos_xy[None, :], axis=1)
        closest = float(np.min(dists))
        detect_prob = self.pedestrian_prob * np.exp(-(closest**2) / (2 * (35.0**2)))
        return self.pedestrian_reward if self.rng.random() < detect_prob else 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.position = self.initial_position.copy()
        self.steps = 0
        self.visited_cells = {self._cell(self.position)}

        n_hotspots = max(3, self.n_directions // 2)
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=n_hotspots)
        radii = self.rng.uniform(20.0, self.area_limit * 0.7, size=n_hotspots)
        hx = self.initial_position[0] + radii * np.cos(angles)
        hy = self.initial_position[1] + radii * np.sin(angles)
        self.hotspots = np.stack([hx, hy], axis=1).astype(np.float32)

        return self._obs(), {}

    def step(self, action):
        dx, dy, step_len = self._decode_action(action)
        self.position[0] += dx
        self.position[1] += dy

        reward = self._hotspot_reward(self.position[:2])
        reward -= self.step_distance_penalty * step_len

        cell = self._cell(self.position)
        if cell in self.visited_cells:
            reward -= self.revisit_penalty
        else:
            self.visited_cells.add(cell)

        self.steps += 1
        done = self.steps >= self.max_steps
        return self._obs(), float(reward), bool(done), False, {}


def _build_model(algorithm, env, random_seed, kwargs):
    algo = algorithm.upper()
    if algo not in ALGORITHMS:
        raise ValueError(
            f"Unsupported rl_algorithm '{algorithm}'. Available: {list(ALGORITHMS)}"
        )
    return ALGORITHMS[algo]("MlpPolicy", env, seed=random_seed, verbose=0, **kwargs)


def _find_existing_model_path(model_dir, model_name):
    for path in (
        os.path.join(model_dir, f"{model_name}_best.zip"),
        os.path.join(model_dir, f"{model_name}_final.zip"),
        os.path.join(model_dir, f"{model_name}.zip"),
    ):
        if os.path.exists(path):
            return path
    return None


def _build_env(cfg, seed_offset=0):
    action_distances = getattr(cfg, "rl_action_distances", (20.0, 40.0, 60.0))
    return DirectionDistanceEnv(
        initial_position=getattr(cfg, "rl_initial_position", (-320.34, -206.58, 128.0)),
        max_steps=int(getattr(cfg, "rl_planner_max_steps", 7)),
        action_distances=action_distances,
        n_directions=int(getattr(cfg, "rl_action_directions", 8)),
        area_limit=float(getattr(cfg, "rl_area_limit", 400.0)),
        revisit_penalty=float(getattr(cfg, "rl_planner_revisit_penalty", 0.25)),
        step_distance_penalty=float(getattr(cfg, "rl_planner_distance_penalty", 0.002)),
        pedestrian_reward=float(getattr(cfg, "rl_planner_pedestrian_reward", 1.0)),
        pedestrian_prob=float(getattr(cfg, "rl_planner_pedestrian_probability", 0.35)),
        seed=int(getattr(cfg, "random_seed", 1)) + seed_offset,
    )


def train_or_load_model(cfg):
    env = _build_env(cfg)

    model_dir = getattr(cfg, "rl_model_dir", "./trained_models")
    model_name = getattr(cfg, "rl_model_name", "trajectory_planner")
    force_retrain = bool(getattr(cfg, "rl_force_retrain", True))
    save_final = bool(getattr(cfg, "rl_save_final_model", True))
    total_timesteps = int(getattr(cfg, "rl_total_timesteps", 5000))
    eval_freq = int(getattr(cfg, "rl_eval_freq", 1000))
    n_eval_episodes = int(getattr(cfg, "rl_eval_episodes", 5))

    os.makedirs(model_dir, exist_ok=True)

    if not force_retrain:
        existing = _find_existing_model_path(model_dir, model_name)
        if existing is not None:
            model_cls = ALGORITHMS[getattr(cfg, "rl_algorithm", "PPO").upper()]
            return model_cls.load(existing, env=env), env, existing

    model = _build_model(
        getattr(cfg, "rl_algorithm", "PPO"),
        env,
        int(getattr(cfg, "random_seed", 1)),
        {
            "learning_rate": float(getattr(cfg, "rl_learning_rate", 3e-4)),
            "gamma": float(getattr(cfg, "rl_gamma", 0.99)),
        },
    )

    eval_env = _build_env(cfg, seed_offset=1000)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=max(1, eval_freq),
        n_eval_episodes=max(1, n_eval_episodes),
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    best_tmp = os.path.join(model_dir, "best_model.zip")
    named_best = os.path.join(model_dir, f"{model_name}_best.zip")
    final_prefix = os.path.join(model_dir, f"{model_name}_final")
    if os.path.exists(best_tmp):
        shutil.copyfile(best_tmp, named_best)
    if save_final:
        model.save(final_prefix)

    model_path = _find_existing_model_path(model_dir, model_name)
    if model_path is None:
        model_path = final_prefix + ".zip"
    return model, env, model_path


def build_rl_trajectory(cfg):
    """Train/load model and generate absolute positions from direction+distance actions."""
    model, env, model_file = train_or_load_model(cfg)

    obs, _ = env.reset(seed=int(getattr(cfg, "random_seed", 1)))
    path_list = [env.position.copy().tolist()]
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(action))
        path_list.append(env.position.copy().tolist())
        if done:
            break

    return path_list, model_file
