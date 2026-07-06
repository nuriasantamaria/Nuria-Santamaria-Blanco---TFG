import numpy as np
import gymnasium as gym


class HistoryWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, history_length: int = 4):
        super().__init__(env)
        assert history_length >= 1, "history_length debe ser >= 1"

        self.history_length = history_length
        self._obs_dim = int(np.prod(env.observation_space.shape))
        self._act_dim = int(np.prod(env.action_space.shape))

        new_dim = (self._obs_dim + self._act_dim) * history_length

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(new_dim,), dtype=np.float32
        )

        # más reciente en índice 0
        self._obs_history: list[np.ndarray] = []
        self._act_history: list[np.ndarray] = []

    def _stacked_obs(self) -> np.ndarray:
        obs_flat = np.concatenate(self._obs_history)
        act_flat = np.concatenate(self._act_history)
        return np.concatenate([obs_flat, act_flat]).astype(np.float32)

    def _reset_obs_buffer(self, first_obs: np.ndarray):
        self._obs_history = [
            first_obs.astype(np.float32).copy()
            for _ in range(self.history_length)
        ]

    def _reset_act_buffer(self):
        self._act_history = [
            np.zeros(self._act_dim, dtype=np.float32)
            for _ in range(self.history_length)
        ]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.asarray(obs, dtype=np.float32).reshape(self._obs_dim)
        self._reset_obs_buffer(obs)
        self._reset_act_buffer()
        return self._stacked_obs(), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self._act_dim)

        self._act_history.insert(0, action.copy())
        self._act_history = self._act_history[: self.history_length]

        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float32).reshape(self._obs_dim)

        self._obs_history.insert(0, obs.copy())
        self._obs_history = self._obs_history[: self.history_length]

        return self._stacked_obs(), reward, terminated, truncated, info