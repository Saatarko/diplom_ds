import gymnasium as gym
from gymnasium import spaces
import random
import json
import os
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class MetaAgentFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, d_model=64, nhead=4, num_layers=2, memory_dim=64):
        super().__init__(observation_space, features_dim=d_model)
        self.obs_keys = list(observation_space.spaces.keys())

        # суммируем размер всех Box
        flat_size = sum(int(np.prod(space.shape)) for space in observation_space.spaces.values())

        self.input_linear = nn.Linear(flat_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=memory_dim, batch_first=True)
        self.output_linear = nn.Linear(memory_dim, d_model)
        self.hidden_state = None

    def forward(self, obs_dict):
        # собираем и конкатенируем значения из словаря
        x_parts = [obs_dict[k].float() for k in self.obs_keys]
        x = th.cat(x_parts, dim=1).unsqueeze(1)  # (batch, seq_len=1, features)

        x = self.input_linear(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        # LSTM память
        batch_size = x.size(0)
        if self.hidden_state is None or self.hidden_state[0].size(1) != batch_size:
            h0 = th.zeros(1, batch_size, self.lstm.hidden_size, device=x.device)
            c0 = th.zeros(1, batch_size, self.lstm.hidden_size, device=x.device)
            self.hidden_state = (h0, c0)

        x, (h, c) = self.lstm(x, self.hidden_state)
        self.hidden_state = (h.detach(), c.detach())

        x = x[:, -1, :]
        return self.output_linear(x)

    def reset_memory(self):
        self.hidden_state = None


# 🚀 Политика для мета-агента


class MetaCustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, d_model=64, memory_dim=64, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=MetaAgentFeatureExtractor,
            features_extractor_kwargs=dict(d_model=d_model, memory_dim=memory_dim)
        )

class MetaMazeEnv(gym.Env):
    """
    Meta-агент управляет генератором и навигатором:
    Действия:
        0 - обучать генератора
        1 - обучать навигатора
        2 - усложнить лабиринт (увеличить размер)
    """

     # глобальный лимит итераций для тренировки

    def __init__(self, state_file="meta_state.json", history_file="history.jsonl", verbose=0):
        super(MetaMazeEnv, self).__init__()
        self.verbose = verbose
        self.state_file = state_file
        self.history_file = history_file
        self.phase = 'choice'
        self.max_iters = 10

        # Действия
        self.action_space = spaces.Discrete(3)

        # Наблюдение: [генератор_score, навигатор_score, сложность]
        self.observation_space = spaces.Dict({
            "gen_score": spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32),
            "nav_score": spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32),
            "difficulty": spaces.Box(low=1, high=100, shape=(1,), dtype=np.float32)
        })

        self.state = self._load_or_init_state()

    # ---------------- JSON Чекпоинт ---------------- #
    def _load_or_init_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {
            "generator_score": 0,
            "navigator_score": 0,
            "difficulty": 7,
            "phase": "idle",
            "iteration": 0,
            "current_step": 0
        }

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def _log_history(self, record):
        with open(self.history_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ---------------- Основные методы gym ---------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._load_or_init_state()
        
        if self.state["phase"] == "idle": 
            self.phase = 'choice'
            self._inference(self.state["difficulty"])
        else:
            self.phase = 'train'  # продолжаем обучение

        return self._get_obs(), {}
    

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        if self.phase == 'choice':
            reward = self._choice_step(action)

            # Получаем наблюдение и сериализуем
            obs_dict = self._get_obs()
            obs_serializable = {k: v.tolist() for k, v in obs_dict.items()}

            # запись в историю (JSONL)
            record = {
                "iteration": self.state["iteration"],
                "step": self.state["current_step"],
                "action": int(action),
                "obs": obs_serializable,
                "reward": float(reward),
                "phase": self.state["phase"]
            }
            self._log_history(record)

            terminated = True
            
        else:  # продолжаем обучение
            if self.state["phase"] == "train_generator":
                self._train_agent("generator")
            elif self.state["phase"] == "train_navigator":
                self._train_agent("navigator")

        # Сохраняем в JSON (чекпоинт)
        self._save_state()
        return self._get_obs(), reward, terminated, truncated, info
    

    def _choice_step(self, action):
        try:
            if action == 0:  # обучение генератора
                reward = -self.state["generator_score"]
                self.state["phase"] = "train_generator"
                self.phase = 'train'

            elif action == 1:  # обучение навигатора
                if self.state["generator_score"] < 0:
                    reward = self.state["generator_score"]  # штраф
                else:
                    reward = -self.state["navigator_score"]
                    self.state["phase"] = "train_navigator"
                    self.phase = 'train'

            elif action == 2:  # усложнение лабиринта
                reward = self._increase_difficulty()
            else:
                raise ValueError("Неверное действие!")

        except Exception as e:
            reward = -200
            self.state["phase"] = "exception"

        return reward
    

    def _get_obs(self):
        """Возвращает наблюдение в виде словаря для SB3 и логирования"""
        return {
            "gen_score": np.array([self.state["generator_score"]], dtype=np.float32),
            "nav_score": np.array([self.state["navigator_score"]], dtype=np.float32),
            "difficulty": np.array([self.state["difficulty"]], dtype=np.float32),
        }

    
     # ---------------- Заглушки ---------------- #
    def _inference(self, size):
        generator_result = random.choice(["generator_good", "generator_bad", "generator_exception"])
        navigator_result = random.choice(["navigator_good", "navigator_bad", "navigator_exception"])

        if generator_result == "generator_good":
            self.state["generator_score"] += 10
        elif generator_result == "generator_bad":
            self.state["generator_score"] -= 10
        elif generator_result == "generator_exception":
            self.state["generator_score"] -= 20

        if navigator_result == "navigator_good":
            self.state["navigator_score"] += 10
        elif navigator_result == "navigator_bad":
            self.state["navigator_score"] -= 10
        elif navigator_result == "navigator_exception":
            self.state["navigator_score"] -= 20

    def _train_agent(self, agent_type: str):
        
        if self.state["iteration"] ==0:
            size = self.state["difficulty"] 
        for i in range(self.state["iteration"], self.max_iters):
            self.state["current_step"] += 1
            self.state["iteration"] += 1
            self._save_state()
            # тут будет сохранение модели

        # после обучения сброс
        self.state["generator_score"] = 0
        self.state["navigator_score"] = 0
        self.state["phase"] = "idle"
        self.state["iteration"] = 0

    def _increase_difficulty(self):
        if self.state["generator_score"] > 0 and self.state["navigator_score"] > 0:
            self.state["difficulty"] += 1
            return +0.5
        else:
            return -100