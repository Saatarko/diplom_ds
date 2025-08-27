import csv
import json
import os
import random
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from gymnasium.core import ObsType
from numpy import ndarray, dtype, floating
from numpy._typing import _32Bit
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from train_agents.maze_env.maze_build_cell import MazeBuilderEnvDFSCell, PPOWithImitationCell, \
    CustomTransformerPolicyForBuilder
from train_agents.maze_env.maze_env import PPOWithImitationNav, CustomTransformerPolicy, MazeEnv


class PositionalEncodingMeta(nn.Module):
    """Класс добавляет позиционное кодирование к входным последовательностям для передачи информации о порядке элементов
        в трансформере.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(
            th.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class MetaAgentFeatureExtractor(BaseFeaturesExtractor):
    """извлекает признаки из наблюдений, комбинируя сверточные карты лабиринта и другие фичи, подготавливая
        их для трансформера и LSTM.
       """

    def __init__(
                self,
                observation_space: spaces.Dict,
                d_model: int = 64,
                nhead: int = 4,
                num_layers: int = 2,
                memory_dim: int = 64
        ):
        super().__init__(observation_space, features_dim=d_model)
        self.obs_keys = list(observation_space.spaces.keys())

        # суммируем размер всех Box
        flat_size = sum(
            int(np.prod(space.shape)) for space in observation_space.spaces.values()
        )

        self.input_linear: nn.Linear = nn.Linear(flat_size, d_model)
        self.pos_encoding: PositionalEncodingMeta = PositionalEncodingMeta(d_model)
        enc: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.lstm: nn.LSTM = nn.LSTM(
            input_size=d_model, hidden_size=memory_dim, batch_first=True
        )
        self.output_linear: nn.Linear = nn.Linear(memory_dim, d_model)
        self.hidden_state = None

    def forward(self, obs_dict):
        # собираем и конкатенируем значения из словаря
        x_parts = [obs_dict[k].float() for k in self.obs_keys]
        x: th.Tensor = th.cat(x_parts, dim=1).unsqueeze(1)  # (batch, seq_len=1, features)

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


def generate_and_infer_best_maze(generator_model_path: str,
                                 navigator_model_path: str,
                                 size: object = 15,
                                 maze_dir: str = "result/mazes",
                                 log_dir: str = "result/logs",
                                 num_valid: int = 30,
                                 max_gen_steps: int = None,
                                 nav_runs: int = 30,
                                 max_nav_steps: int = 1000) -> tuple[Any, list[Any] | None]:
    """
    Генерация лабиринтов + выбор лучшего + инференс навигатора.
    """

    os.makedirs(maze_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # === Шаг 1. Генератор ===
    print(f"[INFO] Загружаем модель генератора: {generator_model_path}")
    generator_env = MazeBuilderEnvDFSCell(size=size, verbose=0, use_stub_eval=True)
    generator_env = ActionMasker(generator_env, lambda env: env.get_action_mask())
    generator_model = PPOWithImitationCell.load(
        generator_model_path,
        custom_objects={"policy_class": CustomTransformerPolicyForBuilder}
    )
    generator_model.set_env(generator_env)

    valid_mazes = []
    attempt_count = 0

    while len(valid_mazes) < num_valid:
        attempt_count += 1
        obs, _ = generator_env.reset()
        done, step_count = False, 0
        limit = max_gen_steps or (generator_env.size * generator_env.size * 2)

        while not done and step_count < limit:
            mask = generator_env.get_action_mask()
            action, _ = generator_model.predict(obs, action_masks=mask, deterministic=False)
            if not mask[action]:
                allowed_actions = np.where(mask)[0]
                action = int(allowed_actions[0])
            obs, reward, done, _, _ = generator_env.step(action)
            step_count += 1

        maze = generator_env.unwrapped.layout.copy()
        meta = generator_env.unwrapped.result_maze.copy()
        meta["attempt"] = attempt_count

        if meta.get("success") and meta.get("has_key") and meta.get("has_exit") and meta.get("turns", 0) > 3:
            valid_mazes.append((maze, meta))
            print(f"[INFO] Валидный лабиринт {len(valid_mazes)}/{num_valid}")
        else:
            print(f"[INFO] Лабиринт не прошёл фильтр (попытка {attempt_count})")

    # === Шаг 2. Выбор лучшего лабиринта ===
    best_maze, best_meta = max(valid_mazes, key=lambda m: m[1].get("rating", 0))
    print(f"[INFO] Выбран лабиринт с рейтингом {best_meta.get('rating')}")

    # === Шаг 3. Навигатор ===
    print(f"[INFO] Загружаем модель навигатора: {navigator_model_path}")
    nav_model = PPOWithImitationNav.load(
        navigator_model_path,
        custom_objects={"policy_class": CustomTransformerPolicy}
    )

    best_run_logs = None
    best_steps = float("inf")

    for run_id in range(nav_runs):
        base_env = MazeEnv(best_maze.copy(), verbose=0)
        obs, _ = base_env.reset()
        logs = []

        for step in range(max_nav_steps):
            action, _ = nav_model.predict(obs, deterministic=False)
            action = int(action)
            obs, reward, terminated, truncated, info = base_env.step(action)
            done = terminated or truncated

            x, y = base_env.cursor_x, base_env.cursor_y
            tile = base_env.layout[y, x]
            health = base_env.health
            has_key = int(base_env.KEY_ID in base_env.inventory)
            logs.append([step, x, y, action, tile, reward, health, has_key])

            if done:
                break

        if len(logs) < best_steps:
            best_steps = len(logs)
            best_run_logs = logs
            print(f"[INFO] Новый лучший результат: {best_steps} шагов (run {run_id+1})")

    # === Шаг 4. Сохранение ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    maze_fname = f"best_maze_{timestamp}.csv"
    log_fname = f"best_run_{timestamp}.csv"

    maze_path = os.path.join(maze_dir, maze_fname)
    log_path = os.path.join(log_dir, log_fname)

    np.savetxt(maze_path, np.array(best_maze, dtype=np.int32), fmt="%d", delimiter=",")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x", "y", "action", "tile", "reward", "health", "has_key"])
        writer.writerows(best_run_logs)

    print(f"[INFO] Лучший лабиринт сохранён в {maze_path}")
    print(f"[INFO] Лог лучшего прохождения сохранён в {log_path}")

    # === Шаг 5. Возврат данных ===
    return best_maze, best_run_logs


class MetaCustomPolicy(ActorCriticPolicy):
    """Кастомная политика RL на основе трансформера для мета агента
        """
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

    def __init__(
        self, state_file="meta_state.json", history_file="history.jsonl", verbose=0
    ):
        super(MetaMazeEnv, self).__init__()
        self.verbose = verbose
        self.state_file = state_file
        self.history_file = history_file
        self.phase = "choice"
        self.max_iters = 1000


        # Действия
        self.action_space = spaces.Discrete(3)

        # Наблюдение: [генератор_score, навигатор_score, сложность]
        self.observation_space = spaces.Dict(
            {
                "gen_score": spaces.Box(
                    low=-1000, high=1000, shape=(1,), dtype=np.float32
                ),
                "nav_score": spaces.Box(
                    low=-1000, high=1000, shape=(1,), dtype=np.float32
                ),
                "difficulty": spaces.Box(low=1, high=100, shape=(1,), dtype=np.float32),
            }
        )

        self.state = self._load_or_init_state()

    # ---------------- JSON Чекпоинт ---------------- #
    def _load_or_init_state(self) -> dict[str, int | str] | Any:
        """
        Метод загрузки состояния из файла
        """
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {
            "generator_score": 0,
            "navigator_score": 0,
            "difficulty": 7,
            "phase": "idle",
            "iteration": 0,
            "current_step": 0,
        }

    def _save_state(self):
        """
        Метод сохранения состояния
        """
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def _log_history(self, record: Any):
        """
        Метод сохранения лога
        """
        with open(self.history_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ---------------- Основные методы gym ---------------- #
    def reset(self, *, seed: object = None, options: object = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.state = self._load_or_init_state()

        if self.state["phase"] == "idle":
            self.phase = "choice"
            self._inference(self.state["difficulty"])
        else:
            self.phase = "train"  # продолжаем обучение

        return self._get_obs(), {}

    def step(self, action: int):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        if self.phase == "choice":
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
                "phase": self.state["phase"],
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

    def _choice_step(self, action: int) -> int | str | float:
        """
        Метод выбора действия
        """
        try:
            if action == 0:  # обучение генератора
                reward = -self.state["generator_score"]
                self.state["phase"] = "train_generator"
                self.phase = "train"

            elif action == 1:  # обучение навигатора
                if self.state["generator_score"] < 0:
                    reward = self.state["generator_score"]  # штраф
                else:
                    reward = -self.state["navigator_score"]
                    self.state["phase"] = "train_navigator"
                    self.phase = "train"

            elif action == 2:  # усложнение лабиринта
                reward = self._increase_difficulty()
            else:
                raise ValueError("Неверное действие!")

        except Exception:
            reward = -200
            self.state["phase"] = "exception"

        return reward

    def _get_obs(self) -> dict[str, ndarray[Any, dtype[floating[_32Bit]]]]:
        """Возвращает наблюдение в виде словаря для SB3 и логирования"""
        return {
            "gen_score": np.array([self.state["generator_score"]], dtype=np.float32),
            "nav_score": np.array([self.state["navigator_score"]], dtype=np.float32),
            "difficulty": np.array([self.state["difficulty"]], dtype=np.float32),
        }

    # ---------------- Заглушки ---------------- #
    def _inference(self, size: object) :
        """
        Метод проведения инференса генератора и навигатора для обределения их качества
        """

        maze, log = generate_and_infer_best_maze(
            "generator_agent", "navigator_agent"
        )

        if maze == "generator_good":
            self.state["generator_score"] += 10
        elif maze == "generator_bad":
            self.state["generator_score"] -= 10
        elif maze == "generator_exception":
            self.state["generator_score"] -= 20

        if log == "navigator_good":
            self.state["navigator_score"] += 10
        elif log == "navigator_bad":
            self.state["navigator_score"] -= 10
        elif log == "navigator_exception":
            self.state["navigator_score"] -= 20

    def _train_agent(self, agent_type: str):
        """
        Метод обучения агентов
        """
        env = MazeBuilderEnvDFSCell(size=self.state["difficulty"], verbose=0, use_stub_eval=True)
        if agent_type == "generator":
            model = PPOWithImitationCell.load(
                "generator_agent",
                custom_objects={"policy_class": CustomTransformerPolicyForBuilder},
                device="cuda",
            )
            model.set_env(env)
        else:
            model = PPOWithImitationNav.load(
                "navigator_agent", custom_objects={"policy_class": CustomTransformerPolicy})
            model.set_env(env)
        for i in range(self.state["iteration"], self.max_iters):
            model.learn(total_timesteps=10240, progress_bar=True)

            if agent_type == "generator":
                model.save("generator_agent")
            else:
                model.save("navigator_agent")
            self.state["current_step"] += 1
            self.state["iteration"] += 1
            self._save_state()


        # после обучения сброс
        self.state["generator_score"] = 0
        self.state["navigator_score"] = 0
        self.state["phase"] = "idle"
        self.state["iteration"] = 0

    def _increase_difficulty(self):
        """
        Метод увеличения сложности (увеличения размера лабиринта)
        """
        if self.state["generator_score"] > 0 and self.state["navigator_score"] > 0:
            self.state["difficulty"] += 8
            return +0.5
        else:
            return -100
