import json
import os
from collections import deque
from typing import Optional, Any, Dict, List, Tuple, SupportsFloat, Generator

import gymnasium as gym
import numpy as np
import torch
import torch as th
import torch.nn as nn
from gymnasium import spaces
from gymnasium.core import ObsType
from numpy import ndarray, dtype, floating
from numpy._typing import _32Bit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor
from torch.optim import Optimizer


class PPOWithImitationNav(PPO):
    """Класс для RL агента генератора"""

    def __init__(
            self,
            *args: Any,
            imitation_coef: float = 1.0,
            imitation_lr: float = 1e-4,
            **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.imitation_coef: float = imitation_coef
        self.imitation_lr: float = imitation_lr

        # Данные эксперта
        self.expert_obs: Optional[Dict[str, Tensor]] = None
        self.expert_actions: Optional[Tensor] = None

        # Оптимизатор имитации
        self.imitation_optimizer: Optional[Optimizer] = None

        if hasattr(self, "policy") and self.policy is not None:
            self._init_imitation_optimizer()

    def _init_imitation_optimizer(self) -> None:
        if isinstance(self.policy, BasePolicy):
            self.imitation_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self.imitation_lr
            )

    def set_expert_data(
            self,
            expert_obs: Dict[str, Tensor],
            expert_actions: Tensor
    ) -> None:
        """Загружает данные эксперта и переносит их на GPU/CPU устройства модели."""
        self.expert_obs = {k: v.to(self.device) for k, v in expert_obs.items()}
        self.expert_actions = expert_actions.to(self.device)

    def train(self, *args: Any, **kwargs: Any) -> None:
        super().train(*args, **kwargs)

        if self.expert_obs is not None and self.expert_actions is not None:
            self.policy.train()

            batch_size = 32
            idxs = torch.randint(0, self.expert_actions.shape[0], (batch_size,))
            obs_batch = {k: v[idxs] for k, v in self.expert_obs.items()}
            actions_batch = self.expert_actions[idxs]

            dist = self.policy.get_distribution(obs_batch)
            log_probs = dist.log_prob(actions_batch)
            imitation_loss = -log_probs.mean()

            if self.imitation_optimizer is not None:
                self.imitation_optimizer.zero_grad()
                (self.imitation_coef * imitation_loss).backward()
                self.imitation_optimizer.step()

            print(f"[i] Imitation loss: {imitation_loss.item():.4f}")

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._init_imitation_optimizer()


class PositionalEncodingNav(nn.Module):
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


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """извлекает признаки из наблюдений, комбинируя сверточные карты лабиринта и другие фичи, подготавливая
        их для трансформера и LSTM.
    """
    def __init__(
        self, observation_space: spaces.Dict,
            d_model: int=128,
            nhead: int=4,
            num_layers: int=2,
            memory_dim: int=128
    ):
        super().__init__(observation_space, features_dim=d_model)
        self.observation_space = observation_space
        self.obs_keys = list(observation_space.spaces.keys())

        flat_size = sum(
            int(np.prod(space.shape)) if isinstance(space, spaces.Box) else 1
            for space in observation_space.spaces.values()
        )

        self.input_linear: nn.Linear = nn.Linear(flat_size, d_model)
        self.pos_encoding: PositionalEncodingNav = PositionalEncodingNav(d_model)
        enc :nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.lstm: nn.LSTM = nn.LSTM(
            input_size=d_model, hidden_size=memory_dim, batch_first=True
        )
        self.output_linear: nn.Linear = nn.Linear(memory_dim, d_model)
        self.hidden_state = None

    def forward(self, obs_dict):
        batch_size: int  = next(iter(obs_dict.values())).shape[0]
        x_parts: List = []

        for k in self.obs_keys:
            if k not in obs_dict:
                continue
            v = obs_dict[k]
            space = self.observation_space.spaces[k]

            # float + NAN/INF guard (делаем сразу)
            v = v.float()
            v = th.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

            # flatten
            if isinstance(space, spaces.Discrete):
                v = v.view(-1, 1)
            else:
                v = v.view(batch_size, -1)

            x_parts.append(v)

        if not x_parts:
            return th.zeros(
                batch_size,
                self.output_linear.out_features,
                device=next(self.parameters()).device,
            )

        x: th.Tensor = th.cat(x_parts, dim=1)
        x = self.input_linear(x).unsqueeze(1)  # (B, 1, d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        # LSTM память
        if self.hidden_state is None or self.hidden_state[0].size(1) != batch_size:
            h0 = th.zeros(1, batch_size, self.lstm.hidden_size, device=x.device)
            c0 = th.zeros(1, batch_size, self.lstm.hidden_size, device=x.device)
            self.hidden_state = (h0, c0)

        x, (h, c) = self.lstm(x, self.hidden_state)
        self.hidden_state = (h.detach(), c.detach())

        x = x[:, -1, :]  # (B, hidden)
        x = th.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # ещё одна страховка
        return self.output_linear(x)

    def reset_memory(self, done_mask: Optional[th.Tensor] = None):
        if self.hidden_state is None:
            return
        h, c = self.hidden_state
        if done_mask is None:
            self.hidden_state = None
        else:
            done_mask = done_mask.to(h.device).view(1, -1, 1)
            h = h * (~done_mask)
            c = c * (~done_mask)
            self.hidden_state = (h, c)


class CustomTransformerPolicy(ActorCriticPolicy):
    """Кастомная политика RL на основе трансформера для агента-строителя с поддержкой сброса памяти
    """
    def __init__(
        self,
        *args,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        memory_dim: int =128,  # заменяем memory_dim
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                memory_dim=memory_dim,
            ),
        )


class ResetMemoryCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Здесь можно ничего не делать, просто вернуть True чтобы продолжить
        return True

    def _on_rollout_start(self) -> None:
        # Вызывается в начале rollout (нового сбора данных)
        # Можно попробовать сбросить hidden state здесь, если есть доступ к политике
        if hasattr(self.model.policy.features_extractor, "reset_memory"):
            self.model.policy.features_extractor.reset_memory()

class MazeEnv(gym.Env):
    """Среда RL для навигации по лабиринту с целью нахождения маршрута для выхода
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    original_maze: np.ndarray
    cursor_x: int
    cursor_y: int
    rows: int
    cols: int
    size: int
    done: bool
    inventory: set[int]
    rating: float
    count_last_wall_hit: int
    last_wall_hit: Any

    max_health: int
    health: int

    coins: int
    max_steps: int
    step_count: int

    verbose: bool
    directions: Dict[int, Tuple[int, int]]

    KEY_ID: int
    EXIT_ID: int
    key_pos: tuple[int, int] | None
    exit_pos: tuple[int, int] | None
    path_to_key: bool
    path_to_exit: bool

    layout: np.ndarray
    heatmap: np.ndarray

    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Dict

    def __init__(self, maze, max_health=300, verbose=False):
        super().__init__()
        self.original_maze = maze.copy()
        self.cursor_x, self.cursor_y = 1, 1
        self.rows, self.cols = maze.shape
        self.size = self.rows
        self.done = False
        self.inventory = set()
        self.rating = 0
        self.count_last_wall_hit = 0
        self.last_wall_hit = None

        self.max_health = max_health
        self.health = self.max_health  # вначале они равны

        self.coins = 0
        self.max_steps = self.size * self.size * 10
        self.step_count = 0

        self.verbose = verbose
        self.directions = {
            0: (-1, 0),  # вверх
            1: (1, 0),  # вниз
            2: (0, -1),  # влево
            3: (0, 1),  # вправо
        }

        self.KEY_ID = 2
        self.EXIT_ID = 7
        self.key_pos = None  # Вначале агенту они не известны
        self.exit_pos = None
        self.path_to_key = False
        self.path_to_exit = False

        # Лабиринт (0 пусто, 1 стена, 2 ключ, 4 ловушка, 5 костёр, 7 выход)
        self.layout = np.full(
            (self.size, self.size), -1, dtype=np.int32
        )  # с учетом видимости/пройденного агентом  -1 незивестно
        self.heatmap = np.full(
            (self.size, self.size), np.nan, dtype=np.float32
        )  # тепловая карта наград

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "maze": spaces.Box(
                    low=-100.0,
                    high=1000,
                    shape=(2, self.size, self.size),
                    dtype=np.float32,
                ),
                "coins": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
                "health": spaces.Box(low=0, high=300, shape=(1,), dtype=np.float32),
                "inventory": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "has_key": spaces.Box(
                    low=0, high=1, shape=(2,), dtype=np.float32
                ),  # добавили
                "rating": spaces.Box(
                    low=-500.0, high=500.0, shape=(1,), dtype=np.float32
                ),
                "position": spaces.Box(
                    low=0, high=self.size, shape=(2,), dtype=np.float32
                ),
            }
        )

        self.reset()

    def reset(self, *, seed=None, options=None, preserve_eval=False):
        super().reset(seed=seed)

        self.done = False
        if self.verbose:
            print("[DEBUG RESET]:")

        self.cursor_x, self.cursor_y = 1, 1

        self.layout = np.full(
            (self.size, self.size), -1, dtype=np.int32
        )  # с учетом видимости/пройденного агентом  -1 незивестно
        self.heatmap = np.full(
            (self.size, self.size), np.nan, dtype=np.float32
        )  # тепловая карта наград

        # Установим те 9 клеток что агент видит изначально (копируя из исходного лабиринта)

        for y in range(0, 3):
            for x in range(0, 3):
                self.layout[y, x] = self.original_maze[y, x]
                if self.layout[y, x] == 1:
                    self.heatmap[y, x] = -100
                elif self.layout[y, x] == 0 or self.layout[y, x] == 5:
                    self.heatmap[y, x] = 10
                elif self.layout[y, x] == 2:
                    self.heatmap[y, x] = 50
                elif self.layout[y, x] == 7:
                    self.heatmap[y, x] = 10

        self.heatmap[1, 1] = -1  # награда для первой клетки (сразу)

        self.health = self.max_health
        self.coins = 0
        self.inventory = set()
        self.path_to_key = False
        self.path_to_exit = False
        self.count_last_wall_hit = 0
        self.step_count = 0
        self.rating = 0
        self.last_wall_hit = None

        if self.verbose:
            obs = self._get_obs()
            print("[DEBUG] obs['position'] =", obs["position"])
            print("[DEBUG] obs['maze'][:,1,1] =", obs["maze"][:, 1, 1])

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        reward, done = self._nav_step(action)

        terminated = False
        truncated = False

        if self.verbose:
            print(f"[STEP DEBUG] ТЕкущее {done}")

        if self.health <= 0:
            terminated = True
            reward = -100
            self.log_episode_summary(result="death")
        elif done:
            terminated = True
            self.log_episode_summary(result="exit")
        elif self.step_count >= self.max_steps:
            truncated = True
            reward = -100
            self.log_episode_summary(result="OVER9000 steps")

        info = {}
        return self._get_obs(), reward, terminated, truncated, info

    def _nav_step(self, action: int) -> tuple[int, bool] | tuple[Any, bool] | tuple[int | Any, bool]:
        """Метод передвижения агента навигатора по лабиринту
        """

        if self.verbose:
            print(f"[step begin] Текущее  действие:\n{action}")
            print(f"[step begin] Текущая тепловая карта:\n{self.heatmap}")
            print(f"[step begin] Текущая  карта:\n{self.layout}")
            print(f"[step begin] Текущий  шаг:\n{self.step_count}")
            print(f"[step begin] Текущий  рейтинг:\n{self.rating}")
            print(f"[step begin] Текущее здоровье:\n{self.health}")
            print(f"[step begin] Текущий курсор :\n{[self.cursor_y,self.cursor_x, ]}")
            if self.key_pos:
                print(f"[step begin] Позиция ключа известна:\n{self.key_pos}")
            if self.exit_pos:
                print(f"[step begin] Позиция выхода известна:\n{self.exit_pos}")

            if self.KEY_ID in self.inventory:
                print(f"[step begin] Ключ в инвентаре")

        py, px = self.cursor_y, self.cursor_x  # начальная позиция шага
        dy, dx = self.directions[int(action)]
        ny, nx = py + dy, px + dx

        # Мягкий возрат если агент пытается выйти не имея ключа в инвентаре
        if self.layout[ny, nx] == 7 and self.KEY_ID not in self.inventory:
            self.step_count += 1
            if self.verbose:
                print(f"[Шаг на выход без ключа]")
            return -1, False  # Мягкий возрат\
        elif (
            self.layout[ny, nx] == 7 and self.KEY_ID in self.inventory
        ):  # Если ключ есть - то выходим и закачиваем
            if self.verbose:
                print(f"[TRUE EXIT] Мы на выходе")
            if self.step_count >= (self.max_steps // 2):
                self.rating += 50
            else:
                self.rating += 200
            self.coins += 500

            # self.log_episode_summary(result="exit")

            return self.heatmap[ny, nx], True

        # Запрет идти внешнюю стену напрямую
        if self.layout[ny, nx] != 7 and (
            nx == 0 or ny == 0 or nx == self.size - 1 or ny == self.size - 1
        ):
            self.health -= 1
            self.step_count += 1
            if self.verbose:
                print(f"[Шаг во внешнюю стену]")
            return self.heatmap[ny, nx], False

        # Жёсткая проверка на "запрещённую" клетку стена
        if self.heatmap[ny, nx] <= -99.9:  # вместо == -100
            if self.last_wall_hit == (ny, nx):
                self.count_last_wall_hit += 1
            else:
                self.count_last_wall_hit = 0
            self.last_wall_hit = (ny, nx)
            self.health -= 1
            self.step_count += 1

            reward = self.heatmap[ny, nx] * self.count_last_wall_hit
            if self.verbose:
                print(f"[Шаг в стену]")
            return reward, False

        # Жёсткая проверка на "запрещённую" клетку пустая клетка
        if self.heatmap[ny, nx] == -1:
            self.health -= 1
            self.step_count += 1

            if self.verbose:
                print(f"[Шаг в пустую клетку]")
            return self.heatmap[ny, nx], False

        target = self.layout[ny, nx]

        if self.verbose:
            print(
                f"[DEBUG] cursor=({self.cursor_x},{self.cursor_y}), хочет пойти ny,nx=({ny},{nx}), layout={self.layout[ny,nx]}"
            )

        if target == 2:  # нашли ключ добавляем его в инвентарь

            self.inventory.add(self.KEY_ID)
            self.coins += 100
            self.rating += 50
            self.layout[ny, nx] = 0  # делаем клетку с ключем - пустой

        if target == 5 and self.health < self.max_health:  # нашли костре- лечимся
            self.health = self.max_health
            self.coins += 5
            self.rating += 5
            self.layout[ny, nx] = 0  # делаем клетку с костром - пустой

        self.cursor_x, self.cursor_y = nx, ny
        self.health -= 1  # ограничиваем движение по лабиринту кол-вом жизней

        reward = self.heatmap[ny, nx]
        self.rating += reward // 100
        # Обновляем layout с учетом видимость агента
        self._update_layout()

        # Пересчитываем тепловую карту
        self.compute_heatmap()

        self.step_count += 1
        if (self.health <= 0) or (self.step_count >= self.max_steps):
            self.rating -= 100
            reward = -100
            done = True

            return reward, done

        return reward, False

    def _update_layout(self):
        """Метод обновления карты (layout)
        """
        for y in range(self.cursor_y - 1, self.cursor_y + 2):
            for x in range(self.cursor_x - 1, self.cursor_x + 2):
                if (
                    self.layout[y, x] == -1
                ):  # Просто обновляем все клетки что None т.е те что еще не видели.
                    self.layout[y, x] = self.original_maze[y, x]

                    if self.layout[y, x] == 2:
                        self.key_pos = (y, x)
                    elif self.layout[y, x] == 7:
                        self.exit_pos = (y, x)

    def compute_heatmap(self):
        """Метод тепловой карты (heatmap)
        """

        # Уменьшаем награду за текущую клетку.

        self.heatmap[self.cursor_y, self.cursor_x] = -1

        mask_new = (self.layout != -1) & np.isnan(self.heatmap)

        # Для новых клеток создаём отдельные маски по типу
        walls = mask_new & (self.layout == 1)
        empty = mask_new & (self.layout == 0)
        camp = mask_new & (self.layout == 5)
        keys = mask_new & (self.layout == 2)
        exits = mask_new & (self.layout == 7)

        # Заполняем heatmap за один проход
        self.heatmap[walls] = -100
        if self.health < 50:
            self.heatmap[camp] = 50
        else:
            self.heatmap[camp] = 5

        self.heatmap[empty] = 10
        self.heatmap[keys] = 50
        self.heatmap[exits] = 10

        #  Прверяем на тупик
        open_cell = False
        mask = None

        for ny, nx in self._neighbors(self.cursor_y, self.cursor_x):
            if self.heatmap[ny, nx] > 0:
                open_cell = True  #  Прверяем если рядом есть клетки с наградой больше 0 то не тупик

        # если не тупик прокладываем аршрут к ближайшей неисследованной области
        if not open_cell:

            for target_step in self._find_exploration_target():
                mask = self.find_path(target_step)
                if mask is not None:
                    self.heatmap[mask] = np.maximum(self.heatmap[mask], 10)
                    break

            if mask is None:
                if (
                    self.KEY_ID not in self.inventory and self.key_pos is not None
                ):  # Т.е ключ увидели ноне взяли
                    mask_for_key = self.find_path(
                        self.key_pos
                    )  # если маршрут до ключа найден всем пустым клеткам до него ставим ценность 40

                    if mask_for_key is not None:
                        self.path_to_key = True
                        self.heatmap[mask_for_key] = 40

                elif self.KEY_ID in self.inventory and self.exit_pos is not None:
                    self.heatmap[exits] = 100
                    mask_for_exit = self.find_path(self.exit_pos)
                    if (
                        mask_for_exit is not None
                    ):  # если маршрут до выхода найден всем пустым клеткам до него ставим ценность 80
                        self.path_to_exit = True
                        self.heatmap[mask_for_exit] = 80

        # далее назначение наград в частных случаях: нет кллюча но он виден, и есть ключ и виден выход

        if not self.path_to_key:

            if (
                self.KEY_ID not in self.inventory and self.key_pos is not None
            ):  # Т.е ключ увидели ноне взяли
                mask_for_key = self.find_path(
                    self.key_pos
                )  # если маршрут до ключа найден всем пустым клеткам до него ставим ценность 40

                if mask_for_key is not None:
                    self.path_to_key = True
                    self.heatmap[mask_for_key] = 40

        if not self.path_to_exit:
            if (
                self.KEY_ID in self.inventory and self.exit_pos is None
            ):  # ключ есть но где выход пока не знаем
                mask_exit = (self.heatmap > 0) & np.isnan(self.heatmap)
                self.heatmap[mask_exit] = 30

            elif self.KEY_ID in self.inventory and self.exit_pos is not None:
                self.heatmap[exits] = 100
                mask_for_exit = self.find_path(self.exit_pos)
                if (
                    mask_for_exit is not None
                ):  # если маршрут до выхода найден всем пустым клеткам до него ставим ценность 80
                    self.path_to_exit = True
                    self.heatmap[mask_for_exit] = 80

    def _get_obs(self) -> dict[str, ndarray]:
        maze_obs : np.ndarray= np.stack([self.layout.astype(np.float32), self.heatmap], axis=0)

        return {
            "maze": maze_obs,
            "rating": np.array([self.rating], dtype=np.float32),
            "position": np.array([self.cursor_y, self.cursor_x], dtype=np.float32),
            "inventory": np.array(
                [int(self.KEY_ID in self.inventory)], dtype=np.float32
            ),
            "has_key": np.array(
                [
                    int(self.KEY_ID in self.inventory),
                    1 - int(self.KEY_ID in self.inventory),
                ],
                dtype=np.float32,
            ),
            "health": np.array([self.health], dtype=np.float32),
            "coins": np.array([self.coins], dtype=np.float32),
        }

    def _find_exploration_target(self) -> list[Any]:
        """Метод поиска и возврата клеток рядом с которыми есть что исследовать
        """
        candidates = []
        for y in range(self.size):
            for x in range(self.size):
                # только пустые клетки, которые агент уже видел
                if self.layout[y, x] == 0:
                    # проверяем, есть ли среди соседей nan (неизвестная область)
                    has_nan_neighbor = any(
                        0 <= ny < self.size
                        and 0 <= nx < self.size
                        and np.isnan(self.heatmap[ny, nx])
                        for ny, nx in self._neighbors(y, x)
                    )
                    if has_nan_neighbor:
                        # проверяем по original_maze, что рядом реально есть пустые клетки
                        has_passage = any(
                            self.size > ny >= 0 == self.original_maze[ny, nx]
                            and 0 <= nx < self.size
                            for ny, nx in self._neighbors(y, x)
                        )
                        if has_passage:
                            candidates.append((y, x))

        if not candidates:
            return []

        # сортируем по манхэттенскому расстоянию от текущей позиции
        cy, cx = self.cursor_y, self.cursor_x
        candidates.sort(key=lambda p: abs(p[0] - cy) + abs(p[1] - cx))

        return candidates

        # выбираем ближайшую клетку к курсору (манхэттен)
        cy, cx = self.cursor_y, self.cursor_x
        target = min(candidates, key=lambda pos: abs(pos[0] - cy) + abs(pos[1] - cx))
        return target

    def find_path(self, target_pos: Tuple[int, int]):
        """
        Поиск кратчайшего пути до target_pos (y, x) по пустым клеткам.
        Возвращает булеву маску пути или None, если путь не найден.
        """
        start = (self.cursor_y, self.cursor_x)
        target = target_pos

        # Если уже стоим на целевой клетке
        if start == target:
            mask = np.zeros_like(self.layout, dtype=bool)
            mask[start] = True
            return mask

        visited = np.zeros_like(self.layout, dtype=bool)
        queue = deque([start])
        visited[start] = True

        # Для восстановления пути
        parent = {start: None}

        # 4 направления движения
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def is_walkable(y: int, x: int) -> bool:
            """Разрешённые клетки: пустые, лагерь, ключ, выход"""
            return self.layout[y, x] in (0, 2, 5, 7)

        while queue:
            y, x = queue.popleft()

            for dy, dx in directions:
                ny, nx = y + dy, x + dx

                # Проверка границ
                if not (
                    0 <= ny < self.layout.shape[0] and 0 <= nx < self.layout.shape[1]
                ):
                    continue

                # Пропускаем непроходимые клетки
                if not is_walkable(ny, nx):
                    continue

                # Пропускаем уже посещённые
                if visited[ny, nx]:
                    continue

                visited[ny, nx] = True
                parent[(ny, nx)] = (y, x)
                queue.append((ny, nx))

                if (ny, nx) == target:
                    # Восстанавливаем путь и превращаем его в маску
                    mask = np.zeros_like(self.layout, dtype=bool)
                    cur = (ny, nx)
                    while cur is not None:
                        mask[cur] = True
                        cur = parent[cur]
                    return mask

        return None

    def log_episode_summary(
        self, result="exit", log_file="saved_mazes/episode_summary.json"
    ):
        """
        Метод логирования результатов 
        """
        # Формируем данные
        summary = {
            "steps": self.step_count,
            "rating": self.rating,
            "health": self.health,
            "coins": self.coins,
            "result": result,
        }

        # Если файл уже есть — подгружаем старые записи
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logs = []  # Если файл битый
        else:
            logs = []

        # Добавляем новую запись
        logs.append(summary)

        # Сохраняем обратно
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"[LOG] Записан итог эпизода: {summary}")

    def _neighbors(self, y: int, x: int) -> Generator[tuple[int | Any, int | Any], Any, None]:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (dy, dx)
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.size and 0 <= nx < self.size:
                yield ny, nx  # возвращаем (y,x)


class FullMapExpert:
    """
    Эксперт, который видит весь лабиринт.
    Оптимальный план: старт -> ключ -> выход
    """

    def __init__(self, env):
        self.env = env
        self.path = None
        self.step_idx = 0
        self.phase = "to_key"  # 'to_key' или 'to_exit'

    def reset(self):
        self.path = None
        self.step_idx = 0
        self.phase = "to_key"

    def _find_path(self, start: tuple[int, int], target: tuple[int, int]) -> list[Any] | None:
        """Метод BFS для поиска кратчайшего пути"""
        visited = np.zeros_like(self.env.layout, dtype=bool)
        parent = {}
        queue = deque([start])
        visited[start] = True
        parent[start] = None

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            y, x = queue.popleft()
            if (y, x) == target:
                # восстановим путь
                path = []
                cur = (y, x)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path

            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.env.size and 0 <= nx < self.env.size:
                    if (
                        self.env.original_maze[ny, nx] in (0, 2, 5, 7)
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        parent[(ny, nx)] = (y, x)
                        queue.append((ny, nx))
        return None

    def plan_path(self) -> None:
        """Метод построения пути"""
        start = (self.env.cursor_y, self.env.cursor_x)
        key_pos = self.env.key_pos or tuple(np.argwhere(self.env.original_maze == 2)[0])
        exit_pos = self.env.exit_pos or tuple(
            np.argwhere(self.env.original_maze == 7)[0]
        )
        if self.phase == "to_key":
            self.path = self._find_path(start, key_pos)
        elif self.phase == "to_exit":
            self.path = self._find_path(start, exit_pos)
        self.step_idx = 0

    def get_action(self)-> int:
        """Метод возврата действия"""
        if self.path is None or self.step_idx >= len(self.path):
            self.plan_path()

        if self.path is None:
            return np.random.randint(0, 4)  # fallback если путь не найден

        cur_pos = (self.env.cursor_y, self.env.cursor_x)
        next_pos = self.path[self.step_idx]
        self.step_idx += 1

        dy = next_pos[0] - cur_pos[0]
        dx = next_pos[1] - cur_pos[1]
        for action, (ady, adx) in self.env.directions.items():
            if (dy, dx) == (ady, adx):
                # если мы дошли до ключа или выхода, меняем фазу
                if self.phase == "to_key" and self.env.layout[next_pos] == 2:
                    self.phase = "to_exit"
                    self.path = None
                    self.step_idx = 0
                return action
        return np.random.randint(0, 4)


class PartialMapExpert:
    """
    Эксперт, который видит только видимую часть лабиринта (layout).
    Логика: как только увидел ключ, строит маршрут к нему, затем к выходу.
    """

    def __init__(self, env: MazeEnv):
        self.env = env
        self.path = None
        self.step_idx = 0
        self.phase = "explore"  # 'explore', 'to_key', 'to_exit'

    def reset(self):
        self.path = None
        self.step_idx = 0
        self.phase = "explore"

    def _find_path(self, start: int, target: int):
        """Метод  gjbcrf пути"""
        visited = np.zeros_like(self.env.layout, dtype=bool)
        parent: Dict[int, Tuple[int, int]] = {}
        queue = deque([start])
        visited[start] = True
        parent[start] = None

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            y, x = queue.popleft()
            if (y, x) == target:
                path = []
                cur = (y, x)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path

            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.env.size[0] and 0 <= nx < self.env.size[1]:
                    if self.env.layout[ny, nx] in (0, 2, 5, 7) and not visited[ny, nx]:
                        visited[ny, nx] = True
                        parent[(ny, nx)] = (y, x)
                        queue.append((ny, nx))
        return None

    def plan_path(self):
        """Метод построения пути"""
        start = (self.env.cursor_y, self.env.cursor_x)
        if self.phase == "to_key" and self.env.key_pos is not None:
            target = self.env.key_pos
        elif self.phase == "to_exit" and self.env.exit_pos is not None:
            target = self.env.exit_pos
        else:
            self.path = None
            return
        self.path = self._find_path(start, target)
        self.step_idx = 0

    def get_action(self)-> int:
        """Метод возврата действия"""
        # Определяем фазу
        if self.phase == "explore" and self.env.key_pos is not None:
            self.phase = "to_key"
        if (
            self.phase == "to_key"
            and 2
            in self.env.layout[
                self.env.cursor_y, self.env.cursor_x : self.env.cursor_y + 1
            ]
        ):
            self.phase = "to_exit"

        if self.path is None or self.step_idx >= len(self.path):
            self.plan_path()

        if self.path is None:
            return np.random.randint(0, 4)  # fallback

        cur_pos = (self.env.cursor_y, self.env.cursor_x)
        next_pos = self.path[self.step_idx]
        self.step_idx += 1

        dy = next_pos[0] - cur_pos[0]
        dx = next_pos[1] - cur_pos[1]
        for action, (ady, adx) in self.env.directions.items():
            if (dy, dx) == (ady, adx):
                return action
        return np.random.randint(0, 4)


