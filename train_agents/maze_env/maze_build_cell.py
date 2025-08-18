import os
import uuid
from datetime import datetime
from typing import Optional
import json
import pandas as pd
import numpy as np
import random
import gymnasium as gym
import torch
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
import numpy as np
import random
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import csv
import uuid
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import uuid
import random
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from scipy.spatial.distance import cityblock
from stable_baselines3.common.utils import obs_as_tensor
import random
import copy
import csv
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from maze_env.maze_env import MazeEnv, CustomTransformerPolicy, PositionalEncoding


class PPOWithImitationCell(MaskablePPO):
    def __init__(self, *args, imitation_coef=1.0, imitation_lr=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.imitation_coef = imitation_coef
        self.imitation_lr = imitation_lr
        self.expert_obs = None
        self.expert_actions = None
        self.imitation_optimizer = None

        if hasattr(self, 'policy') and self.policy is not None:
            self._init_imitation_optimizer()

    def _init_imitation_optimizer(self):
        self.imitation_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.imitation_lr)

    def set_expert_data(self, expert_obs: dict, expert_actions: torch.Tensor):
        self.expert_obs = {k: v.to(self.device) for k, v in expert_obs.items()}
        self.expert_actions = expert_actions.to(self.device)

    def train(self, *args, **kwargs):
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

            self.imitation_optimizer.zero_grad()
            (self.imitation_coef * imitation_loss).backward()
            self.imitation_optimizer.step()

            print(f"[i] Imitation loss: {imitation_loss.item():.4f}")

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_imitation_optimizer()



def shortest_path_info(maze, start, goal):
    rows, cols = maze.shape
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return float('inf'), 0  # цель вне границ

    dist = [[float('inf')] * cols for _ in range(rows)]
    path_count = [[0] * cols for _ in range(rows)]

    dist[start[0]][start[1]] = 0
    path_count[start[0]][start[1]] = 1

    queue = deque([start])
    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                # Разрешаем переход если:
                # - клетка проходима (0 или 5)
                # - или это именно цель
                if maze[nx][ny] in (0, 5) or (nx, ny) == goal:
                    if dist[nx][ny] == float('inf'):
                        dist[nx][ny] = dist[x][y] + 1
                        path_count[nx][ny] = path_count[x][y]
                        queue.append((nx, ny))
                    elif dist[nx][ny] == dist[x][y] + 1:
                        path_count[nx][ny] += path_count[x][y]

    if dist[goal[0]][goal[1]] == float('inf'):
        return float('inf'), 0
    return dist[goal[0]][goal[1]], path_count[goal[0]][goal[1]]



class CustomTransformerPolicyForBuilder(MaskableActorCriticPolicy):
    def __init__(
        self,
        *args,
        d_model=128,
        nhead=4,
        num_layers=2,
        memory_dim=128,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=TransformerFeatureExtractorForBuilder,
            features_extractor_kwargs=dict(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                memory_dim=memory_dim,
            ),
        )

    def reset_memory(self, done_mask: Optional[th.Tensor] = None):
        if hasattr(self, "features_extractor") and hasattr(self.features_extractor, "reset_memory"):
            self.features_extractor.reset_memory(done_mask)

class TransformerFeatureExtractorForBuilder(BaseFeaturesExtractor):
    def __init__(self, observation_space, d_model=128, nhead=4, num_layers=2, memory_dim=128):
        super().__init__(observation_space, features_dim=d_model)
        self.d_model = d_model


        # Сверточный блок для maze (2 канала)
        self.maze_conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        maze_size = observation_space["maze"].shape[-2:]
        dummy_maze = th.zeros(1, 2, *maze_size)
        maze_flat_dim = self.maze_conv(dummy_maze).shape[1]

        # Подсчёт размерности остальных признаков (phase, placed, rating, cursor)
        other_dim = 0
        for key, space in observation_space.spaces.items():
            if key == "maze":
                continue
            if isinstance(space, spaces.Box):
                other_dim += int(np.prod(space.shape))
            elif isinstance(space, spaces.Discrete):
                other_dim += 1  # На всякий случай

        print(f"maze_flat_dim={maze_flat_dim}, other_dim={other_dim}, total input dim={maze_flat_dim + other_dim}")

        self.other_proj = nn.Linear(maze_flat_dim + other_dim, d_model)

        self.pos_encoding = PositionalEncoding(d_model)  
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=memory_dim, batch_first=True)
        self.output = nn.Linear(memory_dim, d_model)
        

        self.hidden_state = None

    def forward(self, obs_dict) -> th.Tensor:
        batch_size = obs_dict["maze"].shape[0]

        maze = obs_dict["maze"].float()  # [B, 2, H, W]
        maze_feat = self.maze_conv(maze)  # [B, maze_flat_dim]

        # Приводим все остальные признаки к float и добавляем измерение, если нужно
        cursor = obs_dict["cursor"].float()
        if cursor.ndim == 1:
            cursor = cursor.unsqueeze(0)
        phase = obs_dict["phase"].float()
        if phase.ndim == 1:
            phase = phase.unsqueeze(0)
        rating = obs_dict["rating"].float()
        if rating.ndim == 1:
            rating = rating.unsqueeze(0)
        placed = obs_dict["placed"].float()
        if placed.ndim == 1:
            placed = placed.unsqueeze(0)

        other = th.cat([cursor, phase, rating, placed], dim=1)  # [B, other_dim]

        combined = th.cat([maze_feat, other], dim=1)  # [B, maze_flat_dim + other_dim]

        x = self.other_proj(combined).unsqueeze(1)  # [B, seq_len=1, d_model]
        x = self.pos_encoding(x)
        x = self.transformer(x)

        if self.hidden_state is None or self.hidden_state[0].size(1) != batch_size:
            h0 = th.zeros(1, batch_size, self.lstm.hidden_size, device=x.device)
            c0 = th.zeros(1, batch_size, self.lstm.hidden_size, device=x.device)
            self.hidden_state = (h0, c0)

        x, (h, c) = self.lstm(x, self.hidden_state)
        self.hidden_state = (h.detach(), c.detach())
        x = x.squeeze(1)

        return self.output(x)

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
        
class MazeBuilderEnvDFSCell(gym.Env):
    def __init__(self, size=7, verbose=0, use_stub_eval=True):
        super().__init__()
        self.size = size
        self.phase = "dig"
        self.rating = 0
        self.use_stub_eval = use_stub_eval
        self.verbose = verbose
        self.done = False
        self.entrance_pos = (1, 1)
        self.exit_pos = None
        self.stuck_mazes = {}
        self.current_maze_hash = None
        self.step_count =0
        self.result_maze ={}
        
        # Лабиринт (0 пусто, 1 стена, 2 ключ, 4 ловушка, 5 костёр, 7 выход)
        self.layout = np.ones((self.size, self.size), dtype=np.int32)
        self.heatmap = np.zeros((self.size, self.size), dtype=np.float32)

        # Позиция курсора
        self.cursor_x, self.cursor_y = 1, 1

        # Доступные элементы
        self.elements = {
            0: "empty",
            1: "wall",
            2: "key",
            4: "trap",
            5: "campfire",
            7: "exit",
        }

        self.allowed_elements = {
            4: 0,  # запретить ловушки
            5: 1,  # 1 костёр
        }
        self.placeable_elements = [
            e for e, max_count in self.allowed_elements.items()
            if max_count != 0 and e not in (0, 1)
        ]
        self.placed = {e: 0 for e in self.allowed_elements}

        self.directions = {
            0: (-1, 0),  # вверх
            1: (1, 0),   # вниз
            2: (0, -1),  # влево
            3: (0, 1),   # вправо
        }

        # Пространства действий и наблюдений
        self.action_space = spaces.Discrete(4 + self.size * self.size)
        self.observation_space = spaces.Dict({
            "maze": spaces.Box(low=-100.0, high=1000,
                               shape=(2, self.size, self.size), dtype=np.float32),
            "phase": spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
            "placed": spaces.Box(low=0, high=10,
                                 shape=(len(self.placeable_elements),), dtype=np.int32),
            "rating": spaces.Box(low=-500.0, high=500.0, shape=(1,), dtype=np.float32),
            "cursor": spaces.Box(low=0, high=self.size,
                                 shape=(2,), dtype=np.float32)
        })

        self.reset()


    # ===============================
    # ИНИЦИАЛИЗАЦИЯ
    # ===============================
    def reset(self, *, seed=None, options=None, preserve_eval=False):
        super().reset(seed=seed)
        
        self.done = False
        if self.verbose:
            print("[DEBUG RESET]:")
        # Весь лабиринт — стены
        self.layout = np.ones((self.size, self.size), dtype=np.int32)
        self.heatmap = np.zeros((self.size, self.size), dtype=np.float32)

        self.layout.fill(1)
        
        # Подготовка heatmap к фазе копания
        for y in range(self.size):
            for x in range(self.size):
                # Внешние стены
                if x == 0 or y == 0 or x == self.size - 1 or y == self.size - 1:
                    self.heatmap[y, x] = -100
                # Cтены примыкающие к внешним    
                elif x == 1 or y == 1 or x == self.size - 2 or y == self.size - 2:
                    self.heatmap[y, x] = random.randint(5, 7)
                    
                else:
                    self.heatmap[y, x] = random.randint(9, 15)
                    
        # Стартовая позиция
        self.cursor_x, self.cursor_y = 1, 1
        self.layout[self.cursor_y, self.cursor_x] = 0

        self.rating = 0
        self.stuck_mazes.clear()
        self.current_maze_hash = None
        step_count = 0
        self.result_maze = None
        # Начальная фаза
        self.phase = "dig"

        obs = self.get_obs()
        info = {}
        return obs, info


    def get_obs(self):
        maze_obs = np.stack([self.layout.astype(np.float32), self.heatmap], axis=0)

        phase_map = {"dig": 0, "place_key": 1, "place_exit": 2,
                     "place_other": 3, "eval": 4, "save_maze": 5}
        phase_vec = np.zeros(6, dtype=np.float32)
        phase_vec[phase_map[self.phase]] = 1.0

        return {
            "maze": maze_obs,
            "phase": phase_vec,
            "placed": np.array([self.placed[e] for e in self.placeable_elements], dtype=np.int32),
            "rating": np.array([self.rating], dtype=np.float32),
            "cursor": np.array([self.cursor_x, self.cursor_y], dtype=np.float32)
        }

    def get_action_mask(self):
        mask = np.zeros(4 + self.size * self.size, dtype=bool)
        if self.phase == "dig":
            mask[:4] = True
        else:
            mask[4:] = True
        return mask
    
    
    def _set_phase(self, phase_name):
        self.phase = phase_name
        
        if self.verbose:
            print(f'phase_name {phase_name}')

        if phase_name == "place_key":
            self.cursor_x, self.cursor_y = 0, 0
            self._init_heatmap_place_key()
        elif phase_name == "place_exit":
            self.cursor_x, self.cursor_y = 1, 1
            self._init_heatmap_place_exit()

        elif phase_name == "place_other":
            self.cursor_x, self.cursor_y = 1, 1
            self._init_heatmap_place_other()

    
    
    def step(self, action):

        if self.done:
            raise Exception("Episode has ended. Call reset() to start a new one.")

        reward = 0
        self.last_info = {}

        if self.verbose:
            print(f'фаза {self.phase}')

        if self.phase == "dig":
            reward, done = self._dig_step(action)
        elif self.phase == "place_key":
            reward, done = self._place_step(action, type_place="key")
        elif self.phase == "place_exit":
            reward, done = self._place_step(action, type_place="exit")
        elif self.phase == "place_other":
            reward, done = self._place_step(action, type_place="other")
        elif self.phase == "eval":
            reward, done = self._eval_phase()
        elif self.phase == "save_maze":
            reward, done = self._save_maze()
        else:
            raise ValueError(f"Unknown phase: {self.phase}")

        if done and self.phase == "save_maze":
            self.done = True

        terminated = done
        truncated = False

        info = getattr(self, "last_info", {})
        info["action_mask"] = np.expand_dims(self.get_action_mask(), axis=0)  # добавить batch dim

        return self.get_obs(), reward, terminated, truncated, info

    def _dig_step(self, action):
        reward = 0.0

        # --- Печать состояния лабиринта ---
        if self.step_count % 5 == 0 and self.verbose:
            print("[DEBUG] Текущий лабиринт:")
            print(self.layout)

        # Проверяем, есть ли куда копать вокруг курсора
        if not self.find_dig_positions():
            teleport_to = self.has_global_dig_positions()
            if teleport_to:
                # Телепорт только на реальную выкопанную клетку с доступным направлением
                self.cursor_y, self.cursor_x = teleport_to
                if self.verbose:
                    print(f"[DIG] Телепорт к развилке: {teleport_to}")
            else:
                if self.verbose:
                    print("[DIG] Нет куда копать. Переход к place_exit.")
                self._set_phase("place_exit")
                self.last_info = {"phase": "place", "reason": "all_are_dug"}
                self.rating += 10
                return 10, False

        py, px = self.cursor_y, self.cursor_x
        dy, dx = self.directions[int(action)]
        ny, nx = py + dy, px + dx

        # Проверка выхода за границы
        if not self._in_bounds(nx, ny):
            return -100.0, False

        # Запрет копать внешнюю стену напрямую
        if nx == 0 or ny == 0 or nx == self.size - 1 or ny == self.size - 1:
            return self.heatmap[ny, nx], False

        # Жёсткая проверка на "запрещённую" клетку
        if self.heatmap[ny, nx] <= -99.9:  # вместо == -100
            return self.heatmap[ny, nx], False

        target = self.layout[ny, nx]

        if target == 0:
            return -1.0, False  # уже выкопано — штраф

        reward = self.heatmap[ny, nx]

        # Копаем
        self.layout[ny, nx] = 0
        self.cursor_x, self.cursor_y = nx, ny

        # Пересчитываем тепловую карту
        self.compute_dig_heatmap(ny, nx, py, px)

        # Пересчитываем маску допустимых действий
        self._update_action_mask()

        if self.verbose:
            print(f'[end dig] Текущая тепловая карта:\n{self.heatmap}')

        self.step_count += 1

        return reward, False

    
    def compute_dig_heatmap(self, y, x, prev_y, prev_x):
        
        # 1. НА всякий случай названчаем отрицательную награду текущей клетке
        self.heatmap[y, x] = -100.0
        
        # 2. Проверяем наличие пустых клеток в 2 клетках от екущей

        step_dict = {(2, 0): (1, 0), (-2, 0): (-1, 0), (0, 2): (0, 1), (0, -2): (0, -1)}
        
        for (dy2, dx2), (dy1, dx1) in step_dict.items():
            ny, nx = y + dy2, x + dx2
            mid_y, mid_x = y + dy1, x + dx1
            if self._in_bounds(nx, ny) and self.layout[ny, nx] == 0:
                self.heatmap[mid_y, mid_x] = -100.0
                
        # 3. Проверяем соседей соседей текущей клетки (и если нулевых соседей + внешних стен >2 или =2 (но это все 0) то назначаем -100

        for ny, nx in self._neighbors(y, x):
            if ny == prev_y and nx == prev_x:
                continue
            temp = {}
            for nyn, nxn in self._neighbors(ny, nx):
                if self.layout[nyn, nxn] == 0:
                    temp[(nyn, nxn)] = "empty"
                elif self._is_outer_wall(nyn, nxn):
                    temp[(nyn, nxn)] = "outer_wall"

            if len(temp) > 2:
                for (ty, tx) in temp.keys():
                    self.heatmap[ty, tx] = -100.0
            elif len(temp) == 2:
                if "outer_wall" not in temp.values():
                    for (ty, tx) in temp.keys():
                        self.heatmap[ty, tx] = -100.0
    
    
    def _init_heatmap_place_exit(self):
        """
        Готовим heatmap для размещения выхода

        """
        # Вначале заполним весь heatmap -100
        self.heatmap.fill(-100)
        
        
        # Ищем внешние стены которые примыкают к прокопанному лабиринту.
        
        border_cells =[]
        for y in range(self.size):
            for x in range(self.size):
                if self.layout[y, x] == 0:
                    for nyn, nxn in self._neighbors(y, x):   
                        if self._is_outer_wall(y,x):   # т.е если сосед с нашим лабиринтом внешняя стена то сохраняем его в список
                            border_cells.append((y,x))
                            
        for dy, dx in border_cells:
            if len(border_cells) ==1:
                self.heatmap[dy,dx] = random.randint(9, 15)
            else:
                if (dy == 0 or dx==0) and abs(dx-dy)==2: # т.е клетка соседняя с самой начально пзицией лабиринта.
                    self.heatmap[dy,dx] = 5
                self.heatmap[dy,dx] = random.randint(9, 15)
                
    def _init_heatmap_place_key(self):
    
        # Обнуляем
        self.heatmap.fill(-100)
        visited = np.zeros_like(self.layout, dtype=bool)
        components = []
        exit_component = None

        # BFS по проходам
        for y in range(self.size):
            for x in range(self.size):
                if self.layout[y, x] == 0 and not visited[y, x]:
                    q = deque([(y, x)])
                    visited[y, x] = True
                    comp = []
                    has_exit = False
                    while q:
                        cy, cx = q.popleft()
                        comp.append((cy, cx))
                        if (cy, cx) == self.exit_pos:
                            has_exit = True
                        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ny, nx = cy+dy, cx+dx
                            if 0 <= ny < self.size and 0 <= nx < self.size:
                                if not visited[ny, nx] and self.layout[ny, nx] == 0:
                                    visited[ny, nx] = True
                                    q.append((ny, nx))
                    components.append((comp, has_exit))
                    if has_exit:
                        exit_component = comp

        # Расставляем награды
        if len(components) > 1:
            # Есть несколько путей — бонус только на путях без выхода
            for comp, has_exit in components:
                if not has_exit:
                    for (y, x) in comp:
                        self.heatmap[y, x] = random.randint(9, 15)
        else:
            # Один путь — даём бонусы по удалённости от выхода
            if exit_component:
                dist_map = self._bfs_distances_from_exit(exit_component)
                max_dist = max(dist_map.values())
                for (y, x), d in dist_map.items():
                    if d > max_dist // 2:
                        self.heatmap[y, x] = random.randint(9, 15)

                        
    def _init_heatmap_place_other(self):
        # Сбросить всё на -100
        self.heatmap.fill(-100)

        # Находим координаты всех проходов
        pass_mask = (self.layout == 0)

        # Случайные значения только для проходов
        self.heatmap[pass_mask] = np.random.randint(9, 16, size=pass_mask.sum())                    
                        
    
    def _place_step(self, action, type_place):
        """
        Размещение объектов в фазах 'place_key', 'place_exit', 'place_other'.
        """
        if self.verbose:
            print("[_place_step] Начинается размещение.")

        reward = 0

        # ---------- Выбор координат ----------
        if type_place == "exit":
            
            if self.verbose:
                print("[PLACE DEBUG] Текущий лабиринт:")
                print(self.layout)
            
            idx = action - 4
            y, x = divmod(idx, self.size)
            
            if self.verbose:
                print(f"ПРобуем ставить y = {y}, x = {x}")
                
            if not (x == 0 or y == 0 or x == self.size - 1 or y == self.size - 1):  # Попытку поставить выход не на внешней стене- пресекаем
                 return -100.0, False
            else:
                 if (x == 0 and y == 0) or \
                   (x == 0 and y == self.size - 1) or \
                   (x == self.size - 1 and y == 0) or \
                   (x == self.size - 1 and y == self.size - 1):
                    return -100.0, False
            
            self.exit_pos = (y, x)
            self.layout[y, x] = 7
            
            reward = self.heatmap[y, x]
            
            self._set_phase("place_key")
            
            return reward, False

        elif type_place == "key":
            # Ищем клетку максимально далёкую И от входа, И от выхода

            idx = action - 4
            y, x = divmod(idx, self.size)
            
            if self.verbose:
                print(f"[_place_step] action={action}, y={y}, x={x}")
                print(f"ПРобуем ставить y = {y}, x = {x}")
                
            if self.layout[y, x] != 0: # Попытку поставить ключ не в пустю клетку персекаем
                return -100.0, False
            
            self.key_pos = (y, x)
            self.layout[y, x] = 2
            
            reward = self.heatmap[y, x]
            
            self._set_phase("place_other")
            
        elif type_place == "other":
            
            # Проверяем, не достигли ли лимита для всех элементов
            all_placed = all(
                self.placed[e] >= self.allowed_elements[e]
                for e in self.placeable_elements
            )
            if all_placed:
                self._set_phase("eval")
                return 0, False  
               

            idx = action - 4
            y, x = divmod(idx, self.size)
            
            if self.layout[y, x] != 0: # Попытку поставить элемент не в пустю клетку персекаем
                return -100.0, False
            
            element = None
            for e in self.placeable_elements:
                if self.placed[e] < self.allowed_elements[e]:
                    # Размещаем элемент e
                    self.placed[e] += 1
                    element = e
                    break
                else:
                    # лимит достигнут, идём к следующему элементу
                    continue
                    
            if element is None:
                # Такое может быть, если лимиты достигнуты, но мы не вышли в фазу eval
                self._set_phase("eval")
                return 0, False
        
            # Размещаем выбранный элемент
            self.layout[y, x] = element
            reward = self.heatmap[y, x]
            
            
            self._init_heatmap_place_other() # обновляем heatmap
            

        return reward, False

    
    def _eval_phase(self):
        if self.verbose:
                print("[EVALUATE] Запуск оценки лабиринта...")
                print("[DEBUG] Текущий лабиринт:")
                print(self.layout)

        try:
            
            # Передаём именно maze, а не размер!
            if self.use_stub_eval:
                eval_info = self._stub_evaluate(self.layout)
            else:
                eval_info = self.navigator.evaluate(self.layout)

            if self.verbose:
                print(f"[EVALUATE] Результаты навигатора: {eval_info}")

            final_rating = self._rate_phase(self.layout, eval_info)
            self.rating += final_rating

            # Тут, судя по твоему коду, result_maze должен быть словарём для сохранения

            self.result_maze = {
                "success": bool(eval_info.get("success", False)),
                "steps": int(eval_info.get("steps", 0)),
                "turns": int(eval_info.get("turns", 0)),
                "has_key": bool(eval_info.get("has_key", False)),
                "has_exit": bool(eval_info.get("has_exit", False)),
                "rating": float(final_rating),
            }
            if self.verbose:
                print(f"[FINALIZE] Оценка завершена. Рейтинг: {final_rating}, Успех: {eval_info.get('success')}")

        except Exception as e:
            print(f"[FINALIZE] Ошибка в навигаторе: {e}")
            self.result_maze = {
                "success": False,
                "steps": 0,
                "turns": 0,
                "has_key": False,
                "has_exit": False,
                "rating": -100.0,
                "error": str(e)
            }

        self.phase = "save_maze"
        return 0.0, False



    def _rate_phase(self, layout, eval_info):
        empty_cells = np.sum(layout == 0)  # просто считаем пустые клетки

        size = self.size
        coeff_maze = (size * size) // 2  # или другой коэффициент по смыслу

        rating = 0

        if empty_cells < coeff_maze * 0.3:
            rating -= 10  # слишком мало пустых клеток
        elif empty_cells > coeff_maze:
            rating -= 50  # слишком много пустых клеток

        empty_per_row = np.sum(layout == 0, axis=1)
        std_empty = np.std(empty_per_row)
        if std_empty > coeff_maze * 0.1:
            rating -= std_empty * 2
        else:
            rating += 5

        # Проверка количества элементов
        for elem, expected in self.allowed_elements.items():
            if expected is not None:
                actual = np.sum(layout == elem)
                if actual != expected:
                    rating -= 5 * abs(actual - expected)
                else:
                    rating += 5

        if not eval_info.get("has_exit", False):
            rating -= 25
            if self.verbose:
                print("[RATING] Наказание: выход не размещён.")
        if not eval_info.get("has_key", False):
            rating -= 25
            if self.verbose:
                print("[RATING] Наказание: ключ не размещён.")
        
        if not eval_info.get("has_exit", False) and not eval_info.get("has_key", False):
            key_pos = np.argwhere(layout == 2)
            exit_pos = np.argwhere(layout == 7)

            if len(key_pos) > 0 and len(exit_pos) > 0:
                dist = np.linalg.norm(key_pos[0] - exit_pos[0])
                if dist < self.size / 3:
                    rating -= 10  # слишком близко
                elif dist > self.size * 0.7:
                    rating += 5  # хорошо разбросано
                    
        key_pos = np.argwhere(layout == 2)
        exit_pos = np.argwhere(layout == 7)
        start_pos = (1, 1)

        if len(key_pos) > 0 and len(exit_pos) > 0:
            key = tuple(key_pos[0])
            exit_ = tuple(exit_pos[0])

            # Функция shortest_path_info возвращает (длина, количество путей)
            start_key_len, start_key_paths = shortest_path_info(layout, start_pos, key)
            key_exit_len, key_exit_paths = shortest_path_info(layout, key, exit_)

            # Штрафы за слишком короткие пути (слишком простой лабиринт)
            min_path_len = size // 2
            if start_key_len < min_path_len:
                rating -= (min_path_len - start_key_len) * 5
            if key_exit_len < min_path_len:
                rating -= (min_path_len - key_exit_len) * 5

            # Штраф за слишком много альтернативных путей (слишком прозрачный лабиринт)
            max_paths = 3
            if start_key_paths > max_paths:
                rating -= (start_key_paths - max_paths) * 2
            if key_exit_paths > max_paths:
                rating -= (key_exit_paths - max_paths) * 2

            # Проверка расстояния между ключом и выходом (евклидово)
            dist = np.linalg.norm(np.array(key) - np.array(exit_))
            if dist < size / 3:
                rating -= 10  # слишком близко
            elif dist > size * 0.7:
                rating += 5  # хорошо разбросано

        if eval_info.get("success"):
            rating += 50
        else:
            rating -= 20

        rating -= 0.01 * eval_info.get("steps", 0)
        rating -= 0.05 * eval_info.get("turns", 0)

        turns = eval_info.get("turns", 0)
        min_turns = coeff_maze * 0.7
        max_turns = coeff_maze * 2

        if turns != 0:
            if turns < min_turns:
                rating -= (min_turns - turns) * 0.1
                if self.verbose:
                    print(f"[RATING] Штраф за слишком малое количество поворотов: {turns}")
            elif min_turns <= turns <= max_turns:
                rating += 10
                if self.verbose:
                    print(f"[RATING] Бонус за хорошее количество поворотов: {turns}")
            else:
                rating -= (turns - max_turns) * 0.2
                if self.verbose:
                    print(f"[RATING] Штраф за слишком большое количество поворотов: {turns}")

        return rating

    def _save_maze(self):
        if self.verbose:
            print(f"[SAVE] Запуск сохранения лабиринта")
        save_dir = "saved_mazes"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"maze_{self.size}x{self.size}_{timestamp}.csv"
        full_path = os.path.join(save_dir, fname)

        # np.savetxt(full_path, self.layout, fmt="%d", delimiter=",")  # пока отключено

        meta_log_path = os.path.join(save_dir, "meta.jsonl")

        meta_entry = self.result_maze.copy()
        meta_entry["csv_path"] = fname
        meta_entry["timestamp"] = timestamp

        with open(meta_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta_entry, ensure_ascii=False) + "\n")

        if self.verbose:
            print(f"[SAVE] Лабиринт сохранён: {full_path} + метаданные в {meta_log_path}")
        return 0.0, True

    def _stub_evaluate(self, layout):
        try:
            start_pos = (1, 1)
            key_pos = np.argwhere(layout == 2)
            exit_pos = np.argwhere(layout == 7)

            has_key = len(key_pos) > 0
            has_exit = len(exit_pos) > 0
            empty_cells = int(np.sum(layout == 0))  # исправлено здесь

            if not has_key or not has_exit:
                return {
                    "success": False,
                    "has_key": has_key,
                    "has_exit": has_exit,
                    "steps": 0,
                    "turns": 0,
                    "empty_cells": empty_cells,
                    "error": "Missing key or exit"
                }

            key = tuple(key_pos[0])
            exit_ = tuple(exit_pos[0])

            start_key_len, start_key_paths = shortest_path_info(layout, start_pos, key)  # исправлено
            key_exit_len, key_exit_paths = shortest_path_info(layout, key, exit_)

            # Проверяем недостижимость (inf или -1)
            if start_key_len == -1 or key_exit_len == -1 or np.isinf(start_key_len) or np.isinf(key_exit_len):
                return {
                    "success": False,
                    "has_key": has_key,
                    "has_exit": has_exit,
                    "steps": 0,
                    "turns": 0,
                    "empty_cells": empty_cells,
                    "error": "Unreachable key or exit"
                }

            total_path_len = start_key_len + key_exit_len
            turns = self._estimate_turns_from_path(layout, start_pos, key, exit_)

            key_exit_dist = np.linalg.norm(np.array(key) - np.array(exit_))
            if np.isinf(key_exit_dist) or np.isnan(key_exit_dist):
                key_exit_dist = -1  # безопасное значение

            return {
                "success": True,
                "has_key": True,
                "has_exit": True,
                "steps": int(total_path_len),
                "turns": int(turns),
                "empty_cells": empty_cells,
                "start_key_len": int(start_key_len),
                "key_exit_len": int(key_exit_len),
                "start_key_paths": int(start_key_paths),
                "key_exit_paths": int(key_exit_paths),
                "key_exit_dist": float(key_exit_dist),
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "has_key": False,
                "has_exit": False,
                "steps": 0,
                "turns": 0,
                "empty_cells": 0,
                "error": str(e)
            }   

    # ===============================
    # ВСПОМОГАТЕЛЬНОЕ
    # ===============================
    def _bfs_distances_from_exit(self, component):
        # BFS для подсчёта дистанции от выхода
        from collections import deque
        dist = {}
        q = deque([self.exit_pos])
        dist[self.exit_pos] = 0
        while q:
            cy, cx = q.popleft()
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = cy+dy, cx+dx
                if (ny, nx) in component and (ny, nx) not in dist:
                    dist[(ny, nx)] = dist[(cy, cx)] + 1
                    q.append((ny, nx))
        return dist
                
        
#     def find_dig_positions(self):
#         if self.verbose:
#             print(f'текущая тепловая карта {self.heatmap} текущий курсор {self.cursor_y, self.cursor_x}')
#         CARDINAL = [(1,0), (-1,0), (0,1), (0,-1)]  # (dy, dx)

#         y, x = self.cursor_y, self.cursor_x
#         if self.layout[y, x] == 0:
#             for dy, dx in CARDINAL:
#                 ny, nx = y + dy, x + dx
#                 if self._in_bounds(nx, ny) and self.layout[ny, nx] == 1:
#                     if self.heatmap[ny, nx] > 0:
#                         return True
#         return False

    def find_dig_positions(self):
        """
        Проверяем, есть ли вокруг текущего курсора хотя бы одна клетка для копания
        с положительной наградой (>0).
        """
        if self.verbose:
            print(f'[find_dig_positions] Текущая тепловая карта:\n{self.heatmap}')
            print(f'[find_dig_positions] Текущий курсор: {(self.cursor_y, self.cursor_x)}')

        CARDINAL = [(1,0), (-1,0), (0,1), (0,-1)]
        y, x = self.cursor_y, self.cursor_x

        if self.layout[y, x] != 0:
            # Текущая клетка должна быть уже выкопана (0), иначе ничего делать
            return False

        for dy, dx in CARDINAL:
            ny, nx = y + dy, x + dx
            if self._in_bounds(nx, ny):
                if self.layout[ny, nx] == 1 and self.heatmap[ny, nx] > 0:
                    return True
        return False


    def _in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size
    
    
    def _neighbors(self, y, x):
        dirs = [(-1,0), (1,0), (0,-1), (0,1)]  # (dy, dx)
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.size and 0 <= nx < self.size:
                yield ny, nx  # возвращаем (y,x)
    
    def _zero_neighbors(self, y, x):
        dirs = [(-1,0), (1,0), (0,-1), (0,1)]  # (dy, dx)
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.size and 0 <= nx < self.size:
                if self.layout[ny, nx] == 0:
                    yield ny, nx  # возвращаем (y,x) все нулевых соседей
                    
    def zero_neighbors_count(self, y, x):
        return sum(1 for _ in self._zero_neighbors(y, x))
                    
                    
    def _is_outer_wall(self, y, x):
        return (
            self.layout[y, x] == 1 and (
                y == 0 or y == self.size - 1 or
                x == 0 or x == self.size - 1
            )
        )

    def _get_main_path(self, start, goal):
        rows, cols = self.maze.shape
        dist = [[float('inf')] * cols for _ in range(rows)]
        prev = [[None] * cols for _ in range(rows)]
        dist[start[0]][start[1]] = 0

        queue = deque([start])
        directions = [(1,0), (-1,0), (0,1), (0,-1)]

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                break
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and self.maze[nx, ny] == 0:
                    if dist[nx][ny] == float('inf'):
                        dist[nx][ny] = dist[x][y] + 1
                        prev[nx][ny] = (x, y)
                        queue.append((nx, ny))

        # Если путь не найден
        if dist[goal[0]][goal[1]] == float('inf'):
            return []

        # Восстанавливаем путь с конца
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur[0]][cur[1]]
        path.reverse()
        return path

    def _estimate_turns_from_path(self, layout, start, key, exit_):
        def reconstruct_path(a, b):
            from collections import deque
            H, W = layout.shape
            prev = {}
            visited = np.full((H, W), False)
            q = deque([a])
            visited[a] = True

            while q:
                y, x = q.popleft()
                if (y, x) == b:
                    break
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and layout[ny, nx] in (0, 2, 7, 0) and not visited[ny, nx]:
                        visited[ny, nx] = True
                        prev[(ny, nx)] = (y, x)
                        q.append((ny, nx))

            path = []
            cur = b
            while cur != a:
                path.append(cur)
                cur = prev.get(cur)
                if cur is None:
                    return []  # Нет пути
            path.append(a)
            path.reverse()
            return path

        path = reconstruct_path(start, key)[:-1] + reconstruct_path(key, exit_)
        if len(path) < 3:
            return 0

        turns = 0
        for i in range(2, len(path)):
            dy1 = path[i-1][0] - path[i-2][0]
            dx1 = path[i-1][1] - path[i-2][1]
            dy2 = path[i][0] - path[i-1][0]
            dx2 = path[i][1] - path[i-1][1]
            if (dy1, dx1) != (dy2, dx2):
                turns += 1
        return turns
    
    
    
    def has_global_dig_positions(self):
        """
        Ищет все глобальные точки для копания и выбирает лучшую.
        Критерий: максимальный heatmap у целевой стены.
        """
        if self.heatmap is None:
            return None

        candidates = []

        for y in range(self.size):
            for x in range(self.size):
                # 1. Клетка должна быть пустой
                if self.layout[y, x] != 0:
                    continue

                # 2. Должен быть сосед-стена с наградой > 0
                target_wall = None
                target_heat = -1
                for dy, dx in self.directions.values():
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.size and 0 <= nx < self.size:
                        if self.layout[ny, nx] == 1 and self.heatmap[ny, nx] > 0:
                            # Запоминаем стену с наибольшим heatmap
                            if self.heatmap[ny, nx] > target_heat:
                                target_heat = self.heatmap[ny, nx]
                                target_wall = (ny, nx)

                if not target_wall:
                    continue

                # 3. Проверяем "широкий коридор"
                ny, nx = target_wall
                if self.zero_neighbors_count(ny, nx) >= 2:
                    continue

                # 4. Добавляем в кандидаты
                candidates.append(((y, x), target_heat))

        if not candidates:
            return None

        # 5. Выбираем лучшего кандидата по heatmap (можно заменить на расстояние до курсора)
        best_pos, _ = max(candidates, key=lambda item: item[1])

        # 6. Телепортируемся и пересчитываем тепловую карту
        self.cursor_y, self.cursor_x = best_pos
        self.compute_dig_heatmap(best_pos[0], best_pos[1], best_pos[0], best_pos[1])
        if self.verbose:
            print(f"[DIG] Телепорт к развилке (лучший выбор): {best_pos}")
        return best_pos
    
    def _update_action_mask(self):
        """
        Обновляем маску допустимых действий так, чтобы агент
        никогда не мог выбрать запрещённый ход.
        Запрещённые ходы:
          - выход за границы
          - внешняя стена
          - клетка с тепловой картой <= -99.9
          - уже выкопанная клетка (0)
        """
        self.action_mask = np.zeros(len(self.directions), dtype=bool)
        py, px = self.cursor_y, self.cursor_x

        for i, (dy, dx) in self.directions.items():
            ny, nx = py + dy, px + dx
            if not self._in_bounds(nx, ny):
                continue
            if self.layout[ny, nx] != 1:  # не стена → копать нельзя
                continue
            if self.heatmap[ny, nx] <= -99.9:  # запрещённая награда
                continue
            # Если все проверки прошли — ход допустим
            self.action_mask[i] = True