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


from maze_env.maze_env import MazeEnv, CustomTransformerPolicy, PositionalEncoding

def find_closest_expert_action(obs_np, expert_trajectories):
    """
    Для данного состояния obs_np (np.ndarray) ищем в expert_trajectories
    ближайшее состояние по лабиринту (L2 или L1 расстояние),
    возвращаем соответствующее экспертное действие (tile_id).
    """

    best_dist = None
    best_action = None

    for traj in expert_trajectories:
        for (exp_obs, (_, _, tile_id)) in traj:
            # расстояние между состояниями лабиринта (можно L2, L1, или просто sum of differences)
            dist = np.sum((obs_np - exp_obs) ** 2)  # L2 квадрат
            
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_action = tile_id
    
    return best_action

def convert_obs_to_tensor_dict(obs):
    """
    Преобразует одно или несколько наблюдений (maze-снимков) в dict[str, torch.Tensor],
    совместимый с policy.get_distribution().

    Аргументы:
        obs: np.ndarray
            - если (H, W) — одиночное наблюдение
            - если (B, H, W) — батч наблюдений

    Возвращает:
        dict[str, torch.Tensor] с согласованными размерностями
    """
    if obs.ndim == 2:
        # Одиночный снимок -> (1, H, W)
        obs = obs[np.newaxis, ...]

    B, H, W = obs.shape
    maze_tensor = torch.tensor(obs, dtype=torch.float32)

    # Остальные параметры — заглушки, размер (B, ...)
    cursor_tensor = torch.zeros((B, 2), dtype=torch.int32)
    steps_tensor = torch.zeros((B, 1), dtype=torch.float32)
    phase_tensor = torch.zeros((B,), dtype=torch.int32)
    rating_tensor = torch.zeros((B, 1), dtype=torch.float32)

    return {
        "maze": maze_tensor,
        "cursor": cursor_tensor,
        "steps": steps_tensor,
        "phase": phase_tensor,
        "rating": rating_tensor,
    }


def get_expert_action_from_index(idx, expert_actions_np):
    """
    Получить действие эксперта по индексу из плоского массива демонстраций
    """
    tile_id = expert_actions_np[idx][2]  # третье значение — tile_id
    return tile_id

def get_expert_action(obs):
    global expert_step_idx
    action = get_expert_action_from_index(expert_step_idx, expert_actions_np)
    expert_step_idx += 1
    return action

def prepare_expert_obs_batch(expert_obs_np):
    """
    expert_obs_np - массив лабиринтов (batch, H, W)
    cursor, steps, phase, rating заполняем "заглушками"
    """
    batch_size = expert_obs_np.shape[0]
    H, W = expert_obs_np.shape[1], expert_obs_np.shape[2]

    maze_tensor = torch.tensor(expert_obs_np, dtype=torch.float32)

    # Для cursor можно взять координаты действия (x, y) из expert_actions_np,
    # но если нет, можно поставить (0,0) или любые данные
    # Предположим, что курсор — это координаты клетки действия, тогда нужно подать отдельно

    # Если хочешь, здесь можно заполнить cursor случайными или нулевыми значениями
    cursor_tensor = torch.zeros((batch_size, 2), dtype=torch.int32)

    steps_tensor = torch.zeros((batch_size, 1), dtype=torch.float32)
    phase_tensor = torch.zeros((batch_size,), dtype=torch.int32)  # 0 - build
    rating_tensor = torch.zeros((batch_size, 1), dtype=torch.float32)

    return {
        "maze": maze_tensor,
        "cursor": cursor_tensor,
        "steps": steps_tensor,
        "phase": phase_tensor,
        "rating": rating_tensor,
    }


class PPOWithImitation(PPO):
    def __init__(self, *args, imitation_coef=1.0, imitation_lr=1e-4, **kwargs):
        super().__init__(*args, **kwargs)

        self.imitation_coef = imitation_coef
        self.imitation_lr = imitation_lr
        self.expert_obs = None
        self.expert_actions = None

        self.imitation_optimizer = None  # ← отложим создание

        # Если policy уже существует (например, при обычном создании модели)
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
        # policy уже загружена к этому моменту — можно создать оптимизатор
        self._init_imitation_optimizer()

def generate_demonstrations_pkl(n=100, size=5, save_path="trajectories.pkl"):
    all_trajectories = []

    for i in range(n):
        _, traj = traceable_generate_maze(size)
        all_trajectories.append(traj)

    with open(save_path, "wb") as f:
        pickle.dump(all_trajectories, f)

    print(f"[✓] Сохранено {n} демонстраций ({sum(len(t) for t in all_trajectories)} шагов) в {save_path}")

def load_demonstrations(pkl_path):
    with open(pkl_path, "rb") as f:
        trajectories = pickle.load(f)
    print(f"[✓] Загружено {len(trajectories)} демонстраций.")
    return trajectories
    
    
#  Преобразование в (obs, action) пары
def flatten_trajectories(trajectories):
    obs_list = []
    action_list = []

    for traj in trajectories:
        for step in traj:
            obs_list.append(step['obs'])
            action_list.append(step['action'])  # (x, y, tile_id)

    obs_array = np.stack(obs_list)
    action_array = np.array(action_list)

    print(f"[✓] Получено {len(obs_array)} пар (obs, action)")
    return obs_array, action_array



class CheckpointCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            model_path = os.path.join(self.save_path, f"builder_step_{self.n_calls}.zip")
            self.model.save(model_path)
            if self.verbose:
                print(f"Saved builder at step {self.n_calls}")
        return True


class NavigatorWrapper:
    def __init__(self, model_path: str, size, max_steps: int = 500):
        self.model = PPO.load(model_path,
                              custom_objects={"policy_class": CustomTransformerPolicy},
                              learning_rate=2.5e-4, clip_range=0.2)
        self.max_steps = max_steps
        if self.model.get_env() is None:
            dummy_maze = np.zeros((size, size), dtype=np.int32)
            dummy_env = MazeEnv(dummy_maze)
            self.model.set_env(dummy_env)

    def evaluate(self, maze: np.ndarray, start_pos=(1, 1)) -> dict:
        try:
            env = MazeEnv(maze, start_pos=start_pos)
            obs, _ = env.reset()
        except AssertionError as e:
            return {
                "success": False,
                "steps": 0,
                "has_key": False,
                "has_exit": False,
                "empty_cells": np.sum(maze == 0),
                "error": str(e)
            }
        except Exception as e:
            return {
                "success": False,
                "steps": 0,
                "has_key": False,
                "has_exit": False,
                "empty_cells": np.sum(maze == 0),
                "error": f"Unexpected error: {e}"
            }
        
        
        done = False
        total_steps = 0
        success = False
        lstm_state = None
        episode_start = True

        path_positions = [start_pos]
        for _ in range(self.max_steps):
            action, lstm_state = self.model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=True
            )
            obs, reward, terminated, truncated, info = env.step(action)
            pos = tuple(info.get("agent_pos", (-1, -1)))
            if pos != path_positions[-1]:
                path_positions.append(pos)
            total_steps += 1
            episode_start = terminated or truncated
            if terminated:
                success = True
                break

        has_key = np.sum(maze == 2) > 0
        has_exit = np.sum(maze == 7) > 0
        empty_cells = np.sum(maze == 0)

        # Повороты маршрута
        turns = 0
        for i in range(2, len(path_positions)):
            dy1 = path_positions[i - 1][0] - path_positions[i - 2][0]
            dx1 = path_positions[i - 1][1] - path_positions[i - 2][1]
            dy2 = path_positions[i][0] - path_positions[i - 1][0]
            dx2 = path_positions[i][1] - path_positions[i - 1][1]
            if (dy1, dx1) != (dy2, dx2):
                turns += 1
                
        if len(path_positions) <= 1:
            return {
                "success": False,
                "steps": total_steps,
                "has_key": has_key,
                "has_exit": has_exit,
                "empty_cells": empty_cells,
                "turns": turns,
                "error": "Agent did not move"
            }

        return {
            "success": success,
            "steps": total_steps,
            "has_key": has_key,
            "has_exit": has_exit,
            "empty_cells": empty_cells,
            "turns": turns,
            "error": None
        }


def load_trained_navigator(size):
    model_path = "navigator/ppo_maze_agent_v4"
    return NavigatorWrapper(model_path, size)

class CustomTransformerPolicyForBuilder(ActorCriticPolicy):
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

        # Список ключей observation для понимания порядка
        self.obs_keys = list(observation_space.spaces.keys())

        # --- conv для maze (1 канал) ---
        self.maze_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # --- conv для place_rewards (каждая карта — 1 канал, всего 3) ---
        self.place_rewards_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        maze_size = observation_space["maze"].shape  # (size, size)
        dummy_maze = th.zeros(1, 1, *maze_size)
        dummy_rewards = th.zeros(1, 3, *maze_size)

        maze_flat_dim = self.maze_conv(dummy_maze).shape[1]
        rewards_flat_dim = self.place_rewards_conv(dummy_rewards).shape[1]

        # Размер остальных признаков (cursor, phase, rating, placed, steps, jump_interval)
        other_dim = 2 + 4 + 1 + observation_space["placed"].shape[0] + 1 + 1

        print(f"maze_flat_dim={maze_flat_dim}, rewards_flat_dim={rewards_flat_dim}, other_dim={other_dim}, total input to linear={maze_flat_dim + rewards_flat_dim + other_dim}")

        # Линейный слой для объединения всех признаков
        self.other_proj = nn.Linear(maze_flat_dim + rewards_flat_dim + other_dim, d_model)

        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=memory_dim, batch_first=True)
        self.output = nn.Linear(memory_dim, d_model)

        self.hidden_state = None


    def forward(self, obs_dict) -> th.Tensor:
        batch_size = obs_dict["cursor"].shape[0]

        maze = obs_dict["maze"].unsqueeze(1).float()  # [B,1,H,W]
        maze_feat = self.maze_conv(maze)
        

        # Склеиваем карты наград в один тензор (B, 3, H, W)
        place_rewards_key = obs_dict.get("place_rewards_key", th.zeros_like(maze.squeeze(1)))
        if place_rewards_key.dim() == 3:  # [B, H, W]
            place_rewards_key = place_rewards_key.unsqueeze(1)  # [B, 1, H, W]

        place_rewards_exit = obs_dict.get("place_rewards_exit", th.zeros_like(maze.squeeze(1)))
        if place_rewards_exit.dim() == 3:
            place_rewards_exit = place_rewards_exit.unsqueeze(1)

        place_rewards_campfire = obs_dict.get("place_rewards_campfire", th.zeros_like(maze.squeeze(1)))
        if place_rewards_campfire.dim() == 3:
            place_rewards_campfire = place_rewards_campfire.unsqueeze(1)

        place_rewards = th.cat([place_rewards_key, place_rewards_exit, place_rewards_campfire], dim=1).float()
        rewards_feat = self.place_rewards_conv(place_rewards)

        cursor = obs_dict["cursor"].float()  # [B, 2]

        phase = obs_dict["phase"].float()
        if phase.ndim == 3:
            phase = phase.squeeze(1)
        if phase.ndim == 1:
            phase = phase.unsqueeze(1)  # [B,1]

        rating = obs_dict["rating"].float()
        if rating.dim() == 3:
            rating = rating.squeeze(-1)
        if rating.dim() == 1:
            rating = rating.unsqueeze(1)

        steps = obs_dict.get("steps", th.zeros((batch_size, 1), device=maze.device)).float()
        if steps.dim() == 3:
            steps = steps.squeeze(-1)
        if steps.dim() == 1:
            steps = steps.unsqueeze(1)

        jump_interval = obs_dict.get("jump_interval", th.zeros((batch_size, 1), device=maze.device)).float()
        if jump_interval.dim() == 3:
            jump_interval = jump_interval.squeeze(-1)
        if jump_interval.dim() == 1:
            jump_interval = jump_interval.unsqueeze(1)

        placed = obs_dict["placed"].float()
        if placed.dim() == 3:
            placed = placed.squeeze(-1)
        if placed.dim() == 1:
            placed = placed.unsqueeze(1)
            
        other = th.cat([cursor, phase, steps, rating, placed, jump_interval], dim=1)

        combined = th.cat([maze_feat, rewards_feat, other], dim=1)
        

        x = self.other_proj(combined).unsqueeze(1)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        if self.hidden_state is None:
            h0 = th.zeros(1, batch_size, self.lstm.hidden_size, device=x.device)
            c0 = th.zeros(1, batch_size, self.lstm.hidden_size, device=x.device)
            self.hidden_state = (h0, c0)
        else:
            h, c = self.hidden_state
            if h.size(1) != batch_size:
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
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0:
                if dist[nx][ny] == float('inf'):
                    dist[nx][ny] = dist[x][y] + 1
                    path_count[nx][ny] = path_count[x][y]
                    queue.append((nx, ny))
                elif dist[nx][ny] == dist[x][y] + 1:
                    path_count[nx][ny] += path_count[x][y]

    if dist[goal[0]][goal[1]] == float('inf'):
        return float('inf'), 0
    return dist[goal[0]][goal[1]], path_count[goal[0]][goal[1]]



class MazeBuilderEnvDFS(gym.Env):
    def __init__(self, size=7, use_stub_eval= False, navigator=None, verbose=0):
        self.size = size
        self.maze = np.ones((size, size), dtype=np.uint8)  # Всё стены
        self.phase = "dig"
        self.cursor = [0, 0]
        self.done = False
        self.rating = 0.0
        self.verbose = verbose
        self.result_maze = {}
        self.last_direction = None
        self.turn_points = []
        self.steps_since_turn_jump = 0
        self.jump_interval = self.size*4 
        self.step_count = 0
        self.dug_count = 0
        self.place_attempts = 0
        self.MAX_ATTEMPTS = 10
        self.same_dir_count = 0
        self.navigator = navigator
        self.use_stub_eval = use_stub_eval
        self.stuck_mazes = {}  # {maze_hash: count}
        self.current_maze_hash = None
        self.place_rewards_key = {}    # координаты -> награды для ключа
        self.place_rewards_exit = {}   # координаты -> награды для выхода
        self.place_rewards_campfire = {}  # координаты -> награды для костра

        self.place_rewards_initialized = False
        
        if not use_stub_eval:
            if self.navigator is None:
                self.navigator = load_trained_navigator(self.size)

        self.elements = {
            0: "empty",
            1: "wall",
            2: "key",
            4: "trap",
            5: "campfire",
            7: "exit",
        }

        self.allowed_elements = {
            0: None,  # можно ставить сколько угодно пустых клеток
            1: None,  # стены не ограничены
            2: 1,     # только 1 ключ
            4: 0,     # запретить ловушки (или разрешить позже)
            5: 1,     # 1 костёр
            7: 1,     # 1 выход
        }

        self.placeable_elements = [e for e, max_count in self.allowed_elements.items() if max_count != 0 and e not in (0, 1)]
        self.placed = {e: 0 for e in self.allowed_elements}

        self.directions = {
            0: (-1, 0),  # вверх
            1: (1, 0),   # вниз
            2: (0, -1),  # влево
            3: (0, 1),   # вправо
        }

        self.action_space = spaces.Discrete(4)  # переключается по фазе
        self.observation_space = spaces.Dict({
            "maze": spaces.Box(low=0, high=10, shape=(self.size, self.size), dtype=np.uint8),
            "phase": spaces.Discrete(4),  # 0 - dig, 1 - place, 2 - eval, 3 - save
            "cursor": spaces.Box(low=0, high=self.size, shape=(2,), dtype=np.int32),
            "placed": spaces.Box(low=0, high=10, shape=(len(self.placeable_elements),), dtype=np.int32),
            "rating": spaces.Box(low=-200.0, high=200.0, shape=(1,), dtype=np.float32),
            "steps": spaces.Box(low=0, high=999, shape=(1,), dtype=np.int32),
            "jump_interval": spaces.Box(low=1, high=999, shape=(1,), dtype=np.int32),  # опционально
            "turn_points": spaces.Box(low=0, high=self.size, shape=(5, 2), dtype=np.int32),

            "place_rewards_key": spaces.Box(low=-np.inf, high=np.inf, shape=(self.size, self.size), dtype=np.float32),
            "place_rewards_exit": spaces.Box(low=-np.inf, high=np.inf, shape=(self.size, self.size), dtype=np.float32),
            "place_rewards_campfire": spaces.Box(low=-np.inf, high=np.inf, shape=(self.size, self.size), dtype=np.float32),
        })


    def reset(self, *, seed=None, options=None, preserve_eval=False):
        super().reset(seed=seed)  # если наследуешься от gym.Env и хочешь поддерживать seed
        
        self.maze.fill(1)
        self.rating = 0.0
        self.cursor = [1, 1]
        self.maze[1, 1] = 0
        self.phase = "dig"
        self.done = False
        self.result_maze = None
        self.dig_steps = 0.0
        self.placed = {e: 0 for e in self.placeable_elements}
        self.last_direction = None
        self.turn_points = []
        self.steps_since_turn_jump = 0
        self.dug_count = 0
        self.step_count = 0
        self.last_direction = None
        self.place_attempts = 0
        self.same_dir_count = 0
        self.stuck_mazes = {}  # {maze_hash: count}
        self.current_maze_hash = None
        self.place_rewards_key = {}    # координаты -> награды для ключа
        self.place_rewards_exit = {}   # координаты -> награды для выхода
        self.place_rewards_campfire = {}  # координаты -> награды для костра

        self.place_rewards_initialized = False
        
        
        obs = self.get_obs()
        info = {}  # можно добавить сюда нужную инфу, если нужно
        if self.verbose:
            print(f"[RESET] Курсор установлен в: {self.cursor}, фаза: {self.phase}")
        return obs, info


    def get_obs(self):
        phase_map = {"dig": 0, "place": 1, "eval": 2, "save_maze": 3}
        placed_arr = np.array([self.placed[e] for e in self.placeable_elements], dtype=np.int32)

        padded_turn_points = np.zeros((5, 2), dtype=np.int32)
        for i, point in enumerate(self.turn_points[:5]):
            padded_turn_points[i] = point

        # Конвертация и нормализация словарей наград (если они есть)
        key_reward_map = self._dict_to_map(self.place_rewards_key) if hasattr(self, 'place_rewards_key') else np.zeros((self.size, self.size), dtype=np.float32)
        exit_reward_map = self._dict_to_map(self.place_rewards_exit) if hasattr(self, 'place_rewards_exit') else np.zeros((self.size, self.size), dtype=np.float32)
        campfire_reward_map = self._dict_to_map(self.place_rewards_campfire) if hasattr(self, 'place_rewards_campfire') else np.zeros((self.size, self.size), dtype=np.float32)
        

        return {
            "maze": self.maze.copy(),
            "phase": phase_map[self.phase],
            "cursor": np.array(self.cursor, dtype=np.int32),
            "placed": placed_arr,
            "rating": np.array([self.rating], dtype=np.float32),
            "steps": np.array([self.step_count], dtype=np.int32),
            "jump_interval": np.array([self.jump_interval], dtype=np.int32),
            "turn_points": padded_turn_points,
            "place_rewards_key": key_reward_map,
            "place_rewards_exit": exit_reward_map,
            "place_rewards_campfire": campfire_reward_map
            
    
        }
    

    def _set_phase(self, phase_name):
        self.phase = phase_name
        
        if self.verbose:
            print(f'phase_name {phase_name}')
            
        if phase_name == "dig":
            self.cursor = [1, 1]
            self.action_space = spaces.Discrete(4)
            self.last_direction = None
            self.turn_points = []
            self.steps_since_turn_jump = 0
            self.dug_count = 0
            self.step_count = 0
        elif phase_name == "place":
            self.cursor = [0, 0]
            self.action_space = spaces.Discrete(len(self.placeable_elements))
            self.last_direction = None
            self.turn_points = []
            self.steps_since_turn_jump = 0
        else:
            self.action_space = spaces.Discrete(1)


    def step(self, action):
        if self.done:
            raise Exception("Episode has ended. Call reset() to start a new one.")
        
        reward = 0   
            
        self.last_info = {}
        
        if self.verbose:
            print(f'фаза {self.phase}')
            
        if self.phase == "dig":
            reward, done = self._dig_step(action)
        elif self.phase == "place":
            reward, done = self._place_step(action)
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

        return self.get_obs(), reward, terminated, truncated, info

    
    def _dig_step(self, action):
        if self.verbose:
            print(f"[DEBUG] Action: {action}, Cursor before step: {self.cursor}")
        
        reward = 0
        # --- Hash лабиринта для проверки застревания ---
        maze_hash = hash(self.maze.tobytes())
        if maze_hash == self.current_maze_hash:
            self.stuck_mazes[maze_hash] += 1
        else:
            self.current_maze_hash = maze_hash
            self.stuck_mazes[maze_hash] = 1

        if self.stuck_mazes[maze_hash] > 10:
            if self.verbose:
                print("[STUCK] Лабиринт не меняется >10 шагов. Переход к place.")
            self._set_phase("place")
            reward += self._rate_dig_phase()
            self.last_info = {"phase": "place", "reason": "stuck_loop_detected"}
            return reward, False  # reward можно 0, т.к. фаза сменена

        # --- Печать состояния лабиринта ---
        if self.step_count % 5 == 0 and self.verbose:
            print("[DEBUG] Текущий лабиринт:")
            print(self.maze)

        # --- Проверка количества пустых клеток ---
        empty_cells = np.sum(self.maze == 0)
        if empty_cells < 2:
            prev_len, prev_paths = float('inf'), 0
        else:
            try:
                prev_len, prev_paths = shortest_path_info(
                    self.maze, (1,1), (self.size-2, self.size-2)
                )
            except IndexError:
                print("[WARNING] shortest_path_info вызвал IndexError — пропуск проверки")
                prev_len, prev_paths = float('inf'), 0

        direction_idx = int(action)
        dx, dy = self.directions[direction_idx]

        
        done = False

        self.last_action = action
        self.step_count += 1
        self.steps_since_turn_jump += 1

        x, y = self.cursor
        nx, ny = x + dx, y + dy

        # --- Выход за границы ---
        if not 0 <= x < self.size and 0 <= y < self.size:
            self.last_info = {"phase": "dig", "reason": "oob"}
            return -50, False

        # --- Запрет копания внешней стены ---
        if self._is_outer_wall(nx, ny):
            self.last_info = {"phase": "dig", "reason": "wall"}
            return -10, False

        # --- Ход в уже пустую клетку ---
        if self.maze[nx, ny] == 0:
            reward -= 1
            self.last_info = {"phase": "dig", "reason": "empty"}
            self.rating += reward
            return reward, False

        # --- Проверка на тупик ---
        if self._is_dig_dead_end(nx, ny):
            reward += 3
            # Очищаем поворотные точки от мёртвых
            self.turn_points = [pt for pt in self.turn_points if not self._is_dig_dead_end(*pt)]

            if self.turn_points:
                new_cursor = random.choice(self.turn_points)
                self.cursor = list(new_cursor)
                reward += 1
                return reward, False
            else:
                if self.verbose:
                    print("[DEBUG] Все поворотные точки — тупики. Пробуем fallback.")
                # Fallback: случайная пустая клетка
                
                self._set_phase("place")
                reward += self._rate_dig_phase()
                self.last_info = {"phase": "place", "reason": "no_turn_points_fallback_failed"}
                return reward, False

        # --- Штраф за широкий проход ---
        empty_neighbors = self._count_empty_neighbors(nx, ny)
        if empty_neighbors > 1:
            reward -= (empty_neighbors - 1) * 2
        if empty_neighbors > 2:
            return -100, False

        # --- Копаем стену ---
        self.maze[nx, ny] = 0
        self.cursor = [nx, ny]
        self.dug_count += 1
        if self.verbose:
            print(f"[DEBUG] Cursor after step: {self.cursor}")

        reward += 1  # базовый
        reward += min(0.1 * self.dug_count, 2.0)  # ограничить линейный рост

        # --- Учёт поворота ---
        if self.last_direction is not None and direction_idx != self.last_direction:
            reward += 2
            self.turn_points.append(tuple(self.cursor))
            if len(self.turn_points) > 30:
                self.turn_points.pop(0)
        self.last_direction = direction_idx

        # --- Shortest path reward ---
        try:
            new_len, new_paths = shortest_path_info(self.maze, (1,1), (self.size-2, self.size-2))
            if new_len > prev_len:
                reward += (new_len - prev_len) * 0.5

            if new_paths > prev_paths:
                reward -= (new_paths - prev_paths) * 0.5
        except IndexError:
            pass  # Игнорируем временные IndexError

        # --- Одно направление подряд ---
        if self.last_direction is not None:
            if direction_idx == self.last_direction:
                self.same_dir_count += 1
                if self.same_dir_count > 3:
                    reward -= 1 + 0.2 * (self.same_dir_count - 3)
            else:
                self.same_dir_count = 0

        self.rating += reward
        self.last_info = {"phase": "dig"}
        return reward, False
    
    
    def _rate_dig_phase(self):
        """
        Оценка завершённой фазы копания:
        - за разнообразие (кол-во поворотов)
        - за общую длину пути
        - за connectedness (кол-во путей от входа до выхода)
        - штраф за слишком короткий путь или изолированные области
        """
        rating = 0

        empty_cells = np.argwhere(self.maze == 0)
        num_empty = len(empty_cells)

        if num_empty < 5:
            return -50  # слишком короткий маршрут

        try:
            sp_len, sp_count = shortest_path_info(self.maze, (1, 1), (self.size - 2, self.size - 2))
        except Exception:
            return -30  # плохая связанность

        if sp_len < self.size:
            rating -= 10  # слишком короткий путь
        elif sp_len > self.size * 1.5:
            rating += 10  # длинный маршрут = +интерес

        if sp_count > 3:
            rating -= (sp_count - 3) * 5  # за слишком много альтернативных путей — плохо
        elif sp_count == 1:
            rating += 5  # идеально, только один путь

        # Оценка разнообразия
        if len(self.turn_points) > 10:
            rating += 10
        else:
            rating += len(self.turn_points)

        return rating

    # --- Вспомогательные функции ---
    def _in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def _is_outer_wall(self, x, y):
        return x == 0 or x == self.size - 1 or y == 0 or y == self.size - 1

    def _count_empty_neighbors(self, x, y):
        count = 0
        # Если сам x,y за границей, возвращаем 0
        if not (0 <= x < self.size and 0 <= y < self.size):
            return 0
        for dx, dy in self.directions.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                
                if self.maze[nx, ny] == 0:
                    count += 1
        return count

    def _is_dig_dead_end(self, x, y):
        # защита от выхода за границы
        if not (0 <= x < self.size and 0 <= y < self.size):
            return True
        dead = True
        for dx, dy in self.directions.values():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                continue
            # проверяем соседнюю клетку
            if self.maze[nx, ny] == 1 and not self._is_outer_wall(nx, ny):
                empty = self._count_empty_neighbors(nx, ny)
                if empty <= 1:
                    dead = False
        if dead and self.verbose:
            print(f"[DEBUG] Dead end detected at ({x},{y})")
        return dead


    def _place_step(self, element_idx):
        if not self.place_rewards_initialized:
            self._init_place_rewards()

        reward = 0
        element_idx = int(element_idx)
        max_idx = len(self.placeable_elements) - 1

        if not (0 <= element_idx <= max_idx):
            return -1.0, False  # ошибка действия

        element = self.placeable_elements[element_idx]
        x, y = self.cursor
        pos = (x, y)

        # Попытка размещения
        can_place = False
        if element == 7:
            if (
                self._is_border_cell(x, y)
                and self.maze[x, y] == 1
                and self._can_place(element)
                and not ((x in [0, self.size - 1]) and (y in [0, self.size - 1]))
            ):
                can_place = True
        elif element in (2, 5):
            if self.maze[x, y] == 0 and self._can_place(element):
                can_place = True

        if can_place:
            self.maze[x, y] = element
            self.placed[element] += 1

            # Добавляем награду из словаря (если есть)
            if element == 7:
                reward += 5.0 + self.place_rewards_exit.get(pos, 0)
            elif element == 2:
                reward += 3.0 + self.place_rewards_key.get(pos, 0)
            elif element == 5:
                reward += 1.5 + self.place_rewards_campfire.get(pos, 0)
            else:
                reward += 0  # Другие элементы (если будут)

            self._update_place_rewards_after_placement(element, pos)
            result = "placed"
            if element == 7:
                result = "placed_exit"
        else:
            reward -= 1.5 if element in (2, 5) else 1.0
            result = "fail_place" if element in (2, 5) else "fail_exit"

        self.last_info = {"phase": "place", "element": element, "result": result}

        # Сдвиг курсора
        self.cursor[1] += 1
        if self.cursor[1] >= self.size:
            self.cursor[1] = 0
            self.cursor[0] += 1

        if self.verbose:
            print(f"[PLACE] Объектов размещено: {dict(self.placed)}")
        self.debug_placement_status()

        # Условия завершения
        if self._all_elements_placed():
            if self.verbose:
                print("[PLACE] Все объекты расставлены. Переход к eval.")
            self._set_phase("eval")
            return reward, False

        if self.cursor[0] >= self.size:
            if self.verbose:
                print("[PLACE] Курсор дошёл до конца, но не все объекты размещены.")
            self.cursor = [0, 0]
            return reward, False

        self.rating += reward

        return reward, False


    def _is_border_cell(self, x, y):
        return x == 0 or y == 0 or x == self.size - 1 or y == self.size - 1

    def _can_place(self, element):
        limit = self.allowed_elements.get(element)
        placed = self.placed.get(element, 0)
        return limit is None or placed < limit
    
    def _all_elements_placed(self):
        for e in self.placeable_elements:
            allowed = self.allowed_elements.get(e, 0)
            placed = self.placed.get(e, 0)

            # Если allowed == None или 0, считаем, что ограничений нет или размещение не требуется
            if allowed is None or allowed == 0:
                continue

            if placed < allowed:
                return False
        return True


    def _eval_phase(self):
        if self.verbose:
                print("[EVALUATE] Запуск оценки лабиринта...")
                print("[DEBUG] Текущий лабиринт:")
                print(self.maze)

        try:
            
            # Передаём именно maze, а не размер!
            if self.use_stub_eval:
                eval_info = self._stub_evaluate(self.maze)
            else:
                eval_info = self.navigator.evaluate(self.maze)

            if self.verbose:
                print(f"[EVALUATE] Результаты навигатора: {eval_info}")

            final_rating = self._rate_phase(self.maze, eval_info)
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



    def _rate_phase(self, maze, eval_info):
        empty_cells = np.sum(maze == 0)  # просто считаем пустые клетки

        size = self.size
        coeff_maze = (size * size) // 2  # или другой коэффициент по смыслу

        rating = 0

        if empty_cells < coeff_maze * 0.3:
            rating -= 10  # слишком мало пустых клеток
        elif empty_cells > coeff_maze:
            rating -= 50  # слишком много пустых клеток

        empty_per_row = np.sum(maze == 0, axis=1)
        std_empty = np.std(empty_per_row)
        if std_empty > coeff_maze * 0.1:
            rating -= std_empty * 2
        else:
            rating += 5

        # Проверка количества элементов
        for elem, expected in self.allowed_elements.items():
            if expected is not None:
                actual = np.sum(maze == elem)
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
            key_pos = np.argwhere(maze == 2)
            exit_pos = np.argwhere(maze == 7)

            if len(key_pos) > 0 and len(exit_pos) > 0:
                dist = np.linalg.norm(key_pos[0] - exit_pos[0])
                if dist < self.size / 3:
                    rating -= 10  # слишком близко
                elif dist > self.size * 0.7:
                    rating += 5  # хорошо разбросано
                    
        key_pos = np.argwhere(maze == 2)
        exit_pos = np.argwhere(maze == 7)
        start_pos = (1, 1)

        if len(key_pos) > 0 and len(exit_pos) > 0:
            key = tuple(key_pos[0])
            exit_ = tuple(exit_pos[0])

            # Функция shortest_path_info возвращает (длина, количество путей)
            start_key_len, start_key_paths = shortest_path_info(maze, start_pos, key)
            key_exit_len, key_exit_paths = shortest_path_info(maze, key, exit_)

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
            print(f"[SAVE] Сохранение лабиринта")
        save_dir = "saved_mazes"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"maze_{self.size}x{self.size}_{timestamp}.csv"
        full_path = os.path.join(save_dir, fname)

        # Сохраняем CSV — можно закомментить, если нужно только метаданные
        # np.savetxt(full_path, self.maze, fmt="%d", delimiter=",")

        # Добавляем путь к maze-файлу в мета-лог
        meta_entry = self.result_maze.copy()
        meta_entry["csv_path"] = full_path
        meta_entry["timestamp"] = timestamp
        
        
        meta_log_path = os.path.join(save_dir, "meta.jsonl")
        with open(meta_log_path, "a") as f:
            f.write(json.dumps(meta_entry, ensure_ascii=False) + "\n")
        if self.verbose:
            print(f"[SAVE] Лабиринт сохранён: {full_path} + метаданные в {meta_log_path}")
        return 0.0, True
      
    def get_dig_action_from_coords(self, next_x, next_y):
        # Текущая позиция курсора
        cur_x, cur_y = self.cursor

        dx = next_x - cur_x
        dy = next_y - cur_y

        for action_idx, (adx, ady) in self.directions.items():
            if (dx, dy) == (adx, ady):
                return action_idx
        # Если не нашли подходящего действия, вернуть например 0 или бросить ошибку
        return 0
    
    def debug_placement_status(self):
        if self.verbose:
            print("[DEBUG] Placement status:")
        for e in self.placeable_elements:
            allowed = self.allowed_elements.get(e, 0)
            placed = self.placed.get(e, 0)
            if self.verbose:
                print(f"Element {e} ({self.elements.get(e)}): placed {placed} / allowed {allowed}")
                
        
    def _stub_evaluate(self, maze):
        try:
            start_pos = (1, 1)
            key_pos = np.argwhere(maze == 2)
            exit_pos = np.argwhere(maze == 7)

            has_key = len(key_pos) > 0
            has_exit = len(exit_pos) > 0
            empty_cells = int(np.sum(maze == 0))

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

            start_key_len, start_key_paths = shortest_path_info(maze, start_pos, key)
            key_exit_len, key_exit_paths = shortest_path_info(maze, key, exit_)

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
            turns = self._estimate_turns_from_path(maze, start_pos, key, exit_)

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
        
    def 
    (self, maze, start, key, exit_):
        def reconstruct_path(a, b):
            # Используем простой BFS путь для оценки поворотов
            from collections import deque
            H, W = maze.shape
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
                    if 0 <= ny < H and 0 <= nx < W and maze[ny, nx] in (0, 2, 7) and not visited[ny, nx]:
                        visited[ny, nx] = True
                        prev[(ny, nx)] = (y, x)
                        q.append((ny, nx))

            # Восстанавливаем путь
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
    
    
    def _init_place_rewards(self):
        start = (1, 1)
        exit_pos_arr = np.argwhere(self.maze == 7)

        if len(exit_pos_arr) == 0:
            self.place_rewards_key = {}
            self.place_rewards_exit = {}
            self.place_rewards_campfire = {}
            self.place_rewards_initialized = True
            return

        exit_pos = tuple(exit_pos_arr[0])
        key_pos_arr = np.argwhere(self.maze == 2)
        key_pos = tuple(key_pos_arr[0]) if len(key_pos_arr) > 0 else None

        main_path = self._get_main_path(start, exit_pos)
        if not main_path:
            self.place_rewards_key = {}
            self.place_rewards_exit = {}
            self.place_rewards_campfire = {}
            self.place_rewards_initialized = True
            return

        path_len = len(main_path)
        self.place_rewards_exit = {}
        self.place_rewards_key = {}
        self.place_rewards_campfire = {}

        # --- Награды / штрафы для выхода ---
        for i, (x, y) in enumerate(main_path):
            dist_to_end = path_len - i - 1
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if self._in_bounds(nx, ny) and self._is_outer_wall(nx, ny):
                    # Проверка достижимости: у стены должен быть сосед == 0
                    has_adjacent_path = False
                    for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nnx, nny = nx + ddx, ny + ddy
                        if self._in_bounds(nnx, nny) and self.maze[nnx, nny] == 0:
                            has_adjacent_path = True
                            break

                    if has_adjacent_path:
                        reward = 20 if dist_to_end == 0 else 20 * (dist_to_end / path_len)
                        self.place_rewards_exit[(nx, ny)] = max(
                            self.place_rewards_exit.get((nx, ny), 0),
                            reward
                        )
                    else:
                        self.place_rewards_exit[(nx, ny)] = -100  # недостижима

        # --- Штрафы за углы (всегда плохое место для выхода) ---
        corners = [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]
        for corner in corners:
            self.place_rewards_exit[corner] = -100

        # --- Награды для ключа ---
        if key_pos is None:
            for i, pos in enumerate(main_path):
                dist_to_end = path_len - i - 1
                self.place_rewards_key[pos] = 15 * (dist_to_end / path_len)
        else:
            self.place_rewards_key = {key_pos: -20}

        # --- Штраф за стартовую точку (ключ) ---
        self.place_rewards_key[start] = -100

        # --- Награды для костра ---
        mid = path_len // 2
        for i, pos in enumerate(main_path):
            dist_to_mid = abs(i - mid)
            self.place_rewards_campfire[pos] = max(0, 10 * (1 - dist_to_mid / mid))

        # --- Штраф за стартовую точку (костёр) ---
        self.place_rewards_campfire[start] = -100

        self.place_rewards_initialized = True

                        
                        
    def _update_place_rewards_after_placement(self, element, pos):
        # Запрещаем повторное размещение на этой позиции
        if element == 7:
            # Выход — внешняя стена, запрещаем там размещать повторно
            self.place_rewards_exit[pos] = -20
        elif element == 2:
            self.place_rewards_key[pos] = -20
        elif element == 5:
            self.place_rewards_campfire[pos] = -20

        # Если ключ размещён — сбросить награды ключа (только запрет на размещение в уже занятом месте)
        if element == 2:
            self.place_rewards_key = {pos: -20}
            
    def _dict_to_map(self, rewards_dict):
        reward_map = np.zeros((self.size, self.size), dtype=np.float32)
        for (x,y), val in rewards_dict.items():
            reward_map[x,y] = val
        # Нормализация к диапазону 0..1 (если нужно)
        max_val = reward_map.max()
        min_val = reward_map.min()
        if max_val > min_val:
            reward_map = (reward_map - min_val) / (max_val - min_val)
        else:
            reward_map = np.zeros_like(reward_map)
        return reward_map
    
    
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