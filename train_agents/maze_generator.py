import os
import uuid
import pandas as pd
import numpy as np
import random
import gymnasium as gym
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


import csv

from tqdm import tqdm

# Пути для лабиринтов и модели
maze_folder = "./mazes"
os.makedirs(maze_folder, exist_ok=True)



def generate_maze(
    width=10,
    height=10,
    start_pos=(1, 1),
    num_traps=0,
    num_campfires=0,
    algo='dfs'
):
    EMPTY = 0
    WALL = 1
    KEY = 2
    DOOR = 3
    TRAP = 4
    CAMPFIRE = 5
    EXIT = 7

    if width % 2 == 0: width += 1
    if height % 2 == 0: height += 1

    maze = np.ones((height, width), dtype=int)  # Всё сначала стены

    # ==== 1. Алгоритм carve DFS ====
    def carve_dfs(x, y):
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 1 <= nx < height - 1 and 1 <= ny < width - 1 and maze[nx, ny] == 1:
                maze[nx, ny] = 0
                maze[x + dx // 2, y + dy // 2] = 0
                carve_dfs(nx, ny)

    # ==== 2. Начинаем с начальной позиции ====
    if algo == 'dfs':
        x, y = start_pos
        maze[x, y] = 0
        carve_dfs(x, y)
    else:
        raise NotImplementedError(f"Алгоритм '{algo}' ещё не реализован")

    # ==== 3. Утилита проверки тупиков ====
    def is_dead_end(x, y):
        empty_neighbors = 0
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and maze[nx, ny] == 0:
                empty_neighbors += 1
        return empty_neighbors <= 1

    # ==== 4. Размещение тайлов (ключ, ловушки и т.д.) ====
    def place_tile(tile_code, count=1, avoid_dead_ends=True):
        placed = 0
        attempts = 0
        max_attempts = 500

        while placed < count and attempts < max_attempts:
            x, y = random.randint(1, height - 2), random.randint(1, width - 2)
            if maze[x, y] == 0:
                if avoid_dead_ends and is_dead_end(x, y):
                    attempts += 1
                    continue
                maze[x, y] = tile_code
                placed += 1

        if placed < count:
            print(f"[!] Предупреждение: удалось разместить только {placed} из {count} тайлов {tile_code}")

    place_tile(KEY, 1)
    # place_tile(DOOR, 1)  # Дверь отключена, если не нужна
    place_tile(TRAP, num_traps)
    place_tile(CAMPFIRE, num_campfires)

    # ==== 5. Размещение ВЫХОДА (EXIT) в стену на границе ====
    edge_candidates = []

    for i in range(1, height - 1):
        if maze[i, 1] == 0:
            edge_candidates.append((i, 0))
        if maze[i, width - 2] == 0:
            edge_candidates.append((i, width - 1))

    for j in range(1, width - 1):
        if maze[1, j] == 0:
            edge_candidates.append((0, j))
        if maze[height - 2, j] == 0:
            edge_candidates.append((height - 1, j))

    if edge_candidates:
        ex, ey = random.choice(edge_candidates)
        maze[ex, ey] = EXIT
    else:
        print("[!] Не удалось разместить выход в стене!")

    return maze


def custom_load_or_generate(maze_size):
    mazes = []
    for i in range(1, 101):
        path = os.path.join(maze_folder, f"maze_{i}.csv")
        if os.path.exists(path):
            print(f"Загружаю лабиринт {path}")
            maze = np.loadtxt(path, delimiter=",", dtype=np.int8)
        else:
            print(f"Генерирую лабиринт {path}")
            maze = generate_maze(maze_size, maze_size, num_traps=0, num_campfires=1)
            custom_save_maze(maze, path)
        mazes.append(maze)
    return mazes


def custom_save_maze(maze, path):
    np.savetxt(path, maze, fmt="%d", delimiter=",")


def save_episode_log(log, path="episode_log.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x", "y", "action", "tile", "reward", "health", "has_key"])
        writer.writerows(log)


#
# if __name__ == "__main__":
#     train_agent_on_mazes()