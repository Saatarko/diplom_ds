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

import csv

from tqdm import tqdm


def main():

    EMPTY = 0
    WALL = 1
    KEY = 2
    DOOR = 3
    TRAP = 4
    CAMPFIRE = 5
    EXIT = 7

    def generate_maze(
        width=10,
        height=10,
        start_pos=(1, 1),
        num_traps=4,
        num_campfires=3,
        algo='dfs'  # пока только dfs, но задел под выбор
    ):
        if width % 2 == 0: width += 1
        if height % 2 == 0: height += 1

        maze = np.ones((height, width), dtype=int)  # Стены

        def carve_dfs(x, y):
            dirs = [(2,0), (-2,0), (0,2), (0,-2)]
            random.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 1 <= nx < height-1 and 1 <= ny < width-1 and maze[nx, ny] == WALL:
                    maze[nx, ny] = EMPTY
                    maze[x + dx//2, y + dy//2] = EMPTY
                    carve_dfs(nx, ny)

        # Выбор алгоритма генерации
        if algo == 'dfs':
            x, y = start_pos
            maze[x, y] = EMPTY
            carve_dfs(x, y)
        else:
            raise NotImplementedError(f"Алгоритм '{algo}' ещё не реализован")

        # Добавим полезную функцию выбора случайной пустой клетки
        def place_tile(tile_code, count=1):
            placed = 0
            while placed < count:
                x, y = random.randint(1, height-2), random.randint(1, width-2)
                if maze[x, y] == EMPTY:
                    maze[x, y] = tile_code
                    placed += 1

        place_tile(KEY, 1)
        place_tile(DOOR, 1)
        place_tile(TRAP, num_traps)
        place_tile(CAMPFIRE, num_campfires)

        # Выход
        while True:
            x, y = random.randint(1, height-2), random.randint(1, width-2)
            if maze[x, y] == EMPTY:
                maze[x, y] = EXIT
                break

        return maze

    def save_maze(maze, path="maze/maze.csv"):
        np.savetxt(path, maze, fmt='%d', delimiter=",")


    def save_episode_log(log, path="episode_log.csv"):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "x", "y", "action", "tile", "reward", "health", "has_key"])
            writer.writerows(log)

    class TransformerFeatureExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: spaces.Box, d_model=128, nhead=4, num_layers=2):
            super().__init__(observation_space, features_dim=d_model)

            self.seq_len = observation_space.shape[0]
            self.d_model = d_model

            self.input_linear = nn.Linear(self.seq_len, d_model)

            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.output = nn.Linear(d_model, d_model)

        def forward(self, observations: th.Tensor) -> th.Tensor:
            # Вход [batch, features] -> [batch, seq_len=1, features]
            x = observations.unsqueeze(1)
            x = self.input_linear(x)
            x = self.transformer(x)
            x = x.squeeze(1)
            return self.output(x)

    class CustomTransformerPolicy(ActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                **kwargs,
                features_extractor_class=TransformerFeatureExtractor,
                features_extractor_kwargs=dict(d_model=128, nhead=4, num_layers=2),
            )

    class MazeEnv(gym.Env):
        def __init__(self, maze, start_pos=(1, 1), max_health=100):
            super().__init__()

            self.maze = maze.copy()
            self.start_pos = start_pos
            self.player_pos = list(start_pos)
            self.has_key = False
            self.health = max_health
            self.max_health = max_health
            self.episode_log = []

            self.height, self.width = maze.shape
            self.observation_space = spaces.Dict({
                "view": spaces.Box(low=0, high=7, shape=self.maze.shape, dtype=np.int8),
                "position": spaces.Box(low=0, high=max(self.height, self.width), shape=(2,), dtype=np.int32),
                "has_key": spaces.Discrete(2),
                "health": spaces.Box(low=0, high=max_health, shape=(), dtype=np.int32),
            })

            self.action_space = spaces.Discrete(4)  # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT

        def reset(self, seed=None, options=None):
            self.player_pos = list(self.start_pos)
            self.has_key = False
            self.health = self.max_health
            self.visible_map = np.full_like(self.maze, fill_value=-1, dtype=np.int8)
            self._update_visibility()
            return self._get_obs(), {}

        def _update_visibility(self):
            x, y = self.player_pos
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.height and 0 <= ny < self.width:
                        self.visible_map[nx, ny] = self.maze[nx, ny]

        def step(self, action):
            dx, dy = 0, 0
            if action == 0:
                dx, dy = -1, 0  # UP
            elif action == 1:
                dx, dy = 1, 0  # DOWN
            elif action == 2:
                dx, dy = 0, -1  # LEFT
            elif action == 3:
                dx, dy = 0, 1  # RIGHT

            new_x = self.player_pos[0] + dx
            new_y = self.player_pos[1] + dy

            reward = 0.0
            terminated = False

            if 0 <= new_x < self.height and 0 <= new_y < self.width:
                tile = self.maze[new_x, new_y]
                if tile != 1:  # не стена
                    self.player_pos = [new_x, new_y]

                    if tile == 2:  # ключ
                        self.has_key = True
                        self.maze[new_x, new_y] = 0
                        reward += 1

                    elif tile == 3:  # дверь
                        if self.has_key:
                            pass  # можно пройти
                        else:
                            reward -= 1  # ударился об дверь

                    elif tile == 4:  # ловушка
                        self.health -= 50
                        reward -= 5
                        if self.health <= 0:
                            terminated = True
                            reward -= 100

                    elif tile == 5:  # костёр
                        self.health = min(self.max_health, self.health + 50)

                    elif tile == 7:  # выход
                        if self.has_key:
                            reward += 10
                            terminated = True
                        else:
                            reward -= 1  # нужен ключ

            self._update_visibility()

            obs = self._get_obs()
            self.episode_log.append([
                len(self.episode_log), *self.player_pos, action, tile, reward, self.health, int(self.has_key)
            ])
            return obs, float(reward), bool(terminated), False, {}

        def _get_obs(self):
            return {
                "view": self.visible_map.copy(),  # весь накопленный вид (вся карта)
                "position": np.array(self.player_pos, dtype=np.int32),
                "has_key": int(self.has_key),
                "health": self.health
            }

        def render(self):
            render_maze = self.maze.copy()
            x, y = self.player_pos
            render_maze[x, y] = 9  # агент
            symbols = {
                0: ' ', 1: '█', 2: 'K', 3: 'D',
                4: 'T', 5: 'C', 7: 'E', 9: 'A'
            }
            # print('\n'.join(''.join(symbols.get(cell, '?') for cell in row) for row in render_maze))
            # print(f"Pos: {self.player_pos}  HP: {self.health}  Key: {self.has_key}")

    class FlattenObservationWrapper(ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            view_shape = env.observation_space.spaces["view"].shape
            self.observation_space = Box(
                low=0,
                high=1,
                shape=(view_shape[0] * view_shape[1] + 2 + 2,),  # view + has_key + health + position(2)
                dtype=np.float32,
            )

        def observation(self, obs):
            view_flat = (obs["view"].flatten().astype(np.float32) + 1) / 8.0  # нормализация сдвигом
            has_key = np.array([obs["has_key"]], dtype=np.float32)
            health = np.array([obs["health"] / self.env.max_health], dtype=np.float32)
            position = obs.get("position")
            if position is not None:
                position = position.astype(np.float32) / max(self.env.height, self.env.width)
                return np.concatenate([view_flat, has_key, health, position])
            else:
                return np.concatenate([view_flat, has_key, health])

    NUM_MAZES = 1000
    PARQUET_LOG_FILE = "episode_log.parquet"
    CSV_LOG_FILE = "episode_log.csv"

    all_logs = []  # Для накопления в памяти, можно выгружать периодически, если будет много
    maze = generate_maze(10, 10, num_traps=0, num_campfires=0)
    env = MazeEnv(maze, start_pos=(1, 1))
    env = FlattenObservationWrapper(env)
    check_env(env, warn=True)

    model_path = "maser_trading_v1.zip"

    if os.path.exists(model_path):
        print("Загружаем существующую модель...")
        model = PPO.load(model_path, env=env)
    else:
        print("Создаём новую модель...")
        model = PPO(CustomTransformerPolicy, env, verbose=0, tensorboard_log="./ppo_maze_tensorboard", n_steps=128)

    for i in tqdm(range(NUM_MAZES), desc='Обучение'):
        print(f"=== Maze {i + 1}/{NUM_MAZES} ===")

        # 1. Генерация лабиринта с уникальным id
        maze_id = str(uuid.uuid4())
        maze_filename = f"maze_{maze_id}.csv"
        maze = generate_maze(10, 10, num_traps=0, num_campfires=0)

        # 2. Обертка среды
        env = MazeEnv(maze, start_pos=(1, 1))
        env = FlattenObservationWrapper(env)
        check_env(env, warn=True)

        # 3. Создаём или загружаем модель (если хочешь дообучать одну и ту же, вынеси модель из цикла)
        model.set_env(env)

        # 4. Обучаем
        model.learn(total_timesteps=100_000, progress_bar=True)

        # # 5. Тестируем обученного агента
        # obs, _ = env.reset()
        # done = False
        # step_count = 0
        # max_steps = 1000000
        # max_steps = maze.shape[0] * maze.shape[1]

        # while not done and step_count < max_steps:
        #     action, _ = model.predict(obs)
        #     obs, reward, done, _, _ = env.step(action)
        #     # env.render()  # Можно раскомментировать, если нужен визуальный вывод
        #     step_count += 1
        #
        # if not done:
        #     print(f"Лабиринт {maze_id} НЕ ПРОЙДЕН")
        #     env.unwrapped.episode_log.append([
        #         step_count, *env.unwrapped.player_pos, -1, -1, -999, env.unwrapped.health, int(env.unwrapped.has_key)
        #     ])

        # 6. Сохраняем лабиринт и логи
        # save_maze(maze, maze_filename)

        # Сохраняем текущий лог в CSV (перезаписываем каждый раз)
        # save_episode_log(env.unwrapped.episode_log, CSV_LOG_FILE)

        # Сохраняем модель
        model.save("maser_trading_v1")

        # # После каждого лабиринта:
        # df_one = pd.DataFrame([
        #     {
        #         "maze_id": maze_id,
        #         "step_idx": entry[0],
        #         "x": entry[1],
        #         "y": entry[2],
        #         "action": entry[3],
        #         "tile": entry[4],
        #         "reward": entry[5],
        #         "health": entry[6],
        #         "has_key": entry[7],
        #     }
        #     for entry in env.unwrapped.episode_log
        # ])
        #
        # df_one.to_parquet(PARQUET_LOG_FILE, engine="pyarrow", index=False, append=True)



if __name__ == "__main__":
    main()