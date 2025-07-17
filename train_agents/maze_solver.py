import os
import uuid
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


import csv

from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from maze_env.maze_env import MazeEnv, CustomTransformerPolicy, ResetMemoryCallback


def train_agent_on_mazes(mazes, model_path, num_episodes=1000):

    # Загрузка/создание модели
    if os.path.exists('./ppo_maze_agent_v3.zip'):
        print("Загружаю модель")
        model = PPO.load(model_path)
    else:
        print("Создаю новую модель")
        env = MazeEnv(mazes[0], start_pos=(1, 1))
        model = PPO(
            policy=CustomTransformerPolicy,
            env=env,
            n_steps=1024,
            batch_size=64,
            learning_rate=2.5e-4,
            ent_coef=0.1,
            vf_coef=0.5,
            clip_range=0.2,
            gae_lambda=0.95,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./ppo_maze_tensorboard"
        )

    reset_memory_cb = ResetMemoryCallback()

    for episode in range(num_episodes):

        maze_idx = episode % len(mazes)
        maze = mazes[maze_idx]
        print(f"[{episode+1}] Используется предобученный лабиринт {maze_idx+1}")

        env = MazeEnv(maze, start_pos=(1, 1))
        model.set_env(env)

        model.learn(total_timesteps=10240, progress_bar=True, callback=reset_memory_cb)

        model.save(model_path)
        print(f"Модель сохранена после эпизода {episode+1}")


def evaluate_agent(agent, maze_list, max_steps_dict, success_threshold=0.9):
    """
    Оценивает агента на списке лабиринтов.

    Args:
        agent: обученный PPO агент
        maze_list: список Maze объектов
        max_steps_dict: словарь {maze_size: max_steps}
        success_threshold: минимальная доля успешных эпизодов (например, 0.9)

    Returns:
        (bool: прошел ли порог, dict: подробная статистика)
    """
    successes = 0
    step_counts = []
    failed_mazes = []

    for i, maze in enumerate(tqdm(maze_list, desc="Evaluating")):
        size = maze.shape[0]
        max_steps = max_steps_dict.get(size, size * size)  # по умолчанию N*N

        env = MazeEnv(maze=maze, max_steps=max_steps)
        vec_env = DummyVecEnv([lambda: env])

        obs = vec_env.reset()
        agent.policy.reset_memory()  # сбрасываем память трансформера

        for step in range(max_steps):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            if done[0]:
                break

        final_info = info[0]
        success = final_info.get("success", False)
        step_counts.append(step + 1)
        if success:
            successes += 1
        else:
            failed_mazes.append(i)

    success_rate = successes / len(maze_list)
    passed = success_rate >= success_threshold

    print(f"\n✅ Success rate: {successes}/{len(maze_list)} = {success_rate * 100:.1f}%")
    if not passed:
        print(f"❌ Недостаточно успешных проходов. Провалено: {failed_mazes}")

    return passed, {
        "success_rate": success_rate,
        "step_counts": step_counts,
        "failed_ids": failed_mazes
    }
