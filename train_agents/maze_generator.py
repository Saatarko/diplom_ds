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
from stable_baselines3.common.utils import obs_as_tensor
import copy
import csv
import pickle
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from maze_env.maze_build_env import MazeBuilderEnv, CustomTransformerPolicyForBuilder, flatten_trajectories, load_demonstrations, PPOWithImitation

# ==== Параметры ====
SAVE_PATH = "./generator/generator_agent.zip"
LOG_CSV = "./generator/generator_log.csv"
NAVIGATOR_PATH = "./navigator/ppo_maze_agent_v4"


CHECKPOINT_DIR = "./generator/builder_checkpoints"
TOTAL_ITERATIONS = 1000  # например 100 итераций обучения
STEPS_PER_ITER = 10000  # сколько шагов на итерацию

def train_generator_agent():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Создание среды
    env = MazeBuilderEnv(size=5, save_dir="generated_mazes")

    # Загрузка или новая модель
    if os.path.exists(SAVE_PATH):
        print("Загружаю существующую модель генератора")
        model = PPO.load(SAVE_PATH)
        model.set_env(env)
    else:
        print("Создаю новую модель генератора")
        model = PPO(
            policy=CustomTransformerPolicyForBuilder,
            env=env,
            n_steps=2048,
            batch_size=64,
            learning_rate=2.5e-4,
            ent_coef=0.01,
            vf_coef=0.5,
            clip_range=0.2,
            gae_lambda=0.95,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./ppo_generator_tensorboard"
        )

    for i in range(TOTAL_ITERATIONS):
        print(f"\n[{i+1}/{TOTAL_ITERATIONS}] Итерация обучения генератора")
        model.learn(total_timesteps=10000, progress_bar=True)
        model.save(SAVE_PATH)

        

if __name__ == "__main__":
    train_generator_agent()
    