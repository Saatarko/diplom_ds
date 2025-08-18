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
import json

import csv

from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from maze_env.maze_env import MazeEnv, CustomTransformerPolicy, ResetMemoryCallback, FullMapExpert, PartialMapExpert, PPOWithImitationNav


class VecEnvWrapper4to5:
    """Обёртка для DummyVecEnv, чтобы step() возвращал 5 значений."""
    def __init__(self, venv):
        self.venv = venv

    def reset(self):
        obs = self.venv.reset()
        return obs, {}  # добавляем пустой info

    def step(self, action):
        obs, reward, done, info = self.venv.step(action)
        terminated = done
        truncated = np.zeros_like(done, dtype=bool)
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.venv, name)

def load_mazes_from_folder(folder_path="learning_mazes"):
    maze_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    mazes = []
    for fname in maze_files:
        full_path = os.path.join(folder_path, fname)
        maze_df = pd.read_csv(full_path, header=None)
        maze_array = maze_df.to_numpy(dtype=np.int32)
        mazes.append(maze_array)
    print(f"Загружено {len(mazes)} лабиринтов из {folder_path}")
    return mazes

def save_dagger_progress(iter_num):
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"dagger_iter": iter_num}, f)

def load_dagger_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            data = json.load(f)
            return data.get("dagger_iter", 0)
    return 0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
# ==== Параметры ====
PROGRESS_PATH = "dagger_nav_progress.json"
TOTAL_DAGGER_ITER=TOTAL_ITER =1000
BETA_0 = 1.0
set_seed(42)
LOAD_PATH = "./navigator/navigator_agent_5.zip"
SAVE_PATH = "./navigator/navigator_agent_5"

mazes = load_mazes_from_folder("learning_mazes")

def train_agent_on_mazes(mazes, dagger=False):
    start_iter = load_dagger_progress()

    if os.path.exists(LOAD_PATH):
        print("Загружаю модель")
        base_env = MazeEnv(mazes[0], verbose=0)
        model = PPOWithImitationNav.load(SAVE_PATH, custom_objects={"policy_class": CustomTransformerPolicy})
    else:
        print("Создаю новую модель")
        base_env = MazeEnv(mazes[0], verbose=0)
        model = PPOWithImitationNav(
            policy=CustomTransformerPolicy,
            env=base_env,
            n_steps=20480,
            batch_size=256,
            learning_rate=2.5e-4,
            ent_coef=0.01,
            vf_coef=0.5,
            clip_range=0.2,
            gae_lambda=0.95,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./ppo_generator_tensorboard",
            imitation_coef=1.0,
        )

    reset_memory_cb = ResetMemoryCallback()

    if not dagger:
        for episode in tqdm(range(start_iter, TOTAL_DAGGER_ITER), desc="Casual Training"):
            maze = mazes[episode % len(mazes)]
            base_env = MazeEnv(maze, verbose=0)
            obs, _ = base_env.reset()
            done = False
            t = 0
            max_t = base_env.max_steps

            while not done and t < max_t:
                agent_action, _ = model.predict(obs, deterministic=False)
                agent_action = int(agent_action)

                obs, reward, terminated, truncated, info = base_env.step(agent_action)
                done = terminated or truncated
                t += 1

            # Обучение PPO на собранных траекториях среды
            model.set_env(base_env)
            model.learn(total_timesteps=20480, progress_bar=True, callback=reset_memory_cb)
            model.save(SAVE_PATH)
            save_dagger_progress(episode + 1)
    else:
        for episode in tqdm(range(start_iter, TOTAL_DAGGER_ITER), desc="DAgger Training"):
            maze = mazes[episode % len(mazes)]
            print(f"[{episode+1}] DAgger на лабиринте {episode % len(mazes) + 1}")

            # env для агента
            train_env = DummyVecEnv([lambda m=maze: MazeEnv(m, verbose=0)])
            model.set_env(train_env)

            # эксперт читает ровно тот же env
            base_env = train_env.envs[0]
            expert = FullMapExpert(base_env) if episode < TOTAL_DAGGER_ITER // 2 else PartialMapExpert(base_env)
            expert.reset()

            # rollout с мешапом действий
            obs = train_env.reset()
            done = np.array([False])
            t, max_t = 0, base_env.max_steps

            while not done.all() and t < max_t:
                beta_i = BETA_0 * np.exp(-episode / (TOTAL_DAGGER_ITER / 2))
                expert_action = np.array([expert.get_action()], dtype=np.int64)
                agent_action, _ = model.predict(obs, deterministic=False)
                agent_action = agent_action.astype(np.int64)

                action = np.where(np.random.rand(*agent_action.shape) < beta_i, expert_action, agent_action)
                obs, rewards, dones, infos = train_env.step(action)
                done = np.array(dones, dtype=bool)
                t += 1

            if t >= max_t:
                print("[WARN] DAgger rollout достиг лимита шагов.")

            # обучение PPO на собранных траекториях среды
            model.learn(total_timesteps=20480, progress_bar=True, callback=reset_memory_cb)
            model.save(SAVE_PATH)
            save_dagger_progress(episode + 1)



train_agent_on_mazes(mazes, dagger= False)