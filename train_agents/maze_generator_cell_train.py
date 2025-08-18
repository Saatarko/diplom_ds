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
import json
import time
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
from sb3_contrib.common.wrappers import ActionMasker

from maze_env.maze_build_cell import MazeBuilderEnvDFSCell, CustomTransformerPolicyForBuilder, PPOWithImitationCell

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


set_seed(42)
LOAD_PATH = "./generator/generator_agent_dagger_9.zip"
SAVE_PATH = "./generator/generator_agent_dagger_9"
NAVIGATOR_PATH = "./navigator/ppo_maze_agent_v4"

PROGRESS_PATH = "dagger_progress.json"

TOTAL_DAGGER_ITER = TOTAL_ITER = 1200
STEPS_PER_ITER = 10000  # количество взаимодействий в среде
IMITATION_BATCH_SIZE = 32
ROLLOUTS_PER_ITER = 10
beta_0 = 1.0


def preprocess_obs(obs):
    """Приводим obs к float32 и нормализуем числовые поля."""
    obs_float = {}
    for k, v in obs.items():
        arr = np.array(v, dtype=np.float32)  # float32
        # Масштабируем поля с большими числами
        if k in ["rating"]:
            arr /= 100.0
        elif k in ["steps", "jump_interval"]:
            arr /= 1000.0
        obs_float[k] = arr
    return obs_float


def run_rollout(env, model, max_total_steps=400):
    """
    Запуск rollout для DAgger с автоматическим управлением фазами.
    
    env          : MazeBuilderEnvDFS
    model        : агент (PPOWithImitation)
    max_total_steps : максимальное количество шагов на весь rollout
    """
    obs_lists = {k: [] for k in env.get_obs().keys()}
    actions_list = []
    expert_actions_list = []

    obs, _ = env.reset()

    for step in range(max_total_steps):
        # Преобразуем наблюдения для модели (float, нормализация и т.д.)
        obs_float = preprocess_obs(obs)

        # Агент выбирает действие
        action, _ = model.predict(obs_float, deterministic=False)

        # Эксперт выбирает действие
        expert_action = env.get_expert_action()

        # Сохраняем данные
        for k, v in obs.items():
            obs_lists[k].append(np.array(v))  # сохраняем оригинальные данные
        actions_list.append(int(action))
        expert_actions_list.append(int(expert_action))

        # Выполняем шаг в среде
        obs, reward, done, truncated, info = env.step(int(action))

        if done:
            break

    # Преобразуем в тензоры для обучения
    obs_batch = {k: torch.tensor(np.stack(v), dtype=torch.float32) for k, v in obs_lists.items()}
    expert_actions_tensor = torch.tensor(expert_actions_list, dtype=torch.long).flatten()

    return obs_batch, expert_actions_tensor



def dagger_training_loop(env, model, start_iter=0):
    all_expert_obs = None
    all_expert_actions = None

    for i in tqdm(range(start_iter, TOTAL_DAGGER_ITER), desc="DAgger Training"):
        beta_i = beta_0 * np.exp(-i / (TOTAL_DAGGER_ITER / 2))
        model.imitation_coef = beta_i
        print(f"\n[DAgger] Итерация {i+1}/{TOTAL_DAGGER_ITER}, imitation_coef = {beta_i:.4f}")

        batch_obs = None
        batch_actions = None

        for _ in range(ROLLOUTS_PER_ITER):
            obs_batch, expert_actions_tensor = run_rollout(env, model)

            # Объединяем с текущим batch
            if batch_obs is None:
                batch_obs = obs_batch
                batch_actions = expert_actions_tensor
            else:
                batch_obs = {k: torch.cat([batch_obs[k], obs_batch[k]], dim=0) for k in batch_obs}
                batch_actions = torch.cat([batch_actions, expert_actions_tensor], dim=0)

        # Объединяем со всеми предыдущими данными
        if all_expert_obs is None:
            all_expert_obs = batch_obs
            all_expert_actions = batch_actions
        else:
            all_expert_obs = {k: torch.cat([all_expert_obs[k], batch_obs[k]], dim=0) for k in all_expert_obs}
            all_expert_actions = torch.cat([all_expert_actions, batch_actions], dim=0)

        print(f"[DAgger] Текущий размер датасета: {len(all_expert_actions)} примеров")

        # Передаем новые данные модели
        model.set_expert_data(all_expert_obs, all_expert_actions)

        # Обучаем модель
        model.learn(total_timesteps=STEPS_PER_ITER)

        # Дополнительные проходы (опционально)
        for _ in range(5):
            model.train()

        # Сохраняем модель и прогресс
        model.save(SAVE_PATH)
        save_dagger_progress(i + 1)
        
# def train_generator_agent_with_dagger(dagger=False):
    
#     env = MazeBuilderEnvDFSCell(size=7, verbose =0, use_stub_eval=True)
#     env = ActionMasker(env, lambda env: env.get_action_mask())

#     start_iter = load_dagger_progress()

#     if os.path.exists(LOAD_PATH):
#         print("Загружаю существующую модель генератора")
#         model = PPOWithImitationCell.load(LOAD_PATH, custom_objects={"policy_class": CustomTransformerPolicyForBuilder})
#         model.set_env(env)
#     else:
#         print("Создаю новую модель генератора")
#         model = PPOWithImitationCell(
#             policy=CustomTransformerPolicyForBuilder,
#             env=env,
#             n_steps=20480,
#             batch_size=256,
#             learning_rate=2.5e-4,
#             ent_coef=0.01,
#             vf_coef=0.5,
#             clip_range=0.2,
#             gae_lambda=0.95,
#             gamma=0.99,
#             verbose=1,
#             tensorboard_log="./ppo_generator_tensorboard",
#             imitation_coef=1.0,
#         )
#         start_iter = 0
    
#     if dagger:
#         dagger_training_loop(env, model, start_iter)
#     else:
#         for i in tqdm(range(start_iter, TOTAL_ITER),desc="Обучение"):
#             beta_i = beta_0 * np.exp(-i / (TOTAL_DAGGER_ITER / 2))
#             model.imitation_coef = beta_i
#             print(f"\n[{i+1}/{TOTAL_ITER}] Итерация обучения генератора")
#             model.learn(total_timesteps=10240, progress_bar=True)
#             model.save(SAVE_PATH)
#             save_dagger_progress(i + 1)
            
            

    
def train_generator_agent_with_dagger(dagger=False, adapt=False):
    env = MazeBuilderEnvDFSCell(size=7, verbose=0, use_stub_eval=True)
    env = ActionMasker(env, lambda env: env.get_action_mask())

    start_iter = load_dagger_progress()

    if os.path.exists(LOAD_PATH):
        print("Загружаю существующую модель генератора")
        model = PPOWithImitationCell.load(LOAD_PATH, custom_objects={"policy_class": CustomTransformerPolicyForBuilder})
        model.set_env(env)
    else:
        print("Создаю новую модель генератора")
        model = PPOWithImitationCell(
            policy=CustomTransformerPolicyForBuilder,
            env=env,
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
        start_iter = 0

    if dagger:
        dagger_training_loop(env, model, start_iter)
    else:
        if adapt and start_iter < 50:  # адаптация только если мы еще не начали или почти не начали обучение
            adaptation_iters = 50 - start_iter
            steps_per_iter = 10240

            original_ent_coef = model.ent_coef
            model.ent_coef = 0.02  # чуть повысить исследование на адаптации

            for i in range(adaptation_iters):
                print(f"\n[ADAPT {start_iter + i + 1}/10] Дообучение в новой среде")
                model.learn(total_timesteps=steps_per_iter, progress_bar=True)
                model.save(SAVE_PATH)
                save_dagger_progress(start_iter + i + 1)

            model.ent_coef = original_ent_coef
            start_iter += adaptation_iters

        # Основное обучение с оставшимися итерациями
        for i in tqdm(range(start_iter, TOTAL_ITER), desc="Обучение"):
            beta_i = beta_0 * np.exp(-i / (TOTAL_DAGGER_ITER / 2))
            model.imitation_coef = beta_i
            print(f"\n[{i + 1}/{TOTAL_ITER}] Итерация обучения генератора")
            model.learn(total_timesteps=10240, progress_bar=True)
            model.save(SAVE_PATH)
            save_dagger_progress(i + 1)

    print("Обучение завершено")
    
    
train_generator_agent_with_dagger(dagger=False, adapt=True)