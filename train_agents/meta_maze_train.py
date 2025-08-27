import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from maze_env.maze_env import PPOWithImitationNav
from maze_env.meta_maze import MetaCustomPolicy, MetaMazeEnv
from stable_baselines3 import PPO
from tqdm import tqdm


# ==== Параметры ====
SAVE_PATH = "./navigator/mega_agent"
LOAD_PATH = "./navigator/mega_agent.zip"
PROGRESS_PATH = "meta_iter_progress.json"

TOTAL_ITERATIONS = 1000  # например 100 итераций обучения


def save__progress(iter_num: int) -> None:
    """
    Сохранение номер итерации
    """
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"dagger_iter": iter_num}, f)


def load_progress():
    """
    Загрузка номера последней итерации
    """
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            data = json.load(f)
            return data.get("dagger_iter", 0)
    return 0


def train_meta_agent():
    """
    Функция обучения агента
    """
    start_iter = load_progress()

    # Создание среды
    env = MetaMazeEnv(verbose=0)

    if os.path.exists(LOAD_PATH):
        print("Загружаю существующую модель мета агента")
        model = PPOWithImitationNav.load(
            SAVE_PATH, custom_objects={"policy_class": MetaCustomPolicy}
        )
        model.set_env(env)
    else:
        print("Создаю новую модель мета агнета")
        model = PPO(
            policy=MetaCustomPolicy,
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
            tensorboard_log="./ppo_generator_tensorboard",
        )

    for episode in tqdm(range(start_iter, TOTAL_ITERATIONS), desc="META Training"):
        model.learn(total_timesteps=20480, progress_bar=True)
        model.save(SAVE_PATH)
        save__progress(episode + 1)


train_meta_agent()
