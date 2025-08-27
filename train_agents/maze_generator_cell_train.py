import json
import os
import random
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
from maze_env.maze_build_cell import (
    CustomTransformerPolicyForBuilder,
    MazeBuilderEnvDFSCell,
    PPOWithImitationCell,
)
from sb3_contrib.common.wrappers import ActionMasker
from tqdm import tqdm


def load_partial_model_for_new_size(
    model_class: Type[Any],  # тип класса модели, например PPO, MaskablePPO и т.д.
    checkpoint_path: str,
    env: MazeBuilderEnvDFSCell,
    custom_objects: Optional[dict[str, Any]] = None,
    device: str = "cpu"
) -> Any:  # возвращает экземпляр модели
    """
    Загружает старую модель в новую, переносит только слои совпадающие по размеру.
    """
    custom_objects = custom_objects or {}
    policy_class = custom_objects.get("policy_class", "MlpPolicy")

    print(f"Создаём новую модель для env: {env}")
    model = model_class(policy=policy_class, env=env, verbose=1, device=device)

    if not os.path.exists(checkpoint_path):
        print("Чекпоинт не найден, создаём новую модель")
        return model

    print(f"Загружаю состояние старой модели из {checkpoint_path}")
    pretrained = model_class.load(
        checkpoint_path, custom_objects=custom_objects, device=device
    )

    pretrained_state = pretrained.policy.state_dict()
    current_state = model.policy.state_dict()

    # Копируем только совпадающие слои
    copied_layers = []
    skipped_layers = []
    for name, param in pretrained_state.items():
        if name in current_state and param.shape == current_state[name].shape:
            current_state[name].copy_(param)
            copied_layers.append(name)
        else:
            skipped_layers.append(name)

    model.policy.load_state_dict(current_state)
    return model

    print(f"Скопировано слоёв: {len(copied_layers)}")
    print(f"Пропущено слоёв (из-за несовпадения размерностей): {len(skipped_layers)}")
    if skipped_layers:
        print("Пропущенные слои:", skipped_layers)

    return model


def save_dagger_progress(iter_num: int):
    """Сохранение последней итерации"""
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"dagger_iter": iter_num}, f)


def load_dagger_progress() -> int | Any:
    """Загрузка последней итерации"""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            data = json.load(f)
            return data.get("dagger_iter", 0)
    return 0


def set_seed(seed=42):
    """Контролируем все варианты сидов"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==== Параметры ====


set_seed(42)
LOAD_PATH = "./generator/generator_agent_15.zip"
SAVE_PATH = "./generator/generator_agent_15"
NAVIGATOR_PATH = "./navigator/ppo_maze_agent_v4"

PROGRESS_PATH = "dagger_progress.json"

TOTAL_DAGGER_ITER = TOTAL_ITER = 1200
STEPS_PER_ITER = 10000  # количество взаимодействий в среде
IMITATION_BATCH_SIZE = 32
ROLLOUTS_PER_ITER = 10
beta_0 = 1.0


def preprocess_obs(obs: Dict) -> dict[Any, Any]:
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


def run_rollout(env: MazeBuilderEnvDFSCell, model: PPOWithImitationCell, max_total_steps=400):
    """
    Запуск rollout для DAgger с автоматическим управлением фазами.

    env          : MazeBuilderEnvDFSCell
    model        : агент (PPOWithImitationCell)
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
    obs_batch = {
        k: torch.tensor(np.stack(v), dtype=torch.float32) for k, v in obs_lists.items()
    }
    expert_actions_tensor = torch.tensor(
        expert_actions_list, dtype=torch.long
    ).flatten()

    return obs_batch, expert_actions_tensor


def dagger_training_loop(env: MazeBuilderEnvDFSCell, model:PPOWithImitationCell , start_iter: int=0):
    """
    Функция применения DAgger .

    env          : MazeBuilderEnvDFSCell
    model        : агент (PPOWithImitationCell)
    start_iter : начальная итерация
     """
    all_expert_obs = None
    all_expert_actions = None

    for i in tqdm(range(start_iter, TOTAL_DAGGER_ITER), desc="DAgger Training"):
        beta_i = beta_0 * np.exp(-i / (TOTAL_DAGGER_ITER / 2))
        model.imitation_coef = beta_i
        print(
            f"\n[DAgger] Итерация {i+1}/{TOTAL_DAGGER_ITER}, imitation_coef = {beta_i:.4f}"
        )

        batch_obs = None
        batch_actions = None

        for _ in range(ROLLOUTS_PER_ITER):
            obs_batch, expert_actions_tensor = run_rollout(env, model)

            # Объединяем с текущим batch
            if batch_obs is None:
                batch_obs = obs_batch
                batch_actions = expert_actions_tensor
            else:
                batch_obs = {
                    k: torch.cat([batch_obs[k], obs_batch[k]], dim=0) for k in batch_obs
                }
                batch_actions = torch.cat([batch_actions, expert_actions_tensor], dim=0)

        # Объединяем со всеми предыдущими данными
        if all_expert_obs is None:
            all_expert_obs = batch_obs
            all_expert_actions = batch_actions
        else:
            all_expert_obs = {
                k: torch.cat([all_expert_obs[k], batch_obs[k]], dim=0)
                for k in all_expert_obs
            }
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


def train_generator_agent_with_dagger_safe(
    dagger=False, adapt=False, size=7, resize=False
):
    """
       Функция обучения.

    dagger=False,  - прменяется ли даггер
    adapt=False,   - применяется ли адаптеция
    size=7,    - размер лабиринта
    resize=False  - в текущей фазе идет изменение размера лабирнта
    """
    env = MazeBuilderEnvDFSCell(size=size, verbose=0, use_stub_eval=True)
    env = ActionMasker(env, lambda env: env.get_action_mask())
    print(f"Продолжение обучения генерации лабиринтов размера {size}")

    start_iter = load_dagger_progress()

    # --- Загружаем или создаем модель ---
    if os.path.exists(LOAD_PATH):
        if not resize and start_iter != 0:
            print("Идет обучение — продолжаем")
            model = PPOWithImitationCell.load(
                LOAD_PATH,
                custom_objects={"policy_class": CustomTransformerPolicyForBuilder},
                device="cuda",
            )
            model.set_env(env)

            # Проверка весов на NaN/Inf
            nan_found = False
            for name, param in model.policy.state_dict().items():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"[WARNING] NaN/Inf в слое {name}, сбрасываем веса")
                    nan_found = True
                    torch.nn.init.normal_(param.data, 0.0, 0.01)
            if nan_found:
                print("[INFO] Веса очищены, модель готова к обучению")

        else:
            print("Новый размер среды — загружаем модель частично")
            model = load_partial_model_for_new_size(
                PPOWithImitationCell,
                LOAD_PATH,
                env,
                custom_objects={"policy_class": CustomTransformerPolicyForBuilder},
                device="cuda",
            )
            model.set_env(env)

    else:
        print("Создаю новую модель генератора")
        model = PPOWithImitationCell(
            policy=CustomTransformerPolicyForBuilder,
            env=env,
            n_steps=10240,
            batch_size=(
                512 if size >= 15 else 256
            ),  # увеличиваем batch_size для больших лабиринтов
            learning_rate=2.5e-4,
            ent_coef=0.01,
            vf_coef=0.5,
            clip_range=0.2,
            gae_lambda=0.95,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./ppo_generator_tensorboard",
            imitation_coef=1.0,
            max_grad_norm=0.5,
        )
        start_iter = 0

    # --- Фаза адаптации для новых размеров ---
    if adapt or (resize or size >= 15 and start_iter == 0):
        adaptation_iters = 20  # можно увеличить при больших лабиринтах
        steps_per_iter = 10240
        original_ent_coef = model.ent_coef
        model.ent_coef = 0.02  # больше энтропии для разогрева

        print(
            f"\n[INFO] Запуск адаптации ({adaptation_iters} итераций) для {size}x{size}"
        )
        for i in range(adaptation_iters):
            env.reset()
            print(
                f"[ADAPT {start_iter + i + 1}/{start_iter + adaptation_iters}] Дообучение"
            )
            model.learn(total_timesteps=steps_per_iter, progress_bar=True)
            model.save(SAVE_PATH)
            save_dagger_progress(start_iter + i + 1)

        model.ent_coef = original_ent_coef
        start_iter += adaptation_iters

    # --- Основное обучение ---
    for i in tqdm(range(start_iter, TOTAL_ITER), desc="Обучение"):
        env.reset()

        # Затухание imitation_coef
        beta_i = beta_0 * np.exp(-i / (TOTAL_DAGGER_ITER / 2))
        model.imitation_coef = beta_i
        print(f"\n[{i + 1}/{TOTAL_ITER}] Итерация обучения генератора")

        model.learn(total_timesteps=20480, progress_bar=True)

        # Сохраняем прогресс
        model.save(SAVE_PATH)
        save_dagger_progress(i + 1)

    print("Обучение завершено")
    save_dagger_progress(0)


# Запуск безопасного продолжения обучения
train_generator_agent_with_dagger_safe(dagger=False, adapt=False, size=15, resize=False)
