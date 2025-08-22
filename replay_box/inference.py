import os
import csv
from typing import Any

import numpy as np
from datetime import datetime
from sb3_contrib.common.wrappers import ActionMasker


from maze_env.maze_build_cell import MazeBuilderEnvDFSCell, PPOWithImitationCell, \
    CustomTransformerPolicyForBuilder
from maze_env.maze_env import PPOWithImitationNav, CustomTransformerPolicy, MazeEnv


def generate_and_infer_best_maze(generator_model_path: str,
                                 navigator_model_path: str,
                                 size = 15,
                                 maze_dir: str = "result/mazes",
                                 log_dir: str = "result/logs",
                                 num_valid: int = 30,
                                 max_gen_steps: int = None,
                                 nav_runs: int = 30,
                                 max_nav_steps: int = 1000)-> tuple[Any, list[Any] | None]:
    """
    Генерация лабиринтов + выбор лучшего + инференс навигатора.
    """

    os.makedirs(maze_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # === Шаг 1. Генератор ===
    print(f"[INFO] Загружаем модель генератора: {generator_model_path}")
    generator_env = MazeBuilderEnvDFSCell(size=size, verbose=0, use_stub_eval=True)
    generator_env = ActionMasker(generator_env, lambda env: env.get_action_mask())
    generator_model = PPOWithImitationCell.load(
        generator_model_path,
        custom_objects={"policy_class": CustomTransformerPolicyForBuilder}
    )
    generator_model.set_env(generator_env)

    valid_mazes = []
    attempt_count = 0

    while len(valid_mazes) < num_valid:
        attempt_count += 1
        obs, _ = generator_env.reset()
        done, step_count = False, 0
        limit = max_gen_steps or (generator_env.size * generator_env.size * 2)

        while not done and step_count < limit:
            mask = generator_env.get_action_mask()
            action, _ = generator_model.predict(obs, action_masks=mask, deterministic=False)
            if not mask[action]:
                allowed_actions = np.where(mask)[0]
                action = int(allowed_actions[0])
            obs, reward, done, _, _ = generator_env.step(action)
            step_count += 1

        maze = generator_env.unwrapped.layout.copy()
        meta = generator_env.unwrapped.result_maze.copy()
        meta["attempt"] = attempt_count

        if meta.get("success") and meta.get("has_key") and meta.get("has_exit") and meta.get("turns", 0) > 3:
            valid_mazes.append((maze, meta))
            print(f"[INFO] Валидный лабиринт {len(valid_mazes)}/{num_valid}")
        else:
            print(f"[INFO] Лабиринт не прошёл фильтр (попытка {attempt_count})")

    # === Шаг 2. Выбор лучшего лабиринта ===
    best_maze, best_meta = max(valid_mazes, key=lambda m: m[1].get("rating", 0))
    print(f"[INFO] Выбран лабиринт с рейтингом {best_meta.get('rating')}")

    # === Шаг 3. Навигатор ===
    print(f"[INFO] Загружаем модель навигатора: {navigator_model_path}")
    nav_model = PPOWithImitationNav.load(
        navigator_model_path,
        custom_objects={"policy_class": CustomTransformerPolicy}
    )

    best_run_logs = None
    best_steps = float("inf")

    for run_id in range(nav_runs):
        base_env = MazeEnv(best_maze.copy(), verbose=0)
        obs, _ = base_env.reset()
        logs = []

        for step in range(max_nav_steps):
            action, _ = nav_model.predict(obs, deterministic=False)
            action = int(action)
            obs, reward, terminated, truncated, info = base_env.step(action)
            done = terminated or truncated

            x, y = base_env.cursor_x, base_env.cursor_y
            tile = base_env.layout[y, x]
            health = base_env.health
            has_key = int(base_env.KEY_ID in base_env.inventory)
            logs.append([step, x, y, action, tile, reward, health, has_key])

            if done:
                break

        if len(logs) < best_steps:
            best_steps = len(logs)
            best_run_logs = logs
            print(f"[INFO] Новый лучший результат: {best_steps} шагов (run {run_id+1})")

    # === Шаг 4. Сохранение ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    maze_fname = f"best_maze_{timestamp}.csv"
    log_fname = f"best_run_{timestamp}.csv"

    maze_path = os.path.join(maze_dir, maze_fname)
    log_path = os.path.join(log_dir, log_fname)

    np.savetxt(maze_path, np.array(best_maze, dtype=np.int32), fmt="%d", delimiter=",")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x", "y", "action", "tile", "reward", "health", "has_key"])
        writer.writerows(best_run_logs)

    print(f"[INFO] Лучший лабиринт сохранён в {maze_path}")
    print(f"[INFO] Лог лучшего прохождения сохранён в {log_path}")

    # === Шаг 5. Возврат данных ===
    return best_maze, best_run_logs
