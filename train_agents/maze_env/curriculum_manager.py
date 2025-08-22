import json
import os

import numpy as np
from agents.maze_generator import custom_load_or_generate
from agents.maze_solver import evaluate_agent, train_agent_on_mazes

STATE_FILE = "curriculum_state.json"


DEFAULT_STATE = {
    "stage": "train",  # "train", "test", "advance"
    "maze_size": 5,
    "iteration": 0,
    "agent_checkpoint": None,
    "generator_checkpoint": None
}


def load_state():
    if not os.path.exists(STATE_FILE):
        return DEFAULT_STATE.copy()
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


def train_stage(state):
    maze_size = state["maze_size"]
    iteration = state["iteration"]
    print(f"[TRAIN] Maze size: {maze_size}x{maze_size}, итерация: {iteration}")

    # 1. Генерация лабиринтов
    maze_folder = f"generated_mazes/{maze_size}x{maze_size}/v{iteration}"
    os.makedirs(maze_folder, exist_ok=True)

    mazes = custom_load_or_generate(maze_size)

    # 2. Обучение агента
    checkpoint_path = f"checkpoints/solver/maze_solver_{maze_size}x{maze_size}_v{iteration}.zip"

    train_agent_on_mazes(
        mazes=mazes,
        model_path=checkpoint_path,
        num_episodes=1000  # или больше, если хочешь
    )

    # 3. Обновляем состояние
    state["agent_checkpoint"] = checkpoint_path
    state["stage"] = "test"
    save_state(state)


def test_stage(state):
    maze_size = state["maze_size"]
    iteration = state["iteration"]
    checkpoint_path = state["agent_checkpoint"]

    print(f"[TEST] Maze size: {maze_size}x{maze_size}, итерация: {iteration}")
    maze_folder = f"generated_mazes/{maze_size}x{maze_size}/v{iteration}"
    maze_paths = [os.path.join(maze_folder, f"maze_{i}.csv") for i in range(1, 101)]
    maze_list = [np.loadtxt(p, delimiter=",", dtype=np.int8) for p in maze_paths]

    # Подгружаем модель и тестим
    results = evaluate_agent(model_path=checkpoint_path, mazes=maze_list)

    # Определим порог допустимого количества шагов
    MAX_STEPS = {
        5: 40,
        10: 90,
        15: 140,
        20: 200
    }.get(maze_size, int(2.2 * maze_size ** 2))

    # Оценка
    success_count = 0
    for result in results:
        if result["success"] and result["steps"] <= MAX_STEPS:
            success_count += 1

    success_rate = success_count / len(results)
    print(f"Успешных прохождений: {success_count}/{len(results)} ({success_rate:.2%})")

    if success_rate >= 0.90:
        print("✅ Агент прошёл текущий уровень сложности. Усложняем.")
        state["maze_size"] += 1  # Усложнение
        state["iteration"] = 0
        state["stage"] = "train"
    else:
        print("🔁 Агент не справился. Повторим обучение.")
        state["iteration"] += 1
        state["stage"] = "train"

    save_state(state)



def advance_stage(state):
    print("[ADVANCE] Increasing maze difficulty.")
    state["maze_size"] += 1
    state["iteration"] = 0
    state["stage"] = "train"
    save_state(state)


def main():
    state = load_state()

    if state["stage"] == "train":
        train_stage(state)
    elif state["stage"] == "test":
        test_stage(state)
    elif state["stage"] == "advance":
        advance_stage(state)
    else:
        print(f"[ERROR] Unknown stage: {state['stage']}")


if __name__ == "__main__":
    main()