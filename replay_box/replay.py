import os

import pygame
import numpy as np
import csv
import time

# --- Настройки ---
TILE_SIZE = 32
FPS = 3
SCALE = 2  # ← Меняешь на 1, 2, 3 и т.д.

COLORS = {
    0: (240, 240, 240),  # empty
    1: (50, 50, 50),      # wall
}


# --- Загрузка данных ---
def load_maze(path):
    return np.loadtxt(path, delimiter=",", dtype=np.int8)

def load_log(path):
    log = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            log.append({
                "step": int(row["step"]),
                "x": int(row["x"]),
                "y": int(row["y"]),
                "action": int(row["action"]),
                "tile": int(row["tile"]),
                "reward": float(row["reward"]),
                "health": int(row["health"]),
                "has_key": bool(int(row["has_key"]))
            })
    return log

# --- Отрисовка ---
def draw_maze(screen, maze, agent_pos, images):
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            tile = maze[i, j]
            rect = pygame.Rect(j * TILE_SIZE * SCALE, i * TILE_SIZE * SCALE,
                               TILE_SIZE * SCALE, TILE_SIZE * SCALE)

            if tile in images:
                img = images[tile]
                img_scaled = pygame.transform.scale(img, (TILE_SIZE * SCALE, TILE_SIZE * SCALE))
                screen.blit(img_scaled, rect.topleft)
            else:
                color = COLORS.get(tile, (100, 100, 100))
                pygame.draw.rect(screen, color, rect)

    # Если есть агент — рисуем
    if agent_pos is not None:
        ay,ax = agent_pos
        rect_agent = pygame.Rect(ay * TILE_SIZE * SCALE, ax * TILE_SIZE * SCALE,
                                 TILE_SIZE * SCALE, TILE_SIZE * SCALE)
        img_agent = images[9]
        img_agent_scaled = pygame.transform.scale(img_agent, (TILE_SIZE * SCALE, TILE_SIZE * SCALE))
        screen.blit(img_agent_scaled, rect_agent.topleft)

def main():
    pygame.init()

    maze = load_maze("maze.csv")

    log = None
    if os.path.exists("episode_log.csv"):
        log = load_log("episode_log.csv")

    h, w = maze.shape
    screen = pygame.display.set_mode((w * TILE_SIZE * SCALE, h * TILE_SIZE * SCALE))
    pygame.display.set_caption("Maze Viewer")
    clock = pygame.time.Clock()

    # Загружаем картинки
    KEY_IMG = pygame.image.load("../replay_box/icon/key.png").convert_alpha()
    DOOR_IMG = pygame.image.load("../replay_box/icon/door.png").convert_alpha()
    TRAP_IMG = pygame.image.load("../replay_box/icon/trap.png").convert_alpha()
    CAMPFIRE_IMG = pygame.image.load("../replay_box/icon/campfire.png").convert_alpha()
    EXIT_IMG = pygame.image.load("../replay_box/icon/exit.png").convert_alpha()
    AGENT_IMG = pygame.image.load("../replay_box/icon/agent.png").convert_alpha()

    IMAGES = {
        2: KEY_IMG,
        3: DOOR_IMG,
        4: TRAP_IMG,
        5: CAMPFIRE_IMG,
        7: EXIT_IMG,
        9: AGENT_IMG,
    }

    running = True
    step_idx = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.MOUSEBUTTONDOWN:
                running = False

        if log:  # Есть лог — воспроизводим
            if step_idx < len(log):
                agent_pos = (log[step_idx]["x"], log[step_idx]["y"])
                draw_maze(screen, maze, agent_pos, IMAGES)
                pygame.display.flip()
                step_idx += 1
                clock.tick(FPS)
            else:
                time.sleep(1)
                running = False
        else:  # Лога нет — просто показываем лабиринт
            draw_maze(screen, maze, None, IMAGES)
            pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()