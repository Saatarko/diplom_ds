import os, csv, time, pygame, numpy as np
from datetime import datetime

from agents import generate_and_infer_best_maze

# --- Настройки ---
TILE_SIZE = 32
FPS = 5
SCALE = 2
PANEL_WIDTH = 200  # справа для кнопок

COLORS = {
    0: (240, 240, 240),  # empty
    1: (50, 50, 50),     # wall
}

# --- Навигатор направления ---
DIRECTIONS = {
    0: (-1, 0),  # вверх
    1: (1, 0),   # вниз
    2: (0, -1),  # влево
    3: (0, 1),   # вправо
}

# --- Загрузка ---
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
def draw_maze(screen, maze, agent_pos, action, images):
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

    # Агент
    if agent_pos is not None:
        x, y = agent_pos
        rect_agent = pygame.Rect(x * TILE_SIZE * SCALE, y * TILE_SIZE * SCALE,
                                 TILE_SIZE * SCALE, TILE_SIZE * SCALE)

        # эффект мигания
        shrink = TILE_SIZE * SCALE - 4
        img_agent = pygame.transform.scale(images[9], (shrink, shrink))
        offset = (rect_agent.x + 2, rect_agent.y + 2)
        screen.blit(img_agent, offset)

        # стрелка действия
        if action in DIRECTIONS:
            dy, dx = DIRECTIONS[action]
            tx, ty = x + dx, y + dy
            if 0 <= tx < maze.shape[1] and 0 <= ty < maze.shape[0]:
                rect_arrow = pygame.Rect(tx * TILE_SIZE * SCALE,
                                         ty * TILE_SIZE * SCALE,
                                         TILE_SIZE * SCALE, TILE_SIZE * SCALE)
                arrow_img = images.get(10 + action)  # стрелки будут 10..13
                if arrow_img:
                    arr_scaled = pygame.transform.scale(arrow_img, (TILE_SIZE * SCALE, TILE_SIZE * SCALE))
                    screen.blit(arr_scaled, rect_arrow.topleft)

# --- Кнопки ---
def draw_panel(screen, font):
    w, h = screen.get_size()
    x0 = w - PANEL_WIDTH
    pygame.draw.rect(screen, (200, 200, 200), (x0, 0, PANEL_WIDTH, h))

    buttons = [("Generate", 50), ("Rerun", 120), ("Exit", 190)]
    btn_rects = {}
    for label, y in buttons:
        rect = pygame.Rect(x0 + 20, y, PANEL_WIDTH - 40, 50)
        pygame.draw.rect(screen, (150, 150, 150), rect)
        text = font.render(label, True, (0, 0, 0))
        screen.blit(text, (rect.x + 10, rect.y + 10))
        btn_rects[label] = rect
    return btn_rects

# --- Main ---
def main():
    pygame.init()
    font = pygame.font.SysFont("Arial", 24)

    maze = None
    log = None
    step_idx = 0

    # экран с панелью
    maze_w, maze_h = 15, 15
    screen = pygame.display.set_mode((maze_w * TILE_SIZE * SCALE + PANEL_WIDTH,
                                      maze_h * TILE_SIZE * SCALE))
    pygame.display.set_caption("Maze Viewer")
    clock = pygame.time.Clock()

    # Загружаем картинки
    KEY_IMG = pygame.image.load("../replay_box/icon/key.png").convert_alpha()
    DOOR_IMG = pygame.image.load("../replay_box/icon/door.png").convert_alpha()
    TRAP_IMG = pygame.image.load("../replay_box/icon/trap.png").convert_alpha()
    CAMPFIRE_IMG = pygame.image.load("../replay_box/icon/campfire.png").convert_alpha()
    EXIT_IMG = pygame.image.load("../replay_box/icon/exit.png").convert_alpha()
    AGENT_IMG = pygame.image.load("../replay_box/icon/agent.png").convert_alpha()

    ARROW_UP = pygame.image.load("../replay_box/icon/up.png").convert_alpha()
    ARROW_DOWN = pygame.image.load("../replay_box/icon/down.png").convert_alpha()
    ARROW_LEFT = pygame.image.load("../replay_box/icon/left.png").convert_alpha()
    ARROW_RIGHT = pygame.image.load("../replay_box/icon/right.png").convert_alpha()

    IMAGES = {
        2: KEY_IMG,
        3: DOOR_IMG,
        4: TRAP_IMG,
        5: CAMPFIRE_IMG,
        7: EXIT_IMG,
        9: AGENT_IMG,
        10: ARROW_UP,
        11: ARROW_DOWN,
        12: ARROW_LEFT,
        13: ARROW_RIGHT,
    }

    running = True
    while running:
        screen.fill((220, 220, 220))

        # Рисуем лабиринт
        if maze is not None:
            if log and step_idx < len(log):
                agent_pos = (log[step_idx]["x"], log[step_idx]["y"])
                action = log[step_idx]["action"]
                draw_maze(screen, maze, agent_pos, action, IMAGES)
                step_idx += 1
                clock.tick(FPS)
            else:
                draw_maze(screen, maze, None, None, IMAGES)

        # Панель
        btn_rects = draw_panel(screen, font)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if btn_rects["Exit"].collidepoint(mx, my):
                    running = False
                elif btn_rects["Rerun"].collidepoint(mx, my):
                    if os.path.exists("result/mazes/best_maze.csv") and os.path.exists("result/logs/best_run.csv"):
                        maze = load_maze("result/mazes/best_maze.csv")
                        log = load_log("result/logs/best_run.csv")
                        step_idx = 0
                elif btn_rects["Generate"].collidepoint(mx, my):

                    # maze, log = generate_and_infer_best_maze(
                    #     "generator_agent_15", "navigator_agent_15"
                    # )
                    # np.savetxt("best_maze.csv", np.array(maze, dtype=np.int32), fmt="%d", delimiter=",")
                    # with open("best_run.csv", "w", newline="") as f:
                    #     fieldnames = ["step", "x", "y", "action", "tile", "reward", "health", "has_key"]
                    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
                    #     writer.writeheader()
                    #     writer.writerows(log)
                    #
                    # step_idx = 0

                    # --- Показываем заставку ---
                    screen.fill((50, 50, 50))  # фон
                    font = pygame.font.SysFont(None, 48)
                    text_surf = font.render("Генерация лабиринта... Пожалуйста, ждите", True, (255, 255, 255))
                    screen.blit(text_surf, (50, screen.get_height() // 2))
                    pygame.display.flip()

                    # --- Вызываем генерацию ---
                    maze, log = generate_and_infer_best_maze(
                        "generator_agent_15", "navigator_agent_15"
                    )

                    # --- Сохраняем результат ---
                    np.savetxt("best_maze.csv", np.array(maze, dtype=np.int32), fmt="%d", delimiter=",")
                    with open("best_run.csv", "w", newline="") as f:
                        fieldnames = ["step", "x", "y", "action", "tile", "reward", "health", "has_key"]
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(log)

                    step_idx = 0

    pygame.quit()

if __name__ == "__main__":
    main()