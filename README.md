# 🧭 RL Multi-Agent Maze Prototype  

**Проект: Исследование прототипа мультиагентной архитектуры с разделением функций: генерация среды, решение задачи в среде, мета-управление.**  

---

## 🚀 Запуск проекта

### Демонстрационный инференс (генерация + навигация)

#### Локально:

- Клонировать репозиторий
- Загрузить модели с диска

```
https://drive.google.com/drive/folders/15SN9VTYMs74h12f8yNwtEcAY2Q9ZwfYn?usp=sharing
```

- Запустить инференс (replay_box/replay_full.py):

```
python replay_full.py
```

### Самостоятельно обучение

#### Для обучения с нуля (кроме метаагента)

- Клонировать репозиторий
- Запустить обучение генератора:
```
python maze_generator_cell_train.py 
```
- запустить  для обучения навигатора 
```
python maze_solver.py 
```

#### Для обучения с нуля метаагента:

- Рационально в meta_maze.py заменить (для первых 1000 -2000 итераций) реальный инференс и обучения агентов заглушками (для серьезной экономии времени)
- запустить meta_maze.py
------
## 🛠 Используемые технологии
- RL фреймворки: Gymnasium, Stable Baselines3 (PPO, MaskablePPO)
- ML-библиотеки: PyTorch
- Визуализация и интерфейсы: Pygame

Подходы: Curriculum learning, Adversarial Curriculum Learning

-------
## 🎓 1. Обоснование актуальности проекта

Задача навигации и генерации среды в рамках обучения с подкреплением (Reinforcement Learning, RL) является фундаментальной для разработки адаптивных интеллектуальных агентов. Несмотря на то, что прохождение лабиринтов может казаться упрощённой или академической задачей, данный проект решает несколько ключевых и актуальных исследовательских направлений в области ИИ и RL:

1.1 Исследование взаимодействия агента и среды
Проект демонстрирует полноценное моделирование цикла «агент ↔ среда», где среда не является фиксированной, а динамически создаётся другим RL-агентом. Это открывает возможность исследовать двустороннюю адаптацию, что соответствует реальным задачам, где окружающая среда может быть изменчивой или враждебной (adversarial).

1.2 Генерация среды как отдельная RL-задача
Разработка генератора лабиринтов, который обучается на основе сложности и проходимости, является примером обучения среды — активно развивающегося направления в AI/ML. Это приближает проект к задачам вроде автоматического дизайна уровней в играх (AI-assisted game design) и обучающих симуляторов.

1.3 Сurriculum learning и adversarial взаимодействия.
Проект внедряет adversarial curriculum learning: лабиринты последовательно усложняются в ответ на успехи навигатора, что позволяет построить более устойчивую траекторию обучения. Введение мета-агента, управляющего этим процессом, расширяет проект до мета-RL, одного из самых перспективных направлений (learning to learn).

1.4 Навигация как базовая задача в робототехнике и CV
Прохождение лабиринтов моделирует задачи, аналогичные навигации роботов или автономных машин, но с преимуществом полной контролируемости и малой вычислительной стоимости. Таким образом, проект:

- служит прототипом для задач реального мира (path planning, obstacle avoidance);
- избегает необходимости работы с громоздкими CV-данными, что делает его реализуемым за ограниченное время (2 месяца);
- сохраняет исследовательскую глубину, сравнимую с более дорогими задачами.

1.5 Гибкость, воспроизводимость и научная ценность
Мы используеи собственные среды, что делает экспериментальную часть воспроизводимой, легко масштабируемой и полностью контролируемой.

### 📌 Вывод:
Проект демонстрирует:
- современные подходы в RL (multi-agent, curriculum, meta-learning);
- оригинальную постановку задачи (агент строит среду для другого агента);
- воспроизводимую исследовательскую базу.

## 🤖 2. Связь с реальными задачами:

- Path planning в автономных авто / дронах — похожий алгоритмический уровень;
- Робототехника
- Сетевая маршрутизация
- Обучение агента в игре (например, RTS) — те же элементы: генератор карты + агент;
- Генерация заданий для обучения — meta-learning;
- Auto-curriculum и self-play — применяется в AlphaZero, OpenAI Five, etc.

## 📊 Сравнение с CV/робототехникой

| Критерий                      | Навигация по лабиринту                  | CV/робототехника            |
| ----------------------------- | --------------------------------------- | --------------------------- |
| Своя среда                    | ✅ Легко сделать и адаптировать          | ❌ Сложно, дорого            |
| Контроль сложности            | ✅ Полный контроль (через генератор)     | ❌ Непредсказуемые кейсы     |
| Кол-во данных                 | ✅ Генерируем на лету                    | ❌ Ограниченные датасеты     |
| Скорость обучения             | ✅ Приемлемая                            | ❌ Очень медленно            |
| Воспроизводимость             | ✅ Лабораторная, контролируемая          | ❌ Зависит от сенсоров, шума |
| Возможность внедрения meta-RL | ✅ Прямо по схеме builder/navigator/meta | ❌ Сложно интегрировать      |


## 3. Логика трёх агентов

### 3.1  Генератор
- Отвечает за формирование среды испытаний (лабиринтов).
- Усложняет или упрощает окружение, реагируя на успехи/неудачи навигатора.
- При работе имет две фазы: копание и размещение элементов. Основные наблюдения получает из карты лабиринта и 
тепловой карты наград

В прикладных аналогиях: 

- это может быть система, которая генерирует сценарии/ситуации — дорожные условия для беспилотников, 
- топологии для сетевой маршрутизации, 
- задания для роботов.

То есть он симулирует мир, который нужно преодолеть.

### 3.2 Навигатор

Задача — находить путь в сгенерированной среде.
Он оперирует алгоритмами поиска и планирования маршрута (по сути решает задачу pathfinding / planning).
Основные наблюдения получает из карты лабиринта и тепловой карты наград (с защитой от застревания в тупиках)

В прикладных аналогиях:

- Для робота → навигация в помещении.
- Для сети → выбор маршрута доставки пакетов.
- Для логистики → оптимизация цепочки поставок.

- Главное — он не изменяет среду, он адаптируется.

### 3.3 Мета-агент

Это «дирижёр» или «оркестратор».

Он не знает деталей реализации подчинённых агентов, но умеет:
- решать, кого запустить,
- оценивать результаты,
- менять сложность сценариев,
- инициировать дообучение нужного агента.

Идея взята из Adversarial Curriculum Learning (ACL): генератор «бросает вызов», навигатор пытается справиться, а мета-агент решает баланс между ними.

В прикладных аналогиях:

- В беспилотниках → мета-агент может решать: тренировать ли сейчас perception-модуль (CV), планирование или контроль.
- В кибербезопасности → агент может решать: стоит ли тренировать генератор атак (Red Team) или навигатор-защитник (Blue Team).
- В сетевых системах → мета-агент регулирует нагрузку на симулятор сети и «маршрутизатор».

### 3.3.1 Аналогия с Mass Effect: Легион 🛡️

Легион — это множество агентов, объединённых общей целью.
У нас  — система агентов, где каждый узкоспециализирован, но мета-агент координирует их для достижения результата.
«Система напоминает принцип "коллективного интеллекта", где отдельные агенты выполняют специализированные функции, а управляющий слой обеспечивает координацию их действий».

# 🧭 RL Multi-Agent Maze Prototype

**Project: Research of a prototype of a multi-agent architecture with separation of functions: environment generation, task solution in the environment, meta-management.**

---

## 🚀 Launch the project

### Demo inference (generation + navigation)

#### Locally:

- Clone repository
- Load models from disk

```
https://drive.google.com/drive/folders/15SN9VTYMs74h12f8yNwtEcAY2Q9ZwfYn?usp=sharing
```

- Run inference (replay_box/replay_full.py):

```
python replay_full.py
```

### Self-training

#### For training from scratch (except metaagent)

- Clone repository
- Run generator training:
```
python maze_generator_cell_train.py
```
- run for navigator training
```
python maze_solver.py
```

For training a meta-agent from scratch:

- Rationally in meta_maze.py replace (for the first 1000-2000 iterations) real inference and agent training with stubs (for significant time savings)
- run meta_maze.py
------
## 🛠 Technologies used
- RL frameworks: Gymnasium, Stable Baselines3 (PPO, MaskablePPO)
- ML libraries: PyTorch
- Visualization and interfaces: Pygame

Approaches: Curriculum learning, Adversarial Curriculum Learning

-------
## 🎓 1. Justification of the relevance of the project

The task of navigation and environment generation within the framework of reinforcement learning (RL) is fundamental for the development of adaptive intelligent agents. While solving a maze may seem like a simplistic or academic task, this project addresses several key and current research areas in AI and RL:

1.1 Study of agent-environment interaction
The project demonstrates a full-fledged simulation of the agent ↔ environment cycle, where the environment is not fixed but dynamically generated by another RL agent. This opens up the possibility of exploring two-way adaptation, which corresponds to real-world problems where the environment can be changeable or adversarial.

1.2 Environment generation as a separate RL task
Developing a maze generator that learns based on complexity and passability is an example of environment learning, a rapidly developing area in AI/ML. This brings the project closer to problems such as AI-assisted game design and educational simulators.

1.3 Curriculum learning and adversarial interactions.
The project implements adversarial curriculum learning: mazes become progressively more complex in response to the navigator's success, which allows for a more stable learning trajectory to be built. The introduction of a meta-agent that controls this process expands the project to meta-RL, one of the most promising areas (learning to learn).

1.4 Navigation as a Basic Task in Robotics and CV
Navigating mazes simulates tasks similar to the navigation of robots or autonomous machines, but with the advantage of complete controllability and low computational cost. Thus, the project:

- serves as a prototype for real-world tasks (path planning, obstacle avoidance);
- avoids the need to work with cumbersome CV data, which makes it feasible in a limited time (2 months);
- maintains a research depth comparable to more expensive tasks.

5. Flexibility, reproducibility and scientific value
We use our own environments, which makes the experimental part reproducible, easily scalable and fully controllable.

### 📌 Conclusion:
The project demonstrates:
- modern approaches in RL (multi-agent, curriculum, meta-learning);
- original problem statement (an agent builds an environment for another agent);
- reproducible research base.

## 🤖 2. Connection with real-world tasks:

- Path planning in autonomous cars / drones - similar algorithmic level;
- Robotics
- Network routing
- Agent training in a game (e.g. RTS) - the same elements: map generator + agent;
- Generation of tasks for training - meta-learning;
- Auto-curriculum and self-play - used in AlphaZero, OpenAI Five, etc.

## 📊 Comparison with CV/robotics

| Criterion | Maze navigation | CV/robotics |
| ----------------------------- | --------------------------------------- | -------------------------- |
| Own environment | ✅ Easy to make and adapt | ❌ Complex, expensive |
| Complexity control | ✅ Full control (via generator) | ❌ Unpredictable cases |
| Amount of data | ✅ Generate on the fly | ❌ Limited datasets |
| Training speed | ✅ Acceptable | ❌ Very slow |
| Reproducibility | ✅ Laboratory, controlled | ❌ Dependent on sensors, noise |
| Possibility of implementing meta-RL | ✅ Directly according to the builder/navigator/meta scheme | ❌ Difficult to integrate |

## 3. Three-agent logic

### 3.1 Generator
- Responsible for generating the testing environment (mazes).
- Complicates or simplifies the environment, reacting to the navigator's successes/failures.
- Has two phases during operation: digging and placing elements. Receives main observations from the labirinta and
reward heat map

In applied analogies:

- it can be a system that generates scenarios/situations — road conditions for drones,
- topologies for network routing,
- tasks for robots.

That is, it simulates the world that needs to be overcome.

### 3.2 Navigator

The task is to find a path in the generated environment.
It operates with search and route planning algorithms (essentially solves the pathfinding / planning problem).
It receives its main observations from the maze map and the reward heat map (with protection against getting stuck in dead ends)

In applied analogies:

- For a robot → indoor navigation.
- For a network → choosing a route for delivering packages.
- For logistics → supply chain optimization.

- The main thing is that it does not change the environment, it adapts.

### 3.3 Meta-agent

This is a "conductor" or "orchestrator".

It does not know the implementation details of subordinate agents, but it can:
- decide who to launch,
- evaluate results,
- change the complexity of scenarios,
- initiate additional training of the required agent.

The idea is taken from Adversarial Curriculum Learning (ACL): the generator "throws down a challenge", the navigator tries to cope, and the meta-agent decides the balance between them.

In applied analogies:

- In drones → the meta-agent can decide whether to train the perception module (CV), planning or control.
- In cybersecurity → the agent can decide whether to train the attack generator (Red Team) or the navigator-defender (Blue Team).
- In network systems → the meta-agent regulates the load on the network simulator and the "router".

### 3.3.1 Analogy with Mass Effect: Legion 🛡️

Legion is a set of agents united by a common goal.
We have a system of agents, where each is highly specialized, but the meta-agent coordinates them to achieve the result.
"The system resembles the principle of "collective intelligence", where individual agents perform specialized functions, and the control layer ensures the coordination of their actions."