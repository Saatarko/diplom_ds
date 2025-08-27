# Базовый образ Python
FROM python:3.10-slim

# Системные зависимости для pygame и VNC
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    fontconfig \
    xvfb \
    x11vnc \
    xterm \
    xfce4 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем requirements и ставим зависимости
COPY replay_box/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем саму папку replay_box
COPY replay_box/ .

# Устанавливаем пароль для VNC (по желанию)
ENV VNC_PASSWORD=secret

# Скрипт запуска Xvfb + VNC + replay_full.py
CMD bash -c "\
    Xvfb :1 -screen 0 1024x768x24 & \
    x11vnc -display :1 -passwd $VNC_PASSWORD -forever -shared & \
    export DISPLAY=:1 && python replay_full.py"