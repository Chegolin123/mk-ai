# syntax=docker/dockerfile:1
# CPU-образ для CI и headless-тренировок. GPU-вариант — в docker-compose.yml (profile gpu).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy

# libGL/libglib нужны pygame даже в headless-режиме
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# CPU-колёса torch: быстрая сборка и маленький образ, для CI этого достаточно
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install pygame numpy pillow

COPY game.py ./
COPY models ./models
COPY tools ./tools
COPY docs ./docs
COPY README.md LICENSE ./

RUN mkdir -p runs
VOLUME ["/app/models", "/app/runs"]

ENTRYPOINT ["python", "game.py"]
CMD ["--headless", "--episodes", "500"]
