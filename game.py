"""MK-AI — два гуманоида учатся драться (2D, вид сбоку).

Идея
----
Два бойца управляются одной нейросетевой политикой (self-play). Каждый эпизод —
это одна «попытка»: бойцы дерутся, получают награды, и сразу после боя политика
обновляется на собранном опыте (actor-critic / A2C). Обучение идёт на GPU через
PyTorch, счётчик итераций и статистика выводятся прямо на экран.

Награды
-------
* успешный удар:        +1.0 и больше (зависит от урона);
* пропущенный удар:     -0.35;
* 2 секунды без ударов: -0.5 (повторяется каждые 2 секунды простоя);
* победа:               +10;
* поражение:            -6;
* ничья по таймеру:     -1 обоим (стимулируем добивать).

Запуск
------
    python game.py                 # окно + обучение
    python game.py --headless --episodes 100
    python game.py --speed 4       # 1..8, можно менять на ходу клавишами 1-4

Управление
----------
    Space  пауза
    1..4   скорость симуляции (1x, 2x, 4x, 8x)
    S      сохранить модель
    Esc    выход (модель сохраняется автоматически)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Константы мира
# ---------------------------------------------------------------------------

W, H = 1280, 720
GROUND_Y = 610
FPS = 60
DECISION_EVERY = 4          # решение принимается 15 раз в секунду
MAX_FRAMES = 20 * FPS       # длительность одной попытки
IDLE_FRAMES = 2 * FPS       # 2 секунды без удара = штраф

N_ACTIONS = 7
# 0 стоять, 1 назад, 2 вперёд, 3 прыжок, 4 удар рукой, 5 удар ногой, 6 блок
ACTION_NAMES = ["стоять", "назад", "вперёд", "прыжок", "удар", "пинок", "блок"]

OBS_DIM = 18
HIDDEN = 128
GAMMA = 0.995

SPEED_FORWARD = 4.4
GRAVITY = 0.78
JUMP_VY = -14.0
PUNCH_RANGE = 96.0
KICK_RANGE = 128.0
PUNCH_DAMAGE = 8.0
KICK_DAMAGE = 13.0

# Награды. Простой должен быть заведомо хуже, чем риск проиграть бой,
# иначе политика находит ловушку «стоять и ничего не делать».
R_HIT = 2.0
R_HIT_DMG = 0.08
R_BLOCKED = 0.5
R_TAKEN = 0.6
R_IDLE = 1.0
R_WIN = 20.0
R_LOSS = 8.0
R_DRAW = 3.0

TEAM_COLORS = [
    {"body": (232, 84, 62), "dark": (150, 38, 30), "trim": (255, 196, 128), "glow": (255, 110, 80)},
    {"body": (64, 156, 232), "dark": (26, 82, 150), "trim": (150, 230, 255), "glow": (90, 200, 255)},
]

MODEL_DIR = Path(__file__).resolve().parent / "models"


# ---------------------------------------------------------------------------
# Политика
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Маленькая MLP: наблюдение -> вероятности действий + оценка ценности."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = HIDDEN):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        # стартуем почти равномерно, чтобы исследовать все действия
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Боец
# ---------------------------------------------------------------------------

@dataclass
class Fighter:
    idx: int
    x: float
    facing: int = 1
    hp: float = 100.0
    y: float = GROUND_Y
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = True
    action: int = 0
    blocking: bool = False
    attack_kind: int = 0          # 0 нет, 1 удар рукой, 2 пинок
    attack_timer: int = 0
    attack_duration: int = 0
    attack_hit_done: bool = False
    cooldown: int = 0
    stun: int = 0
    hit_flash: int = 0
    walk_phase: float = 0.0
    last_hit_frame: int = 0
    last_penalty_frame: int = 0
    ep_reward: float = 0.0
    hits: int = 0
    taken: int = 0
    idle_penalties: int = 0

    # буферы траектории для обучения
    obs_buf: list = field(default_factory=list)
    act_buf: list = field(default_factory=list)
    rew_buf: list = field(default_factory=list)
    val_buf: list = field(default_factory=list)

    def reset(self, x: float, facing: int):
        self.x = x
        self.facing = facing
        self.hp = 100.0
        self.y = GROUND_Y
        self.vx = self.vy = 0.0
        self.on_ground = True
        self.action = 0
        self.blocking = False
        self.attack_kind = 0
        self.attack_timer = 0
        self.attack_duration = 0
        self.attack_hit_done = False
        self.cooldown = 0
        self.stun = 0
        self.hit_flash = 0
        self.walk_phase = 0.0
        self.last_hit_frame = 0
        self.last_penalty_frame = 0
        self.ep_reward = 0.0
        self.hits = 0
        self.taken = 0
        self.idle_penalties = 0
        self.obs_buf.clear()
        self.act_buf.clear()
        self.rew_buf.clear()
        self.val_buf.clear()

    # --- удобные свойства для наблюдения ---
    @property
    def attack_active(self) -> bool:
        if self.attack_timer <= 0:
            return False
        elapsed = self.attack_duration - self.attack_timer
        if self.attack_kind == 1:
            return 5 <= elapsed <= 9
        if self.attack_kind == 2:
            return 10 <= elapsed <= 16
        return False


# ---------------------------------------------------------------------------
# Бой
# ---------------------------------------------------------------------------

class Fight:
    def __init__(self, policy: ActorCritic, device: torch.device):
        self.policy = policy
        self.device = device
        self.p1 = Fighter(0, 560.0, facing=1)
        self.p2 = Fighter(1, 720.0, facing=-1)
        self.fighters = [self.p1, self.p2]
        self.frame = 0
        self.finished = False
        self.winner = 0               # 0 ничья, 1 / 2 — победитель
        self.sparks: list = []
        self.banner_timer = 0
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.p1.reset(560.0, 1)
        self.p2.reset(720.0, -1)
        self.frame = 0
        self.finished = False
        self.winner = 0
        self.sparks.clear()
        self.banner_timer = 0

    # --------------------------------------------------------------- helpers
    def other(self, f: Fighter) -> Fighter:
        return self.p2 if f is self.p1 else self.p1

    def reward(self, f: Fighter, r: float):
        if f.rew_buf:
            f.rew_buf[-1] += r
        f.ep_reward += r

    def dist(self, a: Fighter, b: Fighter) -> float:
        return abs(a.x - b.x)

    # ------------------------------------------------------------------ obs
    def obs(self, f: Fighter) -> np.ndarray:
        o = self.other(f)
        dist = self.dist(f, o)
        since_hit = min(1.0, max(0, self.frame - f.last_hit_frame) / IDLE_FRAMES)
        return np.array([
            f.hp / 100.0,
            o.hp / 100.0,
            (o.x - f.x) / W,
            min(1.0, dist / 600.0),
            (o.y - f.y) / 150.0,
            1.0 if f.on_ground else 0.0,
            1.0 if o.on_ground else 0.0,
            1.0 if f.attack_active else 0.0,
            1.0 if o.attack_active else 0.0,
            1.0 if f.blocking else 0.0,
            f.stun / 14.0,
            f.cooldown / 40.0,
            1.0 if dist <= PUNCH_RANGE else 0.0,
            1.0 if dist <= KICK_RANGE else 0.0,
            max(0.0, (MAX_FRAMES - self.frame) / MAX_FRAMES),
            since_hit,
            (f.vx - o.vx) / 10.0,
            (f.hp - o.hp) / 100.0,
        ], dtype=np.float32)

    # ------------------------------------------------------------- decisions
    def decide(self, f: Fighter):
        obs = self.obs(f)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.policy(x)
            dist = Categorical(logits=logits)
            act = dist.sample()
            logp = float(dist.log_prob(act))
            val = float(value)
        f.obs_buf.append(obs)
        f.act_buf.append(int(act))
        f.rew_buf.append(0.0)
        f.val_buf.append(val)
        f.action = int(act)

    # -------------------------------------------------------------- physics
    def apply_action(self, f: Fighter):
        o = self.other(f)
        f.facing = 1 if o.x >= f.x else -1

        if f.cooldown > 0:
            f.cooldown -= 1
        if f.hit_flash > 0:
            f.hit_flash -= 1

        if f.stun > 0:
            f.stun -= 1
            f.vx *= 0.7
            f.blocking = False
            return

        if f.attack_timer > 0:
            f.attack_timer -= 1
            f.blocking = False
            f.vx *= 0.5
            if f.attack_timer == 0:
                f.attack_kind = 0
            return

        action = f.action
        f.blocking = (action == 6 and f.on_ground)

        if action in (4, 5) and f.on_ground and f.cooldown <= 0:
            if action == 4:
                f.attack_kind = 1
                f.attack_timer = 16
                f.attack_duration = 16
                f.cooldown = 26
            else:
                f.attack_kind = 2
                f.attack_timer = 26
                f.attack_duration = 26
                f.cooldown = 40
            f.attack_hit_done = False
            f.vx = 0.0
            return

        if action == 3 and f.on_ground:
            f.vy = JUMP_VY
            f.on_ground = False
            f.vx = f.facing * SPEED_FORWARD * 0.55
            return

        if action == 2:          # вперёд, к противнику
            f.vx = f.facing * SPEED_FORWARD
            f.walk_phase += 0.35
        elif action == 1:        # назад
            f.vx = -f.facing * SPEED_FORWARD * 0.78
            f.walk_phase += 0.28
        else:
            f.vx *= 0.6

    def physics(self, f: Fighter):
        if not f.on_ground:
            f.vy += GRAVITY
            f.y += f.vy
            if f.y >= GROUND_Y:
                f.y = GROUND_Y
                f.vy = 0.0
                f.on_ground = True

        f.x += f.vx
        f.x = max(70.0, min(W - 70.0, f.x))

    def separate(self):
        """Не даём бойцам проходить друг сквозь друга."""
        min_sep = 56.0
        dx = self.p2.x - self.p1.x
        if abs(dx) < min_sep:
            sign = 1.0 if dx >= 0 else -1.0
            push = (min_sep - abs(dx)) / 2.0
            self.p1.x -= sign * push
            self.p2.x += sign * push
            self.p1.x = max(70.0, self.p1.x)
            self.p2.x = min(W - 70.0, self.p2.x)

    # ----------------------------------------------------------------- hits
    def resolve_hits(self):
        order = self.fighters if self.frame % 2 == 0 else list(reversed(self.fighters))
        for att in order:
            if att.attack_timer <= 0 or att.attack_hit_done or not att.attack_active:
                continue
            dfd = self.other(att)
            if dfd.hp <= 0:
                continue
            rng = PUNCH_RANGE if att.attack_kind == 1 else KICK_RANGE
            dmg = PUNCH_DAMAGE if att.attack_kind == 1 else KICK_DAMAGE
            if self.dist(att, dfd) > rng or abs(att.y - dfd.y) > 70:
                continue

            att.attack_hit_done = True
            att.hits += 1
            att.last_hit_frame = self.frame
            att.last_penalty_frame = self.frame

            contact_x = (att.x + dfd.x) / 2
            contact_y = dfd.y - 120

            if dfd.blocking:
                dmg *= 0.22
                dfd.hp -= dmg
                dfd.vx += att.facing * 2.2
                self.reward(att, R_BLOCKED)
                self.reward(dfd, -0.1)
                self.sparks.append({"x": contact_x, "y": contact_y, "ttl": 10,
                                    "color": (220, 220, 255), "big": False})
            else:
                dfd.hp -= dmg
                dfd.taken += 1
                dfd.stun = 14 if att.attack_kind == 1 else 20
                dfd.hit_flash = 8
                dfd.blocking = False
                dfd.attack_timer = 0
                dfd.attack_kind = 0
                dfd.vx += att.facing * (5.5 if att.attack_kind == 1 else 8.0)
                self.reward(att, R_HIT + dmg * R_HIT_DMG)
                self.reward(dfd, -R_TAKEN)
                self.sparks.append({"x": contact_x, "y": contact_y, "ttl": 14,
                                    "color": TEAM_COLORS[att.idx]["glow"], "big": True})

            dfd.hp = max(0.0, dfd.hp)

    # --------------------------------------------------------------- shaping
    def shaping(self):
        for f in self.fighters:
            if f.stun > 0:
                continue
            o = self.other(f)
            dist = self.dist(f, o)
            near = dist <= PUNCH_RANGE * 0.9

            # Подводим друг к другу: без этого случайная политика может
            # простоять весь эпизод на дистанции и никогда не увидеть удары.
            if f.action == 2 and not near:
                self.reward(f, 0.006)
            elif f.action == 1 and near:
                self.reward(f, -0.004)

            # Атака по цели поощряется, удар «в воздух» — нет.
            if f.action == 4:
                self.reward(f, 0.006 if dist <= PUNCH_RANGE else -0.004)
            elif f.action == 5:
                self.reward(f, 0.006 if dist <= KICK_RANGE else -0.004)

    def idle_penalty(self):
        for f in self.fighters:
            if (self.frame - f.last_hit_frame > IDLE_FRAMES and
                    self.frame - f.last_penalty_frame >= IDLE_FRAMES):
                self.reward(f, -R_IDLE)
                f.idle_penalties += 1
                f.last_penalty_frame = self.frame

    # ------------------------------------------------------------------ step
    def step(self):
        if self.finished:
            return

        if self.frame % DECISION_EVERY == 0:
            self.decide(self.p1)
            self.decide(self.p2)

        self.apply_action(self.p1)
        self.apply_action(self.p2)
        self.physics(self.p1)
        self.physics(self.p2)
        self.separate()
        self.resolve_hits()
        self.shaping()
        self.idle_penalty()

        # искры
        for s in self.sparks:
            s["ttl"] -= 1
        self.sparks = [s for s in self.sparks if s["ttl"] > 0]

        self.frame += 1
        self.check_end()

    def check_end(self):
        p1, p2 = self.p1, self.p2
        if p1.hp <= 0 and p2.hp <= 0:
            winner = 0
        elif p1.hp <= 0:
            winner = 2
        elif p2.hp <= 0:
            winner = 1
        elif self.frame >= MAX_FRAMES:
            if abs(p1.hp - p2.hp) < 1.0:
                winner = 0
            else:
                winner = 1 if p1.hp > p2.hp else 2
        else:
            return

        self.finished = True
        self.winner = winner
        if winner == 1:
            self.reward(p1, R_WIN)
            self.reward(p2, -R_LOSS)
        elif winner == 2:
            self.reward(p2, R_WIN)
            self.reward(p1, -R_LOSS)
        else:
            self.reward(p1, -R_DRAW)
            self.reward(p2, -R_DRAW)


# ---------------------------------------------------------------------------
# Тренер: обновление политики после каждой попытки
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, device: torch.device, model_path: Path, load: bool = True):
        self.device = device
        self.model_path = model_path
        self.policy = ActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=5e-4)
        self.episode = 0
        self.wins = [0, 0, 0]           # p1, p2, ничьи
        self.history = deque(maxlen=120)
        self.reward_history = deque(maxlen=60)
        self.hit_history = deque(maxlen=60)
        self.loss_history = deque(maxlen=60)
        self.last_loss = 0.0
        self.best_hits = 0
        self.loaded = False
        if load and model_path.exists():
            self.load()

    # ------------------------------------------------------------------ io
    def save(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.model_path.with_suffix(".tmp")
        torch.save({
            "model": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episode": self.episode,
            "wins": self.wins,
            "history": list(self.history),
        }, tmp)
        os.replace(tmp, self.model_path)

    def load(self):
        try:
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            self.episode = int(ckpt.get("episode", 0))
            self.wins = list(ckpt.get("wins", [0, 0, 0]))
            self.history = deque(ckpt.get("history", []), maxlen=120)
            self.loaded = True
            print(f"[model] загружена модель: эпизод {self.episode}, победы {self.wins}")
        except Exception as e:
            print(f"[model] не удалось загрузить ({e}), начинаем с нуля")

    # ------------------------------------------------------------- training
    def learn(self, fight: Fight) -> float:
        obs, acts, vals, rets = [], [], [], []
        for f in fight.fighters:
            if not f.rew_buf:
                continue
            r = np.asarray(f.rew_buf, dtype=np.float32)
            g = 0.0
            disc = []
            for x in r[::-1]:
                g = x + GAMMA * g
                disc.append(g)
            disc.reverse()
            obs.append(np.stack(f.obs_buf))
            acts.append(np.asarray(f.act_buf, dtype=np.int64))
            vals.append(np.asarray(f.val_buf, dtype=np.float32))
            rets.append(np.asarray(disc, dtype=np.float32))

        if not obs:
            return 0.0

        obs_t = torch.as_tensor(np.concatenate(obs), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.concatenate(acts), dtype=torch.int64, device=self.device)
        ret_t = torch.as_tensor(np.concatenate(rets), dtype=torch.float32, device=self.device)

        # 2 эпохи по всему эпизоду; эпизод небольшой, GPU успевает
        total_loss = 0.0
        n = obs_t.shape[0]
        for _ in range(2):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, 512):
                idx = perm[start:start + 512]
                logits, values = self.policy(obs_t[idx])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act_t[idx])
                entropy = dist.entropy()
                adv = ret_t[idx] - values.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = -(logp * adv).mean()
                value_loss = F.mse_loss(values, ret_t[idx])
                loss = policy_loss + 0.1 * value_loss - 0.02 * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                total_loss += float(loss)

        self.episode += 1
        self.last_loss = total_loss / 2.0
        self.loss_history.append(self.last_loss)

        if fight.winner == 1:
            self.wins[0] += 1
        elif fight.winner == 2:
            self.wins[1] += 1
        else:
            self.wins[2] += 1
        self.history.append(fight.winner)

        hits = max(fight.p1.hits, fight.p2.hits)
        self.hit_history.append(hits)
        self.best_hits = max(self.best_hits, hits)
        self.reward_history.append((fight.p1.ep_reward + fight.p2.ep_reward) / 2.0)

        if self.episode % 10 == 0:
            self.save()
        return self.last_loss

    # ------------------------------------------------------------ статистика
    def avg(self, dq):
        return sum(dq) / len(dq) if dq else 0.0

    def winrate(self, last: int = 40) -> float:
        h = list(self.history)[-last:]
        if not h:
            return 0.0
        return sum(1 for x in h if x == 1) / len(h)


# ---------------------------------------------------------------------------
# Рендер (pygame)
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self, trainer: Trainer, speed: int):
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption("MK-AI — самообучающиеся бойцы")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.trainer = trainer
        self.speed = speed
        self.fonts = {
            "small": pygame.font.SysFont("segoeui", 14),
            "text": pygame.font.SysFont("segoeui", 18),
            "bold": pygame.font.SysFont("segoeui", 20, bold=True),
            "hud": pygame.font.SysFont("segoeui", 30, bold=True),
            "big": pygame.font.SysFont("segoeui", 54, bold=True),
        }
        self.bg = self._make_background()
        self.crowd = [(random.randint(20, W - 20), random.randint(300, 430),
                       random.choice([(34, 40, 56), (44, 50, 70), (28, 34, 48)]))
                      for _ in range(160)]
        self.t = 0.0

    def _gradient(self, top, bottom):
        pg = self.pg
        surf = pg.Surface((W, H))
        for y in range(H):
            k = y / max(1, H - 1)
            col = tuple(int(top[i] + (bottom[i] - top[i]) * k) for i in range(3))
            pg.draw.line(surf, col, (0, y), (W, y))
        return surf

    def _make_background(self):
        pg = self.pg
        bg = self._gradient((10, 14, 28), (30, 26, 46))

        # луна
        pg.draw.circle(bg, (230, 232, 220), (1020, 120), 58)
        pg.draw.circle(bg, (210, 214, 205), (1000, 104), 10)
        pg.draw.circle(bg, (214, 218, 208), (1042, 142), 7)
        pg.draw.circle(bg, (214, 218, 208), (1014, 152), 5)

        # дальние горы
        for pts, col in [
            ([(-40, 470), (180, 330), (360, 470)], (24, 30, 48)),
            ([(280, 470), (520, 300), (760, 470)], (28, 34, 54)),
            ([(680, 470), (960, 340), (1240, 470)], (24, 30, 48)),
        ]:
            pg.draw.polygon(bg, col, pts)
        pg.draw.rect(bg, (20, 24, 38), (0, 450, W, 180))

        # арена: каменный пол
        pg.draw.rect(bg, (46, 42, 52), (0, GROUND_Y, W, H - GROUND_Y))
        pg.draw.line(bg, (92, 82, 96), (0, GROUND_Y), (W, GROUND_Y), 3)
        for i in range(-6, 14):
            x0 = i * 110
            pg.draw.line(bg, (38, 34, 44), (x0, GROUND_Y), (x0 + 210, H), 2)
        for y in range(GROUND_Y + 28, H, 34):
            pg.draw.line(bg, (38, 34, 44), (0, y), (W, y), 2)

        # факелы по краям
        for x in (70, W - 70):
            pg.draw.rect(bg, (58, 44, 34), (x - 5, 470, 10, 150))
            pg.draw.ellipse(bg, (80, 50, 30), (x - 16, 455, 32, 22))
        return bg

    # ------------------------------------------------------------- primitives
    def text(self, s, pos, color=(230, 236, 245), font="text", center=False):
        pg = self.pg
        surf = self.fonts[font].render(s, True, color)
        rect = surf.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.screen.blit(surf, rect)

    def bar(self, x, y, w, h, frac, color, bg=(18, 20, 30), border=(120, 130, 150)):
        pg = self.pg
        pg.draw.rect(self.screen, bg, (x, y, w, h), border_radius=6)
        fw = int(max(0, min(1, frac)) * (w - 4))
        if fw > 0:
            pg.draw.rect(self.screen, color, (x + 2, y + 2, fw, h - 4), border_radius=5)
        pg.draw.rect(self.screen, border, (x, y, w, h), 2, border_radius=6)

    # --------------------------------------------------------------- fighters
    def draw_fighter(self, f: Fighter, t: float):
        pg = self.pg
        col = TEAM_COLORS[f.idx]
        body = (255, 255, 255) if f.hit_flash > 0 else col["body"]
        dark = (255, 255, 255) if f.hit_flash > 0 else col["dark"]

        x = f.x
        feet = f.y
        walk = math.sin(f.walk_phase)
        bob = 2.0 * math.sin(t * 4 + f.idx) if f.on_ground else 0.0
        hip_y = feet - 82 + bob
        shoulder_y = feet - 140 + bob
        head_y = feet - 164 + bob
        d = f.facing

        # тень
        pg.draw.ellipse(self.screen, (18, 16, 22),
                        (int(x - 34), int(feet - 8), 68, 16))

        # ноги
        if f.attack_kind == 2:
            elapsed = f.attack_duration - f.attack_timer
            p = max(0.0, min(1.0, elapsed / 12.0))
            kick_x = x + d * (26 + 82 * math.sin(min(1.0, p * 1.4) * math.pi))
            kick_y = hip_y + 30 - 22 * p
            knee = ((x + kick_x) / 2 + d * 6, (hip_y + kick_y) / 2)
            pg.draw.line(self.screen, dark, (x, hip_y), knee, 14)
            pg.draw.line(self.screen, dark, knee, (kick_x, kick_y), 12)
            pg.draw.circle(self.screen, col["trim"], (int(kick_x), int(kick_y)), 7)
            # опорная нога
            pg.draw.line(self.screen, dark, (x, hip_y), (x - d * 14, feet - 10), 14)
            pg.draw.line(self.screen, dark, (x - d * 14, feet - 10), (x - d * 20, feet), 12)
        else:
            for side, phase in ((1, walk), (-1, -walk)):
                fx = x + side * 12 + phase * 16
                knee = (x + side * 10 + phase * 8, (hip_y + feet) / 2)
                foot = (fx, feet - (4 if abs(phase) > 0.6 else 0))
                pg.draw.line(self.screen, dark, (x, hip_y), knee, 14)
                pg.draw.line(self.screen, dark, knee, foot, 12)

        # торс
        pg.draw.line(self.screen, body, (x, hip_y), (x, shoulder_y), 26)
        pg.draw.circle(self.screen, col["trim"], (int(x), int(shoulder_y - 2)), 7)

        # руки
        def arm_to(fist, hand_color=col["trim"]):
            elbow = ((x + fist[0]) / 2 + d * 4, (shoulder_y + fist[1]) / 2 + 6)
            pg.draw.line(self.screen, body, (x, shoulder_y), elbow, 12)
            pg.draw.line(self.screen, body, elbow, fist, 10)
            pg.draw.circle(self.screen, hand_color, (int(fist[0]), int(fist[1])), 6)

        if f.blocking:
            arm_to((x + d * 26, shoulder_y + 10), col["glow"])
            arm_to((x + d * 22, shoulder_y + 28), col["glow"])
        elif f.attack_kind == 1:
            elapsed = f.attack_duration - f.attack_timer
            p = max(0.0, min(1.0, elapsed / 8.0))
            reach = math.sin(min(1.0, p * 1.3) * math.pi / 2)
            fist = (x + d * (24 + 88 * reach), shoulder_y + 8)
            arm_to(fist, col["glow"])
            arm_to((x - d * 18, shoulder_y + 26))
        else:
            arm_to((x + d * 20 + d * walk * 12, shoulder_y + 34 - walk * 8))
            arm_to((x - d * 18 - d * walk * 12, shoulder_y + 38 + walk * 8))

        # голова
        pg.draw.circle(self.screen, col["trim"], (int(x), int(head_y)), 18)
        pg.draw.circle(self.screen, body, (int(x - d * 2), int(head_y - 2)), 15)
        # повязка
        pg.draw.rect(self.screen, col["glow"],
                     (int(x - 16), int(head_y - 10), 32, 6), border_radius=3)
        # глаз
        pg.draw.circle(self.screen, (20, 20, 28),
                       (int(x + d * 8), int(head_y - 1)), 2)

        # вспышка удара
        if f.attack_active:
            rng = PUNCH_RANGE if f.attack_kind == 1 else KICK_RANGE
            fx = x + d * rng
            fy = shoulder_y + (6 if f.attack_kind == 1 else 34)
            pg.draw.circle(self.screen, (*col["glow"],), (int(fx), int(fy)), 10, 2)

    # ------------------------------------------------------------------ hud
    def draw_hud(self, fight: Fight, trainer: Trainer, speed: int, paused: bool):
        pg = self.pg
        # здоровье
        hw = 470
        self.bar(40, 52, hw, 30, fight.p1.hp / 100.0,
                 (70, 210, 120) if fight.p1.hp > 35 else (230, 80, 70))
        self.bar(W - 40 - hw, 52, hw, 30, fight.p2.hp / 100.0,
                 (70, 210, 120) if fight.p2.hp > 35 else (230, 80, 70))
        self.text("ALEX-1", (44, 24), TEAM_COLORS[0]["glow"], "bold")
        right = self.fonts["bold"].render("ALEX-2", True, TEAM_COLORS[1]["glow"])
        self.screen.blit(right, (W - 44 - right.get_width(), 24))

        # награда эпизода
        self.text(f"награда {fight.p1.ep_reward:+.1f}", (44, 88), (200, 210, 225), "small")
        r2 = self.fonts["small"].render(f"награда {fight.p2.ep_reward:+.1f}", True, (200, 210, 225))
        self.screen.blit(r2, (W - 44 - r2.get_width(), 88))

        # итерация
        self.text(f"ИТЕРАЦИЯ {trainer.episode}", (W // 2, 30), (255, 220, 120), "hud", center=True)
        self.text(f"победы  {trainer.wins[0]} : {trainer.wins[1]}   ничьи {trainer.wins[2]}",
                  (W // 2, 66), (190, 200, 215), "small", center=True)
        self.text(f"ударов в эпизоде: {fight.p1.hits + fight.p2.hits}   "
                  f"лучший бой: {trainer.best_hits}", (W // 2, 102), (170, 180, 200), "small", center=True)

        # нижняя панель
        dev = "CUDA: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.text(f"обучение: {dev}", (44, H - 64), (150, 190, 225), "small")
        self.text(f"награда ср.: {trainer.avg(trainer.reward_history):+.2f}   "
                  f"loss: {trainer.last_loss:.3f}", (44, H - 42), (160, 170, 190), "small")
        self.text("Space пауза   1/2/3/4 скорость   S сохранить   Esc выход",
                  (44, H - 20), (120, 130, 150), "small")

        # скорость
        self.text(f"скорость {speed}x", (W - 44, H - 64), (230, 210, 150), "bold")
        self.text("каждая попытка = обновление весов", (W - 44, H - 40),
                  (120, 130, 150), "small")

        # история исходов
        hist = list(trainer.history)[-60:]
        bx, by, bw, bh = W - 400, 130, 356, 54
        pg.draw.rect(self.screen, (16, 20, 30), (bx, by, bw, bh), border_radius=8)
        pg.draw.rect(self.screen, (70, 80, 100), (bx, by, bw, bh), 1, border_radius=8)
        if hist:
            step = min(5, (bw - 12) / max(1, len(hist)))
            for i, w in enumerate(hist):
                c = (80, 200, 110) if w == 1 else (80, 150, 235) if w == 2 else (150, 150, 160)
                pg.draw.rect(self.screen, c, (bx + 6 + i * step, by + 10, max(2, step - 1), bh - 20))
        self.text("последние бои (зелёный = победа ALEX-1)", (bx, by - 20), (150, 160, 180), "small")

        # пауза
        if paused:
            self.text("ПАУЗА", (W // 2, H // 2 - 20), (255, 230, 140), "big", center=True)

        # баннер победы
        if fight.finished:
            if fight.winner == 0:
                msg, col = "НИЧЬЯ", (210, 210, 220)
            elif fight.winner == 1:
                msg, col = "ПОБЕДА ALEX-1  +20", TEAM_COLORS[0]["glow"]
            else:
                msg, col = "ПОБЕДА ALEX-2  +20", TEAM_COLORS[1]["glow"]
            overlay = pg.Surface((W, 150), pg.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, H // 2 - 90))
            self.text(msg, (W // 2, H // 2 - 18), col, "big", center=True)

    def draw(self, fight: Fight, trainer: Trainer, speed: int, paused: bool):
        pg = self.pg
        self.t += 0.016
        self.screen.blit(self.bg, (0, 0))

        # толпа
        for (cx, cy, c) in self.crowd:
            pg.draw.circle(self.screen, c, (cx, cy), 6)

        # пламя факелов
        for x in (70, W - 70):
            for k in range(3):
                r = 10 + 5 * math.sin(self.t * 9 + k + x)
                pg.draw.circle(self.screen, (255, 140 + k * 25, 40),
                               (x + int(4 * math.sin(self.t * 7 + k)), 452 - k * 8), int(r))

        # бойцы: дальний/ближний порядок не важен
        self.draw_fighter(fight.p1, self.t)
        self.draw_fighter(fight.p2, self.t)

        # искры
        for s in fight.sparks:
            k = s["ttl"]
            radius = (16 - k) * 2.2 + 4
            pg.draw.circle(self.screen, s["color"], (int(s["x"]), int(s["y"])), int(radius), 2)

        self.draw_hud(fight, trainer, speed, paused)
        pg.display.flip()

    def tick(self):
        return self.clock.tick(FPS)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            x = torch.zeros(1, device="cuda")
            _ = x + 1
            return torch.device("cuda")
        except Exception as e:
            print(f"[cuda] не работает ({e}), переключаюсь на CPU")
    return torch.device("cpu")


def headless(episodes: int, speed: int, device: torch.device, load: bool):
    trainer = Trainer(device, MODEL_DIR / "policy.pt", load=load)
    fight = Fight(trainer.policy, device)
    print(f"[train] устройство: {device}  эпизодов: {episodes}")
    t0 = time.time()
    for _ in range(episodes):
        fight.reset()
        while not fight.finished:
            for _ in range(speed):
                fight.step()
                if fight.finished:
                    break
        loss = trainer.learn(fight)
        if trainer.episode % 5 == 0 or trainer.episode <= 3:
            print(f"эп {trainer.episode:4d}  winner={fight.winner}  "
                  f"hp {fight.p1.hp:5.1f}:{fight.p2.hp:5.1f}  "
                  f"hits {fight.p1.hits}+{fight.p2.hits}  "
                  f"idle {fight.p1.idle_penalties}+{fight.p2.idle_penalties}  "
                  f"R {fight.p1.ep_reward:+6.2f}/{fight.p2.ep_reward:+6.2f}  "
                  f"loss {loss:.3f}  winrate {trainer.winrate():.2f}")
    dt = time.time() - t0
    print(f"[done] {episodes} эпизодов за {dt:.1f} c "
          f"({episodes / max(dt, 1e-9):.1f} эп/с), "
          f"средние удары {trainer.avg(trainer.hit_history):.1f}, "
          f"победы {trainer.wins}")
    trainer.save()
    print(f"[model] сохранено: {MODEL_DIR / 'policy.pt'}")


def windowed(speed: int, device: torch.device, load: bool):
    trainer = Trainer(device, MODEL_DIR / "policy.pt", load=load)
    fight = Fight(trainer.policy, device)
    renderer = Renderer(trainer, speed)

    paused = False
    phase = "fight"        # fight | ko
    ko_until = 0.0
    running = True
    announced = False

    print(f"[game] устройство: {device}; модель: {MODEL_DIR / 'policy.pt'}")
    print("[game] Space — пауза, 1-4 — скорость, S — сохранить, Esc — выход")

    while running:
        pg = renderer.pg
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                elif event.key == pg.K_SPACE:
                    paused = not paused
                elif event.key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
                    speed = int(event.unicode)
                elif event.key == pg.K_s:
                    trainer.save()
                    print("[model] сохранено вручную")

        if not paused:
            if phase == "fight":
                for _ in range(speed):
                    fight.step()
                    if fight.finished:
                        break
                if fight.finished:
                    trainer.learn(fight)
                    phase = "ko"
                    ko_until = time.time() + 1.6
                    announced = False
                    print(f"[ep {trainer.episode:4d}] winner={fight.winner} "
                          f"hits={fight.p1.hits}+{fight.p2.hits} "
                          f"R={fight.p1.ep_reward:+.1f}/{fight.p2.ep_reward:+.1f} "
                          f"loss={trainer.last_loss:.3f}")
            elif phase == "ko" and time.time() >= ko_until:
                fight.reset()
                phase = "fight"

        renderer.draw(fight, trainer, speed, paused)
        renderer.tick()

    trainer.save()
    pg.quit()
    print(f"[game] выход; модель сохранена ({trainer.episode} эпизодов)")


def main():
    ap = argparse.ArgumentParser(description="MK-AI: самообучающиеся бойцы")
    ap.add_argument("--headless", action="store_true", help="без окна, только обучение")
    ap.add_argument("--episodes", type=int, default=100, help="сколько эпизодов в headless")
    ap.add_argument("--speed", type=int, default=4, choices=[1, 2, 4, 8],
                    help="сколько шагов симуляции за кадр")
    ap.add_argument("--no-load", action="store_true", help="не загружать сохранённую модель")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    if args.headless:
        headless(args.episodes, args.speed, device, load=not args.no_load)
    else:
        windowed(args.speed, device, load=not args.no_load)


if __name__ == "__main__":
    main()
