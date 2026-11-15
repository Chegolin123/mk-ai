"""Сборка скриншотов и GIF для документации.

Скрипт запускает игру в невидимом режиме (SDL dummy), прогоняет бои текущей
политикой и сохраняет самые выразительные кадры:

    python tools/capture.py --shots     # docs/screenshots/*.png
    python tools/capture.py --gif       # docs/screenshots/fight.gif

Кадры подбираются по состоянию боя: удар рукой, пинок, нокдаун и т.п.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(ROOT))

import pygame  # noqa: E402

from game import Fight, MODEL_DIR, Renderer, Trainer, pick_device  # noqa: E402

SHOTS = ROOT / "docs" / "screenshots"


def save_png(screen, name: str) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    path = SHOTS / name
    pygame.image.save(screen, str(path))
    print(f"[shot] {path.relative_to(ROOT)}")


def capture_shots(trainer: Trainer, fight: Fight, renderer: Renderer, episodes: int) -> None:
    """Ищем удачные моменты боя и сохраняем их как PNG."""
    wanted = {
        "hero.png": lambda: (fight.p1.attack_active or fight.p2.attack_active)
        and len(fight.sparks) > 0,
        "punch.png": lambda: any(f.attack_kind == 1 and f.attack_active for f in fight.fighters),
        "kick.png": lambda: any(f.attack_kind == 2 and f.attack_active for f in fight.fighters),
        "ko.png": lambda: fight.finished,
    }
    taken: set[str] = set()

    for _ in range(episodes):
        fight.reset()
        while not fight.finished:
            fight.step()
            renderer.draw(fight, trainer, 4, False)
            for name, check in wanted.items():
                if name in taken:
                    continue
                if name == "ko.png" and not fight.finished:
                    continue
                if check():
                    save_png(renderer.screen, name)
                    taken.add(name)
            if len(taken) == len(wanted):
                return
        if len(taken) == len(wanted):
            return

    missing = set(wanted) - taken
    if missing:
        print("[shot] не найдены кадры:", ", ".join(sorted(missing)))


def capture_gif(trainer: Trainer, fight: Fight, renderer: Renderer,
                step: int = 3, max_frames: int = 220, width: int = 800) -> None:
    """Записывает бой в анимированный GIF (для README)."""
    from PIL import Image

    frames = []
    fight.reset()
    i = 0
    while not fight.finished and len(frames) < max_frames:
        fight.step()
        renderer.draw(fight, trainer, 4, False)
        if i % step == 0:
            raw = pygame.image.tobytes(renderer.screen, "RGB")
            img = Image.frombytes("RGB", (renderer.screen.get_width(),
                                          renderer.screen.get_height()), raw)
            h = int(width * img.height / img.width)
            frames.append(img.resize((width, h), Image.LANCZOS))
        i += 1

    if not frames:
        print("[gif] кадров нет")
        return

    SHOTS.mkdir(parents=True, exist_ok=True)
    path = SHOTS / "fight.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=50, loop=0, optimize=True)
    size_mb = path.stat().st_size / 1e6
    print(f"[gif] {path.relative_to(ROOT)} — {len(frames)} кадров, {size_mb:.1f} МБ")


def main() -> None:
    ap = argparse.ArgumentParser(description="скриншоты и GIF для документации")
    ap.add_argument("--shots", action="store_true", help="сохранить PNG-кадры")
    ap.add_argument("--gif", action="store_true", help="сохранить анимированный GIF")
    ap.add_argument("--episodes", type=int, default=12, help="сколько боёв просмотреть")
    ap.add_argument("--no-load", action="store_true", help="не загружать модель")
    args = ap.parse_args()

    if not (args.shots or args.gif):
        args.shots = args.gif = True

    device = pick_device()
    trainer = Trainer(device, MODEL_DIR / "policy.pt", load=not args.no_load)
    fight = Fight(trainer.policy, device)
    renderer = Renderer(trainer, 4)

    if args.shots:
        capture_shots(trainer, fight, renderer, args.episodes)
    if args.gif:
        capture_gif(trainer, fight, renderer)
    pygame.quit()


if __name__ == "__main__":
    main()
