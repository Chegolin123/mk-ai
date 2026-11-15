@echo off
rem Запуск окна игры. Приоритет: .venv рядом с проектом, затем MK_AI_PYTHON, затем python из PATH.
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" game.py %*
    goto :eof
)

if defined MK_AI_PYTHON (
    "%MK_AI_PYTHON%" game.py %*
    goto :eof
)

python game.py %*
