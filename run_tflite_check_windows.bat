@echo off
setlocal

if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found.
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python codes\model_input.py
pause
