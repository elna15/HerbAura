@echo off
setlocal

if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found.
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)

if not exist dataset\raw (
    echo Dataset folder not found.
    echo Put the Kaggle dataset inside dataset\raw first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python codes\train_cnn.py
pause
