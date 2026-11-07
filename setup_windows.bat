@echo off
setlocal

echo ========================================
echo Herbaura teammate setup
echo ========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found.
    echo Please install Python 3.10 or 3.11, then run this file again.
    pause
    exit /b 1
)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing project requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Requirement installation failed.
    pause
    exit /b 1
)

if not exist dataset\raw (
    mkdir dataset\raw
)

echo.
echo ========================================
echo Setup complete
echo ========================================
echo.
echo Put the Kaggle dataset class folders inside:
echo dataset\raw\
echo.
echo Then double-click run_training_windows.bat
echo.
pause
