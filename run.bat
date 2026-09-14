@echo off
setlocal enabledelayedexpansion

REM Sofia House Price Forecast - Windows launcher.
REM
REM On first run this script creates a Python virtual environment in .venv
REM and installs all dependencies from requirements.txt. Subsequent runs
REM skip straight to launching the Streamlit app.
REM
REM Requirements: Python 3.12 (or 3.11) installed and on PATH.
REM Download from: https://www.python.org/downloads/

cd /d "%~dp0"

REM ----- Locate Python --------------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Python was not found on PATH.
    echo.
    echo Install Python 3.12 from https://www.python.org/downloads/
    echo During install, tick "Add python.exe to PATH".
    echo Then re-run this script.
    echo.
    pause
    exit /b 1
)

REM ----- Create venv on first run --------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo First-time setup: creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )

    echo Installing dependencies. This downloads about 500 MB and takes 2-5 minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies.
        echo Check the messages above. The most common cause is a network problem
        echo or a missing C++ build tool for pmdarima.
        pause
        exit /b 1
    )
    echo.
    echo Setup complete.
)

REM ----- Launch the Streamlit app --------------------------------------------
echo.
echo Starting the Sofia House Price Forecast app...
echo A browser tab will open at http://localhost:8501
echo Close this window or press Ctrl+C to stop the app.
echo.

".venv\Scripts\streamlit.exe" run app\predict_price.py

pause
