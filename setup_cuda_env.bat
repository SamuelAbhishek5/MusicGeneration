@echo off
REM Setup script to create a conda environment with CUDA-enabled PyTorch
REM This script creates a new environment called 'musicgen-cuda' with Python 3.11 and CUDA support

echo Creating conda environment with Python 3.11 and CUDA support...
conda create -n musicgen-cuda python=3.11 -y

echo Activating environment...
call conda activate musicgen-cuda

echo Installing PyTorch with CUDA 11.8 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Installing other dependencies...
pip install flask flask-cors requests numpy scipy transformers sentencepiece gunicorn uvicorn

echo.
echo Setup complete!
echo.
echo To use this environment in the future, run:
echo   conda activate musicgen-cuda
echo.
echo Then start the server with:
echo   python musicserver.py
echo.
