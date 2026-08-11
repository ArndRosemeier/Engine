@echo off
cd /d "%~dp0"
python run_examples.py
if errorlevel 1 pause
