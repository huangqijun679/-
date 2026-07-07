@echo off
call py -m pip install -r requirements.txt >nul 2>&1
py -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
pause
