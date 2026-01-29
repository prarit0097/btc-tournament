@echo off
cd /d e:\btc
if exist e:\btc\.venv311\Scripts\python.exe (
  e:\btc\.venv311\Scripts\python.exe -m btc_tournament.run_hourly >> e:\btc\data\tournament_task.log 2>&1
  exit /b
)
if exist e:\btc\.venv\Scripts\python.exe (
  e:\btc\.venv\Scripts\python.exe -m btc_tournament.run_hourly >> e:\btc\data\tournament_task.log 2>&1
  exit /b
)
py -3 -m btc_tournament.run_hourly >> e:\btc\data\tournament_task.log 2>&1
