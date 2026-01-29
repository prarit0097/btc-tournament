@echo off
cd /d e:\btc
e:\btc\.venv311\Scripts\python.exe -m btc_tournament.run_hourly >> e:\btc\data\tournament_task.log 2>&1
