@echo off
REM Setup Windows Task Scheduler for automated dependency scanning
REM This creates a weekly scheduled task to run dependency scans

echo.
echo ================================================
echo  Setting up automated dependency scanning
echo ================================================
echo.

set TASK_NAME=SnakeAI_DependencyScan
set SCRIPT_PATH=%~dp0scan_dependencies.bat
set LOG_PATH=%~dp0logs\scheduled_scans.log

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

echo Task Name: %TASK_NAME%
echo Script Path: %SCRIPT_PATH%
echo Log Path: %LOG_PATH%
echo.

REM Delete existing task if it exists
schtasks /query /tn "%TASK_NAME%" >nul 2>&1
if %errorlevel% == 0 (
    echo Removing existing scheduled task...
    schtasks /delete /tn "%TASK_NAME%" /f
)

REM Create new scheduled task (runs every Monday at 9:00 AM)
echo Creating scheduled task...
schtasks /create /tn "%TASK_NAME%" /tr "\"%SCRIPT_PATH%\" > \"%LOG_PATH%\" 2>&1" /sc weekly /d MON /st 09:00 /ru SYSTEM

if %errorlevel% == 0 (
    echo.
    echo ✓ Successfully created scheduled task!
    echo.
    echo The task will run every Monday at 9:00 AM
    echo Logs will be saved to: %LOG_PATH%
    echo.
    echo To manage the task:
    echo - View: schtasks /query /tn "%TASK_NAME%"
    echo - Run now: schtasks /run /tn "%TASK_NAME%"
    echo - Delete: schtasks /delete /tn "%TASK_NAME%" /f
    echo.
) else (
    echo ✗ Failed to create scheduled task
    echo You may need to run this script as Administrator
)

pause
