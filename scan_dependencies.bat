@echo off
REM Streamlined Dependency Security Scanner for Snake AI
REM Fast scan with pip-audit + basic checks

echo.
echo ========================================
echo  Snake AI Security Scanner (Fast Mode)
echo ========================================
echo.

REM Change to the project directory
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found
)

REM Run the streamlined Python scanner
python check_dependencies.py

echo.
echo For comprehensive scan, run: python check_dependencies_full.py
echo.
echo Scan complete. Press any key to exit...
pause > nul
