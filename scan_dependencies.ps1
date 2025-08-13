# Streamlined Dependency Security Scanner for Snake AI (PowerShell version)
# Fast scan with pip-audit + basic checks

Write-Host ""
Write-Host "========================================"
Write-Host " Snake AI Security Scanner (Fast Mode)"
Write-Host "========================================"
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Activate virtual environment if it exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..."
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "Warning: Virtual environment not found" -ForegroundColor Yellow
}

# Run the streamlined Python scanner
python check_dependencies.py

Write-Host ""
Write-Host "For comprehensive scan: python check_dependencies_full.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "Scan complete. Check the output above for any issues." -ForegroundColor Green
