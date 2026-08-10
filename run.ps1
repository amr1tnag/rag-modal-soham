# Start the app on Windows.  Usage:  .\run.ps1
#
# Calls the virtual environment's python.exe by full path, so it works in a
# fresh terminal with nothing activated — the failure everyone hits otherwise.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "No virtual environment found. Creating one..." -ForegroundColor Yellow
    python -m venv .venv
    if (-not (Test-Path $py)) {
        Write-Host "Could not create it. Is Python installed and on PATH?" -ForegroundColor Red
        Write-Host "Get it from https://www.python.org/downloads/ and tick" -ForegroundColor Red
        Write-Host '"Add python.exe to PATH" during setup.' -ForegroundColor Red
        exit 1
    }
}

# Cheap check — reinstalling every launch would add seconds to startup.
& $py -c "import uvicorn, fastapi, numpy, ollama, pypdf" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    & $py -m pip install --quiet --upgrade pip
    & $py -m pip install --quiet -r requirements.txt
}

try {
    Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null
} catch {
    Write-Host "Ollama does not seem to be running." -ForegroundColor Yellow
    Write-Host "Open the Ollama app, or run 'ollama serve' in another terminal."
    Write-Host "Starting anyway — the sidebar will show a red dot until it is up."
    Write-Host ""
}

Write-Host "Open http://localhost:8000  (Ctrl+C to stop)" -ForegroundColor Green
Write-Host ""
& $py -m uvicorn app.main:app --reload
