# Start the FastAPI backend and the Streamlit frontend (Windows PowerShell).
$projectPath = "$PSScriptRoot"
$venvPath = Join-Path $projectPath "venv"

if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}
. (Join-Path $venvPath "Scripts/Activate.ps1")
pip install -r (Join-Path $projectPath "requirements.txt")

$env:BLURANYTHING_API_URL = "http://localhost:8000"
$api = "cd '$projectPath'; uvicorn backend.app:app --reload"
$ui = "cd '$projectPath'; `$env:BLURANYTHING_API_URL='http://localhost:8000'; streamlit run frontend/streamlit_app.py"

Start-Process powershell -ArgumentList "-NoExit -Command `"$api`""
Start-Process powershell -ArgumentList "-NoExit -Command `"$ui`""
