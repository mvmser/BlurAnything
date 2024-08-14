# Define paths
$projectPath = "$PSScriptRoot"  # Assuming the script is located in the root of the project
$venvPath = Join-Path $projectPath "venv"
$requirementsFile = Join-Path $projectPath "requirements.txt"
$backendApp = Join-Path $projectPath "backend/app.py"
$frontendApp = Join-Path $projectPath "frontend/streamlit_app.py"

# Create virtual environment if it doesn't exist
if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

# Activate virtual environment
$activateScript = Join-Path $venvPath "Scripts/Activate.ps1"
. $activateScript

# Install requirements if not already installed
if (-not (Test-Path $requirementsFile)) {
    Write-Host "Requirements file not found."
} else {
    pip install -r $requirementsFile
}

$command1 = "cd '$projectPath'; uvicorn backend.app:app --reload"
$command2 = "cd '$projectPath\frontend'; streamlit run streamlit_app.py"

Start-Process powershell -ArgumentList "-NoExit -Command `"$command1`"" 
Start-Process powershell -ArgumentList "-NoExit -Command `"$command2`"" 
