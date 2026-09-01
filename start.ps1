param()

$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Project .venv not found. Follow README.md to install dependencies first.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $projectPython -m streamlit run app.py
}
finally {
    Pop-Location
}
