$ErrorActionPreference = 'Stop'
$gesturePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $gesturePython)) {
    throw 'Missing environment. Follow miniProject/README.md to install dependencies.'
}
Push-Location $PSScriptRoot
try {
    & $gesturePython -m gesture_project.demo @args
    $gestureExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $gestureExitCode
