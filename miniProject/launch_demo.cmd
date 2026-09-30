@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Missing environment. Follow README.md to install dependencies.
    pause
    popd
    exit /b 1
)
".venv\Scripts\python.exe" -m gesture_project.demo %*
set "gestureExitCode=%ERRORLEVEL%"
if not "%gestureExitCode%"=="0" pause
popd
exit /b %gestureExitCode%
