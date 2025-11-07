@echo off
setlocal enabledelayedexpansion

REM Resolve the directory of this script
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%"

REM Ensure input and output directories exist
if not exist "input" mkdir "input"
if not exist "output" mkdir "output"

REM Run the Python script with Windows launcher. Pass through any args.
REM Without args, the script auto-discovers two .xlsx files in .\input and writes to .\output
py -3 "%SCRIPT_DIR%compare_excel_sec_id_stream.py" %*
set EXITCODE=%ERRORLEVEL%

popd

REM Keep the window open if launched by double-click
if "%EXITCODE%" NEQ "0" (
  echo.
  echo Script exited with code %EXITCODE%.
  pause
)

exit /b %EXITCODE%
