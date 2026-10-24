@echo off
rem ---------------------------------------------------------------------------
rem  Drag one or more .brstm files onto this file to convert them to MP3.
rem  The MP3s are written next to the source files.
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo.
    echo   Drag a .brstm file onto convert.bat, or run:
    echo       convert.bat "C:\path\song.brstm"
    echo       convert.bat "C:\path\folder" -f flac --loops 2 --fade-out 8
    echo.
    pause
    exit /b 1
)

python -m brstm2mp3 %*
set RESULT=%ERRORLEVEL%
echo.
if %RESULT%==0 (echo Done.) else (echo Finished with errors, see messages above.)
pause
exit /b %RESULT%
