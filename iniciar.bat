@echo off
REM ============================================================
REM  NeuroGuard 3D - Inicio rapido (doble clic en este archivo)
REM  1. Crea el entorno virtual con Python 3.13 (si no existe)
REM  2. Instala las librerias de requirements.txt
REM  3. Entrena la red si todavia no hay modelo guardado
REM  4. Inicia la interfaz y abre el navegador
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creando entorno virtual con Python 3.13...
    py -3.13 -m venv .venv
    if errorlevel 1 (
        echo.
        echo No se encontro Python 3.13. Instalelo desde https://www.python.org/downloads/
        echo TensorFlow todavia no funciona con Python 3.14.
        pause
        exit /b 1
    )
    echo Instalando librerias, puede tardar unos minutos...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

if not exist "modelo\datos.npz" (
    echo Entrenando la red neuronal...
    ".venv\Scripts\python.exe" red_neuronal_estudiantes.py
)

echo Abriendo http://127.0.0.1:5000 ...
start "" cmd /c "timeout /t 8 >nul & start http://127.0.0.1:5000"
".venv\Scripts\python.exe" app.py
pause
