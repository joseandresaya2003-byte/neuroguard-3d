@echo off
chcp 65001 >nul
REM =============================================================
REM  subir_a_huggingface.bat — Publica NeuroGuard 3D en internet GRATIS
REM  -------------------------------------------------------------
REM  Usa Hugging Face Spaces (servidor gratuito con 16 GB de RAM, suficiente
REM  para TensorFlow). Antes de ejecutarlo:
REM    1. Cree una cuenta gratis en https://huggingface.co/join
REM    2. Cree un token de ESCRITURA en https://huggingface.co/settings/tokens
REM  Luego haga doble clic en este archivo. Se puede repetir para ACTUALIZAR
REM  la página cuando cambie el código.
REM =============================================================

cd /d "%~dp0"

echo.
echo  === Publicar NeuroGuard 3D en Hugging Face Spaces ===
echo.

REM 1) Herramienta oficial de Hugging Face (comando "hf").
.venv\Scripts\python -m pip install -q -U huggingface_hub
if errorlevel 1 goto error

REM 2) Iniciar sesión (pega el token cuando lo pida; solo la primera vez).
.venv\Scripts\hf auth whoami >nul 2>&1
if errorlevel 1 .venv\Scripts\hf auth login
if errorlevel 1 goto error

set /p USUARIO=Escriba su nombre de usuario de Hugging Face:
set ESPACIO=%USUARIO%/neuroguard-3d

REM 3) Crear el Space (tipo Docker, público). Si ya existe, no pasa nada.
.venv\Scripts\hf repos create %ESPACIO% --repo-type space --space-sdk docker --public --exist-ok
if errorlevel 1 goto error

REM 4) Subir el proyecto (sin el entorno virtual ni archivos temporales).
.venv\Scripts\hf upload %ESPACIO% . . --repo-type space --commit-message "Publicar NeuroGuard 3D" --exclude ".venv/*" --exclude "__pycache__/*" --exclude "*.pyc" --exclude ".vscode/*" --exclude "README.md" --exclude "despliegue/*" --exclude "iniciar.bat" --exclude "subir_a_huggingface.bat"
if errorlevel 1 goto error

REM 5) README especial que le dice a Hugging Face que es una app Docker en el puerto 7860.
.venv\Scripts\hf upload %ESPACIO% despliegue\README_space.md README.md --repo-type space --commit-message "Configuracion del Space"
if errorlevel 1 goto error

echo.
echo  Listo. El servidor tarda unos 5-10 minutos en construirse la primera vez.
echo  Pagina publica:  https://huggingface.co/spaces/%ESPACIO%
echo  Enlace directo:  https://%USUARIO%-neuroguard-3d.hf.space
echo.
pause
exit /b 0

:error
echo.
echo  Hubo un error. Revise el mensaje de arriba (usuario, token o conexion a internet).
pause
exit /b 1
