# =============================================================
# Dockerfile — NeuroGuard 3D en un servidor (Hugging Face Spaces)
# -------------------------------------------------------------
# Describe la "máquina" donde corre la página en internet:
#   1. Parte de Linux con Python 3.13 (la misma versión del proyecto).
#   2. Instala las librerías de requirements-servidor.txt.
#   3. Copia el proyecto (incluye el modelo ya entrenado en modelo/).
#   4. Arranca el servidor web con gunicorn en el puerto 7860.
# =============================================================

FROM python:3.13-slim

# Mensajes de Python inmediatos y menos avisos de TensorFlow en los registros.
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# Hugging Face ejecuta el contenedor con el usuario 1000 (no como administrador).
RUN useradd -m -u 1000 usuario
WORKDIR /home/usuario/app

# Primero solo las librerías: así Docker reutiliza este paso si solo cambia el código.
COPY --chown=usuario requirements-servidor.txt .
RUN pip install -r requirements-servidor.txt

# Luego el proyecto completo (lo que no sirve en el servidor está en .dockerignore).
COPY --chown=usuario . .
USER usuario

EXPOSE 7860

# Gunicorn = servidor web para producción.
#   --workers 1  : UN solo proceso, porque el laboratorio guarda su estado en memoria.
#   --threads 8  : atiende a varias personas a la vez dentro de ese proceso.
#   --timeout 180: margen para la primera carga del modelo.
CMD gunicorn --workers 1 --threads 8 --timeout 180 --bind 0.0.0.0:${PORT} app:app
