"""Servidor de la interfaz gráfica "NeuroGuard 3D".

Tecnología: Flask (Python) + HTML, CSS y JavaScript (Three.js para el 3D).

¿CÓMO SE COMUNICA LA INTERFAZ CON EL MODELO?
    1. red_neuronal_estudiantes.py entrena la red y guarda en "modelo/":
         - red_neuronal.keras  -> arquitectura, pesos y sesgos entrenados
         - normalizador.pkl    -> media y desviación del StandardScaler
         - info_modelo.json    -> métricas, historial por época y datos de prueba
    2. Este servidor carga esos archivos UNA sola vez al iniciar.
    3. El navegador (static/js/app.js) envía los datos escritos por el usuario
       con fetch() a la ruta POST /api/analizar en formato JSON.
    4. El servidor valida los datos, los normaliza con el normalizador
       entrenado y llama a modelo.predict(). La probabilidad que se muestra
       sale de la red real: no hay resultados fijos ni inventados.
    5. Además calcula, con los MISMOS pesos de la red, los valores z y la
       activación de cada neurona para que la vista 3D los dibuje.

CÓMO EJECUTAR
    python app.py        y abrir  http://127.0.0.1:5000
"""

import json  # Lee el archivo de información del modelo.
import mimetypes  # Tipos de archivo correctos para MediaPipe (.mjs y .wasm).
import math  # Permite revisar si un número es válido (no NaN / infinito).
import os  # Manejo de rutas.
import runpy  # Ejecuta el script de entrenamiento si falta el modelo.
import sys
import threading  # El laboratorio entrena en segundo plano sin bloquear la interfaz.

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Menos mensajes de TF.

import joblib  # Carga el normalizador guardado.
import numpy as np  # Cálculos numéricos.
from flask import Flask, has_request_context, jsonify, render_template, request  # Servidor web.
import keras  # Keras (viene con TensorFlow): carga la red neuronal entrenada.

# El control con las manos (MediaPipe) necesita que el navegador reciba estos
# archivos con su tipo correcto; Windows a veces no los tiene registrados.
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
from sklearn.metrics import (  # Métricas que se recalculan al cambiar el umbral.
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ------------------------------------------------------------
# Rutas de los archivos generados por el entrenamiento
# ------------------------------------------------------------
CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA_MODELO = os.path.join(CARPETA_BASE, "modelo")
RUTA_MODELO = os.path.join(CARPETA_MODELO, "red_neuronal.keras")
RUTA_NORMALIZADOR = os.path.join(CARPETA_MODELO, "normalizador.pkl")
RUTA_INFO = os.path.join(CARPETA_MODELO, "info_modelo.json")
RUTA_DATOS = os.path.join(CARPETA_MODELO, "datos.npz")

# Límites de validación de las entradas.
MAX_INTENTOS = 100  # Máximo aceptado (la red se entrenó con 0-20).
MAX_TRAFICO = 10000  # Máximo aceptado en MB (la red se entrenó con 0-2000).


# ------------------------------------------------------------
# Carga del modelo (si no existe, se entrena automáticamente)
# ------------------------------------------------------------
def cargar_o_entrenar():
    """Carga el modelo entrenado. Si no existe, ejecuta el entrenamiento."""
    faltan = not all(
        os.path.exists(r)
        for r in (RUTA_MODELO, RUTA_NORMALIZADOR, RUTA_INFO, RUTA_DATOS)
    )
    if faltan:
        print("No se encontró el modelo entrenado. Entrenando la red...")
        runpy.run_path(os.path.join(CARPETA_BASE, "red_neuronal_estudiantes.py"))

    modelo_cargado = keras.models.load_model(RUTA_MODELO)
    normalizador_cargado = joblib.load(RUTA_NORMALIZADOR)
    with open(RUTA_INFO, encoding="utf-8") as archivo:
        info_cargada = json.load(archivo)
    return modelo_cargado, normalizador_cargado, info_cargada


modelo, normalizador, info = cargar_o_entrenar()

# ------------------------------------------------------------
# UMBRAL DE DECISIÓN (modificable desde la interfaz)
# ------------------------------------------------------------
# El umbral oficial (0.5) viene del entrenamiento (info_modelo.json). La
# interfaz puede cambiarlo con POST /api/umbral SIN tocar el código ni volver
# a entrenar: los pesos de la red no cambian, solo la regla "p >= umbral".
UMBRAL_OFICIAL = float(info["umbral"])
UMBRAL_MIN, UMBRAL_MAX = 0.05, 0.95  # Límites permitidos en la interfaz.


def umbral_actual():
    """
    Umbral de decisión de QUIEN hace la petición.

    Cada navegador envía su umbral en la cabecera HTTP "X-Umbral". Así, si la
    página está publicada en internet, varias personas pueden usarla a la vez
    y cada una mueve SU umbral sin cambiárselo a las demás. Si la cabecera no
    llega o no es válida, se usa el umbral oficial del taller (0.5).
    """
    if has_request_context():
        try:
            valor = float(request.headers.get("X-Umbral", ""))
            if math.isfinite(valor) and UMBRAL_MIN <= valor <= UMBRAL_MAX:
                return round(valor, 2)
        except ValueError:
            pass
    return UMBRAL_OFICIAL


# Datos normalizados del entrenamiento oficial (prueba y laboratorio).
DATOS = np.load(RUTA_DATOS)
# Probabilidades del modelo oficial para los 400 datos de prueba. Se calculan
# una sola vez: al cambiar el umbral solo hay que volver a compararlas.
PROB_PRUEBA = modelo.predict(DATOS["X_prueba"], verbose=0).ravel()


def metricas_con(probabilidades, umbral):
    """Exactitud, precisión, recall, F1 y matriz de confusión para un umbral.

    Las probabilidades no cambian; solo cambia cuáles se consideran
    sospechosas (p >= umbral). Por eso no hace falta volver a entrenar.
    """
    y_real = DATOS["y_prueba"]
    y_pred = (np.asarray(probabilidades) >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    return {
        "exactitud": float(accuracy_score(y_real, y_pred)),
        "precision": float(precision_score(y_real, y_pred, zero_division=0)),
        "recall": float(recall_score(y_real, y_pred, zero_division=0)),
        "f1": float(f1_score(y_real, y_pred, zero_division=0)),
        "matriz_confusion": {"vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp)},
    }


def metricas_oficiales():
    """Métricas del modelo oficial con el umbral vigente (la pérdida no depende del umbral)."""
    return {"perdida": info["metricas"]["perdida"], **metricas_con(PROB_PRUEBA, umbral_actual())}


def muestra_prueba():
    """Datos de prueba con su probabilidad y la clase predicha con el umbral vigente."""
    u = umbral_actual()
    return [
        {**punto, "probabilidad": round(float(p), 4), "predicho": int(p >= u)}
        for punto, p in zip(info["muestra_prueba"], PROB_PRUEBA)
    ]

# Pesos y sesgos de cada capa: [(W, b, activación), ...]
# W tiene forma (entradas_de_la_capa, neuronas_de_la_capa).
CAPAS = [
    (capa.get_weights()[0], capa.get_weights()[1], capa.get_config()["activation"])
    for capa in modelo.layers
]


# ------------------------------------------------------------
# Matemática de la red (la misma que hace Keras por dentro)
# ------------------------------------------------------------
def relu(z):
    """ReLU(z) = max(0, z)."""
    return np.maximum(0.0, z)


def sigmoide(z):
    """Sigmoide(z) = 1 / (1 + e^(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


# 🧠 CEREBRO EN LA INTERFAZ: misma operación que hace la red, capa por capa.
def propagacion_adelante(x_normalizado):
    """Recorre la red capa por capa y guarda z y la activación de cada neurona.

    Para cada capa:  z = a_anterior · W + b   y   a = f(z)
    """
    activacion = np.asarray(x_normalizado, dtype=float)
    recorrido = []
    for W, b, nombre_activacion in CAPAS:
        # 🧠 AQUÍ SE HACE LA OPERACIÓN DE CADA NEURONA: z = Σ(w·x) + b
        z = activacion @ W + b
        activacion = relu(z) if nombre_activacion == "relu" else sigmoide(z)
        recorrido.append(
            {"activacion": nombre_activacion, "z": z.tolist(), "a": activacion.tolist()}
        )
    return recorrido


def validar(datos):
    """Valida lo que envía el usuario. Devuelve (intentos, tráfico, errores)."""
    errores = {}
    intentos = trafico = None

    # --- Intentos fallidos: entero, no negativo ---
    valor = datos.get("intentos")
    try:
        numero = float(valor)
        if not math.isfinite(numero):
            raise ValueError
        if not numero.is_integer():
            errores["intentos"] = "Los intentos fallidos deben ser un número entero."
        elif numero < 0:
            errores["intentos"] = "Los intentos fallidos no pueden ser negativos."
        elif numero > MAX_INTENTOS:
            errores["intentos"] = f"El máximo permitido es {MAX_INTENTOS} intentos."
        else:
            intentos = int(numero)
    except (TypeError, ValueError):
        errores["intentos"] = "Escriba un número válido de intentos fallidos."

    # --- Tráfico enviado: decimal, no negativo ---
    valor = datos.get("trafico")
    try:
        numero = float(valor)
        if not math.isfinite(numero):
            raise ValueError
        if numero < 0:
            errores["trafico"] = "El tráfico enviado no puede ser negativo."
        elif numero > MAX_TRAFICO:
            errores["trafico"] = f"El máximo permitido es {MAX_TRAFICO} MB."
        else:
            trafico = round(numero, 2)
    except (TypeError, ValueError):
        errores["trafico"] = "Escriba un número válido de megabytes."

    return intentos, trafico, errores


def construir_explicacion(p, p_sin_intentos, p_sin_trafico, p_promedio):
    """Arma una explicación en lenguaje sencillo a partir de la salida de la red.

    Se compara la conexión con una "conexión promedio" (media de los datos de
    entrenamiento). Para repartir la diferencia entre las dos entradas de forma
    justa se usa el valor de Shapley con dos variables:
        efecto_x1 = ½·[(p − p_sin_x1) + (p_sin_x2 − p_promedio)]
        efecto_x2 = ½·[(p − p_sin_x2) + (p_sin_x1 − p_promedio)]
    donde p_sin_xi = probabilidad si la entrada i tuviera su valor promedio.
    Así efecto_x1 + efecto_x2 = p − p_promedio exactamente.
    Todas las probabilidades salen de la red real (modelo.predict).
    """
    efecto_intentos = 0.5 * ((p - p_sin_intentos) + (p_sin_trafico - p_promedio))
    efecto_trafico = 0.5 * ((p - p_sin_trafico) + (p_sin_intentos - p_promedio))
    umbral = umbral_actual()
    sospechosa = p >= umbral

    frases = []
    if sospechosa:
        frases.append(
            f"La red calculó {p:.1%}, valor mayor o igual al umbral {umbral:.2f}, "
            "por eso clasifica la conexión como SOSPECHOSA."
        )
    else:
        frases.append(
            f"La red calculó {p:.1%}, valor menor que el umbral {umbral:.2f}, "
            "por eso clasifica la conexión como NORMAL."
        )

    def describir(nombre, efecto):
        if abs(efecto) < 0.02:
            return f"{nombre} casi no cambió la decisión"
        sentido = "subió" if efecto > 0 else "bajó"
        return f"{nombre} {sentido} la probabilidad {abs(efecto) * 100:.1f} puntos"

    frases.append(
        f"Frente a una conexión promedio (p = {p_promedio:.1%}): "
        + describir("el número de intentos fallidos", efecto_intentos)
        + " y "
        + describir("el tráfico enviado", efecto_trafico)
        + "."
    )
    distancia = abs(p - umbral)
    if distancia < 0.10:
        frases.append("La probabilidad está cerca del umbral: es un caso dudoso.")
    elif distancia > 0.40:
        frases.append("La probabilidad está lejos del umbral: la red está muy segura.")
    return " ".join(frases), {"intentos": efecto_intentos, "trafico": efecto_trafico}


# ------------------------------------------------------------
# Servidor web
# ------------------------------------------------------------
app = Flask(__name__)


@app.get("/")
def inicio():
    """Entrega la página principal de la interfaz."""
    return render_template("index.html")


@app.get("/api/modelo")
def datos_modelo():
    """Envía al navegador la estructura, pesos, métricas e historial reales."""
    capas = [
        {
            "neuronas": int(W.shape[1]),
            "activacion": act,
            "pesos": W.tolist(),
            "sesgos": b.tolist(),
        }
        for W, b, act in CAPAS
    ]
    return jsonify(
        {
            "arquitectura": info["arquitectura"],
            "umbral": umbral_actual(),
            "umbral_oficial": UMBRAL_OFICIAL,
            "umbral_limites": [UMBRAL_MIN, UMBRAL_MAX],
            "capas": capas,
            "parametros_entrenables": info["parametros_entrenables"],
            "normalizador": {
                "media": normalizador.mean_.tolist(),
                "desviacion": normalizador.scale_.tolist(),
            },
            "datos": info["datos"],
            "metricas": metricas_oficiales(),
            # Métricas con el umbral oficial, para comparar cuando se mueve el umbral.
            "metricas_umbral_oficial": {
                "perdida": info["metricas"]["perdida"],
                **metricas_con(PROB_PRUEBA, UMBRAL_OFICIAL),
            },
            "historial": info["historial"],
            "muestra_prueba": muestra_prueba(),
            "limites": {"intentos": MAX_INTENTOS, "trafico": MAX_TRAFICO},
        }
    )


@app.post("/api/analizar")
def analizar():
    """Recibe {intentos, trafico}, consulta la red y devuelve el resultado."""
    datos = request.get_json(silent=True) or {}
    intentos, trafico, errores = validar(datos)
    if errores:
        return jsonify({"ok": False, "errores": errores}), 400

    # 1) Se arma un lote con la conexión del usuario y 3 variantes de comparación
    #    (cada entrada reemplazada por su promedio) y se consulta la red UNA vez.
    media_intentos, media_trafico = normalizador.mean_
    lote = np.array(
        [
            [intentos, trafico],  # conexión del usuario
            [media_intentos, trafico],  # intentos en su valor promedio
            [intentos, media_trafico],  # tráfico en su valor promedio
            [media_intentos, media_trafico],  # conexión promedio
        ],
        dtype=float,
    )
    lote_normalizado = normalizador.transform(lote)  # Misma normalización del entrenamiento.
    probabilidades = modelo.predict(lote_normalizado, verbose=0).ravel()  # Red real.
    p, p_sin_intentos, p_sin_trafico, p_promedio = (float(v) for v in probabilidades)
    x_normalizado = lote_normalizado[0]

    # 2) Recorrido neurona por neurona con los mismos pesos (para el 3D).
    recorrido = propagacion_adelante(x_normalizado)

    # 3) Explicación e influencia de cada entrada.
    explicacion, efectos = construir_explicacion(
        p, p_sin_intentos, p_sin_trafico, p_promedio
    )

    # Advertencias: valores válidos pero fuera del rango de entrenamiento.
    advertencias = []
    if intentos > 20:
        advertencias.append(
            "La red se entrenó con 0 a 20 intentos; con valores mayores está extrapolando."
        )
    if trafico > 2000:
        advertencias.append(
            "La red se entrenó con 0 a 2000 MB; con valores mayores está extrapolando."
        )

    umbral = umbral_actual()
    sospechosa = p >= umbral
    return jsonify(
        {
            "ok": True,
            "entrada": {"intentos": intentos, "trafico": trafico},
            "normalizada": x_normalizado.tolist(),
            "probabilidad": p,
            "umbral": umbral,
            "clase": int(sospechosa),
            "clasificacion": "SOSPECHOSA" if sospechosa else "NORMAL",
            "explicacion": explicacion,
            "efectos": efectos,
            "advertencias": advertencias,
            "recorrido": recorrido,
        }
    )


@app.post("/api/umbral")
def cambiar_umbral():
    """Cambia el umbral de decisión y devuelve las métricas recalculadas.

    Recibe {"umbral": 0.35}. La red NO se vuelve a entrenar: sus
    probabilidades son las mismas, solo cambia la regla p >= umbral.
    """
    datos = request.get_json(silent=True) or {}
    try:
        valor = float(datos.get("umbral"))
        if not math.isfinite(valor):
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "El umbral debe ser un número."}), 400
    if not UMBRAL_MIN <= valor <= UMBRAL_MAX:
        return jsonify(
            {"ok": False, "error": f"El umbral debe estar entre {UMBRAL_MIN} y {UMBRAL_MAX}."}
        ), 400
    umbral = round(valor, 2)  # No se guarda en el servidor: es el umbral de este usuario.
    return jsonify(
        {
            "ok": True,
            "umbral": umbral,
            "metricas": {"perdida": info["metricas"]["perdida"], **metricas_con(PROB_PRUEBA, umbral)},
            "predicciones": [int(p >= umbral) for p in PROB_PRUEBA],
        }
    )


# ------------------------------------------------------------
# Mapa de decisión: la red evaluada en una cuadrícula de puntos
# ------------------------------------------------------------
def cuadricula(n):
    """Puntos (intentos, tráfico) de una cuadrícula n x n, ya normalizados."""
    valores_intentos = np.linspace(0, 20, n)
    valores_trafico = np.linspace(0, 2000, n)
    malla_i, malla_t = np.meshgrid(valores_intentos, valores_trafico)
    puntos = np.column_stack((malla_i.ravel(), malla_t.ravel()))
    return valores_intentos, valores_trafico, normalizador.transform(puntos)


def mapa_de(red, n):
    """Probabilidad que da 'red' en cada punto de la cuadrícula n x n."""
    valores_intentos, valores_trafico, puntos = cuadricula(n)
    # Se llama a la red directamente (más rápido que predict para lotes pequeños).
    probabilidades = np.asarray(red(puntos, training=False)).ravel()
    return {
        "n": n,
        "intentos": valores_intentos.tolist(),
        "trafico": valores_trafico.tolist(),
        # Fila = valor de tráfico, columna = valor de intentos.
        "probabilidades": probabilidades.reshape(n, n).round(4).tolist(),
    }


_mapa_cache = None  # El mapa del modelo oficial se calcula una sola vez.


@app.get("/api/mapa")
def mapa_decision():
    """Mapa del modelo oficial en 61 x 61 puntos (0-20 intentos, 0-2000 MB)."""
    global _mapa_cache
    if _mapa_cache is None:
        _mapa_cache = mapa_de(modelo, 61)
    return jsonify(_mapa_cache)


# ============================================================
# LABORATORIO: entrenar redes experimentales y compararlas
# ============================================================
# Objetivo pedagógico del taller: "Modificar la arquitectura y observar cómo
# cambian los resultados". El usuario elige la activación de las capas ocultas,
# la tasa de aprendizaje, las neuronas y las épocas. La red se entrena con LOS
# MISMOS datos del modelo oficial (modelo/datos.npz) y, cada pocas épocas, se
# envía su mapa de decisión a la interfaz para ver cómo aprende "en vivo".
# El modelo oficial 2-8-4-1 con ReLU NO se modifica.

ACTIVACIONES = {"relu": "ReLU", "tanh": "Tanh", "sigmoid": "Sigmoide", "linear": "Lineal"}
TASAS = [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

_candado = threading.Lock()  # Evita que dos hilos modifiquen el estado a la vez.
laboratorio = {
    "estado": "inactivo",  # inactivo | entrenando | terminado | detenido | error
    "config": None,
    "epoca": 0,
    "historial": {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []},
    "mapa": None,
    "resultado": None,
    "detener": False,
    "version": 0,  # Aumenta con cada cambio: la interfaz sabe si hay algo nuevo.
    "generacion": 0,  # Aumenta al reiniciar: un hilo viejo ya no puede escribir.
}
# Tabla de experimentos; la primera fila es el modelo oficial del taller.
# Cada fila guarda sus probabilidades de prueba ("_prob") para recalcular
# exactitud, precisión y recall cuando el usuario cambia el umbral.
def fila_oficial():
    """Fila de la tabla que corresponde al modelo oficial 2-8-4-1."""
    return {
        "nombre": "Oficial",
        "activacion": "ReLU",
        "arquitectura": info["arquitectura"],
        "tasa": 0.001,
        "epocas": len(info["historial"]["loss"]),
        "perdida": info["metricas"]["perdida"],
        "_prob": PROB_PRUEBA,
    }


experimentos = [fila_oficial()]


def tabla_experimentos():
    """Filas de la tabla con las métricas calculadas con el umbral vigente."""
    filas = []
    for e in experimentos:
        m = metricas_con(e["_prob"], umbral_actual())
        filas.append(
            {
                **{k: v for k, v in e.items() if k != "_prob"},
                "exactitud": m["exactitud"],
                "precision": m["precision"],
                "recall": m["recall"],
            }
        )
    return filas


def validar_config(datos):
    """Revisa la configuración enviada por el laboratorio de la interfaz."""
    try:
        config = {
            "activacion": str(datos.get("activacion", "relu")),
            "tasa": float(datos.get("tasa", 0.001)),
            "epocas": int(datos.get("epocas", 100)),
            "neuronas1": int(datos.get("neuronas1", 8)),
            "neuronas2": int(datos.get("neuronas2", 4)),
        }
    except (TypeError, ValueError):
        return None, "La configuración contiene valores no numéricos."
    if config["activacion"] not in ACTIVACIONES:
        return None, "Activación no válida."
    if config["tasa"] not in TASAS:
        return None, "Tasa de aprendizaje no válida."
    if not 5 <= config["epocas"] <= 300:
        return None, "Las épocas deben estar entre 5 y 300."
    if not 1 <= config["neuronas1"] <= 16 or not 1 <= config["neuronas2"] <= 8:
        return None, "Neuronas fuera de rango (capa 1: 1-16, capa 2: 1-8)."
    return config, None


class ProgresoLaboratorio(keras.callbacks.Callback):
    """Keras ejecuta esta clase al final de cada época: guarda progreso y mapa."""

    def __init__(self, total_epocas, generacion):
        super().__init__()
        self.generacion = generacion  # Si la plataforma se reinicia, deja de escribir.
        # El mapa se recalcula unas 60 veces en total (no en cada época, por velocidad).
        self.cada = max(1, total_epocas // 60)
        self.total = total_epocas

    def on_epoch_end(self, epoca, logs=None):
        logs = logs or {}
        actualizar_mapa = epoca % self.cada == 0 or epoca == self.total - 1
        mapa = mapa_de(self.model, 41) if actualizar_mapa else None
        with _candado:
            if laboratorio["generacion"] != self.generacion:
                self.model.stop_training = True  # Se reinició la plataforma.
                return
            for clave in laboratorio["historial"]:
                laboratorio["historial"][clave].append(float(logs.get(clave, 0.0)))
            laboratorio["epoca"] = epoca + 1
            if mapa:
                laboratorio["mapa"] = mapa
            laboratorio["version"] += 1
            if laboratorio["detener"]:
                self.model.stop_training = True  # El usuario pulsó "Detener".


def entrenar_experimento(config, generacion):
    """Construye, entrena y evalúa una red experimental (se ejecuta en un hilo)."""
    try:
        keras.utils.set_random_seed(42)  # Misma semilla: comparación justa.
        activacion = config["activacion"]
        red = keras.Sequential(
            [
                keras.Input(shape=(2,)),
                keras.layers.Dense(config["neuronas1"], activation=activacion),
                keras.layers.Dense(config["neuronas2"], activation=activacion),
                # La salida siempre es sigmoide: debe entregar una probabilidad.
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        red.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config["tasa"]),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        red.fit(
            DATOS["X_entrenamiento"],
            DATOS["y_entrenamiento"],
            epochs=config["epocas"],
            batch_size=32,
            validation_split=0.20,
            verbose=0,
            callbacks=[ProgresoLaboratorio(config["epocas"], generacion)],
        )
        # Evaluación con los datos de prueba (los mismos del modelo oficial).
        y_real = DATOS["y_prueba"]
        prob = red.predict(DATOS["X_prueba"], verbose=0).ravel()
        # Aquí (hilo de entrenamiento) se usa el umbral oficial; la tabla de
        # experimentos se recalcula con el umbral de cada usuario al pedirla.
        m = metricas_con(prob, UMBRAL_OFICIAL)
        resultado = {
            "perdida": float(red.evaluate(DATOS["X_prueba"], y_real, verbose=0)[0]),
            "exactitud": m["exactitud"],
            "precision": m["precision"],
            "recall": m["recall"],
            "parametros": int(red.count_params()),
        }
        mapa_final = mapa_de(red, 41)
        with _candado:
            if laboratorio["generacion"] != generacion:
                return  # La plataforma se reinició mientras entrenaba: se descarta.
            detenido = laboratorio["detener"]
            laboratorio["resultado"] = resultado
            laboratorio["mapa"] = mapa_final
            laboratorio["estado"] = "detenido" if detenido else "terminado"
            laboratorio["version"] += 1
            experimentos.append(
                {
                    "nombre": f"Exp. {len(experimentos)}",
                    "activacion": ACTIVACIONES[activacion],
                    "arquitectura": f"2-{config['neuronas1']}-{config['neuronas2']}-1",
                    "tasa": config["tasa"],
                    "epocas": laboratorio["epoca"],
                    "perdida": resultado["perdida"],
                    "_prob": prob,
                }
            )
    except Exception as error:  # Si algo falla, la interfaz muestra el mensaje.
        with _candado:
            if laboratorio["generacion"] != generacion:
                return
            laboratorio["estado"] = "error"
            laboratorio["resultado"] = {"error": str(error)}
            laboratorio["version"] += 1


@app.post("/api/laboratorio/entrenar")
def laboratorio_entrenar():
    """Inicia el entrenamiento de una red experimental en segundo plano."""
    config, error = validar_config(request.get_json(silent=True) or {})
    if error:
        return jsonify({"ok": False, "error": error}), 400
    with _candado:
        if laboratorio["estado"] == "entrenando":
            return jsonify({"ok": False, "error": "Ya hay un entrenamiento en curso."}), 409
        laboratorio.update(
            estado="entrenando",
            config=config,
            epoca=0,
            historial={"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []},
            mapa=None,
            resultado=None,
            detener=False,
        )
        laboratorio["version"] += 1
        generacion = laboratorio["generacion"]
    threading.Thread(target=entrenar_experimento, args=(config, generacion), daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/laboratorio/detener")
def laboratorio_detener():
    """Pide detener el entrenamiento al terminar la época actual."""
    with _candado:
        laboratorio["detener"] = True
    return jsonify({"ok": True})


@app.get("/api/laboratorio/estado")
def laboratorio_estado():
    """La interfaz consulta esta ruta varias veces por segundo mientras entrena."""
    with _candado:
        datos = {k: v for k, v in laboratorio.items() if k not in ("detener", "generacion")}
        datos["experimentos"] = tabla_experimentos()
        datos["umbral"] = umbral_actual()
        return jsonify(datos)


# ============================================================
# REINICIAR LA PLATAFORMA DESDE CERO
# ============================================================
@app.post("/api/reiniciar")
def reiniciar():
    """Deja todo como al abrir la plataforma por primera vez.

    - Umbral de vuelta al oficial (0.5).
    - Se detiene el entrenamiento del laboratorio (si hay uno) y se borran
      los experimentos; solo queda la fila del modelo oficial.
    El modelo oficial entrenado NO se borra ni se vuelve a entrenar.
    (El historial de análisis vive en el navegador y lo borra la interfaz.)
    """
    with _candado:
        laboratorio.update(
            estado="inactivo",
            config=None,
            epoca=0,
            historial={"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []},
            mapa=None,
            resultado=None,
            detener=False,
        )
        laboratorio["generacion"] += 1  # El hilo que estuviera entrenando se descarta.
        laboratorio["version"] += 1
        experimentos[:] = [fila_oficial()]
    return jsonify({"ok": True, "umbral": UMBRAL_OFICIAL})


if __name__ == "__main__":
    # En el computador: http://127.0.0.1:5000
    # En un servidor (Hugging Face, Render…) se usa gunicorn (ver Dockerfile),
    # pero también se puede indicar el puerto con la variable PORT.
    puerto = int(os.environ.get("PORT", 5000))
    anfitrion = os.environ.get("HOST", "127.0.0.1")
    print(f"\nNeuroGuard 3D listo en  http://{anfitrion}:{puerto}  (Ctrl+C para detener)\n")
    app.run(host=anfitrion, port=puerto, debug=False, threaded=True)

