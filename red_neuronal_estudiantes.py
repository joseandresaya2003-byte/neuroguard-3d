"""Taller práctico: red neuronal de tres capas (arquitectura 2-8-4-1).

Asignatura : Inteligencia Artificial - Ingeniería Informática - UNIPAZ
Docente    : Ing. Esp. Jhonis Ríos M.

¿QUÉ HACE ESTE PROGRAMA?
    Construye, entrena y evalúa una red neuronal que clasifica conexiones
    informáticas SIMULADAS como:
        0 -> conexión normal
        1 -> conexión sospechosa
    a partir de dos entradas:
        x1 -> número de intentos fallidos de inicio de sesión
        x2 -> tráfico enviado en megabytes (MB)

    Todas las marcas TODO del código entregado por el docente están resueltas
    (se indica "TODO n (RESUELTO)" en cada una). Además se agregaron las
    partes que pide el taller: conteo de clases, parámetros entrenables,
    precisión, recall, matriz de confusión y la Fase 6 de predicción con
    explicación de cada resultado.

OBJETIVO GENERAL DEL TALLER (y dónde se cumple en este archivo)
    "Construir, entrenar y evaluar una red neuronal de tres capas que permita
     clasificar conexiones informáticas simuladas como normales o sospechosas."
        CONSTRUIR  -> sección 5   🧠  keras.Sequential con 3 capas Dense
        ENTRENAR   -> sección 7   🔥  modelo.fit()
        EVALUAR    -> sección 8   📊  modelo.evaluate() + métricas
        CLASIFICAR -> sección 9   🔮  analizar_conexion(): normal o sospechosa

MAPA DEL CÓDIGO (busque cada emoji con Ctrl+F)
    🟦 ENTRADAS ........... x1 = intentos fallidos, x2 = tráfico en MB
    🟨 DATOS .............. generación, etiquetas, división y normalización
    🧠 CEREBRO ............ la red neuronal 2-8-4-1 (sección 5)
        🟩 CAPA OCULTA 1 .. 8 neuronas ReLU
        🟩 CAPA OCULTA 2 .. 4 neuronas ReLU
        🟥 CAPA DE SALIDA . 1 neurona sigmoide -> probabilidad de 0 a 1
    ⚙️ CONFIGURACIÓN ...... optimizador Adam y función de pérdida (sección 6)
    🔥 ENTRENAMIENTO ...... aquí la red APRENDE (sección 7)
    📊 EVALUACIÓN ......... métricas y matriz de confusión (sección 8)
    🔮 PREDICCIÓN ......... clasificar una conexión nueva (sección 9)
    💾 GUARDAR ............ archivos que usa la interfaz gráfica (sección 10)

    Al terminar, el programa GUARDA el modelo entrenado en la carpeta
    "modelo/" para que la interfaz gráfica (app.py) lo cargue y lo use.
    Así la interfaz se conecta con la red REAL y no muestra datos inventados.

CÓMO EJECUTAR
    python -m pip install -r requirements.txt
    python red_neuronal_estudiantes.py

AVISO ACADÉMICO
    Los datos son simulados. El resultado no confirma ni descarta la
    existencia de un ataque informático real.
"""

# ============================================================
# INSTALACIÓN DE LIBRERÍAS
# ============================================================
# Abra la terminal de Visual Studio Code y ejecute:
# python -m pip install -r requirements.txt

# ============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================
import json  # Guarda métricas e historial en un archivo de texto (JSON).
import os  # Maneja rutas de carpetas y archivos.
import sys  # Permite configurar la salida de la consola.

# En Windows la consola puede no aceptar algunos caracteres (tildes, cuadros
# del resumen de Keras). Esta línea fuerza UTF-8 para evitar errores al imprimir.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Reduce los mensajes informativos internos de TensorFlow en la consola.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib  # Guarda el normalizador entrenado (viene con scikit-learn).
import numpy as np  # Permite crear y organizar datos numéricos.
import tensorflow as tf  # Librería principal de deep learning (el "motor" de Keras).
import keras  # Keras (viene con TensorFlow): permite crear y entrenar la red neuronal.
from keras import layers  # Contiene las capas neuronales (Dense).
from sklearn.model_selection import train_test_split  # Divide los datos.
from sklearn.preprocessing import StandardScaler  # Normaliza las entradas.
from sklearn.metrics import (  # Métricas de evaluación de clasificación.
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Carpeta donde se guardan los archivos que usa la interfaz gráfica.
CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA_MODELO = os.path.join(CARPETA_BASE, "modelo")
RUTA_MODELO = os.path.join(CARPETA_MODELO, "red_neuronal.keras")
RUTA_NORMALIZADOR = os.path.join(CARPETA_MODELO, "normalizador.pkl")
RUTA_INFO = os.path.join(CARPETA_MODELO, "info_modelo.json")

# Umbral de decisión: si p >= 0.5 la conexión se considera sospechosa.
UMBRAL = 0.5


# Colores para la consola (códigos ANSI): cada fase sale de un color distinto
# para reconocerla rápido. En la terminal de VS Code funcionan directamente;
# en la consola clásica de Windows (cmd) hay que activarlos con SetConsoleMode.
if os.name == "nt":
    try:
        import ctypes

        consola = ctypes.windll.kernel32
        consola.SetConsoleMode(consola.GetStdHandle(-11), 7)  # 7 = permitir colores ANSI
    except (AttributeError, OSError):
        pass  # Si no se puede, el programa sigue igual (solo sin colores).
COLORES = {
    "amarillo": "\033[93m",  # Datos
    "morado": "\033[95m",  # Cerebro / arquitectura
    "rojo": "\033[91m",  # Entrenamiento / conexión sospechosa
    "verde": "\033[92m",  # Evaluación / mensajes positivos
    "azul": "\033[94m",  # Predicción
    "cian": "\033[96m",  # Conexión normal / guardado
}
NEGRITA = "\033[1m"
FIN_COLOR = "\033[0m"  # Vuelve al color normal de la consola.


def color(texto, nombre):
    """Devuelve el texto pintado con el color indicado."""
    return f"{COLORES[nombre]}{texto}{FIN_COLOR}"


def titulo(texto, nombre_color="cian"):
    """Imprime un título visible (y de color) para separar las etapas en la consola."""
    linea = "=" * 64
    print("\n" + color(linea, nombre_color))
    print(NEGRITA + color(texto, nombre_color))
    print(color(linea, nombre_color))


# ============================================================
# 2. CONFIGURACIÓN DEL EXPERIMENTO (SEMILLA)
# ============================================================
# TODO 1 (RESUELTO): se define una semilla entera.
# La semilla NO es un dato de la red: solo controla los números aleatorios
# para que cada ejecución genere los mismos datos y resultados similares.
SEMILLA = 42

if SEMILLA is None:
    raise NotImplementedError("Complete el TODO 1: defina la SEMILLA.")

np.random.seed(SEMILLA)  # Fija los números aleatorios de NumPy.
keras.utils.set_random_seed(SEMILLA)  # Fija los de TensorFlow/Keras/Python.


# ============================================================
# 🟨 3. GENERACIÓN DE DATOS DIDÁCTICOS (SIMULADOS)
#       Aquí nacen las ENTRADAS 🟦 de la red (x1 y x2).
# ============================================================
CANTIDAD_CONEXIONES = 2000  # Número de conexiones simuladas.

# 🟦 ENTRADA x1: intentos fallidos, números enteros entre 0 y 20.
intentos_fallidos = np.random.randint(
    0,  # Valor mínimo.
    21,  # Genera valores hasta 20 (el 21 no se incluye).
    size=CANTIDAD_CONEXIONES,  # Un valor por conexión.
)

# 🟦 ENTRADA x2: tráfico enviado, números decimales entre 0 y 2000 MB.
trafico_mb = np.random.uniform(
    0,  # Tráfico mínimo.
    2000,  # Tráfico máximo aproximado.
    size=CANTIDAD_CONEXIONES,  # Un valor por conexión.
)

# Ruido aleatorio: hace los datos más realistas, porque en la práctica dos
# conexiones parecidas no siempre reciben la misma etiqueta.
ruido = np.random.normal(
    0,  # Media del ruido.
    0.08,  # Variación de los datos.
    size=CANTIDAD_CONEXIONES,
)

# "Regla oculta" con la que se fabrican las respuestas correctas.
# Cada entrada se lleva a escala 0-1 y se pondera:
#   55 % de peso para los intentos fallidos y 45 % para el tráfico.
# La red NO conoce esta fórmula: debe descubrir el patrón a partir de ejemplos.
puntaje_riesgo = (
    0.55 * (intentos_fallidos / 20)
    + 0.45 * (trafico_mb / 2000)
    + ruido
)


# ============================================================
# 🟨 3.1 CREACIÓN DE LAS ETIQUETAS (la "respuesta correcta" de cada conexión)
# ============================================================
# TODO 2 (RESUELTO): etiqueta 1 si el puntaje de riesgo es >= 0.55, si no 0.
# (puntaje_riesgo >= 0.55) produce True/False; astype(int) lo vuelve 1/0.
etiquetas = (puntaje_riesgo >= 0.55).astype(int)

if etiquetas is None:
    raise NotImplementedError("Complete el TODO 2: cree las etiquetas.")

# 🟦🟦 LAS DOS ENTRADAS DE LA RED, JUNTAS ------------------------------------
# Une intentos y tráfico para formar las dos entradas de la red.
# Resultado: matriz de 2000 filas (conexiones) x 2 columnas (x1, x2).
entradas = np.column_stack((intentos_fallidos, trafico_mb))

# Fase 2.4: contar cuántas conexiones de cada clase se generaron.
total_normales = int(np.sum(etiquetas == 0))
total_sospechosas = int(np.sum(etiquetas == 1))

titulo("FASE 2 - COMPRENSIÓN DE LOS DATOS", "amarillo")
print("Entradas : x1 = intentos fallidos (0-20), x2 = tráfico enviado (0-2000 MB)")
print("Etiquetas: 0 = conexión normal, 1 = conexión sospechosa")
print(f"Total de conexiones simuladas : {CANTIDAD_CONEXIONES}")
print(f"Conexiones normales  (0)      : {total_normales}")
print(f"Conexiones sospechosas (1)    : {total_sospechosas}")
print("\nPrimeras 5 conexiones generadas [intentos, tráfico MB] -> etiqueta:")
for i in range(5):
    print(f"  [{intentos_fallidos[i]:>2}, {trafico_mb[i]:8.2f}] -> {etiquetas[i]}")


# ============================================================
# 🟨 4. DIVISIÓN Y NORMALIZACIÓN DE LAS ENTRADAS 🟦
# ============================================================
# 80 % de los datos para entrenar y 20 % para probar con datos que la red
# nunca vio. stratify mantiene la misma proporción de 0 y 1 en ambos grupos.
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    entradas,
    etiquetas,
    test_size=0.20,  # Reserva 20 % para probar el modelo.
    random_state=SEMILLA,
    stratify=etiquetas,
)

# StandardScaler transforma cada columna con: x_norm = (x - media) / desviación.
# Sin esto, el tráfico (valores hasta 2000) dominaría a los intentos (hasta 20).
normalizador = StandardScaler()  # Crea el normalizador.

# TODO 3 (RESUELTO): fit_transform APRENDE la media y la desviación SOLO con los
# datos de entrenamiento y luego los transforma.
X_entrenamiento_normalizado = normalizador.fit_transform(X_entrenamiento)

# TODO 4 (RESUELTO): transform usa la MISMA media y desviación ya aprendidas.
# No se usa fit en prueba para no "filtrar" información de los datos de prueba.
X_prueba_normalizado = normalizador.transform(X_prueba)

if X_entrenamiento_normalizado is None or X_prueba_normalizado is None:
    raise NotImplementedError("Complete los TODO 3 y 4: normalice los datos.")

print(f"\nDatos de entrenamiento: {len(X_entrenamiento)} | Datos de prueba: {len(X_prueba)}")
print(f"Media aprendida        [intentos, tráfico]: {np.round(normalizador.mean_, 3)}")
print(f"Desviación aprendida   [intentos, tráfico]: {np.round(normalizador.scale_, 3)}")


# ############################################################
# 🧠🧠🧠  5. CEREBRO DEL CÓDIGO: LA RED NEURONAL (2-8-4-1)  🧠🧠🧠
# ############################################################
#
#   🟦 ENTRADAS        🟩 OCULTA 1         🟩 OCULTA 2        🟥 SALIDA
#   x1 intentos ──┐     8 neuronas ReLU     4 neuronas ReLU    1 neurona sigmoide
#                 ├──►  ● ● ● ● ● ● ● ● ──► ● ● ● ●  ───────►  ●  ──► p (0 a 1)
#   x2 tráfico  ──┘
#
# Cada capa Dense hace en cada neurona:  z = Σ(wᵢ·xᵢ) + b   y luego  a = f(z)
#   - ReLU(z)     = max(0, z)          -> capas ocultas (permite aprender curvas)
#   - Sigmoide(z) = 1 / (1 + e^(-z))   -> salida (convierte z en probabilidad 0-1)
# TODO 5 (RESUELTO): entrada de 2 valores + capas 8 ReLU, 4 ReLU y 1 sigmoide.
modelo = keras.Sequential(
    [
        # 🟦 CAPA DE ENTRADA ---------------------------------------------
        # Recibe los 2 valores normalizados (x1 = intentos, x2 = tráfico).
        # No tiene neuronas que calculen: solo entrega los datos a la red.
        keras.Input(shape=(2,), name="entradas"),

        # 🟩 CAPA OCULTA 1 -----------------------------------------------
        # 8 neuronas con ReLU. Cada una combina x1 y x2 con sus propios
        # pesos y detecta un "patrón" distinto (2·8 pesos + 8 sesgos = 24).
        layers.Dense(8, activation="relu", name="oculta_1_relu"),

        # 🟩 CAPA OCULTA 2 -----------------------------------------------
        # 4 neuronas con ReLU. Combinan los 8 patrones anteriores en
        # patrones más completos (8·4 pesos + 4 sesgos = 36).
        layers.Dense(4, activation="relu", name="oculta_2_relu"),

        # 🟥 CAPA DE SALIDA ----------------------------------------------
        # 1 neurona con SIGMOIDE: convierte el resultado en una probabilidad
        # entre 0 y 1 de que la conexión sea sospechosa (4 pesos + 1 sesgo = 5).
        layers.Dense(1, activation="sigmoid", name="salida_sigmoide"),
    ],
    name="red_estudiantes",
)
# 🧠 FIN DEL CEREBRO ##########################################

if len(modelo.layers) == 0:
    raise NotImplementedError("Complete el TODO 5: construya las tres capas.")


# ============================================================
# ⚙️ 6. CONFIGURACIÓN DEL APRENDIZAJE (COMPILACIÓN)
# ============================================================
# TODO 6 (RESUELTO):
#   optimizer = "adam"               -> algoritmo que ajusta pesos y sesgos.
#                                       (tasa de aprendizaje 0.001 = tamaño de cada ajuste).
#   loss = "binary_crossentropy"     -> mide el error en problemas de 2 clases.
#   metrics = exactitud, precisión y recall, calculadas en cada época.
modelo.compile(
    optimizer="adam",  # Adam con su tasa de aprendizaje por defecto (0.001).
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

titulo("FASE 3 - ARQUITECTURA DE LA RED NEURONAL (EL CEREBRO)", "morado")
print(f"Motor: TensorFlow {tf.__version__} · Keras {keras.__version__}")
modelo.summary()  # Muestra la estructura construida.

# Fase 3.5: parámetros entrenables = pesos + sesgos de todas las capas.
#   Capa 1: 2·8 pesos + 8 sesgos = 24
#   Capa 2: 8·4 pesos + 4 sesgos = 36
#   Salida: 4·1 pesos + 1 sesgo  =  5      Total = 65
parametros_entrenables = int(
    sum(np.prod(peso.shape) for peso in modelo.trainable_weights)
)
print(f"Parámetros entrenables (pesos + sesgos): {parametros_entrenables}")
for capa in modelo.layers:
    w, b = capa.get_weights()
    print(f"  {capa.name:<16} pesos {w.shape} = {w.size:>2} | sesgos {b.shape} = {b.size}")


# ############################################################
# 🔥🔥🔥  7. ENTRENAMIENTO  ->  AQUÍ APRENDE LA RED  🔥🔥🔥
# ############################################################
# En cada época, modelo.fit():
#   1) propaga los datos hacia adelante y calcula una predicción,
#   2) compara la predicción con la etiqueta y calcula la pérdida,
#   3) propaga el error hacia atrás (retropropagación / backpropagation),
#   4) el optimizador Adam actualiza los pesos y sesgos para reducir el error.
pesos_antes = modelo.layers[0].get_weights()[0].copy()  # Para comparar después.

titulo("FASE 4 - ENTRENAMIENTO (modelo.fit): LA RED ESTÁ APRENDIENDO", "rojo")
# TODO 7 (RESUELTO): entrenamiento por 100 épocas, lotes de 32 ejemplos y 20 %
# de los datos de entrenamiento reservados para validación.
# 🔥 ESTA ES LA LÍNEA QUE ENTRENA LA RED 🔥
historial = modelo.fit(
    X_entrenamiento_normalizado,
    y_entrenamiento,
    epochs=100,
    batch_size=32,
    validation_split=0.20,
    verbose=2,  # Una línea por época: muestra cómo baja la pérdida.
)

if historial is None:
    raise NotImplementedError("Complete el TODO 7: entrene con modelo.fit().")

perdidas = historial.history["loss"]
print(f"\nPérdida en la época 1  : {perdidas[0]:.4f}")
print(f"Pérdida en la época 100: {perdidas[-1]:.4f}")
if perdidas[-1] < perdidas[0]:
    print(color("La pérdida DISMINUYÓ -> la red SÍ está aprendiendo el patrón.", "verde"))
else:
    print("La pérdida no disminuyó -> revise los datos o la configuración.")

pesos_despues = modelo.layers[0].get_weights()[0]
print("\nLos pesos cambian porque el optimizador los corrige en cada lote:")
print(f"  w(x1 -> neurona 1) antes: {pesos_antes[0, 0]: .4f} | después: {pesos_despues[0, 0]: .4f}")
print(f"  w(x2 -> neurona 1) antes: {pesos_antes[1, 0]: .4f} | después: {pesos_despues[1, 0]: .4f}")


# ============================================================
# 📊 8. EVALUACIÓN CON DATOS DE PRUEBA (datos que la red nunca vio)
# ============================================================
titulo("FASE 5 - EVALUACIÓN CON DATOS DE PRUEBA", "verde")
# TODO 8 (RESUELTO): evaluate devuelve [pérdida, exactitud, precisión, recall].
resultados = modelo.evaluate(X_prueba_normalizado, y_prueba, verbose=0)

if resultados is None:
    raise NotImplementedError("Complete el TODO 8: evalúe el modelo.")

perdida_prueba, exactitud_keras, precision_keras, recall_keras = resultados

# Se calculan también con scikit-learn a partir de las predicciones (umbral 0.5).
probabilidades_prueba = modelo.predict(X_prueba_normalizado, verbose=0).ravel()
predicciones_prueba = (probabilidades_prueba >= UMBRAL).astype(int)

exactitud = accuracy_score(y_prueba, predicciones_prueba)
precision = precision_score(y_prueba, predicciones_prueba, zero_division=0)
recall = recall_score(y_prueba, predicciones_prueba, zero_division=0)
f1 = f1_score(y_prueba, predicciones_prueba, zero_division=0)
# Matriz de confusión: [[VN, FP], [FN, VP]]
vn, fp, fn, vp = confusion_matrix(y_prueba, predicciones_prueba).ravel()

print(f"Pérdida  (binary crossentropy): {perdida_prueba:.4f}")
print(f"Exactitud (accuracy)          : {exactitud:.2%}")
print(f"Precisión (precision)         : {precision:.2%}")
print(f"Recall   (sensibilidad)       : {recall:.2%}")
print(f"F1-score                      : {f1:.2%}")
print("\nMatriz de confusión (filas = real, columnas = predicho):")
print("                 Pred. normal   Pred. sospechosa")
print(f"  Real normal        {vn:>5}          {fp:>5}")
print(f"  Real sospechosa    {fn:>5}          {vp:>5}")
print("\nInterpretación:")
print(f"  - Exactitud: de cada 100 conexiones, la red acierta ~{exactitud * 100:.0f}.")
print(f"  - Precisión: de las alertas de 'sospechosa', el {precision:.0%} eran realmente sospechosas")
print(f"    ({fp} falsas alarmas).")
print(f"  - Recall: detectó el {recall:.0%} de las conexiones sospechosas reales")
print(f"    ({fn} se escaparon sin ser detectadas).")


# ============================================================
# 🔮 9. PREDICCIÓN DE UNA CONEXIÓN NUEVA (usar el cerebro ya entrenado)
# ============================================================
def analizar_conexion(intentos, trafico):
    """Recibe dos valores, los pasa por la red y muestra la clasificación.

    Devuelve un diccionario con la probabilidad y la clasificación para que
    pueda reutilizarse (por ejemplo, desde la interfaz).
    """
    nueva_conexion = np.array([[intentos, trafico]], dtype=float)

    # TODO 9 (RESUELTO): se normaliza con el MISMO normalizador del entrenamiento.
    nueva_conexion_normalizada = normalizador.transform(nueva_conexion)

    # TODO 10 (RESUELTO): predict devuelve [[p]]; float(...[0][0]) extrae p.
    probabilidad = float(modelo.predict(nueva_conexion_normalizada, verbose=0)[0][0])

    if nueva_conexion_normalizada is None or probabilidad is None:
        raise NotImplementedError("Complete los TODO 9 y 10: realice la predicción.")

    # TODO 11 (RESUELTO): decisión con el umbral 0.5.
    if probabilidad >= UMBRAL:
        resultado = "CONEXIÓN SOSPECHOSA (1)"
        explicacion = (
            f"La probabilidad ({probabilidad:.2%}) es mayor o igual al umbral "
            f"({UMBRAL}). La combinación de intentos fallidos y tráfico se parece "
            "a los ejemplos sospechosos con los que se entrenó la red."
        )
    else:
        resultado = "CONEXIÓN NORMAL (0)"
        explicacion = (
            f"La probabilidad ({probabilidad:.2%}) es menor que el umbral "
            f"({UMBRAL}). Los valores se parecen a los ejemplos normales "
            "vistos durante el entrenamiento."
        )

    print("\nANÁLISIS DE LA CONEXIÓN")
    print(f"Intentos fallidos : {intentos}")
    print(f"Tráfico enviado   : {trafico:.2f} MB")
    print(f"Datos normalizados: {np.round(nueva_conexion_normalizada[0], 4)}")
    print(f"Probabilidad      : {probabilidad:.2%}")
    print(f"Umbral utilizado  : {UMBRAL}")
    # Sospechosa en rojo, normal en cian.
    print(f"Resultado         : {color(resultado, 'rojo' if probabilidad >= UMBRAL else 'cian')}")
    print(f"Explicación       : {explicacion}")

    return {"probabilidad": probabilidad, "resultado": resultado}


# TODO 12 (RESUELTO): se prueban los casos del docente (8, 600) y (2, 100)
# y los tres casos mínimos de la Fase 6 (bajo, intermedio y alto).
titulo("FASE 6 - PREDICCIÓN DE CONEXIONES NUEVAS", "azul")
casos_prueba = [
    ("Pocos intentos y poco tráfico", 2, 100),
    ("Valores intermedios (caso docente)", 8, 600),
    ("Valores intermedios altos", 11, 1000),
    ("Muchos intentos y mucho tráfico", 18, 1800),
]
for descripcion, intentos, trafico in casos_prueba:
    print(f"\n>>> Caso: {descripcion}")
    analizar_conexion(intentos, trafico)


# ############################################################
# 🧮 9.1 LA OPERACIÓN DEL CEREBRO, PASO A PASO
# ############################################################
# modelo.predict() hace estas cuentas automáticamente y no se ven. Aquí se
# repiten "a mano" con los pesos (W) y sesgos (b) que la red APRENDIÓ, para
# mostrar DÓNDE y CÓMO se hace la operación en cada una de las 3 capas:
#
#     z = entradas · W + b         -> suma ponderada de cada neurona  Σ(wᵢ·xᵢ) + b
#     a = ReLU(z)  o  Sigmoide(z)  -> función de activación de la capa
#
# Al final se comprueba que el resultado es IGUAL al de modelo.predict().
def operacion_del_cerebro(intentos, trafico):
    """Recorre la red capa por capa mostrando z y la activación de cada neurona."""
    # 🟦 ENTRADAS: los dos valores, normalizados igual que en el entrenamiento.
    x = normalizador.transform(np.array([[intentos, trafico]], dtype=float))[0]
    print(f"🟦 Entradas normalizadas  x = {np.round(x, 4)}")

    nombres = ["🟩 Capa oculta 1 (8 ReLU)", "🟩 Capa oculta 2 (4 ReLU)", "🟥 Capa de salida (1 sigmoide)"]
    activacion = x  # Lo que entra a la primera capa son las entradas.
    for capa, nombre in zip(modelo.layers, nombres):
        W, b = capa.get_weights()  # Pesos y sesgos aprendidos en el entrenamiento.

        # 🧠 AQUÍ SE HACE LA OPERACIÓN DE LA NEURONA: z = Σ(w·x) + b
        z = activacion @ W + b

        # 🧠 Y AQUÍ LA ACTIVACIÓN: ReLU en las ocultas, sigmoide en la salida.
        if capa.get_config()["activation"] == "relu":
            activacion = np.maximum(0, z)  # ReLU(z) = max(0, z)
        else:
            activacion = 1 / (1 + np.exp(-z))  # Sigmoide(z) = 1 / (1 + e^(-z))

        print(f"\n{nombre}")
        print(f"   z = {np.round(z, 3)}")
        print(f"   a = {np.round(activacion, 3)}")

    p_a_mano = float(activacion[0])
    p_keras = float(modelo.predict(x.reshape(1, 2), verbose=0)[0][0])
    print(f"\nProbabilidad calculada a mano : {p_a_mano:.6f}")
    print(f"Probabilidad de modelo.predict: {p_keras:.6f}")
    if abs(p_a_mano - p_keras) < 1e-5:
        print(color("Son iguales: así es exactamente como 'piensa' la red.", "verde"))


titulo("LA OPERACIÓN DEL CEREBRO PASO A PASO (caso docente: 8 intentos, 600 MB)", "morado")
operacion_del_cerebro(8, 600)


# ============================================================
# 💾 10. GUARDAR EL MODELO PARA LA INTERFAZ
# ============================================================
# La interfaz (app.py) carga estos archivos: así usa la red ENTRENADA real.
os.makedirs(CARPETA_MODELO, exist_ok=True)
modelo.save(RUTA_MODELO)  # Arquitectura + pesos + sesgos.
joblib.dump(normalizador, RUTA_NORMALIZADOR)  # Media y desviación aprendidas.

# Datos ya divididos y normalizados: el "Laboratorio" de la interfaz los usa
# para entrenar redes experimentales con LOS MISMOS datos y compararlas.
np.savez(
    os.path.join(CARPETA_MODELO, "datos.npz"),
    X_entrenamiento=X_entrenamiento_normalizado,
    y_entrenamiento=y_entrenamiento,
    X_prueba=X_prueba_normalizado,
    y_prueba=y_prueba,
)

# Muestra de datos de prueba (escala original) para dibujarlos en la interfaz.
muestra = [
    {
        "intentos": int(X_prueba[i, 0]),
        "trafico": round(float(X_prueba[i, 1]), 2),
        "real": int(y_prueba[i]),
        "predicho": int(predicciones_prueba[i]),
    }
    for i in range(len(X_prueba))
]

info = {
    "semilla": SEMILLA,
    "umbral": UMBRAL,
    "datos": {
        "total": CANTIDAD_CONEXIONES,
        "normales": total_normales,
        "sospechosas": total_sospechosas,
        "entrenamiento": int(len(X_entrenamiento)),
        "prueba": int(len(X_prueba)),
        "rango_intentos": [0, 20],
        "rango_trafico": [0, 2000],
    },
    "arquitectura": "2-8-4-1",
    "parametros_entrenables": parametros_entrenables,
    "metricas": {
        "perdida": float(perdida_prueba),
        "exactitud": float(exactitud),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matriz_confusion": {"vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp)},
    },
    # Historial por época (convertido a float normal para poder guardarlo).
    "historial": {k: [float(v) for v in vals] for k, vals in historial.history.items()},
    "muestra_prueba": muestra,
}
with open(RUTA_INFO, "w", encoding="utf-8") as archivo:
    json.dump(info, archivo, ensure_ascii=False, indent=1)

titulo("MODELO GUARDADO", "cian")
print(f"Modelo      : {RUTA_MODELO}")
print(f"Normalizador: {RUTA_NORMALIZADOR}")
print(f"Información : {RUTA_INFO}")
print("\nAhora ejecute la interfaz con:  python app.py")
print("\nAviso académico: los datos son simulados. El resultado no confirma ni")
print("descarta la existencia de un ataque informático real.")
