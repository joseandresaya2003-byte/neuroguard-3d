# Informe — Taller práctico de redes neuronales

**Asignatura:** Inteligencia Artificial · **Programa:** Ingeniería Informática · **UNIPAZ**
**Docente:** Ing. Esp. Jhonis Ríos M.
**Proyecto:** NeuroGuard 3D — clasificación de conexiones con una red neuronal 2-8-4-1

---

### 1. ¿Cuáles son las entradas de la red neuronal?

Son dos valores numéricos por conexión:

- **x₁ — intentos fallidos de inicio de sesión:** número entero entre 0 y 20.
- **x₂ — tráfico enviado en MB:** número decimal entre 0 y 2000.

En el código se unen con `np.column_stack((intentos_fallidos, trafico_mb))` y forman una matriz de 2000 filas × 2 columnas. La capa de entrada (`keras.Input(shape=(2,))`) no hace cálculos: solo pasa estos dos valores, ya normalizados, a la primera capa oculta.

### 2. ¿Qué representa cada etiqueta?

La etiqueta es la respuesta correcta que la red debe aprender:

- **0 = conexión normal.**
- **1 = conexión sospechosa.**

Se crea con `etiquetas = (puntaje_riesgo >= 0.55).astype(int)`, donde `puntaje_riesgo = 0.55·(intentos/20) + 0.45·(tráfico/2000) + ruido`. Se generaron **1191 conexiones normales y 809 sospechosas**. La red nunca ve esta fórmula: la descubre a partir de los ejemplos.

### 3. ¿Cuál es la función de los pesos?

Los pesos (w) indican **cuánta importancia** le da cada neurona a cada uno de sus datos de entrada. Un peso positivo grande hace que esa entrada empuje la neurona a activarse. Un peso negativo la frena. Un peso cercano a cero significa que la entrada casi no importa. Al inicio los pesos son aleatorios y `modelo.fit()` los ajusta. En nuestra ejecución, el peso de x₁ hacia la neurona 1 cambió de un valor inicial aleatorio a otro aprendido. En la interfaz 3D, cada conexión muestra su peso real: cian si es positivo, rosa si es negativo, y más grueso cuanto mayor es su magnitud.

### 4. ¿Cuál es la función de los sesgos?

El sesgo (b) es un valor que se **suma** a la combinación lineal: `z = Σ(wᵢ·xᵢ) + b`. Desplaza el punto en el que la neurona se activa. Gracias a él, la neurona puede dar una salida distinta de cero aunque todas las entradas valgan 0, y puede necesitar más o menos estímulo para activarse. Cada neurona tiene un sesgo propio: 8 + 4 + 1 = 13 sesgos en total.

### 5. ¿Por qué se normalizan los datos?

Las dos entradas tienen escalas muy distintas: los intentos llegan a 20 y el tráfico a 2000. Sin normalizar, el tráfico dominaría la suma `Σ(w·x)` solo por tener números más grandes, y el entrenamiento sería lento e inestable. Usamos `StandardScaler`, que aplica `x_norm = (x − media) / desviación`. Así ambas variables quedan con media 0 y desviación 1. La media y la desviación se aprenden **solo con los datos de entrenamiento** (`fit_transform`) y luego se aplican iguales a los datos de prueba y a las conexiones nuevas (`transform`). Así no se filtra información de la prueba al entrenamiento.

### 6. ¿Qué sucede dentro de una neurona?

Una neurona hace dos pasos:

1. **Combinación lineal:** multiplica cada entrada por su peso, suma los resultados y agrega el sesgo: `z = w₁x₁ + w₂x₂ + … + b`.
2. **Activación:** aplica una función a z. En las capas ocultas es ReLU, `a = max(0, z)`. En la salida es la sigmoide, `p = 1/(1 + e^(−z))`.

Por ejemplo, para la conexión (18 intentos, 1800 MB), la neurona de salida calculó `z = (1.1985·5.3352) + (−0.4559·0) + (−0.1839·0) + (−1.2397·0) − 0.2343 = 6.1599`, y luego `σ(6.1599) = 0.9979`. La interfaz muestra este cálculo al hacer clic en cualquier neurona.

### 7. ¿Qué función cumplen las capas ocultas?

Transforman las entradas en **representaciones intermedias** que facilitan la decisión final. La primera capa oculta (8 neuronas) detecta combinaciones simples de intentos y tráfico. Por ejemplo, una neurona puede activarse solo cuando ambos valores son altos. La segunda capa (4 neuronas) combina esos patrones en otros más abstractos. Sin capas ocultas, el modelo sería una regresión logística y solo podría trazar una frontera recta. Con ellas, la red puede aprender fronteras curvas. Se ve en el mapa de decisión de la interfaz.

### 8. ¿Por qué se utiliza ReLU en las capas ocultas?

- **Introduce no linealidad:** si todas las capas fueran lineales, varias capas juntas equivaldrían a una sola. ReLU permite aprender relaciones que una recta no puede representar.
- **Es muy simple y rápida:** solo calcula `max(0, z)`.
- **Evita que el gradiente se desvanezca** para valores positivos (su derivada es 1), así que la red aprende más rápido que con sigmoide en las capas ocultas.
- **Produce activaciones dispersas:** las neuronas con z < 0 dan exactamente 0 y quedan "apagadas" para ese caso. En la conexión (18, 1800), por ejemplo, 3 de 8 neuronas de la primera capa quedaron apagadas.

### 9. ¿Por qué se utiliza sigmoide en la salida?

Porque el problema es de **clasificación binaria** y necesitamos una **probabilidad**. La sigmoide convierte cualquier número real z en un valor entre 0 y 1, que se interpreta como la probabilidad de que la conexión sea sospechosa. Después se aplica el umbral: p ≥ 0.5 → sospechosa y p < 0.5 → normal. Además, la sigmoide se combina naturalmente con la función de pérdida `binary_crossentropy`.

### 10. ¿Qué sucede cuando se ejecuta `modelo.fit()`?

Es el momento en que **la red aprende**. En cada una de las 100 épocas, con lotes de 32 ejemplos:

1. Recibe un lote de datos de entrenamiento.
2. Hace la **propagación hacia adelante** y calcula una predicción para cada ejemplo.
3. Compara las predicciones con las etiquetas y calcula la **pérdida** (binary crossentropy).
4. Hace la **retropropagación**: calcula, con la regla de la cadena, cuánto contribuyó cada peso y cada sesgo al error (el gradiente).
5. El optimizador **Adam** actualiza los 65 pesos y sesgos en la dirección que reduce el error.
6. Al final de la época evalúa con el 20 % de validación.

En nuestra ejecución, la pérdida bajó de **0.6149 a 0.2220** y la exactitud de entrenamiento subió de **72.0 % a 90.4 %**. Eso demuestra que la red aprendió. En la pestaña *Entrenamiento* de la interfaz se pueden ver las curvas y reproducir el proceso.

### 11. ¿Qué diferencia existe entre exactitud, precisión y recall?

Con los 400 datos de prueba obtuvimos VN = 219, FP = 19, FN = 14 y VP = 148.

| Métrica | Fórmula | Pregunta que responde | Resultado |
|---|---|---|---|
| **Exactitud** | (VP + VN) / total | ¿Qué porcentaje de todas las conexiones clasificó bien? | 367/400 = **91.75 %** |
| **Precisión** | VP / (VP + FP) | De las conexiones que marcó como sospechosas, ¿cuántas lo eran? | 148/167 = **88.62 %** |
| **Recall** | VP / (VP + FN) | De las conexiones sospechosas reales, ¿cuántas detectó? | 148/162 = **91.36 %** |

La exactitud mide el desempeño general. La **precisión** mide la calidad de las alertas: si es baja, hay muchas falsas alarmas. El **recall** mide la capacidad de detección: si es bajo, se escapan casos sospechosos. En seguridad suele importar más el recall, porque dejar pasar un ataque es más grave que revisar una falsa alarma.

### 12. ¿Cómo se conectó la interfaz con el modelo?

1. `red_neuronal_estudiantes.py` entrena la red y guarda en `modelo/` el modelo (`red_neuronal.keras`), el normalizador (`normalizador.pkl`, con joblib) y las métricas (`info_modelo.json`).
2. `app.py` (Flask) carga esos archivos una sola vez al iniciar.
3. Cuando el usuario pulsa *Analizar*, `app.js` valida los datos y los envía con `fetch()` a `POST /api/analizar` en formato JSON.
4. El servidor vuelve a validar los datos, los normaliza con el **mismo** normalizador del entrenamiento y llama a `modelo.predict()`. También calcula los valores z y la activación de cada neurona con los pesos reales de la red.
5. El servidor responde con un JSON (probabilidad, umbral, clasificación, explicación y recorrido), y la interfaz actualiza el medidor, la red 3D, el mapa de decisión y el historial.

Ningún resultado es fijo: todos salen del modelo entrenado.

### 13. ¿Qué dificultades se presentaron durante el desarrollo?

- **Versión de Python:** el equipo tenía Python 3.14 y TensorFlow todavía no lo soporta. Lo resolvimos con un entorno virtual de Python 3.13 (`py -3.13 -m venv .venv`).
- **Caracteres especiales en la consola de Windows:** el resumen de Keras y las tildes generaban errores de codificación. Lo resolvimos con `sys.stdout.reconfigure(encoding="utf-8")`.
- **Normalizar los datos nuevos:** al principio es fácil olvidar que una conexión nueva debe pasar por el mismo `StandardScaler`. Por eso guardamos el normalizador junto con el modelo.
- **Tasa de aprendizaje:** con una tasa alta (0.01) la red aprendía casi todo en la primera época y la curva de pérdida no mostraba bien el aprendizaje. Con Adam y su tasa por defecto (0.001), la pérdida baja de forma progresiva.
- **Visualización 3D:** encuadrar la red en pantallas de distinto tamaño y sincronizar la animación con los valores reales de cada neurona.
- **Explicar el resultado:** cuando la probabilidad está saturada (cerca de 0 o 1), cambiar una sola entrada casi no mueve el resultado. Para repartir la influencia de forma justa usamos valores de Shapley con dos variables.

### 14. ¿Qué mejoras podrían aplicarse al proyecto?

- Entrenar con **datos reales** de registros de red (por ejemplo, conjuntos públicos como CICIDS o NSL-KDD) y con más variables: puertos, duración, horario y dirección IP de origen.
- Usar **validación cruzada** y **EarlyStopping** para detener el entrenamiento cuando la validación deje de mejorar.
- **Ajustar el umbral** según el costo de cada error, para priorizar el recall en seguridad, y mostrar la curva ROC y el AUC.
- Probar otras arquitecturas (más neuronas, Dropout, regularización) y comparar resultados.
- Permitir en la interfaz **reentrenar** el modelo cambiando la arquitectura o las épocas, y cargar archivos CSV con muchas conexiones.
- Publicar la interfaz con un servidor de producción (por ejemplo, Waitress o Gunicorn) y agregar autenticación.

### 15. ¿Por qué este modelo no debe utilizarse como un sistema real de seguridad?

- **Los datos son simulados:** las etiquetas salen de una fórmula inventada (`0.55·intentos + 0.45·tráfico + ruido`), no de ataques reales. La red solo aprendió esa fórmula.
- **Solo usa dos variables.** Un ataque real puede tener pocos intentos fallidos y poco tráfico (por ejemplo, un robo de credenciales lento), y el modelo lo marcaría como normal.
- **Comete errores:** con los datos de prueba hubo 14 conexiones sospechosas no detectadas y 19 falsas alarmas. En un entorno real, esos errores tienen consecuencias.
- **No generaliza fuera de su rango:** con más de 20 intentos o más de 2000 MB, la red extrapola. La interfaz lo advierte.
- **No fue validada** con tráfico real ni con expertos en seguridad, y no considera ataques que intenten engañar al modelo.

Por eso es un **modelo didáctico**: sirve para entender cómo aprende una red neuronal, pero el resultado no confirma ni descarta la existencia de un ataque informático real.
