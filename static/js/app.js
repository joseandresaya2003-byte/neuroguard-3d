/* =============================================================
   app.js - Controlador principal de la interfaz NeuroGuard 3D
   -------------------------------------------------------------
   Flujo de comunicación con la red neuronal:
     1. Al abrir la página se piden al servidor Flask:
          GET /api/modelo -> pesos, sesgos, métricas e historial REALES
          GET /api/mapa   -> probabilidades de la red en una cuadrícula
     2. Cuando el usuario pulsa "Analizar", se validan los campos aquí
        (validación rápida) y se envían con fetch():
          POST /api/analizar  { intentos, trafico }
        El servidor vuelve a validar, normaliza y ejecuta modelo.predict().
     3. Con la respuesta se actualizan: medidor, clasificación, explicación,
        animación 3D, mapa de decisión, cálculo paso a paso e historial.
     4. UMBRAL: al mover el deslizador del umbral se envía
          POST /api/umbral  { umbral }
        y TODO se actualiza a la vez (medidor, veredicto, explicación, mapa,
        superficie 3D, red 3D, métricas, matriz de confusión y laboratorio).
     5. REINICIAR: POST /api/reiniciar deja el servidor como recién abierto
        y la página se recarga con el historial vacío.
   ============================================================= */

import { Red3D } from "./red3d.js";
import { MapaDecision, GraficaLineas, CurvaSigmoide } from "./graficas.js";
import { Superficie3D } from "./superficie3d.js";
import { Laboratorio } from "./laboratorio.js";
import { umbral, esSospechosa, textoUmbral, cabeceraUmbral } from "./umbral.js";
import { descargarImagenAnalisis } from "./exportar.js";
import { ControlManos } from "./manos.js"; // ✋ control de la red 3D con la cámara web

// Atajo para buscar elementos por id.
const $ = (id) => document.getElementById(id);

// Estado global de la interfaz.
const estado = {
  modelo: null, // datos de /api/modelo
  ultimo: null, // última respuesta de /api/analizar
  historial: [],
  red3d: null,
  manos: null, // ControlManos (se crea al pulsar "✋ Manos")
  mapa: null,
  superficie: null,
  sigmoide: null,
  laboratorio: null,
  ocupado: false,
};

/* =============================================================
   1. ARRANQUE
   ============================================================= */
async function iniciar() {
  try {
    const [modelo, mapa] = await Promise.all([pedirJSON("/api/modelo"), pedirJSON("/api/mapa")]);
    estado.modelo = modelo;
    umbral.valor = modelo.umbral; // Empieza con el umbral oficial (0.5).
    umbral.oficial = modelo.umbral_oficial;
    $("rangoUmbral").min = modelo.umbral_limites[0];
    $("rangoUmbral").max = modelo.umbral_limites[1];
    $("rangoUmbral").value = umbral.valor;

    // Valores iniciales de los campos (el navegador a veces recuerda los de antes).
    ponerValores($("intentos").defaultValue, $("trafico").defaultValue);

    // Encabezado con datos reales del modelo.
    $("chipParametros").textContent = modelo.parametros_entrenables;

    // Curva sigmoide de la neurona de salida.
    estado.sigmoide = new CurvaSigmoide($("lienzoSigmoide"), $("tooltipSigmoide"));

    // Red 3D con los pesos reales.
    estado.red3d = new Red3D($("escena3d"), modelo.capas, mostrarDetalleNeurona);

    // Mapa de decisión 2D y superficie 3D: al hacer clic se analiza ese punto.
    const elegirPunto = (intentos, trafico) => {
      ponerValores(intentos, trafico);
      analizar({ silencioso: true });
    };
    estado.mapa = new MapaDecision($("lienzoMapa"), $("tooltipMapa"), elegirPunto);
    estado.mapa.setMapa(mapa);
    estado.mapa.setPuntos(modelo.muestra_prueba);
    estado.superficie = new Superficie3D($("superficieMapa"), elegirPunto);
    estado.superficie.setMapa(mapa);

    // Laboratorio para entrenar redes experimentales en vivo.
    estado.laboratorio = new Laboratorio(modelo.muestra_prueba, notificar);

    construirGraficasEntrenamiento(modelo.historial);
    pintarUmbral();
    construirMetricas(modelo);
    configurarEventos();

    $("cargando").classList.add("oculto");
    // Primer análisis con los valores por defecto (no se guarda en el historial).
    analizar({ registrar: false });
    if (sessionStorage.getItem("reiniciado")) {
      sessionStorage.removeItem("reiniciado");
      notificar("Plataforma reiniciada: historial vacío, umbral 0.5 y laboratorio limpio.");
    }
  } catch (error) {
    $("cargando").innerHTML =
      `<p style="color:#ff8da0">No se pudo conectar con el servidor de la red neuronal.</p>` +
      `<p>Ejecute <code>python app.py</code> y recargue la página.</p>`;
    console.error(error);
  }
}

async function pedirJSON(url, opciones = {}) {
  // Toda petición lleva el umbral de este usuario (cabecera X-Umbral).
  const cabeceras = { ...cabeceraUmbral(), ...(opciones.headers || {}) };
  const respuesta = await fetch(url, { ...opciones, headers: cabeceras });
  const datos = await respuesta.json();
  if (!respuesta.ok && !datos.errores) throw new Error(`Error ${respuesta.status} en ${url}`);
  return datos;
}

const porcentaje = (v, dec = 1) => `${(v * 100).toFixed(dec)}%`;

/* =============================================================
   2. EVENTOS DEL FORMULARIO
   ============================================================= */
function configurarEventos() {
  const parejas = [
    ["intentos", "rangoIntentos"],
    ["trafico", "rangoTrafico"],
  ];
  // Sincroniza cada campo numérico con su deslizador.
  for (const [idCampo, idRango] of parejas) {
    $(idCampo).addEventListener("input", () => {
      $(idRango).value = $(idCampo).value;
      rellenarDeslizador($(idRango));
      validarCampo(idCampo);
    });
    rellenarDeslizador($(idRango));
    $(idRango).addEventListener("input", () => {
      $(idCampo).value = $(idRango).value;
      rellenarDeslizador($(idRango));
      validarCampo(idCampo);
      if ($("modoVivo").checked) analizarConPausa();
    });
  }

  // Botones + y −.
  document.querySelectorAll(".paso").forEach((boton) =>
    boton.addEventListener("click", () => {
      const campo = $(boton.dataset.objetivo);
      const nuevo = Math.max(0, (parseFloat(campo.value) || 0) + Number(boton.dataset.delta));
      campo.value = boton.dataset.objetivo === "trafico" ? nuevo.toFixed(2).replace(/\.00$/, "") : nuevo;
      campo.dispatchEvent(new Event("input"));
      if ($("modoVivo").checked) analizarConPausa();
    })
  );

  // Botón Analizar (envío del formulario).
  $("formulario").addEventListener("submit", (e) => {
    e.preventDefault();
    analizar();
  });

  // Casos de prueba de la Fase 6.
  document.querySelectorAll(".caso[data-intentos]").forEach((boton) =>
    boton.addEventListener("click", () => {
      ponerValores(boton.dataset.intentos, boton.dataset.trafico);
      analizar();
    })
  );
  $("casoAleatorio").addEventListener("click", () => {
    ponerValores(Math.floor(Math.random() * 21), Math.round(Math.random() * 2000));
    analizar();
  });

  // Controles de la vista 3D.
  $("botonRotar").addEventListener("click", (e) => {
    const activo = e.currentTarget.classList.toggle("activo");
    estado.red3d.setAutoRotar(activo);
  });
  $("botonEtiquetas").addEventListener("click", (e) => {
    const activo = e.currentTarget.classList.toggle("activo");
    estado.red3d.setValores(activo);
  });
  $("botonVista").addEventListener("click", () => {
    estado.red3d.centrar();
    $("detalleNeurona").innerHTML =
      '<p class="placeholder">Haga clic en una neurona de la red para ver su cálculo: <code>z = Σ(wᵢ·xᵢ) + b</code> y su activación.</p>';
  });

  // ✋ Control con las manos: abre la red en PANTALLA COMPLETA y enciende la cámara.
  $("botonManos").addEventListener("click", () => {
    if (estado.manos?.activo) return apagarManos();
    entrarInmersivo();
    encenderManos();
  });
  $("botonManosInmersivo").addEventListener("click", () =>
    estado.manos?.activo ? apagarManos() : encenderManos()
  );
  // ⛶ Pantalla completa sin cámara (para usar con ratón o con los dedos).
  $("botonCompleta").addEventListener("click", entrarInmersivo);
  // ← Regresar: apaga la cámara y vuelve a la página normal.
  $("botonRegresar").addEventListener("click", salirInmersivo);
  $("botonCentrarInmersivo").addEventListener("click", () => $("botonVista").click());
  // Si el usuario sale con la tecla Esc del navegador, también se regresa.
  document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement && $("escena3d").classList.contains("inmersivo")) salirInmersivo();
  });
  // El cálculo de la neurona seleccionada se copia al recuadro del modo inmersivo.
  new MutationObserver(() => {
    const caja = $("detalleNeurona");
    $("detalleInmersivo").innerHTML = caja.innerHTML;
    $("detalleInmersivo").hidden = !!caja.querySelector(".placeholder");
  }).observe($("detalleNeurona"), { childList: true, subtree: true });

  // Pestañas.
  document.querySelectorAll(".pestana").forEach((pestana) =>
    pestana.addEventListener("click", () => {
      document.querySelectorAll(".pestana, .contenido-pestana").forEach((el) => el.classList.remove("activa"));
      pestana.classList.add("activa");
      $(pestana.dataset.panel).classList.add("activa");
    })
  );

  // Descargar la imagen explicada del resultado actual.
  $("botonDescargar").addEventListener("click", () => descargarImagen(estado.ultimo));

  // Umbral de decisión: la vista cambia al instante y el servidor confirma.
  $("rangoUmbral").addEventListener("input", (e) => cambiarUmbral(Number(e.target.value)));
  $("umbralOficial").addEventListener("click", () => cambiarUmbral(umbral.oficial));

  // Reiniciar todo (con confirmación).
  $("botonReiniciar").addEventListener("click", () => abrirModal(true));
  $("cancelarReiniciar").addEventListener("click", () => abrirModal(false));
  $("modalReiniciar").addEventListener("click", (e) => e.target.id === "modalReiniciar" && abrirModal(false));
  document.addEventListener("keydown", (e) => e.key === "Escape" && abrirModal(false));
  $("confirmarReiniciar").addEventListener("click", reiniciarTodo);

  $("verPuntos").addEventListener("change", (e) => estado.mapa.setVerPuntos(e.target.checked));
  $("verNiveles").addEventListener("change", (e) => estado.mapa.setVerNiveles(e.target.checked));
  $("girarSuperficie").addEventListener("change", (e) => estado.superficie.setAutoRotar(e.target.checked));
}

/** Pinta la parte "llena" del deslizador según su valor. */
function rellenarDeslizador(rango) {
  const porcentajeLleno = ((rango.value - rango.min) / (rango.max - rango.min)) * 100;
  rango.style.setProperty("--relleno", `${Math.min(Math.max(porcentajeLleno, 0), 100)}%`);
}

function ponerValores(intentos, trafico) {
  $("intentos").value = intentos;
  $("trafico").value = trafico;
  $("rangoIntentos").value = intentos;
  $("rangoTrafico").value = trafico;
  rellenarDeslizador($("rangoIntentos"));
  rellenarDeslizador($("rangoTrafico"));
  validarCampo("intentos");
  validarCampo("trafico");
}

let temporizador = null;
function analizarConPausa() {
  clearTimeout(temporizador);
  temporizador = setTimeout(() => analizar({ silencioso: true }), 160);
}

/* =============================================================
   3. VALIDACIÓN DE DATOS (en el navegador; el servidor valida otra vez)
   ============================================================= */
function validarCampo(id) {
  const texto = $(id).value.trim();
  const valor = Number(texto);
  const limite = estado.modelo?.limites?.[id] ?? (id === "intentos" ? 100 : 10000);
  let error = "";
  if (texto === "" || Number.isNaN(valor)) error = "Este campo es obligatorio y debe ser numérico.";
  else if (valor < 0) error = "No se permiten valores negativos.";
  else if (id === "intentos" && !Number.isInteger(valor)) error = "Los intentos deben ser un número entero.";
  else if (valor > limite) error = `El máximo permitido es ${limite}${id === "trafico" ? " MB" : ""}.`;
  mostrarError(id, error);
  return !error;
}

function mostrarError(id, mensaje) {
  const nombre = id === "intentos" ? "Intentos" : "Trafico";
  $(`error${nombre}`).textContent = mensaje;
  $(`campo${nombre}`).classList.toggle("invalido", Boolean(mensaje));
}

/* =============================================================
   4. ANÁLISIS: comunicación con la red neuronal en el servidor
   ============================================================= */
async function analizar({ silencioso = false, registrar = true } = {}) {
  const valido = [validarCampo("intentos"), validarCampo("trafico")].every(Boolean);
  if (!valido) {
    notificar("Revise los datos marcados en rojo.", true);
    return;
  }
  if (estado.ocupado && !silencioso) return;
  estado.ocupado = true;
  $("botonAnalizar").disabled = true;

  try {
    const respuesta = await pedirJSON("/api/analizar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ intentos: $("intentos").value, trafico: $("trafico").value }),
    });
    if (!respuesta.ok) {
      // Errores de validación devueltos por el servidor.
      for (const [campo, mensaje] of Object.entries(respuesta.errores)) mostrarError(campo, mensaje);
      notificar("El servidor rechazó los datos: revise los campos.", true);
      return;
    }
    estado.ultimo = respuesta;
    mostrarResultado(respuesta, registrar);
  } catch (error) {
    notificar("No hay conexión con el servidor (¿está ejecutándose app.py?).", true);
    console.error(error);
  } finally {
    estado.ocupado = false;
    $("botonAnalizar").disabled = false;
  }
}

/* =============================================================
   5. PRESENTACIÓN DEL RESULTADO
   ============================================================= */
function mostrarResultado(r, registrar = true) {
  const p = r.probabilidad;

  // Medidor: el arco mide ~283 unidades (media circunferencia de radio 90).
  $("arcoValor").style.strokeDashoffset = 283 * (1 - p);
  $("aguja").style.transform = `rotate(${-90 + 180 * p}deg)`;
  animarNumero($("textoProbabilidad"), p);

  // Veredicto.
  pintarVeredicto(r.clase === 1);

  // Explicación e influencia de cada entrada.
  $("textoExplicacion").textContent = r.explicacion;
  $("barrasEfecto").hidden = false;
  pintarEfecto("Intentos", r.efectos.intentos);
  pintarEfecto("Trafico", r.efectos.trafico);
  $("listaAdvertencias").innerHTML = r.advertencias.map((a) => `<li>⚠ ${a}</li>`).join("");

  // Animación 3D, mapa, cálculo e historial.
  estado.red3d.mostrarRecorrido(r.recorrido, r.normalizada);
  estado.mapa.setActual(r.entrada.intentos, r.entrada.trafico, p);
  estado.superficie.setActual(r.entrada.intentos, r.entrada.trafico, p);
  construirCalculo(r);
  estado.sigmoide.setZ(r.recorrido[r.recorrido.length - 1].z[0]); // z de la neurona de salida
  $("botonDescargar").disabled = false;
  explicarSigmoide();
  if (registrar) agregarHistorial(r);
  if (estado.red3d.seleccion) {
    mostrarDetalleNeurona(estado.red3d.seleccion.capa, estado.red3d.seleccion.indice);
  }
}

/** Anima el número de la probabilidad desde su valor anterior al nuevo. */
function animarNumero(elemento, destino) {
  const origen = Number(elemento.dataset.valor ?? 0);
  const inicio = performance.now();
  const paso = (ahora) => {
    const t = Math.min((ahora - inicio) / 800, 1);
    const suave = 1 - Math.pow(1 - t, 3);
    elemento.textContent = porcentaje(origen + (destino - origen) * suave, 2);
    if (t < 1) requestAnimationFrame(paso);
  };
  requestAnimationFrame(paso);
  elemento.dataset.valor = destino;
}

/** Barra divergente: hacia la derecha (rosa) sube la probabilidad, a la izquierda (cian) la baja. */
function pintarEfecto(nombre, efecto) {
  const barra = $(`efecto${nombre}`);
  const ancho = Math.min(Math.abs(efecto), 1) * 50;
  barra.style.width = `${ancho}%`;
  barra.style.left = efecto >= 0 ? "50%" : `${50 - ancho}%`;
  barra.style.background = efecto >= 0 ? "var(--sospechosa)" : "var(--normal)";
  $(`valorEfecto${nombre}`).textContent = `${efecto >= 0 ? "+" : "−"}${Math.abs(efecto * 100).toFixed(1)} pts`;
}

/* =============================================================
   6. DETALLE MATEMÁTICO DE UNA NEURONA (clic en la red 3D)
   ============================================================= */
function mostrarDetalleNeurona(capa, indice) {
  const caja = $("detalleNeurona");
  const r = estado.ultimo;
  const m = estado.modelo;
  if (!r) {
    caja.innerHTML = '<p class="placeholder">Primero analice una conexión.</p>';
    return;
  }
  const f = (v) => v.toFixed(4);
  const signo = (v) => `<span class="${v >= 0 ? "pos" : "neg"}">${f(v)}</span>`;

  if (capa === 0) {
    // Neurona de entrada: muestra la normalización.
    const nombre = indice === 0 ? "intentos fallidos" : "tráfico (MB)";
    const original = indice === 0 ? r.entrada.intentos : r.entrada.trafico;
    const media = m.normalizador.media[indice];
    const desv = m.normalizador.desviacion[indice];
    caja.innerHTML = `<h4>Entrada x${indice + 1} · ${nombre}</h4>
      <div class="formula">x${indice + 1} = (valor − media) / desviación = (${original} − ${media.toFixed(3)}) / ${desv.toFixed(3)} = <b>${f(r.normalizada[indice])}</b></div>
      <p class="placeholder">Las entradas no calculan nada: solo entregan el valor normalizado a la capa oculta 1.</p>`;
    return;
  }

  // Neurona de una capa densa: z = Σ(w·a) + b y activación.
  const capaModelo = m.capas[capa - 1];
  const entradas = capa === 1 ? r.normalizada : r.recorrido[capa - 2].a;
  const pesos = capaModelo.pesos.map((fila) => fila[indice]);
  const sesgo = capaModelo.sesgos[indice];
  const z = r.recorrido[capa - 1].z[indice];
  const a = r.recorrido[capa - 1].a[indice];
  const nombreEntrada = capa === 1 ? "x" : "h";
  const terminos = pesos.map((w, i) => `(${signo(w)}·${f(entradas[i])})`).join(" + ");
  const titulo = capa === 3 ? "Neurona de salida ŷ" : `Capa oculta ${capa} · neurona h${indice + 1}`;
  const activacion =
    capaModelo.activacion === "relu"
      ? `ReLU(z) = max(0, ${f(z)}) = <b>${f(a)}</b> ${a === 0 ? "→ neurona apagada (no aporta a la siguiente capa)" : "→ neurona activa"}`
      : `σ(z) = 1 / (1 + e^(−${f(z)})) = <b>${f(a)}</b> → ${porcentaje(a, 2)} ${esSospechosa(a) ? "≥" : "<"} umbral ${textoUmbral()} → <b>${esSospechosa(a) ? "SOSPECHOSA" : "NORMAL"}</b>`;
  caja.innerHTML = `<h4>${titulo} · activación ${capaModelo.activacion}</h4>
    <div class="formula">z = Σ(w·${nombreEntrada}) + b = ${terminos} + ${signo(sesgo)} = <b>${f(z)}</b></div>
    <div class="formula">${activacion}</div>`;
}

/* =============================================================
   7. PESTAÑA "CÁLCULO PASO A PASO"
   ============================================================= */
function construirCalculo(r) {
  const m = estado.modelo;
  const f = (v) => v.toFixed(4);
  const [x1, x2] = r.normalizada;
  let html = `<div class="paso-capa">
      <h4>1 · Normalización</h4>
      <div class="formula-capa">x = (valor − media) / desv.</div>
      <div class="fila-neurona cab"><span></span><span>original</span><span>normalizado</span></div>
      <div class="fila-neurona" data-capa="0" data-i="0"><span>x₁</span><span>${r.entrada.intentos}</span><span>${f(x1)}</span></div>
      <div class="fila-neurona" data-capa="0" data-i="1"><span>x₂</span><span>${r.entrada.trafico}</span><span>${f(x2)}</span></div>
      <p class="placeholder" style="margin-top:8px;font-size:12px">media = [${m.normalizador.media.map((v) => v.toFixed(2)).join(", ")}]<br>desv. = [${m.normalizador.desviacion.map((v) => v.toFixed(2)).join(", ")}]</p>
    </div>`;
  const nombres = ["2 · Oculta 1 (8 ReLU)", "3 · Oculta 2 (4 ReLU)", "4 · Salida (sigmoide)"];
  const formulas = ["a = max(0, Σw·x + b)", "a = max(0, Σw·h + b)", "p = 1 / (1 + e^(−z))"];
  r.recorrido.forEach((paso, l) => {
    html += `<div class="paso-capa"><h4>${nombres[l]}</h4><div class="formula-capa">${formulas[l]}</div>
      <div class="fila-neurona cab"><span></span><span>z</span><span>activación</span></div>`;
    paso.z.forEach((z, i) => {
      const a = paso.a[i];
      const etiqueta = l === 2 ? "ŷ" : `h${i + 1}`;
      html += `<div class="fila-neurona" data-capa="${l + 1}" data-i="${i}"><span>${etiqueta}</span><span>${f(z)}</span><span class="${a === 0 ? "apagada" : ""}">${f(a)}${a === 0 ? " (apagada)" : ""}</span></div>`;
    });
    if (l === 2) {
      html += `<p style="margin:10px 0 0;font-size:13px">p = <b>${porcentaje(r.probabilidad, 2)}</b> ${r.clase ? "≥" : "<"} umbral ${r.umbral.toFixed(2)}<br>→ <b style="color:${r.clase ? "#ff8da0" : "#5fd3ea"}">${r.clasificacion}</b></p>`;
    }
    html += "</div>";
  });
  $("calculoPasos").innerHTML = html;
  // Clic en una fila = seleccionar esa neurona en la vista 3D.
  $("calculoPasos").querySelectorAll(".fila-neurona[data-capa]").forEach((fila) =>
    fila.addEventListener("click", () => {
      estado.red3d.seleccionar(Number(fila.dataset.capa), Number(fila.dataset.i));
      $("escena3d").scrollIntoView({ behavior: "smooth", block: "center" });
    })
  );
}

/* =============================================================
   8. HISTORIAL DE ANÁLISIS
   ============================================================= */
function agregarHistorial(r) {
  const ultimo = estado.historial[0];
  // Evita repetir la misma conexión seguida (por ejemplo, al arrastrar).
  if (ultimo && ultimo.entrada.intentos === r.entrada.intentos && ultimo.entrada.trafico === r.entrada.trafico && ultimo.umbral === r.umbral) return;
  estado.historial.unshift(r);
  estado.historial = estado.historial.slice(0, 50);
  $("contadorHistorial").textContent = estado.historial.length;
  $("cuerpoHistorial").innerHTML = estado.historial
    .map((h, k) => {
      const clase = h.clase ? "sospechosa" : "normal";
      return `<tr><td>${estado.historial.length - k}</td><td>${h.entrada.intentos}</td><td>${h.entrada.trafico}</td>
        <td>${porcentaje(h.probabilidad, 2)}</td><td>${h.umbral.toFixed(2)}</td><td><span class="etiqueta-clase ${clase}">${h.clasificacion}</span></td>
        <td><button class="mini" data-k="${k}">Repetir</button> <button class="mini" data-img="${k}" title="Descargar imagen explicada">⬇ Imagen</button></td></tr>`;
    })
    .join("");
  $("cuerpoHistorial").querySelectorAll("button[data-img]").forEach((b) =>
    b.addEventListener("click", () => descargarImagen(estado.historial[Number(b.dataset.img)]))
  );
  $("cuerpoHistorial").querySelectorAll("button[data-k]").forEach((b) =>
    b.addEventListener("click", () => {
      const h = estado.historial[Number(b.dataset.k)];
      ponerValores(h.entrada.intentos, h.entrada.trafico);
      analizar();
    })
  );
}

/* =============================================================
   9. PESTAÑAS "ENTRENAMIENTO" Y "MÉTRICAS" (datos reales del entrenamiento)
   ============================================================= */
function construirGraficasEntrenamiento(h) {
  const perdida = new GraficaLineas($("lienzoPerdida"), [
    { nombre: "entrenamiento", valores: h.loss, color: "#0ea5c4" },
    { nombre: "validación", valores: h.val_loss, color: "#c27a14" },
  ]);
  const exactitud = new GraficaLineas(
    $("lienzoExactitud"),
    [
      { nombre: "entrenamiento", valores: h.accuracy, color: "#0ea5c4" },
      { nombre: "validación", valores: h.val_accuracy, color: "#c27a14" },
    ],
    { formato: (v) => porcentaje(v) }
  );
  $("botonRepetir").addEventListener("click", () => {
    perdida.reproducir();
    exactitud.reproducir();
  });
  const n = h.loss.length;
  // Diferencia final entre entrenamiento y validación: si es grande, hay sobreajuste.
  const brecha = h.accuracy[n - 1] - h.val_accuracy[n - 1];
  const textoBrecha =
    Math.abs(brecha) < 0.05
      ? `La exactitud de validación (${porcentaje(h.val_accuracy[n - 1])}) queda cerca de la de entrenamiento: no hay sobreajuste importante.`
      : `La exactitud de validación (${porcentaje(h.val_accuracy[n - 1])}) se separa de la de entrenamiento: posible sobreajuste.`;
  $("resumenEntrenamiento").innerHTML =
    `En ${n} épocas la pérdida bajó de <b>${h.loss[0].toFixed(4)}</b> a <b>${h.loss[n - 1].toFixed(4)}</b> ` +
    `y la exactitud subió de <b>${porcentaje(h.accuracy[0])}</b> a <b>${porcentaje(h.accuracy[n - 1])}</b>: la red aprendió ajustando sus ${estado.modelo.parametros_entrenables} pesos y sesgos con <code>modelo.fit()</code>. ` +
    textoBrecha;
}

function construirMetricas(m) {
  const { exactitud, precision, recall, perdida, f1, matriz_confusion: mc } = m.metricas;
  $("chipExactitud").textContent = porcentaje(exactitud);
  $("notaMetricas").innerHTML =
    umbral.valor === umbral.oficial
      ? `Métricas con los ${m.datos.prueba} datos de prueba y el <b>umbral oficial ${textoUmbral()}</b>. Mueva el umbral en el panel <b>Resultado</b> para ver cómo cambian.`
      : `Métricas recalculadas con el <b>umbral ${textoUmbral()}</b> (el oficial es ${umbral.oficial.toFixed(2)}). La red y sus probabilidades son las mismas: solo cambia la regla <code>p ≥ umbral</code>. ` +
        (umbral.valor < umbral.oficial
          ? "Al bajarlo hay <b>más alertas</b>: sube el recall y aparecen más falsas alarmas."
          : "Al subirlo hay <b>menos alertas</b>: sube la precisión, pero se escapan más conexiones sospechosas.");
  const tarjetas = [
    ["Exactitud (accuracy)", exactitud, `De ${m.datos.prueba} conexiones de prueba acertó ${mc.vn + mc.vp}. Es el porcentaje general de aciertos.`],
    ["Precisión", precision, `De ${mc.vp + mc.fp} alertas de "sospechosa", ${mc.vp} eran reales. Hubo ${mc.fp} falsas alarmas.`],
    ["Recall (sensibilidad)", recall, `Detectó ${mc.vp} de ${mc.vp + mc.fn} conexiones sospechosas reales; se le escaparon ${mc.fn}.`],
    ["F1-score", f1, "Promedio armónico entre precisión y recall: resume el equilibrio entre ambas."],
  ];
  $("tarjetasMetricas").innerHTML =
    tarjetas
      .map(
        ([nombre, v, texto]) => `<div class="tarjeta"><div class="etiqueta">${nombre}</div>
        <div class="cifra">${porcentaje(v)}</div><div class="barra"><i data-ancho="${v * 100}"></i></div><p>${texto}</p></div>`
      )
      .join("") +
    `<div class="tarjeta" style="grid-column:1/-1"><div class="etiqueta">Pérdida (binary crossentropy)</div>
      <div class="cifra">${perdida.toFixed(4)}</div><p>Error promedio de las probabilidades frente a las etiquetas reales. Entre más cerca de 0, mejor.</p></div>`;
  // Anima las barras después de pintarlas.
  requestAnimationFrame(() =>
    document.querySelectorAll(".tarjeta .barra i").forEach((i) => (i.style.width = `${i.dataset.ancho}%`))
  );

  $("matrizConfusion").innerHTML = `
    <span></span><span class="cab">Predijo normal</span><span class="cab">Predijo sospechosa</span>
    <span class="lado">Real<br>normal</span>
    <div class="celda acierto"><b>${mc.vn}</b><span>verdaderos negativos</span></div>
    <div class="celda fallo"><b>${mc.fp}</b><span>falsas alarmas</span></div>
    <span class="lado">Real<br>sospechosa</span>
    <div class="celda fallo"><b>${mc.fn}</b><span>no detectadas</span></div>
    <div class="celda acierto"><b>${mc.vp}</b><span>verdaderos positivos</span></div>`;

  const d = m.datos;
  $("datosDataset").innerHTML = `
    <div>Conexiones simuladas: <b>${d.total}</b></div>
    <div>Normales (0): <b>${d.normales}</b></div>
    <div>Sospechosas (1): <b>${d.sospechosas}</b></div>
    <div>Entrenamiento: <b>${d.entrenamiento}</b></div>
    <div>Prueba: <b>${d.prueba}</b></div>
    <div>Arquitectura: <b>${m.arquitectura}</b></div>
    <div>Parámetros entrenables: <b>${m.parametros_entrenables}</b></div>`;
}

/* =============================================================
   10. UMBRAL DE DECISIÓN MODIFICABLE
   ============================================================= */
let temporizadorUmbral = null;

/** Cambia el umbral: actualiza la vista al instante y luego le avisa al servidor. */
function cambiarUmbral(valor) {
  valor = Math.round(valor * 100) / 100;
  umbral.valor = valor;
  $("rangoUmbral").value = valor;
  pintarUmbral();

  // Lo que se puede recalcular en el navegador cambia de inmediato.
  estado.mapa.refrescar(); // frontera, colores y errores de los datos de prueba
  estado.laboratorio.refrescarMapa();
  if (estado.ultimo) {
    pintarVeredicto(esSospechosa(estado.ultimo.probabilidad));
    explicarSigmoide();
  }

  // El servidor recalcula métricas, explicación y laboratorio (con una pausa
  // corta para no enviar una petición por cada pixel del deslizador).
  clearTimeout(temporizadorUmbral);
  temporizadorUmbral = setTimeout(async () => {
    try {
      const datos = await pedirJSON("/api/umbral", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ umbral: valor }),
      });
      if (!datos.ok) return notificar(datos.error, true);
      estado.modelo.metricas = datos.metricas;
      estado.modelo.muestra_prueba.forEach((punto, i) => (punto.predicho = datos.predicciones[i]));
      construirMetricas(estado.modelo);
      pintarResumenUmbral();
      estado.laboratorio.refrescar();
      // Explicación, red 3D y cálculo con el umbral nuevo.
      if (estado.ultimo) analizar({ silencioso: true, registrar: false });
      document.querySelectorAll("#matrizConfusion .celda").forEach((c) => {
        c.classList.remove("destello");
        void c.offsetWidth; // reinicia la animación
        c.classList.add("destello");
      });
    } catch (error) {
      notificar("No se pudo cambiar el umbral en el servidor.", true);
      console.error(error);
    }
  }, 180);
}

/** Pinta el umbral en el medidor, la regla, las etiquetas y la escala de color. */
function pintarUmbral() {
  const u = umbral.valor;
  const t = textoUmbral();
  const modificado = u !== umbral.oficial;
  $("textoUmbral").textContent = t;
  $("textoRegla").textContent = `Regla: p ≥ ${t} → sospechosa · p < ${t} → normal`;
  $("insigniaUmbral").textContent = modificado ? "modificado" : "oficial";
  $("controlUmbral").classList.toggle("modificado", modificado);
  $("umbralOficial").disabled = !modificado;
  $("etiquetaUmbral").textContent = `umbral ${t}`;
  document.querySelectorAll(".valor-umbral").forEach((el) => (el.textContent = t));
  rellenarDeslizador($("rangoUmbral"));

  // Marca del umbral sobre el arco del medidor (centro 110,115 y radio 90).
  const angulo = Math.PI * (1 - u);
  const punto = (radio) => [110 + radio * Math.cos(angulo), 115 - radio * Math.sin(angulo)];
  const [x1, y1] = punto(97);
  const [x2, y2] = punto(81);
  const [xt, yt] = punto(106);
  const marca = $("marcaUmbral");
  marca.setAttribute("x1", x1);
  marca.setAttribute("y1", y1);
  marca.setAttribute("x2", x2);
  marca.setAttribute("y2", y2);
  const etiqueta = $("etiquetaUmbral");
  etiqueta.setAttribute("x", Math.min(Math.max(xt, 42), 178));
  etiqueta.setAttribute("y", Math.max(yt - 2, 10));
  // El gris del degradado del arco queda justo en el umbral.
  $("paradaUmbral").setAttribute("offset", (1 - Math.cos(Math.PI * u)) / 2);
  document.documentElement.style.setProperty("--umbral-pct", `${u * 100}%`);
  pintarResumenUmbral();
}

/** Mini resumen bajo el deslizador: cómo cambian precisión, recall y errores. */
function pintarResumenUmbral() {
  const m = estado.modelo?.metricas;
  const o = estado.modelo?.metricas_umbral_oficial;
  if (!m || !o) return;
  const mc = m.matriz_confusion;
  const oc = o.matriz_confusion;
  // "mejor" indica si un aumento es bueno (+1) o malo (-1) para esa medida.
  const celda = (nombre, valor, base, texto, mejor) => {
    const d = valor - base;
    const clase = Math.abs(d) < 1e-9 ? "" : d * mejor > 0 ? "sube" : "baja";
    const cambio = Math.abs(d) < 1e-9 ? "" : `${d > 0 ? "+" : "−"}${texto(Math.abs(d))}`;
    return `<div><span>${nombre}</span><b>${texto(valor)}</b><small class="${clase}">${cambio}</small></div>`;
  };
  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  const n = (v) => String(v);
  $("resumenUmbral").innerHTML =
    celda("Precisión", m.precision, o.precision, pct, 1) +
    celda("Recall", m.recall, o.recall, pct, 1) +
    celda("Falsas alarmas", mc.fp, oc.fp, n, -1) +
    celda("No detectadas", mc.fn, oc.fn, n, -1);
}

/** Texto bajo la curva sigmoide: z, probabilidad y por qué cae de un lado del umbral. */
function explicarSigmoide() {
  const r = estado.ultimo;
  if (!r) return;
  const z = r.recorrido[r.recorrido.length - 1].z[0];
  const p = r.probabilidad;
  const u = umbral.valor;
  const zCorte = Math.log(u / (1 - u)); // z donde la curva vale exactamente el umbral
  const sospechosa = esSospechosa(p);
  $("textoSigmoide").innerHTML =
    `La neurona de salida sumó <b>z = ${z.toFixed(2)}</b> y la sigmoide lo convirtió en ` +
    `<b>p = ${porcentaje(p, 2)}</b>. El umbral ${textoUmbral()} equivale a cortar la curva en ` +
    `z = ${zCorte.toFixed(2)}: como z ${sospechosa ? "≥" : "<"} ${zCorte.toFixed(2)}, la conexión es ` +
    `<b class="${sospechosa ? "txt-sospechosa" : "txt-normal"}">${sospechosa ? "SOSPECHOSA" : "NORMAL"}</b>.`;
}

/* =============================================================
   ⛶ MODO PANTALLA COMPLETA + ✋ CONTROL CON LAS MANOS
   ============================================================= */

/** La red 3D ocupa toda la pantalla (como un videojuego). */
function entrarInmersivo() {
  if ($("escena3d").classList.contains("inmersivo")) return;
  $("escena3d").classList.add("inmersivo");
  document.body.classList.add("modo-inmersivo");
  // Tras el cambio de tamaño, encuadra la red: en pantallas anchas se acerca
  // un poco; en un celular vertical se deja completa a lo ancho.
  setTimeout(() => estado.red3d.encuadrar(innerWidth > innerHeight ? 0.8 : 1), 250);
  // Pantalla completa real (oculta la barra del navegador). Si el navegador
  // no lo permite, igual queda ocupando toda la ventana.
  document.documentElement.requestFullscreen?.().catch(() => {});
}

/** ← Regresar: apaga la cámara y vuelve a la vista normal de la página. */
function salirInmersivo() {
  apagarManos();
  $("escena3d").classList.remove("inmersivo");
  setTimeout(() => estado.red3d.encuadrar(), 250); // vuelve al encuadre del panel
  document.body.classList.remove("modo-inmersivo");
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
}

/** Pinta el estado de los dos botones de manos (página e inmersivo). */
function pintarBotonesManos(texto, activo, cargando = false) {
  for (const id of ["botonManos", "botonManosInmersivo"]) {
    $(id).textContent = texto;
    $(id).classList.toggle("activo", activo);
    $(id).classList.toggle("cargando", cargando);
  }
}

async function encenderManos() {
  estado.manos ??= new ControlManos(estado.red3d, $("escena3d"), notificar);
  if (estado.manos.activo || $("botonManos").classList.contains("cargando")) return;
  pintarBotonesManos("✋ Cargando…", false, true);
  try {
    await estado.manos.iniciar();
    pintarBotonesManos("✋ Apagar manos", true);
    notificar("Cámara activa: pellizque (pulgar + índice) para seleccionar y girar.");
  } catch (error) {
    console.error(error);
    estado.manos.detener();
    pintarBotonesManos("✋ Manos", false);
    const sinPermiso = error?.name === "NotAllowedError";
    const sinCamara = error?.name === "NotFoundError";
    notificar(
      sinPermiso ? "Permiso de cámara denegado." : sinCamara ? "No se encontró una cámara web." : "No se pudo iniciar el control con las manos.",
      true
    );
  }
}

function apagarManos() {
  if (estado.manos?.activo) estado.manos.detener();
  pintarBotonesManos("✋ Manos", false);
}

/** Genera y descarga la imagen PNG explicada de un resultado. */
async function descargarImagen(r) {
  if (!r) return notificar("Primero analice una conexión.", true);
  try {
    await descargarImagenAnalisis(r, estado.modelo);
    notificar("Imagen descargada: revise su carpeta de Descargas.");
  } catch (error) {
    notificar("No se pudo generar la imagen.", true);
    console.error(error);
  }
}

/** Veredicto (icono, color y texto) según la clase. */
function pintarVeredicto(sospechosa) {
  const veredicto = $("veredicto");
  veredicto.className = `veredicto ${sospechosa ? "sospechosa" : "normal"}`;
  veredicto.querySelector(".icono").textContent = sospechosa ? "!" : "✓";
  $("textoClase").textContent = sospechosa ? "Conexión SOSPECHOSA" : "Conexión NORMAL";
}

/* =============================================================
   11. REINICIAR LA PLATAFORMA DESDE CERO
   ============================================================= */
function abrirModal(abrir) {
  $("modalReiniciar").hidden = !abrir;
  if (abrir) $("cancelarReiniciar").focus();
}

async function reiniciarTodo() {
  const boton = $("confirmarReiniciar");
  boton.disabled = true;
  boton.textContent = "Reiniciando…";
  try {
    // El servidor vuelve al umbral 0.5, detiene el laboratorio y borra los experimentos.
    await pedirJSON("/api/reiniciar", { method: "POST" });
    // Al recargar, el historial (que vive en el navegador) queda vacío y
    // todas las vistas arrancan como la primera vez.
    sessionStorage.setItem("reiniciado", "1");
    window.scrollTo(0, 0);
    location.reload();
  } catch (error) {
    boton.disabled = false;
    boton.textContent = "Sí, reiniciar todo";
    notificar("No se pudo reiniciar: ¿está ejecutándose app.py?", true);
    console.error(error);
  }
}

/* =============================================================
   12. NOTIFICACIONES
   ============================================================= */
let temporizadorAviso = null;
function notificar(texto, esError = false) {
  const n = $("notificacion");
  n.textContent = texto;
  n.className = `notificacion${esError ? " error" : ""}`;
  n.hidden = false;
  clearTimeout(temporizadorAviso);
  temporizadorAviso = setTimeout(() => (n.hidden = true), 3200);
}

iniciar();
