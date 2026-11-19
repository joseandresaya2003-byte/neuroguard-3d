/* =============================================================
   laboratorio.js - Pestaña "Laboratorio": entrenar redes en vivo
   -------------------------------------------------------------
   Objetivo del taller: "Modificar la arquitectura y observar cómo
   cambian los resultados".
   Flujo:
     1. El usuario elige activación, tasa de aprendizaje, neuronas y épocas.
     2. POST /api/laboratorio/entrenar -> el servidor entrena una red
        nueva con Keras en segundo plano (mismos datos del modelo oficial).
     3. Mientras entrena, se consulta GET /api/laboratorio/estado cada
        250 ms: llegan la época, la pérdida y el mapa de decisión, y el mapa
        2D y la superficie 3D se transforman en vivo.
     4. Al terminar se muestran exactitud, precisión, recall y pérdida, y
        el experimento se agrega a la tabla de comparación.
   ============================================================= */

import { MapaDecision, GraficaLineas } from "./graficas.js";
import { cabeceraUmbral } from "./umbral.js";
import { Superficie3D } from "./superficie3d.js";

const $ = (id) => document.getElementById(id);
const porcentaje = (v) => `${(v * 100).toFixed(1)}%`;

// Explicación breve de cada función de activación.
const DESCRIPCIONES = {
  relu: "<b>ReLU</b> · max(0, z). Deja pasar los valores positivos y apaga los negativos. Es rápida, evita que el gradiente se desvanezca y es la que exige el taller.",
  tanh: "<b>Tanh</b> · (eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ). Curva en S entre −1 y 1, centrada en cero. Aprende fronteras suaves, pero se satura con valores grandes.",
  sigmoid: "<b>Sigmoide</b> · 1/(1 + e⁻ᶻ). Curva en S entre 0 y 1. En capas ocultas suele aprender más lento porque su gradiente es pequeño (máximo 0.25).",
  linear: "<b>Lineal</b> · f(z) = z. Sin no linealidad, varias capas equivalen a una sola: la red solo puede trazar una frontera RECTA. Aquí acierta casi igual porque la regla de los datos simulados es lineal (0.55·intentos + 0.45·tráfico); con datos reales y fronteras curvas fallaría.",
};
const NOMBRES = { relu: "ReLU", tanh: "Tanh", sigmoid: "Sigmoide", linear: "Lineal" };

export class Laboratorio {
  constructor(muestraPrueba, notificar) {
    this.notificar = notificar;
    this.config = { activacion: "relu", tasa: 0.001, epocas: 100, neuronas1: 8, neuronas2: 4 };
    this.entrenando = false;
    this.version = -1;

    // Mapa 2D de solo lectura (sin clic) y superficie 3D sin selección.
    this.mapa = new MapaDecision($("lienzoLab"), $("tooltipLab"), null);
    // En el laboratorio los puntos se dibujan solo por su clase real.
    this.mapa.setPuntos(muestraPrueba.map((p) => ({ ...p, predicho: p.real })));
    this.superficie = new Superficie3D($("superficieLab"));
    this.grafica = new GraficaLineas(
      $("lienzoLabPerdida"),
      [
        { nombre: "entrenamiento", valores: [], color: "#0ea5c4" },
        { nombre: "validación", valores: [], color: "#c27a14" },
      ],
      { epocas: 100 }
    );

    this._eventos();
    this._pintarConfig();
    this._consultar(); // Carga la tabla de experimentos (incluye el modelo oficial).
  }

  _eventos() {
    // Botones de activación (grupo de opciones).
    document.querySelectorAll("[data-activacion]").forEach((boton) =>
      boton.addEventListener("click", () => {
        if (this.entrenando) return;
        this.config.activacion = boton.dataset.activacion;
        this._pintarConfig();
      })
    );
    // Steppers de neuronas.
    document.querySelectorAll("[data-lab]").forEach((boton) =>
      boton.addEventListener("click", () => {
        if (this.entrenando) return;
        const clave = boton.dataset.lab;
        const maximo = clave === "neuronas1" ? 16 : 8;
        this.config[clave] = Math.min(maximo, Math.max(1, this.config[clave] + Number(boton.dataset.delta)));
        this._pintarConfig();
      })
    );
    $("labTasa").addEventListener("change", (e) => (this.config.tasa = Number(e.target.value)));
    $("labEpocas").addEventListener("change", (e) => {
      this.config.epocas = Number(e.target.value);
      this._pintarConfig();
    });
    $("labBoton").addEventListener("click", () => (this.entrenando ? this.detener() : this.entrenar()));
  }

  /** Refleja la configuración elegida en los controles. */
  _pintarConfig() {
    const c = this.config;
    document.querySelectorAll("[data-activacion]").forEach((b) => {
      const activa = b.dataset.activacion === c.activacion;
      b.classList.toggle("activa", activa);
      b.setAttribute("aria-checked", activa);
    });
    $("labNeuronas1").textContent = c.neuronas1;
    $("labNeuronas2").textContent = c.neuronas2;
    $("labEpocaTotal").textContent = c.epocas;
    $("labDescripcion").innerHTML = DESCRIPCIONES[c.activacion];
    $("labArquitectura").textContent = `2-${c.neuronas1}-${c.neuronas2}-1 · ${NOMBRES[c.activacion]}`;
  }

  /** Vuelve a pedir el estado (por ejemplo, la tabla con un umbral nuevo). */
  refrescar() {
    this.version = -1; // Obliga a redibujar aunque el entrenamiento no haya cambiado.
    if (!this.entrenando) this._consultar();
  }

  /** Redibuja el mapa del laboratorio (la frontera sigue al umbral). */
  refrescarMapa() {
    this.mapa.refrescar();
  }

  async entrenar() {
    const respuesta = await fetch("/api/laboratorio/entrenar", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...cabeceraUmbral() },
      body: JSON.stringify(this.config),
    });
    const datos = await respuesta.json();
    if (!datos.ok) {
      this.notificar(datos.error, true);
      return;
    }
    this._modoEntrenando(true);
    ["labExactitud", "labPrecision", "labRecall", "labPerdida"].forEach((id) => ($(id).textContent = "…"));
    this._consultar();
  }

  async detener() {
    await fetch("/api/laboratorio/detener", { method: "POST" });
  }

  _modoEntrenando(activo) {
    this.entrenando = activo;
    const boton = $("labBoton");
    boton.classList.toggle("detener", activo);
    boton.querySelector("span").textContent = activo ? "Detener" : "Entrenar";
    document.querySelector(".controles-lab").classList.toggle("bloqueado", activo);
    $("labVivo1").classList.toggle("activo", activo);
    $("labVivo2").classList.toggle("activo", activo);
  }

  /** Pregunta al servidor por el progreso y actualiza todo lo que cambió. */
  async _consultar() {
    let estado;
    try {
      estado = await (await fetch("/api/laboratorio/estado", { headers: cabeceraUmbral() })).json();
    } catch {
      this.notificar("Se perdió la conexión con el servidor.", true);
      this._modoEntrenando(false);
      return;
    }
    if (estado.version !== this.version) {
      this.version = estado.version;
      this._mostrar(estado);
    }
    if (estado.estado === "entrenando") {
      if (!this.entrenando) this._modoEntrenando(true);
      setTimeout(() => this._consultar(), 250);
    } else if (this.entrenando) {
      this._modoEntrenando(false);
      if (estado.estado === "error") this.notificar(`Error al entrenar: ${estado.resultado?.error}`, true);
      else this.notificar(estado.estado === "detenido" ? "Entrenamiento detenido." : "Entrenamiento terminado.");
    }
  }

  _mostrar(estado) {
    const total = estado.config?.epocas ?? this.config.epocas;
    $("labEpoca").textContent = String(estado.epoca).padStart(3, "0");
    $("labEpocaTotal").textContent = total;
    $("labProgreso").style.width = `${(estado.epoca / total) * 100}%`;

    if (estado.mapa) {
      $("labVacio").hidden = true;
      this.mapa.setMapa(estado.mapa);
      this.superficie.setMapa(estado.mapa);
    }
    const h = estado.historial;
    this.grafica.actualizar(
      [
        { nombre: "entrenamiento", valores: h.loss, color: "#0ea5c4" },
        { nombre: "validación", valores: h.val_loss, color: "#c27a14" },
      ],
      total
    );

    const r = estado.resultado;
    if (r && !r.error) {
      // La última fila de la tabla es este experimento, con sus métricas
      // calculadas con el umbral vigente (cambian si se mueve el umbral).
      const fila = estado.experimentos[estado.experimentos.length - 1];
      const m = fila && fila.nombre !== "Oficial" ? fila : r;
      $("labExactitud").textContent = porcentaje(m.exactitud);
      $("labPrecision").textContent = porcentaje(m.precision);
      $("labRecall").textContent = porcentaje(m.recall);
      $("labPerdida").textContent = r.perdida.toFixed(4);
    }
    this._tabla(estado.experimentos);
  }

  /** Tabla de comparación: barra en la exactitud y la mejor fila resaltada. */
  _tabla(experimentos) {
    const mejor = Math.max(...experimentos.map((e) => e.exactitud));
    $("cuerpoExperimentos").innerHTML = experimentos
      .slice()
      .reverse()
      .map((e) => {
        const destacado = e.exactitud === mejor ? " class=\"mejor\"" : "";
        return `<tr${destacado}>
          <td>${e.nombre === "Oficial" ? '<span class="etiqueta-oficial">Oficial</span>' : e.nombre}</td>
          <td>${e.activacion}</td><td>${e.arquitectura}</td><td>${e.tasa}</td><td>${e.epocas}</td>
          <td>${e.perdida.toFixed(4)}</td>
          <td><div class="celda-barra"><i style="width:${e.exactitud * 100}%"></i><span>${porcentaje(e.exactitud)}</span></div></td>
          <td>${porcentaje(e.precision)}</td><td>${porcentaje(e.recall)}</td></tr>`;
      })
      .join("");
  }
}
