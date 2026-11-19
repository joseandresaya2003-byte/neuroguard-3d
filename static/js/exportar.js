/* =============================================================
   exportar.js - Descarga una IMAGEN EXPLICADA del análisis actual
   -------------------------------------------------------------
   Con el resultado que devolvió la red (POST /api/analizar) se dibuja
   un diagrama de las 3 capas con los valores REALES de esa conexión:
     - qué neuronas se encendieron y cuáles se apagaron (ReLU = 0),
     - el valor de activación de cada neurona,
     - z y la probabilidad de la neurona de salida (sigmoide),
     - el umbral usado y la clasificación final,
     - la explicación en texto.
   El dibujo se arma como SVG, se convierte a PNG en un <canvas> y se
   descarga. Todo ocurre en el navegador: no se envía nada a internet.
   ============================================================= */

const NS = "http://www.w3.org/2000/svg";
const AZUL = "#4f8df7", VERDE = "#34c27a", ROJO = "#e8455f", VIOLETA = "#a78bfa";
const CIAN = "#0ea5c4";
const ANCHO = 1600, ALTO = 1260;

/** Corta un texto largo en líneas de máximo "max" caracteres. */
function partirTexto(texto, max) {
  const lineas = [];
  let actual = "";
  for (const palabra of texto.split(" ")) {
    if ((actual + " " + palabra).trim().length > max) {
      lineas.push(actual.trim());
      actual = palabra;
    } else actual += " " + palabra;
  }
  if (actual.trim()) lineas.push(actual.trim());
  return lineas;
}

/** Construye el SVG con el diagrama explicado del resultado "r". */
function construirSVG(r, modelo) {
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("xmlns", NS);
  svg.setAttribute("width", ANCHO);
  svg.setAttribute("height", ALTO);
  svg.setAttribute("viewBox", `0 0 ${ANCHO} ${ALTO}`);
  const el = (tag, at, txt) => {
    const e = document.createElementNS(NS, tag);
    for (const k in at) e.setAttribute(k, at[k]);
    if (txt != null) e.textContent = txt;
    svg.appendChild(e);
    return e;
  };
  const SANS = "Segoe UI, Arial, sans-serif";
  const MONO = "Consolas, Courier New, monospace";
  const texto = (x, y, t, at = {}) =>
    el("text", { x, y, fill: "#ecebf5", "font-size": 18, "font-family": SANS, ...at }, t);

  // Datos reales del análisis.
  const { intentos, trafico } = r.entrada;
  const [x1, x2] = r.normalizada;
  const capasOcultas = r.recorrido; // [oculta 1, oculta 2, salida]
  const zSalida = capasOcultas[2].z[0];
  const p = r.probabilidad;
  const u = r.umbral;
  const sospechosa = r.clase === 1;
  const colorClase = sospechosa ? ROJO : CIAN;
  const pct = (v, d = 2) => `${(v * 100).toFixed(d)} %`;
  const f2 = (v) => v.toFixed(2);

  // Fondo y título.
  el("rect", { x: 0, y: 0, width: ANCHO, height: ALTO, fill: "#0b0a12" });
  texto(800, 50, `Análisis de una conexión: ${intentos} intentos fallidos y ${trafico} MB`, {
    "text-anchor": "middle", "font-size": 32, "font-weight": 700,
  });
  texto(800, 84, "Recorrido por la red neuronal 2-8-4-1 · las 3 capas son OCULTA 1 + OCULTA 2 + SALIDA (la entrada no cuenta como capa)", {
    "text-anchor": "middle", "font-size": 17, fill: "#b4b1c8",
  });

  // Marco del "cerebro" (las 3 capas).
  el("rect", { x: 500, y: 128, width: 950, height: 700, rx: 28, fill: "rgba(167,139,250,.06)", stroke: VIOLETA, "stroke-width": 3, "stroke-dasharray": "10 8" });
  el("rect", { x: 700, y: 110, width: 550, height: 40, rx: 20, fill: "#1d1730", stroke: VIOLETA, "stroke-width": 2 });
  texto(975, 137, `EL CEREBRO = las 3 capas (${modelo.parametros_entrenables} pesos y sesgos)`, {
    "text-anchor": "middle", "font-size": 20, "font-weight": 700, fill: "#ddd5ff",
  });

  // Columnas de neuronas con las activaciones reales.
  const cy = 490;
  const capas = [
    { x: 290, sep: 170, color: AZUL, act: [1, 1], valores: [x1, x2] },
    { x: 690, sep: 56, color: VERDE, act: capasOcultas[0].a },
    { x: 1020, sep: 100, color: VERDE, act: capasOcultas[1].a },
    { x: 1330, sep: 0, color: ROJO, act: [1], valores: [p] },
  ];
  capas.forEach((c) => (c.ys = c.act.map((_, i) => cy + (i - (c.act.length - 1) / 2) * c.sep)));

  // Cajas de cada capa.
  const activas = (a) => a.filter((v) => v > 0).length;
  const cajas = [
    [capas[0], 230, "ENTRADA", "2 valores · no es capa", null],
    [capas[1], 190, "CAPA 1 · OCULTA", `8 ReLU · ${activas(capas[1].act)} activas`, 1],
    [capas[2], 190, "CAPA 2 · OCULTA", `4 ReLU · ${activas(capas[2].act)} activas`, 2],
    [capas[3], 190, "CAPA 3 · SALIDA", "1 neurona · sigmoide", 3],
  ];
  for (const [c, w, titulo, sub, num] of cajas) {
    el("rect", { x: c.x - w / 2, y: 185, width: w, height: 580, rx: 20, fill: "rgba(255,255,255,.025)", stroke: c.color, "stroke-width": 2, "stroke-opacity": 0.7 });
    texto(c.x, 222, titulo, { "text-anchor": "middle", "font-size": 19, "font-weight": 700, fill: c.color });
    texto(c.x, 247, sub, { "text-anchor": "middle", "font-size": 15, fill: "#b4b1c8" });
    if (num) {
      el("circle", { cx: c.x + w / 2 - 8, cy: 185, r: 24, fill: c.color });
      texto(c.x + w / 2 - 8, 194, String(num), { "text-anchor": "middle", "font-size": 26, "font-weight": 800, fill: "#0b0a12" });
    }
  }

  // Conexiones: resaltadas si la neurona de origen está encendida.
  for (let l = 0; l < capas.length - 1; l++) {
    const a = capas[l], b = capas[l + 1];
    a.ys.forEach((ya, i) => b.ys.forEach((yb) => {
      const on = a.act[i] > 0;
      el("line", { x1: a.x, y1: ya, x2: b.x, y2: yb, stroke: on ? "#9d8cf0" : "#3a3550", "stroke-width": on ? 1.6 : 1, "stroke-opacity": on ? 0.55 : 0.35 });
    }));
  }

  // Neuronas y su valor.
  capas.forEach((c, l) => c.ys.forEach((y, i) => {
    const on = c.act[i] > 0;
    const salida = l === 3;
    const relleno = salida ? colorClase : on ? c.color : "#2b2740";
    if (on) el("circle", { cx: c.x, cy: y, r: salida ? 38 : 26, fill: relleno, opacity: 0.18 });
    el("circle", { cx: c.x, cy: y, r: salida ? 26 : 19, fill: relleno, stroke: on ? "#fff" : "#4a4466", "stroke-width": 2 });
    if (l === 0) {
      texto(c.x - 34, y - 4, i === 0 ? `x₁ = ${intentos}` : `x₂ = ${trafico}`, { "text-anchor": "end", "font-size": 17, fill: "#dfe8ff" });
      texto(c.x - 34, y + 18, `norm. ${f2(c.valores[i])}`, { "text-anchor": "end", "font-size": 13, fill: "#8f8aa8", "font-family": MONO });
    } else if (!salida) {
      texto(c.x + 27, y + 5, on ? f2(c.act[i]) : "0", { "font-size": 13, fill: on ? "#d6f5e3" : "#6b6585", "font-family": MONO });
    }
  }));

  // Salida -> probabilidad -> clasificación.
  el("line", { x1: 1362, y1: cy, x2: 1462, y2: cy, stroke: "#fff", "stroke-width": 2.5 });
  el("path", { d: `M1462,${cy - 7} L1474,${cy} L1462,${cy + 7} z`, fill: "#fff" });
  texto(1530, cy - 14, `p = ${pct(p)}`, { "text-anchor": "middle", "font-size": 19, "font-weight": 700, "font-family": MONO });
  el("rect", { x: 1470, y: cy + 2, width: 122, height: 34, rx: 17, fill: sospechosa ? "rgba(232,69,95,.18)" : "rgba(14,165,196,.18)", stroke: colorClase });
  texto(1531, cy + 25, sospechosa ? "SOSPECHOSA" : "NORMAL", { "text-anchor": "middle", "font-size": 15, "font-weight": 700, fill: sospechosa ? "#ff8da0" : "#7fdcef" });
  texto(1531, cy + 62, `p ${sospechosa ? "≥" : "<"} ${f2(u)}`, { "text-anchor": "middle", "font-size": 15, fill: "#b4b1c8", "font-family": MONO });
  texto(1531, cy + 84, "(umbral)", { "text-anchor": "middle", "font-size": 13, fill: "#77748e" });

  texto(975, 805, "Neuronas grises = apagadas (ReLU dio 0) · las de color se activaron · junto a cada neurona: su activación a", {
    "text-anchor": "middle", "font-size": 15, fill: "#77748e",
  });

  // Tarjetas con la operación de cada capa y sus valores reales.
  const media = modelo.normalizador.media, desv = modelo.normalizador.desviacion;
  const tarjetas = [
    [290, AZUL, "Normalización", `x₁ = (${intentos} − ${media[0].toFixed(2)}) / ${desv[0].toFixed(2)} = ${f2(x1)}`, `x₂ = (${trafico} − ${media[1].toFixed(0)}) / ${desv[1].toFixed(0)} = ${f2(x2)}`, "keras.Input(shape=(2,))"],
    [690, VERDE, "z = Σ(w·x) + b", "a = ReLU(z) = max(0, z)", `${activas(capas[1].act)} de 8 neuronas activas`, "Dense(8, \"relu\") · 24 parámetros"],
    [1020, VERDE, "z = Σ(w·h) + b", "a = ReLU(z) = max(0, z)", `${activas(capas[2].act)} de 4 neuronas activas`, "Dense(4, \"relu\") · 36 parámetros"],
    [1330, ROJO, `z = ${f2(zSalida)}`, `p = 1/(1 + e^(${f2(-zSalida)}))`, `p = ${pct(p)}`, "Dense(1, \"sigmoid\") · 5 parámetros"],
  ];
  for (const [x, color, l1, l2, l3, l4] of tarjetas) {
    el("rect", { x: x - 145, y: 850, width: 290, height: 135, rx: 16, fill: "#14111f", stroke: color, "stroke-width": 2 });
    texto(x, 880, l1, { "text-anchor": "middle", "font-size": 18, "font-weight": 700, "font-family": MONO });
    texto(x, 907, l2, { "text-anchor": "middle", "font-size": 14, fill: "#c9c4e6", "font-family": MONO });
    texto(x, 933, l3, { "text-anchor": "middle", "font-size": 15, fill: "#ecebf5", "font-family": MONO });
    texto(x, 962, l4, { "text-anchor": "middle", "font-size": 14, fill: color, "font-family": MONO });
  }

  // Explicación en texto (la misma que muestra la interfaz).
  el("rect", { x: 40, y: 1008, width: 1520, height: 150, rx: 16, fill: sospechosa ? "rgba(232,69,95,.07)" : "rgba(14,165,196,.07)", stroke: colorClase });
  texto(62, 1040, `EXPLICACIÓN · clasificación: ${r.clasificacion}`, { "font-size": 19, "font-weight": 700, fill: sospechosa ? "#ff8da0" : "#7fdcef" });
  partirTexto(r.explicacion, 150).slice(0, 4).forEach((linea, k) =>
    texto(62, 1072 + k * 24, linea, { "font-size": 16, fill: "#d6d2ea" })
  );

  // Pie: aviso académico y fecha.
  const fecha = new Date().toLocaleString("es-CO");
  texto(40, 1200, "NeuroGuard 3D · Inteligencia Artificial · UNIPAZ", { "font-size": 15, fill: "#8f8aa8" });
  texto(40, 1226, "Demostración académica con datos simulados: el resultado no confirma ni descarta un ataque informático real.", { "font-size": 14, fill: "#77748e" });
  texto(1560, 1200, `Umbral usado: ${f2(u)} · ${fecha}`, { "text-anchor": "end", "font-size": 15, fill: "#8f8aa8" });
  return svg;
}

/** Convierte el SVG a PNG (a doble resolución) y lo descarga. */
export async function descargarImagenAnalisis(r, modelo) {
  const svg = construirSVG(r, modelo);
  const datos = new XMLSerializer().serializeToString(svg);
  const url = URL.createObjectURL(new Blob([datos], { type: "image/svg+xml;charset=utf-8" }));
  try {
    const img = new Image();
    await new Promise((ok, falla) => {
      img.onload = ok;
      img.onerror = falla;
      img.src = url;
    });
    const canvas = document.createElement("canvas");
    canvas.width = ANCHO * 2;
    canvas.height = ALTO * 2;
    const ctx = canvas.getContext("2d");
    ctx.scale(2, 2);
    ctx.drawImage(img, 0, 0);
    const png = await new Promise((ok) => canvas.toBlob(ok, "image/png"));
    const enlace = document.createElement("a");
    enlace.href = URL.createObjectURL(png);
    enlace.download = `analisis_${r.entrada.intentos}intentos_${r.entrada.trafico}MB_umbral${r.umbral.toFixed(2)}.png`;
    document.body.appendChild(enlace);
    enlace.click();
    enlace.remove();
    setTimeout(() => URL.revokeObjectURL(enlace.href), 4000);
  } finally {
    URL.revokeObjectURL(url);
  }
}
