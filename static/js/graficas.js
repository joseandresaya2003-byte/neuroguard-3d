/* =============================================================
   graficas.js - Gráficas 2D dibujadas con <canvas> (sin librerías)
   -------------------------------------------------------------
   1. MapaDecision: mapa de calor con la probabilidad que da la red para
      cada combinación (intentos, tráfico).
        - Frontera de decisión p = umbral (línea blanca; el umbral se
          puede mover desde la interfaz) y curvas de nivel p = 0.25 y
          p = 0.75 (líneas punteadas).
        - Datos de prueba: círculo = real normal, triángulo = real
          sospechosa, X amarilla = la red se equivocó.
        - Animado: cuando llega un mapa nuevo se transforma suavemente;
          el marcador de la conexión actual se desliza y pulsa; una "lupa"
          sigue al ratón mostrando la probabilidad; un clic crea una onda.
   2. GraficaLineas: curvas de pérdida / exactitud por época, con línea
      guía y tooltip. Se puede actualizar en vivo (laboratorio).
   ============================================================= */

import { umbral, esSospechosa } from "./umbral.js";

// Colores de datos (validados para daltonismo sobre el fondo oscuro).
const C_NORMAL = [14, 165, 196]; // #0ea5c4
const C_NEUTRO = [62, 58, 82]; // gris violáceo en p = umbral (punto medio divergente)
const C_SOSPECHOSA = [232, 69, 95]; // #e8455f
const TEXTO_SUAVE = "#77748e";
const REJILLA = "rgba(160,150,200,.14)";
const SUPERFICIE = "#110f1a";

/** Prepara un canvas nítido en pantallas de alta densidad (dibuja a 2x). */
function prepararCanvas(canvas) {
  const ancho = Number(canvas.getAttribute("width"));
  const alto = Number(canvas.getAttribute("height"));
  canvas.dataset.w = ancho;
  canvas.dataset.h = alto;
  canvas.width = ancho * 2;
  canvas.height = alto * 2;
  const ctx = canvas.getContext("2d");
  ctx.scale(2, 2);
  return { ctx, W: ancho, H: alto };
}

/** Convierte la posición del ratón a coordenadas lógicas del canvas. */
function posicionRaton(canvas, e) {
  const r = canvas.getBoundingClientRect();
  return {
    x: ((e.clientX - r.left) / r.width) * canvas.dataset.w,
    y: ((e.clientY - r.top) / r.height) * canvas.dataset.h,
    px: e.clientX - r.left,
    py: e.clientY - r.top,
  };
}

/**
 * Escala divergente: cian (normal) -> gris (umbral) -> rosa (sospechosa).
 * El gris queda justo en el umbral, así el color dice la clase asignada.
 */
export function colorProbabilidad(p) {
  const u = umbral.valor;
  const [a, b, t] = p < u ? [C_NORMAL, C_NEUTRO, p / u] : [C_NEUTRO, C_SOSPECHOSA, (p - u) / (1 - u)];
  return a.map((v, i) => Math.round(v + (b[i] - v) * t));
}

/* =============================================================
   1. MAPA DE DECISIÓN
   ============================================================= */
export class MapaDecision {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {HTMLElement} tooltip
   * @param {Function|null} alElegir  (intentos, tráfico) al hacer clic/arrastrar. null = solo lectura.
   */
  constructor(canvas, tooltip, alElegir = null) {
    this.canvas = canvas;
    this.tooltip = tooltip;
    this.alElegir = alElegir;
    ({ ctx: this.ctx, W: this.W, H: this.H } = prepararCanvas(canvas));
    this.m = { izq: 62, der: 16, arr: 14, aba: 44 }; // márgenes
    this.ancho = this.W - this.m.izq - this.m.der;
    this.alto = this.H - this.m.arr - this.m.aba;
    this.puntos = [];
    this.verPuntos = true;
    this.verNiveles = true;
    this.actual = null; // conexión analizada (destino del marcador)
    this.marcador = null; // posición dibujada del marcador (se desliza)
    this.raton = null; // posición del ratón para la "lupa"
    this.ondas = []; // animaciones de clic
    this.P = null; // probabilidades mostradas (Float32Array n*n)
    this.Pdestino = null;

    // Capa de fondo (mapa + ejes + curvas + puntos) que solo se redibuja si cambia.
    this.fondo = document.createElement("canvas");
    this.fondo.width = this.W * 2;
    this.fondo.height = this.H * 2;
    this.fctx = this.fondo.getContext("2d");
    this.fctx.scale(2, 2);
    this.fondoSucio = true;

    this._eventos();
    this._bucle = this._bucle.bind(this);
    requestAnimationFrame(this._bucle);
  }

  // Conversión entre valores reales y pixeles.
  xDe(intentos) { return this.m.izq + (intentos / 20) * this.ancho; }
  yDe(trafico) { return this.m.arr + this.alto - (trafico / 2000) * this.alto; }
  intentosDe(x) { return ((x - this.m.izq) / this.ancho) * 20; }
  traficoDe(y) { return ((this.m.arr + this.alto - y) / this.alto) * 2000; }

  /** Recibe una cuadrícula de probabilidades calculada por la red en el servidor. */
  setMapa(mapa) {
    const nuevo = Float32Array.from(mapa.probabilidades.flat());
    if (!this.P || mapa.n !== this.n) {
      this.n = mapa.n;
      this.P = Float32Array.from(nuevo);
    }
    this.Pdestino = nuevo;
    this.fondoSucio = true;
  }

  setPuntos(puntos) { this.puntos = puntos; this.fondoSucio = true; }
  /** Vuelve a dibujar todo (por ejemplo, cuando cambia el umbral). */
  refrescar() { this.fondoSucio = true; }
  setVerPuntos(v) { this.verPuntos = v; this.fondoSucio = true; }
  setVerNiveles(v) { this.verNiveles = v; this.fondoSucio = true; }
  setActual(intentos, trafico, p) {
    this.actual = { intentos: Math.min(intentos, 20), trafico: Math.min(trafico, 2000), p };
    if (!this.marcador) this.marcador = { ...this.actual };
  }

  /** Probabilidad en cualquier punto (interpolación bilineal de la cuadrícula). */
  probabilidadEn(intentos, trafico) {
    const n = this.n, P = this.P;
    const fx = Math.min(Math.max((intentos / 20) * (n - 1), 0), n - 1);
    const fy = Math.min(Math.max((trafico / 2000) * (n - 1), 0), n - 1);
    const x0 = Math.floor(fx), y0 = Math.floor(fy);
    const x1 = Math.min(x0 + 1, n - 1), y1 = Math.min(y0 + 1, n - 1);
    const tx = fx - x0, ty = fy - y0;
    const v = (j, i) => P[j * n + i];
    return (v(y0, x0) * (1 - tx) + v(y0, x1) * tx) * (1 - ty) + (v(y1, x0) * (1 - tx) + v(y1, x1) * tx) * ty;
  }

  /** Marching squares: segmentos donde la probabilidad cruza "nivel". */
  _curvaNivel(nivel) {
    const n = this.n, P = this.P;
    const seg = [];
    const pos = (i, j) => [(i / (n - 1)) * 20, (j / (n - 1)) * 2000];
    for (let j = 0; j < n - 1; j++) {
      for (let i = 0; i < n - 1; i++) {
        const esq = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
        const pts = [];
        for (let k = 0; k < 4; k++) {
          const [ia, ja] = esq[k], [ib, jb] = esq[(k + 1) % 4];
          const va = P[ja * n + ia], vb = P[jb * n + ib];
          if ((va >= nivel) !== (vb >= nivel)) {
            const t = (nivel - va) / (vb - va);
            const [xa, ya] = pos(ia, ja), [xb, yb] = pos(ib, jb);
            pts.push([xa + (xb - xa) * t, ya + (yb - ya) * t]);
          }
        }
        if (pts.length >= 2) seg.push([pts[0], pts[1]]);
        if (pts.length === 4) seg.push([pts[2], pts[3]]);
      }
    }
    return seg;
  }

  /** Dibuja la capa de fondo: mapa de calor, rejilla, curvas y datos de prueba. */
  _dibujarFondo() {
    const ctx = this.fctx, { W, H, m } = this;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = SUPERFICIE;
    ctx.fillRect(0, 0, W, H);
    if (!this.P) return;

    // Mapa de calor: imagen pequeña n x n ampliada con suavizado.
    const n = this.n;
    const img = document.createElement("canvas");
    img.width = img.height = n;
    const g = img.getContext("2d");
    const datos = g.createImageData(n, n);
    for (let fila = 0; fila < n; fila++) {
      for (let col = 0; col < n; col++) {
        const p = this.P[(n - 1 - fila) * n + col]; // fila 0 de la imagen = tráfico más alto
        const [r, gg, b] = colorProbabilidad(p);
        datos.data.set([r, gg, b, 255], (fila * n + col) * 4);
      }
    }
    g.putImageData(datos, 0, 0);
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(m.izq, m.arr, this.ancho, this.alto, 8);
    ctx.clip();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, m.izq, m.arr, this.ancho, this.alto);

    // Rejilla interna.
    ctx.strokeStyle = "rgba(255,255,255,.07)";
    ctx.lineWidth = 1;
    for (let v = 0; v <= 20; v += 2) {
      ctx.beginPath(); ctx.moveTo(this.xDe(v), m.arr); ctx.lineTo(this.xDe(v), m.arr + this.alto); ctx.stroke();
    }
    for (let v = 0; v <= 2000; v += 250) {
      ctx.beginPath(); ctx.moveTo(m.izq, this.yDe(v)); ctx.lineTo(m.izq + this.ancho, this.yDe(v)); ctx.stroke();
    }

    // Curvas de nivel p = 0.25 y 0.75 (punteadas) y frontera p = umbral (gruesa).
    const trazar = (segmentos) => {
      ctx.beginPath();
      for (const [[x1, y1], [x2, y2]] of segmentos) {
        ctx.moveTo(this.xDe(x1), this.yDe(y1));
        ctx.lineTo(this.xDe(x2), this.yDe(y2));
      }
      ctx.stroke();
    };
    if (this.verNiveles) {
      ctx.strokeStyle = "rgba(255,255,255,.45)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);
      trazar(this._curvaNivel(0.25));
      trazar(this._curvaNivel(0.75));
    }
    ctx.setLineDash([]);
    ctx.strokeStyle = "rgba(0,0,0,.35)";
    ctx.lineWidth = 5;
    const frontera = this._curvaNivel(umbral.valor);
    trazar(frontera);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2.2;
    trazar(frontera);

    // Datos de prueba.
    if (this.verPuntos) {
      for (const p of this.puntos) {
        const x = this.xDe(p.intentos), y = this.yDe(p.trafico);
        ctx.fillStyle = p.real ? "#e8455f" : "#0ea5c4";
        ctx.strokeStyle = SUPERFICIE;
        ctx.lineWidth = 2; // anillo de superficie para separar marcas superpuestas
        ctx.beginPath();
        if (p.real) { ctx.moveTo(x, y - 5); ctx.lineTo(x + 5, y + 4); ctx.lineTo(x - 5, y + 4); ctx.closePath(); }
        else ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.stroke(); ctx.fill();
        if (p.real !== this._predicho(p)) {
          ctx.strokeStyle = "#ffd34d"; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5);
          ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5); ctx.stroke();
        }
      }
    }
    ctx.restore();

    // Ejes y títulos.
    ctx.font = "11px Consolas, monospace";
    ctx.fillStyle = TEXTO_SUAVE;
    ctx.textAlign = "center";
    for (let v = 0; v <= 20; v += 2) ctx.fillText(v, this.xDe(v), m.arr + this.alto + 16);
    ctx.textAlign = "right";
    for (let v = 0; v <= 2000; v += 250) ctx.fillText(v, m.izq - 8, this.yDe(v) + 4);
    ctx.textAlign = "center";
    ctx.fillStyle = "#b4b1c8";
    ctx.fillText("x₁ · intentos fallidos", m.izq + this.ancho / 2, H - 8);
    ctx.save();
    ctx.translate(14, m.arr + this.alto / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("x₂ · tráfico enviado (MB)", 0, 0);
    ctx.restore();
  }

  /** Clase que la red asigna a un dato de prueba con el umbral vigente. */
  _predicho(p) {
    return p.probabilidad != null ? Number(esSospechosa(p.probabilidad)) : p.predicho;
  }

  /** Bucle de animación: transforma el mapa, desliza el marcador, ondas y lupa. */
  _bucle(ahora) {
    requestAnimationFrame(this._bucle);
    if (!this.canvas.offsetParent) return; // La pestaña está oculta: no se dibuja.

    // 1) Transición suave del mapa hacia el más reciente.
    if (this.Pdestino) {
      let cambio = false;
      for (let k = 0; k < this.P.length; k++) {
        const d = this.Pdestino[k] - this.P[k];
        if (Math.abs(d) > 5e-4) { this.P[k] += d * 0.18; cambio = true; }
        else this.P[k] = this.Pdestino[k];
      }
      if (cambio) this.fondoSucio = true;
    }
    if (this.fondoSucio) { this._dibujarFondo(); this.fondoSucio = false; }

    const { ctx, W, H, m } = this;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(this.fondo, 0, 0, W, H);
    if (!this.P) return;

    // 2) Lupa que sigue al ratón.
    if (this.raton) {
      const { x, y } = this.raton;
      const p = this.probabilidadEn(this.intentosDe(x), this.traficoDe(y));
      ctx.strokeStyle = "rgba(255,255,255,.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(m.izq, y); ctx.lineTo(m.izq + this.ancho, y);
      ctx.moveTo(x, m.arr); ctx.lineTo(x, m.arr + this.alto);
      ctx.stroke();
      ctx.setLineDash([]);
      const [r, g, b] = colorProbabilidad(p);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 13, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 9px Consolas, monospace";
      ctx.textAlign = "center";
      ctx.fillText(Math.round(p * 100), x, y + 3);
    }

    // 3) Ondas de los clics.
    this.ondas = this.ondas.filter((o) => ahora - o.inicio < 700);
    for (const o of this.ondas) {
      const t = (ahora - o.inicio) / 700;
      ctx.strokeStyle = `rgba(255,255,255,${1 - t})`;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(o.x, o.y, 6 + t * 34, 0, Math.PI * 2); ctx.stroke();
    }

    // 4) Marcador de la conexión actual: se desliza hasta su destino y pulsa.
    if (this.actual) {
      const mk = this.marcador;
      mk.intentos += (this.actual.intentos - mk.intentos) * 0.15;
      mk.trafico += (this.actual.trafico - mk.trafico) * 0.15;
      const x = this.xDe(mk.intentos), y = this.yDe(mk.trafico);
      const pulso = (ahora % 1600) / 1600;
      const color = esSospechosa(this.actual.p) ? "232,69,95" : "14,165,196";
      ctx.strokeStyle = `rgba(${color},${1 - pulso})`;
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(x, y, 9 + pulso * 22, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = SUPERFICIE;
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.stroke(); ctx.fill();
      ctx.fillStyle = `rgb(${color})`;
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
      if (this.actual.p != null) {
        const texto = `p = ${(this.actual.p * 100).toFixed(1)}%`;
        ctx.font = "bold 12px Consolas, monospace";
        const tw = ctx.measureText(texto).width + 14;
        const bx = Math.min(x + 14, m.izq + this.ancho - tw);
        const by = Math.max(y - 34, m.arr + 4);
        ctx.fillStyle = "rgba(10,9,16,.9)";
        ctx.beginPath(); ctx.roundRect(bx, by, tw, 22, 6); ctx.fill();
        ctx.fillStyle = "#ffffff";
        ctx.textAlign = "left";
        ctx.fillText(texto, bx + 7, by + 15);
      }
    }
  }

  _eventos() {
    let arrastrando = false;
    let ultimo = 0;
    const dentro = (pos) =>
      pos.x >= this.m.izq && pos.x <= this.m.izq + this.ancho && pos.y >= this.m.arr && pos.y <= this.m.arr + this.alto;

    const elegir = (pos) => {
      const intentos = Math.round(Math.min(Math.max(this.intentosDe(pos.x), 0), 20));
      const trafico = Math.round(Math.min(Math.max(this.traficoDe(pos.y), 0), 2000) / 10) * 10;
      this.alElegir(intentos, trafico);
    };

    this.canvas.style.cursor = this.alElegir ? "crosshair" : "default";
    this.canvas.addEventListener("pointerdown", (e) => {
      const pos = posicionRaton(this.canvas, e);
      if (!this.alElegir || !dentro(pos) || !this.P) return;
      arrastrando = true;
      this.canvas.setPointerCapture(e.pointerId);
      this.ondas.push({ x: pos.x, y: pos.y, inicio: performance.now() });
      elegir(pos);
    });
    this.canvas.addEventListener("pointerup", () => (arrastrando = false));
    this.canvas.addEventListener("pointermove", (e) => {
      const pos = posicionRaton(this.canvas, e);
      if (!this.P || !dentro(pos)) { this.tooltip.hidden = true; this.raton = null; return; }
      this.raton = { x: pos.x, y: pos.y };
      // Mientras se arrastra, se analiza como máximo cada 120 ms.
      if (arrastrando && performance.now() - ultimo > 120) { ultimo = performance.now(); elegir(pos); }

      const intentos = this.intentosDe(pos.x), trafico = this.traficoDe(pos.y);
      let cercano = null; // ¿Hay un dato de prueba cerca del ratón?
      if (this.verPuntos) {
        let mejor = 7;
        for (const p of this.puntos) {
          const d = Math.hypot(this.xDe(p.intentos) - pos.x, this.yDe(p.trafico) - pos.y);
          if (d < mejor) { mejor = d; cercano = p; }
        }
      }
      if (cercano) {
        const real = cercano.real ? "sospechosa" : "normal";
        const predicho = this._predicho(cercano);
        const pred = predicho ? "sospechosa" : "normal";
        const prob = cercano.probabilidad != null ? ` (p = ${(cercano.probabilidad * 100).toFixed(1)}%)` : "";
        this.tooltip.innerHTML = `<b>Dato de prueba</b><br>${cercano.intentos} intentos · ${cercano.trafico} MB<br>real: ${real} · red: ${pred}${prob} ${cercano.real === predicho ? "✓" : "✗"}`;
      } else {
        const p = this.probabilidadEn(intentos, trafico);
        this.tooltip.innerHTML = `${intentos.toFixed(1)} intentos · ${trafico.toFixed(0)} MB<br>p(red) ≈ <b>${(p * 100).toFixed(1)}%</b> → ${esSospechosa(p) ? "sospechosa" : "normal"}`;
      }
      this.tooltip.hidden = false;
      const ancho = this.canvas.clientWidth;
      this.tooltip.style.left = `${pos.px > ancho * 0.65 ? pos.px - 210 : pos.px + 18}px`;
      this.tooltip.style.top = `${pos.py + 18}px`;
    });
    this.canvas.addEventListener("pointerleave", () => {
      this.tooltip.hidden = true;
      this.raton = null;
    });
  }
}

/* =============================================================
   2. GRÁFICA DE LÍNEAS (pérdida / exactitud por época)
   ============================================================= */
export class GraficaLineas {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {Array} series  [{nombre, valores:[...], color}]
   * @param {Object} opciones {formato: (v) => texto, epocas: total de épocas del eje X}
   */
  constructor(canvas, series, opciones = {}) {
    this.canvas = canvas;
    this.tooltip = canvas.parentElement.querySelector(".tooltip-grafica");
    this.formato = opciones.formato ?? ((v) => v.toFixed(3));
    ({ ctx: this.ctx, W: this.W, H: this.H } = prepararCanvas(canvas));
    this.m = { izq: 54, der: 14, arr: 12, aba: 34 };
    this.ancho = this.W - this.m.izq - this.m.der;
    this.alto = this.H - this.m.arr - this.m.aba;
    this.guia = null;
    this._eventos();
    this.actualizar(series, opciones.epocas);
  }

  /** Cambia los datos (por ejemplo, cada época del laboratorio) y redibuja. */
  actualizar(series, totalEpocas) {
    this.series = series;
    this.largo = series[0].valores.length;
    this.epocas = Math.max(totalEpocas ?? this.largo, 2);
    this.visibles = this.largo;
    const todos = series.flatMap((s) => s.valores);
    if (todos.length) {
      const min = Math.min(...todos), max = Math.max(...todos);
      const holgura = (max - min) * 0.1 || 0.05;
      this.yMin = Math.max(0, min - holgura);
      this.yMax = max + holgura;
    } else {
      this.yMin = 0;
      this.yMax = 1;
    }
    this.dibujar();
  }

  x(i) { return this.m.izq + (i / (this.epocas - 1)) * this.ancho; }
  y(v) { return this.m.arr + this.alto - ((v - this.yMin) / (this.yMax - this.yMin)) * this.alto; }

  dibujar() {
    const { ctx, W, H, m } = this;
    ctx.clearRect(0, 0, W, H);
    ctx.font = "11px Consolas, monospace";

    ctx.textAlign = "right";
    for (let k = 0; k <= 4; k++) {
      const v = this.yMin + ((this.yMax - this.yMin) * k) / 4;
      const yy = this.y(v);
      ctx.strokeStyle = REJILLA;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(m.izq, yy); ctx.lineTo(m.izq + this.ancho, yy); ctx.stroke();
      ctx.fillStyle = TEXTO_SUAVE;
      ctx.fillText(this.formato(v), m.izq - 6, yy + 4);
    }
    ctx.textAlign = "center";
    const paso = this.epocas > 150 ? 50 : this.epocas > 60 ? 20 : 10;
    for (let e = 0; e <= this.epocas; e += paso) {
      ctx.fillText(e === 0 ? 1 : e, this.x(Math.max(e - 1, 0)), H - 16);
    }
    ctx.fillText("época", m.izq + this.ancho / 2, H - 2);

    // Series: línea de 2 px con un leve relleno degradado bajo la primera.
    this.series.forEach((s, idx) => {
      const vals = s.valores.slice(0, this.visibles);
      if (vals.length < 2) return;
      ctx.beginPath();
      vals.forEach((v, i) => (i ? ctx.lineTo(this.x(i), this.y(v)) : ctx.moveTo(this.x(i), this.y(v))));
      if (idx === 0) {
        ctx.save();
        ctx.lineTo(this.x(vals.length - 1), m.arr + this.alto);
        ctx.lineTo(this.x(0), m.arr + this.alto);
        const grad = ctx.createLinearGradient(0, m.arr, 0, m.arr + this.alto);
        grad.addColorStop(0, "rgba(14,165,196,.18)");
        grad.addColorStop(1, "rgba(14,165,196,0)");
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.restore();
        ctx.beginPath();
        vals.forEach((v, i) => (i ? ctx.lineTo(this.x(i), this.y(v)) : ctx.moveTo(this.x(i), this.y(v))));
      }
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.stroke();
    });

    if (this.guia != null && this.guia < Math.min(this.visibles, this.largo)) {
      const xx = this.x(this.guia);
      ctx.strokeStyle = "rgba(236,235,245,.35)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(xx, m.arr); ctx.lineTo(xx, m.arr + this.alto); ctx.stroke();
      for (const s of this.series) {
        ctx.fillStyle = s.color;
        ctx.strokeStyle = SUPERFICIE;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(xx, this.y(s.valores[this.guia]), 4.5, 0, Math.PI * 2); ctx.stroke(); ctx.fill();
      }
    }
  }

  /** Reproduce el entrenamiento dibujando las curvas época por época. */
  reproducir(duracionMs = 2500) {
    const inicio = performance.now();
    const paso = (ahora) => {
      const t = Math.min((ahora - inicio) / duracionMs, 1);
      this.visibles = Math.max(2, Math.round(t * this.largo));
      this.dibujar();
      if (t < 1) requestAnimationFrame(paso);
    };
    requestAnimationFrame(paso);
  }

  _eventos() {
    this.canvas.addEventListener("pointermove", (e) => {
      const pos = posicionRaton(this.canvas, e);
      const i = Math.round(((pos.x - this.m.izq) / this.ancho) * (this.epocas - 1));
      if (i < 0 || i >= this.largo) { this.guia = null; this.tooltip.hidden = true; this.dibujar(); return; }
      this.guia = i;
      this.dibujar();
      this.tooltip.innerHTML =
        `<b>Época ${i + 1}</b><br>` +
        this.series.map((s) => `<span style="color:${s.color}">■</span> ${s.nombre}: ${this.formato(s.valores[i])}`).join("<br>");
      this.tooltip.hidden = false;
      this.tooltip.style.left = `${pos.px > this.canvas.clientWidth * 0.6 ? pos.px - 190 : pos.px + 14}px`;
      this.tooltip.style.top = `${Math.max(pos.py - 30, 0)}px`;
    });
    this.canvas.addEventListener("pointerleave", () => {
      this.guia = null;
      this.tooltip.hidden = true;
      this.dibujar();
    });
  }
}

/* =============================================================
   3. CURVA SIGMOIDE EN VIVO (neurona de salida)
   -------------------------------------------------------------
   La neurona de salida calcula un número z (puede ser cualquier valor)
   y la función sigmoide lo convierte en una probabilidad entre 0 y 1:
        p = σ(z) = 1 / (1 + e^(−z))
   Esta gráfica dibuja la curva en "S" y un punto que se desliza hasta el
   z de la conexión analizada. La línea horizontal es el umbral: si el
   punto queda por encima la conexión es sospechosa; si queda por debajo,
   normal. El umbral en p equivale a un corte en z:
        p ≥ umbral   <=>   z ≥ ln(umbral / (1 − umbral))
   ============================================================= */
export const sigmoide = (z) => 1 / (1 + Math.exp(-z));

export class CurvaSigmoide {
  constructor(canvas, tooltip) {
    this.canvas = canvas;
    this.tooltip = tooltip;
    ({ ctx: this.ctx, W: this.W, H: this.H } = prepararCanvas(canvas));
    this.m = { izq: 34, der: 12, arr: 12, aba: 34 }; // márgenes
    this.ancho = this.W - this.m.izq - this.m.der;
    this.alto = this.H - this.m.arr - this.m.aba;
    this.Z = 10; // El eje z va de −10 a 10.
    this.z = null; // z de la conexión analizada (destino)
    this.zDibujado = null; // z del punto dibujado (se desliza hacia this.z)
    this.raton = null;
    this._eventos();
    this._bucle = this._bucle.bind(this);
    requestAnimationFrame(this._bucle);
  }

  // Conversión entre valores y pixeles.
  xDe(z) { return this.m.izq + ((z + this.Z) / (2 * this.Z)) * this.ancho; }
  yDe(p) { return this.m.arr + (1 - p) * this.alto; }
  zDe(x) { return ((x - this.m.izq) / this.ancho) * 2 * this.Z - this.Z; }

  /** Recibe el z de la neurona de salida para la conexión analizada. */
  setZ(z) {
    this.z = z;
    if (this.zDibujado === null) this.zDibujado = z;
  }

  _eventos() {
    this.canvas.addEventListener("pointermove", (e) => {
      const r = this.canvas.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width) * this.W;
      if (x < this.m.izq || x > this.m.izq + this.ancho) { this.raton = null; this.tooltip.hidden = true; return; }
      this.raton = x;
      const z = this.zDe(x), p = sigmoide(z);
      this.tooltip.innerHTML = `z = ${z.toFixed(2)}<br>σ(z) = <b>${(p * 100).toFixed(1)}%</b> → ${esSospechosa(p) ? "sospechosa" : "normal"}`;
      this.tooltip.hidden = false;
      const px = e.clientX - r.left;
      this.tooltip.style.left = `${px > r.width * 0.55 ? px - 150 : px + 14}px`;
      this.tooltip.style.top = `${e.clientY - r.top + 14}px`;
    });
    this.canvas.addEventListener("pointerleave", () => { this.raton = null; this.tooltip.hidden = true; });
  }

  _bucle(ahora) {
    requestAnimationFrame(this._bucle);
    if (!this.canvas.offsetParent) return; // No visible: no se dibuja.
    const { ctx, W, H, m, Z } = this;
    const u = umbral.valor;
    const zUmbral = Math.log(u / (1 - u)); // Punto de la curva donde p = umbral.

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = SUPERFICIE;
    ctx.fillRect(0, 0, W, H);

    // Zonas: a la izquierda del corte = normal, a la derecha = sospechosa.
    const xu = this.xDe(Math.max(-Z, Math.min(Z, zUmbral)));
    ctx.fillStyle = "rgba(14,165,196,.08)";
    ctx.fillRect(m.izq, m.arr, xu - m.izq, this.alto);
    ctx.fillStyle = "rgba(232,69,95,.08)";
    ctx.fillRect(xu, m.arr, m.izq + this.ancho - xu, this.alto);

    // Rejilla y ejes.
    ctx.strokeStyle = REJILLA;
    ctx.lineWidth = 1;
    ctx.font = "10px Consolas, monospace";
    ctx.fillStyle = TEXTO_SUAVE;
    ctx.textAlign = "center";
    for (let z = -Z; z <= Z; z += 5) {
      ctx.beginPath(); ctx.moveTo(this.xDe(z), m.arr); ctx.lineTo(this.xDe(z), m.arr + this.alto); ctx.stroke();
      ctx.fillText(z, this.xDe(z), m.arr + this.alto + 13);
    }
    ctx.textAlign = "right";
    for (const p of [0, 0.5, 1]) {
      ctx.beginPath(); ctx.moveTo(m.izq, this.yDe(p)); ctx.lineTo(m.izq + this.ancho, this.yDe(p)); ctx.stroke();
      ctx.fillText(p, m.izq - 6, this.yDe(p) + 3);
    }
    ctx.textAlign = "center";
    ctx.fillText("z (entrada de la neurona de salida)", m.izq + this.ancho / 2, H - 4);

    // Umbral: línea horizontal en p = umbral y vertical en el z equivalente.
    ctx.strokeStyle = "rgba(196,181,253,.85)";
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(m.izq, this.yDe(u)); ctx.lineTo(m.izq + this.ancho, this.yDe(u));
    ctx.moveTo(xu, m.arr); ctx.lineTo(xu, m.arr + this.alto);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#c4b5fd";
    ctx.textAlign = "left";
    ctx.fillText(`umbral ${u.toFixed(2)}`, m.izq + 4, this.yDe(u) - 5);

    // Curva en S, coloreada según la probabilidad (cian -> gris -> rosa).
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    const pasos = 120;
    for (let k = 0; k < pasos; k++) {
      const z1 = -Z + (2 * Z * k) / pasos, z2 = -Z + (2 * Z * (k + 1)) / pasos;
      const [r, g, b] = colorProbabilidad(sigmoide((z1 + z2) / 2));
      ctx.strokeStyle = `rgb(${r},${g},${b})`;
      ctx.beginPath();
      ctx.moveTo(this.xDe(z1), this.yDe(sigmoide(z1)));
      ctx.lineTo(this.xDe(z2), this.yDe(sigmoide(z2)));
      ctx.stroke();
    }

    // Línea guía del ratón.
    if (this.raton !== null) {
      const z = this.zDe(this.raton);
      ctx.strokeStyle = "rgba(255,255,255,.3)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(this.raton, m.arr); ctx.lineTo(this.raton, m.arr + this.alto); ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.beginPath(); ctx.arc(this.raton, this.yDe(sigmoide(z)), 3.5, 0, Math.PI * 2); ctx.fill();
    }

    // Punto de la conexión analizada: se desliza por la curva hasta su z.
    if (this.z !== null) {
      this.zDibujado += (this.z - this.zDibujado) * 0.12;
      const zv = Math.max(-Z, Math.min(Z, this.zDibujado)); // fuera de rango: se queda en el borde
      const p = sigmoide(this.zDibujado);
      const x = this.xDe(zv), y = this.yDe(p);
      const color = esSospechosa(sigmoide(this.z)) ? "232,69,95" : "14,165,196";
      // Proyecciones punteadas a los ejes (z abajo, p a la izquierda).
      ctx.strokeStyle = "rgba(255,255,255,.45)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(x, y); ctx.lineTo(x, m.arr + this.alto);
      ctx.moveTo(x, y); ctx.lineTo(m.izq, y);
      ctx.stroke();
      ctx.setLineDash([]);
      const pulso = (ahora % 1600) / 1600;
      ctx.strokeStyle = `rgba(${color},${1 - pulso})`;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 6 + pulso * 14, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = SUPERFICIE;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.stroke(); ctx.fill();
      ctx.fillStyle = `rgb(${color})`;
      ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
      // Etiqueta con los valores.
      const texto = `z=${this.z.toFixed(2)} → p=${(sigmoide(this.z) * 100).toFixed(1)}%`;
      ctx.font = "bold 11px Consolas, monospace";
      const tw = ctx.measureText(texto).width + 12;
      const bx = Math.min(Math.max(x - tw / 2, m.izq), m.izq + this.ancho - tw);
      const by = p > 0.6 ? y + 12 : y - 30;
      ctx.fillStyle = "rgba(10,9,16,.9)";
      ctx.beginPath(); ctx.roundRect(bx, by, tw, 19, 5); ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.textAlign = "left";
      ctx.fillText(texto, bx + 6, by + 13);
    }
  }
}
