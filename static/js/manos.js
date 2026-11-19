/* =============================================================
   manos.js — ✋ Controlar la red 3D con las MANOS (cámara web)
   -------------------------------------------------------------
   Usa MediaPipe Hand Landmarker (Google), guardado en
   static/vendor/mediapipe, así que funciona SIN internet.
   La cámara se procesa en el propio navegador: no se envía
   ninguna imagen al servidor.

   MediaPipe entrega 21 puntos por mano. Usamos estos:
       0 = muñeca            8 = punta del índice
       5, 9, 13, 17 = base de los dedos (centro de la palma)
       6, 10, 14, 18 = nudillos medios   12, 16, 20 = puntas de los otros dedos

   GESTOS (modo fácil: sin pellizcos ni dobles gestos)
     ✋  Mano ABIERTA y moverla ...... gira la red (la red sigue a la mano)
     ☝️  APUNTAR con el índice ....... mueve el cursor
     ☝️  Quieto 1 s sobre neurona .... la selecciona (ve su cálculo)
     ☝️  … y 1 s más ................. ENTRA a la neurona
     ☝️  Quieto 1 s sobre un botón ... lo pulsa (Regresar, Centrar, Salir)
     ✋✋ Dos manos abiertas .......... separarlas = acercar · juntarlas = alejar
     ✊  Puño cerrado 1 s ............ sale de la neurona / centra la vista
   ============================================================= */

const RUTA = "/static/vendor/mediapipe";

const TIEMPO_APUNTAR = 900; // ms quieto apuntando para seleccionar / pulsar
const TIEMPO_PUNO = 900; // ms con el puño cerrado para salir
const RADIO_QUIETO = 45; // px que puede temblar el dedo sin reiniciar la cuenta
const CUADROS_CAMBIO = 3; // cuadros seguidos para aceptar un gesto nuevo (evita parpadeos)

const distancia = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// Texto que se muestra en la cámara según el gesto reconocido.
const TEXTO_GESTO = {
  abierta: "✋ Mano abierta: moviéndola gira la red",
  apunta: "☝️ Apuntando: quédese quieto 1 s",
  puno: "✊ Puño: mantenga 1 s para salir",
  otra: "Abra la mano ✋ o apunte con el índice ☝️",
};

/** Reconoce la postura de la mano: "abierta", "apunta", "puno" u "otra". */
function postura(p) {
  // Un dedo está estirado si su punta queda más lejos de la muñeca que su nudillo.
  const estirado = ([punta, nudillo]) => distancia(p[punta], p[0]) > distancia(p[nudillo], p[0]) * 1.1;
  const [indice, medio, anular, menique] = [[8, 6], [12, 10], [16, 14], [20, 18]].map(estirado);
  const cuantos = [indice, medio, anular, menique].filter(Boolean).length;
  if (cuantos >= 3) return "abierta";
  if (indice && !anular && !menique) return "apunta"; // índice (el medio puede estar a medias)
  if (cuantos === 0) return "puno";
  return "otra";
}

export class ControlManos {
  /**
   * @param {Red3D} red          La red 3D (red3d.js) que se va a mover.
   * @param {HTMLElement} panel  El contenedor #escena3d.
   * @param {Function} avisar    Para mostrar mensajes (toast) en la página.
   */
  constructor(red, panel, avisar) {
    this.red = red;
    this.panel = panel;
    this.avisar = avisar;
    this.activo = false;
    this.manos = []; // estado de cada mano entre un cuadro y el siguiente
    this.espera = null; // cuenta del "quedarse quieto" apuntando
    this.zoomPrevio = null;
  }

  /* ---------- Encender / apagar ---------- */

  async iniciar() {
    // 1) Cargar MediaPipe (solo la primera vez; pesa ~22 MB y queda en caché).
    if (!this.detector) {
      const { FilesetResolver, HandLandmarker } = await import(`${RUTA}/vision_bundle.mjs`);
      const archivos = await FilesetResolver.forVisionTasks(`${RUTA}/wasm`);
      const opciones = (delegate) => ({
        baseOptions: { modelAssetPath: `${RUTA}/hand_landmarker.task`, delegate },
        runningMode: "VIDEO",
        numHands: 2,
      });
      try {
        this.detector = await HandLandmarker.createFromOptions(archivos, opciones("GPU"));
      } catch {
        this.detector = await HandLandmarker.createFromOptions(archivos, opciones("CPU"));
      }
    }

    // 2) Pedir permiso para usar la cámara web.
    this.flujo = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, frameRate: { ideal: 60 }, facingMode: "user" },
      audio: false,
    });

    // 3) Vista previa pequeña de la cámara con los puntos de la mano dibujados.
    this.caja = document.createElement("div");
    this.caja.className = "camara-manos";
    this.caja.innerHTML = `<video playsinline muted></video><canvas width="320" height="240"></canvas>
      <div class="estado-manos">Buscando manos…</div>`;
    this.video = this.caja.querySelector("video");
    this.lienzo = this.caja.querySelector("canvas").getContext("2d");
    this.textoEstado = this.caja.querySelector(".estado-manos");

    // Guía de gestos (la fila del gesto actual se ilumina).
    this.ayuda = document.createElement("div");
    this.ayuda.className = "ayuda-manos";
    this.ayuda.innerHTML = `<b>Cómo se usa</b>
      <div data-g="abierta"><span>✋</span><p>Mano abierta y moverla = <b>girar</b></p></div>
      <div data-g="apunta"><span>☝️</span><p>Apuntar con el índice = <b>cursor</b></p></div>
      <div data-g="apunta"><span>⏱️</span><p>Quieto 1 s = <b>ver</b> · 1 s más = <b>entrar</b></p></div>
      <div data-g="apunta"><span>🔘</span><p>Quieto 1 s en un botón = <b>pulsarlo</b></p></div>
      <div data-g="zoom"><span>✋✋</span><p>Dos manos: separar / juntar = <b>zoom</b></p></div>
      <div data-g="puno"><span>✊</span><p>Puño 1 s = <b>salir</b> / centrar</p></div>`;

    // Un cursor por mano (con un anillo de progreso y un texto de acción).
    this.cursores = [0, 1].map(() => {
      const c = document.createElement("div");
      c.className = "cursor-mano";
      c.innerHTML = '<div class="progreso"></div><span class="icono"></span><span class="accion"></span>';
      c.hidden = true;
      this.panel.appendChild(c);
      return c;
    });
    this.panel.append(this.caja, this.ayuda);

    this.video.srcObject = this.flujo;
    await this.video.play();
    this.activo = true;
    this.red.pausarAutoRotar(true);
    this._bucle();
    this._pintarCursores();
  }

  detener() {
    this.activo = false;
    this.flujo?.getTracks().forEach((pista) => pista.stop()); // apaga la cámara
    this.caja?.remove();
    this.ayuda?.remove();
    this.cursores?.forEach((c) => c.remove());
    this.red.mostrarTooltip(null);
    this.red.pausarAutoRotar(false);
    this.manos = [];
    this.espera = null;
  }

  /* ---------- Bucle: un cuadro de la cámara = una detección ---------- */

  _bucle() {
    if (!this.activo) return;
    if (this.video.readyState >= 2) {
      const resultado = this.detector.detectForVideo(this.video, performance.now());
      this._dibujarPuntos(resultado.landmarks);
      this._interpretar(resultado.landmarks);
    }
    // Solo se analiza cuando la cámara entrega una imagen NUEVA (no se repite
    // trabajo), dejando libre el procesador para dibujar la red a 60 cuadros/s.
    if (this.video.requestVideoFrameCallback) this.video.requestVideoFrameCallback(() => this._bucle());
    else requestAnimationFrame(() => this._bucle());
  }

  /**
   * Los cursores se dibujan a 60 cuadros/s deslizándose hacia la última
   * posición detectada: se ven fluidos aunque la cámara vaya a 30.
   */
  _pintarCursores() {
    if (!this.activo) return;
    const ahora = performance.now();
    const dt = Math.min(0.05, (ahora - (this.ultimoPintado ?? ahora)) / 1000);
    this.ultimoPintado = ahora;
    const k = 1 - Math.exp(-dt * 30);
    this.manos.forEach((m, i) => {
      const c = this.cursores[i];
      c.vx = c.vx == null ? m.x : c.vx + (m.x - c.vx) * k;
      c.vy = c.vy == null ? m.y : c.vy + (m.y - c.vy) * k;
      c.style.transform = `translate(${c.vx}px, ${c.vy}px)`;
    });
    requestAnimationFrame(() => this._pintarCursores());
  }

  /** Dibuja los 21 puntos de cada mano sobre la vista previa. */
  _dibujarPuntos(manos) {
    const g = this.lienzo;
    g.clearRect(0, 0, 320, 240);
    for (const puntos of manos) {
      g.fillStyle = "#a78bfa";
      for (const p of puntos) {
        g.beginPath();
        g.arc(p.x * 320, p.y * 240, 2.6, 0, Math.PI * 2);
        g.fill();
      }
      g.fillStyle = "#22d3ee"; // punta del índice resaltada (es la que apunta)
      g.beginPath();
      g.arc(puntos[8].x * 320, puntos[8].y * 240, 5, 0, Math.PI * 2);
      g.fill();
    }
  }

  /* ---------- Convertir los puntos de la mano en gestos ---------- */

  _interpretar(manosDetectadas) {
    const ancho = this.panel.clientWidth;
    const alto = this.panel.clientHeight;
    const ahora = performance.now();

    // 1) Postura y posición de cada mano.
    const actuales = manosDetectadas.slice(0, 2).map((p, i) => {
      const anterior = this.manos[i] || {};
      // El gesto solo cambia si se repite varios cuadros seguidos (sin parpadeos).
      const vista = postura(p);
      const racha = vista === anterior.vista ? (anterior.racha ?? 0) + 1 : 1;
      const gesto = !anterior.gesto || racha >= CUADROS_CAMBIO ? vista : anterior.gesto;

      // Apuntando: el cursor es la punta del índice. Si no: el centro de la palma.
      const ancla = gesto === "apunta" ? p[8] : {
        x: (p[0].x + p[5].x + p[9].x + p[13].x + p[17].x) / 5,
        y: (p[0].y + p[5].y + p[9].y + p[13].y + p[17].y) / 5,
      };
      // En espejo (como un espejo real) y ampliando el centro de la imagen para
      // llegar a los bordes de la pantalla sin estirar el brazo.
      const nx = Math.min(1, Math.max(0, (1 - ancla.x - 0.15) / 0.7));
      const ny = Math.min(1, Math.max(0, (ancla.y - 0.15) / 0.7));
      let x = nx * ancho, y = ny * alto;
      // Suavizado ADAPTATIVO: si la mano va rápido se sigue casi sin retraso;
      // si está quieta se suaviza más para que el cursor no tiemble.
      const mismaAncla = (anterior.gesto === "apunta") === (gesto === "apunta");
      if (anterior.x != null && mismaAncla) {
        const salto = Math.hypot(x - anterior.x, y - anterior.y);
        const alfa = Math.min(0.9, 0.25 + salto / 60);
        x = anterior.x + (x - anterior.x) * alfa;
        y = anterior.y + (y - anterior.y) * alfa;
      }
      return {
        x, y, vista, racha, gesto,
        previoX: mismaAncla ? anterior.x : null, // para medir cuánto se movió
        previoY: mismaAncla ? anterior.y : null,
        punoDesde: gesto === "puno" ? anterior.punoDesde ?? ahora : null,
      };
    });

    // 2) Dos manos abiertas: ZOOM (separar = acercar, juntar = alejar).
    const dosAbiertas = actuales.length === 2 && actuales.every((m) => m.gesto === "abierta");
    if (dosAbiertas) {
      const separacion = Math.hypot(actuales[0].x - actuales[1].x, actuales[0].y - actuales[1].y);
      if (this.zoomPrevio) this.red.zoomSuave((this.zoomPrevio / separacion) ** 1.8);
      this.zoomPrevio = separacion;
    } else {
      this.zoomPrevio = null;
    }

    let textoAccion = null;
    actuales.forEach((m, i) => {
      const cursor = this.cursores[i];
      cursor.hidden = false; // la posición la pinta _pintarCursores()
      cursor.dataset.gesto = dosAbiertas ? "zoom" : m.gesto;
      cursor.querySelector(".icono").textContent =
        dosAbiertas ? "↔" : { abierta: "✋", apunta: "", puno: "✊", otra: "" }[m.gesto];
      let progreso = 0;
      let accion = "";

      // ---- ✋ Mano abierta: GIRAR (la red sigue el movimiento de la mano) ----
      // (solo con UNA mano y si la postura es estable, para no girar al cambiar de gesto)
      if (m.gesto === "abierta" && m.vista === "abierta" && actuales.length === 1 && m.previoX != null) {
        const dx = (m.x - m.previoX) / ancho;
        const dy = (m.y - m.previoY) / alto;
        this.red.girarSuave(dx * Math.PI * 2.6, dy * Math.PI * 1.3);
      }

      // ---- ✊ Puño 1 segundo: SALIR / CENTRAR ----
      if (m.punoDesde) {
        progreso = Math.max(0, Math.min(1, (ahora - m.punoDesde) / TIEMPO_PUNO));
        accion = this.red.dentro ? "Saliendo…" : "Centrando…";
        if (progreso === 1) {
          this.red.centrar();
          m.punoDesde = ahora + 1e9; // no repetir hasta abrir la mano
        }
      }

      // ---- ☝️ Apuntar (solo la primera mano): tooltip + "quedarse quieto" ----
      if (i === 0) {
        const apunta = m.gesto === "apunta";
        const n = apunta ? this.red.neuronaEn(m.x, m.y) : null;
        this.red.mostrarTooltip(n, m.x, m.y);
        cursor.classList.toggle("sobre", !!n);
        if (apunta) {
          const r = this._esperar(m, n, ahora);
          progreso = r.progreso;
          accion = r.accion;
        } else {
          this.espera = null;
        }
      }

      cursor.style.setProperty("--p", progreso);
      cursor.classList.toggle("entrar", this.espera?.fase === 1 && i === 0);
      cursor.querySelector(".accion").textContent = accion;
      if (i === 0) textoAccion = dosAbiertas ? "✋✋ Dos manos: separe o junte para zoom" : TEXTO_GESTO[m.gesto];
    });

    // Esconde los cursores de las manos que salieron de la imagen.
    for (let i = actuales.length; i < 2; i++) this.cursores[i].hidden = true;
    if (!actuales.length) {
      this.red.mostrarTooltip(null);
      this.espera = null;
    }
    this.textoEstado.textContent = textoAccion ?? "Muestre la mano a la cámara ✋";
    const activo = dosAbiertas ? "zoom" : actuales[0]?.gesto;
    this.ayuda.querySelectorAll("[data-g]").forEach((fila) => fila.classList.toggle("activo", fila.dataset.g === activo));
    this.manos = actuales;
  }

  /**
   * ☝️ "Quedarse quieto" apuntando: llena el anillo del cursor y, al
   * completarse, pulsa el botón o selecciona / entra a la neurona.
   */
  _esperar(m, neurona, ahora) {
    // ¿Hay un botón debajo del dedo? (Regresar, Centrar, Salir…)
    const caja = this.panel.getBoundingClientRect();
    const boton = document
      .elementFromPoint(caja.left + m.x, caja.top + m.y)
      ?.closest(".barra-inmersiva button, .hud3d button, .controles-3d button");
    const botonValido = boton && !["botonManos", "botonManosInmersivo"].includes(boton.id) ? boton : null;
    const clave = botonValido ? `b:${botonValido.id || botonValido.textContent}`
      : neurona ? `n:${neurona.capa}-${neurona.indice}` : null;

    const e = this.espera;
    const seMovio = e && Math.hypot(m.x - e.x, m.y - e.y) > RADIO_QUIETO;
    if (!clave) { this.espera = null; return { progreso: 0, accion: "" }; }
    if (!e || e.clave !== clave || seMovio) {
      this.espera = { clave, x: m.x, y: m.y, desde: ahora, fase: 0, listo: false };
      return { progreso: 0, accion: "" };
    }
    if (e.listo) return { progreso: 0, accion: "" };

    const progreso = Math.min(1, (ahora - e.desde) / TIEMPO_APUNTAR);
    const yaDentro = neurona && this.red.dentro?.capa === neurona.capa && this.red.dentro?.indice === neurona.indice;
    let accion = botonValido ? `Pulsar «${botonValido.textContent.trim()}»`
      : e.fase === 0 ? "Ver cálculo…" : "Entrar…";
    if (yaDentro) accion = "Ya está dentro";

    if (progreso === 1) {
      if (botonValido) {
        botonValido.click();
        e.listo = true;
      } else if (yaDentro) {
        e.listo = true;
      } else if (e.fase === 0) {
        this.red.seleccionar(neurona.capa, neurona.indice); // 1 s: ver su cálculo
        Object.assign(e, { fase: 1, desde: ahora }); // sigue contando para entrar
      } else {
        this.red.entrar(neurona.capa, neurona.indice); // 2 s: meterse en la neurona
        e.listo = true;
      }
    }
    return { progreso, accion };
  }
}
