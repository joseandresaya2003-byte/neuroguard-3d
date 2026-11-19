/* =============================================================
   red3d.js - Dibujo de la red neuronal 2-8-4-1 en 3D con Three.js
   -------------------------------------------------------------
   ¿Qué dibuja?
     - Cada NEURONA es una esfera. Las capas van de izquierda a derecha:
       entradas (2) -> oculta 1 (8 ReLU) -> oculta 2 (4 ReLU) -> salida (1 sigmoide).
     - Cada CONEXIÓN es un cilindro delgado entre dos neuronas. Usa los
       PESOS REALES de la red entrenada (recibidos del servidor):
         color cian = peso positivo, color rosa = peso negativo,
         grosor y brillo = tamaño del peso |w|.
     - Al analizar una conexión, unos PULSOS de luz viajan por las
       conexiones capa por capa (propagación hacia adelante). El tamaño del
       pulso es proporcional a |w · a| (lo que realmente aporta esa conexión).
       Si una neurona ReLU da 0, no envía pulsos: se ve que está "apagada".
     - Las neuronas brillan según su activación real.
   Interacción: arrastrar = girar, rueda = zoom, pasar el ratón = ver valores,
   clic = ver el cálculo matemático de esa neurona.
   ============================================================= */

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { umbral, esSospechosa } from "./umbral.js";

// Colores (los mismos de la hoja de estilos).
const COLOR_POSITIVO = new THREE.Color("#0ea5c4");
const COLOR_NEGATIVO = new THREE.Color("#e8455f");
const COLOR_APAGADA = new THREE.Color("#2b2740");
const COLOR_ENCENDIDA = new THREE.Color("#ddd5ff");
const COLOR_ACTIVA = new THREE.Color("#8b7cf6"); // violeta: neurona oculta encendida

// Posición horizontal (eje X) de cada capa y radio del anillo de neuronas.
const POS_X = [-6.6, -2.2, 2.2, 6.6];
const NOMBRES_CAPAS = ["Entradas", "Oculta 1 · ReLU", "Oculta 2 · ReLU", "Salida · Sigmoide"];
const DURACION_ETAPA = 0.75; // Segundos que tarda la señal en pasar de una capa a otra.

/** Crea una textura circular difuminada (halo de luz) dibujada en un canvas. */
export function texturaHalo() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.25, "rgba(255,255,255,.45)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

/** Crea un texto flotante (sprite) que siempre mira a la cámara. */
export function crearTexto(texto, { tam = 0.5, color = "#b4b1c8", negrita = false } = {}) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 96;
  const textura = new THREE.CanvasTexture(canvas);
  textura.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: textura, transparent: true, depthWrite: false })
  );
  sprite.scale.set(tam * 5.33, tam, 1);
  // Función para cambiar el texto sin crear un sprite nuevo.
  sprite.userData.escribir = (nuevo, nuevoColor = color) => {
    const g = canvas.getContext("2d");
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.font = `${negrita ? "700" : "500"} 54px Consolas, "Cascadia Code", monospace`;
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.lineWidth = 10;
    g.strokeStyle = "rgba(8,7,13,.9)";
    g.strokeText(nuevo, 256, 48);
    g.fillStyle = nuevoColor;
    g.fillText(nuevo, 256, 48);
    textura.needsUpdate = true;
  };
  sprite.userData.escribir(texto);
  return sprite;
}

/** Posiciones de n neuronas formando un anillo en el plano Y-Z. */
function posicionesCapa(n, x) {
  if (n === 1) return [new THREE.Vector3(x, 0, 0)];
  if (n === 2) return [new THREE.Vector3(x, 1.5, 0), new THREE.Vector3(x, -1.5, 0)];
  const radio = n >= 8 ? 3.0 : 2.0;
  return Array.from({ length: n }, (_, i) => {
    const ang = (i / n) * Math.PI * 2 + Math.PI / 2;
    return new THREE.Vector3(x, Math.sin(ang) * radio, Math.cos(ang) * radio);
  });
}

export class Red3D {
  /**
   * @param {HTMLElement} contenedor  Elemento donde se inserta el canvas 3D.
   * @param {Array} capasModelo       Capas recibidas de /api/modelo (pesos y sesgos reales).
   * @param {Function} alSeleccionar  Se llama con (capa, neurona) al hacer clic en una neurona.
   */
  constructor(contenedor, capasModelo, alSeleccionar) {
    this.contenedor = contenedor;
    this.capasModelo = capasModelo;
    this.alSeleccionar = alSeleccionar;
    this.tooltip = contenedor.querySelector("#tooltip3d");
    this.recorrido = null; // Últimos valores z y a calculados por la red.
    this.animacion = null; // Estado de la animación de pulsos.
    this.seleccion = null;
    this.mostrarValores = true;
    this.hud = contenedor.querySelector("#hud3d"); // aviso "dentro de la neurona"
    this.dentro = null; // { capa, indice } si la cámara entró a una neurona
    this.vuelo = null; // animación de cámara en curso (entrar / salir)
    this.autoRotarUsuario = true; // lo que eligió el usuario con el botón Auto-rotar
    // Giro y zoom "pendientes" que pide la mano: se aplican poco a poco en cada
    // cuadro (60 por segundo), así el movimiento es fluido aunque la cámara web
    // solo entregue ~30 imágenes por segundo.
    this.pendiente = { h: 0, v: 0, zoom: 0 };

    this._crearEscena();
    this._crearRed();
    this._crearEventos();
    this.reloj = new THREE.Clock();
    this.renderer.setAnimationLoop(() => this._cuadro());
  }

  /* ---------- Escena, cámara, luces y controles ---------- */
  _crearEscena() {
    this.escena = new THREE.Scene();
    this.escena.fog = new THREE.FogExp2(0x08070d, 0.026);

    this.camara = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
    this.posicionInicial = new THREE.Vector3(2.5, 4.5, 17);
    this.camara.position.copy(this.posicionInicial);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // nitidez sin perder fluidez
    this.contenedor.appendChild(this.renderer.domElement);

    this.controles = new OrbitControls(this.camara, this.renderer.domElement);
    this.controles.enableDamping = true; // Movimiento suave.
    this.controles.autoRotate = true;
    this.controles.autoRotateSpeed = 0.6;
    this.controles.minDistance = 7;
    this.controles.maxDistance = 34;

    this.escena.add(new THREE.AmbientLight(0x8899bb, 0.7));
    const luz = new THREE.PointLight(0xc4b5fd, 60, 60);
    luz.position.set(0, 8, 10);
    this.escena.add(luz);

    // Campo de estrellas de fondo (decorativo).
    const n = 600;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) pos[i] = (Math.random() - 0.5) * 80;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.estrellas = new THREE.Points(
      geo,
      new THREE.PointsMaterial({ color: 0x4a4366, size: 0.12, transparent: true, opacity: 0.8 })
    );
    this.escena.add(this.estrellas);

    // Ajusta el tamaño del canvas cuando cambia el tamaño del panel.
    new ResizeObserver(() => this._redimensionar()).observe(this.contenedor);
    this._redimensionar();
  }

  _redimensionar() {
    const { clientWidth: w, clientHeight: h } = this.contenedor;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camara.aspect = w / h;
    this.camara.updateProjectionMatrix();
    // Calcula a qué distancia debe estar la cámara para que quepa toda la red
    // (ancho ≈ 18 unidades, alto ≈ 9) sin importar la forma del panel.
    const tanVertical = Math.tan(THREE.MathUtils.degToRad(this.camara.fov / 2));
    const distancia = Math.max(9.5 / (tanVertical * this.camara.aspect), 5.5 / tanVertical, 14);
    this.posicionInicial = new THREE.Vector3(0.15, 0.25, 1).normalize().multiplyScalar(distancia);
    this.controles.maxDistance = Math.max(34, distancia * 1.6);
    // En pantallas angostas (celular vertical) la cámara queda más lejos:
    // se reduce la niebla para que la red no se vea oscura.
    this.escena.fog.density = 0.026 * Math.min(1, 17 / distancia);
    if (!this.yaEncuadrada) {
      this.camara.position.copy(this.posicionInicial);
      this.yaEncuadrada = true;
    }
  }

  /* ---------- Construcción de neuronas y conexiones ---------- */
  _crearRed() {
    const halo = texturaHalo();
    const tamanos = [2, ...this.capasModelo.map((c) => c.neuronas)]; // [2, 8, 4, 1]
    this.neuronas = []; // neuronas[capa][i] = { malla, halo, texto, pos }
    this.conexiones = []; // conexiones[capa] = lista de { malla, origen, destino, peso }
    this.mallasNeuronas = []; // Para detectar clics (raycasting).

    tamanos.forEach((n, capa) => {
      const posiciones = posicionesCapa(n, POS_X[capa]);
      const radio = capa === 3 ? 0.62 : 0.42;
      const lista = posiciones.map((p, i) => {
        const malla = new THREE.Mesh(
          new THREE.SphereGeometry(radio, 32, 32),
          new THREE.MeshStandardMaterial({
            color: COLOR_APAGADA, emissive: COLOR_ENCENDIDA, emissiveIntensity: 0.05,
            roughness: 0.35, metalness: 0.2,
          })
        );
        malla.position.copy(p);
        malla.userData = { capa, indice: i };
        this.escena.add(malla);
        this.mallasNeuronas.push(malla);

        const brillo = new THREE.Sprite(
          new THREE.SpriteMaterial({
            map: halo, color: COLOR_ENCENDIDA, transparent: true, opacity: 0,
            blending: THREE.AdditiveBlending, depthWrite: false,
          })
        );
        brillo.position.copy(p);
        brillo.scale.setScalar(radio * 5);
        this.escena.add(brillo);

        const texto = crearTexto("", { tam: 0.42 });
        texto.position.copy(p).add(new THREE.Vector3(0, radio + 0.42, 0));
        this.escena.add(texto);

        return { malla, halo: brillo, texto, pos: p, radio };
      });
      this.neuronas.push(lista);

      // Nombre de la capa flotando debajo.
      const nombre = crearTexto(NOMBRES_CAPAS[capa], { tam: 0.5, color: "#77748e" });
      nombre.position.set(POS_X[capa], capa === 0 ? -3.2 : -(n >= 8 ? 3.0 : 2.0) - 1.3, 0);
      this.escena.add(nombre);
    });

    // Etiquetas fijas de las entradas.
    this.neuronas[0][0].texto.userData.escribir("x₁", "#22d3ee");
    this.neuronas[0][1].texto.userData.escribir("x₂", "#22d3ee");

    // Conexiones con los pesos reales.
    const maxPeso = Math.max(...this.capasModelo.flatMap((c) => c.pesos.flat().map(Math.abs)));
    const eje = new THREE.Vector3(0, 1, 0);
    this.capasModelo.forEach((capaModelo, l) => {
      const lista = [];
      capaModelo.pesos.forEach((filaPesos, i) => {
        filaPesos.forEach((peso, j) => {
          const a = this.neuronas[l][i].pos;
          const b = this.neuronas[l + 1][j].pos;
          const largo = a.distanceTo(b);
          const fuerza = Math.abs(peso) / maxPeso; // 0..1
          const malla = new THREE.Mesh(
            new THREE.CylinderGeometry(1, 1, largo, 6, 1, true),
            new THREE.MeshBasicMaterial({
              color: peso >= 0 ? COLOR_POSITIVO : COLOR_NEGATIVO,
              transparent: true, opacity: 0.12 + 0.5 * fuerza,
              blending: THREE.AdditiveBlending, depthWrite: false,
            })
          );
          const grosor = 0.012 + 0.05 * fuerza;
          malla.scale.set(grosor, 1, grosor);
          malla.position.copy(a).add(b).multiplyScalar(0.5);
          malla.quaternion.setFromUnitVectors(eje, b.clone().sub(a).normalize());
          malla.userData.opacidadBase = malla.material.opacity;
          this.escena.add(malla);

          // Pulso de luz que recorrerá esta conexión durante la animación.
          const pulso = new THREE.Sprite(
            new THREE.SpriteMaterial({
              map: halo, color: peso >= 0 ? COLOR_POSITIVO : COLOR_NEGATIVO,
              transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false,
            })
          );
          pulso.visible = false;
          this.escena.add(pulso);

          lista.push({ malla, pulso, origen: i, destino: j, peso, a, b });
        });
      });
      this.conexiones.push(lista);
    });

    // Anillo que marca la neurona seleccionada.
    this.anillo = new THREE.Mesh(
      new THREE.TorusGeometry(0.72, 0.035, 12, 48),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 })
    );
    this.anillo.visible = false;
    this.escena.add(this.anillo);
  }

  /* ---------- Ratón: tooltip y selección ---------- */
  _crearEventos() {
    const raycaster = new THREE.Raycaster();
    const puntero = new THREE.Vector2();
    const canvas = this.renderer.domElement;
    let inicioClic = null;

    const neuronaBajo = (e) => {
      const r = canvas.getBoundingClientRect();
      puntero.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      raycaster.setFromCamera(puntero, this.camara);
      const hit = raycaster.intersectObjects(this.mallasNeuronas)[0];
      return hit ? hit.object.userData : null;
    };

    canvas.addEventListener("pointermove", (e) => {
      const n = neuronaBajo(e);
      if (!n) { this.tooltip.hidden = true; canvas.style.cursor = ""; return; }
      canvas.style.cursor = "pointer";
      const r = this.contenedor.getBoundingClientRect();
      this.tooltip.style.left = `${e.clientX - r.left}px`;
      this.tooltip.style.top = `${e.clientY - r.top}px`;
      this.tooltip.innerHTML = this._textoTooltip(n.capa, n.indice);
      this.tooltip.hidden = false;
    });
    canvas.addEventListener("pointerleave", () => (this.tooltip.hidden = true));
    // Se distingue un clic de un arrastre (girar la cámara).
    // Funciona igual con ratón, lápiz o dedo (pointer events).
    let ultimoToque = { t: 0, clave: null };
    canvas.addEventListener("pointerdown", (e) => (inicioClic = [e.clientX, e.clientY]));
    canvas.addEventListener("pointerup", (e) => {
      const tolerancia = e.pointerType === "touch" ? 12 : 5; // el dedo tiembla más que el ratón
      if (!inicioClic || Math.hypot(e.clientX - inicioClic[0], e.clientY - inicioClic[1]) > tolerancia) return;
      const n = neuronaBajo(e);
      const clave = n ? `${n.capa}-${n.indice}` : "vacio";
      const dobleToque = performance.now() - ultimoToque.t < 380 && ultimoToque.clave === clave;
      ultimoToque = { t: performance.now(), clave };
      if (n && dobleToque) this.entrar(n.capa, n.indice); // 2 clics / 2 toques: entrar
      else if (n) this.seleccionar(n.capa, n.indice); // 1 clic / 1 toque: ver su cálculo
      else if (dobleToque && this.dentro) this.salir(); // doble clic en el vacío: salir
    });
    // Escape = salir de la neurona.
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.dentro) this.salir();
    });
    this._neuronaBajo = neuronaBajo;
  }

  /**
   * Neurona que hay debajo de un punto del panel (coordenadas en píxeles del
   * contenedor). La usa el control con las manos para su cursor.
   */
  neuronaEn(x, y) {
    const r = this.renderer.domElement.getBoundingClientRect();
    return this._neuronaBajo({ clientX: r.left + x, clientY: r.top + y });
  }

  /** Muestra (o esconde con null) el tooltip de una neurona en (x, y) del panel. */
  mostrarTooltip(n, x, y) {
    if (!n) { this.tooltip.hidden = true; return; }
    this.tooltip.style.left = `${x}px`;
    this.tooltip.style.top = `${y}px`;
    this.tooltip.innerHTML = this._textoTooltip(n.capa, n.indice);
    this.tooltip.hidden = false;
  }

  _textoTooltip(capa, i) {
    const nombre = capa === 0 ? `x${i + 1}` : capa === 3 ? "ŷ salida" : `h${i + 1}`;
    let t = `<b>${NOMBRES_CAPAS[capa]} · ${nombre}</b>`;
    if (!this.recorrido) return t + "<br>analice una conexión";
    if (capa === 0) return t + `<br>valor normalizado = ${this.entrada[i].toFixed(4)}`;
    const paso = this.recorrido[capa - 1];
    return t + `<br>z = ${paso.z[i].toFixed(4)}<br>a = ${paso.a[i].toFixed(4)}`;
  }

  /* ---------- API pública ---------- */

  /** Marca una neurona, resalta sus conexiones de entrada y avisa a app.js. */
  seleccionar(capa, indice) {
    this.seleccion = { capa, indice };
    const n = this.neuronas[capa][indice];
    this.anillo.visible = true;
    this.anillo.position.copy(n.pos);
    this.anillo.scale.setScalar(n.radio / 0.42);
    // Resalta solo las conexiones que llegan a la neurona elegida.
    this.conexiones.forEach((lista, l) =>
      lista.forEach((c) => {
        const llega = capa > 0 && l === capa - 1 && c.destino === indice;
        const sale = l === capa && c.origen === indice;
        c.malla.material.opacity = llega ? 0.95 : sale ? 0.45 : c.malla.userData.opacidadBase * 0.25;
      })
    );
    this.alSeleccionar?.(capa, indice);
  }

  limpiarSeleccion() {
    this.seleccion = null;
    this.anillo.visible = false;
    this.conexiones.flat().forEach((c) => (c.malla.material.opacity = c.malla.userData.opacidadBase));
  }

  /**
   * Recibe el resultado real del servidor y lanza la animación.
   * @param {Array} recorrido  [{z:[...], a:[...]}, ...] por capa (calculado con los pesos reales).
   * @param {Array} entrada    Entradas normalizadas [x1, x2].
   */
  mostrarRecorrido(recorrido, entrada) {
    this.recorrido = recorrido;
    this.entrada = entrada;
    // Valores de activación por capa (la capa 0 son las entradas normalizadas).
    this.valores = [entrada.map(Math.abs), ...recorrido.map((p) => p.a)];
    // Aporte de cada conexión = |w · a_origen|, normalizado por capa.
    this.conexiones.forEach((lista, l) => {
      const fuente = l === 0 ? entrada : recorrido[l - 1].a;
      const aportes = lista.map((c) => Math.abs(c.peso * fuente[c.origen]));
      const max = Math.max(...aportes, 1e-6);
      lista.forEach((c, k) => (c.aporte = aportes[k] / max));
    });
    // Reinicia brillo de neuronas (se encienden cuando llega la señal).
    this.neuronas.forEach((capa) => capa.forEach((n) => (n.objetivo = 0)));
    this.animacion = { t: 0, etapaEncendida: -1 };
    this._escribirValores();
    this._pintarHud(); // si está dentro de una neurona, actualiza sus valores
  }

  setAutoRotar(activo) {
    this.autoRotarUsuario = activo;
    this.controles.autoRotate = activo && !this.dentro;
  }

  /** Gira la cámara alrededor del punto que mira (ángulos en radianes). */
  girar(dHorizontal, dVertical) {
    if (this.vuelo) return;
    const desde = this.camara.position.clone().sub(this.controles.target);
    const esfera = new THREE.Spherical().setFromVector3(desde);
    esfera.theta -= dHorizontal;
    esfera.phi = Math.min(Math.PI - 0.15, Math.max(0.15, esfera.phi - dVertical));
    desde.setFromSpherical(esfera);
    this.camara.position.copy(this.controles.target).add(desde);
  }

  /** Igual que girar(), pero repartido en varios cuadros (movimiento suave). */
  girarSuave(dHorizontal, dVertical) {
    this.pendiente.h += dHorizontal;
    this.pendiente.v += dVertical;
  }

  /** Igual que zoom(), pero repartido en varios cuadros. */
  zoomSuave(factor) {
    this.pendiente.zoom += Math.log(factor);
  }

  /** Acerca (factor < 1) o aleja (factor > 1) la cámara. */
  zoom(factor) {
    if (this.vuelo) return;
    const desde = this.camara.position.clone().sub(this.controles.target);
    const largo = Math.min(this.controles.maxDistance, Math.max(this.controles.minDistance, desde.length() * factor));
    this.camara.position.copy(this.controles.target).add(desde.setLength(largo));
  }

  /**
   * Vuelve a encuadrar toda la red con un vuelo suave (por ejemplo al pasar a
   * pantalla completa). La distancia ya se calculó según la forma de la
   * pantalla en _redimensionar(), así cabe igual en celular vertical o en PC.
   * @param {number} factor  < 1 acerca un poco (pantallas anchas).
   */
  encuadrar(factor = 1) {
    if (this.dentro) return; // si está dentro de una neurona, se queda ahí
    this._volar(this.posicionInicial.clone().multiplyScalar(factor), new THREE.Vector3(0, 0, 0));
  }

  /** Detiene el giro automático mientras el usuario usa las manos. */
  pausarAutoRotar(pausar) {
    this.controles.autoRotate = !pausar && this.autoRotarUsuario && !this.dentro;
  }

  /**
   * 🔍 "Meterse" en una neurona: la cámara vuela hasta ella, la selecciona y
   * muestra arriba su valor. Dentro se puede seguir girando alrededor de ella.
   */
  entrar(capa, indice) {
    const n = this.neuronas[capa][indice];
    this.seleccionar(capa, indice);
    this.dentro = { capa, indice };
    this.controles.autoRotate = false;
    this.controles.minDistance = 0.8; // permite acercarse mucho a la neurona
    // Se acerca desde el lado donde ya está la cámara (el movimiento es natural).
    const direccion = this.camara.position.clone().sub(n.pos).normalize();
    // Distancia para que la neurona ocupe ~30 % del lado más corto del panel
    // (se ve completa y alrededor se alcanzan a ver sus conexiones).
    const fovV = THREE.MathUtils.degToRad(this.camara.fov);
    const fovH = 2 * Math.atan(Math.tan(fovV / 2) * this.camara.aspect);
    const lejos = n.radio / (0.3 * Math.tan(Math.min(fovV, fovH) / 2));
    const destino = n.pos.clone().add(direccion.multiplyScalar(Math.min(14, Math.max(n.radio * 4, lejos))));
    this._volar(destino, n.pos.clone());
    this._pintarHud();
  }

  /** Sale de la neurona y vuelve a la vista general. */
  salir() {
    this.dentro = null;
    this.hud.hidden = true;
    this.limpiarSeleccion();
    this._volar(this.posicionInicial.clone(), new THREE.Vector3(0, 0, 0), () => {
      this.controles.minDistance = 7;
      this.controles.autoRotate = this.autoRotarUsuario;
    });
  }

  _volar(posicion, objetivo, alTerminar) {
    this.controles.enabled = false; // durante el vuelo no se puede arrastrar
    this.vuelo = {
      t: 0, duracion: 1.1,
      desdePos: this.camara.position.clone(), haciaPos: posicion,
      desdeObj: this.controles.target.clone(), haciaObj: objetivo,
      alTerminar,
    };
  }

  _pintarHud() {
    if (!this.dentro) return;
    const { capa, indice } = this.dentro;
    const [titulo, ...valores] = this._textoTooltip(capa, indice).split("<br>");
    this.hud.innerHTML = `<span>🔍 Dentro de ${titulo}</span>
      <span class="valores">${valores.join(" · ")}</span>
      <button type="button" class="mini">Salir ✕</button>`;
    this.hud.querySelector("button").addEventListener("click", () => this.salir());
    this.hud.hidden = false;
  }
  setValores(activo) {
    this.mostrarValores = activo;
    this.neuronas.slice(1).flat().forEach((n) => (n.texto.visible = activo));
  }
  centrar() {
    this.salir(); // vuelo suave a la vista inicial (también sale de una neurona)
  }

  /* ---------- Lógica interna de animación ---------- */
  _escribirValores() {
    this.recorrido.forEach((paso, l) =>
      paso.a.forEach((a, i) => {
        const esSalida = l === this.recorrido.length - 1;
        const texto = esSalida ? `p=${(a * 100).toFixed(1)}%` : a.toFixed(2);
        const color = esSalida ? (esSospechosa(a) ? "#ff8da0" : "#5fd3ea") : a > 0 ? "#e6ecf5" : "#4d5a75";
        this.neuronas[l + 1][i].texto.userData.escribir(texto, color);
      })
    );
  }

  /** Enciende las neuronas de una capa según su activación real. */
  _encenderCapa(capa) {
    const vals = this.valores[capa];
    const max = Math.max(...vals, 1e-6);
    this.neuronas[capa].forEach((n, i) => {
      if (capa === 3) {
        const p = vals[i];
        // Brilla más cuanto más lejos esté p del umbral (la red está más segura).
        const u = umbral.valor;
        n.objetivo = 0.4 + 0.6 * Math.min(1, Math.abs(p - u) / Math.max(u, 1 - u));
        const col = esSospechosa(p) ? COLOR_NEGATIVO : COLOR_POSITIVO;
        n.malla.material.emissive.copy(col);
        n.halo.material.color.copy(col);
        n.malla.material.color.copy(col).multiplyScalar(0.5);
      } else {
        n.objetivo = vals[i] / max; // 0 = neurona ReLU apagada
        n.malla.material.color.copy(COLOR_APAGADA).lerp(COLOR_ACTIVA, n.objetivo * 0.8);
      }
    });
  }

  _cuadro() {
    const dt = Math.min(this.reloj.getDelta(), 0.05);
    const tiempo = this.reloj.elapsedTime;

    // Vuelo de la cámara al entrar o salir de una neurona.
    if (this.vuelo) {
      const v = this.vuelo;
      v.t = Math.min(1, v.t + dt / v.duracion);
      const e = v.t < 0.5 ? 4 * v.t ** 3 : 1 - (-2 * v.t + 2) ** 3 / 2; // suave al inicio y al final
      this.camara.position.lerpVectors(v.desdePos, v.haciaPos, e);
      this.controles.target.lerpVectors(v.desdeObj, v.haciaObj, e);
      if (v.t === 1) {
        this.vuelo = null;
        this.controles.enabled = true;
        v.alTerminar?.();
      }
    }
    // Aplica una parte del giro/zoom pendiente (rápido pero sin saltos).
    const p = this.pendiente;
    if (Math.abs(p.h) + Math.abs(p.v) + Math.abs(p.zoom) > 1e-5) {
      const k = 1 - Math.exp(-dt * 16);
      this.girar(p.h * k, p.v * k);
      this.zoom(Math.exp(p.zoom * k));
      p.h -= p.h * k;
      p.v -= p.v * k;
      p.zoom -= p.zoom * k;
    }
    this.controles.update();
    this.estrellas.rotation.y += dt * 0.01;

    // Animación de la propagación hacia adelante.
    if (this.animacion) {
      const an = this.animacion;
      an.t += dt;
      const etapaActual = Math.floor(an.t / DURACION_ETAPA); // 0,1,2 = capas de conexiones
      // Enciende la capa a la que ya llegó la señal.
      while (an.etapaEncendida < Math.min(etapaActual, 3)) {
        an.etapaEncendida++;
        this._encenderCapa(an.etapaEncendida);
      }
      this.conexiones.forEach((lista, l) => {
        const progreso = (an.t - l * DURACION_ETAPA) / DURACION_ETAPA;
        lista.forEach((c) => {
          const activo = progreso > 0 && progreso < 1 && c.aporte > 0.02;
          c.pulso.visible = activo;
          if (activo) {
            const e = progreso * progreso * (3 - 2 * progreso); // suavizado
            c.pulso.position.lerpVectors(c.a, c.b, e);
            c.pulso.scale.setScalar(0.25 + 0.75 * c.aporte);
            c.pulso.material.opacity = 0.35 + 0.65 * c.aporte;
          }
        });
      });
      if (an.t > DURACION_ETAPA * 3 + 0.2) {
        this.conexiones.flat().forEach((c) => (c.pulso.visible = false));
        this.animacion = null;
      }
    }

    // Transición suave del brillo de cada neurona hacia su valor objetivo.
    this.neuronas.forEach((capa, l) =>
      capa.forEach((n) => {
        const obj = n.objetivo ?? 0;
        n.brillo = (n.brillo ?? 0) + (obj - (n.brillo ?? 0)) * Math.min(1, dt * 6);
        const latido = l === 3 && this.recorrido ? 1 + 0.08 * Math.sin(tiempo * 4) : 1;
        n.malla.material.emissiveIntensity = 0.05 + n.brillo * 1.4;
        n.halo.material.opacity = n.brillo * 0.85;
        n.halo.scale.setScalar(n.radio * (3 + 3 * n.brillo) * latido);
      })
    );

    if (this.anillo.visible) this.anillo.lookAt(this.camara.position);
    this.renderer.render(this.escena, this.camara);
  }
}
