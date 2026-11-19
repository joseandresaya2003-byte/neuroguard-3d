/* =============================================================
   superficie3d.js - Superficie 3D de probabilidad (Three.js)
   -------------------------------------------------------------
   Convierte el mapa de decisión en un "relieve":
     - eje X (izquierda → derecha): intentos fallidos 0-20
     - eje Z (adelante → atrás):    tráfico enviado 0-2000 MB
     - altura (eje Y):               probabilidad p que calcula la red
   Un plano translúcido representa el UMBRAL de decisión (0.5 por
   defecto; al moverlo en la interfaz el plano sube o baja animado):
   todo lo que sobresale del plano es "sospechosa" y lo que queda
   debajo es "normal". La línea donde la superficie corta el plano es
   la FRONTERA DE DECISIÓN.
   Cuando llegan mapas nuevos (por ejemplo, mientras el laboratorio
   entrena) la superficie se transforma suavemente: se ve aprender a la red.
   ============================================================= */

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { crearTexto, texturaHalo } from "./red3d.js";
import { colorProbabilidad } from "./graficas.js";
import { umbral, esSospechosa, textoUmbral } from "./umbral.js";

const LADO = 12; // Tamaño de la base (unidades 3D).
const ALTURA = 4.5; // Altura que corresponde a p = 1.

// Conversión entre valores reales y coordenadas 3D.
const xDe = (intentos) => -LADO / 2 + (intentos / 20) * LADO;
const zDe = (trafico) => LADO / 2 - (trafico / 2000) * LADO;

export class Superficie3D {
  /**
   * @param {HTMLElement} contenedor  Donde se dibuja.
   * @param {Function} alElegir       (intentos, tráfico) al hacer clic en la superficie (opcional).
   */
  constructor(contenedor, alElegir = null) {
    this.contenedor = contenedor;
    this.alElegir = alElegir;
    this.n = 0;
    this.alturas = null; // Alturas mostradas ahora (se animan hacia "destino").
    this.destino = null;
    this.actual = null; // Conexión actual {intentos, trafico, p}

    this._crearEscena();
    this._crearDecorado();
    this._crearEventos();
    this.reloj = new THREE.Clock();
    this.renderer.setAnimationLoop(() => this._cuadro());
  }

  _crearEscena() {
    this.escena = new THREE.Scene();
    this.camara = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
    this.camara.position.set(-11, 10, 15);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.contenedor.appendChild(this.renderer.domElement);

    this.controles = new OrbitControls(this.camara, this.renderer.domElement);
    this.controles.target.set(0, 1.4, 0);
    this.controles.enableDamping = true;
    this.controles.autoRotate = true;
    this.controles.autoRotateSpeed = 0.5;
    this.controles.minDistance = 8;
    this.controles.maxDistance = 40;
    this.controles.maxPolarAngle = Math.PI * 0.49; // No dejar pasar por debajo del piso.

    this.escena.add(new THREE.AmbientLight(0xb8b0ff, 0.9));
    const luz = new THREE.DirectionalLight(0xffffff, 1.6);
    luz.position.set(-6, 12, 8);
    this.escena.add(luz);

    new ResizeObserver(() => this._redimensionar()).observe(this.contenedor);
    this._redimensionar();
  }

  _redimensionar() {
    const { clientWidth: w, clientHeight: h } = this.contenedor;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camara.aspect = w / h;
    // En paneles angostos se aleja la cámara para que quepa toda la superficie.
    const distancia = Math.max(27, 24 / Math.max(this.camara.aspect, 0.5));
    if (!this.encuadrada) {
      this.camara.position.setLength(distancia);
      this.encuadrada = true;
    }
    this.camara.updateProjectionMatrix();
  }

  /** Piso, plano del umbral, ejes y marcador de la conexión actual. */
  _crearDecorado() {
    // Piso con rejilla.
    const rejilla = new THREE.GridHelper(LADO, 10, 0x4b4566, 0x2a2640);
    rejilla.material.transparent = true;
    rejilla.material.opacity = 0.6;
    this.escena.add(rejilla);

    // Plano del umbral (se mueve a la altura p = umbral en _cuadro).
    const plano = new THREE.Mesh(
      new THREE.PlaneGeometry(LADO, LADO),
      new THREE.MeshBasicMaterial({
        color: 0xffffff, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false,
      })
    );
    plano.rotation.x = -Math.PI / 2;
    this.escena.add(plano);
    const borde = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(LADO, LADO)),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 })
    );
    borde.rotation.x = -Math.PI / 2;
    this.escena.add(borde);
    this.textoUmbral = crearTexto(`umbral p = ${textoUmbral()}`, { tam: 0.5, color: "#e9e6ff" });
    this.textoUmbral.position.set(LADO / 2 + 1.2, 0, -LADO / 2);
    this.escena.add(this.textoUmbral);
    this.planoUmbral = [plano, borde, this.textoUmbral];
    this.alturaUmbral = umbral.valor * ALTURA; // Altura dibujada (se anima).
    this.umbralPintado = umbral.valor; // Umbral con el que se colorearon los vértices.

    // Ejes con marcas.
    const etiqueta = (texto, x, y, z, color = "#8f8aa8", tam = 0.42) => {
      const s = crearTexto(texto, { tam, color });
      s.position.set(x, y, z);
      this.escena.add(s);
    };
    [0, 5, 10, 15, 20].forEach((v) => etiqueta(String(v), xDe(v), 0, LADO / 2 + 0.8));
    [0, 500, 1000, 1500, 2000].forEach((v) => etiqueta(String(v), LADO / 2 + 1.1, 0, zDe(v)));
    etiqueta("intentos fallidos →", 0, -0.2, LADO / 2 + 1.8, "#c9c4e6", 0.5);
    etiqueta("tráfico MB →", LADO / 2 + 2.6, -0.2, 0, "#c9c4e6", 0.5);
    [0, 0.5, 1].forEach((v) => etiqueta(`p=${v}`, -LADO / 2 - 0.9, v * ALTURA, LADO / 2));
    const ejeP = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-LADO / 2, 0, LADO / 2),
        new THREE.Vector3(-LADO / 2, ALTURA, LADO / 2),
      ]),
      new THREE.LineBasicMaterial({ color: 0x8f8aa8 })
    );
    this.escena.add(ejeP);

    // Marcador: esfera + línea vertical + halo pulsante.
    this.marcador = new THREE.Group();
    this.esfera = new THREE.Mesh(
      new THREE.SphereGeometry(0.28, 24, 24),
      new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.6 })
    );
    this.halo = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: texturaHalo(), color: 0xffffff, transparent: true, opacity: 0.8,
        blending: THREE.AdditiveBlending, depthWrite: false,
      })
    );
    this.lineaMarcador = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 1, 0)]),
      new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.15, gapSize: 0.1, transparent: true, opacity: 0.7 })
    );
    this.textoMarcador = crearTexto("", { tam: 0.55, negrita: true, color: "#ffffff" });
    this.marcador.add(this.esfera, this.halo, this.lineaMarcador, this.textoMarcador);
    this.marcador.visible = false;
    this.escena.add(this.marcador);
    this.posMarcador = new THREE.Vector3();
  }

  /** Construye la malla (n x n vértices) la primera vez o si cambia el tamaño. */
  _crearMalla(n) {
    if (this.malla) {
      this.escena.remove(this.malla, this.alambre);
      this.malla.geometry.dispose();
    }
    this.n = n;
    const geo = new THREE.BufferGeometry();
    const posiciones = new Float32Array(n * n * 3);
    const colores = new Float32Array(n * n * 3);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const k = (j * n + i) * 3;
        posiciones[k] = xDe((i / (n - 1)) * 20);
        posiciones[k + 2] = zDe((j / (n - 1)) * 2000);
      }
    }
    const indices = [];
    for (let j = 0; j < n - 1; j++) {
      for (let i = 0; i < n - 1; i++) {
        const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }
    geo.setIndex(indices);
    geo.setAttribute("position", new THREE.BufferAttribute(posiciones, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colores, 3));
    this.malla = new THREE.Mesh(
      geo,
      new THREE.MeshStandardMaterial({
        vertexColors: true, side: THREE.DoubleSide, roughness: 0.45, metalness: 0.15,
        transparent: true, opacity: 0.96,
      })
    );
    this.alambre = new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.05 })
    );
    this.escena.add(this.malla, this.alambre);
    this.alturas = new Float32Array(n * n);
  }

  /**
   * Recibe un mapa de probabilidades (de /api/mapa o del laboratorio).
   * La superficie se transforma suavemente hacia los nuevos valores.
   */
  setMapa(mapa) {
    if (mapa.n !== this.n) this._crearMalla(mapa.n);
    this.destino = Float32Array.from(mapa.probabilidades.flat());
    this.mapa = mapa;
  }

  /** Coloca el marcador de la conexión analizada sobre la superficie. */
  setActual(intentos, trafico, p) {
    this.actual = { intentos: Math.min(intentos, 20), trafico: Math.min(trafico, 2000), p };
    this.marcador.visible = true;
    this._escribirMarcador();
  }

  _escribirMarcador() {
    const p = this.actual.p;
    this.textoMarcador.userData.escribir(`p = ${(p * 100).toFixed(1)}%`, esSospechosa(p) ? "#ff8da0" : "#7fdcef");
  }

  setAutoRotar(v) { this.controles.autoRotate = v; }

  _crearEventos() {
    if (!this.alElegir) return;
    const raycaster = new THREE.Raycaster();
    const puntero = new THREE.Vector2();
    const canvas = this.renderer.domElement;
    let inicio = null;
    canvas.addEventListener("pointerdown", (e) => (inicio = [e.clientX, e.clientY]));
    canvas.addEventListener("pointerup", (e) => {
      // Un clic (sin arrastrar) sobre la superficie elige esa conexión.
      if (!inicio || Math.hypot(e.clientX - inicio[0], e.clientY - inicio[1]) > 5 || !this.malla) return;
      const r = canvas.getBoundingClientRect();
      puntero.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      raycaster.setFromCamera(puntero, this.camara);
      const hit = raycaster.intersectObject(this.malla)[0];
      if (!hit) return;
      const intentos = Math.round(((hit.point.x + LADO / 2) / LADO) * 20);
      const trafico = Math.round((((LADO / 2 - hit.point.z) / LADO) * 2000) / 10) * 10;
      this.alElegir(Math.min(Math.max(intentos, 0), 20), Math.min(Math.max(trafico, 0), 2000));
    });
  }

  _cuadro() {
    const dt = Math.min(this.reloj.getDelta(), 0.05);
    const t = this.reloj.elapsedTime;
    this.controles.update();

    // Plano del umbral: sube o baja suavemente hasta p = umbral.
    if (umbral.valor !== this.umbralPintado) {
      this.umbralPintado = umbral.valor;
      this.pintada = false; // Los colores dependen del umbral: repintar.
      this.textoUmbral.userData.escribir(`umbral p = ${textoUmbral()}`);
      if (this.actual) this._escribirMarcador();
    }
    this.alturaUmbral += (umbral.valor * ALTURA - this.alturaUmbral) * Math.min(1, dt * 8);
    this.planoUmbral.forEach((o) => (o.position.y = this.alturaUmbral));

    // Transformación suave de la superficie hacia el mapa más reciente.
    if (this.destino && this.malla) {
      const pos = this.malla.geometry.attributes.position;
      const col = this.malla.geometry.attributes.color;
      const factor = Math.min(1, dt * 7);
      let cambio = 0;
      for (let k = 0; k < this.alturas.length; k++) {
        const d = this.destino[k] - this.alturas[k];
        if (Math.abs(d) > 1e-4) {
          this.alturas[k] += d * factor;
          cambio++;
        }
      }
      if (cambio || !this.pintada) {
        for (let k = 0; k < this.alturas.length; k++) {
          const p = this.alturas[k];
          pos.array[k * 3 + 1] = p * ALTURA;
          const [r, g, b] = colorProbabilidad(p);
          col.array[k * 3] = (r / 255) ** 2.2; // a espacio lineal de color
          col.array[k * 3 + 1] = (g / 255) ** 2.2;
          col.array[k * 3 + 2] = (b / 255) ** 2.2;
        }
        pos.needsUpdate = col.needsUpdate = true;
        this.malla.geometry.computeVertexNormals();
        this.pintada = true;
      }
    }

    // El marcador sigue la altura actual de la superficie en su posición.
    if (this.actual && this.alturas) {
      const n = this.n;
      const i = Math.round((this.actual.intentos / 20) * (n - 1));
      const j = Math.round((this.actual.trafico / 2000) * (n - 1));
      const alturaSuperficie = this.alturas[j * n + i] * ALTURA;
      const objetivo = new THREE.Vector3(xDe(this.actual.intentos), 0, zDe(this.actual.trafico));
      this.posMarcador.lerp(objetivo, Math.min(1, dt * 6));
      this.marcador.position.copy(this.posMarcador);
      const y = alturaSuperficie + 0.28;
      this.esfera.position.y = y;
      this.halo.position.y = y;
      this.halo.scale.setScalar(1.6 + 0.5 * Math.sin(t * 4));
      this.textoMarcador.position.y = y + 0.8;
      this.lineaMarcador.scale.y = Math.max(y, 0.01);
      this.lineaMarcador.computeLineDistances();
      const color = esSospechosa(this.actual.p) ? 0xe8455f : 0x0ea5c4;
      this.esfera.material.emissive.setHex(color);
      this.halo.material.color.setHex(color);
    }

    this.renderer.render(this.escena, this.camara);
  }
}
