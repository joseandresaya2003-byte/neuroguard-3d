/* =============================================================
   umbral.js - Umbral de decisión compartido por toda la interfaz
   -------------------------------------------------------------
   La red entrega una probabilidad p. La REGLA DE DECISIÓN es:
       p >= umbral  ->  sospechosa      p < umbral  ->  normal
   El umbral oficial del taller es 0.5, pero se puede mover desde el
   panel "Resultado". Todas las vistas (medidor, mapa 2D, superficie 3D,
   red 3D, cálculo, métricas, laboratorio) leen el valor de aquí, así un
   solo cambio las actualiza a todas.
   ============================================================= */

export const umbral = {
  valor: 0.5, // Umbral vigente (lo fija app.js con el valor del servidor).
  oficial: 0.5, // Umbral con el que se entrenó y evaluó el modelo oficial.
};

/** ¿La probabilidad p se clasifica como sospechosa con el umbral vigente? */
export const esSospechosa = (p) => p >= umbral.valor;

/**
 * Cabecera HTTP con el umbral de ESTE navegador. Se envía en cada petición
 * para que el servidor calcule con él (cada usuario tiene su propio umbral).
 */
export const cabeceraUmbral = () => ({ "X-Umbral": umbral.valor.toFixed(2) });

/** Texto corto del umbral, por ejemplo "0.50". */
export const textoUmbral = () => umbral.valor.toFixed(2);
