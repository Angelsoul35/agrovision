# 02 — Generación sintética de hojas

## Por qué datos sintéticos

Necesitamos imágenes etiquetadas, pero no podemos descargar un dataset de campo
(restricción offline) ni licenciar uno. La solución: **dibujar** las hojas con
NumPy (y Pillow solo para codificar PNG en la API). Todo sale de un generador
determinista con semilla, así que los datos son reproducibles y los entendemos
por completo.

## Cómo se dibuja una hoja

1. Un **fondo** neutro con ruido.
2. Una **elipse** rotada como cuerpo de la hoja (variamos ejes y ángulo).
3. Un **color base** según la clase.
4. **Nervios**: se oscurece el nervio central y unas venas laterales.
5. Las **lesiones** de la enfermedad (manchas), si las hay.
6. Un **degradado de iluminación multiplicativo** y **ruido** de píxel.

## Las cuatro clases

- **healthy (sana)**: verde dominante, sin lesiones.
- **chlorosis (clorosis)**: **amarilleo global** de toda la hoja (pérdida de
  clorofila): sube R y G, baja la saturación (más pálida). La severidad va de
  suave (casi sana) a fuerte.
- **blight (tizón)**: **pocas manchas marrones grandes** (tejido necrótico) sobre
  hoja verde.
- **rust (roya)**: **muchas manchas naranja-marrón pequeñas** (pústulas).

## El detalle clave: solapamiento a propósito

Si las clases fueran perfectamente separables, el problema sería trivial y la
precisión, un 1.0 sin valor. Por eso las hacemos **solaparse como en la realidad**:

- la clorosis suave se parece a una hoja sana;
- la roya con pocas manchas grandes se parece al tizón;
- el color de la lesión se muestrea en un rango marrón↔naranja **compartido**,
  así que el color por sí solo no distingue roya de tizón → hay que mirar también
  la **geometría** (muchas pequeñas vs pocas grandes).

Además, cada "mando" del generador (tamaño y número de manchas, severidad de la
clorosis, iluminación) se divide en **bandas**. Esto permite la evaluación
rigurosa del documento 04: entrenar con unas bandas y probar con las **disjuntas**.
