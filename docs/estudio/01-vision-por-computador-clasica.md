# 01 — Visión por computador clásica

## Por qué visión clásica (una decisión de diseño)

Lo "fácil" sería afinar una CNN preentrenada. Aquí se elige a propósito **visión
clásica** por dos razones que un reclutador valora: (1) **explicabilidad** — en una
herramienta que aconseja tratar un cultivo, cada veredicto debe poder leerse y
revocarse, y una CNN es una caja negra; (2) **dominio del problema** — construir las
features a mano demuestra que entiendo *qué* distingue una enfermedad de otra, no solo
que sé llamar a `model.fit`. El núcleo corre offline y determinista; el benchmark con
fotos reales usa un subconjunto pequeño vendoreado. Una CNN daría más accuracy
aprendiendo sus propios rasgos — es la alternativa "accuracy primero", un trade-off
consciente, no una carencia de este diseño.

## Segmentar la hoja del fondo

Primero separamos los píxeles de hoja de los del fondo. El fondo es neutro y poco
saturado (un tono tierra); la hoja —verde, amarilla, marrón o naranja— está
**más saturada**. Así, la máscara de hoja es simplemente `saturación > umbral`.
Truco importante: el degradado de iluminación que añade el generador es
**multiplicativo**, y la saturación **no cambia** al multiplicar los tres canales
por el mismo factor. Por eso la segmentación por saturación es robusta a la luz.

## Los features (qué medimos y por qué)

Sobre los píxeles de hoja calculamos estadísticos con significado:

- **Color**: medias de R, G, B; **fracción de verde** `G/(R+G+B)` (una hoja sana
  es verde-dominante); un **índice de amarillo** `min(r,g) − b` (sube con la
  clorosis); un **índice de marrón** `max(r−g, 0)` (sube con las lesiones).
- **HSV calculado desde RGB**: media de tono (hue) y de saturación. La clorosis
  es **pálida** → saturación baja.
- **Cobertura de lesión**: una máscara por umbral de color (rojizo-marrón sobre
  verde, azul bajo) marca el tejido enfermo; medimos qué fracción de la hoja
  ocupa.
- **Geometría de las lesiones** con `scipy.ndimage.label`: número de
  componentes conexas y su tamaño. Esto es lo que separa **muchas manchas
  pequeñas** (roya) de **pocas manchas grandes** (tizón).
- **Textura**: varianza de la magnitud del gradiente. La roya es "moteada" →
  textura alta.

Cada feature es una línea de código que un agrónomo podría entender. Esa es la
gracia de la visión clásica: nada es una caja negra.

## Ejemplo trabajado: los features sobre hojas reales

Para ver que estos rasgos **capturan señal de verdad**, estas son las medias de
tres de ellos sobre las fotos reales del dataset Beans (30 hojas por clase):

| Clase (real) | `green_fraction` | `spot_coverage` | `n_spots` |
|---|---|---|---|
| healthy (sana) | 0.421 | 0.171 | 5.2 |
| bean_rust (roya) | 0.387 | 0.274 | 7.6 |
| angular_leaf_spot | 0.418 | 0.326 | 8.7 |

La lectura es directa y es la que usa el clasificador: a medida que la hoja enferma,
la **cobertura de lesión** sube (0.17 → 0.27 → 0.33) y el **número de componentes de
lesión** crece (5 → 8 → 9), mientras la fracción de verde baja un poco. No es una
separación perfecta —en hojas reales las sombras, nervios y el fondo elevan algo la
cobertura incluso en una hoja sana, por eso `spot_coverage` sana no es 0—, pero la
**tendencia relativa** es real y es la que da los ~0.67 de accuracy. Ahí se ve, con
números, por qué un rasgo *con significado* funciona: no hace falta una caja negra
para que la cobertura y el conteo de lesiones ordenen las clases.

## Ejemplo trabajado: por qué una roya es una roya

Para una roya sintética que el modelo clasifica (`p=0.91`), los rasgos que más la
empujan a esa clase son: **`high mean_saturation` (1.55)**, `high mean_hue` (0.75),
`high n_spots` (0.73), `high brown_index` (0.17). Se lee como lo diría un patólogo: la
roya es **saturada y anaranjada** (hue/saturación altos), con **muchas** pústulas
(n_spots alto) de color lesión (brown_index). La explicación no es un adorno: es la
misma combinación de color + geometría con la que se definió la clase.
