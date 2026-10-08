# 01 — Visión por computador clásica

## Por qué sin redes neuronales

Lo "fácil" sería coger una red convolucional (CNN) preentrenada y afinarla. Pero
el proyecto tiene una restricción dura: **sin descargas, sin red, determinista**.
Y, sobre todo, un objetivo de portafolio: demostrar que entiendo el problema, no
importar el conocimiento de otro. Una CNN preentrenada haría que la precisión
fuera una propiedad del dataset ajeno, no de mi trabajo. Por eso usamos **visión
clásica**: cada paso es una decisión explícita y explicable.

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
