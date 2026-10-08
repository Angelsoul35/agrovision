# 04 — Evaluación honesta

## El autoengaño que evitamos

Con cuatro clases codificadas por color, un split **aleatorio** da ~1.0 de
precisión. Ese número **no significa nada**: solo dice que entrenamiento y test
salen de la misma distribución. Reportarlo sería el mismo autoengaño de las demos
de "99% de precisión" que no sirven en campo. De hecho, una primera versión de
este proyecto daba ~0.99 con un split interpolado; esa fue la señal de que la
tarea era demasiado fácil y el número, poco fiable.

## Qué hacemos en su lugar

Dos cosas, juntas:

1. **Hacer que las clases se solapen** como la enfermedad real (clorosis suave ≈
   sana; roya grande ≈ tizón; color de lesión en un rango compartido). Esto crea
   ambigüedad irreducible, así que incluso dentro de distribución la precisión es
   ~0.95, no 1.0.
2. **Partir por parámetro, no al azar.** Cada mando del generador (tamaño y número
   de manchas, severidad, iluminación) se divide en **bandas intercaladas**; se
   entrena con unas y se prueba con las **disjuntas**. Ninguna hoja de test tiene
   un tamaño, número, severidad o iluminación visto en entrenamiento. Y
   promediamos varias semillas para que el titular sea robusto, no un split con
   suerte.

## El número (medido, no inventado)

```
accuracy (media ± std, 5 seeds): 0.93 ± 0.02   # 4 clases; el azar es ~0.25
macro-F1 (media, 5 seeds)       : 0.93
train accuracy                  : 0.95
train-test gap                  : 0.01          # acotada -> aprende features, no un aspecto fijo
```

## Lo que el número sí dice y lo que no dice

- **Sí** dice que el modelo generaliza a hojas que no ha visto apoyándose en los
  features de color/textura: la precisión en entrenamiento es **0.95** (no 1.0,
  porque las clases se solapan de verdad) y la brecha train-test es casi nula.
- **No** dice que esté listo para el campo. Es un **benchmark sintético**. Las
  hojas reales traen fondos complejos, oclusión, enfermedades mezcladas y
  variación de cámara que ningún renderizador captura. Los errores que quedan son
  los honestos: la frontera **roya/tizón**, donde el color y la geometría se
  confunden. Cerrar esa brecha pide una **CNN entrenada con fotos reales
  etiquetadas**, midiendo contra ese test real; este número held-out es la base a
  superar, no el techo.

## Dónde caen los errores

La matriz de confusión lo confirma: sana y clorosis salen casi perfectas, y casi
todo el error está en roya↔tizón. Eso es exactamente el solapamiento que
diseñamos a propósito, no un fallo aleatorio del modelo.
