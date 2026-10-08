# 04 — Evaluación honesta

## El autoengaño que evitamos

Con datos sintéticos, un split **aleatorio** (entrenamiento y test de la misma
distribución) da un número que **no significa nada**: hasta un modelo que solo
memorizara el aspecto sintético fijo saldría alto. Es la trampa de las demos de
"99% de precisión" que no sirven en campo.

Pero bajar el número **lisiando las clases** (que la roya y el tizón se confundan
a propósito) es deshonesto en el otro sentido: la roya real (pústulas pequeñas
anaranjadas) y el tizón real (manchas grandes marrones) **sí** se distinguen. Un
benchmark que finge que no, engaña igual que uno trivialmente fácil.

## Qué hacemos en su lugar

Modelamos las clases **con fidelidad** y hacemos honesta la **evaluación** por
método, no por ambigüedad artificial:

1. **Clases fieles a la realidad:** cada enfermedad con su color y geometría reales
   (roya = muchas pústulas pequeñas anaranjadas; tizón = pocas manchas grandes
   marrones; clorosis = amarilleo global que, en severidad baja, sí tiende a sana —
   una frontera suave real que conservamos). Así son separables, como en el campo.
2. **Partir por parámetro, no al azar.** Cada mando del generador (tamaño y número
   de manchas, severidad, iluminación) se divide en **bandas intercaladas**; se
   entrena con unas y se prueba con las **disjuntas**. Ninguna hoja de test tiene
   un tamaño, número, severidad o iluminación visto en entrenamiento. Y
   promediamos varias semillas para que el titular sea robusto, no un split con
   suerte.

## El número (medido, no inventado)

```
accuracy (media ± std, 5 seeds): 0.99 ± 0.01   # 4 clases; el azar es ~0.25
macro-F1 (media, 5 seeds)       : 0.99
train accuracy                  : 1.00
train-test gap                  : 0.01          # ~0 -> generaliza de verdad, no memoriza
```

## Lo que el número sí dice y lo que no dice

- **Sí** generaliza a hojas que no ha visto: la honestidad **no** es un tope de
  precisión, es la **brecha train-test casi nula sobre parámetros no vistos**. Un
  número alto con brecha ~0.01 es generalización real; un número alto con brecha
  grande sería memorización. El problema es alto porque las clases son de verdad
  separables.
- **No** dice que esté listo para el campo. Es un **benchmark sintético**. Las
  hojas reales traen fondos complejos, oclusión, enfermedades mezcladas y
  variación de cámara que ningún renderizador captura — ahí el problema es de
  verdad más difícil. En producción se entrena una **CNN con fotos reales
  etiquetadas**, midiendo contra ese test real; este número held-out es la base a
  superar, no el techo.

## Dónde caen los errores

La matriz de confusión lo confirma: las cuatro clases salen casi perfectas sobre
parámetros no vistos; el poco error que queda está en algún caso límite de tamaño
de lesión no visto en entrenamiento, no en una confusión sistemática de clases.
