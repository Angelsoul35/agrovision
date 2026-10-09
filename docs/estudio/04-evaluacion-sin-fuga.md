# 04 — Evaluación sin fuga

## El autoengaño que evitamos

Con datos sintéticos, un split **aleatorio** (entrenamiento y test de la misma
distribución) da un número que **no significa nada**: hasta un modelo que solo
memorizara el aspecto sintético fijo saldría alto. Es la trampa de las demos de
"99% de precisión" que no sirven en campo.

Pero el extremo opuesto también engaña: **lisiar las clases** —meter ruido o un
solapamiento artificial que borre la señal— baja el número sin que signifique nada.
La frontera es sutil, así que la cuidamos: el solapamiento que añadimos es el que
**existe de verdad en el campo**, no uno inventado.

## Dos cosas hacen el número realista (y son distintas)

1. **Las clases se solapan como en la realidad.** La roya (pústulas anaranjadas) y
   el tizón (manchas marrones) comparten una banda de color ancha, sus tamaños y
   conteos de lesión se cruzan, y **cada enfermedad tiene una severidad**: una
   lesión temprana es tenue y escasa, cercana a una hoja sana. Por eso una parte de
   las hojas cae en una zona genuinamente ambigua. Lo comprobamos: la confusión que
   queda es **roya↔tizón** (y temprano→sana), exactamente donde un agrónomo también
   duda; en cambio sana y clorosis, que no tienen esa ambigüedad, salen casi
   perfectas. No es borrar la señal: es reproducir la dificultad real.
2. **Partir por parámetro, no al azar.** Cada mando del generador (tamaño y número
   de manchas, severidad, iluminación) se divide en **bandas intercaladas**; se
   entrena con unas y se prueba con las **disjuntas**. Ninguna hoja de test tiene un
   tamaño, número, severidad o iluminación visto en entrenamiento. Se promedian
   varias semillas para que el titular sea robusto, no un split con suerte.

Lo primero hace el problema difícil; lo segundo hace la medición confiable.

## El número (medido, no inventado)

```
accuracy (media ± std, 5 seeds): 0.90 ± 0.03   # 4 clases; el azar es ~0.25
macro-F1 (media, 5 seeds)       : 0.89
train accuracy                  : 0.86
train-test gap                  : -0.04         # ~0 -> aprende rasgos, no memoriza
```

El ~0.90 **no es 1.0** (por el solapamiento) y la brecha train-test ~0 dice que
**tampoco sobreajusta**. Si quitas los confundidores, la accuracy salta de vuelta a
1.0 — y hay un test de regresión que **falla** si eso pasa (prueba de mutación).

## Validación con imágenes reales (Beans)

El sintético da una medición controlada de 4 clases; el dataset real **Makerere
Beans** (180 fotos de campo, vendoreadas en `datasets/beans/`) mide los **mismos
rasgos** sobre hojas reales. Con `python -m agrovision.real_benchmark`:

```
modelo                    CV accuracy       train   test    gap
Logística (regularizada)  0.672 ± 0.051     0.733   0.711   0.022
Random forest             0.667 ± 0.046     1.000   0.667   0.333   # memoriza, sin ganancia
```

Sobre hojas reales los rasgos a mano dan un clasificador **usable y totalmente
explicable**: ~0.67 (3 clases, azar 0.33) con un logístico bien ajustado, y cada
veredicto sigue nombrando los rasgos que lo motivaron. Enriquecer a 66 rasgos
(histogramas HSV, LBP, textura GLCM) y un SVM-RBF solo llega a ~0.72, así que los
rasgos simples y legibles no dejan mucho sobre la mesa. Una CNN marcaría más alto
aprendiendo sus propios rasgos, pero pierde esa explicación por rasgo que un agrónomo
revisa y puede revocar — es el **trade-off de interpretabilidad** que este proyecto
elige a propósito ([ADR-0001](../adr/0001-classical-cv-not-pretrained-cnn.md)), no un
veredicto de que el pipeline no sirva. El random forest, sin regularizar, memoriza
las 180 imágenes (train ~1.0) sin ganar nada en validación cruzada: la misma lección
de "más capacidad no es mejor".

## Dónde caen los errores

- **Sintético:** la confusión es roya↔tizón y temprano→sana, no un error al azar.
- **Real:** la roya se detecta mejor (el naranja es distintivo); roya/tizón se
  confunden entre sí y algo con sana, como esperan los rasgos de color/textura.
