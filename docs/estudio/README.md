# Guía de estudio — AgroVision

Una guía en español para entender el proyecto a fondo y poder defenderlo en una
entrevista. El código, los comentarios y la documentación técnica están en
inglés (como en un entorno profesional real); esta guía explica el *porqué* de
cada decisión.

## Orden sugerido

1. [01 — Visión por computador clásica](01-vision-por-computador-clasica.md)
   Qué features de color y textura usamos y por qué, sin redes neuronales.
2. [02 — Generación sintética de hojas](02-generacion-sintetica.md)
   Cómo se dibujan las hojas y las cuatro clases de salud.
3. [03 — El clasificador y la regla determinista](03-clasificador-y-regla.md)
   Cómo se decide la clase, cómo se explica y cuándo manda la regla.
4. [04 — Evaluación honesta](04-evaluacion-honesta.md)
   Por qué NO reportamos un 1.0 y qué significa de verdad el número.
5. [05 — Preguntas de entrevista](05-preguntas-de-entrevista.md)
   Preguntas típicas y cómo responderlas con este proyecto.

## La idea en una frase

Un detector de salud de hojas **totalmente offline y determinista**, hecho con
**visión por computador clásica** (color + textura, sin modelos preentrenados),
que clasifica una hoja en sana, clorosis, tizón (*blight*) o roya (*rust*), da una
explicación y una recomendación, y se evalúa con **honestidad** reservando rangos
de parámetros nunca vistos en el entrenamiento.
