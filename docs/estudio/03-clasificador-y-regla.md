# 03 — El clasificador y la regla determinista

## Qué hace

Recibe el vector de features de una hoja y predice la clase de salud (`healthy`,
`chlorosis`, `blight`, `rust`) con una **confianza** y una **explicación**. Es un
`StandardScaler` seguido de una **regresión logística multinomial**. Al ser
lineal, cada veredicto viene con los features que más pesaron: un agrónomo ve
*por qué* y puede corregir.

## La explicación por instancia

Para la clase predicha, la contribución de cada feature es `valor_estandarizado ×
peso`. Mostramos los de mayor contribución positiva, y les ponemos el prefijo
`high`/`low` según si el valor estaba por encima o por debajo de la media. Así se
lee bien incluso cuando lo que apoya la clase es un valor **bajo** (por ejemplo,
`low spot_coverage` apoya `healthy`).

## La regla determinista (el atajo del caso claro)

Antes del modelo hay una regla para la hoja **obviamente sana**:

```
si green_fraction >= AV_RULE_GREEN_FRACTION  (por defecto 0.47)
   y spot_coverage <= AV_RULE_MAX_SPOT_COVERAGE  (por defecto 0.02)
→ veredicto: healthy (por regla, confianza 1.0)
```

¿Por qué una regla y no dejar que decida el modelo? Porque el caso evidente no
debería depender de un vector de pesos que puede derivar: una hoja muy verde y
sin lesiones es sana por definición operativa, y eso es más robusto y más
explicable como **regla declarada**. Los umbrales tienen margen: ninguna clase
enferma del benchmark alcanza esa fracción de verde con cobertura de lesión casi
nula, así que la regla no se dispara por error (lo verificamos).

## Cuando el modelo no está seguro → revisión

Si la confianza del modelo está por debajo de `AV_CLASSIFIER_MIN_CONFIDENCE`, el
veredicto se marca `needs_review`: se devuelve para que lo confirme una persona
en vez de afirmarlo con falsa seguridad. En una herramienta que aconseja sobre
tratar un cultivo, esa es la conducta correcta.

## La recomendación

Cada clase lleva una acción en lenguaje claro, por ejemplo: *"rust detected →
consider a targeted fungicide and improve air circulation"*. El objetivo es que
la salida sea accionable, no solo una etiqueta.
