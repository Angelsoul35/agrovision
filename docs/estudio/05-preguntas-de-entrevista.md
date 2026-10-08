# 05 — Preguntas de entrevista

Preguntas típicas sobre este proyecto y cómo responderlas.

### "¿Por qué visión clásica y no una CNN preentrenada?"
Por la restricción (offline, determinista, sin descargas) y por el objetivo de
portafolio: demostrar que entiendo el problema, no importar los pesos de otro.
Una CNN batiría a estos features en hojas reales, y lo digo abiertamente; pero la
visión clásica hace cada decisión explícita y explicable (segmentación por regla,
features con significado, modelo lineal). La CNN sobre datos reales es el camino a
producción, lo dejo escrito (ADR 0001).

### "¿No es un problema que los datos sean sintéticos?"
Sí, y es la limitación principal; la reconozco en cada sitio donde doy un número.
Pero también me da algo valioso: como controlo los parámetros de generación, puedo
reservar rangos enteros para una evaluación honesta de generalización, cosa que un
dataset fijo no regala (ADR 0002).

### "Tu clasificador, ¿no está sobreajustado? ¿No es un 1.0 sospechoso?"
Da **0.99 ± 0.01**, y lo que lo hace fiable no es un tope artificial sino la
**brecha train-test de ~0.01 sobre parámetros no vistos**. Evalúo con un **split
disjunto por parámetros** (tamaños, cantidades, severidad e iluminación que nunca
aparecen en entrenamiento) y promedio varias semillas. El número es alto porque
las cuatro clases son de verdad separables (roya, tizón, clorosis y hoja sana se
distinguen en el campo); un 0.99 **con brecha casi nula** es generalización real,
mientras que un 0.99 con brecha grande sería memorización. No bajé el número
lisiando las clases a propósito: eso engañaría igual que un benchmark trivial.

### "¿Qué pasa cuando el modelo no está seguro?"
Si la confianza baja del umbral, el veredicto va a **revisión humana** en vez de
afirmarse. Y el caso obvio (hoja muy verde, sin lesiones) lo resuelve una **regla
determinista**, no el modelo. Es la conducta honesta para una herramienta que
aconseja tratar un cultivo.

### "¿Cómo distingues roya de tizón si las dos son manchas marrones?"
Por **geometría**, no solo por color: uso `scipy.ndimage.label` para contar
componentes y medir su tamaño. La roya son **muchas manchas pequeñas**; el tizón,
**pocas grandes**. Como el color lo hago solaparse a propósito, el modelo tiene
que apoyarse en esa geometría. Ahí es donde cae casi todo el error, y es un error
honesto: esa frontera es genuinamente ambigua.

### "¿Cómo segmentas la hoja del fondo sin máscara?"
Por **saturación**: el fondo es neutro (poco saturado) y la hoja no. Además, la
iluminación del generador es multiplicativa, y la saturación es invariante a
multiplicar los canales, así que la segmentación aguanta la luz.

### "¿Qué harías para llevarlo a producción?"
Entrenar y evaluar con **fotos reales de campo etiquetadas** usando una CNN;
añadir segmentación real de hoja para escenas con fondo; manejar **varias
enfermedades a la vez** y grados de severidad; y medir contra ese test real,
tomando este score held-out como la base a superar, no como el techo.
