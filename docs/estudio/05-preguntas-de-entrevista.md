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
El sintético da algo valioso: como controlo los parámetros, reservo rangos enteros
para una evaluación sin fuga (ADR 0002). Pero no me quedo ahí: el mismo extractor de
rasgos se mide también sobre **fotos reales de campo** (dataset Makerere Beans, 180
imágenes vendoreadas), con `python -m agrovision.real_benchmark`. Así reporto el
número controlado y el número real.

### "Tu clasificador, ¿no está sobreajustado? ¿No es un 1.0 sospechoso?"
Ya no da 1.0: da **0.90 ± 0.03**, porque las clases **se solapan a propósito como en
el campo** (roya y tizón comparten banda de color y geometría; una lesión temprana es
tenue y se parece a una hoja sana). Y lo que prueba que **tampoco** sobreajusta es la
**brecha train-test ~0** sobre un **split disjunto por parámetros** (tamaños,
cantidades, severidad, iluminación nunca vistos), promediado en varias semillas. No es
que lisie las clases: sana y clorosis siguen casi perfectas; la confusión es roya↔tizón,
donde un agrónomo también duda. Si quito los confundidores, vuelve a 1.0 — y un test de
mutación falla si eso pasa. Sobre hojas reales (Beans) los rasgos a mano llegan a ~0.67
(azar 0.33): ese es el techo de los rasgos interpretables, y el argumento para una CNN.

### "¿Qué pasa cuando el modelo no está seguro?"
Si la confianza baja del umbral, el veredicto va a **revisión humana** en vez de
afirmarse. Y el caso obvio (hoja muy verde, sin lesiones) lo resuelve una **regla
determinista**, no el modelo. Es la conducta correcta para una herramienta que
aconseja tratar un cultivo.

### "¿Cómo distingues roya de tizón si las dos son manchas marrones?"
Por **geometría**, no solo por color: uso `scipy.ndimage.label` para contar
componentes y medir su tamaño. La roya son **muchas manchas pequeñas**; el tizón,
**pocas grandes**. Como el color lo hago solaparse a propósito, el modelo tiene
que apoyarse en esa geometría. Ahí es donde cae casi todo el error, y es un error
riguroso: esa frontera es genuinamente ambigua.

### "¿Cómo segmentas la hoja del fondo sin máscara?"
Por **saturación**: el fondo es neutro (poco saturado) y la hoja no. Además, la
iluminación del generador es multiplicativa, y la saturación es invariante a
multiplicar los canales, así que la segmentación aguanta la luz.

### "¿Qué harías para llevarlo a producción?"
Entrenar y evaluar con **fotos reales de campo etiquetadas** usando una CNN;
añadir segmentación real de hoja para escenas con fondo; manejar **varias
enfermedades a la vez** y grados de severidad; y medir contra ese test real,
tomando este score held-out como la base a superar, no como el techo.
