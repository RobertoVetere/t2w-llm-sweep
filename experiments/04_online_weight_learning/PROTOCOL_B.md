# Protocolo 04-B: el LLM "mete la mano del todo" (sin entrenar)

Congelado antes del nuevo análisis del feedback y antes de cargar ninguna señal de la prueba limpia.

## Qué cambia respecto a la fase A
- **Acceso total del LLM a los 20 casos de feedback** (los mismos de la fase A): señales en crudo y cualquier análisis que programe (curtosis espectral, cepstrum, estadísticos temporales, espectros en bandas finas, etc.). Puede reescribir la red entera.
- **Sigue prohibido entrenar:** nada de descenso de gradiente, optimizadores, mínimos cuadrados, búsquedas automáticas (rejilla, aleatorias) que maximicen el acierto en el feedback, ni copiar ejemplos o promedios de ejemplos a los pesos. Los valores los elige el LLM razonando sobre lo que ve, y lo deja escrito en el código.
- Máximo 2 iteraciones nuevas (iter3, iter4), cada una con su commit.

## Por qué hay una prueba nueva
En la fase A ya se miró el acierto de iter2 en la prueba 1 (1772/1797 rpm, 48 kHz). Para que la fase B sea limpia se añade la **prueba 2**, que nadie ha mirado:
- Grabaciones **distintas** de la misma máquina a 1772 y 1797 rpm: lado de accionamiento, adquiridas a **12 kHz**, con fallos de 7, 14, 21 y **28 milésimas** (el tamaño de 28 no aparece en el feedback).
- Sanas: las grabaciones sanas de 1772/1797 rpm, pasadas por la misma cadena que el resto.
- Cadena común para todas: filtro y diezmado a 12 kHz y vuelta a 48 kHz (`resample_poly`), de modo que todas las clases tienen el mismo ancho de banda (hasta 6 kHz) y entran a las redes en su formato original.
- Ventanas de 0,5 s sin solapamiento.

Se reporta todo en las dos pruebas: v1, iter2, iter3/iter4 y los métodos por gradiente entrenados con los mismos 20 casos (ajuste de la red escrita, corrección lineal sola y CNN, con 5 semillas).
