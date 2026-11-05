# Protocolo 04: aprendizaje sobre la marcha editando pesos con un LLM (sin gradientes)

Congelado antes de generar ningún informe de fallos y antes de mirar ningún caso del conjunto de feedback.

## Pregunta
Una red ya desplegada carece de una capacidad: la red v1 de rodamientos casi nunca detecta los fallos de bola. Si llegan unos pocos casos etiquetados, ¿puede un LLM (Claude Opus 5.5) **instalar esa capacidad editando los pesos directamente**, en 1–2 iteraciones y sin descenso de gradiente? ¿Cómo se compara con aprender esos mismos casos por gradiente?

## Escenario
- **Modelo desplegado:** red v1 escrita a mano (`../01_cwru_zero_data/handnet.py`, SHA-256 `4d536ec1…5b24d`).
- **Feedback (lo único de lo que se aprende):** 5 ventanas por clase (20 en total), elegidas con semilla 0 entre las velocidades **1730 y 1750 rpm**.
- **Prueba (no se mira hasta el final):** todas las ventanas de **1772 y 1797 rpm**, es decir, otras condiciones de operación.

## Bucle (máximo 2 iteraciones)
1. `report.py` genera automáticamente un informe de fallos de la red actual sobre el feedback: matriz de confusión, puntuaciones internas por clase y banda, y picos del espectro de envolvente expresados en órdenes del eje. El informe se guarda en `reports/`.
2. El LLM lee el informe y escribe la edición como código (`edits/iterN.py`), razonando cada cambio por escrito en el propio archivo.
3. Cada iteración es un commit.

## Reglas para el LLM
- Prohibido: descenso de gradiente, optimizadores, búsquedas automáticas de parámetros que evalúen el acierto sobre el feedback, y copiar ejemplos a los pesos (memorizar ventanas).
- Permitido: razonar sobre el informe y escribir pesos o reglas nuevas justificados por la física del rodamiento.
- El conjunto de prueba solo se evalúa al final, sobre v1, iter1 e iter2 a la vez.

## Comparación (mismos 20 casos de feedback, misma prueba)
- Ajuste por gradiente de la red escrita (cabeza + corrección lineal, como en el experimento 01), 5 semillas.
- Corrección lineal sola sobre el mismo frente espectral (sin la red escrita), 5 semillas.
- CNN 1D desde cero sobre la señal, 5 semillas.
- Robustez: los métodos por gradiente se repiten también con 5 sorteos distintos del feedback.

## Métricas
Exactitud equilibrada (4 clases), acierto por clase (sobre todo Bola), detección sano/averiado, falsas alarmas y fracción de pesos modificados.
