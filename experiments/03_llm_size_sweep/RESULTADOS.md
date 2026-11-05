# Resultados del barrido (5 nov 2026)

Mismo prompt (`prompt_bearing.txt`) y mismo evaluador (datos reales del CWRU, 947 ventanas) para todos los modelos. Los GGUF descargados coinciden con el SHA-256 que publica Hugging Face (ver `results/*/meta.json`). Se generaron 3 respuestas por modelo con temperatura 0,7.

| Modelo | Respuestas que se ejecutan | Calculan frecuencias de fallo | Exactitud equilibrada |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0 / 3 | 0 / 3 | — |
| Qwen2.5-1.5B-Instruct | 0 / 3 | 0 / 3 | — |
| Qwen2.5-3B-Instruct | 0 / 3 | 0 / 3 | — |
| Qwen2.5-7B-Instruct | 0 / 3 | 0 / 3 | — |
| Qwen2.5-14B-Instruct | 0 / 3 | 1 / 3, con fórmulas incorrectas | — |
| Claude (red v1, escrita en la sesión) | sí | sí, deducidas de la geometría | 58,5% (azar 25%) |

Errores en `EVALUACION.txt`. Los fallos son de dos tipos:

1. **El código no funciona.** Hay formas de tensor incompatibles, variables sin definir, respuestas cortadas y una red de 786 millones de parámetros que agota la memoria.
2. **No hay conocimiento compilado.** Incluso cuando el modelo de 14B nombra BPFO, BPFI y BSF, las fórmulas son incorrectas y los "pesos" son marcadores de posición. En el resto de respuestas, las capas se dejan con su inicialización aleatoria.

Límite de esta comparación: Claude no recibió el prompt en frío. Escribió su red con el contexto de la sesión y con herramientas para probar el código, mientras que los Qwen respondieron de una sola vez y sin herramientas.
