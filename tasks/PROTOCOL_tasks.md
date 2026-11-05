# Protocolo: batería de tareas (punto 2) y punto de partida (réplica del punto 1)

Escrito y congelado antes de cargar ninguna muestra de estos datasets. Las redes
`digits_net.py`, `har_net.py` y `fsdd_net.py` se escribieron solo con la documentación
de cada dataset y conocimiento general.

## Tareas y particiones (fijadas a priori)
| Tarea | Modalidad | Clases | Prueba | Fuente de ejemplos para ajustar |
|---|---|---|---|---|
| Rodamientos CWRU | vibración | 4 | dejando fuera una velocidad | las otras 3 velocidades |
| Dígitos 8x8 (sklearn) | visión | 10 | muestras 900..1796 | muestras 0..899 |
| UCI HAR (561 rasgos) | sensores del móvil | 6 | conjunto de test oficial (otras personas) | conjunto de entrenamiento oficial |
| FSDD | audio | 10 | grabaciones con índice 0..4 (convención del dataset) | índices 5..49 |

## Evaluación
1. Cero ejemplos: la red escrita a mano tal cual, una sola vez. Métrica: exactitud equilibrada.
2. Con k = 1, 5 y 20 ejemplos por clase, 5 semillas, mismas muestras para todas las condiciones:
   - **Escrita + ajuste**: se parte de los pesos escritos y se ajustan todos con Adam
     (lr 3e-3, 200 pasos) y una penalización L2 hacia los pesos escritos (λ = 1e-3).
   - **Misma arquitectura, inicio aleatorio**: mismo entrenamiento sin penalización.
   - **Regresión logística**: sobre la entrada (dígitos, HAR) o sobre los rasgos del frente
     escrito a mano (FSDD), como referencia estándar.
3. Nada se ajusta mirando el conjunto de prueba. Cualquier cambio posterior se reporta como nueva versión.
