# Resultados v1: evaluación ciega única

Pesos congelados en el commit `66cde38` (SHA-256 `4d536ec1…5b24d`) antes de cargar ninguna señal del dataset.

## Red escrita a mano (0 ejemplos de entrenamiento)
- 3,6 M de pesos por red (una por velocidad nominal; la misma fórmula).
- 947 ventanas de 0,5 s, CWRU, lado de accionamiento, 48 kHz.

| Métrica | Valor |
|---|---|
| Exactitud equilibrada (4 clases; azar 25%) | **58,5%** |
| Sano vs averiado (equilibrada) | **80,8%** |
| Falsas alarmas en máquinas sanas | **0 de 70** |

Matriz de confusión (filas = real; columnas = predicho Normal/IR/B/OR):

| | Normal | IR | B | OR |
|---|---|---|---|---|
| Normal | 70 | 0 | 0 | 0 |
| IR | 101 | 96 | 0 | 0 |
| B | 165 | 0 | 1 | 44 |
| OR | 71 | 0 | 0 | 399 |

- OR: 85%. IR: 49% (100% con fallos de 7 milésimas, 0% con los de 14). Bola: prácticamente 0%.
- Los fallos de 14 milésimas casi nunca se detectan (2,5%).
- Por velocidad: entre 55% y 69%.

## CNN 1D entrenada (dejando fuera una velocidad, 3 semillas)

| Ejemplos de entrenamiento por clase | Exactitud equilibrada |
|---|---|
| 0 (red escrita a mano) | 58,5% |
| 1 | 56,9% ± 7,9 |
| 5 | 77,0% ± 2,1 |
| 20 | 87,1% ± 2,0 |
| todos (~700 ventanas) | 87,7% ± 1,4 |

## Lectura honesta
- Sin ningún dato, la red escrita a mano rinde como una CNN entrenada con 1 ejemplo por clase, y no da ninguna falsa alarma.
- Con 5 ejemplos por clase, la CNN ya la supera.
- Los fallos de bola y de 14 milésimas son conocidos por ser difíciles en CWRU con análisis de envolvente (Smith y Randall, MSSP 2015). Además, las ventanas de bola puntúan más alto en OR que en bola, algo que esa literatura también describe. El análisis posterior (puntuaciones medias por clase) está en `eval_outputs.npz`.
- No hubo ajuste posterior. Cualquier v2 se publicará junto a esta v1.
