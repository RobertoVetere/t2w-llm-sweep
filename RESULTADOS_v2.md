# Resultados: tres puntos (5 nov 2026)

Exactitud equilibrada. Media de 5 semillas en los casos con pocos ejemplos. Las redes escritas a mano se congelaron en git antes de cargar los datos de cada tarea.

## Punto 2: la red escrita a mano sin ningún ejemplo

| Tarea | Modalidad | Clases | Azar | Red escrita, 0 ejemplos |
|---|---|---|---|---|
| Rodamientos CWRU | vibración | 4 | 25% | **58,5%** |
| Dígitos 8x8 | visión | 10 | 10% | **60,1%** |
| UCI HAR | sensores del móvil | 6 | 16,7% | **60,5%** |
| FSDD | voz | 10 | 10% | **47,0%** |

En las cuatro tareas, la red supera claramente al azar (entre 2,3 y 6 veces), sin haber visto ningún dato.

## Puntos 1 y 2: ¿sirve como punto de partida con pocos ejemplos?

Esquema: red escrita congelada + corrección lineal que empieza en cero. Control: la misma corrección lineal sin la red escrita, con el mismo optimizador.

| Tarea | k=1: escrita / control | k=5 | k=20 |
|---|---|---|---|
| Rodamientos (velocidad no vista)* | **80,4 / 54,3** | **93,0 / 82,4** | **94,6 / 84,4** |
| Dígitos | **65,8 / 58,5** | 81,0 / 80,6 | 86,6 / 88,5 |
| HAR (personas no vistas) | **63,4 / 55,3** | 75,0 / 73,5 | 84,3 / 83,3 |
| FSDD | **51,3 / 39,7** | 68,9 / 67,5 | 79,5 / 82,0 |

\* En rodamientos se ajustó también la cabeza de la red escrita (esquema del punto 1).

**Patrón:**
- Con 1 ejemplo por clase, la red escrita aporta entre 7 y 26 puntos en las cuatro tareas.
- Con 5 o 20 ejemplos, la ventaja desaparece en las tareas sin cambio de dominio.
- En rodamientos se mantiene con todos los datos (+8 a +10 puntos). Ahí la prueba es en una velocidad que no aparece en el ajuste, y el conocimiento físico (las frecuencias de fallo escaladas con la velocidad) generaliza donde los datos no llegan.

**Equivalencia aproximada:** la descripción escrita vale entre 1 y 3 ejemplos etiquetados por clase. Su acierto con 0 ejemplos iguala o supera al de una regresión logística con 1 ejemplo por clase: 60 frente a 61 en dígitos, 61 frente a 58 en HAR, 47 frente a 39 en FSDD, 58 frente a 57 de la CNN en rodamientos.

## Lo que salió mal (se reporta)
- **v1 del ajuste en las tareas nuevas:** ajustar la red escrita entera, sin la corrección lineal, ayuda en FSDD (k=1: 44,6 frente a 22,0 con inicio aleatorio). Pero en dígitos y HAR queda por debajo de la regresión logística con k≥5, porque la red escrita es un cuello de botella (la de HAR usa solo 3 rasgos). El esquema con corrección lineal (v2) se escribió **después** de ver la v1.
- Con k≥5, una regresión logística estándar (sklearn) iguala o supera a los dos esquemas con corrección lineal.

## Límites
- Un solo LLM (yo). Las cuatro tareas son datasets públicos conocidos, que probablemente vi durante mi entrenamiento.
- Los umbrales de la red de HAR son conjeturas a priori.
- Falta un segundo dataset de rodamientos de otro laboratorio.
- Punto 3 pendiente: el kit está en `llm_sweep/` (prompt, evaluador e instrucciones).
