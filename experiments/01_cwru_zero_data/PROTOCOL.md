# Protocolo — Text-to-Weights: diagnóstico de rodamientos sin datos de entrenamiento

Escrito y congelado **antes** de mirar ninguna señal del dataset.

## Datos de evaluación (solo para evaluar)
- CWRU Bearing Data Center, copia en NumPy (github.com/srigas/CWRU_Bearing_NumPy).
- Canal: acelerómetro del lado de accionamiento (DE), 48 kHz.
- Ficheros: todos los `*_DE48.npz` y `*_Normal.npz` de las 4 velocidades (1730, 1750, 1772, 1797 rpm): 56 ficheros.
- Clases (4): Normal, IR (pista interior), B (bola), OR (pista exterior, en cualquier posición @3/@6/@12).
- Ventanas: 24 000 muestras (0,5 s), sin solapamiento, desde el inicio de cada fichero.

## Lo único que la red puede saber (metadatos de ingeniería, no datos)
- Frecuencia de muestreo (48 kHz).
- Velocidad nominal del eje (rpm del nombre del fichero; en planta la da un tacómetro).
- Geometría del rodamiento SKF 6205-2RS: 9 bolas, diámetro de bola 0,3126 in, diámetro primitivo 1,537 in, ángulo de contacto 0°.
- Conocimiento general de física de rodamientos (análisis de envolvente).

## Reglas
1. Ningún peso se ajusta con datos. Todos salen de fórmulas escritas por el LLM a partir de lo anterior.
2. Los hiperparámetros (bandas, umbral, tolerancias) se fijan antes de evaluar y quedan en el código.
3. Se permiten pruebas unitarias con señales **sintéticas generadas por el propio código**, nunca con el dataset.
4. Los pesos se guardan y se registra su SHA-256 en git **antes** de la evaluación.
5. Una única evaluación. Si después se cambia algo, se reporta como versión nueva junto a la primera, nunca en su lugar.

## Métricas
- Exactitud equilibrada (media de la exactitud por clase) en 4 clases. Azar = 25%.
- Detección binaria sano/averiado (exactitud equilibrada).
- Matriz de confusión y desglose por tamaño de fallo y velocidad.

## Comparaciones (después de la evaluación ciega)
- Azar.
- CNN 1D entrenada con datos, validación cruzada por velocidad (entrena con 3, prueba con la 4ª).
- La misma CNN con pocos ejemplos por clase (1, 5, 20 ventanas).
