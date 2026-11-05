# Informe de fallos: red `v1` sobre el feedback (20 ventanas, 1730/1750 rpm)

Exactitud equilibrada en feedback: 0.55

## Matriz de confusión (filas = real, columnas = predicho)

| | Normal | IR | B | OR |
|---|---|---|---|---|
| Normal | 5 | 0 | 0 | 0 |
| IR | 3 | 2 | 0 | 0 |
| B | 4 | 0 | 0 | 1 |
| OR | 1 | 0 | 0 | 4 |

Órdenes de referencia (múltiplos de la velocidad del eje): BPFO=3.585, 2xBPFO=7.170, BPFI=5.415, BSF=2.357, 2xBSF=4.713, FTF=0.398, 1x eje=1.000, 2x eje=2.000


## Clase real: Normal (5 ventanas; tamaños de fallo: [0])
| Banda (Hz) | máx z medio | curtosis señal filtrada* | z en BPFO / 2xBPFO / BPFI / BSF / 2xBSF / FTF / 1x eje / 2x eje |
|---|---|---|---|
| 1000-3000 | 1.2 | – | 0.1 / 0.3 / 0.1 / 0.1 / 0.0 / 0.1 / 1.0 / 0.9 |
| 3000-6000 | 3.7 | – | 0.1 / 0.0 / -0.1 / 0.0 / 0.1 / -0.0 / 0.2 / 0.4 |
| 6000-10000 | 1.3 | – | -0.0 / 0.0 / 0.0 / 0.0 / -0.0 / -0.0 / 0.1 / 0.1 |
| 10000-16000 | 0.1 | – | 0.0 / -0.0 / -0.0 / -0.0 / -0.0 / -0.0 / 0.1 / 0.0 |
| 16000-22000 | 0.1 | – | 0.0 / 0.0 / 0.0 / 0.0 / 0.0 / 0.0 / 0.0 / 0.0 |

Picos principales (banda 3000-6000 Hz), orden: z = 0.24: 0.2, 1.00: 0.2, 1.52: 0.2, 2.02: 0.4, 2.80: 0.2, 3.96: 3.5, 4.14: 0.4, 4.82: 0.2, 6.86: 0.2, 7.92: 1.6
Ventanas con pico z>3 en alguna banda: BPFO 0%, 2xBPFO 0%, BPFI 0%, BSF 0%, 2xBSF 0%, FTF 0%, 1x eje 0%, 2x eje 0%

## Clase real: IR (5 ventanas; tamaños de fallo: [7, 14, 21])
| Banda (Hz) | máx z medio | curtosis señal filtrada* | z en BPFO / 2xBPFO / BPFI / BSF / 2xBSF / FTF / 1x eje / 2x eje |
|---|---|---|---|
| 1000-3000 | 7.9 | – | 0.3 / -0.2 / 5.7 / 2.6 / 0.0 / 1.3 / 4.1 / 4.2 |
| 3000-6000 | 6.9 | – | -0.0 / -0.1 / 5.8 / 2.4 / -0.2 / 0.2 / 3.7 / 2.5 |
| 6000-10000 | 1.6 | – | 0.0 / -0.1 / 1.5 / 0.2 / 0.0 / -0.0 / 0.2 / 0.1 |
| 10000-16000 | 0.2 | – | -0.0 / -0.0 / 0.1 / 0.0 / -0.0 / 0.0 / 0.1 / 0.1 |
| 16000-22000 | 0.2 | – | 0.0 / -0.0 / 0.1 / 0.1 / 0.0 / 0.0 / 0.1 / 0.1 |

Picos principales (banda 1000-3000 Hz), orden: z = 1.00: 4.1, 1.40: 2.2, 2.00: 4.2, 2.40: 3.1, 3.00: 2.5, 3.40: 2.6, 4.42: 1.8, 5.38: 5.7, 7.40: 1.6, 10.80: 2.6
Ventanas con pico z>3 en alguna banda: BPFO 0%, 2xBPFO 0%, BPFI 100%, BSF 60%, 2xBSF 0%, FTF 0%, 1x eje 60%, 2x eje 100%

## Clase real: B (5 ventanas; tamaños de fallo: [7, 14, 21])
| Banda (Hz) | máx z medio | curtosis señal filtrada* | z en BPFO / 2xBPFO / BPFI / BSF / 2xBSF / FTF / 1x eje / 2x eje |
|---|---|---|---|
| 1000-3000 | 4.5 | – | 2.0 / 1.5 / 1.2 / 0.7 / 0.8 / 1.3 / 1.1 / 2.7 |
| 3000-6000 | 3.1 | – | 1.2 / 0.2 / 1.1 / 0.6 / 0.8 / 1.2 / 1.5 / 1.4 |
| 6000-10000 | 0.3 | – | 0.0 / 0.0 / 0.0 / 0.0 / 0.1 / 0.1 / 0.1 / 0.1 |
| 10000-16000 | 0.1 | – | 0.0 / 0.0 / 0.0 / 0.0 / 0.0 / 0.0 / 0.1 / 0.0 |
| 16000-22000 | 0.1 | – | 0.0 / -0.0 / 0.0 / 0.0 / -0.0 / 0.0 / 0.1 / 0.1 |

Picos principales (banda 1000-3000 Hz), orden: z = 0.36: 1.3, 0.62: 1.4, 1.62: 2.0, 2.00: 2.7, 2.68: 1.9, 3.40: 2.0, 3.60: 2.0, 4.56: 1.3, 5.20: 1.3, 7.20: 1.5
Ventanas con pico z>3 en alguna banda: BPFO 20%, 2xBPFO 0%, BPFI 20%, BSF 0%, 2xBSF 0%, FTF 0%, 1x eje 20%, 2x eje 60%

## Clase real: OR (5 ventanas; tamaños de fallo: [7, 14, 21])
| Banda (Hz) | máx z medio | curtosis señal filtrada* | z en BPFO / 2xBPFO / BPFI / BSF / 2xBSF / FTF / 1x eje / 2x eje |
|---|---|---|---|
| 1000-3000 | 8.0 | – | 6.1 / 2.1 / -0.2 / 0.0 / -0.2 / 1.5 / 3.2 / 3.0 |
| 3000-6000 | 6.3 | – | 5.1 / 1.1 / -0.3 / 0.0 / -0.1 / 3.0 / 3.1 / 2.8 |
| 6000-10000 | 1.0 | – | 0.8 / 0.5 / 0.1 / -0.0 / -0.0 / 0.1 / 0.3 / 0.1 |
| 10000-16000 | 0.4 | – | 0.3 / 0.2 / -0.0 / -0.0 / -0.0 / 0.1 / 0.2 / 0.1 |
| 16000-22000 | 0.2 | – | 0.2 / 0.1 / -0.0 / 0.0 / -0.0 / 0.1 / 0.2 / 0.0 |

Picos principales (banda 1000-3000 Hz), orden: z = 0.40: 1.5, 1.00: 3.2, 1.60: 3.5, 1.98: 3.0, 2.60: 1.8, 3.20: 1.9, 3.60: 6.1, 5.18: 1.5, 5.58: 1.6, 7.18: 2.1
Ventanas con pico z>3 en alguna banda: BPFO 80%, 2xBPFO 40%, BPFI 0%, BSF 0%, 2xBSF 0%, FTF 40%, 1x eje 40%, 2x eje 80%

*Columna reservada; no se calcula en esta versión.