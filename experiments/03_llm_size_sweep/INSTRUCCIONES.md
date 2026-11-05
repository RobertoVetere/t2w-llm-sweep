# Punto 3: otros LLM escribiendo la red de rodamientos

Desde la sesión no puedo ejecutar LLM (Hugging Face tiene desactivada la ejecución de Spaces y no puedo descargar modelos). Hay dos formas de hacerlo:

## Opción A: en tu ordenador con Ollama (recomendada)
1. Instala Ollama (ollama.com) y descarga los modelos:
   `ollama pull qwen2.5:0.5b`, `qwen2.5:1.5b`, `qwen2.5:3b`, `qwen2.5:7b`, `qwen2.5:14b`
   (y `qwen2.5:32b` si tu equipo lo aguanta).
2. Para cada modelo, 3 intentos (para medir la variabilidad):
   `ollama run qwen2.5:7b --nowordwrap < prompt_bearing.txt > qwen7b_1.txt`
3. Pásame todos los `.txt`. Los evalúo con `eval_llm_code.py`, con el mismo evaluador que mi red.

## Opción B: activar los Spaces de Hugging Face
En la configuración del conector MCP de Hugging Face, quita `gradio=none` o añade el Space
`Fsezai33/llm-zero-gpu-playground`. Con eso puedo lanzar yo los modelos desde aquí.

## Qué se mide
- Si el código se ejecuta (los modelos pequeños suelen fallar aquí).
- Exactitud equilibrada en 4 clases (azar 25%) y detección de sano/averiado.
- Si aparece un salto con el tamaño del modelo, es una capacidad emergente.
