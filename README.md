# t2w-llm-sweep

Parte del proyecto Text-to-Weights: ¿pueden los LLM de distintos tamaños escribir a mano los pesos de una red neuronal que diagnostique rodamientos, sin datos ni entrenamiento?

- `prompt_bearing.txt`: el prompt, idéntico para todos los modelos.
- `run_llm.py`: descarga el GGUF oficial de Qwen desde Hugging Face, **verifica que su SHA-256 coincide con el publicado por Hugging Face** y genera 3 respuestas por modelo.
- `.github/workflows/sweep.yml`: lo ejecuta en GitHub Actions para Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B y 14B (cuantización q4_k_m).
- `results/`: respuestas y metadatos (revisión del repo, hashes, tiempos). La evaluación se hace con datos reales del CWRU, con el mismo evaluador que la red escrita por Claude.
