"""Genera docs/img/resultado_principal.png a partir de results_final_B.json."""
import json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
d = json.load(open(os.path.join(ROOT, "experiments/04_online_weight_learning/results_final_B.json")))
g = d["gradiente_mismo_feedback"]; P1, P2 = "prueba1_48k", "prueba2_12k_limpia"
rows = [("Red desplegada (sin ajustar)", d["v1_desplegada"][P1]["balanced_acc"], d["v1_desplegada"][P2]["balanced_acc"]),
        ("CNN desde cero · gradiente", g["cnn"][P1]["balanced_acc"][0], g["cnn"][P2]["balanced_acc"][0]),
        ("Corrección lineal · gradiente", g["residual_only"][P1]["balanced_acc"][0], g["residual_only"][P2]["balanced_acc"][0]),
        ("Red escrita + ajuste · gradiente", g["hand_ft"][P1]["balanced_acc"][0], g["hand_ft"][P2]["balanced_acc"][0]),
        ("LLM edita pesos · iter. 2", d["llm_iter2"][P1]["balanced_acc"], d["llm_iter2"][P2]["balanced_acc"]),
        ("LLM reescribe · iter. 3 (sin gradiente)", d["llm_iter3"][P1]["balanced_acc"], d["llm_iter3"][P2]["balanced_acc"])][::-1]
ink, ink2, muted, grid, surf, c1, c2 = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200); fig.patch.set_facecolor(surf); ax.set_facecolor(surf)
h = 0.36
for i, (lab, a, b) in enumerate(rows):
    ax.barh(i + h / 2 + 0.01, a * 100, h, color=c1, edgecolor=surf, linewidth=1)
    ax.barh(i - h / 2 - 0.01, b * 100, h, color=c2, edgecolor=surf, linewidth=1)
    ax.text(a * 100 + 0.8, i + h / 2, f"{a * 100:.1f}", va="center", fontsize=8, color=ink2)
    ax.text(b * 100 + 0.8, i - h / 2, f"{b * 100:.1f}", va="center", fontsize=8, color=ink2)
ax.set_yticks(np.arange(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
for t in ax.get_yticklabels():
    t.set_color(ink)
    if t.get_text().startswith("LLM"): t.set_fontweight("bold")
ax.axvline(25, color=muted, lw=1, ls=(0, (2, 3))); ax.text(25.5, len(rows) - 0.45, "azar", fontsize=8, color=muted)
ax.set_xlim(0, 100); ax.set_xlabel("Exactitud equilibrada (%)", fontsize=9, color=ink2)
ax.xaxis.grid(True, color=grid, lw=0.8); ax.set_axisbelow(True)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7"); ax.tick_params(axis="x", colors=muted, length=0); ax.tick_params(axis="y", length=0)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c1), plt.Rectangle((0, 0), 1, 1, color=c2)],
          labels=["Prueba 1: velocidades no vistas (48 kHz)", "Prueba 2 limpia: otras grabaciones a 12 kHz, nunca miradas"],
          loc="lower center", bbox_to_anchor=(0.4, -0.27), frameon=False, fontsize=8.5, labelcolor=ink2)
fig.suptitle("Instalar una capacidad con 20 casos: editar pesos razonando vs. entrenar", x=0.02, ha="left", fontsize=11.5, color=ink, fontweight="bold")
ax.set_title("Diagnóstico de rodamientos CWRU · 4 clases · mismos 20 casos etiquetados para todos", loc="left", fontsize=8.5, color=ink2, pad=8)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "docs/img/resultado_principal.png"), facecolor=surf, bbox_inches="tight")
