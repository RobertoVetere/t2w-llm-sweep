"""Informe automático de fallos sobre el conjunto de feedback (lo único que ve el LLM).

Uso: python report.py NOMBRE_RED  (v1 | iter1 | iter2)
"""
import importlib, sys, numpy as np, torch, torch.nn as nn
from common import (dataset, feedback_idx, HandNet, fault_orders, FGRID, BANDS, NPH, CLASSES, bal_acc, predict)

ORD = np.arange(0.5, 20.0, 0.02)          # rejilla en órdenes del eje


def make(name):
    if name == "v1":
        return HandNet
    return importlib.import_module(f"edits.{name}").make_net


def env_spectra(X):
    """Espectro de envolvente normalizado (z) por banda, con el frente físico de v1 como instrumento."""
    net = HandNet(1750)
    with torch.no_grad():
        x = torch.tensor(X[:, None])
        e = net.pool(net.absum(torch.relu(net.fb(x))))
        z = torch.relu(net.fourier(e)); B, K, _ = z.shape
        m = nn.functional.max_pool1d(z.view(B * K, 1, -1), NPH, NPH).view(B, K, net.F)
        return net.norm(m).numpy()            # (B, K, F)


def kurt(x):
    x = x - x.mean(); return float((x ** 4).mean() / (x ** 2).mean() ** 2)


def main(name):
    X, y, rpm, size, fn = dataset()
    idx = feedback_idx()
    mk = make(name)
    p = predict(mk, idx); t = y[idx]
    out = [f"# Informe de fallos: red `{name}` sobre el feedback ({len(idx)} ventanas, 1730/1750 rpm)\n"]
    out.append(f"Exactitud equilibrada en feedback: {bal_acc(t, p):.2f}\n")
    out.append("## Matriz de confusión (filas = real, columnas = predicho)\n")
    out.append("| | " + " | ".join(CLASSES) + " |\n|---|" + "---|" * 4)
    for c in range(4):
        out.append(f"| {CLASSES[c]} | " + " | ".join(str(int(((t == c) & (p == d)).sum())) for d in range(4)) + " |")
    o = fault_orders()
    marks = {"BPFO": o["bpfo"], "2xBPFO": 2 * o["bpfo"], "BPFI": o["bpfi"], "BSF": o["bsf"], "2xBSF": 2 * o["bsf"],
             "FTF": o["ftf"], "1x eje": 1.0, "2x eje": 2.0}
    out.append("\nÓrdenes de referencia (múltiplos de la velocidad del eje): " +
               ", ".join(f"{k}={v:.3f}" for k, v in marks.items()) + "\n")
    Z = env_spectra(X[idx])                       # (n, K, F)
    for c in range(4):
        sel = np.where(t == c)[0]
        out.append(f"\n## Clase real: {CLASSES[c]} ({len(sel)} ventanas; tamaños de fallo: {sorted(set(size[idx][sel].tolist()))})")
        # pasar cada ventana a órdenes del eje y promediar
        zo = np.zeros((len(sel), len(BANDS), len(ORD)))
        for i, j in enumerate(sel):
            fr = rpm[idx][j] / 60.0
            for k in range(len(BANDS)):
                zo[i, k] = np.interp(ORD * fr, FGRID, Z[j, k])
        mean = zo.mean(0)
        out.append("| Banda (Hz) | máx z medio | curtosis señal filtrada* | z en " + " / ".join(marks) + " |\n|---|---|---|---|")
        for k, (lo, hi) in enumerate(BANDS):
            vals = []
            for v in marks.values():
                w = np.abs(ORD - v) <= max(0.03, 0.01 * v)
                vals.append(f"{mean[k][w].max():.1f}")
            out.append(f"| {lo}-{hi} | {Z[sel, k].max(1).mean():.1f} | – | " + " / ".join(vals) + " |")
        kb = int(np.argmax([Z[sel, k].max(1).mean() for k in range(len(BANDS))]))
        sp = mean[kb]
        pk = [i for i in range(1, len(ORD) - 1) if sp[i] >= sp[i - 1] and sp[i] >= sp[i + 1]]
        pk = sorted(pk, key=lambda i: -sp[i])[:10]
        out.append(f"\nPicos principales (banda {BANDS[kb][0]}-{BANDS[kb][1]} Hz), orden: z = " +
                   ", ".join(f"{ORD[i]:.2f}: {sp[i]:.1f}" for i in sorted(pk)))
        # consistencia: fracción de ventanas con pico > 3 en cada marca
        cons = []
        for nm, v in marks.items():
            w = np.abs(ORD - v) <= max(0.03, 0.01 * v)
            cons.append(f"{nm} {np.mean(zo[:, :, w].max(2).max(1) > 3):.0%}")
        out.append("Ventanas con pico z>3 en alguna banda: " + ", ".join(cons))
    out.append("\n*Columna reservada; no se calcula en esta versión.")
    txt = "\n".join(out)
    open(f"reports/report_{name}.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1])
