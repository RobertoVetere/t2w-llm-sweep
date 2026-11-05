"""Prueba unitaria con señales sintéticas generadas aquí (no se usa el dataset)."""
import numpy as np, torch
from handnet import HandNet, fault_orders, FS, WIN, CLASSES, n_params

rng = np.random.default_rng(0)


def synth(kind, rpm, snr=1.0, fres=None):
    fr = rpm / 60
    o = fault_orders()
    t = np.arange(WIN) / FS
    x = rng.normal(0, 1, WIN) + 0.5 * np.sin(2 * np.pi * fr * t) + 0.3 * np.sin(2 * np.pi * 2 * fr * t)
    if kind == "Normal":
        return x
    ff = {"IR": o["bpfi"], "B": 2 * o["bsf"], "OR": o["bpfo"]}[kind] * fr
    fres = fres or rng.uniform(2500, 15000)
    ring = np.exp(-np.arange(300) / 40) * np.sin(2 * np.pi * fres * np.arange(300) / FS)
    imp = np.zeros(WIN)
    tk = rng.uniform(0, 1 / ff)
    while tk < WIN / FS:
        i = int(tk * FS)
        a = 1.0
        if kind == "IR":
            a = 1 + 0.8 * np.cos(2 * np.pi * fr * tk)
        if kind == "B":
            a = 1 + 0.8 * np.cos(2 * np.pi * o["ftf"] * fr * tk)
        imp[i] += a
        tk += (1 / ff) * (1 + rng.normal(0, 0.01))
    s = np.convolve(imp, ring)[:WIN]
    s *= snr * 3 / s.std()
    return x + s


if __name__ == "__main__":
    for rpm in (1730, 1797):
        net = HandNet(rpm)
        print("rpm", rpm, "parámetros", n_params(net))
        for snr in (1.0, 0.3):
            for kind in CLASSES:
                xs = np.stack([synth(kind, rpm, snr) for _ in range(20)])[:, None].astype(np.float32)
                out, sc = net(torch.tensor(xs), return_scores=True)
                pred = out.argmax(1).numpy()
                acc = np.mean([CLASSES[p] == kind for p in pred])
                print(f"  snr={snr} {kind:6s} acierto={acc:.2f}  scores medios IR/B/OR={sc.mean(0).numpy().round(1)}")
