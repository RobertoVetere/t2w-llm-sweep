"""Análisis libre de los 20 casos de feedback (fase B). Solo feedback; la prueba no se toca.
Instrumentos: curtosis por bandas finas (kurtograma simple), espectro de envolvente en la banda
más impulsiva, cepstrum de la envolvente y estadísticos temporales."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from common import dataset, feedback_idx, CLASSES, fault_orders

FS = 48000
EDGES = [(lo, lo + 500) for lo in range(500, 6000, 500)] + [(6000, 9000), (9000, 14000), (14000, 22000)]


def kurt(x):
    x = x - x.mean(); return float((x ** 4).mean() / ((x ** 2).mean() ** 2 + 1e-12))


def band_env(x, lo, hi):
    sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
    return np.abs(hilbert(sosfiltfilt(sos, x)))


def env_orders(env, fr, orders, fs=FS):
    e = env - env.mean(); n = len(e)
    S = np.abs(np.fft.rfft(e * np.hanning(n))); f = np.fft.rfftfreq(n, 1 / fs)
    sel = (f > 3) & (f < 600); base = np.median(S[sel]); mad = np.median(np.abs(S[sel] - base)) + 1e-12
    out = []
    for o in orders:
        w = np.abs(f - o * fr) <= 0.01 * o * fr + 2.0
        out.append((S[w].max() - base) / (1.4826 * mad))
    return out


if __name__ == "__main__":
    X, y, rpm, size, fn = dataset()
    idx = feedback_idx()
    o = fault_orders()
    names = ["1x", "2x", "FTF", "BSF", "2BSF", "2BSF-FTF", "2BSF+FTF", "BPFO", "2BPFO", "BPFI", "BPFI-1x", "BPFI+1x"]
    ords = [1, 2, o["ftf"], o["bsf"], 2 * o["bsf"], 2 * o["bsf"] - o["ftf"], 2 * o["bsf"] + o["ftf"],
            o["bpfo"], 2 * o["bpfo"], o["bpfi"], o["bpfi"] - 1, o["bpfi"] + 1]
    print("órdenes:", dict(zip(names, np.round(ords, 3))))
    print("\nreal size rpm | rms  kurt  cresta | curtosis por banda (500 Hz hasta 6 kHz; luego 6-9k, 9-14k, 14-22k)")
    rows = []
    for j in idx:
        x = X[j].astype(np.float64); fr = rpm[j] / 60
        kb = [kurt(band_env(x, lo, hi) ** 1) for lo, hi in EDGES]   # curtosis de la envolvente por banda
        rows.append((j, kb))
        print(f"{CLASSES[y[j]]:6s} {size[j]:2d} {rpm[j]} | {x.std():.3f} {kurt(x):5.1f} {np.abs(x).max() / x.std():5.1f} | " +
              " ".join(f"{k:4.1f}" for k in kb))
    print("\nEspectro de envolvente (z robusto) en la banda 2-5 kHz y en la de mayor curtosis (<6 kHz):")
    print("real size rpm banda* | " + " ".join(f"{n:>8s}" for n in names))
    for j, kb in rows:
        x = X[j].astype(np.float64); fr = rpm[j] / 60
        z1 = env_orders(band_env(x, 2000, 5000), fr, ords)
        kbest = int(np.argmax(kb[:11])); lo, hi = EDGES[kbest]
        z2 = env_orders(band_env(x, lo, hi), fr, ords)
        print(f"{CLASSES[y[j]]:6s} {size[j]:2d} {rpm[j]} 2-5k   | " + " ".join(f"{v:8.1f}" for v in z1))
        print(f"{'':6s} {'':2s} {'':4s} {lo}-{hi} | " + " ".join(f"{v:8.1f}" for v in z2))
