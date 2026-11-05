"""Prueba 2 (limpia): grabaciones a 12 kHz de 1772/1797 rpm + sanas, misma cadena de remuestreo."""
import glob, os
import numpy as np
from scipy.signal import resample_poly
from common import CLASSES
from data import ROOT
from handnet import WIN


def load_test2():
    X, y, rpm, size, fn = [], [], [], [], []
    files = []
    for r in (1772, 1797):
        files += sorted(glob.glob(f"{ROOT}/{r} RPM/{r}_*_DE12.npz")) + [f"{ROOT}/{r} RPM/{r}_Normal.npz"]
    for f in files:
        b = os.path.basename(f); r = int(b.split("_")[0])
        sig = np.load(f)["DE"].ravel().astype(np.float64)
        if "Normal" in b:
            c, s = "Normal", 0
            sig = resample_poly(sig, 1, 4)              # 48 kHz -> 12 kHz
        else:
            c = b.split("_")[1].split("@")[0]; s = int(b.split("_")[2])
        sig = resample_poly(sig, 4, 1).astype(np.float32)   # 12 kHz -> 48 kHz (ancho de banda 6 kHz)
        for i in range(len(sig) // WIN):
            X.append(sig[i * WIN:(i + 1) * WIN]); y.append(CLASSES.index(c)); rpm.append(r); size.append(s); fn.append(b)
    return np.stack(X), np.array(y), np.array(rpm), np.array(size), np.array(fn)
