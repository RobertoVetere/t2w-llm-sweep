import glob, os, re
import numpy as np
from handnet import WIN

DATA = os.environ.get("T2W_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
ROOT = os.path.join(DATA, "CWRU_Bearing_NumPy", "Data")
CLASSES = ["Normal", "IR", "B", "OR"]


def load_windows():
    """Devuelve X (N, WIN), y, rpm, size, file. Solo DE a 48 kHz, según el protocolo."""
    X, y, rpm, size, fn = [], [], [], [], []
    files = sorted(glob.glob(f"{ROOT}/*/*_DE48.npz") + glob.glob(f"{ROOT}/*/*_Normal.npz"))
    for f in files:
        b = os.path.basename(f)
        r = int(b.split("_")[0])
        if "Normal" in b:
            c, s = "Normal", 0
        else:
            c = b.split("_")[1].split("@")[0]
            s = int(b.split("_")[2])
        sig = np.load(f)["DE"].ravel().astype(np.float32)
        n = len(sig) // WIN
        for i in range(n):
            X.append(sig[i * WIN:(i + 1) * WIN]); y.append(CLASSES.index(c))
            rpm.append(r); size.append(s); fn.append(b)
    return np.stack(X), np.array(y), np.array(rpm), np.array(size), np.array(fn)
