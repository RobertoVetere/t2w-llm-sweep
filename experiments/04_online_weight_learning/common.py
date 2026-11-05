"""Utilidades compartidas: datos, particiones fijas y evaluación."""
import os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01_cwru_zero_data"))
from handnet import HandNet, fault_orders, FGRID, BANDS, NPH, CLASSES   # noqa: E402
from data import load_windows                                         # noqa: E402

FEEDBACK_RPM = (1730, 1750)
TEST_RPM = (1772, 1797)
K = 5

_cache = {}


def dataset():
    if "d" not in _cache:
        _cache["d"] = load_windows()
    return _cache["d"]


def feedback_idx(seed=0, k=K):
    X, y, rpm, size, fn = dataset()
    rng = np.random.default_rng(seed)
    pool = np.where(np.isin(rpm, FEEDBACK_RPM))[0]
    return np.concatenate([rng.choice(pool[y[pool] == c], k, replace=False) for c in range(4)])


def test_idx():
    X, y, rpm, size, fn = dataset()
    return np.where(np.isin(rpm, TEST_RPM))[0]


def bal_acc(y, p, k=4):
    return float(np.mean([np.mean(p[y == c] == c) for c in range(k) if np.any(y == c)]))


def predict(make_net, idx):
    """make_net(rpm) -> nn.Module con salida (B, 4)."""
    X, y, rpm, size, fn = dataset()
    pred = np.zeros(len(idx), int)
    for r in np.unique(rpm[idx]):
        m = rpm[idx] == r
        net = make_net(int(r))
        with torch.no_grad():
            pred[m] = net(torch.tensor(X[idx][m][:, None])).argmax(1).numpy()
    return pred


def metrics(make_net, idx):
    X, y, rpm, size, fn = dataset()
    p = predict(make_net, idx); t = y[idx]
    return {"balanced_acc": bal_acc(t, p),
            "per_class": {CLASSES[c]: float(np.mean(p[t == c] == c)) for c in range(4)},
            "healthy_vs_fault": bal_acc((t > 0).astype(int), (p > 0).astype(int), 2),
            "false_alarms": f"{int(((p > 0) & (t == 0)).sum())}/{int((t == 0).sum())}"}
