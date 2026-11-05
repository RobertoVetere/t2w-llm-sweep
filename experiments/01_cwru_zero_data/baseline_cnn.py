"""CNN 1D entrenada (tipo WDCNN), validación dejando una velocidad fuera, con k ventanas por clase."""
import json, sys, numpy as np, torch, torch.nn as nn
from data import load_windows
from evaluate import bal_acc

torch.set_num_threads(2)


def model():
    return nn.Sequential(
        nn.Conv1d(1, 16, 64, 16, padding=24), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        nn.Flatten(), nn.Linear(64, 4))


def run(k, seed, X, y, rpm):
    accs = []
    for test_r in np.unique(rpm):
        rng = np.random.default_rng(seed)
        tr = np.where(rpm != test_r)[0]
        if k is not None:
            tr = np.concatenate([rng.choice(tr[y[tr] == c], min(k, (y[tr] == c).sum()), replace=False) for c in range(4)])
        te = np.where(rpm == test_r)[0]
        torch.manual_seed(seed)
        net = model(); opt = torch.optim.Adam(net.parameters(), 1e-3)
        # pesos de clase para el desequilibrio
        cw = torch.tensor([1 / max(1, (y[tr] == c).sum()) for c in range(4)], dtype=torch.float32)
        lossf = nn.CrossEntropyLoss(weight=cw / cw.sum() * 4)
        Xt = torch.tensor(X[tr][:, None]); yt = torch.tensor(y[tr])
        mu, sd = Xt.mean(), Xt.std()
        steps = 300
        for it in range(steps):
            idx = torch.randint(0, len(tr), (min(32, len(tr)),))
            xb = (Xt[idx] - mu) / sd
            sh = np.random.randint(0, 2000); xb = torch.roll(xb, sh, -1)
            net.train(); opt.zero_grad(); l = lossf(net(xb), yt[idx]); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = net((torch.tensor(X[te][:, None]) - mu) / sd).argmax(1).numpy()
        accs.append(bal_acc(y[te], p, 4))
    return float(np.mean(accs))


if __name__ == "__main__":
    X, y, rpm, size, fn = load_windows()
    res = {}
    for k in [1, 5, 20, None]:
        vals = [run(k, s, X, y, rpm) for s in range(3)]
        res[str(k if k else "todos")] = (float(np.mean(vals)), float(np.std(vals)))
        print(k, res[str(k if k else "todos")], flush=True)
    json.dump(res, open("results_cnn.json", "w"), indent=1)
