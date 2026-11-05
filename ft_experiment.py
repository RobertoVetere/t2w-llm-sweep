"""Punto 1: ¿sirve la red escrita a mano como punto de partida para entrenar con pocos datos?

Validación dejando fuera una velocidad (entrena con 3, prueba con la 4ª), k ventanas por clase.
Condiciones:
  hand_ft     : red escrita a mano; se ajustan su cabeza (pesos de armónicos y salida)
                y una corrección lineal sobre el espectro que empieza en cero.
  front_only  : mismo frente escrito a mano (filtros + espectro de envolvente) pero la
                cabeza se aprende desde cero (lineal sobre el espectro). Aísla el valor
                de la cabeza escrita.
  cnn         : CNN 1D desde cero (resultados de baseline_cnn.py, mismas particiones).
"""
import json, numpy as np, torch, torch.nn as nn
from handnet import HandNet, NPH
from data import load_windows
from evaluate import bal_acc

torch.set_num_threads(2)


def spectra(X):
    net = HandNet(1750)  # el frente (filtros, Fourier, LayerNorm) no depende de la velocidad
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            x = torch.tensor(X[i:i + 64][:, None])
            y = torch.relu(net.fb(x)); e = net.pool(net.absum(y))
            z = torch.relu(net.fourier(e)); B, K, _ = z.shape
            m = nn.functional.max_pool1d(z.view(B * K, 1, -1), NPH, NPH).view(B, K, net.F)
            out.append(net.norm(m).numpy())
    return np.concatenate(out)


class Head(nn.Module):
    """Cabeza escrita a mano (dependiente de rpm) + corrección lineal entrenable."""
    def __init__(self, rpms, use_hand, K, F):
        super().__init__()
        self.use_hand = use_hand
        ref = HandNet(1750)
        self.G = {r: torch.tensor(HandNet(r).gather.weight.numpy()) for r in rpms}
        self.S, self.nh = ref.S, ref.nh
        self.harm = nn.Parameter(ref.harm.weight.detach().clone())
        self.W = nn.Parameter(ref.head.weight.detach().clone())
        self.b = nn.Parameter(ref.head.bias.detach().clone())
        self.corr = nn.Linear(K * F, 4)
        nn.init.zeros_(self.corr.weight); nn.init.zeros_(self.corr.bias)
        if not use_hand:
            nn.init.normal_(self.corr.weight, 0, 0.01)

    def forward(self, z, r):
        out = self.corr(z.flatten(1))
        if self.use_hand:
            B, K, F = z.shape
            g = (z @ self.G[r].T).view(B * K, 1, -1)
            g = nn.functional.max_pool1d(g, self.S, self.S).view(B, K, self.nh)
            sc = (g @ self.harm.T).max(1).values
            out = out + sc @ self.W.T + self.b
        return out


def run(Z, y, rpm, k, seed, use_hand, steps=300):
    rpms = list(np.unique(rpm)); accs = []
    for test_r in rpms:
        rng = np.random.default_rng(seed)
        tr = np.where(rpm != test_r)[0]
        if k is not None:
            tr = np.concatenate([rng.choice(tr[y[tr] == c], min(k, (y[tr] == c).sum()), replace=False) for c in range(4)])
        te = np.where(rpm == test_r)[0]
        torch.manual_seed(seed)
        h = Head(rpms, use_hand, Z.shape[1], Z.shape[2])
        if k != 0:
            opt = torch.optim.Adam([{"params": [h.harm, h.W, h.b], "lr": 1e-2},
                                    {"params": h.corr.parameters(), "lr": 1e-3, "weight_decay": 1e-2}])
            cw = torch.tensor([1 / max(1, (y[tr] == c).sum()) for c in range(4)], dtype=torch.float32)
            lossf = nn.CrossEntropyLoss(weight=cw / cw.sum() * 4)
            for it in range(steps):
                # minilotes por velocidad (la cabeza escrita depende de rpm)
                r = int(rng.choice(np.unique(rpm[tr])))
                idx = tr[rpm[tr] == r]
                idx = rng.choice(idx, min(32, len(idx)), replace=False)
                opt.zero_grad()
                l = lossf(h(torch.tensor(Z[idx]), r), torch.tensor(y[idx]))
                l.backward(); opt.step()
        with torch.no_grad():
            p = h(torch.tensor(Z[te]), test_r).argmax(1).numpy()
        accs.append(bal_acc(y[te], p, 4))
    return float(np.mean(accs))


if __name__ == "__main__":
    X, y, rpm, size, fn = load_windows()
    Z = spectra(X).astype(np.float32)
    res = {}
    for name, use_hand in [("hand_ft", True), ("front_only", False)]:
        for k in [0, 1, 5, 20, None]:
            if k == 0 and not use_hand:
                continue
            vals = [run(Z, y, rpm, k, s, use_hand) for s in range(5 if k != 0 else 1)]
            res[f"{name}_k{k if k is not None else 'all'}"] = (float(np.mean(vals)), float(np.std(vals)))
            print(name, k, res[f"{name}_k{k if k is not None else 'all'}"], flush=True)
    json.dump(res, open("results_ft.json", "w"), indent=1)
