"""v2 del punto de partida (añadido DESPUÉS de ver v1; se reporta junto a v1).

En v1 se ajustaba la red escrita entera; como algunas redes escritas son un cuello de
botella (HAR usa 3 rasgos), el ajuste no puede ir más allá. Aquí se usa el mismo esquema
que en rodamientos (punto 1): logits = red_escrita(x) + corrección lineal que empieza en
cero. Control limpio: la misma corrección lineal, mismo optimizador, sin la red escrita.
"""
import copy, json, sys, numpy as np, torch, torch.nn as nn
from run_tasks import load, NETS, to_t, bal_acc, front

torch.set_num_threads(2)


class Residual(nn.Module):
    def __init__(self, hand, dim, ncls, use_hand):
        super().__init__()
        self.hand, self.use_hand = hand, use_hand
        self.corr = nn.Linear(dim, ncls)
        nn.init.zeros_(self.corr.weight); nn.init.zeros_(self.corr.bias)

    def forward(self, xh, f):
        out = self.corr(f)
        if self.use_hand:
            out = out + xh
        return out


def run(task):
    Xtr, ytr, Xte, yte = load(task)
    hand = NETS[task]()
    with torch.no_grad():
        if task == "fsdd":
            Htr = torch.cat([hand(to_t(x, task)) for x in Xtr]); Hte = torch.cat([hand(to_t(x, task)) for x in Xte])
        else:
            Htr, Hte = hand(to_t(Xtr, task)), hand(to_t(Xte, task))
    Ftr, Fte = front(hand, Xtr, task), front(hand, Xte, task)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6          # estandarización con el conjunto de ajuste
    Ftr, Fte = torch.tensor((Ftr - mu) / sd, dtype=torch.float32), torch.tensor((Fte - mu) / sd, dtype=torch.float32)
    ncls = len(np.unique(ytr)); res = {}
    for k in [1, 5, 20]:
        r = {"hand+residual": [], "residual_only": []}
        for seed in range(5):
            rng = np.random.default_rng(100 + seed)
            idx = np.concatenate([rng.choice(np.where(ytr == c)[0], k, replace=False) for c in range(ncls)])
            for name, use in [("hand+residual", True), ("residual_only", False)]:
                torch.manual_seed(seed)
                m = Residual(None, Ftr.shape[1], ncls, use)
                opt = torch.optim.Adam(m.corr.parameters(), 1e-2, weight_decay=1e-3)
                yb = torch.tensor(ytr[idx])
                for _ in range(300):
                    loss = nn.functional.cross_entropy(m(Htr[idx], Ftr[idx]), yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    p = m(Hte, Fte).argmax(1).numpy()
                r[name].append(bal_acc(yte, p))
        res[f"k{k}"] = {c: (float(np.mean(v)), float(np.std(v))) for c, v in r.items()}
        print(task, k, res[f"k{k}"], flush=True)
    json.dump(res, open(f"results_residual_{task}.json", "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1])
