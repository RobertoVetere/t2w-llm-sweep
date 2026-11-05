"""Evaluación final única: v1, iter1, iter2 y métodos por gradiente con el mismo feedback."""
import json, os, sys, numpy as np, torch, torch.nn as nn
from common import dataset, feedback_idx, test_idx, metrics, bal_acc, HandNet, CLASSES, HERE
import edits.iter1 as it1, edits.iter2 as it2, edits.iter3 as it3
sys.path.insert(0, os.path.join(HERE, "..", "01_cwru_zero_data"))
from ft_experiment import spectra, Head            # noqa: E402
from baseline_cnn import model as cnn_model        # noqa: E402

torch.set_num_threads(2)


def per_class(t, p):
    return {CLASSES[c]: float(np.mean(p[t == c] == c)) for c in range(4)}


def train_head(Z, y, rpm, tr, use_hand, seed, steps=300):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    h = Head(list(np.unique(rpm)), use_hand, Z.shape[1], Z.shape[2])
    opt = torch.optim.Adam([{"params": [h.harm, h.W, h.b], "lr": 1e-2},
                            {"params": h.corr.parameters(), "lr": 1e-3, "weight_decay": 1e-2}])
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        r = int(rng.choice(np.unique(rpm[tr]))); idx = tr[rpm[tr] == r]
        idx = rng.choice(idx, min(32, len(idx)), replace=False)
        l = lossf(h(torch.tensor(Z[idx]), r), torch.tensor(y[idx]))
        opt.zero_grad(); l.backward(); opt.step()
    return h


def eval_head(h, Z, y, rpm, te):
    p = np.zeros(len(te), int)
    with torch.no_grad():
        for r in np.unique(rpm[te]):
            m = rpm[te] == r
            p[m] = h(torch.tensor(Z[te][m]), int(r)).argmax(1).numpy()
    return p


def train_cnn(X, y, tr, seed, steps=300):
    torch.manual_seed(seed); net = cnn_model(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt = torch.tensor(X[tr][:, None]); yt = torch.tensor(y[tr]); mu, sd = Xt.mean(), Xt.std()
    for _ in range(steps):
        i = torch.randint(0, len(tr), (min(32, len(tr)),))
        xb = torch.roll((Xt[i] - mu) / sd, int(np.random.randint(0, 2000)), -1)
        net.train(); opt.zero_grad(); nn.functional.cross_entropy(net(xb), yt[i]).backward(); opt.step()
    net.eval()
    return net, mu, sd


if __name__ == "__main__":
    X, y, rpm, size, fn = dataset()
    te = test_idx(); t = y[te]
    res = {"test_windows": int(len(te)), "test_rpm": [1772, 1797], "feedback_windows": 20}
    # --- LLM: v1, iter1, iter2 ---
    for name, mk in [("v1_desplegada", HandNet), ("llm_iter1", it1.make_net), ("llm_iter2", it2.make_net)]:
        res[name] = metrics(mk, te); print(name, res[name], flush=True)
    # --- gradiente con el mismo feedback (seed 0) y con 5 sorteos de feedback ---
    Z = spectra(X).astype(np.float32)
    for fseeds, tag in [([0], "feedback_semilla0"), ([0, 1, 2, 3, 4], "feedback_5_sorteos")]:
        out = {"hand_ft": [], "residual_only": [], "cnn": []}
        pc = {"hand_ft": [], "residual_only": [], "cnn": []}
        for fs in fseeds:
            tr = feedback_idx(fs)
            for s in range(5):
                for nm, use in [("hand_ft", True), ("residual_only", False)]:
                    p = eval_head(train_head(Z, y, rpm, tr, use, s), Z, y, rpm, te)
                    out[nm].append(bal_acc(t, p)); pc[nm].append(per_class(t, p))
                net, mu, sd = train_cnn(X, y, tr, s)
                with torch.no_grad():
                    p = net((torch.tensor(X[te][:, None]) - mu) / sd).argmax(1).numpy()
                out["cnn"].append(bal_acc(t, p)); pc["cnn"].append(per_class(t, p))
        res[tag] = {k: {"balanced_acc": [float(np.mean(v)), float(np.std(v))],
                        "per_class_mean": {c: float(np.mean([d[c] for d in pc[k]])) for c in CLASSES}}
                    for k, v in out.items()}
        print(tag, json.dumps(res[tag]), flush=True)
    json.dump(res, open("results_final.json", "w"), indent=1)
