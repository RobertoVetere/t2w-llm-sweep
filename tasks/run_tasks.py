"""Evalúa las tres tareas nuevas: cero ejemplos + pocos ejemplos (tres condiciones)."""
import copy, glob, json, os, sys, numpy as np, torch, torch.nn as nn
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from digits_net import DigitsNet
from har_net import HARNet
from fsdd_net import FSDDNet

torch.set_num_threads(2)
HAR = "/home/claude/data/HumanActivityRecognition"
FSDD = "/home/claude/data/free-spoken-digit-dataset/recordings"


def bal_acc(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


# ---------------- datos ----------------
def load(task):
    if task == "digits":
        from sklearn.datasets import load_digits
        d = load_digits(); X = d.data.astype(np.float32); y = d.target
        return list(X[:900]), y[:900], list(X[900:]), y[900:]
    if task == "har":
        ld = lambda f: np.loadtxt(f"{HAR}/{f}", dtype=np.float32)
        Xtr, Xte = ld("X_train.txt"), ld("X_test.txt")
        ytr, yte = ld("y_train.txt").astype(int) - 1, ld("y_test.txt").astype(int) - 1
        return list(Xtr), ytr, list(Xte), yte
    if task == "fsdd":
        tr, te = ([], []), ([], [])
        for f in sorted(glob.glob(f"{FSDD}/*.wav")):
            d, _, idx = os.path.basename(f)[:-4].split("_")
            sr, s = wavfile.read(f)
            s = s.astype(np.float32); s = s / (np.abs(s).max() + 1e-9)
            tgt = te if int(idx) <= 4 else tr
            tgt[0].append(s); tgt[1].append(int(d))
        return tr[0], np.array(tr[1]), te[0], np.array(te[1])


NETS = {"digits": DigitsNet, "har": HARNet, "fsdd": FSDDNet}


def to_t(x, task):
    return torch.tensor(x)[None, None] if task == "fsdd" else torch.tensor(np.stack(x))


def predict(net, X, task):
    with torch.no_grad():
        if task == "fsdd":
            return np.array([net(to_t(x, task)).argmax(1).item() for x in X])
        return net(to_t(X, task)).argmax(1).numpy()


def logits_batch(net, xs, task):
    if task == "fsdd":
        return torch.cat([net(to_t(x, task)) for x in xs])
    return net(to_t(xs, task))


def finetune(net, X, y, task, prior, seed, steps=200):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    theta0 = [p.detach().clone() for p in net.parameters()]
    for p in net.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(steps):
        idx = rng.choice(len(X), min(16 if task != "fsdd" else 8, len(X)), replace=False)
        out = logits_batch(net, [X[i] for i in idx], task)
        loss = nn.functional.cross_entropy(out, torch.tensor(y[idx]))
        if prior:
            loss = loss + 1e-3 * sum(((p - p0) ** 2).sum() for p, p0 in zip(net.parameters(), theta0))
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def front(net, X, task):
    if task != "fsdd":
        return np.stack(X)
    with torch.no_grad():
        return np.stack([net.features(to_t(x, task))[0].numpy() for x in X])


if __name__ == "__main__":
    task = sys.argv[1]
    Xtr, ytr, Xte, yte = load(task)
    res = {"n_train_pool": len(ytr), "n_test": len(yte)}
    hand = NETS[task]()
    res["zero_shot"] = bal_acc(yte, predict(hand, Xte, task))
    print(task, "cero ejemplos:", res["zero_shot"], flush=True)
    Fte = front(hand, Xte, task); Ftr_all = front(hand, Xtr, task)
    for k in [1, 5, 20]:
        r = {"hand_ft": [], "random_init": [], "logreg": []}
        for seed in range(5):
            rng = np.random.default_rng(100 + seed)
            idx = np.concatenate([rng.choice(np.where(ytr == c)[0], k, replace=False) for c in np.unique(ytr)])
            Xs, ys = [Xtr[i] for i in idx], ytr[idx]
            n1 = finetune(copy.deepcopy(hand), Xs, ys, task, True, seed)
            r["hand_ft"].append(bal_acc(yte, predict(n1, Xte, task)))
            n2 = copy.deepcopy(hand)
            torch.manual_seed(seed)
            for m in n2.modules():
                if hasattr(m, "reset_parameters") and m is not n2:
                    m.reset_parameters()
            n2 = finetune(n2, Xs, ys, task, False, seed)
            r["random_init"].append(bal_acc(yte, predict(n2, Xte, task)))
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(Ftr_all[idx], ys)
            r["logreg"].append(bal_acc(yte, lr.predict(Fte)))
        res[f"k{k}"] = {c: (float(np.mean(v)), float(np.std(v))) for c, v in r.items()}
        print(task, k, res[f"k{k}"], flush=True)
    json.dump(res, open(f"results_{task}.json", "w"), indent=1)
