"""Evaluación ciega única de la red escrita a mano."""
import json, numpy as np, torch
from handnet import HandNet, weights_hash, n_params
from data import load_windows, CLASSES


def bal_acc(y, p, k):
    return float(np.mean([np.mean(p[y == c] == c) for c in range(k) if np.any(y == c)]))


if __name__ == "__main__":
    X, y, rpm, size, fn = load_windows()
    pred = np.zeros(len(y), int); scores = np.zeros((len(y), 3))
    for r in np.unique(rpm):
        net = HandNet(int(r)); m = rpm == r
        with torch.no_grad():
            out, sc = net(torch.tensor(X[m][:, None]), return_scores=True)
        pred[m] = out.argmax(1).numpy(); scores[m] = sc.numpy()
    res = {"weights_sha256": weights_hash(), "params_per_net": n_params(HandNet(1750)),
           "n_windows": int(len(y)), "per_class_n": [int((y == c).sum()) for c in range(4)]}
    res["balanced_acc_4class"] = bal_acc(y, pred, 4)
    res["acc_raw"] = float(np.mean(pred == y))
    yb, pb = (y > 0).astype(int), (pred > 0).astype(int)
    res["balanced_acc_healthy_vs_fault"] = bal_acc(yb, pb, 2)
    cm = np.zeros((4, 4), int)
    for a, b in zip(y, pred): cm[a, b] += 1
    res["confusion(rows=true)"] = cm.tolist()
    res["by_size"] = {int(s): float(np.mean(pred[size == s] == y[size == s])) for s in np.unique(size)}
    res["by_rpm"] = {int(r): bal_acc(y[rpm == r], pred[rpm == r], 4) for r in np.unique(rpm)}
    res["by_file"] = {f: float(np.mean(pred[fn == f] == y[fn == f])) for f in np.unique(fn)}
    np.savez("eval_outputs.npz", y=y, pred=pred, scores=scores, rpm=rpm, size=size, fn=fn)
    json.dump(res, open("results_v1.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "by_file"}, indent=1))
