"""Evaluación final de la fase B, una sola vez: prueba 1 (48 kHz, 1772/1797) y prueba 2 limpia (12 kHz)."""
import json, numpy as np, torch
from common import dataset, feedback_idx, test_idx, bal_acc, HandNet, CLASSES
import edits.iter2 as it2, edits.iter3 as it3
from final_eval import train_head, eval_head, train_cnn, per_class, spectra
from test2 import load_test2

torch.set_num_threads(2)


def run_nets(mk, X, rpm, idx):
    p = np.zeros(len(idx), int)
    for r in np.unique(rpm[idx]):
        m = rpm[idx] == r; net = mk(int(r))
        with torch.no_grad():
            for s in range(0, m.sum(), 64):
                sel = np.where(m)[0][s:s + 64]
                p[sel] = net(torch.tensor(X[idx][sel][:, None])).argmax(1).numpy()
    return p


def summary(t, p):
    return {"balanced_acc": bal_acc(t, p), "per_class": per_class(t, p),
            "healthy_vs_fault": bal_acc((t > 0).astype(int), (p > 0).astype(int), 2),
            "false_alarms": f"{int(((p > 0) & (t == 0)).sum())}/{int((t == 0).sum())}"}


if __name__ == "__main__":
    X, y, rpm, size, fn = dataset()
    X2, y2, rpm2, size2, fn2 = load_test2()
    te = test_idx(); i2 = np.arange(len(y2))
    tests = {"prueba1_48k": (X, y, rpm, te), "prueba2_12k_limpia": (X2, y2, rpm2, i2)}
    res = {"n": {k: int(len(v[3])) for k, v in tests.items()},
           "prueba2_por_clase_n": [int((y2 == c).sum()) for c in range(4)]}
    for name, mk in [("v1_desplegada", HandNet), ("llm_iter2", it2.make_net), ("llm_iter3", it3.make_net)]:
        res[name] = {k: summary(Y[I], run_nets(mk, XX, R, I)) for k, (XX, Y, R, I) in tests.items()}
        print(name, json.dumps({k: round(v["balanced_acc"], 3) for k, v in res[name].items()}),
              {k: {c: round(a, 2) for c, a in v["per_class"].items()} for k, v in res[name].items()}, flush=True)
    # tamaño 28 (no visto) en prueba 2
    p3 = run_nets(it3.make_net, X2, rpm2, i2)
    res["llm_iter3_prueba2_por_tamano"] = {int(s): float(np.mean(p3[size2 == s] == y2[size2 == s])) for s in np.unique(size2)}
    # gradiente con el mismo feedback (semilla 0), 5 semillas de entrenamiento
    Z1 = spectra(X).astype(np.float32); Z2 = spectra(X2).astype(np.float32)
    tr = feedback_idx(0)
    allrpm = np.concatenate([rpm, rpm2])
    acc = {m: {k: [] for k in tests} for m in ["hand_ft", "residual_only", "cnn"]}
    pcs = {m: {k: [] for k in tests} for m in acc}
    for s in range(5):
        for nm, use in [("hand_ft", True), ("residual_only", False)]:
            h = train_head(Z1, y, rpm, tr, use, s)
            for k, Zt, Y, R, I in [("prueba1_48k", Z1, y, rpm, te), ("prueba2_12k_limpia", Z2, y2, rpm2, i2)]:
                # la cabeza escrita necesita las velocidades de prueba
                p = eval_head(h, Zt, Y, R, I); acc[nm][k].append(bal_acc(Y[I], p)); pcs[nm][k].append(per_class(Y[I], p))
        net, mu, sd = train_cnn(X, y, tr, s)
        for k, (XX, Y, R, I) in tests.items():
            with torch.no_grad():
                p = np.concatenate([net((torch.tensor(XX[I][b:b + 64][:, None]) - mu) / sd).argmax(1).numpy()
                                    for b in range(0, len(I), 64)])
            acc["cnn"][k].append(bal_acc(Y[I], p)); pcs["cnn"][k].append(per_class(Y[I], p))
    res["gradiente_mismo_feedback"] = {m: {k: {"balanced_acc": [float(np.mean(v)), float(np.std(v))],
                                                "per_class_mean": {c: float(np.mean([d[c] for d in pcs[m][k]])) for c in CLASSES}}
                                            for k, v in acc[m].items()} for m in acc}
    print(json.dumps(res["gradiente_mismo_feedback"], indent=0)[:3000])
    json.dump(res, open("results_final_B.json", "w"), indent=1)
