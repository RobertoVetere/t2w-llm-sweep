"""Evalúa el código que escribe otro LLM con prompt_bearing.txt.

Uso: python3 eval_llm_code.py respuesta_modelo.py [más archivos...]
Cada archivo debe definir build_model(rpm). Si no se puede ejecutar, cuenta como fallo.
"""
import json, re, sys, traceback, numpy as np, torch
sys.path.insert(0, "/home/claude/t2w")
from data import load_windows
from evaluate import bal_acc


def extract(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    if m:
        return m[0]
    m = re.search(r"```(?:python)?\n(.*)", text, re.S)   # bloque sin cerrar (respuesta cortada)
    return m.group(1) if m else text


if __name__ == "__main__":
    X, y, rpm, size, fn = load_windows()
    out = {}
    for path in sys.argv[1:]:
        src = extract(open(path).read())
        res = {"runs": False}
        try:
            ns = {}
            exec(src, ns)
            pred = np.zeros(len(y), int)
            for r in np.unique(rpm):
                net = ns["build_model"](float(r)); net.eval(); m = rpm == r
                with torch.no_grad():
                    o = net(torch.tensor(X[m][:, None]))
                assert o.shape == (m.sum(), 4), f"forma de salida {tuple(o.shape)}"
                pred[m] = o.argmax(1).numpy()
            res.update(runs=True, balanced_acc=bal_acc(y, pred, 4),
                       healthy_vs_fault=bal_acc((y > 0).astype(int), (pred > 0).astype(int), 2),
                       pred_hist=np.bincount(pred, minlength=4).tolist())
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out[path] = res
        print(path, res, flush=True)
    json.dump(out, open("results_llm_sweep.json", "w"), indent=1)
