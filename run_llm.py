"""Descarga un GGUF oficial de Qwen, verifica su SHA-256 contra el que publica Hugging Face
y genera N respuestas al prompt. Uso: python run_llm.py REPO PATRON SALIDA"""
import hashlib, json, os, sys, time
from huggingface_hub import HfApi, hf_hub_download
from llama_cpp import Llama

repo, pattern, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(os.environ.get("N_SAMPLES", "3"))
os.makedirs(outdir, exist_ok=True)

info = HfApi().model_info(repo, files_metadata=True)
files = sorted(s for s in info.siblings if s.rfilename.endswith(".gguf") and pattern in s.rfilename.lower())
assert files, f"no hay archivos con {pattern}"
meta = {"repo": repo, "revision": info.sha, "files": []}
paths = []
for s in files:
    p = hf_hub_download(repo, s.rfilename, revision=info.sha)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    ok = h.hexdigest() == s.lfs.sha256
    meta["files"].append({"file": s.rfilename, "sha256": h.hexdigest(), "official_sha256": s.lfs.sha256, "match": ok})
    assert ok, f"SHA no coincide en {s.rfilename}"
    paths.append(p)

llm = Llama(model_path=paths[0], n_ctx=4096, n_threads=os.cpu_count(), verbose=False)
prompt = open("prompt_bearing.txt").read()
for i in range(1, N + 1):
    t = time.time()
    r = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                   temperature=0.7, top_p=0.95, max_tokens=2048, seed=i)
    txt = r["choices"][0]["message"]["content"]
    open(f"{outdir}/sample_{i}.txt", "w").write(txt)
    meta[f"sample_{i}"] = {"seconds": round(time.time() - t, 1), "tokens": r["usage"]["completion_tokens"]}
    print(outdir, i, meta[f"sample_{i}"], flush=True)
json.dump(meta, open(f"{outdir}/meta.json", "w"), indent=1)
