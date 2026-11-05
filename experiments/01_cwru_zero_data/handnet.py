"""Red escrita a mano para diagnóstico de rodamientos (Text-to-Weights).

TODOS los pesos salen de fórmulas basadas en física (análisis de envolvente)
y en la geometría del rodamiento. Nada se ajusta con datos.

Arquitectura (solo capas estándar de PyTorch):
  Conv1d (banco de filtros pasa-banda, pares +h/-h) -> ReLU
  -> Conv1d 1x1 (suma de pares = valor absoluto = rectificación)
  -> AvgPool1d (paso bajo + diezmado: envolvente a 1500 Hz)
  -> Linear (proyecciones de Fourier en 8 fases) -> ReLU -> MaxPool (módulo)
  -> LayerNorm sin parámetros (z-score del espectro de envolvente)
  -> Linear (selección de bins alrededor de cada frecuencia de fallo) -> MaxPool
  -> Linear (suma ponderada de armónicos) -> MaxPool sobre bandas
  -> logits [Normal, IR, B, OR]
"""
import math, hashlib
import numpy as np
import torch, torch.nn as nn

FS = 48000
WIN = 24000
TAPS = 255
POOL = 32                      # envolvente a 48000/32 = 1500 Hz
FENV = FS / POOL
BANDS = [(1000, 3000), (3000, 6000), (6000, 10000), (10000, 16000), (16000, 22000)]
FGRID = np.arange(5.0, 600.0 + 1e-9, 1.0)   # rejilla del espectro de envolvente (Hz)
NPH = 8                        # fases para aproximar el módulo con ReLU + max
TAU = 3.0                      # umbral (z-score) para declarar avería; fijado a priori
CLASSES = ["Normal", "IR", "B", "OR"]

# Geometría SKF 6205-2RS (lado accionamiento)
NB, DBALL, DPITCH, ANG = 9, 0.3126, 1.537, 0.0


def fault_orders():
    r = DBALL / DPITCH * math.cos(ANG)
    bpfo = NB / 2 * (1 - r)
    bpfi = NB / 2 * (1 + r)
    bsf = DPITCH / (2 * DBALL) * (1 - r ** 2)
    ftf = 0.5 * (1 - r)
    return dict(bpfo=bpfo, bpfi=bpfi, bsf=bsf, ftf=ftf)


def templates():
    """(clase, [(múltiplo del eje, peso)]) — armónicos elegidos para no solaparse."""
    o = fault_orders()
    return {
        "IR": [(o["bpfi"], 1.0), (3 * o["bpfi"], 1.0), (o["bpfi"] - 1, 0.5), (o["bpfi"] + 1, 0.5)],
        "B":  [(2 * o["bsf"], 1.0), (4 * o["bsf"], 1.0), (o["bsf"], 0.5)],
        "OR": [(o["bpfo"], 1.0), (2 * o["bpfo"], 1.0)],
    }


def bandpass(lo, hi, n=TAPS):
    t = np.arange(n) - (n - 1) / 2
    def lp(fc):
        return 2 * fc / FS * np.sinc(2 * fc / FS * t)
    h = (lp(hi) - lp(lo)) * np.hamming(n)
    return h / np.sqrt((h ** 2).sum())


class HandNet(nn.Module):
    def __init__(self, rpm):
        super().__init__()
        K = len(BANDS)
        fr = rpm / 60.0
        self.K = K
        # 1) banco de filtros: canal 2k = +h_k, 2k+1 = -h_k
        self.fb = nn.Conv1d(1, 2 * K, TAPS, bias=False)
        w = np.zeros((2 * K, 1, TAPS))
        for k, (lo, hi) in enumerate(BANDS):
            h = bandpass(lo, hi)[::-1]           # conv1d de torch es correlación
            w[2 * k, 0] = h
            w[2 * k + 1, 0] = -h
        # 2) suma de pares -> |y|
        self.absum = nn.Conv1d(2 * K, K, 1, bias=False)
        a = np.zeros((K, 2 * K, 1))
        for k in range(K):
            a[k, 2 * k, 0] = a[k, 2 * k + 1, 0] = 1.0
        # 3) paso bajo y diezmado
        self.pool = nn.AvgPool1d(POOL, POOL)
        L = (WIN - TAPS + 1) // POOL
        self.L = L
        # 4) Fourier en NPH fases con ventana de Hann, sin componente continua
        tt = np.arange(L) / FENV
        hann = np.hanning(L)
        F = len(FGRID)
        P = np.zeros((F * NPH, L))
        for i, f in enumerate(FGRID):
            for j in range(NPH):
                v = hann * np.cos(2 * np.pi * f * tt - 2 * np.pi * j / NPH)
                P[i * NPH + j] = v - v.mean() * hann / hann.mean()
        P /= hann.sum()
        self.fourier = nn.Linear(L, F * NPH, bias=False)
        self.F = F
        self.norm = nn.LayerNorm(F, elementwise_affine=False)
        # 5) selección de bins: para cada (clase, armónico) S ranuras one-hot
        tpl = templates()
        slots, harm_w = [], []
        for c in ["IR", "B", "OR"]:
            hw = []
            for mult, wt in tpl[c]:
                f0 = mult * fr
                tol = 0.01 * f0 + 1.0
                idx = np.where(np.abs(FGRID - f0) <= tol)[0]
                slots.append(idx)
                hw.append(wt)
            harm_w.append(np.array(hw) / np.sum(hw))
        S = max(len(s) for s in slots)
        self.S, self.nh = S, len(slots)
        G = np.zeros((len(slots) * S, F))
        for s, idx in enumerate(slots):
            for q in range(S):
                G[s * S + q, idx[min(q, len(idx) - 1)]] = 1.0   # rellena repitiendo
        self.gather = nn.Linear(F, len(slots) * S, bias=False)
        self.slotmax = nn.MaxPool1d(S, S)
        H = np.zeros((3, len(slots)))
        s = 0
        for c, hw in enumerate(harm_w):
            H[c, s:s + len(hw)] = hw
            s += len(hw)
        self.harm = nn.Linear(len(slots), 3, bias=False)
        # 6) logits: Normal = constante TAU
        self.head = nn.Linear(3, 4)
        Wh = np.zeros((4, 3)); bh = np.zeros(4)
        bh[0] = TAU
        for c in range(3):
            Wh[c + 1, c] = 1.0
        with torch.no_grad():
            for m, v in [(self.fb, w), (self.absum, a), (self.fourier, P), (self.gather, G),
                         (self.harm, H), (self.head, Wh)]:
                m.weight.copy_(torch.tensor(v, dtype=torch.float32))
            self.head.bias.copy_(torch.tensor(bh, dtype=torch.float32))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x, return_scores=False):          # x: (B, 1, WIN)
        y = torch.relu(self.fb(x))
        e = self.pool(self.absum(y))                    # (B, K, L)
        z = torch.relu(self.fourier(e))                 # (B, K, F*NPH)
        B, K, _ = z.shape
        m = z.view(B * K, 1, -1)
        m = nn.functional.max_pool1d(m, NPH, NPH).view(B, K, self.F)
        zs = self.norm(m)                               # z-score por banda
        g = self.gather(zs).view(B * K, 1, -1)
        g = self.slotmax(g).view(B, K, self.nh)
        sc = self.harm(g)                               # (B, K, 3)
        sc = sc.max(dim=1).values                       # mejor banda (MaxPool sobre K)
        out = self.head(sc)
        return (out, sc) if return_scores else out


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def weights_hash(rpms=(1730, 1750, 1772, 1797)):
    h = hashlib.sha256()
    for r in rpms:
        for p in HandNet(r).state_dict().values():
            h.update(p.numpy().tobytes())
    return h.hexdigest()
