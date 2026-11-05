"""Iteración 3 (fase B): red reescrita por Claude Opus 5.5 tras el análisis libre del feedback
(reports/analysis_B_feedback.txt). Sin gradientes: todos los pesos salen de fórmulas o de valores
elegidos razonando sobre el informe.

RAZONAMIENTO
------------
1. Nivel de vibración. En el feedback, las 5 ventanas sanas tienen RMS 0,063–0,065 g y las 15
   averiadas tienen 0,14 g o más. La v1 normalizaba cada espectro (LayerNorm) y tiraba este
   dato, que es el primer indicador que mira un técnico (severidad según ISO 10816).
   -> Canal nuevo: amplitud media |x| de la señal filtrada por debajo de 5,5 kHz. Se filtra
      para que mida lo mismo en las dos pruebas, porque la prueba 2 solo tiene hasta 6 kHz.
2. Líneas de pista: el espectro de envolvente de la banda 2–5 kHz muestra BPFI muy marcada en
   los IR (z robusto 10–56) y BPFO en los OR (13–33). La excepción es un OR de 14 milésimas sin
   líneas. Se usa una sola banda, 2–5 kHz, presente en ambas pruebas, en vez de las 5 bandas
   hasta 22 kHz de la v1.
3. Bola por exclusión: las líneas de bola (BSF, 2xBSF) no aparecen de forma consistente en
   ninguna de las 5 bolas. Física: una bola dañada gira y no siempre golpea con el defecto. Regla
   de técnico: si vibra por encima de lo sano y no hay línea de pista que lo explique, es la bola.
4. Lógica de decisión, toda con ReLU y capas lineales:
      Normal = 3
      IR     = 3 + 2·(zBPFI − 3)
      OR     = 3 + 2·(zBPFO + 0,3·z2BPFO − 3)
      B      = 3 + min(k·(A − A0), 3) − 2·relu(max(zBPFI, zBPFO) − 2,5)
   Los umbrales A0 y k se fijan en el paso 5, mirando los rasgos del feedback con `inspect`.
5. Valores elegidos tras ver los rasgos del feedback (reports/iter3_features.txt):
   - Las sanas tienen A = 0,045–0,047 y la avería más suave, A = 0,1135.
     A0 = 0,075 queda en el punto medio geométrico, a unas 1,6 veces el nivel sano: el
     criterio habitual de alarma en mantenimiento (de 1,6 a 2,5 veces la línea base).
   - K = 80 hace que la severidad sature (3) hacia A ≈ 0,11, el nivel de la avería más leve.
   Previsión en el feedback: 18/20 aciertos. Fallan la bola con BPFO = 5,6 (que se lee como OR)
   y el OR de 14 milésimas sin líneas (que se lee como bola). Son los dos casos que la física de
   la envolvente no separa, y no se fuerzan.
   No hay iteración 4: los errores que quedan no tienen una explicación física que corregir.
"""
import sys
import numpy as np, torch, torch.nn as nn
from common import HandNet, fault_orders, FGRID, NPH

FS, TAPS, POOL = 48000, 255, 32
A0 = 0.075
K = 80.0


def fir_lowpass(fc, n=TAPS):
    t = np.arange(n) - (n - 1) / 2
    h = 2 * fc / FS * np.sinc(2 * fc / FS * t) * np.hamming(n)
    return h / h.sum()


def fir_bandpass(lo, hi, n=TAPS):
    t = np.arange(n) - (n - 1) / 2
    lp = lambda fc: 2 * fc / FS * np.sinc(2 * fc / FS * t)
    h = (lp(hi) - lp(lo)) * np.hamming(n)
    return h / np.sqrt((h ** 2).sum())


class Iter3(nn.Module):
    def __init__(self, rpm, a0=None, k=None):
        super().__init__()
        a0 = A0 if a0 is None else a0; k = K if k is None else k
        base = HandNet(rpm)
        self.fourier, self.norm, self.F = base.fourier, base.norm, base.F
        # canal de amplitud: paso bajo 5,5 kHz en pares ±, |x| y media
        self.lp = nn.Conv1d(1, 2, TAPS, bias=False)
        # canal de líneas: paso banda 2–5 kHz en pares ±, envolvente a 1500 Hz
        self.bp = nn.Conv1d(1, 2, TAPS, bias=False)
        self.pool = nn.AvgPool1d(POOL, POOL)
        fr = rpm / 60.0; o = fault_orders()
        lines = {"BPFI": o["bpfi"], "BPFO": o["bpfo"], "2BPFO": 2 * o["bpfo"], "2BSF": 2 * o["bsf"], "FTF": o["ftf"], "1X": 1.0}
        self.names = list(lines); self.ix = {n: i for i, n in enumerate(self.names)}
        slots = [np.where(np.abs(FGRID - v * fr) <= 0.01 * v * fr + 1.0)[0] for v in lines.values()]
        S = max(len(s) for s in slots); self.S = S
        G = np.zeros((len(slots) * S, self.F))
        for s, idx in enumerate(slots):
            for q in range(S):
                G[s * S + q, idx[min(q, len(idx) - 1)]] = 1
        self.gather = nn.Linear(self.F, len(slots) * S, bias=False)
        with torch.no_grad():
            hl = fir_lowpass(5500)[::-1].copy(); hb = fir_bandpass(2000, 5000)[::-1].copy()
            self.lp.weight.copy_(torch.tensor(np.stack([hl, -hl])[:, None], dtype=torch.float32))
            self.bp.weight.copy_(torch.tensor(np.stack([hb, -hb])[:, None], dtype=torch.float32))
            self.gather.weight.copy_(torch.tensor(G, dtype=torch.float32))
        self.a0, self.k = a0, k
        for p in self.parameters():
            p.requires_grad_(False)

    def features(self, x):
        amp = torch.relu(self.lp(x)).sum(1).mean(1, keepdim=True)                 # A = media |x| (<5,5 kHz)
        env = self.pool(torch.relu(self.bp(x)).sum(1, keepdim=True))              # (B,1,L) envolvente
        z = torch.relu(self.fourier(env)); B = z.shape[0]
        m = nn.functional.max_pool1d(z.view(B, 1, -1), NPH, NPH).view(B, self.F)
        zs = self.norm(m)
        g = nn.functional.max_pool1d(self.gather(zs)[:, None], self.S, self.S)[:, 0]   # z por línea
        return amp, g

    def forward(self, x):
        A, g = self.features(x); ix = self.ix
        zI, zO = g[:, [ix["BPFI"]]], g[:, [ix["BPFO"]]] + 0.3 * g[:, [ix["2BPFO"]]]
        race = torch.maximum(g[:, [ix["BPFI"]]], g[:, [ix["BPFO"]]])
        a = self.k * (A - self.a0)
        sev = a - torch.relu(a - 3.0)                                               # min(a, 3)
        normal = torch.full_like(A, 3.0)
        ir = 3 + 2 * (zI - 3); orr = 3 + 2 * (zO - 3)
        b = 3 + sev - 2 * torch.relu(race - 2.5)
        return torch.cat([normal, ir, b, orr], 1)


def make_net(rpm):
    return Iter3(rpm)


if __name__ == "__main__":       # paso 5: inspeccionar rasgos en el feedback (solo feedback)
    from common import dataset, feedback_idx, CLASSES
    X, y, rpm, size, fn = dataset(); idx = feedback_idx()
    print("real tam rpm |  A(media|x|<5.5k)  zBPFI  zBPFO  z2BPFO  z2BSF  zFTF  z1X")
    for j in idx:
        net = Iter3(int(rpm[j]), a0=0.0, k=1.0)
        with torch.no_grad():
            A, g = net.features(torch.tensor(X[j][None, None]))
        print(f"{CLASSES[y[j]]:6s} {size[j]:2d} {rpm[j]} | {A.item():.4f} " + " ".join(f"{v:6.1f}" for v in g[0].numpy()))
