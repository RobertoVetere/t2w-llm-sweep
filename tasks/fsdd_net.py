"""Tarea: dígitos hablados en inglés (Free Spoken Digit Dataset, 8 kHz). Red escrita a mano.

Conocimiento usado (documentación + fonética, no datos): grabaciones a 8 kHz con el
silencio recortado. Cada dígito se describe por sus fonemas al principio, en medio y al
final, y cada fonema por su forma espectral: formantes de vocales (F1, F2, F3 típicos del
inglés), fricativas (/s/ energía alta >3 kHz; /f/, /θ/ planas y débiles; /z/ = /s/ + sonoridad),
nasales (energía concentrada <300 Hz), oclusiva /t/ (ráfaga de banda ancha, más aguda).

Arquitectura: Conv1d (banco de Gabor, pares +h/-h) -> ReLU -> Conv1d 1x1 (|.|)
 -> AvgPool (tramas de 25 ms) -> LayerNorm sobre bandas (forma espectral)
 -> AdaptiveAvgPool(3 tramos: inicio/medio/fin) -> Linear (plantillas fonéticas) -> 10 logits.
"""
import numpy as np, torch, torch.nn as nn

FS = 8000
NB = 18
CENTERS = np.geomspace(150, 3800, NB)
KER = 129

# --- fonemas como perfil de amplitud lineal sobre CENTERS -----------------------------
f = CENTERS


def vowel(F1, F2, F3=2600):
    a = sum(w / np.sqrt(1 + ((f - F) / B) ** 2) for F, w, B in [(F1, 1.0, 90), (F2, 0.6, 120), (F3, 0.3, 160)])
    return a / (1 + f / 800)          # inclinación glotal


PH = {
    "s":  np.exp(-0.5 * ((f - 3600) / 500) ** 2) + 0.05,
    "f":  np.full(NB, 0.3),
    "th": np.full(NB, 0.3),
    "z":  np.exp(-0.5 * ((f - 3600) / 500) ** 2) + 0.6 * np.exp(-0.5 * ((f - 180) / 60) ** 2),
    "n":  np.exp(-0.5 * ((f - 250) / 80) ** 2) + 0.05 * vowel(500, 1500),
    "w":  vowel(300, 700, 2300),
    "t":  0.2 + f / 4000,
    "r":  vowel(420, 1100, 1650),
    "v":  0.5 * np.exp(-0.5 * ((f - 180) / 70) ** 2) + 0.15,
    "k":  0.2 + 0.8 * np.exp(-0.5 * ((f - 2000) / 600) ** 2),
    "i":  vowel(280, 2300, 3000),
    "I":  vowel(400, 1950, 2600),
    "e":  vowel(450, 2000, 2600),
    "E":  vowel(580, 1800, 2500),
    "a":  vowel(750, 1300, 2500),
    "A":  vowel(650, 1200, 2500),     # /ʌ/
    "o":  vowel(450, 900, 2400),
    "O":  vowel(550, 850, 2400),      # /ɔ/
    "u":  vowel(300, 900, 2300),
    "@":  vowel(500, 1500, 2500),     # schwa
}

# (inicio, medio, fin), cada tramo es una mezcla de fonemas
WORDS = {
    0: ([("z", .6), ("I", .4)], [("I", .4), ("r", .6)], [("o", 1.0)]),
    1: ([("w", 1.0)], [("A", 1.0)], [("A", .3), ("n", .7)]),
    2: ([("t", .5), ("u", .5)], [("u", 1.0)], [("u", 1.0)]),
    3: ([("th", .5), ("r", .5)], [("i", 1.0)], [("i", 1.0)]),
    4: ([("f", .6), ("O", .4)], [("O", 1.0)], [("O", .4), ("r", .6)]),
    5: ([("f", .6), ("a", .4)], [("a", 1.0)], [("I", .5), ("v", .5)]),
    6: ([("s", 1.0)], [("I", .7), ("k", .3)], [("s", 1.0)]),
    7: ([("s", 1.0)], [("E", .7), ("v", .3)], [("@", .3), ("n", .7)]),
    8: ([("e", 1.0)], [("I", 1.0)], [("t", .7), ("I", .3)]),
    9: ([("n", .7), ("a", .3)], [("a", 1.0)], [("I", .3), ("n", .7)]),
}


def gabor(fc):
    t = (np.arange(KER) - KER // 2) / FS
    sf = max(110.0, 0.18 * fc)                 # ancho de banda
    st = 1 / (2 * np.pi * sf)
    h = np.exp(-0.5 * (t / st) ** 2) * np.cos(2 * np.pi * fc * t)
    return h / np.abs(h).sum()


class FSDDNet(nn.Module):
    def __init__(self, scale=6.0):
        super().__init__()
        self.fb = nn.Conv1d(1, 2 * NB, KER, padding=KER // 2, bias=False)
        w = np.zeros((2 * NB, 1, KER))
        for k, fc in enumerate(CENTERS):
            w[2 * k, 0] = gabor(fc); w[2 * k + 1, 0] = -gabor(fc)
        self.absum = nn.Conv1d(2 * NB, NB, 1, bias=False)
        a = np.zeros((NB, 2 * NB, 1))
        for k in range(NB):
            a[k, 2 * k, 0] = a[k, 2 * k + 1, 0] = 1
        self.frames = nn.AvgPool1d(200, 80)
        self.norm = nn.LayerNorm(NB, elementwise_affine=False)
        self.seg = nn.AdaptiveAvgPool1d(3)
        T = np.zeros((10, NB * 3))
        for d, segs in WORDS.items():
            for s, mix in enumerate(segs):
                p = sum(wt * PH[ph] for ph, wt in mix)
                p = (p - p.mean()) / p.std()
                T[d, s::3] = p / np.sqrt(NB * 3)     # disposición (banda, tramo) aplanada
        self.match = nn.Linear(NB * 3, 10, bias=False)
        with torch.no_grad():
            self.fb.weight.copy_(torch.tensor(w, dtype=torch.float32))
            self.absum.weight.copy_(torch.tensor(a, dtype=torch.float32))
            self.match.weight.copy_(torch.tensor(T * scale, dtype=torch.float32))

    def features(self, x):                   # x: (1, 1, n)
        e = self.frames(self.absum(torch.relu(self.fb(x))))      # (1, NB, frames)
        z = self.norm(e.transpose(1, 2)).transpose(1, 2)         # forma espectral por trama
        return self.seg(z).flatten(1)                            # (1, NB*3)

    def forward(self, x):
        return self.match(self.features(x))
