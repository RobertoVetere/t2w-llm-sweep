"""Tarea: dígitos manuscritos 8x8 (sklearn load_digits). Red escrita a mano.

Conocimiento usado (documentación del dataset, no datos): cada imagen sale de un
mapa de bits de 32x32 dividido en bloques de 4x4; cada píxel cuenta los puntos
negros del bloque (0..16). Los pesos son plantillas generadas dibujando cada
dígito con trazos (rectas y arcos) descritos por el LLM, sin ver ninguna imagen.

Arquitectura: LayerNorm(64) -> Linear(64 -> 10*9 plantillas desplazadas) -> MaxPool(9) -> logits.
"""
import numpy as np, torch, torch.nn as nn

TH = 1.6  # medio grosor del trazo en píxeles de 32x32


def seg(a, b):
    return [("seg", a, b)]


def arc(c, r, t0, t1, ry=None):
    return [("arc", c, r, ry or r, t0, t1)]


STROKES = {
    0: arc((16, 16), 8, 0, 360, 13),
    1: seg((16, 3), (16, 29)) + seg((11, 8), (16, 3)),
    2: arc((16, 10), 7, 180, 390) + seg((22, 13.5), (8, 29)) + seg((8, 29), (25, 29)),
    3: arc((15, 9), 6.5, -150, 90) + arc((15, 22), 7.5, -90, 150),
    4: seg((19, 3), (7, 21)) + seg((7, 21), (25, 21)) + seg((19, 3), (19, 30)),
    5: seg((9, 3), (23, 3)) + seg((9, 3), (10, 14)) + arc((15, 20), 8, -125, 150),
    6: seg((21, 3), (9, 18)) + arc((15, 22), 7, 0, 360),
    7: seg((7, 3), (25, 3)) + seg((25, 3), (13, 30)),
    8: arc((16, 9), 6, 0, 360) + arc((16, 22), 7.5, 0, 360),
    9: arc((16, 10), 7, 0, 360) + seg((23, 10), (20, 30)),
}


def render(strokes):
    pts = []
    for s in strokes:
        if s[0] == "seg":
            (x0, y0), (x1, y1) = s[1], s[2]
            for u in np.linspace(0, 1, 200):
                pts.append((x0 + u * (x1 - x0), y0 + u * (y1 - y0)))
        else:
            (cx, cy), rx, ry, t0, t1 = s[1], s[2], s[3], s[4], s[5]
            for t in np.radians(np.linspace(t0, t1, 400)):
                pts.append((cx + rx * np.cos(t), cy + ry * np.sin(t)))
    pts = np.array(pts)
    yy, xx = np.mgrid[0:32, 0:32] + 0.5
    d = np.min((xx[..., None] - pts[:, 0]) ** 2 + (yy[..., None] - pts[:, 1]) ** 2, axis=-1)
    bmp = (d <= TH ** 2).astype(float)
    return bmp.reshape(8, 4, 8, 4).sum((1, 3))      # 0..16 como en el dataset


def shift(img, dy, dx):
    out = np.zeros_like(img)
    ys, yd = (slice(dy, 8), slice(0, 8 - dy)) if dy >= 0 else (slice(0, 8 + dy), slice(-dy, 8))
    xs, xd = (slice(dx, 8), slice(0, 8 - dx)) if dx >= 0 else (slice(0, 8 + dx), slice(-dx, 8))
    out[ys, xs] = img[yd, xd]
    return out


class DigitsNet(nn.Module):
    SH = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]

    def __init__(self, scale=8.0):
        super().__init__()
        self.norm = nn.LayerNorm(64, elementwise_affine=False)
        W = []
        for d in range(10):
            base = render(STROKES[d])
            for dy, dx in self.SH:
                t = shift(base, dy, dx).ravel()
                t = (t - t.mean()) / (t.std() * np.sqrt(64))
                W.append(t * scale)
        self.match = nn.Linear(64, 10 * len(self.SH), bias=False)
        with torch.no_grad():
            self.match.weight.copy_(torch.tensor(np.array(W), dtype=torch.float32))
        self.pool = nn.MaxPool1d(len(self.SH))

    def forward(self, x):                  # x: (B, 64), valores 0..16
        s = self.match(self.norm(x))
        return self.pool(s[:, None]).squeeze(1)


if __name__ == "__main__":
    for d in range(10):
        print(d); print(render(STROKES[d]).astype(int))
