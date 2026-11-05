"""Iteración 2: edición escrita por Claude Opus 5.5 leyendo reports/report_iter1.md (detalle por ventana).

RAZONAMIENTO
------------
Errores de iter1 en el feedback: 1 IR->B, 2 B->IR, 1 B->OR y 1 OR->B.

a) IR de 14 milésimas leídos como bola: BPFI es moderado (z 3,6–3,9), pero la línea del eje
   1x es muy fuerte (6,3–6,4). Física: la pista interior gira con el eje, así que sus impactos
   salen modulados a 1x. Esa modulación es firma de pista interior, no de bola.
   -> IR recibe +0,5·z(1x) y B recibe -0,5·z(1x).
   El G1 alto de estas ventanas es precisamente esa línea 1x, que antes se contaba como
   evidencia de bola.

b) Bolas de 21 milésimas leídas como IR: su BPFI es débil (2,4–2,7), pero la puntuación lineal
   de iter1 la daba por buena.
   -> IR pasa a tener rodilla: 2·(z(BPFI) - 2,5). Una línea BPFI solo cuenta como evidencia
      de pista interior cuando claramente sobresale del ruido (z>2,5); los IR reales del
      feedback tienen z≥3,6.

c) Bola: margen de +0,5 sobre el umbral de sano, porque las sanas tienen G1≤1,5 y queda
   holgura de sobra:
      logit_B = G1 + 0,5·z(FTF) - 0,5·max(z(BPFI), z(BPFO)) - 0,5·z(1x) + 0,5

d) Se aceptan, sin forzarlos, dos casos que la envolvente no puede separar físicamente:
   una bola con BPFO=5,5 (literatura: los registros de bola de CWRU muestran a veces líneas
   de pista) y un OR con BPFO=1,7. Ajustarlos exigiría reglas ad hoc que no generalizarían.

Previsión en el feedback: Normal 5/5, IR 5/5, B 3/5, OR 4/5.
"""
import numpy as np, torch, torch.nn as nn
from common import HandNet, fault_orders, FGRID, NPH

TAU = 3.0


class Iter2(nn.Module):
    def __init__(self, rpm):
        super().__init__()
        base = HandNet(rpm)
        self.fb, self.absum, self.pool, self.fourier, self.norm = base.fb, base.absum, base.pool, base.fourier, base.norm
        self.F = base.F
        fr = rpm / 60.0
        o = fault_orders()
        lines = {"BPFI": o["bpfi"], "2BPFI": 2 * o["bpfi"], "BPFI-1": o["bpfi"] - 1, "BPFI+1": o["bpfi"] + 1,
                 "BPFO": o["bpfo"], "2BPFO": 2 * o["bpfo"], "FTF": o["ftf"], "1X": 1.0}
        self.names = list(lines)
        slots = []
        for v in lines.values():
            f0 = v * fr; tol = 0.01 * f0 + 1.0
            slots.append(np.where(np.abs(FGRID - f0) <= tol)[0])
        S = max(len(s) for s in slots); self.S = S
        G = np.zeros((len(slots) * S, self.F))
        for s, idx in enumerate(slots):
            for q in range(S):
                G[s * S + q, idx[min(q, len(idx) - 1)]] = 1.0
        self.gather = nn.Linear(self.F, len(slots) * S, bias=False)
        n = len(self.names); ix = {k: i for i, k in enumerate(self.names)}
        W = np.zeros((4, n + 2)); b = np.zeros(4)
        b[0] = TAU                                                                     # Normal
        W[1, ix["BPFI"]] = 2.0; b[1] = -5.0                                            # IR: 2·(BPFI-2,5)
        W[1, ix["2BPFI"]] = 0.3; W[1, ix["BPFI-1"]] = 0.2; W[1, ix["BPFI+1"]] = 0.2; W[1, ix["1X"]] = 0.5
        W[2, n] = 1.0; W[2, ix["FTF"]] = 0.5; W[2, n + 1] = -0.5; W[2, ix["1X"]] = -0.5; b[2] = 0.5   # B
        W[3, ix["BPFO"]] = 1.0; W[3, ix["2BPFO"]] = 0.3                                # OR
        self.head = nn.Linear(n + 2, 4)
        with torch.no_grad():
            self.gather.weight.copy_(torch.tensor(G, dtype=torch.float32))
            self.head.weight.copy_(torch.tensor(W, dtype=torch.float32))
            self.head.bias.copy_(torch.tensor(b, dtype=torch.float32))
        self.ix = ix
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        e = self.pool(self.absum(torch.relu(self.fb(x))))
        z = torch.relu(self.fourier(e)); B, K, _ = z.shape
        m = nn.functional.max_pool1d(z.view(B * K, 1, -1), NPH, NPH).view(B, K, self.F)
        zs = self.norm(m)
        g = nn.functional.max_pool1d(self.gather(zs).view(B * K, 1, -1), self.S, self.S).view(B, K, -1)
        lines = g.max(1).values
        G1 = zs[:, 0, :].max(1, keepdim=True).values
        race = torch.maximum(lines[:, [self.ix["BPFI"]]], lines[:, [self.ix["BPFO"]]])
        return self.head(torch.cat([lines, G1, race], 1))


def make_net(rpm):
    return Iter2(rpm)
