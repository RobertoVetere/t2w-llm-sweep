"""Iteración 1: edición de pesos escrita por Claude Opus 5.5 leyendo reports/report_v1.md.

RAZONAMIENTO (a partir del informe del feedback, 20 ventanas a 1730/1750 rpm)
---------------------------------------------------------------------------------
1. Pista interior (IR): 3 de 5 ventanas se clasifican como sanas, aunque BPFI aparece con
   z≈5,7 en el 100 % de ellas. Causa: v1 promedia BPFI con 3xBPFI y dos bandas laterales,
   que son débiles, y la media diluida (≈2,3) queda por debajo del umbral (3).
   -> Edición: la puntuación IR pasa a ser la línea fundamental BPFI con peso 1 más apoyos
      pequeños sin normalizar (2xBPFI 0,3; BPFI±1x 0,2), en las mismas unidades z.

2. Pista exterior (OR): bien detectada por BPFO (z≈6,1). Misma corrección de escala:
   OR = BPFO + 0,3·2xBPFO (antes era la media, que diluía la línea fundamental).

3. Bola (B): es la capacidad que falta (0 de 5). El informe muestra por qué: las líneas de
   bola (BSF, 2xBSF) NO aparecen (0 % de ventanas con z>3). Es coherente con la física de
   este banco: la bola dañada gira, su defecto no golpea en cada paso y la línea 2xBSF se
   emborrona. Lo que sí aparece:
     - envolvente modulada con fuerza en la banda de resonancia baja (1–3 kHz):
       pico máximo z≈4,5 frente a 1,2 en sanas;
     - modulación a la frecuencia de jaula (FTF, z≈1,3 frente a 0,1 en sanas), que es la
       firma física de una bola dañada, ya que esta entra en la zona de carga una vez por
       vuelta de la jaula;
     - evidencia débil de pista (BPFO≈2), no dominante.
   -> Capacidad nueva: "diagnóstico por exclusión con firma de jaula". Una ventana es de
      bola si hay impactos modulados en 1–3 kHz y modulación a FTF, pero no domina una
      línea de pista. Se escribe así:
         logit_B = G1 - 0,5 + 0,5·z(FTF) - 0,5·max(z(BPFI), z(BPFO))
      donde G1 es el máximo del espectro de envolvente (z) en la banda 1–3 kHz.
      Se usa solo la banda 1–3 kHz porque en 3–6 kHz las ventanas sanas tienen un tono
      propio de la máquina (orden 3,96, z≈3,5) que no es de rodamiento.
   Con las medias del informe: B≈3,6 (>3, el umbral de sano); sana≈0,6; IR: B≈5,1 < IR≈6,6;
   OR: B≈5,2 < OR≈6,7. El término de supresión evita que IR y OR se lean como bola.

4. Sano: sin cambios (logit constante 3).

No se usa gradiente ni búsqueda automática; todos los números salen del razonamiento anterior.
"""
import numpy as np, torch, torch.nn as nn
from common import HandNet, fault_orders, FGRID, NPH

TAU = 3.0


class Iter1(nn.Module):
    def __init__(self, rpm):
        super().__init__()
        base = HandNet(rpm)                       # reutiliza el frente físico de v1, intacto
        self.fb, self.absum, self.pool, self.fourier, self.norm = base.fb, base.absum, base.pool, base.fourier, base.norm
        self.F = base.F
        fr = rpm / 60.0
        o = fault_orders()
        # líneas a leer (orden del eje): índice -> nombre
        lines = {"BPFI": o["bpfi"], "2BPFI": 2 * o["bpfi"], "BPFI-1": o["bpfi"] - 1, "BPFI+1": o["bpfi"] + 1,
                 "BPFO": o["bpfo"], "2BPFO": 2 * o["bpfo"], "FTF": o["ftf"]}
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
        # cabeza: features = [7 líneas (máx sobre bandas)] + [G1] + [max(BPFI,BPFO)]
        n = len(self.names); ix = {k: i for i, k in enumerate(self.names)}
        W = np.zeros((4, n + 2)); b = np.zeros(4)
        b[0] = TAU                                                   # Normal
        W[1, ix["BPFI"]] = 1.0; W[1, ix["2BPFI"]] = 0.3; W[1, ix["BPFI-1"]] = 0.2; W[1, ix["BPFI+1"]] = 0.2   # IR
        W[2, n] = 1.0; b[2] = -0.5; W[2, ix["FTF"]] = 0.5; W[2, n + 1] = -0.5                                  # B
        W[3, ix["BPFO"]] = 1.0; W[3, ix["2BPFO"]] = 0.3                                                       # OR
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
        zs = self.norm(m)                                             # (B, K, F)
        g = nn.functional.max_pool1d(self.gather(zs).view(B * K, 1, -1), self.S, self.S).view(B, K, -1)
        lines = g.max(1).values                                      # mejor banda por línea
        G1 = zs[:, 0, :].max(1, keepdim=True).values                 # banda 1–3 kHz
        race = torch.maximum(lines[:, [self.ix["BPFI"]]], lines[:, [self.ix["BPFO"]]])
        return self.head(torch.cat([lines, G1, race], 1))


def make_net(rpm):
    return Iter1(rpm)
