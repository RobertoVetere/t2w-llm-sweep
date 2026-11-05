"""Tarea: reconocimiento de actividad con el móvil en la cintura (UCI HAR, 561 rasgos).

Conocimiento usado (documentación, no datos): rasgos normalizados a [-1, 1]; el móvil
va en la cintura con el eje X vertical cuando la persona está de pie; la gravedad y el
movimiento corporal están separados en rasgos 'Gravity' y 'Body'.

Reglas escritas por el LLM (umbrales fijados a priori):
- Estático vs dinámico: tBodyAccMag-mean() bajo => quieto.  Umbral -0.75.
- Tumbado: la gravedad deja de caer sobre X => tGravityAcc-mean()-X bajo. Umbral 0.4.
- Sentado vs de pie: sentado, la pelvis se inclina y X recibe algo menos de gravedad
  que de pie. Umbral en X: 0.82.
- Bajar escaleras = más impactos (jerk alto); subir = cadencia lenta (jerk bajo);
  andar en medio. Umbrales en tBodyAccJerkMag-mean(): -0.15 y -0.45.

Arquitectura: Linear(561->5 rasgos, one-hot) -> Linear(5->6 logits) a trozos con ReLU.
"""
import numpy as np, torch, torch.nn as nn

FEAT = {"gX": 41, "magMean": 201, "jerkMean": 227}   # índices 1-based de features.txt
CLASSES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


class HARNet(nn.Module):
    def __init__(self, k=10.0):
        super().__init__()
        names = list(FEAT)
        self.pick = nn.Linear(561, len(names), bias=False)
        P = np.zeros((len(names), 561))
        for i, n in enumerate(names):
            P[i, FEAT[n] - 1] = 1
        # capa oculta: unidades ReLU que miden "cuánto pasa" cada umbral
        # h0 = dyn = relu(mag + 0.75) ; h1 = static = relu(-0.75 - mag)
        # h2 = lay = relu(0.4 - gX)   ; h3 = upright = relu(gX - 0.4)
        # h4 = sit = relu(0.82 - gX)  ; h5 = stand = relu(gX - 0.82)
        # h6 = down = relu(jerk + 0.15); h7 = up = relu(-0.45 - jerk)
        # h8 = walk_mid = relu(0.15 - |jerk + 0.30|) aproximado con dos ReLU
        H = np.zeros((10, 3)); b = np.zeros(10)
        g, m, j = 0, 1, 2
        H[0, m], b[0] = 1, 0.75
        H[1, m], b[1] = -1, -0.75
        H[2, g], b[2] = -1, 0.4
        H[3, g], b[3] = 1, -0.4
        H[4, g], b[4] = -1, 0.82
        H[5, g], b[5] = 1, -0.82
        H[6, j], b[6] = 1, 0.15
        H[7, j], b[7] = -1, -0.45
        H[8, j], b[8] = 1, 0.30      # relu(j+0.30)
        H[9, j], b[9] = -1, -0.30    # relu(-(j+0.30))
        self.hid = nn.Linear(3, 10)
        # salida: dinámicas suman dyn, estáticas suman static; luego subreglas
        O = np.zeros((6, 10)); ob = np.zeros(6)
        for c in (0, 1, 2):
            O[c, 0] = 1; O[c, 1] = -1
        for c in (3, 4, 5):
            O[c, 1] = 1; O[c, 0] = -1
        O[5, 2] = 2;  O[3, 3] = 0.5; O[4, 3] = 0.5       # tumbado vs erguido
        O[3, 4] = 1;  O[4, 5] = 1                         # sentado vs de pie
        O[3, 2] = -1                                      # "sentado" satura si está tumbado
        O[2, 6] = 1;  O[1, 7] = 1                         # bajar / subir
        O[0, 8] = -1; O[0, 9] = -1; ob[0] = 0.15          # andar: jerk cerca de -0.30
        O[1, 8] = -0.3; O[2, 9] = -0.3
        self.out = nn.Linear(10, 6)
        with torch.no_grad():
            self.pick.weight.copy_(torch.tensor(P, dtype=torch.float32))
            self.hid.weight.copy_(torch.tensor(H * k, dtype=torch.float32))
            self.hid.bias.copy_(torch.tensor(b * k, dtype=torch.float32))
            self.out.weight.copy_(torch.tensor(O, dtype=torch.float32))
            self.out.bias.copy_(torch.tensor(ob * k, dtype=torch.float32))

    def forward(self, x):
        return self.out(torch.relu(self.hid(self.pick(x))))
