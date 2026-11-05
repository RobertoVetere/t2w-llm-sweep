#!/usr/bin/env bash
# Descarga los datasets públicos usados en los experimentos en ./data (o en $T2W_DATA).
set -euo pipefail
DATA="${T2W_DATA:-$(cd "$(dirname "$0")/.." && pwd)/data}"
mkdir -p "$DATA" && cd "$DATA"

# 1) CWRU Bearing Data Center (copia en NumPy): solo lado de accionamiento a 48 kHz + sanos
if [ ! -d CWRU_Bearing_NumPy ]; then
  git clone --depth 1 --filter=blob:none --no-checkout https://github.com/srigas/CWRU_Bearing_NumPy.git
  (cd CWRU_Bearing_NumPy && git ls-tree -r --name-only HEAD | grep -E 'DE48|Normal' > /tmp/t2w_sel.txt \
     && git checkout HEAD --pathspec-from-file=/tmp/t2w_sel.txt)
fi

# 2) UCI HAR (561 rasgos, particiones oficiales)
if [ ! -d HumanActivityRecognition ]; then
  git clone --depth 1 --filter=blob:none --no-checkout https://github.com/juandapradam12/HumanActivityRecognition.git
  (cd HumanActivityRecognition && git checkout HEAD -- X_train.txt X_test.txt y_train.txt y_test.txt \
     subject_train.txt subject_test.txt features.txt activity_labels.txt)
fi

# 3) Free Spoken Digit Dataset
[ -d free-spoken-digit-dataset ] || git clone --depth 1 https://github.com/Jakobovski/free-spoken-digit-dataset.git

echo "Datos en $DATA"
