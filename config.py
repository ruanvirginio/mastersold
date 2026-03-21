# =============================================================================
# config.py — Configuração Central do Projeto
# Mestrado em TI — Previsão de Potência Aparente em Transformadores
# =============================================================================
# EDITE APENAS ESTE ARQUIVO para alterar IDs, datas ou hiperparâmetros.
# Todos os scripts importam daqui, garantindo consistência total.
# =============================================================================

import os
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CAMINHOS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = "bases_tratadas/daily_peak_transformers_dataset.csv"
WEATHER_PATH = "bases_tratadas/weather_dataset_2022-2024.csv"

OUTPUT_EDA         = "eda_out"
OUTPUT_UV          = "resultados_UV"
OUTPUT_MV          = "resultados_MV"
OUTPUT_DL          = "resultados_DL"
OUTPUT_SARIMAX     = "resultados_SARIMAX"
OUTPUT_SVR         = "resultados_SVR"
OUTPUT_TRANSFORMER = "resultados_Transformer"
OUTPUT_COMPARE     = "resultados_comparacao"
OUTPUT_FUTURE      = "previsao_futura"
OUTPUT_REPORT      = "relatorio"

for d in [OUTPUT_EDA, OUTPUT_UV, OUTPUT_MV, OUTPUT_DL,
          OUTPUT_SARIMAX, OUTPUT_SVR, OUTPUT_TRANSFORMER,
          OUTPUT_COMPARE, OUTPUT_FUTURE, OUTPUT_REPORT]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMADORES  ← mesmos do EDA
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORMADORES = ["T21a", "T22a", "T70a"]

MAPA_PLOT = {
    "T21a": "T1",
    "T22a": "T2",
    "T70a": "T3",
}

# ─────────────────────────────────────────────────────────────────────────────
# JANELA TEMPORAL — Partição 3 conjuntos (sem data leakage)
#
#   ┌─────────────────┬──────────────┬─────────────┐
#   │    TREINO       │  VALIDAÇÃO   │   TESTE     │
#   │  2021 – 2022    │    2023      │    2024     │
#   └─────────────────┴──────────────┴─────────────┘
#   Treino  → ajuste dos parâmetros do modelo
#   Validação → seleção de hiperparâmetros (HPT) e early stopping
#   Teste   → avaliação final única (nunca tocado durante HPT)
# ─────────────────────────────────────────────────────────────────────────────
START_DATE     = "2021-01-01"
END_TRAIN      = "2022-12-31"
END_VALIDATION = "2023-12-31"
# Tudo após END_VALIDATION → conjunto de TESTE

# ─────────────────────────────────────────────────────────────────────────────
# PREVISÃO FUTURA REAL
# ─────────────────────────────────────────────────────────────────────────────
# O modelo será retreinado em TODOS os dados (2021–2024) e projetará
# até FUTURE_END_DATE. Sem ground truth → sem métricas, com IC bootstrap.
FUTURE_END_DATE = "2027-12-31"   # 3 anos além de 2024

# ─────────────────────────────────────────────────────────────────────────────
# SÉRIES TEMPORAIS
# ─────────────────────────────────────────────────────────────────────────────
PERIOD         = 365
MAX_GAP_INTERP = 7                            # dias máximos para interpolar
LAGS           = [1, 2, 7, 14, 30, 60, 180, 365]   # lags seletivos
ROLLING_WINDOW = 365
STEPS          = 365                          # horizonte backtesting (1 ano)

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUTIBILIDADE
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
# DEEP LEARNING (LSTM / CNN-LSTM)
# ─────────────────────────────────────────────────────────────────────────────
DL_LOOKBACK    = 365
DL_HORIZON     = 1       # previsão 1-step; iterado para multi-step
DL_EPOCHS      = 150
DL_BATCH_SIZE  = 32
DL_PATIENCE    = 20
DL_LR          = 1e-3

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────
# Modo "leve" → roda em CPU (Ryzen 5700G) em tempo razoável
# Modo "completo" → melhores resultados, pode levar horas sem GPU
TF_LITE_D_MODEL   = 64     # dimensão do modelo (leve: 64 | completo: 256)
TF_LITE_N_HEADS   = 4      # cabeças de atenção (leve: 4 | completo: 8)
TF_LITE_N_LAYERS  = 2      # camadas encoder (leve: 2 | completo: 4)
TF_LITE_FFN_DIM   = 128    # dim feed-forward (leve: 128 | completo: 512)
TF_LITE_DROPOUT   = 0.1

TF_FULL_D_MODEL   = 256
TF_FULL_N_HEADS   = 8
TF_FULL_N_LAYERS  = 4
TF_FULL_FFN_DIM   = 512
TF_FULL_DROPOUT   = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# SVR
# ─────────────────────────────────────────────────────────────────────────────
SVR_LAGS = [1, 2, 7, 14, 30, 60, 180, 365]   # features de lag para SVR

# ─────────────────────────────────────────────────────────────────────────────
# SARIMAX
# ─────────────────────────────────────────────────────────────────────────────
# Ordens testadas na grade HPT
SARIMAX_ORDERS = [
    # (p, d, q) × (P, D, Q, s)
    ((1,1,1), (1,1,1,7)),    # sazonalidade semanal
    ((2,1,1), (1,1,1,7)),
    ((1,1,2), (1,1,1,7)),
    ((2,1,2), (0,1,1,7)),
]

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"