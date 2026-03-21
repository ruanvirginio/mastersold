# =============================================================================
# debug.py — Diagnóstico do problema dtype='string' no Keras
# Execute: python debug.py
# =============================================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd

print("=" * 60)
print("1. VERSÕES")
print("=" * 60)
import tensorflow as tf
import keras
print(f"   TensorFlow : {tf.__version__}")
print(f"   Keras      : {keras.__version__}")
print(f"   NumPy      : {np.__version__}")
print(f"   Pandas     : {pd.__version__}")

print("\n" + "=" * 60)
print("2. CARREGANDO DADOS")
print("=" * 60)
DATA_PATH    = "bases_tratadas/daily_peak_transformers_dataset.csv"
WEATHER_PATH = "bases_tratadas/weather_dataset_2022-2024.csv"

df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
print(f"   Colunas do dataset: {list(df.columns)}")
print(f"   dtypes:\n{df.dtypes}")

print("\n" + "=" * 60)
print("3. CARREGANDO WEATHER")
print("=" * 60)
try:
    w = pd.read_csv(WEATHER_PATH, sep=";", encoding="latin-1")
    print(f"   Colunas weather: {list(w.columns)}")
    print(f"   dtypes weather:\n{w.dtypes}")
except Exception as e:
    print(f"   ERRO ao carregar weather: {e}")
    w = None

print("\n" + "=" * 60)
print("4. TESTANDO FEATURES CALENDÁRIO")
print("=" * 60)
import holidays

tid = df["id"].unique()[0]
df_tr = df[df["id"] == tid].copy()
df_tr["Time"] = pd.to_datetime(df_tr["datahora"])
df_tr = df_tr.set_index("Time").sort_index()
full_idx = pd.date_range(df_tr.index.min().normalize(),
                         df_tr.index.max().normalize(), freq="D")
df_tr = df_tr.reindex(full_idx)
df_tr = df_tr.loc["2021-01-01":]
df_tr["Smax"] = df_tr["Smax"].interpolate(method="time", limit=7).ffill(limit=2)
y = df_tr["Smax"]

br = holidays.Brazil()
cal = pd.DataFrame(index=y.index)
cal["weekday"]       = y.index.weekday
cal["month"]         = y.index.month
cal["dayofyear"]     = y.index.dayofyear
cal["year"]          = y.index.year
cal["weekday_sin"]   = np.sin(2*np.pi*cal["weekday"] / 7)
cal["weekday_cos"]   = np.cos(2*np.pi*cal["weekday"] / 7)
cal["month_sin"]     = np.sin(2*np.pi*cal["month"]   / 12)
cal["month_cos"]     = np.cos(2*np.pi*cal["month"]   / 12)
cal["dayofyear_sin"] = np.sin(2*np.pi*cal["dayofyear"] / 365)
cal["dayofyear_cos"] = np.cos(2*np.pi*cal["dayofyear"] / 365)
cal["is_weekend"]    = (y.index.weekday >= 5).astype(int)
cal["is_holiday"]    = y.index.isin(br).astype(int)

print(f"   Colunas cal: {list(cal.columns)}")
print(f"   dtypes cal:\n{cal.dtypes}")

print("\n" + "=" * 60)
print("5. AFTER JOIN COM WEATHER")
print("=" * 60)
if w is not None:
    w["DATA"] = pd.to_datetime(w["DATA"])
    w = w.set_index("DATA").sort_index()
    cols = ["temp_max_anos","temp_min_anos","temp_media_anos","precipitacao_anos"]
    w = w[[c for c in cols if c in w.columns]]
    feat_df = cal.join(w, how="left").bfill().ffill()
else:
    feat_df = cal

print(f"   Todas as colunas após join:\n{feat_df.dtypes}")
print(f"\n   Colunas NÃO numéricas:")
non_num = feat_df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"   {non_num if non_num else 'Nenhuma — tudo OK!'}")

print("\n" + "=" * 60)
print("6. TESTANDO ARRAY NUMPY")
print("=" * 60)
feat_df_num = feat_df.select_dtypes(include=[np.number])
for col in feat_df_num.columns:
    feat_df_num[col] = pd.to_numeric(feat_df_num[col], errors="coerce").astype(np.float32)
feat_df_num = feat_df_num.fillna(0.0)

X = feat_df_num.to_numpy(dtype=np.float32)
print(f"   X.shape  : {X.shape}")
print(f"   X.dtype  : {X.dtype}")
print(f"   X tem NaN: {np.isnan(X).any()}")
print(f"   X tem Inf: {np.isinf(X).any()}")

print("\n" + "=" * 60)
print("7. TESTANDO MODELO KERAS MÍNIMO")
print("=" * 60)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam

# Criar janelas mínimas
lb = 10
Xw = np.array([X[i-lb:i] for i in range(lb, lb+50)], dtype=np.float32)
yw = y.values[lb:lb+50].astype(np.float32)

print(f"   Xw.shape : {Xw.shape}  dtype={Xw.dtype}")
print(f"   yw.shape : {yw.shape}  dtype={yw.dtype}")

model = Sequential([
    Input(shape=(lb, X.shape[1])),
    LSTM(16),
    Dense(1)
])
model.compile(Adam(1e-3), "mse")

try:
    model.fit(Xw, yw, epochs=1, verbose=0)
    print("   ✅ model.fit() funcionou sem erros!")
except Exception as e:
    print(f"   ❌ ERRO no model.fit(): {e}")
    print(f"\n   Tipos dos arrays passados para fit:")
    print(f"   Xw dtype = {Xw.dtype}")
    print(f"   yw dtype = {yw.dtype}")

print("\n" + "=" * 60)
print("DIAGNÓSTICO CONCLUÍDO")
print("=" * 60)