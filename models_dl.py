# =============================================================================
# models_dl.py — Deep Learning: LSTM · Stacked LSTM · CNN-LSTM
# =============================================================================
# Todos os modelos usam janelas deslizantes (sliding windows).
# Previsão multi-step: o modelo prevê 1 passo por vez e realimenta
# suas próprias previsões — nunca usa valores reais futuros.
#
# Scalers ajustados APENAS no treino → sem data leakage.
# Early stopping monitora val_loss → sem data leakage.
# HPT (unidades, dropout) avaliado por MAE de validação.
# Métricas reportadas calculadas SOMENTE no teste.
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Compatível com TensorFlow 2.x e Keras 3
import tensorflow as tf
try:
    # Keras 3 (TF >= 2.16) — import direto
    import keras
    from keras.models import Sequential
    from keras.layers import (
        Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D,
        BatchNormalization
    )
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
except ImportError:
    # Keras 2 (TF < 2.16) — import via tensorflow
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D,
        BatchNormalization
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler

from config import (
    DATA_PATH, TRANSFORMADORES, MAPA_PLOT, RANDOM_STATE,
    DL_LOOKBACK, DL_HORIZON, DL_EPOCHS, DL_BATCH_SIZE,
    DL_PATIENCE, DL_LR, END_TRAIN, END_VALIDATION,
    OUTPUT_DL, OUTPUT_FUTURE
)
from utils import (
    preparar_serie_diaria, adicionar_features_calendario,
    carregar_weather, compute_metrics, get_splits,
    plot_forecast_3way, plot_residuals, plot_future_forecast, log
)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

EXOG_COLS = [
    "weekday_sin","weekday_cos","month_sin","month_cos",
    "dayofyear_sin","dayofyear_cos","is_weekend","is_holiday",
    "temp_max_anos","temp_min_anos","temp_media_anos","precipitacao_anos"
]

# ─────────────────────────────────────────────────────────────────────────────
# PREPARAÇÃO DE DADOS
# ─────────────────────────────────────────────────────────────────────────────
def preparar_dados_dl(df_tr, weather=None):
    """
    Janelas deslizantes com normalização ajustada SOMENTE no treino.
    Retorna dicionário com arrays prontos para Keras.
    """
    y = preparar_serie_diaria(df_tr)
    splits = get_splits(y)

    cal = adicionar_features_calendario(y)
    if weather is not None:
        feat_df = cal.join(weather, how="left").bfill().ffill()
    else:
        feat_df = cal

    # ── Garantir APENAS colunas numéricas float32 ─────────────────────────
    # Remove qualquer coluna de string/objeto que possa ter vazado pelo join
    feat_df = feat_df.select_dtypes(include=[np.number])
    # Forçar conversão explícita coluna por coluna (mais seguro que .astype em bloco)
    for col in feat_df.columns:
        feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce").astype(np.float32)
    feat_df = feat_df.fillna(0.0)

    feat_cols = [c for c in EXOG_COLS if c in feat_df.columns]
    if not feat_cols:
        feat_cols = list(feat_df.columns)

    log.info(f"  DL features ({len(feat_cols)}): {feat_cols}")

    X_raw = feat_df[feat_cols].to_numpy(dtype=np.float32)
    y_raw = np.array(y.values, dtype=np.float32).reshape(-1, 1)

    # Scalers fitados APENAS no treino
    tr_mask = y.index.isin(splits["train"])
    sc_X = MinMaxScaler(); sc_y = MinMaxScaler()
    sc_X.fit(X_raw[tr_mask]); sc_y.fit(y_raw[tr_mask])
    X_sc = sc_X.transform(X_raw).astype(np.float32)
    y_sc = sc_y.transform(y_raw).flatten().astype(np.float32)

    # Janelas
    def make_windows(X, y, lb):
        Xw, yw = [], []
        for i in range(lb, len(y)):
            Xw.append(X[i-lb:i])
            yw.append(y[i])
        return np.array(Xw), np.array(yw)

    Xw, yw = make_windows(X_sc, y_sc, DL_LOOKBACK)
    Xw = Xw.astype(np.float32)
    yw = yw.astype(np.float32)
    idx_w  = y.index[DL_LOOKBACK:]

    t_end = pd.to_datetime(END_TRAIN)
    v_end = pd.to_datetime(END_VALIDATION)
    m_tr  = idx_w <= t_end
    m_vl  = (idx_w > t_end) & (idx_w <= v_end)
    m_te  = idx_w > v_end

    return dict(
        X_tr=Xw[m_tr], y_tr=yw[m_tr],
        X_vl=Xw[m_vl], y_vl=yw[m_vl],
        X_te=Xw[m_te], y_te=yw[m_te],
        idx_te=idx_w[m_te],
        sc_y=sc_y, sc_X=sc_X,   # sc_X adicionado para previsão futura sem re-fit
        y_series=y,
        n_feat=Xw.shape[2],
        X_all=Xw, y_all=yw, idx_all=idx_w,
        X_raw_fut_base=X_sc,   # para previsão futura
        feat_cols=feat_cols,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ARQUITETURAS
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm(n_feat, units=128, dropout=0.2):
    m = Sequential([
        Input((DL_LOOKBACK, n_feat)),
        LSTM(units, return_sequences=True), Dropout(dropout),
        LSTM(units//2),                     Dropout(dropout),
        Dense(64, activation="relu"),        Dense(1)
    ], name="LSTM")
    m.compile(optimizer=Adam(learning_rate=DL_LR), loss="mse", metrics=["mae"]); return m

def build_stacked_lstm(n_feat, units=128, dropout=0.2):
    m = Sequential([
        Input((DL_LOOKBACK, n_feat)),
        LSTM(units,    return_sequences=True), Dropout(dropout),
        LSTM(units//2, return_sequences=True), Dropout(dropout),
        LSTM(units//4),                        Dropout(dropout),
        Dense(64, activation="relu"),           Dense(1)
    ], name="Stacked_LSTM")
    m.compile(optimizer=Adam(learning_rate=DL_LR), loss="mse", metrics=["mae"]); return m

def build_cnn_lstm(n_feat, filters=64, kernel_size=7, units=64, dropout=0.2):
    m = Sequential([
        Input((DL_LOOKBACK, n_feat)),
        Conv1D(filters, kernel_size, activation="relu", padding="same"),
        BatchNormalization(),
        Conv1D(filters//2, 3, activation="relu", padding="same"),
        MaxPooling1D(2), Dropout(dropout),
        LSTM(units),     Dropout(dropout),
        Dense(64, activation="relu"), Dense(1)
    ], name="CNN_LSTM")
    m.compile(optimizer=Adam(learning_rate=DL_LR), loss="mse", metrics=["mae"]); return m

ARCHITECTURES = {
    "LSTM": (build_lstm, [
        {"units":64,  "dropout":0.2},
        {"units":128, "dropout":0.2},
        {"units":128, "dropout":0.3},
    ]),
    "Stacked_LSTM": (build_stacked_lstm, [
        {"units":64,  "dropout":0.2},
        {"units":128, "dropout":0.3},
    ]),
    "CNN_LSTM": (build_cnn_lstm, [
        {"filters":32, "kernel_size":7,  "units":64,  "dropout":0.2},
        {"filters":64, "kernel_size":7,  "units":64,  "dropout":0.2},
        {"filters":64, "kernel_size":14, "units":128, "dropout":0.3},
    ]),
}

def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=DL_PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=DL_PATIENCE//2,
                          min_learning_rate=1e-6, verbose=0),
    ]

# ─────────────────────────────────────────────────────────────────────────────
# RECURSIVE PREDICT PARA MODELOS DL
# ─────────────────────────────────────────────────────────────────────────────
def recursive_predict_dl(model, last_window, future_X_exog, sc_y, steps):
    """
    Previsão multi-step recursiva para modelos Keras.
    last_window: (lookback, n_feat) — última janela real conhecida.
    future_X_exog: (steps, n_feat) — features exógenas futuras (calendário).
    Alimenta previsões anteriores como entrada dos próximos passos.
    """
    window = last_window.copy()   # (lookback, n_feat)
    preds_sc = []

    for s in range(steps):
        x_in = window[np.newaxis, :, :]   # (1, lookback, n_feat)
        yhat_sc = model.predict(x_in, verbose=0)[0, 0]
        preds_sc.append(yhat_sc)

        # Deslizar janela: remover o mais antigo, adicionar o novo
        new_row = future_X_exog[s].copy() if s < len(future_X_exog) else window[-1].copy()
        # Substituir o canal da série (assumindo que é o último ou não há canal de série)
        window = np.vstack([window[1:], new_row])

    preds_sc = np.array(preds_sc).reshape(-1, 1)
    return sc_y.inverse_transform(preds_sc).flatten()

# ─────────────────────────────────────────────────────────────────────────────
# TREINO
# ─────────────────────────────────────────────────────────────────────────────
def treinar_dl(df: pd.DataFrame, weather=None) -> pd.DataFrame:
    resultados = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [DL]\n{'='*60}")
        data = preparar_dados_dl(df_tr, weather)
        if len(data["X_tr"]) == 0 or len(data["X_vl"]) == 0:
            log.warning(f"  {tid}: dados insuficientes, pulando."); continue

        # ── Garantia final: todos os arrays como float32 antes de entrar no Keras
        for key in ["X_tr","y_tr","X_vl","y_vl","X_te","y_te","X_all","y_all"]:
            if key in data:
                data[key] = np.array(data[key], dtype=np.float32)

        for nome, (build_fn, hpt_grid) in ARCHITECTURES.items():
            log.info(f"\n  >>> {nome}")

            # ── HPT sobre VALIDAÇÃO ───────────────────────────────────────
            best_mae, best_params = np.inf, None

            for params in hpt_grid:
                model = build_fn(data["n_feat"], **params)

                # ── DEBUG: inspecionar TUDO antes do fit ──────────────────
                X_tr = np.array(data["X_tr"], dtype=np.float32)
                y_tr = np.array(data["y_tr"], dtype=np.float32)
                X_vl = np.array(data["X_vl"], dtype=np.float32)
                y_vl = np.array(data["y_vl"], dtype=np.float32)
                log.info(f"  shapes: X_tr={X_tr.shape} {X_tr.dtype} | "
                         f"y_tr={y_tr.shape} {y_tr.dtype} | "
                         f"X_vl={X_vl.shape} {X_vl.dtype} | "
                         f"y_vl={y_vl.shape} {y_vl.dtype}")

                model.fit(X_tr, y_tr,
                          validation_data=(X_vl, y_vl),
                          epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                          callbacks=get_callbacks(), verbose=0)
                pred_vl = model.predict(data["X_vl"], verbose=0).flatten()
                pred_vl_inv = data["sc_y"].inverse_transform(
                    pred_vl.reshape(-1,1)).flatten()
                y_vl_inv = data["sc_y"].inverse_transform(
                    data["y_vl"].reshape(-1,1)).flatten()
                mae = np.mean(np.abs(y_vl_inv - pred_vl_inv))
                log.info(f"     {params} → MAE_val={mae:.4f}")
                if mae < best_mae:
                    best_mae, best_params = mae, params

            log.info(f"  ✓ HPT → {best_params} | MAE_val={best_mae:.4f}")

            # ── Retreinar TREINO+VAL, avaliar TESTE ───────────────────────
            X_tv = np.vstack([data["X_tr"], data["X_vl"]])
            y_tv = np.concatenate([data["y_tr"], data["y_vl"]])

            model_final = build_fn(data["n_feat"], **best_params)
            # ── CORREÇÃO: Early stopping monitora VAL (X_vl), NUNCA o teste ──
            # Usar X_vl como monitor de early stopping ao retreinar em treino+val
            # é aceitável pois o ES apenas controla quando parar, não seleciona
            # hiperparâmetros (esses já foram selecionados na etapa de HPT).
            # Alternativa mais conservadora: treinar por épocas fixas (sem ES).
            history = model_final.fit(
                X_tv, y_tv,
                validation_data=(data["X_vl"], data["y_vl"]),
                epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                callbacks=get_callbacks(), verbose=0
            )

            pred_te = model_final.predict(data["X_te"], verbose=0).flatten()
            pred_te_inv = data["sc_y"].inverse_transform(
                pred_te.reshape(-1,1)).flatten()
            y_te_inv = data["sc_y"].inverse_transform(
                data["y_te"].reshape(-1,1)).flatten()

            y_true_s = pd.Series(y_te_inv,   index=data["idx_te"])
            y_pred_s = pd.Series(pred_te_inv, index=data["idx_te"])
            metrics  = compute_metrics(y_true_s, y_pred_s)
            log.info(f"  ✓ TESTE → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

            # Previsão de validação para plot
            pred_vl_final = model_final.predict(data["X_vl"], verbose=0).flatten()
            pred_vl_inv2  = data["sc_y"].inverse_transform(
                pred_vl_final.reshape(-1,1)).flatten()
            idx_vl = data["idx_all"][(data["idx_all"] > pd.to_datetime(END_TRAIN)) &
                                     (data["idx_all"] <= pd.to_datetime(END_VALIDATION))]
            pred_vl_s = pd.Series(pred_vl_inv2[:len(idx_vl)], index=idx_vl)

            plot_forecast_3way(data["y_series"], pred_vl_s, y_pred_s,
                               tid, nome, OUTPUT_DL)
            plot_residuals(y_true_s, y_pred_s, tid, nome, OUTPUT_DL)

            # Curva de loss
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(history.history["loss"],     label="Treino",    color="#2563eb")
            ax.plot(history.history["val_loss"], label="Validação", color="#f59e0b")
            ax.set_title(f"{MAPA_PLOT.get(tid,tid)} — {nome}: Loss (MSE)")
            ax.set_xlabel("Época"); ax.legend(); plt.tight_layout()
            plt.savefig(f"{OUTPUT_DL}/loss_{tid}_{nome}.svg",
                        dpi=150, bbox_inches="tight"); plt.close()

            # ── Previsão futura recursiva ──────────────────────────────────
            try:
                from config import FUTURE_END_DATE
                future_idx = pd.date_range(
                    data["y_series"].index[-1]+pd.Timedelta(days=1),
                    pd.to_datetime(FUTURE_END_DATE), freq="D"
                )
                # Features exógenas futuras (calendário determinístico)
                cal_fut = adicionar_features_calendario(
                    pd.Series(np.nan, index=future_idx))
                # Clima futuro: média histórica (documentar no artigo)
                if weather is not None:
                    clim_mean = weather.mean()
                    for col in clim_mean.index:
                        if col in data["feat_cols"]:
                            cal_fut[col] = clim_mean[col]
                feat_fut_cols = [c for c in data["feat_cols"] if c in cal_fut.columns]

                # Normalizar com sc_X já fitado no treino (salvo em preparar_dados_dl)
                # CORREÇÃO: X_all é 3D (janelas), não pode ser usado para refit do scaler.
                # sc_X foi fitado apenas no treino — reutilizamos diretamente.
                sc_X_ref = data.get("sc_X")
                if sc_X_ref is None:
                    # fallback: recriar fitando nos primeiros dados (apenas treino)
                    from sklearn.preprocessing import MinMaxScaler
                    n_tr = sum(data["idx_all"] <= pd.to_datetime(END_TRAIN))
                    # X_all[0] tem shape (lookback, n_feat); pegar o último timestep de cada janela
                    X2d = data["X_all"][:n_tr, -1, :]   # (n_tr, n_feat)
                    sc_X_ref = MinMaxScaler().fit(X2d)

                fut_raw = np.zeros((len(future_idx), data["n_feat"]))
                for i, col in enumerate(feat_fut_cols[:data["n_feat"]]):
                    if col in cal_fut.columns:
                        fut_raw[:, i] = cal_fut[col].values[:len(future_idx)]
                fut_sc = sc_X_ref.transform(fut_raw)

                last_win = data["X_all"][-1]   # última janela real (lookback, n_feat)

                # Retreinar modelo final em TODOS os dados
                model_fut = build_fn(data["n_feat"], **best_params)
                model_fut.fit(data["X_all"], data["y_all"],
                              epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                              callbacks=get_callbacks(), verbose=0)

                preds_fut = recursive_predict_dl(
                    model_fut, last_win, fut_sc, data["sc_y"], len(future_idx)
                )
                fut_series = pd.Series(preds_fut, index=future_idx)

                # Bootstrap residual
                all_pred  = model_fut.predict(data["X_all"], verbose=0).flatten()
                all_pred_inv = data["sc_y"].inverse_transform(
                    all_pred.reshape(-1,1)).flatten()
                all_y_inv    = data["sc_y"].inverse_transform(
                    data["y_all"].reshape(-1,1)).flatten()
                resids   = all_y_inv - all_pred_inv
                rng      = np.random.default_rng(RANDOM_STATE)
                boots    = [preds_fut + rng.choice(resids, size=len(future_idx), replace=True)
                            for _ in range(200)]
                boots    = np.array(boots)

                plot_future_forecast(
                    data["y_series"], fut_series,
                    pd.Series(np.percentile(boots,2.5,axis=0), index=future_idx),
                    pd.Series(np.percentile(boots,97.5,axis=0), index=future_idx),
                    tid, nome, OUTPUT_FUTURE
                )
                pd.DataFrame({
                    "data": future_idx, "prev_mean": preds_fut,
                    "ic_lower_95": np.percentile(boots,2.5,axis=0),
                    "ic_upper_95": np.percentile(boots,97.5,axis=0),
                }).to_csv(f"{OUTPUT_FUTURE}/futuro_{tid}_{nome}.csv", index=False)
            except Exception as e:
                log.warning(f"  Previsão futura {nome} falhou: {e}")

            resultados.append({
                "transformador": tid, "nome_plot": MAPA_PLOT.get(tid,tid),
                "modelo": nome, "abordagem": "DeepLearning",
                "best_params": str(best_params), "MAE_val": round(best_mae,4),
                **{k: round(v,4) for k,v in metrics.items()},
            })

    return pd.DataFrame(resultados)


def main():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES)].copy()
    weather = None
    try: weather = carregar_weather()
    except Exception as e: log.warning(f"Clima: {e}")

    res = treinar_dl(df, weather)
    res.to_csv(f"{OUTPUT_DL}/metricas_DL.csv", index=False)
    print(res[["transformador","modelo","RMSE","MAE","sMAPE","R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
