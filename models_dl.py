# =============================================================================
# models_dl.py — Deep Learning: LSTM · CNN-LSTM  ·  UV e MV
# =============================================================================
# Estrutura espelhada ao models_boosting.py para comparação simétrica.
#
# Modo UV: janela de entrada contém APENAS a série Smax normalizada.
#          n_feat = 1 (só y escalado).
#
# Modo MV: janela de entrada contém Smax + features exógenas selecionadas.
#          Seleção idêntica ao boosting: proxy LGBM, gain ≥ 95%, só treino.
#
# Previsão multi-step: recursiva — alimenta previsões como entrada do próximo
# passo, nunca usa valores reais futuros.
#
# Scalers fitados APENAS no treino → sem data leakage.
# Early stopping monitora val_loss (X_vl) → nunca o teste.
# HPT avalia MAE na validação (invertido para escala original).
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
try:
    import keras
    from keras.models import Sequential
    from keras.layers import (Input, LSTM, Dense, Dropout,
                              Conv1D, MaxPooling1D, BatchNormalization)
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
except ImportError:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,
                                         Conv1D, MaxPooling1D, BatchNormalization)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor

from config import (
    DATA_PATH, TRANSFORMADORES, MAPA_PLOT, RANDOM_STATE,
    DL_LOOKBACK, DL_EPOCHS, DL_BATCH_SIZE, DL_PATIENCE, DL_LR,
    END_TRAIN, END_VALIDATION, OUTPUT_DL, OUTPUT_FUTURE, FUTURE_END_DATE
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
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "dayofyear_sin", "dayofyear_cos", "is_weekend", "is_holiday",
    "temp_max_anos", "temp_min_anos", "temp_media_anos", "precipitacao_anos"
]
FEATURE_IMPORTANCE_THRESHOLD = 0.95


# ─────────────────────────────────────────────────────────────────────────────
# SELEÇÃO DE FEATURES (idêntica ao boosting)
# ─────────────────────────────────────────────────────────────────────────────
def selecionar_features_mv(y_train, exog_train, tid, output_dir):
    proxy = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1,
        verbose=-1, importance_type="gain"
    )
    from skforecast.recursive import ForecasterRecursive
    from skforecast.preprocessing import RollingFeatures
    from config import LAGS, ROLLING_WINDOW
    fc = ForecasterRecursive(
        estimator=proxy, lags=LAGS,
        window_features=RollingFeatures(stats=["mean"], window_sizes=[ROLLING_WINDOW])
    )
    fc.fit(y=y_train, exog=exog_train)
    imp_df = fc.get_feature_importances()

    exog_imp = imp_df[imp_df["feature"].isin(exog_train.columns)].copy()
    exog_imp = exog_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    if exog_imp.empty or exog_imp["importance"].sum() == 0:
        return list(exog_train.columns)

    exog_imp["imp_cum"] = (exog_imp["importance"] / exog_imp["importance"].sum()).cumsum()
    selected = exog_imp.loc[
        exog_imp["imp_cum"].shift(1, fill_value=0) < FEATURE_IMPORTANCE_THRESHOLD,
        "feature"
    ].tolist()
    if not selected:
        selected = [exog_imp.iloc[0]["feature"]]

    log.info(f"  [FS-DL] {tid}: {len(selected)}/{len(exog_imp)} → {selected}")

    colors = ["#2563eb" if f in selected else "#94a3b8" for f in exog_imp["feature"]]
    fig, ax = plt.subplots(figsize=(8, max(4, len(exog_imp) * 0.45 + 1)))
    ax.barh(exog_imp["feature"][::-1], exog_imp["importance"][::-1], color=colors[::-1])
    ax.set_title(f"{MAPA_PLOT.get(tid,tid)} — DL: Feature Selection\n"
                 f"Blue = selected ({len(selected)}/{len(exog_imp)})", fontsize=9)
    ax.set_xlabel("Importance (gain)")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/feature_selection_{tid}_DL_MV.svg",
                dpi=150, bbox_inches="tight")
    plt.close()
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# PREPARAÇÃO DE DADOS
# ─────────────────────────────────────────────────────────────────────────────
def preparar_dados_dl(df_tr, weather=None, mode="UV"):
    """
    UV: X = janela de (lookback, 1) — só Smax normalizado.
    MV: X = janela de (lookback, 1 + n_exog_sel) — Smax + features selecionadas.
    Scaler fitado APENAS no treino.
    """
    y = preparar_serie_diaria(df_tr)
    splits = get_splits(y)

    sel_features = None

    if mode == "MV":
        cal = adicionar_features_calendario(y)
        if weather is not None:
            exog_df = cal.join(weather, how="left").bfill().ffill()
        else:
            exog_df = cal
        cols_disp = [c for c in EXOG_COLS if c in exog_df.columns]
        exog_df = exog_df[cols_disp]

        sel_features = selecionar_features_mv(
            y.loc[splits["train"]],
            exog_df.loc[splits["train"]],
            df_tr["id"].iloc[0] if "id" in df_tr.columns else "?",
            OUTPUT_DL
        )
        exog_sel = exog_df[[c for c in sel_features if c in exog_df.columns]]

        # Scaler das exógenas — fitado só no treino
        sc_exog = MinMaxScaler()
        sc_exog.fit(exog_sel.loc[splits["train"]].values)
        exog_sc = sc_exog.transform(exog_sel.values)
    else:
        exog_sc    = None
        sc_exog    = None
        sel_features = []

    # Scaler do target — fitado só no treino
    y_raw = y.values.reshape(-1, 1).astype(np.float32)
    sc_y  = MinMaxScaler()
    sc_y.fit(y_raw[y.index.isin(splits["train"])])
    y_sc  = sc_y.transform(y_raw).flatten().astype(np.float32)

    # Montar array de features: [y_sc] ou [y_sc, exog_sc...]
    if exog_sc is not None:
        X_raw = np.hstack([y_sc.reshape(-1, 1), exog_sc]).astype(np.float32)
    else:
        X_raw = y_sc.reshape(-1, 1).astype(np.float32)

    # Janelas deslizantes
    def make_windows(X, y, lb):
        Xw, yw = [], []
        for i in range(lb, len(y)):
            Xw.append(X[i - lb:i])
            yw.append(y[i])
        return np.array(Xw, dtype=np.float32), np.array(yw, dtype=np.float32)

    Xw, yw = make_windows(X_raw, y_sc, DL_LOOKBACK)
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
        idx_vl=idx_w[m_vl], idx_te=idx_w[m_te],
        sc_y=sc_y, sc_exog=sc_exog,
        y_series=y, n_feat=Xw.shape[2],
        X_all=Xw, y_all=yw, idx_all=idx_w,
        sel_features=sel_features,
        X_raw_full=X_raw,   # para previsão futura
        y_sc_full=y_sc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ARQUITETURAS (apenas LSTM e CNN-LSTM)
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm(n_feat, units=128, dropout=0.2):
    m = Sequential([
        Input((DL_LOOKBACK, n_feat)),
        LSTM(units, return_sequences=True), Dropout(dropout),
        LSTM(units // 2),                   Dropout(dropout),
        Dense(64, activation="relu"),        Dense(1)
    ], name="LSTM")
    m.compile(Adam(DL_LR), "mse", metrics=["mae"])
    return m


def build_cnn_lstm(n_feat, filters=64, kernel_size=7, units=64, dropout=0.2):
    m = Sequential([
        Input((DL_LOOKBACK, n_feat)),
        Conv1D(filters, kernel_size, activation="relu", padding="same"),
        BatchNormalization(),
        Conv1D(filters // 2, 3, activation="relu", padding="same"),
        MaxPooling1D(2), Dropout(dropout),
        LSTM(units),     Dropout(dropout),
        Dense(64, activation="relu"), Dense(1)
    ], name="CNN_LSTM")
    m.compile(Adam(DL_LR), "mse", metrics=["mae"])
    return m


ARCHITECTURES = {
    "LSTM": (build_lstm, [
        {"units": 64,  "dropout": 0.2},
        {"units": 128, "dropout": 0.2},
        {"units": 128, "dropout": 0.3},
    ]),
    "CNN_LSTM": (build_cnn_lstm, [
        {"filters": 32, "kernel_size": 7,  "units": 64,  "dropout": 0.2},
        {"filters": 64, "kernel_size": 7,  "units": 64,  "dropout": 0.2},
        {"filters": 64, "kernel_size": 14, "units": 128, "dropout": 0.3},
    ]),
}


def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=DL_PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=DL_PATIENCE // 2, min_lr=1e-6, verbose=0),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# RECURSIVE PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def recursive_predict_dl(model, last_window, fut_exog_sc, sc_y, steps):
    """
    UV: fut_exog_sc = None → desliza janela só com previsão anterior.
    MV: fut_exog_sc = (steps, n_exog) → substitui canal exógeno em cada passo.
    """
    window   = last_window.copy()   # (lookback, n_feat)
    preds_sc = []
    n_feat   = window.shape[1]

    for s in range(steps):
        x_in = window[np.newaxis]
        yhat = model.predict(x_in, verbose=0)[0, 0]
        preds_sc.append(yhat)

        # Nova linha: [yhat_sc, exog_s...] ou só [yhat_sc]
        new_row = np.zeros(n_feat, dtype=np.float32)
        new_row[0] = yhat
        if fut_exog_sc is not None and s < len(fut_exog_sc):
            new_row[1:] = fut_exog_sc[s]
        window = np.vstack([window[1:], new_row])

    preds_sc = np.array(preds_sc).reshape(-1, 1)
    return sc_y.inverse_transform(preds_sc).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def treinar_dl(df: pd.DataFrame, weather=None, mode="UV") -> pd.DataFrame:
    resultados = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [DL-{mode}]\n{'='*60}")
        data = preparar_dados_dl(df_tr, weather=weather, mode=mode)

        if len(data["X_tr"]) == 0 or len(data["X_vl"]) == 0:
            log.warning(f"  {tid}: dados insuficientes."); continue

        for key in ["X_tr", "y_tr", "X_vl", "y_vl", "X_te", "y_te", "X_all", "y_all"]:
            data[key] = np.array(data[key], dtype=np.float32)

        for nome, (build_fn, hpt_grid) in ARCHITECTURES.items():
            log.info(f"\n  >>> {nome} [{mode}]")

            # ── HPT sobre VALIDAÇÃO ───────────────────────────────────────
            best_mae, best_params = np.inf, None
            for params in hpt_grid:
                model = build_fn(data["n_feat"], **params)
                model.fit(data["X_tr"], data["y_tr"],
                          validation_data=(data["X_vl"], data["y_vl"]),
                          epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                          callbacks=get_callbacks(), verbose=0)
                pv_sc  = model.predict(data["X_vl"], verbose=0).flatten()
                pv_inv = data["sc_y"].inverse_transform(pv_sc.reshape(-1, 1)).flatten()
                yv_inv = data["sc_y"].inverse_transform(data["y_vl"].reshape(-1, 1)).flatten()
                mae    = np.mean(np.abs(yv_inv - pv_inv))
                log.info(f"     {params} → MAE_val={mae:.4f}")
                if mae < best_mae:
                    best_mae, best_params = mae, params

            log.info(f"  ✓ HPT → {best_params} | MAE_val={best_mae:.4f}")

            # ── Retreinar TREINO+VAL, early stopping monitora VAL ─────────
            X_tv = np.vstack([data["X_tr"], data["X_vl"]])
            y_tv = np.concatenate([data["y_tr"], data["y_vl"]])
            model_final = build_fn(data["n_feat"], **best_params)
            history = model_final.fit(
                X_tv, y_tv,
                validation_data=(data["X_vl"], data["y_vl"]),
                epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                callbacks=get_callbacks(), verbose=0
            )

            # ── Avaliação no TESTE ────────────────────────────────────────
            pt_sc  = model_final.predict(data["X_te"], verbose=0).flatten()
            pt_inv = data["sc_y"].inverse_transform(pt_sc.reshape(-1, 1)).flatten()
            yt_inv = data["sc_y"].inverse_transform(data["y_te"].reshape(-1, 1)).flatten()
            y_true_s = pd.Series(yt_inv, index=data["idx_te"])
            y_pred_s = pd.Series(pt_inv, index=data["idx_te"])
            metrics  = compute_metrics(y_true_s, y_pred_s)
            log.info(f"  ✓ TESTE → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

            # Previsão de validação para plot
            pv2_sc  = model_final.predict(data["X_vl"], verbose=0).flatten()
            pv2_inv = data["sc_y"].inverse_transform(pv2_sc.reshape(-1, 1)).flatten()
            pred_vl_s = pd.Series(pv2_inv, index=data["idx_vl"])

            nome_modelo = f"{nome}_{mode}"
            plot_forecast_3way(data["y_series"], pred_vl_s, y_pred_s,
                               tid, nome_modelo, OUTPUT_DL, suffix="")
            plot_residuals(y_true_s, y_pred_s, tid, nome_modelo, OUTPUT_DL)

            # Curva de loss
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history.history["loss"],     label="Training",  color="#2563eb")
            ax.plot(history.history["val_loss"], label="Validation", color="#f59e0b")
            ax.set_title(f"{MAPA_PLOT.get(tid,tid)} — {nome_modelo}: Training Loss (MSE)")
            ax.set_xlabel("Epoch"); ax.legend(); plt.tight_layout()
            plt.savefig(f"{OUTPUT_DL}/loss_{tid}_{nome_modelo}.svg",
                        dpi=150, bbox_inches="tight"); plt.close()

            # ── Previsão futura ───────────────────────────────────────────
            try:
                future_idx = pd.date_range(
                    data["y_series"].index[-1] + pd.Timedelta(days=1),
                    pd.to_datetime(FUTURE_END_DATE), freq="D"
                )
                # Features exógenas futuras (determinísticas)
                fut_exog_sc = None
                if mode == "MV" and data["sc_exog"] is not None:
                    cal_fut = adicionar_features_calendario(
                        pd.Series(np.nan, index=future_idx))
                    if weather is not None:
                        for col in [c for c in data["sel_features"]
                                    if c not in cal_fut.columns and c in weather.columns]:
                            cal_fut[col] = weather[col].mean()
                    sel_fut = [c for c in data["sel_features"] if c in cal_fut.columns]
                    fut_raw = cal_fut[sel_fut].values.astype(np.float32)
                    # Pad/trim para n_exog correto
                    n_exog = data["n_feat"] - 1
                    if fut_raw.shape[1] < n_exog:
                        pad = np.zeros((len(future_idx), n_exog - fut_raw.shape[1]),
                                       dtype=np.float32)
                        fut_raw = np.hstack([fut_raw, pad])
                    fut_exog_sc = data["sc_exog"].transform(fut_raw[:, :n_exog])

                # Retreinar em todos os dados
                model_fut = build_fn(data["n_feat"], **best_params)
                model_fut.fit(data["X_all"], data["y_all"],
                              epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                              callbacks=get_callbacks(), verbose=0)

                preds_fut = recursive_predict_dl(
                    model_fut, data["X_all"][-1], fut_exog_sc,
                    data["sc_y"], len(future_idx)
                )
                fut_series = pd.Series(preds_fut, index=future_idx)

                all_pred_inv = data["sc_y"].inverse_transform(
                    model_fut.predict(data["X_all"], verbose=0).reshape(-1, 1)).flatten()
                all_y_inv = data["sc_y"].inverse_transform(
                    data["y_all"].reshape(-1, 1)).flatten()
                resids = all_y_inv - all_pred_inv
                rng    = np.random.default_rng(RANDOM_STATE)
                boots  = np.array([preds_fut + rng.choice(resids, size=len(future_idx),
                                                           replace=True)
                                   for _ in range(200)])
                plot_future_forecast(
                    data["y_series"], fut_series,
                    pd.Series(np.percentile(boots, 2.5,  axis=0), index=future_idx),
                    pd.Series(np.percentile(boots, 97.5, axis=0), index=future_idx),
                    tid, nome_modelo, OUTPUT_FUTURE
                )
            except Exception as e:
                log.warning(f"  Previsão futura {nome_modelo} falhou: {e}")

            resultados.append({
                "transformador": tid, "nome_plot": MAPA_PLOT.get(tid, tid),
                "modelo": nome, "abordagem": mode,
                "features_sel": str(data["sel_features"]),
                "n_features_sel": len(data["sel_features"]),
                "best_params": str(best_params), "MAE_val": round(best_mae, 4),
                **{k: round(v, 4) for k, v in metrics.items()},
            })

    return pd.DataFrame(resultados)


def main():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES)].copy()

    weather = None
    try:
        weather = carregar_weather()
    except Exception as e:
        log.warning(f"Clima: {e}")

    log.info("=== DL UNIVARIADO ===")
    res_uv = treinar_dl(df, weather=None, mode="UV")
    res_uv.to_csv(f"{OUTPUT_DL}/metricas_DL_UV.csv", index=False)

    log.info("=== DL MULTIVARIADO ===")
    res_mv = treinar_dl(df, weather=weather, mode="MV")
    res_mv.to_csv(f"{OUTPUT_DL}/metricas_DL_MV.csv", index=False)

    res = pd.concat([res_uv, res_mv])
    res.to_csv(f"{OUTPUT_DL}/metricas_DL.csv", index=False)
    print(res[["transformador", "modelo", "abordagem", "RMSE", "MAE", "sMAPE", "R2"]]
          .sort_values(["transformador", "modelo", "abordagem"]).to_string(index=False))


if __name__ == "__main__":
    main()
