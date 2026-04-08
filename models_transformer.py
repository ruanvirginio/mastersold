# =============================================================================
# models_transformer.py — Temporal Fusion Transformer (TFT) para Séries Temporais
# =============================================================================
# Implementação do Transformer adaptado para previsão de séries temporais.
# Inclui dois modos:
#   --mode lite   → d_model=64, 4 heads, 2 layers   (Ryzen 5700G, ~20-40 min)
#   --mode full   → d_model=256, 8 heads, 4 layers  (melhor resultado, ~2-4h CPU)
#
# Arquitetura:
#   - Positional Encoding → Multi-Head Self-Attention → Feed-Forward Network
#   - Projetado para capturar dependências de longo alcance (até 365 dias de lag)
#
# Referências:
#   Vaswani et al. (2017). Attention is all you need. NeurIPS.
#   Lim et al. (2021). Temporal Fusion Transformers for interpretable
#     multi-horizon time series forecasting. IJF 37(4).
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, random, argparse
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
    from keras.layers import (
        Input, Dense, Dropout, LayerNormalization,
        MultiHeadAttention, GlobalAveragePooling1D, Add
    )
    from keras.models import Model
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
except ImportError:
    # Keras 2 (TF < 2.16) — import via tensorflow
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, LayerNormalization,
        MultiHeadAttention, GlobalAveragePooling1D, Add
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler

from config import (
    DATA_PATH, TRANSFORMADORES, MAPA_PLOT, RANDOM_STATE,
    DL_LOOKBACK, DL_EPOCHS, DL_BATCH_SIZE, DL_PATIENCE, DL_LR,
    END_TRAIN, END_VALIDATION,
    TF_LITE_D_MODEL, TF_LITE_N_HEADS, TF_LITE_N_LAYERS, TF_LITE_FFN_DIM, TF_LITE_DROPOUT,
    TF_FULL_D_MODEL, TF_FULL_N_HEADS, TF_FULL_N_LAYERS, TF_FULL_FFN_DIM, TF_FULL_DROPOUT,
    OUTPUT_TRANSFORMER, OUTPUT_FUTURE
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
# POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(tf.keras.layers.Layer):
    """
    Codificação posicional sinusoidal clássica (Vaswani et al., 2017).
    Injeta informação de posição temporal na sequência.
    """
    def __init__(self, max_len=500, d_model=64, **kwargs):
        super().__init__(**kwargs)
        pos = np.arange(max_len)[:, np.newaxis]
        i   = np.arange(d_model)[np.newaxis, :]
        angle = pos / np.power(10000, (2*(i//2)) / np.float32(d_model))
        angle[:, 0::2] = np.sin(angle[:, 0::2])
        angle[:, 1::2] = np.cos(angle[:, 1::2])
        self.pos_enc = tf.constant(angle[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_enc[:, :seq_len, :]

# ─────────────────────────────────────────────────────────────────────────────
# BLOCO TRANSFORMER ENCODER
# ─────────────────────────────────────────────────────────────────────────────
def transformer_encoder_block(x, d_model, n_heads, ffn_dim, dropout_rate):
    """
    Um bloco encoder: Multi-Head Attention + FFN + LayerNorm + Residual.
    """
    # Multi-Head Self-Attention
    attn_out = MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model//n_heads, dropout=dropout_rate
    )(x, x)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn_out]))

    # Feed-Forward Network
    ffn = Dense(ffn_dim, activation="relu")(x)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(d_model)(ffn)
    x   = LayerNormalization(epsilon=1e-6)(Add()([x, ffn]))
    return x

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUÇÃO DO MODELO
# ─────────────────────────────────────────────────────────────────────────────
def build_transformer(lookback, n_features, d_model, n_heads,
                      n_layers, ffn_dim, dropout_rate):
    """
    Transformer completo para previsão de série temporal (1-step output).

    Entrada : (batch, lookback, n_features)
    Saída   : (batch, 1)
    """
    inp = Input(shape=(lookback, n_features))

    # Projeção linear para d_model
    x = Dense(d_model)(inp)

    # Positional encoding
    x = PositionalEncoding(max_len=lookback+10, d_model=d_model)(x)
    x = Dropout(dropout_rate)(x)

    # N blocos encoder empilhados
    for _ in range(n_layers):
        x = transformer_encoder_block(x, d_model, n_heads, ffn_dim, dropout_rate)

    # Pooling global + cabeça de regressão
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(64, activation="relu")(x)
    x   = Dropout(dropout_rate)(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out, name="Transformer")
    model.compile(optimizer=Adam(DL_LR), loss="mse", metrics=["mae"])
    return model

# ─────────────────────────────────────────────────────────────────────────────
# PREPARAÇÃO DE DADOS
# ─────────────────────────────────────────────────────────────────────────────
def preparar_dados_transformer(df_tr, weather=None):
    """Idêntico ao DL, reutilizado aqui para independência do script."""
    y = preparar_serie_diaria(df_tr)
    splits = get_splits(y)
    cal = adicionar_features_calendario(y)
    if weather is not None:
        feat_df = cal.join(weather, how="left").bfill().ffill()
    else:
        feat_df = cal

    # ── Garantir APENAS colunas numéricas float32 ─────────────────────────
    feat_df = feat_df.select_dtypes(include=[np.number])
    for col in feat_df.columns:
        feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce").astype(np.float32)
    feat_df = feat_df.fillna(0.0)

    feat_cols = [c for c in EXOG_COLS if c in feat_df.columns]
    if not feat_cols:
        feat_cols = list(feat_df.columns)

    X_raw = feat_df[feat_cols].to_numpy(dtype=np.float32)
    y_raw = np.array(y.values, dtype=np.float32).reshape(-1,1)

    tr_mask = y.index.isin(splits["train"])
    sc_X = MinMaxScaler(); sc_y = MinMaxScaler()
    sc_X.fit(X_raw[tr_mask]); sc_y.fit(y_raw[tr_mask])
    X_sc = sc_X.transform(X_raw).astype(np.float32)
    y_sc = sc_y.transform(y_raw).flatten().astype(np.float32)

    def windows(X, y, lb):
        Xw, yw = [], []
        for i in range(lb, len(y)):
            Xw.append(X[i-lb:i]); yw.append(y[i])
        return np.array(Xw, dtype=np.float32), np.array(yw, dtype=np.float32)

    Xw, yw = windows(X_sc, y_sc, DL_LOOKBACK)
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
        sc_y=sc_y, y_series=y, n_feat=Xw.shape[2],
        X_all=Xw, y_all=yw, idx_all=idx_w,
        feat_cols=feat_cols, sc_X=sc_X,
    )


def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=DL_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=DL_PATIENCE//2, min_lr=1e-6, verbose=0),
    ]

# ─────────────────────────────────────────────────────────────────────────────
# RECURSIVE PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def recursive_predict_transformer(model, last_window, fut_sc, sc_y, steps):
    window = last_window.copy()
    preds_sc = []
    for s in range(steps):
        x_in = window[np.newaxis]
        yhat = model.predict(x_in, verbose=0)[0,0]
        preds_sc.append(yhat)
        new_row = fut_sc[s].copy() if s < len(fut_sc) else window[-1].copy()
        window  = np.vstack([window[1:], new_row])
    preds_sc = np.array(preds_sc).reshape(-1,1)
    return sc_y.inverse_transform(preds_sc).flatten()

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def treinar_transformer(df: pd.DataFrame, weather=None, mode: str = "full") -> pd.DataFrame:
    """
    mode='lite'  → configuração leve para CPU (Ryzen 5700G)
    mode='full'  → melhor resultado, demora mais sem GPU
    """
    if mode == "lite":
        d_model, n_heads, n_layers, ffn_dim, dr = (
            TF_LITE_D_MODEL, TF_LITE_N_HEADS, TF_LITE_N_LAYERS,
            TF_LITE_FFN_DIM, TF_LITE_DROPOUT
        )
        hpt_grid = [
            {"d_model":64,  "n_heads":4, "n_layers":2, "ffn_dim":128, "dropout_rate":0.1},
            {"d_model":64,  "n_heads":4, "n_layers":2, "ffn_dim":256, "dropout_rate":0.2},
        ]
    else:
        d_model, n_heads, n_layers, ffn_dim, dr = (
            TF_FULL_D_MODEL, TF_FULL_N_HEADS, TF_FULL_N_LAYERS,
            TF_FULL_FFN_DIM, TF_FULL_DROPOUT
        )
        hpt_grid = [
            {"d_model":128, "n_heads":4, "n_layers":3, "ffn_dim":256, "dropout_rate":0.1},
            {"d_model":256, "n_heads":8, "n_layers":4, "ffn_dim":512, "dropout_rate":0.1},
            {"d_model":256, "n_heads":8, "n_layers":4, "ffn_dim":512, "dropout_rate":0.2},
        ]

    nome_modelo = f"Transformer_{mode}"
    resultados  = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [Transformer-{mode}]\n{'='*60}")
        data = preparar_dados_transformer(df_tr, weather)
        if len(data["X_tr"]) == 0 or len(data["X_vl"]) == 0:
            log.warning(f"  {tid}: dados insuficientes."); continue

        # ── HPT sobre VALIDAÇÃO ───────────────────────────────────────────
        best_mae, best_params = np.inf, None

        for params in hpt_grid:
            model = build_transformer(DL_LOOKBACK, data["n_feat"], **params)
            model.fit(data["X_tr"], data["y_tr"],
                      validation_data=(data["X_vl"], data["y_vl"]),
                      epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                      callbacks=get_callbacks(), verbose=0)
            pv    = model.predict(data["X_vl"], verbose=0).flatten()
            pv_inv= data["sc_y"].inverse_transform(pv.reshape(-1,1)).flatten()
            yv_inv= data["sc_y"].inverse_transform(data["y_vl"].reshape(-1,1)).flatten()
            mae   = np.mean(np.abs(yv_inv - pv_inv))
            log.info(f"     {params} → MAE_val={mae:.4f}")
            if mae < best_mae:
                best_mae, best_params = mae, params

        log.info(f"  ✓ HPT → {best_params} | MAE_val={best_mae:.4f}")

        # ── Retreinar TREINO+VAL, TESTE ───────────────────────────────────
        X_tv = np.vstack([data["X_tr"], data["X_vl"]])
        y_tv = np.concatenate([data["y_tr"], data["y_vl"]])
        model_final = build_transformer(DL_LOOKBACK, data["n_feat"], **best_params)
        # ── CORREÇÃO: Early stopping monitora VAL (X_vl), NUNCA o teste ──
        history = model_final.fit(
            X_tv, y_tv,
            validation_data=(data["X_vl"], data["y_vl"]),
            epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
            callbacks=get_callbacks(), verbose=0
        )
        pt    = model_final.predict(data["X_te"], verbose=0).flatten()
        pt_inv= data["sc_y"].inverse_transform(pt.reshape(-1,1)).flatten()
        yt_inv= data["sc_y"].inverse_transform(data["y_te"].reshape(-1,1)).flatten()
        y_true_s = pd.Series(yt_inv, index=data["idx_te"])
        y_pred_s = pd.Series(pt_inv, index=data["idx_te"])
        metrics  = compute_metrics(y_true_s, y_pred_s)
        log.info(f"  ✓ TESTE → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

        # Plot previsão validação
        pv2    = model_final.predict(data["X_vl"], verbose=0).flatten()
        pv2_inv= data["sc_y"].inverse_transform(pv2.reshape(-1,1)).flatten()
        pred_vl_s = pd.Series(pv2_inv, index=data["idx_vl"])
        plot_forecast_3way(data["y_series"], pred_vl_s, y_pred_s,
                           tid, nome_modelo, OUTPUT_TRANSFORMER)
        plot_residuals(y_true_s, y_pred_s, tid, nome_modelo, OUTPUT_TRANSFORMER)

        # Curva de loss
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(history.history["loss"],     label="Treino",    color="#2563eb")
        ax.plot(history.history["val_loss"], label="Validação", color="#f59e0b")
        ax.set_title(f"{MAPA_PLOT.get(tid,tid)} — {nome_modelo}: Loss")
        ax.set_xlabel("Época"); ax.legend(); plt.tight_layout()
        plt.savefig(f"{OUTPUT_TRANSFORMER}/loss_{tid}_{nome_modelo}.svg",
                    dpi=150, bbox_inches="tight"); plt.close()

        # Atenção: visualizar pesos médios da última camada de atenção
        try:
            attn_layers = [l for l in model_final.layers
                           if isinstance(l, MultiHeadAttention)]
            if attn_layers:
                log.info(f"  Transformer tem {len(attn_layers)} camadas de atenção.")
        except Exception:
            pass

        # ── Previsão futura recursiva ──────────────────────────────────────
        try:
            from config import FUTURE_END_DATE
            future_idx = pd.date_range(
                data["y_series"].index[-1]+pd.Timedelta(days=1),
                pd.to_datetime(FUTURE_END_DATE), freq="D"
            )
            cal_fut = adicionar_features_calendario(pd.Series(np.nan, index=future_idx))
            if weather is not None:
                for col, val in weather.mean().items():
                    if col in data["feat_cols"]:
                        cal_fut[col] = val
            feat_fut = np.zeros((len(future_idx), data["n_feat"]))
            for i, col in enumerate(data["feat_cols"][:data["n_feat"]]):
                if col in cal_fut.columns:
                    feat_fut[:, i] = cal_fut[col].values[:len(future_idx)]
            fut_sc = data["sc_X"].transform(feat_fut)

            model_fut = build_transformer(DL_LOOKBACK, data["n_feat"], **best_params)
            model_fut.fit(data["X_all"], data["y_all"],
                          epochs=DL_EPOCHS, batch_size=DL_BATCH_SIZE,
                          callbacks=get_callbacks(), verbose=0)

            preds_fut = recursive_predict_transformer(
                model_fut, data["X_all"][-1], fut_sc, data["sc_y"], len(future_idx)
            )
            fut_series = pd.Series(preds_fut, index=future_idx)

            all_pred = model_fut.predict(data["X_all"], verbose=0).flatten()
            all_pred_inv = data["sc_y"].inverse_transform(
                all_pred.reshape(-1,1)).flatten()
            all_y_inv = data["sc_y"].inverse_transform(
                data["y_all"].reshape(-1,1)).flatten()
            resids = all_y_inv - all_pred_inv
            rng    = np.random.default_rng(RANDOM_STATE)
            boots  = [preds_fut + rng.choice(resids, size=len(future_idx), replace=True)
                      for _ in range(200)]
            boots  = np.array(boots)
            plot_future_forecast(
                data["y_series"], fut_series,
                pd.Series(np.percentile(boots,2.5,axis=0),  index=future_idx),
                pd.Series(np.percentile(boots,97.5,axis=0), index=future_idx),
                tid, nome_modelo, OUTPUT_FUTURE
            )
            pd.DataFrame({
                "data": future_idx, "prev_mean": preds_fut,
                "ic_lower_95": np.percentile(boots,2.5,axis=0),
                "ic_upper_95": np.percentile(boots,97.5,axis=0),
            }).to_csv(f"{OUTPUT_FUTURE}/futuro_{tid}_{nome_modelo}.csv", index=False)
        except Exception as e:
            log.warning(f"  Previsão futura Transformer falhou: {e}")

        resultados.append({
            "transformador": tid, "nome_plot": MAPA_PLOT.get(tid,tid),
            "modelo": nome_modelo, "abordagem": "Transformer",
            "best_params": str(best_params), "MAE_val": round(best_mae,4),
            **{k: round(v,4) for k,v in metrics.items()},
        })

    return pd.DataFrame(resultados)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lite","full","both"], default="lite",
                        help="lite: roda no Ryzen 5700G em ~20-40min | "
                             "full: melhores resultados, ~2-4h CPU | "
                             "both: roda os dois e compara")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES)].copy()

    weather = None
    try: weather = carregar_weather()
    except Exception as e: log.warning(f"Clima: {e}")

    modes = ["lite","full"] if args.mode == "both" else [args.mode]
    all_res = []

    for m in modes:
        log.info(f"\n=== TRANSFORMER [{m.upper()}] ===")
        res = treinar_transformer(df, weather, mode=m)
        res.to_csv(f"{OUTPUT_TRANSFORMER}/metricas_Transformer_{m}.csv", index=False)
        all_res.append(res)

    df_all = pd.concat(all_res, ignore_index=True)
    print(df_all[["transformador","modelo","RMSE","MAE","sMAPE","R2"]].to_string(index=False))


if __name__ == "__main__":
    import argparse
    main()
