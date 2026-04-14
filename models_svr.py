# =============================================================================
# models_svr.py — Support Vector Regression (SVR) · UV e MV
# =============================================================================
# Estrutura espelhada ao models_boosting.py para comparação simétrica.
#
# Fluxo UV:
#   TREINO → HPT (validação) → retreino treino+val → teste
#
# Fluxo MV (idêntico ao boosting):
#   TREINO → seleção features (proxy LGBM, gain 95%) → HPT (validação)
#          → retreino treino+val → teste
#
# Previsão de teste: direta (usa lags reais do passado conhecido).
# Lags de calendário são determinísticos — sem leakage em usá-los no teste.
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, random
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

from config import (
    DATA_PATH, TRANSFORMADORES, MAPA_PLOT, RANDOM_STATE,
    SVR_LAGS, END_TRAIN, END_VALIDATION, OUTPUT_SVR, OUTPUT_FUTURE,
    FUTURE_END_DATE
)
from utils import (
    preparar_serie_diaria, adicionar_features_calendario,
    carregar_weather, build_lag_matrix, compute_metrics, get_splits,
    plot_forecast_3way, plot_residuals, plot_future_forecast, log
)

random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

# Todas as features exógenas candidatas (modo MV)
EXOG_COLS = [
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "dayofyear_sin", "dayofyear_cos", "is_weekend", "is_holiday",
    "temp_max_anos", "temp_min_anos", "temp_media_anos", "precipitacao_anos"
]

FEATURE_IMPORTANCE_THRESHOLD = 0.95

# Grade HPT
SVR_PARAM_GRID = [
    {"svr__C": 1,    "svr__epsilon": 0.1, "svr__gamma": "scale"},
    {"svr__C": 10,   "svr__epsilon": 0.1, "svr__gamma": "scale"},
    {"svr__C": 100,  "svr__epsilon": 0.1, "svr__gamma": "scale"},
    {"svr__C": 10,   "svr__epsilon": 0.5, "svr__gamma": "scale"},
    {"svr__C": 100,  "svr__epsilon": 0.5, "svr__gamma": "auto"},
    {"svr__C": 1000, "svr__epsilon": 0.1, "svr__gamma": "scale"},
]


# ─────────────────────────────────────────────────────────────────────────────
# SELEÇÃO DE FEATURES (igual ao boosting — proxy LGBM sobre treino)
# ─────────────────────────────────────────────────────────────────────────────
def selecionar_features_mv(y_train, exog_train, tid, output_dir):
    """Seleciona features exógenas por gain acumulado ≥ 95% no treino."""
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

    exog_imp["imp_pct"] = exog_imp["importance"] / exog_imp["importance"].sum()
    exog_imp["imp_cum"] = exog_imp["imp_pct"].cumsum()
    selected = exog_imp.loc[
        exog_imp["imp_cum"].shift(1, fill_value=0) < FEATURE_IMPORTANCE_THRESHOLD,
        "feature"
    ].tolist()
    if not selected:
        selected = [exog_imp.iloc[0]["feature"]]

    log.info(f"  [FS-SVR] {tid}: {len(selected)}/{len(exog_imp)} features → {selected}")

    # Plot seleção
    colors = ["#2563eb" if f in selected else "#94a3b8" for f in exog_imp["feature"]]
    fig, ax = plt.subplots(figsize=(8, max(4, len(exog_imp) * 0.45 + 1)))
    ax.barh(exog_imp["feature"][::-1], exog_imp["importance"][::-1],
            color=colors[::-1])
    ax.set_title(f"{MAPA_PLOT.get(tid,tid)} — SVR: Feature Selection\n"
                 f"Blue = selected ({len(selected)}/{len(exog_imp)})", fontsize=9)
    ax.set_xlabel("Importance (gain)")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/feature_selection_{tid}_SVR_MV.svg",
                dpi=150, bbox_inches="tight")
    plt.close()
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUÇÃO DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def construir_X(y, exog_cols_sel=None, weather=None):
    """
    Monta matriz de features: lags + calendário [+ clima se MV].
    Retorna DataFrame alinhado com y (sem NaN iniciais removidos).
    """
    cal = adicionar_features_calendario(y)
    lag_df = build_lag_matrix(y, SVR_LAGS)   # inclui coluna 'y'

    if exog_cols_sel is not None:
        # MV: adicionar features exógenas selecionadas
        if weather is not None:
            cal = cal.join(weather, how="left").bfill().ffill()
        exog_sel = cal[[c for c in exog_cols_sel if c in cal.columns]]
        df = lag_df.join(exog_sel, how="left").dropna()
    else:
        # UV: só lags + calendário básico
        cal_cols = [c for c in cal.columns]
        df = lag_df.join(cal[cal_cols], how="left").dropna()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def treinar_svr(df: pd.DataFrame, weather=None, mode="UV") -> pd.DataFrame:
    """
    mode='UV' → lags + calendário básico (sem clima)
    mode='MV' → lags + features exógenas selecionadas (calendário + clima)
    """
    resultados = []
    t_end = pd.to_datetime(END_TRAIN)
    v_end = pd.to_datetime(END_VALIDATION)

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [SVR-{mode}]\n{'='*60}")
        y      = preparar_serie_diaria(df_tr)
        splits = get_splits(y)

        # ── Seleção de features (MV apenas, sobre treino) ─────────────────
        sel_features = None
        if mode == "MV":
            cal_full = adicionar_features_calendario(y)
            if weather is not None:
                exog_full_df = cal_full.join(weather, how="left").bfill().ffill()
            else:
                exog_full_df = cal_full
            cols_disp = [c for c in EXOG_COLS if c in exog_full_df.columns]
            exog_full_df = exog_full_df[cols_disp]

            exog_train_only = exog_full_df.loc[splits["train"]]
            y_train_only    = y.loc[splits["train"]]
            sel_features = selecionar_features_mv(
                y_train_only, exog_train_only, tid, OUTPUT_SVR
            )

        # ── Construir feature matrix ───────────────────────────────────────
        feat_df = construir_X(
            y,
            exog_cols_sel=sel_features,
            weather=weather if mode == "MV" else None
        )
        feat_cols = [c for c in feat_df.columns if c != "y"]
        X_all   = feat_df[feat_cols].values
        y_all   = feat_df["y"].values
        idx_all = feat_df.index

        mask_tr  = idx_all <= t_end
        mask_val = (idx_all > t_end) & (idx_all <= v_end)
        mask_te  = idx_all > v_end

        X_tr, y_tr = X_all[mask_tr],  y_all[mask_tr]
        X_vl, y_vl = X_all[mask_val], y_all[mask_val]
        X_te, y_te = X_all[mask_te],  y_all[mask_te]
        idx_te     = idx_all[mask_te]

        # ── HPT sobre VALIDAÇÃO ───────────────────────────────────────────
        best_mae, best_params, best_pipe = np.inf, None, None
        for params in SVR_PARAM_GRID:
            pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
            pipe.set_params(**params)
            pipe.fit(X_tr, y_tr)
            mae = np.mean(np.abs(y_vl - pipe.predict(X_vl)))
            log.info(f"     {params} → MAE_val={mae:.4f}")
            if mae < best_mae:
                best_mae, best_params, best_pipe = mae, params, pipe

        log.info(f"  ✓ HPT → {best_params} | MAE_val={best_mae:.4f}")

        # ── Retreinar TREINO+VAL, avaliar TESTE ───────────────────────────
        X_tv = np.vstack([X_tr, X_vl])
        y_tv = np.concatenate([y_tr, y_vl])
        pipe_final = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
        pipe_final.set_params(**best_params)
        pipe_final.fit(X_tv, y_tv)

        pred_test   = pipe_final.predict(X_te)
        y_true_test = pd.Series(y_te, index=idx_te)
        y_pred_test = pd.Series(pred_test, index=idx_te)
        metrics     = compute_metrics(y_true_test, y_pred_test)
        log.info(f"  ✓ TESTE → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

        # Previsão validação para plot
        pred_val_plot = pd.Series(best_pipe.predict(X_vl), index=idx_all[mask_val])
        nome_modelo   = f"SVR_{mode}"
        plot_forecast_3way(y, pred_val_plot, y_pred_test,
                           tid, nome_modelo, OUTPUT_SVR, suffix="")
        plot_residuals(y_true_test, y_pred_test, tid, nome_modelo, OUTPUT_SVR)

        # ── Previsão futura ───────────────────────────────────────────────
        try:
            future_idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1),
                                       pd.to_datetime(FUTURE_END_DATE), freq="D")
            # Features futuras (calendário determinístico + média climática histórica)
            cal_fut = adicionar_features_calendario(pd.Series(np.nan, index=future_idx))
            if mode == "MV" and weather is not None and sel_features:
                for col in [c for c in sel_features if c not in cal_fut.columns]:
                    if col in weather.columns:
                        cal_fut[col] = weather[col].mean()
            fut_feat_df = build_lag_matrix(
                pd.concat([y, pd.Series(np.nan, index=future_idx)]), SVR_LAGS
            ).loc[future_idx]
            if mode == "MV" and sel_features:
                for col in sel_features:
                    if col in cal_fut.columns:
                        fut_feat_df[col] = cal_fut[col].values
            fut_feat_df = fut_feat_df[[c for c in feat_cols if c in fut_feat_df.columns]].fillna(0)

            X_all_fut = X_all
            pipe_fut  = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
            pipe_fut.set_params(**best_params)
            pipe_fut.fit(X_all_fut, y_all)

            preds_fut  = pipe_fut.predict(fut_feat_df.values)
            fut_series = pd.Series(preds_fut, index=future_idx)
            resids     = y_all - pipe_fut.predict(X_all_fut)
            rng        = np.random.default_rng(RANDOM_STATE)
            boots      = np.array([preds_fut + rng.choice(resids, size=len(future_idx), replace=True)
                                   for _ in range(200)])
            plot_future_forecast(
                y, fut_series,
                pd.Series(np.percentile(boots, 2.5,  axis=0), index=future_idx),
                pd.Series(np.percentile(boots, 97.5, axis=0), index=future_idx),
                tid, nome_modelo, OUTPUT_FUTURE
            )
        except Exception as e:
            log.warning(f"  Previsão futura SVR-{mode} falhou: {e}")

        resultados.append({
            "transformador": tid, "nome_plot": MAPA_PLOT.get(tid, tid),
            "modelo": "SVR", "abordagem": mode,
            "features_sel": str(sel_features) if sel_features else "N/A",
            "n_features_sel": len(sel_features) if sel_features else 0,
            "best_params": str(best_params), "MAE_val": round(best_mae, 4),
            **{k: round(v, 4) for k, v in metrics.items()},
        })

    return pd.DataFrame(resultados)


def main():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES)].copy()

    log.info("=== SVR UNIVARIADO ===")
    res_uv = treinar_svr(df, weather=None, mode="UV")
    res_uv.to_csv(f"{OUTPUT_SVR}/metricas_SVR_UV.csv", index=False)

    log.info("=== SVR MULTIVARIADO ===")
    try:
        weather = carregar_weather()
    except Exception as e:
        log.warning(f"Clima não carregado: {e}"); weather = None
    res_mv = treinar_svr(df, weather=weather, mode="MV")
    res_mv.to_csv(f"{OUTPUT_SVR}/metricas_SVR_MV.csv", index=False)

    res = pd.concat([res_uv, res_mv])
    res.to_csv(f"{OUTPUT_SVR}/metricas_SVR.csv", index=False)
    print(res[["transformador", "modelo", "abordagem", "RMSE", "MAE", "sMAPE", "R2"]]
          .sort_values(["transformador", "abordagem"]).to_string(index=False))


if __name__ == "__main__":
    main()
