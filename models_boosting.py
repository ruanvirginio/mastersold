# =============================================================================
# models_boosting.py — XGBoost · LightGBM · GradientBoosting
# Univariado (UV) e Multivariado (MV) com HPT sem data leakage
# =============================================================================
# Metodologia (sem data leakage):
#   TREINO (2021-2022) → ajuste
#   VALIDAÇÃO (2023)   → HPT: seleciona melhores hiperparâmetros
#   TESTE (2024)       → avaliação final única, reportada no artigo
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, random
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from lightgbm  import LGBMRegressor
from xgboost   import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures

from config import (
    DATA_PATH, TRANSFORMADORES, MAPA_PLOT, RANDOM_STATE,
    LAGS, ROLLING_WINDOW, END_TRAIN, END_VALIDATION,
    OUTPUT_UV, OUTPUT_MV, OUTPUT_FUTURE, FUTURE_END_DATE
)
from utils import (
    preparar_serie_diaria, adicionar_features_calendario,
    carregar_weather, compute_metrics, get_splits,
    plot_forecast_3way, plot_residuals, plot_future_forecast,
    selecionar_features, log
)

random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

EXOG_COLS = [
    "weekday_sin","weekday_cos","month_sin","month_cos",
    "dayofyear_sin","dayofyear_cos","is_weekend","is_holiday",
    "temp_max_anos","temp_min_anos","temp_media_anos","precipitacao_anos"
]

# ─────────────────────────────────────────────────────────────────────────────
# GRADE DE HIPERPARÂMETROS
# ─────────────────────────────────────────────────────────────────────────────
def get_param_grid():
    return {
        "LGBM": {
            "estimator": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            "params": [
                {"n_estimators":200, "learning_rate":0.05, "max_depth":5, "num_leaves":31},
                {"n_estimators":300, "learning_rate":0.03, "max_depth":6, "num_leaves":50},
                {"n_estimators":500, "learning_rate":0.01, "max_depth":6, "num_leaves":50},
                {"n_estimators":200, "learning_rate":0.05, "max_depth":4, "num_leaves":20},
            ]
        },
        "XGBoost": {
            "estimator": XGBRegressor(
                objective="reg:squarederror", random_state=RANDOM_STATE,
                n_jobs=-1, verbosity=0),
            "params": [
                {"n_estimators":200, "learning_rate":0.05, "max_depth":4, "subsample":0.8},
                {"n_estimators":300, "learning_rate":0.03, "max_depth":5, "subsample":0.8},
                {"n_estimators":500, "learning_rate":0.01, "max_depth":4, "subsample":0.9},
                {"n_estimators":200, "learning_rate":0.05, "max_depth":3, "subsample":1.0},
            ]
        },
        "GradientBoosting": {
            "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "params": [
                {"n_estimators":200, "learning_rate":0.05, "max_depth":3, "subsample":0.8},
                {"n_estimators":300, "learning_rate":0.03, "max_depth":4, "subsample":0.8},
                {"n_estimators":100, "learning_rate":0.10, "max_depth":3, "subsample":1.0},
            ]
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# PREVISÃO FUTURA COM BOOTSTRAP IC
# ─────────────────────────────────────────────────────────────────────────────
def prever_futuro_bootstrap(forecaster, y_full, exog_full, tid, modelo,
                             output_dir, n_bootstrap=200):
    """
    Retreina o modelo em TODOS os dados e projeta para o futuro.
    Intervalo de confiança via bootstrap residual (Efron & Tibshirani, 1993).

    NÃO usa dados futuros no treino — projeta iterativamente para frente.
    """
    from config import FUTURE_END_DATE
    future_idx = pd.date_range(
        y_full.index[-1] + pd.Timedelta(days=1),
        pd.to_datetime(FUTURE_END_DATE), freq="D"
    )
    if len(future_idx) == 0:
        return

    # Retreinar em todos os dados disponíveis
    forecaster.fit(y=y_full, exog=exog_full if exog_full is not None else None)

    # Previsão pontual
    if exog_full is not None:
        # Gerar exógenas futuras (calendário — clima não disponível, usar média histórica)
        cal_future = adicionar_features_calendario(
            pd.Series(np.nan, index=future_idx)
        )
        weather_mean = exog_full[[c for c in exog_full.columns
                                  if c not in cal_future.columns]].mean()
        for col, val in weather_mean.items():
            cal_future[col] = val
        exog_future = cal_future[[c for c in EXOG_COLS if c in cal_future.columns]]
        pred_future = forecaster.predict(steps=len(future_idx), exog=exog_future)
    else:
        pred_future = forecaster.predict(steps=len(future_idx))

    pred_series = pd.Series(pred_future.values, index=future_idx)

    # Bootstrap residual para IC
    # 1) Obter resíduos in-sample
    y_fitted = forecaster.predict(steps=len(y_full))
    resids   = (y_full.values - y_fitted.values[:len(y_full)])
    resids   = resids[~np.isnan(resids)]

    boots = []
    rng   = np.random.default_rng(RANDOM_STATE)
    for _ in range(n_bootstrap):
        noise = rng.choice(resids, size=len(future_idx), replace=True)
        boots.append(pred_series.values + noise)

    boots     = np.array(boots)
    lower_95  = np.percentile(boots, 2.5,  axis=0)
    upper_95  = np.percentile(boots, 97.5, axis=0)

    plot_future_forecast(y_full, pred_series,
                         pd.Series(lower_95, index=future_idx),
                         pd.Series(upper_95, index=future_idx),
                         tid, modelo, output_dir)

    # Salvar CSV da previsão futura
    df_fut = pd.DataFrame({
        "data":        future_idx,
        "prev_mean":   pred_series.values,
        "ic_lower_95": lower_95,
        "ic_upper_95": upper_95,
    })
    df_fut.to_csv(f"{output_dir}/futuro_{tid}_{modelo}.csv", index=False)
    log.info(f"  Previsão futura salva: {output_dir}/futuro_{tid}_{modelo}.csv")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def treinar_boosting(df: pd.DataFrame, weather=None, mode="UV") -> pd.DataFrame:
    """
    mode="UV" → univariado (sem exógenas)
    mode="MV" → multivariado (+ calendário + clima)
    """
    output_dir = OUTPUT_UV if mode == "UV" else OUTPUT_MV
    param_grid = get_param_grid()
    resultados = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [{mode}]\n{'='*60}")
        y      = preparar_serie_diaria(df_tr)
        splits = get_splits(y)

        # Exógenas (apenas MV)
        exog = None
        if mode == "MV":
            cal = adicionar_features_calendario(y)
            if weather is not None:
                exog_df = cal.join(weather, how="left").bfill().ffill()
            else:
                exog_df = cal
            cols_disp = [c for c in EXOG_COLS if c in exog_df.columns]
            exog = exog_df[cols_disp]

        # Conjuntos
        y_tv  = y.loc[splits["train"].union(splits["val"])]
        exog_tv = exog.loc[y_tv.index] if exog is not None else None

        cv_val  = TimeSeriesFold(steps=len(splits["val"]),
                                 initial_train_size=len(splits["train"]), refit=False)
        cv_test = TimeSeriesFold(steps=len(splits["test"]),
                                 initial_train_size=len(y_tv),            refit=False)

        for nome, info in param_grid.items():
            log.info(f"  → {nome}")
            best_mae, best_params = np.inf, None

            # ── ETAPA 1: HPT sobre VALIDAÇÃO ──────────────────────────────
            for params in info["params"]:
                est = info["estimator"].set_params(**params)
                fc  = ForecasterRecursive(
                    estimator=est, lags=LAGS,
                    window_features=RollingFeatures(stats=["mean"],
                                                    window_sizes=[ROLLING_WINDOW])
                )
                kw = dict(forecaster=fc, y=y_tv, cv=cv_val,
                          metric="mean_absolute_error", verbose=False)
                if exog_tv is not None:
                    kw["exog"] = exog_tv
                _, pv = backtesting_forecaster(**kw)
                mae = mean_absolute_error(y.loc[pv.index], pv["pred"])
                if mae < best_mae:
                    best_mae, best_params = mae, params

            log.info(f"  ✓ HPT → {best_params} | MAE_val={best_mae:.4f}")

            # ── ETAPA 2: Retreinar TREINO+VAL com melhores params ─────────
            best_est = info["estimator"].set_params(**best_params)
            fc_final = ForecasterRecursive(
                estimator=best_est, lags=LAGS,
                window_features=RollingFeatures(stats=["mean"],
                                                window_sizes=[ROLLING_WINDOW])
            )

            # ── ETAPA 3: TESTE — acesso único ─────────────────────────────
            kw_test = dict(forecaster=fc_final, y=y, cv=cv_test,
                           metric="mean_absolute_error", verbose=False)
            if exog is not None:
                kw_test["exog"] = exog
            _, pt = backtesting_forecaster(**kw_test)

            y_true = y.loc[pt.index]
            metrics = compute_metrics(y_true, pt["pred"])
            log.info(f"  ✓ TESTE → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

            # ── Plots ──────────────────────────────────────────────────────
            # Previsão de validação (só para plot)
            fc_val_plot = ForecasterRecursive(
                estimator=info["estimator"].set_params(**best_params),
                lags=LAGS,
                window_features=RollingFeatures(stats=["mean"],
                                                window_sizes=[ROLLING_WINDOW])
            )
            kw_vp = dict(forecaster=fc_val_plot, y=y_tv, cv=cv_val,
                         metric="mean_absolute_error", verbose=False)
            if exog_tv is not None:
                kw_vp["exog"] = exog_tv
            _, pv_plot = backtesting_forecaster(**kw_vp)

            plot_forecast_3way(y, pv_plot["pred"], pt["pred"],
                               tid, nome, output_dir, suffix=f"_{mode}")
            plot_residuals(y_true, pt["pred"], tid, nome, output_dir)

            # Feature importance via API skforecast
            try:
                imp = fc_final.get_feature_importances()
                imp_s = imp.sort_values("importance", ascending=True).tail(20)
                plt.figure(figsize=(7,5))
                plt.barh(imp_s["feature"], imp_s["importance"], color="#2563eb")
                plt.title(f"{MAPA_PLOT.get(tid,tid)} — {nome} [{mode}]: Feature Importance")
                plt.tight_layout()
                plt.savefig(f"{output_dir}/importance_{tid}_{nome}_{mode}.svg",
                            dpi=150, bbox_inches="tight"); plt.close()
            except Exception as e:
                log.warning(f"Importância não disponível: {e}")

            # ── Previsão futura (retreina em todos os dados) ───────────────
            fc_fut = ForecasterRecursive(
                estimator=info["estimator"].set_params(**best_params),
                lags=LAGS,
                window_features=RollingFeatures(stats=["mean"],
                                                window_sizes=[ROLLING_WINDOW])
            )
            try:
                prever_futuro_bootstrap(fc_fut, y, exog, tid, f"{nome}_{mode}",
                                        OUTPUT_FUTURE)
            except Exception as e:
                log.warning(f"Previsão futura falhou ({nome}): {e}")

            resultados.append({
                "transformador": tid, "nome_plot": MAPA_PLOT.get(tid, tid),
                "modelo": nome, "abordagem": mode,
                "best_params": str(best_params), "MAE_val": round(best_mae, 4),
                **{k: round(v, 4) for k, v in metrics.items()},
            })

    return pd.DataFrame(resultados)


def main():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES)].copy()

    log.info("=== BOOSTING UNIVARIADO ===")
    res_uv = treinar_boosting(df, weather=None, mode="UV")
    res_uv.to_csv(f"{OUTPUT_UV}/metricas_UV.csv", index=False)

    log.info("=== BOOSTING MULTIVARIADO ===")
    try:
        weather = carregar_weather()
    except Exception as e:
        log.warning(f"Clima não carregado: {e}"); weather = None
    res_mv = treinar_boosting(df, weather=weather, mode="MV")
    res_mv.to_csv(f"{OUTPUT_MV}/metricas_MV.csv", index=False)

    print(pd.concat([res_uv, res_mv])
          [["transformador","modelo","abordagem","RMSE","MAE","sMAPE","R2"]]
          .sort_values(["transformador","RMSE"]).to_string(index=False))


if __name__ == "__main__":
    main()
