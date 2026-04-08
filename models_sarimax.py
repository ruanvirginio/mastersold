# =============================================================================
# models_sarimax.py — SARIMAX (baseline estatístico obrigatório em artigos)
# =============================================================================
# SARIMAX é o benchmark estatístico canônico para séries temporais sazonais.
# Sua inclusão é EXIGIDA por revisores de revistas como Electric Power Systems
# Research, Applied Energy e Energy & Buildings.
#
# HPT: seleção de ordem (p,d,q)(P,D,Q,s) pelo AIC no conjunto de TREINO.
# Avaliação final: TESTE (2024) — acesso único.
#
# Referência:
#   Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
#   Time Series Analysis: Forecasting and Control (5th ed.). Wiley.
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, random
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import (
    DATA_PATH, TRANSFORMADORES, MAPA_PLOT, RANDOM_STATE,
    SARIMAX_ORDERS, END_TRAIN, END_VALIDATION,
    OUTPUT_SARIMAX, OUTPUT_FUTURE
)
from utils import (
    preparar_serie_diaria, adicionar_features_calendario,
    compute_metrics, get_splits,
    plot_forecast_3way, plot_residuals, plot_future_forecast, log
)

random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)


def treinar_sarimax(df: pd.DataFrame) -> pd.DataFrame:
    resultados = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [SARIMAX]\n{'='*60}")
        y      = preparar_serie_diaria(df_tr)
        splits = get_splits(y)

        y_train = y.loc[splits["train"]]
        y_val   = y.loc[splits["val"]]
        y_test  = y.loc[splits["test"]]
        y_tv    = pd.concat([y_train, y_val])

        # ── Exógenas de calendário (capturam sazonalidade anual que s=7 não captura)
        # JUSTIFICATIVA: SARIMAX com s=7 captura padrão semanal. Para capturar
        # sazonalidade anual (s=365 é computacionalmente inviável), adicionamos
        # regressores de calendário cíclicos como variáveis exógenas.
        # Referência: Hyndman & Athanasopoulos (2021), Forecasting: P&P, Seção 10.1
        cal_full = adicionar_features_calendario(y)
        exog_cols_sarimax = ["dayofyear_sin", "dayofyear_cos",
                             "month_sin", "month_cos",
                             "weekday_sin", "weekday_cos",
                             "is_weekend", "is_holiday"]
        exog_full  = cal_full[exog_cols_sarimax]
        exog_train = exog_full.loc[splits["train"]]
        exog_val   = exog_full.loc[splits["val"]]
        exog_test  = exog_full.loc[splits["test"]]
        exog_tv    = exog_full.loc[y_tv.index]

        # ── HPT: seleciona ordem pelo AIC no TREINO ───────────────────────
        best_aic, best_order, best_sorder = np.inf, None, None

        for order, sorder in SARIMAX_ORDERS:
            try:
                model = SARIMAX(y_train, exog=exog_train,
                                order=order, seasonal_order=sorder,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                result = model.fit(disp=False, maxiter=200)
                log.info(f"     {order}x{sorder} → AIC={result.aic:.2f}")
                if result.aic < best_aic:
                    best_aic   = result.aic
                    best_order = order
                    best_sorder= sorder
            except Exception as e:
                log.warning(f"  SARIMAX {order}x{sorder} falhou: {e}")

        if best_order is None:
            log.error(f"  {tid}: nenhuma ordem SARIMAX convergiu.")
            continue

        log.info(f"  ✓ HPT → {best_order}x{best_sorder} AIC={best_aic:.2f}")

        # ── Validação (para plot) ─────────────────────────────────────────
        try:
            m_val = SARIMAX(y_train, exog=exog_train,
                            order=best_order, seasonal_order=best_sorder,
                            enforce_stationarity=False, enforce_invertibility=False)
            r_val = m_val.fit(disp=False, maxiter=200)
            pred_val = r_val.forecast(steps=len(y_val), exog=exog_val)
            pred_val.index = y_val.index
        except Exception as e:
            log.warning(f"  Previsão de validação falhou: {e}")
            pred_val = pd.Series(np.nan, index=y_val.index)

        # ── Retreinar em TREINO+VAL, avaliar TESTE ────────────────────────
        try:
            m_tv = SARIMAX(y_tv, exog=exog_tv,
                           order=best_order, seasonal_order=best_sorder,
                           enforce_stationarity=False, enforce_invertibility=False)
            r_tv = m_tv.fit(disp=False, maxiter=200)
            pred_test = r_tv.forecast(steps=len(y_test), exog=exog_test)
            pred_test.index = y_test.index
        except Exception as e:
            log.error(f"  Previsão de teste falhou: {e}")
            pred_test = pd.Series(np.nan, index=y_test.index)

        metrics = compute_metrics(y_test, pred_test)
        log.info(f"  ✓ TESTE → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

        plot_forecast_3way(y, pred_val, pred_test, tid, "SARIMAX", OUTPUT_SARIMAX)
        plot_residuals(y_test, pred_test, tid, "SARIMAX", OUTPUT_SARIMAX)

        # Diagnóstico de resíduos (plot padrão statsmodels)
        try:
            fig = r_tv.plot_diagnostics(figsize=(12, 8))
            fig.suptitle(f"{MAPA_PLOT.get(tid,tid)} — SARIMAX Diagnóstico de Resíduos",
                         fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_SARIMAX}/diagnostics_{tid}_SARIMAX.svg",
                        dpi=150, bbox_inches="tight"); plt.close()
        except Exception:
            pass

        # ── Previsão futura ───────────────────────────────────────────────
        try:
            from config import FUTURE_END_DATE
            future_idx = pd.date_range(y.index[-1]+pd.Timedelta(days=1),
                                       pd.to_datetime(FUTURE_END_DATE), freq="D")
            exog_future = adicionar_features_calendario(
                pd.Series(np.nan, index=future_idx)
            )[exog_cols_sarimax]

            m_full = SARIMAX(y, exog=exog_full,
                             order=best_order, seasonal_order=best_sorder,
                             enforce_stationarity=False, enforce_invertibility=False)
            r_full = m_full.fit(disp=False, maxiter=200)
            fc_obj = r_full.get_forecast(steps=len(future_idx), exog=exog_future)
            pred_fut   = fc_obj.predicted_mean
            pred_fut.index = future_idx
            ci         = fc_obj.conf_int(alpha=0.05)
            lower_fut  = pd.Series(ci.iloc[:,0].values, index=future_idx)
            upper_fut  = pd.Series(ci.iloc[:,1].values, index=future_idx)

            plot_future_forecast(y, pred_fut, lower_fut, upper_fut,
                                 tid, "SARIMAX", OUTPUT_FUTURE)
            pd.DataFrame({
                "data": future_idx,
                "prev_mean":   pred_fut.values,
                "ic_lower_95": lower_fut.values,
                "ic_upper_95": upper_fut.values,
            }).to_csv(f"{OUTPUT_FUTURE}/futuro_{tid}_SARIMAX.csv", index=False)
        except Exception as e:
            log.warning(f"  Previsão futura SARIMAX falhou: {e}")

        resultados.append({
            "transformador": tid, "nome_plot": MAPA_PLOT.get(tid,tid),
            "modelo": "SARIMAX", "abordagem": "Estatístico",
            "best_params": f"{best_order}x{best_sorder}",
            "AIC_treino": round(best_aic, 2),
            **{k: round(v,4) for k,v in metrics.items()},
        })

    return pd.DataFrame(resultados)


def main():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES)].copy()
    res = treinar_sarimax(df)
    res.to_csv(f"{OUTPUT_SARIMAX}/metricas_SARIMAX.csv", index=False)
    print(res[["transformador","modelo","RMSE","MAE","sMAPE","R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
