# =============================================================================
# models_svr.py — Support Vector Regression (SVR) para Previsão de Séries Temporais
# =============================================================================
# SVR com kernel RBF usando features de lag + calendário.
# HPT via GridSearch sobre o conjunto de VALIDAÇÃO (sem data leakage).
#
# Abordagem: recursive multi-step — o modelo prevê 1 passo por vez,
# realimentando as previsões como lags para os próximos passos.
# Isso garante que NUNCA usamos y[t] real para prever y[t+1] no período de teste.
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

from config import (
    DATA_PATH, TRANSFORMADORES, MAPA_PLOT, RANDOM_STATE,
    SVR_LAGS, END_TRAIN, END_VALIDATION, OUTPUT_SVR, OUTPUT_FUTURE,
    GENERATE_FUTURE_FORECAST
)
from utils import (
    preparar_serie_diaria, adicionar_features_calendario,
    build_lag_matrix, compute_metrics, get_splits,
    plot_forecast_3way, plot_residuals, plot_future_forecast,
    selecionar_features, log
)

random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

# Grade HPT para SVR
SVR_PARAM_GRID = [
    {"svr__C": 1,    "svr__epsilon": 0.1, "svr__gamma": "scale"},
    {"svr__C": 10,   "svr__epsilon": 0.1, "svr__gamma": "scale"},
    {"svr__C": 100,  "svr__epsilon": 0.1, "svr__gamma": "scale"},
    {"svr__C": 10,   "svr__epsilon": 0.5, "svr__gamma": "scale"},
    {"svr__C": 100,  "svr__epsilon": 0.5, "svr__gamma": "auto"},
    {"svr__C": 1000, "svr__epsilon": 0.1, "svr__gamma": "scale"},
]


def construir_features_svr(y: pd.Series) -> pd.DataFrame:
    """
    Constrói feature matrix para SVR:
    lags da série + features de calendário.
    Todos os lags são deslocamentos PASSADOS → sem lookahead.
    """
    cal = adicionar_features_calendario(y)
    lag_df = build_lag_matrix(y, SVR_LAGS)

    df = lag_df.join(cal, how="left")
    df = df.dropna()
    return df


def recursive_predict(model, last_known: np.ndarray, selected_features: list,
                      cal_future: pd.DataFrame, steps: int) -> np.ndarray:
    """
    Previsão multi-step recursiva para SVR.

    A cada passo t:
      1. Reconstrói o vetor de features NA MESMA ORDEM usada no treino do pipeline.
      2. Features de lag: extraídas do buffer (reais + previsões anteriores).
      3. Features de calendário: lidas de cal_future (determinísticas — sem leakage).
      4. Acrescenta ŷ[t] ao buffer para os próximos passos.

    Parâmetros
    ----------
    model             : pipeline (StandardScaler + SVR) já treinado
    last_known        : array com valores históricos reais (pelo menos max_lag pontos)
    selected_features : lista ordenada de features EXATAMENTE como treinado
                        (e.g. ["lag_1", "lag_7", "weekday_sin", "is_holiday", ...])
    cal_future        : DataFrame indexado por data com colunas de calendário
    steps             : número de passos a prever
    """
    lag_nums = [int(f.split("_")[1]) for f in selected_features if f.startswith("lag_")]
    max_lag  = max(lag_nums) if lag_nums else 1
    buffer   = list(last_known[-max_lag:])
    preds    = []

    for step in range(steps):
        row = []
        for feat in selected_features:
            if feat.startswith("lag_"):
                lag_n = int(feat.split("_")[1])
                row.append(buffer[-lag_n])
            else:
                if step < len(cal_future) and feat in cal_future.columns:
                    row.append(float(cal_future.iloc[step][feat]))
                else:
                    row.append(0.0)
        x    = np.array(row, dtype=np.float64).reshape(1, -1)
        yhat = model.predict(x)[0]
        preds.append(yhat)
        buffer.append(yhat)   # alimenta com previsão, NUNCA com valor real futuro

    return np.array(preds)


def treinar_svr(df: pd.DataFrame) -> pd.DataFrame:
    resultados = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [SVR]\n{'='*60}")
        y      = preparar_serie_diaria(df_tr)
        splits = get_splits(y)

        feat_df = construir_features_svr(y)
        feature_names = [c for c in feat_df.columns if c != "y"]
        X_all = feat_df[feature_names].values
        y_all = feat_df["y"].values
        idx_all = feat_df.index

        # Máscaras temporais alinhadas ao feat_df (que começa após max_lag)
        t_end  = pd.to_datetime(END_TRAIN)
        v_end  = pd.to_datetime(END_VALIDATION)

        mask_tr  = idx_all <= t_end
        mask_val = (idx_all > t_end) & (idx_all <= v_end)
        mask_te  = idx_all > v_end

        X_tr, y_tr   = X_all[mask_tr],  y_all[mask_tr]
        X_vl, y_vl   = X_all[mask_val], y_all[mask_val]
        X_te, y_te   = X_all[mask_te],  y_all[mask_te]
        idx_te       = idx_all[mask_te]

        # ── Seleção de features (sobre treino) ────────────────────────────
        selected = selecionar_features(X_tr, y_tr, feature_names,
                                       OUTPUT_SVR, tid, "SVR")
        sel_idx  = [list(feature_names).index(f) for f in selected]
        X_tr_s   = X_tr[:, sel_idx]
        X_vl_s   = X_vl[:, sel_idx]
        X_te_s   = X_te[:, sel_idx]

        # ── HPT sobre VALIDAÇÃO ───────────────────────────────────────────
        best_mae, best_params, best_pipe = np.inf, None, None

        for params in SVR_PARAM_GRID:
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("svr", SVR(kernel="rbf"))])
            pipe.set_params(**params)
            pipe.fit(X_tr_s, y_tr)
            pred_val = pipe.predict(X_vl_s)
            mae = np.mean(np.abs(y_vl - pred_val))
            log.info(f"     {params} → MAE_val={mae:.4f}")
            if mae < best_mae:
                best_mae, best_params, best_pipe = mae, params, pipe

        log.info(f"  ✓ HPT → {best_params} | MAE_val={best_mae:.4f}")

        # ── Retreinar em TREINO+VAL, avaliar TESTE ────────────────────────
        X_tv = np.vstack([X_tr_s, X_vl_s])
        y_tv = np.concatenate([y_tr, y_vl])

        pipe_final = Pipeline([("scaler", StandardScaler()),
                               ("svr", SVR(kernel="rbf"))])
        pipe_final.set_params(**best_params)
        pipe_final.fit(X_tv, y_tv)

        # Previsão de TESTE: recursiva — nunca usa valores reais futuros como lags
        # Correção do bug: substituído predict() direto por recursive_predict()
        y_hist_tv = y.loc[y.index <= v_end].values
        cal_test_full = adicionar_features_calendario(y.loc[idx_te])

        pred_test = recursive_predict(
            pipe_final, y_hist_tv, selected, cal_test_full, len(idx_te)
        )

        y_true_test = pd.Series(y_te, index=idx_te)
        y_pred_test = pd.Series(pred_test, index=idx_te)
        metrics     = compute_metrics(y_true_test, y_pred_test)
        log.info(f"  ✓ TESTE → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

        # Previsão de validação para plot
        pred_val_plot = pd.Series(best_pipe.predict(X_vl_s), index=idx_all[mask_val])
        plot_forecast_3way(y, pred_val_plot, y_pred_test, tid, "SVR", OUTPUT_SVR)
        plot_residuals(y_true_test, y_pred_test, tid, "SVR", OUTPUT_SVR)

        # ── Previsão futura (desabilitada: foco no gabarito 2024) ─────────
        if GENERATE_FUTURE_FORECAST:
            try:
                from config import FUTURE_END_DATE
                future_idx = pd.date_range(y.index[-1]+pd.Timedelta(days=1),
                                           pd.to_datetime(FUTURE_END_DATE), freq="D")
                cal_fut = adicionar_features_calendario(pd.Series(np.nan, index=future_idx))

                X_all_s  = X_all[:, sel_idx]
                pipe_fut = Pipeline([("scaler", StandardScaler()),
                                     ("svr", SVR(kernel="rbf"))])
                pipe_fut.set_params(**best_params)
                pipe_fut.fit(X_all_s, y_all)

                preds_fut = recursive_predict(
                    pipe_fut, y.values, selected, cal_fut, len(future_idx)
                )
                fut_series = pd.Series(preds_fut, index=future_idx)

                resids = y_all - pipe_fut.predict(X_all_s)
                rng    = np.random.default_rng(RANDOM_STATE)
                boots  = [preds_fut + rng.choice(resids, size=len(future_idx), replace=True)
                          for _ in range(200)]
                boots  = np.array(boots)
                plot_future_forecast(y, fut_series,
                                     pd.Series(np.percentile(boots, 2.5,  axis=0), index=future_idx),
                                     pd.Series(np.percentile(boots, 97.5, axis=0), index=future_idx),
                                     tid, "SVR", OUTPUT_FUTURE)
                pd.DataFrame({"data": future_idx, "prev_mean": preds_fut,
                              "ic_lower_95": np.percentile(boots,2.5,axis=0),
                              "ic_upper_95": np.percentile(boots,97.5,axis=0)
                             }).to_csv(f"{OUTPUT_FUTURE}/futuro_{tid}_SVR.csv", index=False)
            except Exception as e:
                log.warning(f"Previsão futura SVR falhou: {e}")

        resultados.append({
            "transformador": tid, "nome_plot": MAPA_PLOT.get(tid,tid),
            "modelo": "SVR", "abordagem": "SVR",
            "best_params": str(best_params), "MAE_val": round(best_mae,4),
            **{k: round(v,4) for k,v in metrics.items()},
        })

    return pd.DataFrame(resultados)


def main():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES)].copy()
    res = treinar_svr(df)
    res.to_csv(f"{OUTPUT_SVR}/metricas_SVR.csv", index=False)
    print(res[["transformador","modelo","RMSE","MAE","sMAPE","R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
