# =============================================================================
# models_boosting.py — XGBoost · LightGBM · GradientBoosting
# Univariado (UV) e Multivariado (MV) com HPT sem data leakage
# =============================================================================
# Metodologia (sem data leakage):
#
#   TREINO (2021-2022) → (1) seleção de features exógenas
#                      → (2) ajuste dos parâmetros do modelo
#   VALIDAÇÃO (2023)   → (3) HPT: seleciona melhores hiperparâmetros
#   TESTE (2024)       → (4) avaliação final única, reportada no artigo
#
# Seleção de features (modo MV):
#   - Executada ANTES do HPT, usando APENAS dados de treino.
#   - Método: importância acumulada de um LightGBM rápido (proxy).
#     Features são ranqueadas por importância média (gain) e as que
#     contribuem com os primeiros FEATURE_IMPORTANCE_THRESHOLD % da
#     importância total acumulada são retidas.
#   - Garante que nenhuma informação de validação ou teste contamina
#     a decisão de quais features incluir.
#   - As features selecionadas são fixas para todos os modelos do mesmo
#     transformador (consistência metodológica).
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
    plot_forecast_3way, plot_residuals, plot_future_forecast, log
)

random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

EXOG_COLS = [
    "weekday_sin","weekday_cos","month_sin","month_cos",
    "dayofyear_sin","dayofyear_cos","is_weekend","is_holiday",
    "temp_max_anos","temp_min_anos","temp_media_anos","precipitacao_anos"
]

# Limiar de importância acumulada para seleção de features (modo MV).
# Features são adicionadas em ordem decrescente de importância até que
# a soma acumulada atinja este percentual da importância total.
# Valor 0.95 → retém as features que juntas explicam 95% da importância.
FEATURE_IMPORTANCE_THRESHOLD = 0.95


# ─────────────────────────────────────────────────────────────────────────────
# SELEÇÃO DE FEATURES (apenas modo MV, apenas sobre treino)
# ─────────────────────────────────────────────────────────────────────────────
def selecionar_features_mv(y_train: pd.Series,
                           exog_train: pd.DataFrame,
                           tid: str,
                           output_dir: str) -> list:
    """
    Seleciona as features exógenas mais relevantes usando um modelo LightGBM
    rápido (proxy) treinado EXCLUSIVAMENTE nos dados de treino.

    Procedimento (sem data leakage):
      1. Treina um ForecasterRecursive com LightGBM leve sobre y_train/exog_train.
      2. Extrai importância de cada feature pelo critério 'gain' (contribuição
         média de cada split para redução do erro).
      3. Ordena features por importância decrescente e acumula até atingir
         FEATURE_IMPORTANCE_THRESHOLD da importância total.
      4. Retorna a lista de features selecionadas e salva gráfico para o artigo.

    Garantias metodológicas:
      - Nenhum dado de validação ou teste é acessado nesta etapa.
      - As features selecionadas são determinadas antes do HPT.
      - O mesmo conjunto de features é usado em todos os modelos (LGBM,
        XGBoost, GradientBoosting) do mesmo transformador, garantindo
        comparabilidade fair entre algoritmos.

    Parâmetros
    ----------
    y_train     : série temporal de treino (2021-2022)
    exog_train  : DataFrame de features exógenas, alinhado com y_train
    tid         : identificador do transformador (para log e plot)
    output_dir  : diretório de saída para salvar o gráfico de importância

    Retorna
    -------
    selected : lista de nomes das features selecionadas
    """
    nome_plot = MAPA_PLOT.get(tid, tid)
    log.info(f"  [FS] Seleção de features para {tid} sobre treino "
             f"({len(y_train)} dias, {len(exog_train.columns)} candidatas)...")

    # Modelo proxy leve — serve apenas para rankear features, não para prever
    proxy = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        importance_type="gain",   # gain é mais discriminativo que split count
    )
    fc_proxy = ForecasterRecursive(
        estimator=proxy,
        lags=LAGS,
        window_features=RollingFeatures(stats=["mean"], window_sizes=[ROLLING_WINDOW])
    )
    fc_proxy.fit(y=y_train, exog=exog_train)

    # Importâncias das features EXÓGENAS (excluindo lags e rolling — sempre incluídos)
    imp_df = fc_proxy.get_feature_importances()

    # Filtrar apenas as features exógenas (não lags nem rolling mean)
    exog_imp = imp_df[imp_df["feature"].isin(exog_train.columns)].copy()
    exog_imp = exog_imp.sort_values("importance", ascending=False).reset_index(drop=True)

    if exog_imp.empty:
        log.warning(f"  [FS] Nenhuma feature exógena encontrada na importância — usando todas.")
        return list(exog_train.columns)

    # Seleção por importância acumulada
    total_imp = exog_imp["importance"].sum()
    if total_imp == 0:
        log.warning(f"  [FS] Importância total zero — usando todas as features.")
        return list(exog_train.columns)

    exog_imp["importance_pct"]  = exog_imp["importance"] / total_imp
    exog_imp["importance_cum"]  = exog_imp["importance_pct"].cumsum()

    # Retém features até atingir o limiar acumulado
    selected_mask = exog_imp["importance_cum"].shift(1, fill_value=0) < FEATURE_IMPORTANCE_THRESHOLD
    selected = exog_imp.loc[selected_mask, "feature"].tolist()

    # Garantia: pelo menos 1 feature sempre selecionada
    if len(selected) == 0:
        selected = [exog_imp.iloc[0]["feature"]]

    n_total = len(exog_imp)
    n_sel   = len(selected)
    log.info(f"  [FS] {tid}: {n_sel}/{n_total} features selecionadas "
             f"(limiar acumulado={FEATURE_IMPORTANCE_THRESHOLD:.0%}): {selected}")

    # ── Gráfico de importância com limiar — salvo para o artigo ───────────
    fig, ax = plt.subplots(figsize=(8, max(4, n_total * 0.45 + 1)))
    colors = ["#2563eb" if f in selected else "#94a3b8" for f in exog_imp["feature"]]
    ax.barh(exog_imp["feature"][::-1], exog_imp["importance"][::-1], color=colors[::-1])

    # Linha de corte (importância acumulada)
    if n_sel < n_total:
        cutoff_val = exog_imp.iloc[n_sel - 1]["importance"]
        ax.axvline(cutoff_val, color="#dc2626", ls="--", lw=1.2,
                   label=f"Threshold ({FEATURE_IMPORTANCE_THRESHOLD:.0%} accumulated)")
        ax.legend(fontsize=8)

    ax.set_title(f"{nome_plot} — Feature Selection (proxy LightGBM, training set)\n"
                 f"Blue = selected ({n_sel}/{n_total}), Grey = discarded",
                 fontsize=10)
    ax.set_xlabel("Importance (gain)")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/feature_selection_{tid}_MV.svg",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Salvar CSV das importâncias (tabela suplementar para o artigo)
    exog_imp["selecionada"] = exog_imp["feature"].isin(selected)
    exog_imp.to_csv(f"{output_dir}/feature_importance_{tid}_proxy.csv", index=False)

    return selected


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
    """
    future_idx = pd.date_range(
        y_full.index[-1] + pd.Timedelta(days=1),
        pd.to_datetime(FUTURE_END_DATE), freq="D"
    )
    if len(future_idx) == 0:
        return

    forecaster.fit(y=y_full, exog=exog_full if exog_full is not None else None)

    if exog_full is not None:
        cal_future = adicionar_features_calendario(pd.Series(np.nan, index=future_idx))
        weather_mean = exog_full[[c for c in exog_full.columns
                                  if c not in cal_future.columns]].mean()
        for col, val in weather_mean.items():
            cal_future[col] = val
        # Usar apenas as colunas que o modelo conhece (features selecionadas)
        exog_future = cal_future[[c for c in exog_full.columns if c in cal_future.columns]]
        pred_future = forecaster.predict(steps=len(future_idx), exog=exog_future)
    else:
        pred_future = forecaster.predict(steps=len(future_idx))

    pred_series = pd.Series(pred_future.values, index=future_idx)

    resids = forecaster.in_sample_residuals_["_unknown_level"]
    resids = resids[~np.isnan(resids)]

    rng  = np.random.default_rng(RANDOM_STATE)
    boots = np.array([
        pred_series.values + rng.choice(resids, size=len(future_idx), replace=True)
        for _ in range(n_bootstrap)
    ])
    lower_95 = np.percentile(boots, 2.5,  axis=0)
    upper_95 = np.percentile(boots, 97.5, axis=0)

    plot_future_forecast(y_full, pred_series,
                         pd.Series(lower_95, index=future_idx),
                         pd.Series(upper_95, index=future_idx),
                         tid, modelo, output_dir)

    pd.DataFrame({
        "data":        future_idx,
        "prev_mean":   pred_series.values,
        "ic_lower_95": lower_95,
        "ic_upper_95": upper_95,
    }).to_csv(f"{output_dir}/futuro_{tid}_{modelo}.csv", index=False)
    log.info(f"  Previsão futura salva: {output_dir}/futuro_{tid}_{modelo}.csv")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def treinar_boosting(df: pd.DataFrame, weather=None, mode="UV") -> pd.DataFrame:
    """
    mode="UV" → univariado (sem exógenas, sem seleção de features)
    mode="MV" → multivariado (+ calendário + clima, COM seleção de features)

    Fluxo MV (sem data leakage):
      TREINO → seleção de features (proxy LightGBM, gain acumulado ≥ 95%)
      TREINO → HPT sobre VALIDAÇÃO com features selecionadas
      TREINO+VAL → retreino com melhores hiperparâmetros
      TESTE → avaliação final única
    """
    output_dir = OUTPUT_UV if mode == "UV" else OUTPUT_MV
    param_grid = get_param_grid()
    resultados = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  {tid}  [{mode}]\n{'='*60}")
        y      = preparar_serie_diaria(df_tr)
        splits = get_splits(y)

        # ── Exógenas (apenas MV) ──────────────────────────────────────────
        exog = None
        selected_features = None

        if mode == "MV":
            cal = adicionar_features_calendario(y)
            if weather is not None:
                exog_df = cal.join(weather, how="left").bfill().ffill()
            else:
                exog_df = cal
            cols_disp = [c for c in EXOG_COLS if c in exog_df.columns]
            exog_full_df = exog_df[cols_disp]

            # ── SELEÇÃO DE FEATURES — somente sobre conjunto de TREINO ────
            # Executada antes do HPT; validação e teste não são acessados.
            exog_train_only = exog_full_df.loc[splits["train"]]
            y_train_only    = y.loc[splits["train"]]

            selected_features = selecionar_features_mv(
                y_train_only, exog_train_only, tid, output_dir
            )

            # Aplicar seleção a todo o DataFrame de exógenas
            exog = exog_full_df[selected_features]
            log.info(f"  Features exógenas usadas em HPT/treino/teste: {selected_features}")

        # ── Conjuntos temporais ───────────────────────────────────────────
        y_tv    = y.loc[splits["train"].union(splits["val"])]
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

            y_true  = y.loc[pt.index]
            metrics = compute_metrics(y_true, pt["pred"])
            log.info(f"  ✓ TESTE → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

            # ── Fit explícito em TREINO+VAL para expor importâncias ───────
            # backtesting_forecaster usa fc_final internamente mas não deixa
            # o forecaster fitado ao retornar — precisamos do fit explícito
            # para chamar get_feature_importances() a seguir.
            # Este fit usa exatamente os mesmos dados do cv_test (y_tv / exog_tv),
            # portanto não introduz nenhum acesso adicional ao conjunto de teste.
            kw_fit = dict(y=y_tv)
            if exog_tv is not None:
                kw_fit["exog"] = exog_tv
            fc_final.fit(**kw_fit)

            # ── Plots ──────────────────────────────────────────────────────
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

            # Feature importance final (modelo fitado em treino+val)
            try:
                imp = fc_final.get_feature_importances()

                # Separar features exógenas das autorregressivas para clareza
                exog_feats = [c for c in imp["feature"] if c in (exog.columns if exog is not None else [])]
                lag_feats  = [c for c in imp["feature"] if c not in exog_feats]

                imp_sorted = imp.sort_values("importance", ascending=True)

                # Colorir: exógenas em azul, lags/rolling em cinza
                colors = [
                    "#2563eb" if f in exog_feats else "#94a3b8"
                    for f in imp_sorted["feature"]
                ]

                n_show = min(20, len(imp_sorted))
                imp_top = imp_sorted.tail(n_show)
                colors_top = colors[-n_show:]

                fig, ax = plt.subplots(figsize=(7, max(4, n_show * 0.35 + 1)))
                ax.barh(imp_top["feature"], imp_top["importance"], color=colors_top)
                ax.set_title(
                    f"{MAPA_PLOT.get(tid,tid)} — {nome} [{mode}]: Feature Importance\n"
                    f"Blue = selected exogenous | Grey = lag / rolling",
                    fontsize=9
                )
                ax.set_xlabel("Importance (gain)")
                plt.tight_layout()
                plt.savefig(f"{output_dir}/importance_{tid}_{nome}_{mode}.svg",
                            dpi=150, bbox_inches="tight")
                plt.close()
                log.info(f"  Importância salva: importance_{tid}_{nome}_{mode}.svg")
            except Exception as e:
                log.warning(f"  Importância não disponível: {e}")

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
                log.warning(f"  Previsão futura falhou ({nome}): {e}")

            resultados.append({
                "transformador":     tid,
                "nome_plot":         MAPA_PLOT.get(tid, tid),
                "modelo":            nome,
                "abordagem":         mode,
                "features_sel":      str(selected_features) if selected_features else "N/A",
                "n_features_sel":    len(selected_features) if selected_features else 0,
                "best_params":       str(best_params),
                "MAE_val":           round(best_mae, 4),
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
          [["transformador","modelo","abordagem","n_features_sel","RMSE","MAE","sMAPE","R2"]]
          .sort_values(["transformador","RMSE"]).to_string(index=False))


if __name__ == "__main__":
    main()
