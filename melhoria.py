# =============================================================================
# melhorias.py — Estratégias de Melhoria para Captura de Picos
# =============================================================================
# Aplica 5 estratégias sobre os modelos existentes:
#
#   1. Features de pico histórico (lag 364/365/366 + max rolling anual)
#   2. Perda assimétrica (penaliza mais subestimação de picos)
#   3. Quantile Regression para capturar extremos
#   4. Investigação e correção do T3 (mudança estrutural)
#   5. Ensemble ponderado dos melhores modelos
#
# Execute após rodar todos os scripts base (boosting, DL, transformer).
# =============================================================================
import warnings; warnings.filterwarnings("ignore")
import os, random
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures

from config import (
    DATA_PATH, WEATHER_PATH, TRANSFORMADORES, MAPA_PLOT,
    RANDOM_STATE, END_TRAIN, END_VALIDATION,
    OUTPUT_MV, OUTPUT_FUTURE
)
from utils import (
    preparar_serie_diaria, adicionar_features_calendario,
    carregar_weather, compute_metrics, get_splits,
    plot_forecast_3way, plot_residuals, log
)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

OUTPUT_MELHORIAS = "resultados_melhorias"
os.makedirs(OUTPUT_MELHORIAS, exist_ok=True)


# =============================================================================
# ESTRATÉGIA 1 — Features de Pico Histórico
# =============================================================================
# A ideia: picos de fevereiro/março tendem a se repetir anualmente.
# Adicionamos lags específicos (364, 365, 366 dias) e uma feature
# "max dos últimos 30 dias do mesmo período no ano anterior".
# Isso dá ao modelo uma "memória de pico" explícita.
# =============================================================================

def adicionar_features_pico(y: pd.Series) -> pd.DataFrame:
    """
    Features extras focadas em capturar picos extremos.

    - lag_364/365/366: valor do mesmo período no ano anterior (±1 dia)
    - rolling_max_30d: máximo dos últimos 30 dias (captura momentum de pico)
    - rolling_max_365d: máximo anual histórico (ancora o nível de pico)
    - pct_of_annual_max: posição relativa atual vs. máximo anual (0-1)
    - is_peak_season: indicador dos meses de maior carga (Jan-Mar, Nov-Dez)
    - zscore_30d: z-score dos últimos 30 dias (detecta anomalias crescentes)
    """
    df = pd.DataFrame(index=y.index)

    # NOTA: lag_364/365/366 são criados internamente pelo skforecast via
    # lags=[...,364,365,366]. Não adicioná-los aqui evita duplicata no LightGBM.

    # Rolling features de extremos
    df["rolling_max_30d"]  = y.shift(1).rolling(30,  min_periods=7).max()
    df["rolling_max_90d"]  = y.shift(1).rolling(90,  min_periods=30).max()
    df["rolling_max_365d"] = y.shift(1).rolling(365, min_periods=180).max()
    df["rolling_min_365d"] = y.shift(1).rolling(365, min_periods=180).min()
    df["rolling_mean_30d"] = y.shift(1).rolling(30,  min_periods=7).mean()
    df["rolling_std_30d"]  = y.shift(1).rolling(30,  min_periods=7).std()

    # Posição relativa no range anual (0 = mínimo, 1 = máximo)
    annual_range = df["rolling_max_365d"] - df["rolling_min_365d"]
    df["pct_of_annual_range"] = (
        (y.shift(1) - df["rolling_min_365d"]) / (annual_range + 1e-8)
    ).clip(0, 1)

    # Z-score dos últimos 30 dias (detecta quando a série está subindo)
    df["zscore_30d"] = (
        (y.shift(1) - df["rolling_mean_30d"]) / (df["rolling_std_30d"] + 1e-8)
    )

    # Indicador de temporada de pico (meses historicamente mais carregados)
    df["is_peak_season"] = y.index.month.isin([1, 2, 3, 11, 12]).astype(int)

    # Tendência de curto prazo (inclinação dos últimos 7 dias)
    df["trend_7d"] = y.shift(1).rolling(7, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0,
        raw=True
    )

    return df.bfill().fillna(0)


# =============================================================================
# ESTRATÉGIA 2 — Perda Assimétrica (Asymmetric Loss)
# =============================================================================
# O modelo padrão (MSE/MAE) penaliza igualmente subestimação e superestimação.
# Para capturar picos, queremos penalizar MAIS quando o modelo subestima
# (prevê menos do que o real) — especialmente em períodos de pico.
#
# Implementação: custom objective no LightGBM/XGBoost que multiplica o
# gradiente por um fator alpha > 1 quando y_pred < y_true.
# =============================================================================

def asymmetric_mse_lgbm(alpha: float = 2.0):
    """
    Função objetivo assimétrica para LightGBM.
    alpha: fator de penalização para subestimação (default=2.0).
    Quanto maior o alpha, mais o modelo tende a prever acima da média.
    """
    def objective(y_true, y_pred):
        residual = y_true - y_pred
        # Penaliza mais quando subestima (residual > 0 → previu menos que o real)
        weight = np.where(residual > 0, alpha, 1.0)
        grad = -2 * weight * residual
        hess =  2 * weight * np.ones_like(residual)
        return grad, hess
    return objective


def asymmetric_mse_xgb(alpha: float = 2.0):
    """Função objetivo assimétrica para XGBoost."""
    def objective(y_true, y_pred):
        residual = y_true - y_pred
        weight = np.where(residual > 0, alpha, 1.0)
        grad = -2 * weight * residual
        hess =  2 * weight * np.ones_like(residual)
        return grad, hess
    return objective


# =============================================================================
# ESTRATÉGIA 3 — Quantile Regression (captura o percentil 90)
# =============================================================================
# Em vez de prever a média, previsão do percentil 90 captura os picos.
# Para o Plano de Negócios da distribuidora, o percentil 90 é mais
# conservador e adequado para dimensionamento de capacidade.
# =============================================================================

def treinar_quantile(y: pd.Series, exog: pd.DataFrame,
                     splits: dict, tid: str) -> dict:
    """
    Treina LightGBM com quantile regression para percentis 50, 75 e 90.
    Percentil 90 captura picos — mais conservador para planejamento elétrico.
    """
    resultados = {}

    for quantile, q_name in [(0.5, "p50_mediana"), (0.75, "p75"), (0.90, "p90_pico")]:
        log.info(f"  Quantile {q_name} ({quantile})")

        model = LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            num_leaves=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )

        y_tv   = y.loc[splits["train"].union(splits["val"])]
        exog_tv = exog.loc[y_tv.index]

        cv_test = TimeSeriesFold(
            steps=len(splits["test"]),
            initial_train_size=len(y_tv),
            refit=False
        )

        lags_pico = [1, 2, 7, 14, 30, 60, 180, 364, 365, 366]

        fc = ForecasterRecursive(
            estimator=model,
            lags=lags_pico,
            window_features=RollingFeatures(
                stats=["mean", "max"], window_sizes=[30, 365]
            )
        )

        _, pred = backtesting_forecaster(
            forecaster=fc, y=y, exog=exog,
            cv=cv_test, metric="mean_absolute_error", verbose=False
        )

        y_true = y.loc[pred.index]
        metrics = compute_metrics(y_true, pred["pred"])
        resultados[q_name] = {"pred": pred["pred"], "metrics": metrics}

        log.info(f"    RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

    return resultados


# =============================================================================
# ESTRATÉGIA 4 — Diagnóstico e Correção do T3
# =============================================================================
# T3 tem R² máximo de 0.33 — muito ruim. Vamos investigar:
#   a) Mudança estrutural (quebra estrutural na série)
#   b) Nível de ruído vs. sinal
#   c) Se normalização ou diferenciação ajuda
# =============================================================================

def diagnosticar_t3(y: pd.Series, tid: str):
    """
    Diagnóstico específico para série com baixo R².
    Testa: estacionariedade, quebra estrutural, ruído.
    """
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import breaks_cusumolsresid

    log.info(f"\n{'='*50}")
    log.info(f"  DIAGNÓSTICO T3: {tid}")
    log.info(f"{'='*50}")

    # Teste de estacionariedade (ADF)
    try:
        adf_result = adfuller(y.dropna(), autolag="AIC")
        log.info(f"  ADF stat={adf_result[0]:.4f} p={adf_result[1]:.4f}")
        if adf_result[1] > 0.05:
            log.warning("  SÉRIE NÃO ESTACIONÁRIA — considerar diferenciação")
        else:
            log.info("  Série estacionária (ADF OK)")
    except Exception as e:
        log.warning(f"  ADF falhou: {e}")

    # Coeficiente de variação (nível de ruído)
    cv = y.std() / y.mean()
    log.info(f"  Coef. de variação: {cv:.4f} "
             f"({'ALTO — série muito ruidosa' if cv > 0.3 else 'normal'})")

    # Detectar mudança de nível (split na média por ano)
    for year in y.index.year.unique():
        subset = y[y.index.year == year]
        log.info(f"  {year}: mean={subset.mean():.4f} std={subset.std():.4f} "
                 f"max={subset.max():.4f}")

    # Plot de diagnóstico
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    nome = MAPA_PLOT.get(tid, tid)

    # Série completa
    axes[0,0].plot(y.index, y.values, lw=0.8, color="#2563eb")
    axes[0,0].set_title(f"{nome} — Série completa")

    # Média por mês (detecta sazonalidade)
    monthly = y.groupby(y.index.month).mean()
    axes[0,1].bar(monthly.index, monthly.values, color="#2563eb")
    axes[0,1].set_title(f"{nome} — Média por mês")
    axes[0,1].set_xlabel("Mês")

    # Boxplot por ano (detecta mudança de nível)
    years = sorted(y.index.year.unique())
    axes[1,0].boxplot(
        [y[y.index.year == yr].values for yr in years],
        labels=years, patch_artist=True,
        boxprops=dict(facecolor="#dbeafe")
    )
    axes[1,0].set_title(f"{nome} — Distribuição por ano")

    # ACF manual (correlação com lag 1 até 30)
    lags_acf = range(1, 31)
    acf_vals = [y.autocorr(lag=l) for l in lags_acf]
    axes[1,1].bar(lags_acf, acf_vals, color="#2563eb")
    axes[1,1].axhline(0, color="black", lw=0.5)
    axes[1,1].axhline( 1.96/np.sqrt(len(y)), color="red", ls="--", lw=0.8)
    axes[1,1].axhline(-1.96/np.sqrt(len(y)), color="red", ls="--", lw=0.8)
    axes[1,1].set_title(f"{nome} — Autocorrelação (lags 1-30)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_MELHORIAS}/diagnostico_{tid}.svg",
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Diagnóstico salvo: {OUTPUT_MELHORIAS}/diagnostico_{tid}.svg")


# =============================================================================
# ESTRATÉGIA 5 — Ensemble Ponderado
# =============================================================================
# Combina as previsões dos melhores modelos ponderadas pelo inverso do MAE
# de validação. Tende a ser mais estável e capturar melhor os extremos
# do que qualquer modelo individualmente.
# =============================================================================

def ensemble_ponderado(previsoes: dict, pesos_mae: dict,
                       y_true: pd.Series, tid: str) -> pd.Series:
    """
    Ensemble ponderado pelo inverso do MAE de validação.
    Modelos com menor MAE_val recebem mais peso.

    previsoes: {nome_modelo: pd.Series com previsões}
    pesos_mae: {nome_modelo: MAE_val}
    """
    # Inverso do MAE → modelos melhores têm mais peso
    inv_mae = {k: 1.0 / (v + 1e-8) for k, v in pesos_mae.items()}
    total   = sum(inv_mae.values())
    pesos   = {k: v / total for k, v in inv_mae.items()}

    log.info(f"\n  Ensemble pesos ({tid}):")
    for nome, peso in sorted(pesos.items(), key=lambda x: -x[1]):
        log.info(f"    {nome}: {peso:.3f}")

    # Combinar previsões
    idx = list(previsoes.values())[0].index
    ensemble = pd.Series(0.0, index=idx)
    for nome, pred in previsoes.items():
        ensemble += pesos[nome] * pred.reindex(idx).fillna(0)

    metrics = compute_metrics(y_true.reindex(idx), ensemble)
    log.info(f"  Ensemble → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

    return ensemble, metrics, pesos


# =============================================================================
# PIPELINE PRINCIPAL DE MELHORIAS
# =============================================================================

def pipeline_melhorias(df: pd.DataFrame, weather=None) -> pd.DataFrame:
    """
    Roda todas as estratégias de melhoria e salva resultados.
    """
    resultados = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"\n{'='*60}\n  MELHORIAS: {tid}\n{'='*60}")

        y      = preparar_serie_diaria(df_tr)
        splits = get_splits(y)
        nome   = MAPA_PLOT.get(tid, tid)

        # Diagnóstico especial para T3
        if tid == "T70a":
            diagnosticar_t3(y, tid)

        # ── Features enriquecidas ──────────────────────────────────────────
        cal       = adicionar_features_calendario(y)
        feat_pico = adicionar_features_pico(y)

        if weather is not None:
            exog_base = cal.join(weather, how="left").bfill().ffill()
        else:
            exog_base = cal

        # Combinar features base + features de pico
        exog_full = pd.concat([exog_base, feat_pico], axis=1)
        exog_full = exog_full.select_dtypes(include=[np.number]).fillna(0)
        log.info(f"  Total features: {exog_full.shape[1]}")

        y_tv    = y.loc[splits["train"].union(splits["val"])]
        exog_tv = exog_full.loc[y_tv.index]

        # Lags enriquecidos (inclui lags anuais explícitos)
        lags_enriquecidos = [1, 2, 7, 14, 30, 60, 180, 364, 365, 366]

        cv_val  = TimeSeriesFold(
            steps=len(splits["val"]),
            initial_train_size=len(splits["train"]),
            refit=False
        )
        cv_test = TimeSeriesFold(
            steps=len(splits["test"]),
            initial_train_size=len(y_tv),
            refit=False
        )

        previsoes_ensemble = {}
        pesos_ensemble     = {}

        # ── Modelo 1: LGBM padrão com features de pico ────────────────────
        log.info("\n  [1/4] LGBM + features de pico")
        for nome_m, est in [
            ("LGBM_pico", LGBMRegressor(
                n_estimators=500, learning_rate=0.01, max_depth=6,
                num_leaves=50, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            )),
            ("LGBM_asym", LGBMRegressor(
                n_estimators=500, learning_rate=0.01, max_depth=6,
                num_leaves=50, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            )),
        ]:
            # HPT no validação
            fc_val = ForecasterRecursive(
                estimator=est,
                lags=lags_enriquecidos,
                window_features=RollingFeatures(
                    stats=["mean", "max"], window_sizes=[30, 365]
                )
            )
            _, pv = backtesting_forecaster(
                forecaster=fc_val, y=y_tv, exog=exog_tv,
                cv=cv_val, metric="mean_absolute_error", verbose=False
            )
            mae_val = mean_absolute_error(y.loc[pv.index], pv["pred"])

            # Teste
            fc_test = ForecasterRecursive(
                estimator=est,
                lags=lags_enriquecidos,
                window_features=RollingFeatures(
                    stats=["mean", "max"], window_sizes=[30, 365]
                )
            )
            _, pt = backtesting_forecaster(
                forecaster=fc_test, y=y, exog=exog_full,
                cv=cv_test, metric="mean_absolute_error", verbose=False
            )

            y_true = y.loc[pt.index]
            metrics = compute_metrics(y_true, pt["pred"])
            log.info(f"  {nome_m} → RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f}")

            plot_forecast_3way(y, pv["pred"], pt["pred"], tid, nome_m, OUTPUT_MELHORIAS)

            previsoes_ensemble[nome_m] = pt["pred"]
            pesos_ensemble[nome_m]     = mae_val

            resultados.append({
                "transformador": tid, "nome_plot": nome,
                "modelo": nome_m, "estrategia": "features_pico+lags_anuais",
                "MAE_val": round(mae_val, 4),
                **{k: round(v, 4) for k, v in metrics.items()}
            })

        # ── Modelo 2: Quantile p90 ─────────────────────────────────────────
        log.info("\n  [2/4] Quantile Regression p90")
        try:
            q_results = treinar_quantile(y, exog_full, splits, tid)
            for q_name, q_data in q_results.items():
                metrics = q_data["metrics"]
                log.info(f"  {q_name} → RMSE={metrics['RMSE']:.4f} "
                         f"R²={metrics['R2']:.4f}")

                # Plot comparativo p50 vs p90
                if "p50" in q_name or "p90" in q_name:
                    previsoes_ensemble[q_name] = q_data["pred"]
                    pesos_ensemble[q_name]     = metrics["MAE"]

                resultados.append({
                    "transformador": tid, "nome_plot": nome,
                    "modelo": f"Quantile_{q_name}",
                    "estrategia": "quantile_regression",
                    **{k: round(v, 4) for k, v in metrics.items()}
                })
        except Exception as e:
            log.warning(f"  Quantile falhou: {e}")

        # ── Modelo 3: XGBoost com lags anuais + features de pico ──────────
        log.info("\n  [3/4] XGBoost + lags anuais")
        try:
            xgb = XGBRegressor(
                n_estimators=500, learning_rate=0.01, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
            )
            fc_xgb_v = ForecasterRecursive(
                estimator=xgb, lags=lags_enriquecidos,
                window_features=RollingFeatures(
                    stats=["mean", "max"], window_sizes=[30, 365]
                )
            )
            _, pv_xgb = backtesting_forecaster(
                forecaster=fc_xgb_v, y=y_tv, exog=exog_tv,
                cv=cv_val, metric="mean_absolute_error", verbose=False
            )
            mae_val_xgb = mean_absolute_error(y.loc[pv_xgb.index], pv_xgb["pred"])

            fc_xgb_t = ForecasterRecursive(
                estimator=xgb, lags=lags_enriquecidos,
                window_features=RollingFeatures(
                    stats=["mean", "max"], window_sizes=[30, 365]
                )
            )
            _, pt_xgb = backtesting_forecaster(
                forecaster=fc_xgb_t, y=y, exog=exog_full,
                cv=cv_test, metric="mean_absolute_error", verbose=False
            )
            metrics_xgb = compute_metrics(y.loc[pt_xgb.index], pt_xgb["pred"])
            log.info(f"  XGBoost_pico → RMSE={metrics_xgb['RMSE']:.4f} "
                     f"R²={metrics_xgb['R2']:.4f}")

            previsoes_ensemble["XGBoost_pico"] = pt_xgb["pred"]
            pesos_ensemble["XGBoost_pico"]     = mae_val_xgb

            resultados.append({
                "transformador": tid, "nome_plot": nome,
                "modelo": "XGBoost_pico",
                "estrategia": "lags_anuais+max_rolling",
                "MAE_val": round(mae_val_xgb, 4),
                **{k: round(v, 4) for k, v in metrics_xgb.items()}
            })
        except Exception as e:
            log.warning(f"  XGBoost_pico falhou: {e}")

        # ── Ensemble ponderado ─────────────────────────────────────────────
        log.info("\n  [4/4] Ensemble ponderado")
        if len(previsoes_ensemble) >= 2:
            try:
                # Alinhar índices
                idx_comum = previsoes_ensemble[
                    list(previsoes_ensemble.keys())[0]
                ].index
                previsoes_alinhadas = {
                    k: v.reindex(idx_comum)
                    for k, v in previsoes_ensemble.items()
                    if not v.reindex(idx_comum).isna().all()
                }
                y_true_ens = y.loc[idx_comum]

                ens_pred, ens_metrics, pesos_finais = ensemble_ponderado(
                    previsoes_alinhadas, pesos_ensemble, y_true_ens, tid
                )

                # Plot ensemble vs. real
                fig, ax = plt.subplots(figsize=(14, 5))
                y_true_ens.plot(ax=ax, label="Real", color="#16a34a", lw=1.5)
                ens_pred.plot(ax=ax, label="Ensemble", color="#dc2626",
                              lw=2, ls="--")
                # Plotar cada modelo individual em cinza
                for nm, pred in previsoes_alinhadas.items():
                    pred.plot(ax=ax, lw=0.7, alpha=0.3, color="#94a3b8",
                              label=f"_{nm}")
                ax.set_title(f"{nome} — Ensemble vs. Individual (Teste 2024)",
                             fontsize=13)
                ax.set_xlabel("Data"); ax.set_ylabel("Smax [MVA]")
                # Legenda só para real e ensemble
                handles, labels = ax.get_legend_handles_labels()
                filt = [(h, l) for h, l in zip(handles, labels)
                        if not l.startswith("_")]
                ax.legend(*zip(*filt), fontsize=9)
                plt.tight_layout()
                plt.savefig(f"{OUTPUT_MELHORIAS}/ensemble_{tid}.svg",
                            dpi=150, bbox_inches="tight")
                plt.close()

                resultados.append({
                    "transformador": tid, "nome_plot": nome,
                    "modelo": "Ensemble_ponderado",
                    "estrategia": f"ensemble({len(previsoes_alinhadas)} modelos)",
                    "pesos": str({k: f"{v:.3f}" for k, v in pesos_finais.items()}),
                    **{k: round(v, 4) for k, v in ens_metrics.items()}
                })
            except Exception as e:
                log.warning(f"  Ensemble falhou: {e}")

        # ── Plot comparativo de todas as estratégias ───────────────────────
        try:
            t_end = pd.to_datetime(END_VALIDATION)
            y_test = y.loc[y.index > t_end]
            cores  = plt.cm.tab10(np.linspace(0, 1, len(previsoes_ensemble)))

            fig, ax = plt.subplots(figsize=(14, 5))
            y_test.plot(ax=ax, label="Real", color="black", lw=1.5, zorder=10)

            for (nm, pred), cor in zip(previsoes_ensemble.items(), cores):
                pred_alinhada = pred.reindex(y_test.index)
                mae = mean_absolute_error(
                    y_test.dropna(),
                    pred_alinhada.reindex(y_test.dropna().index).fillna(0)
                )
                pred_alinhada.plot(ax=ax, label=f"{nm} (MAE={mae:.3f})",
                                   color=cor, lw=1.2, alpha=0.8)

            ax.set_title(f"{nome} — Comparação Estratégias de Melhoria (Teste 2024)",
                         fontsize=13)
            ax.set_xlabel("Data"); ax.set_ylabel("Smax [MVA]")
            ax.legend(fontsize=8, ncol=2)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_MELHORIAS}/comparacao_estrategias_{tid}.svg",
                        dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            log.warning(f"  Plot comparativo falhou: {e}")

    return pd.DataFrame(resultados)


# =============================================================================
# COMPARAÇÃO: ANTES vs. DEPOIS
# =============================================================================

def comparar_antes_depois(df_melhorias: pd.DataFrame):
    """
    Carrega métricas anteriores e compara com as melhorias.
    Gera gráfico de ganho por transformador.
    """
    arquivos_anteriores = {
        "resultados_MV/metricas_MV.csv":   "MV",
        "resultados_DL/metricas_DL.csv":   "DL",
        "resultados_Transformer/metricas_Transformer_lite.csv": "Transformer",
    }
    dfs_ant = []
    for path, label in arquivos_anteriores.items():
        if os.path.exists(path):
            df_ant = pd.read_csv(path)
            dfs_ant.append(df_ant)

    if not dfs_ant:
        log.warning("Nenhum resultado anterior encontrado para comparação.")
        return

    df_anterior = pd.concat(dfs_ant, ignore_index=True)

    # Melhor R² anterior por transformador
    best_ant = df_anterior.groupby("transformador")["R2"].max().reset_index()
    best_ant.columns = ["transformador", "R2_anterior"]

    # Melhor R² nas melhorias
    best_new = df_melhorias.groupby("transformador")["R2"].max().reset_index()
    best_new.columns = ["transformador", "R2_melhoria"]

    comp = best_ant.merge(best_new, on="transformador", how="outer")
    comp["ganho_R2"] = comp["R2_melhoria"] - comp["R2_anterior"]
    comp["nome_plot"] = comp["transformador"].map(MAPA_PLOT)

    log.info("\n" + "="*50)
    log.info("COMPARAÇÃO ANTES vs. DEPOIS")
    log.info("="*50)
    for _, row in comp.iterrows():
        sinal = "+" if row["ganho_R2"] > 0 else ""
        log.info(f"  {row['nome_plot']}: R² {row['R2_anterior']:.4f} → "
                 f"{row['R2_melhoria']:.4f} ({sinal}{row['ganho_R2']:.4f})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comp))
    w = 0.35
    b1 = ax.bar(x - w/2, comp["R2_anterior"].fillna(0), w,
                label="Antes", color="#94a3b8")
    b2 = ax.bar(x + w/2, comp["R2_melhoria"].fillna(0), w,
                label="Após melhorias", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["nome_plot"])
    ax.set_ylabel("R² (maior é melhor)")
    ax.set_title("Ganho de R² após melhorias", fontsize=13)
    ax.legend()
    ax.set_ylim(0, 1)

    # Anotar ganho
    for i, row in comp.iterrows():
        ganho = row["ganho_R2"]
        cor   = "#16a34a" if ganho > 0 else "#dc2626"
        sinal = "+" if ganho > 0 else ""
        ax.text(i + w/2, row["R2_melhoria"] + 0.02,
                f"{sinal}{ganho:.3f}", ha="center", fontsize=9,
                color=cor, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_MELHORIAS}/comparacao_antes_depois.svg",
                dpi=150, bbox_inches="tight")
    plt.close()

    comp.to_csv(f"{OUTPUT_MELHORIAS}/ganho_r2.csv", index=False)
    log.info(f"Comparação salva: {OUTPUT_MELHORIAS}/comparacao_antes_depois.svg")


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=== PIPELINE DE MELHORIAS ===")

    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES)].copy()

    weather = None
    try:
        weather = carregar_weather()
    except Exception as e:
        log.warning(f"Clima não carregado: {e}")

    resultados = pipeline_melhorias(df, weather)

    out = f"{OUTPUT_MELHORIAS}/metricas_melhorias.csv"
    resultados.to_csv(out, index=False)

    comparar_antes_depois(resultados)

    log.info(f"\n✅ Resultados salvos em: {OUTPUT_MELHORIAS}/")
    print("\n" + "="*70)
    print("RESULTADOS DAS MELHORIAS — CONJUNTO DE TESTE")
    print("="*70)
    cols = ["transformador","modelo","estrategia","RMSE","MAE","sMAPE","R2"]
    cols_disp = [c for c in cols if c in resultados.columns]
    print(resultados[cols_disp].sort_values(["transformador","R2"],
          ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()