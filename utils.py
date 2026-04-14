# =============================================================================
# utils.py — Funções Utilitárias Compartilhadas
# =============================================================================
import warnings, logging, os
import numpy as np
import pandas as pd
import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from config import (
    START_DATE, END_TRAIN, END_VALIDATION,
    MAX_GAP_INTERP, RANDOM_STATE, PERIOD, MAPA_PLOT
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    """
    RMSE, MAE, MAPE (protegido), sMAPE (simétrico, recomendado em artigos) e R².
    Referência: Makridakis (1993), IJF 9(4).
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return dict(RMSE=np.nan, MAE=np.nan, MAPE=np.nan, sMAPE=np.nan, R2=np.nan)
    eps   = 1e-8
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    mape  = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)
    smape = float(np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)
    ) * 100)
    return dict(RMSE=rmse, MAE=mae, MAPE=mape, sMAPE=smape, R2=r2)

# ─────────────────────────────────────────────────────────────────────────────
# PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────────────────────────────────────
def preparar_serie_diaria(df_tr: pd.DataFrame, log_gaps: bool = True) -> pd.Series:
    """
    Constrói série diária contínua com interpolação limitada a MAX_GAP_INTERP dias.
    Gaps maiores geram aviso de log para rastreabilidade metodológica.
    """
    s = df_tr.copy()
    s["Time"] = pd.to_datetime(s["datahora"])
    s = s.set_index("Time").sort_index()
    full_idx = pd.date_range(s.index.min().normalize(),
                             s.index.max().normalize(), freq="D")
    s = s.reindex(full_idx).loc[pd.to_datetime(START_DATE):]

    if log_gaps:
        missing = s["Smax"].isna()
        if missing.any():
            grp = (missing != missing.shift()).cumsum()
            lens = missing[missing].groupby(grp[missing]).count()
            large = lens[lens > MAX_GAP_INTERP]
            if not large.empty:
                log.warning(f"Gaps > {MAX_GAP_INTERP} dias: {large.to_dict()}")

    s["Smax"] = s["Smax"].interpolate(method="time", limit=MAX_GAP_INTERP).ffill(limit=2)
    return s["Smax"]


def adicionar_features_calendario(serie: pd.Series) -> pd.DataFrame:
    """Features de calendário com codificação cíclica sin/cos."""
    idx = serie.index
    br  = holidays.Brazil()
    df  = pd.DataFrame(index=idx)
    df["weekday"]        = idx.weekday
    df["month"]          = idx.month
    df["dayofyear"]      = idx.dayofyear
    df["year"]           = idx.year
    df["weekday_sin"]    = np.sin(2*np.pi*df["weekday"]   / 7)
    df["weekday_cos"]    = np.cos(2*np.pi*df["weekday"]   / 7)
    df["month_sin"]      = np.sin(2*np.pi*df["month"]     / 12)
    df["month_cos"]      = np.cos(2*np.pi*df["month"]     / 12)
    df["dayofyear_sin"]  = np.sin(2*np.pi*df["dayofyear"] / 365)
    df["dayofyear_cos"]  = np.cos(2*np.pi*df["dayofyear"] / 365)
    df["is_weekend"]     = (idx.weekday >= 5).astype(int)
    df["is_holiday"]     = idx.isin(br).astype(int)
    return df


def carregar_weather() -> pd.DataFrame:
    from config import WEATHER_PATH
    w = pd.read_csv(WEATHER_PATH, sep=";", encoding="latin-1")
    w["DATA"] = pd.to_datetime(w["DATA"])
    w = w.set_index("DATA").sort_index()
    cols = ["temp_max_anos","temp_min_anos","temp_media_anos","precipitacao_anos"]
    return w[[c for c in cols if c in w.columns]].copy()


# ─────────────────────────────────────────────────────────────────────────────
# SPLITS SEM DATA LEAKAGE
# ─────────────────────────────────────────────────────────────────────────────
def get_splits(y: pd.Series) -> dict:
    """
    Treino / Validação / Teste — janelas temporais não sobrepostas.
    HPT usa APENAS validação. Teste acessado UMA vez no final.
    """
    t = pd.to_datetime(END_TRAIN)
    v = pd.to_datetime(END_VALIDATION)
    tr  = y.index[y.index <= t]
    val = y.index[(y.index > t) & (y.index <= v)]
    te  = y.index[y.index > v]
    log.info(f"Split → Train:{len(tr)} | Val:{len(val)} | Test:{len(te)} dias")
    return dict(train=tr, val=val, test=te)


def build_lag_matrix(y: pd.Series, lags: list) -> pd.DataFrame:
    """
    Constrói matriz de features de lag para modelos tabulares (SVR, XGBoost).
    Garante que lag_k usa y[t-k] para prever y[t] — sem ver o futuro.
    """
    df = pd.DataFrame({"y": y})
    for lag in lags:
        df[f"lag_{lag}"] = y.shift(lag)
    return df.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# SELEÇÃO DE FEATURES (para modelos tabulares)
# ─────────────────────────────────────────────────────────────────────────────
def selecionar_features(X_train, y_train, feature_names, output_dir, tid, modelo):
    """
    Seleção de features via importância de Random Forest + correlação de Spearman.
    Retorna lista das features selecionadas e salva gráfico.
    """
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import spearmanr

    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

    # Correlação de Spearman
    corr = {}
    for col in feature_names:
        idx_c = list(feature_names).index(col)
        c, _ = spearmanr(X_train[:, idx_c], y_train)
        corr[col] = abs(c)
    corr_s = pd.Series(corr).sort_values(ascending=False)

    # Selecionar features: importância > média OU correlação > 0.1
    threshold_imp  = imp.mean()
    selected = imp[imp >= threshold_imp].index.tolist()
    selected += [f for f in corr_s[corr_s > 0.1].index if f not in selected]
    selected = list(dict.fromkeys(selected))   # preservar ordem, sem duplicatas

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(feature_names)*0.3+2)))
    imp.head(20).sort_values().plot.barh(ax=axes[0], color="#2563eb")
    axes[0].axvline(threshold_imp, color="red", ls="--", lw=1, label="threshold")
    axes[0].set_title(f"{MAPA_PLOT.get(tid,tid)} — {modelo}: RF Importance")
    axes[0].legend()

    corr_s.head(20).sort_values().plot.barh(ax=axes[1], color="#16a34a")
    axes[1].set_title(f"{MAPA_PLOT.get(tid,tid)} — {modelo}: Spearman |corr|")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/feature_selection_{tid}_{modelo}.svg",
                dpi=150, bbox_inches="tight")
    plt.close()

    log.info(f"  {tid}/{modelo}: {len(selected)}/{len(feature_names)} features selecionadas")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────────────────
def plot_forecast_3way(y, pred_val, pred_test, tid, modelo, output_dir, suffix=""):
    """Gráfico padrão artigo: treino / validação / teste + previsões."""
    nome  = MAPA_PLOT.get(tid, tid)
    t_end = pd.to_datetime(END_TRAIN)
    v_end = pd.to_datetime(END_VALIDATION)

    fig, ax = plt.subplots(figsize=(14, 5))
    y.loc[:t_end].plot(ax=ax, label="Training",       color="#2563eb", lw=1.0)
    y.loc[t_end:v_end].plot(ax=ax, label="Validation", color="#f59e0b", lw=1.0)
    y.loc[v_end:].plot(ax=ax, label="Test (real)",    color="#16a34a", lw=1.5)
    if pred_val  is not None and len(pred_val)  > 0:
        pred_val.plot(ax=ax,  label="Val. forecast", color="#f59e0b", lw=1.5, ls="--")
    if pred_test is not None and len(pred_test) > 0:
        pred_test.plot(ax=ax, label="Test forecast",     color="#dc2626", lw=1.5, ls="--")
    ax.axvline(t_end, color="gray", ls=":", lw=1)
    ax.axvline(v_end, color="gray", ls=":", lw=1)
    ax.set_title(f"{nome} — {modelo}{' '+suffix if suffix else ''}", fontsize=13)
    ax.set_xlabel("Date"); ax.set_ylabel("Smax [MVA]")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = f"{output_dir}/forecast_{tid}_{modelo}{suffix}.svg"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_residuals(y_true, y_pred, tid, modelo, output_dir):
    """Resíduos no tempo + histograma."""
    nome = MAPA_PLOT.get(tid, tid)
    res  = np.asarray(y_true) - np.asarray(y_pred)
    idx  = getattr(y_true, "index", range(len(res)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(idx, res, lw=0.8, color="#6366f1")
    axes[0].axhline(0, color="red", ls="--", lw=1)
    axes[0].set_title(f"{nome} — {modelo}: Residuals over Time")
    axes[0].set_xlabel("Data"); axes[0].set_ylabel("Residual [MVA]")

    axes[1].hist(res, bins=40, color="#6366f1", edgecolor="white")
    axes[1].set_title(f"{nome} — {modelo}: Residual Distribution")
    axes[1].set_xlabel("Residual [MVA]"); axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals_{tid}_{modelo}.svg",
                dpi=150, bbox_inches="tight"); plt.close()


def plot_future_forecast(y_hist, future_mean, future_lower, future_upper,
                         tid, modelo, output_dir):
    """
    Previsão futura com intervalo de confiança via bootstrap (95%).
    Inclui todos os dados históricos para contextualizar a projeção.
    """
    nome = MAPA_PLOT.get(tid, tid)
    fig, ax = plt.subplots(figsize=(16, 5))
    y_hist.plot(ax=ax, label="Historical", color="#2563eb", lw=1.2)
    ax.plot(future_mean.index, future_mean.values,
            label="Future forecast", color="#dc2626", lw=2, ls="--")
    ax.fill_between(future_mean.index, future_lower, future_upper,
                    alpha=0.2, color="#dc2626", label="95% CI (bootstrap)")
    ax.axvline(y_hist.index[-1], color="gray", ls=":", lw=1.5, label="Último dado real")
    ax.set_title(f"{nome} — {modelo}: Future Forecast with 95% CI", fontsize=13)
    ax.set_xlabel("Date"); ax.set_ylabel("Smax [MVA]")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/future_{tid}_{modelo}.svg",
                dpi=150, bbox_inches="tight"); plt.close()
