# =============================================================================
# EDA.py — Análise Exploratória de Dados (EDA)
# =============================================================================
# Objetivo: caracterizar a série temporal de Potência Aparente Máxima (Smax)
# dos transformadores de força, identificando:
#   - Tendência e sazonalidade
#   - Outliers e gaps de dados
#   - Autocorrelação (ACF/PACF) com lags suficientes para sazonalidade anual
#   - Relação com variáveis meteorológicas
#   - Perfis por dia da semana, mês e ano
#
# Todos os transformadores definidos em config.TRANSFORMADORES são analisados.
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import (
    DATA_PATH, WEATHER_PATH, TRANSFORMADORES, MAPA_PLOT,
    START_DATE, OUTPUT_EDA, PERIOD
)
from utils import (
    preparar_serie_diaria, adicionar_features_calendario,
    carregar_weather, get_splits, log
)


# =============================================================================
# FUNÇÕES DE PLOT
# =============================================================================

def savefig(path: str, dpi: int = 160):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_serie_com_splits(y: pd.Series, tid: str):
    """Série completa com marcação visual dos conjuntos treino/val/teste."""
    splits = get_splits(y)
    nome   = MAPA_PLOT.get(tid, tid)

    fig, ax = plt.subplots(figsize=(14, 4))
    y.loc[splits["train"]].plot(ax=ax, label="Treino",     color="#2563eb", lw=1.0)
    y.loc[splits["val"]  ].plot(ax=ax, label="Validação",  color="#f59e0b", lw=1.0)
    y.loc[splits["test"] ].plot(ax=ax, label="Teste",      color="#16a34a", lw=1.0)

    for vline in [splits["train"][-1], splits["val"][-1]]:
        ax.axvline(vline, color="gray", linestyle=":", lw=1)

    ax.set_title(f"{nome} — Série Temporal com Partições", fontsize=13)
    ax.set_xlabel("Data")
    ax.set_ylabel("Smax [MVA]")
    ax.legend()
    savefig(os.path.join(OUTPUT_EDA, f"{tid}_01_serie_splits.svg"))


def plot_hist_boxplot(y: pd.Series, tid: str):
    """Histograma e boxplot da distribuição de Smax."""
    nome = MAPA_PLOT.get(tid, tid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(y.dropna(), bins=60, color="#2563eb", edgecolor="white")
    axes[0].set_title(f"{nome} — Histograma de Smax")
    axes[0].set_xlabel("Smax [MVA]")
    axes[0].set_ylabel("Frequência")

    axes[1].boxplot(y.dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#dbeafe"))
    axes[1].set_title(f"{nome} — Boxplot de Smax")
    axes[1].set_ylabel("Smax [MVA]")

    savefig(os.path.join(OUTPUT_EDA, f"{tid}_02_hist_box.svg"))


def plot_decomposicao(y: pd.Series, tid: str, period: int = PERIOD):
    """
    Decomposição sazonal aditiva.
    Revela tendência de longo prazo, sazonalidade anual e resíduo.
    """
    nome = MAPA_PLOT.get(tid, tid)

    # Necessário pelo menos 2 × period de dados
    if len(y.dropna()) < 2 * period:
        log.warning(f"{tid}: série curta demais para decomposição (period={period}).")
        return

    result = seasonal_decompose(y.dropna(), model="additive", period=period)
    fig = result.plot()
    fig.set_size_inches(13, 9)
    fig.suptitle(f"{nome} — Decomposição Sazonal Aditiva (período={period} dias)",
                 fontsize=12, y=1.01)
    savefig(os.path.join(OUTPUT_EDA, f"{tid}_03_decomposicao.svg"))


def plot_acf_pacf(y: pd.Series, tid: str, nlags: int = 400):
    """
    ACF e PACF com nlags >= 366 para revelar autocorrelação anual.
    Fundamental para definir os lags do modelo e confirmar sazonalidade.
    """
    nome = MAPA_PLOT.get(tid, tid)
    y_clean = y.dropna()
    nlags = min(nlags, len(y_clean) // 2 - 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    plot_acf(y_clean,  lags=nlags, ax=axes[0], zero=False)
    axes[0].set_title(f"{nome} — ACF (lags={nlags})")

    plot_pacf(y_clean, lags=nlags, ax=axes[1], method="ywm", zero=False)
    axes[1].set_title(f"{nome} — PACF (lags={nlags})")

    plt.tight_layout()
    savefig(os.path.join(OUTPUT_EDA, f"{tid}_04_acf_pacf.svg"))


def plot_boxplot_mensal(y: pd.Series, tid: str):
    """Distribuição de Smax por mês — revela sazonalidade intra-anual."""
    nome = MAPA_PLOT.get(tid, tid)
    df = y.to_frame("Smax")
    df["month"] = df.index.month
    data = [df.loc[df["month"] == m, "Smax"].dropna().values for m in range(1, 13)]
    meses = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]

    plt.figure(figsize=(13, 4))
    plt.boxplot(data, labels=meses, patch_artist=True,
                boxprops=dict(facecolor="#dbeafe"), showfliers=False)
    plt.title(f"{nome} — Distribuição Mensal de Smax", fontsize=13)
    plt.xlabel("Mês")
    plt.ylabel("Smax [MVA]")
    savefig(os.path.join(OUTPUT_EDA, f"{tid}_05_boxplot_mensal.svg"))


def plot_perfil_semanal(y: pd.Series, tid: str):
    """Média de Smax por dia da semana — detecta padrão de carga útil vs. fim de semana."""
    nome  = MAPA_PLOT.get(tid, tid)
    df    = y.to_frame("Smax")
    df["weekday"] = df.index.weekday
    means = df.groupby("weekday")["Smax"].mean()
    dias  = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]

    plt.figure(figsize=(9, 4))
    plt.bar(dias, means.values, color="#2563eb", edgecolor="white")
    plt.title(f"{nome} — Média de Smax por Dia da Semana", fontsize=13)
    plt.ylabel("Média Smax [MVA]")
    savefig(os.path.join(OUTPUT_EDA, f"{tid}_06_perfil_semanal.svg"))


def plot_heatmap_ano_dia(y: pd.Series, tid: str):
    """
    Heatmap Ano × Dia-do-Ano.
    Permite visualizar padrões sazonais repetitivos e anomalias entre anos.
    """
    nome = MAPA_PLOT.get(tid, tid)
    df = y.to_frame("Smax")
    df["year"] = df.index.year
    df["doy"]  = df.index.dayofyear
    pivot = df.pivot_table(index="year", columns="doy", values="Smax", aggfunc="mean")

    plt.figure(figsize=(15, max(3, len(pivot) * 1.2)))
    im = plt.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, label="Smax [MVA]")
    plt.title(f"{nome} — Heatmap Ano × Dia-do-Ano", fontsize=13)
    plt.ylabel("Ano")
    plt.xlabel("Dia do Ano")
    plt.yticks(range(len(pivot.index)), pivot.index)
    savefig(os.path.join(OUTPUT_EDA, f"{tid}_07_heatmap.svg"))


def plot_outliers(y: pd.Series, tid: str):
    """
    Detecção de outliers via IQR e Z-score.
    Apresenta ambas as abordagens para robustez metodológica.
    """
    nome = MAPA_PLOT.get(tid, tid)
    s = y.dropna()

    # IQR
    q1, q3 = np.percentile(s, [25, 75])
    iqr     = q3 - q1
    low_iqr = q1 - 1.5 * iqr
    high_iqr= q3 + 1.5 * iqr
    outliers_iqr = s[(s < low_iqr) | (s > high_iqr)]

    # Z-score
    z = (s - s.mean()) / s.std()
    outliers_z = s[np.abs(z) > 3]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(s.index, s.values, linewidth=0.8, color="#64748b", label="Smax")
    ax.scatter(outliers_iqr.index, outliers_iqr.values,
               color="#f59e0b", s=20, zorder=5, label=f"Outlier IQR (n={len(outliers_iqr)})")
    ax.scatter(outliers_z.index, outliers_z.values,
               color="#dc2626", s=20, zorder=5, marker="x",
               label=f"Outlier Z>3 (n={len(outliers_z)})")
    ax.set_title(f"{nome} — Detecção de Outliers", fontsize=13)
    ax.set_xlabel("Data")
    ax.set_ylabel("Smax [MVA]")
    ax.legend()
    savefig(os.path.join(OUTPUT_EDA, f"{tid}_08_outliers.svg"))


def plot_relacoes_climaticas(y: pd.Series, weather: pd.DataFrame, tid: str):
    """
    Relação entre Smax e variáveis meteorológicas.
    Scatter plots + série temporal sobreposta para análise de correlação visual.
    """
    nome = MAPA_PLOT.get(tid, tid)
    df = y.to_frame("Smax").join(weather, how="left").ffill().bfill()

    climate_cols = [c for c in weather.columns if c in df.columns]
    if not climate_cols:
        return

    corr = df[["Smax"] + climate_cols].corr()["Smax"].drop("Smax")
    corr.to_csv(os.path.join(OUTPUT_EDA, f"{tid}_corr_climatica.csv"))
    log.info(f"{tid} — Correlações com clima:\n{corr.to_string()}")

    ncols = min(2, len(climate_cols))
    nrows = (len(climate_cols) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(climate_cols):
        axes[i].scatter(df[col], df["Smax"], s=6, alpha=0.5, color="#2563eb")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Smax [MVA]")
        r = corr[col] if col in corr else 0
        axes[i].set_title(f"{nome}: Smax vs {col} (r={r:.3f})")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    savefig(os.path.join(OUTPUT_EDA, f"{tid}_09_scatter_clima.svg"))


# =============================================================================
# ESTATÍSTICAS RESUMO
# =============================================================================

def compute_summary(y: pd.Series, tid: str) -> dict:
    """Estatísticas descritivas para tabela do artigo."""
    s = y.dropna()
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1

    # Taxa de dados faltantes original (antes de interpolar)
    return {
        "id":              tid,
        "nome_plot":       MAPA_PLOT.get(tid, tid),
        "inicio":          str(s.index.min().date()),
        "fim":             str(s.index.max().date()),
        "n_dias":          int(len(y)),       # inclui interpolados
        "n_validos":       int(s.count()),
        "media":           round(float(s.mean()), 4),
        "desvio_pad":      round(float(s.std()), 4),
        "minimo":          round(float(s.min()), 4),
        "p25":             round(float(q1), 4),
        "mediana":         round(float(s.median()), 4),
        "p75":             round(float(q3), 4),
        "maximo":          round(float(s.max()), 4),
        "iqr":             round(float(iqr), 4),
        "coef_variacao_%": round(float(s.std() / s.mean() * 100), 2),
        "tx_outlier_iqr":  round(float(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).mean() * 100), 2),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=== EDA INICIADO ===")

    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES)].copy()

    if df.empty:
        raise ValueError(
            f"Nenhum dado encontrado para {TRANSFORMADORES}. "
            "Verifique os IDs em config.py e o arquivo de dados."
        )

    # Carrega clima se disponível
    weather = None
    try:
        weather = carregar_weather()
    except Exception as e:
        log.warning(f"Dataset climático não carregado: {e}")

    summaries = []

    for tid, df_tr in df.groupby("id"):
        log.info(f"--- EDA: {tid} ---")
        y = preparar_serie_diaria(df_tr)
        summaries.append(compute_summary(y, tid))

        plot_serie_com_splits(y, tid)
        plot_hist_boxplot(y, tid)
        plot_decomposicao(y, tid)
        plot_acf_pacf(y, tid, nlags=400)   # ← nlags >= 366 para sazonalidade anual
        plot_boxplot_mensal(y, tid)
        plot_perfil_semanal(y, tid)
        plot_heatmap_ano_dia(y, tid)
        plot_outliers(y, tid)

        if weather is not None:
            plot_relacoes_climaticas(y, weather, tid)

    # Salvar estatísticas resumo (tabela para o artigo)
    df_sum = pd.DataFrame(summaries).sort_values("id")
    out_path = os.path.join(OUTPUT_EDA, "tabela_resumo_descritivo.csv")
    df_sum.to_csv(out_path, index=False)
    log.info(f"Resumo estatístico salvo em: {out_path}")
    log.info("=== EDA CONCLUÍDO ===")
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
