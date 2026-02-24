import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

DATA_PATH = "bases_tratadas/daily_peak_transformers_dataset.csv"
WEATHER_PATH = "bases_tratadas/weather_dataset_2022-2024.csv"  # opcional
USE_WEATHER = os.path.exists(WEATHER_PATH)

TRANSFORMERS = ["T21a", "T22a", "T70a"]

START_DATE = "2021-01-01"
END_DATE = None  # ou "2024-12-31"

OUT_DIR = "eda_out"
os.makedirs(OUT_DIR, exist_ok=True)

PERIOD = 365  # sazonalidade anual

def savefig(path, dpi=160):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def preparar_serie_diaria(df_tr: pd.DataFrame) -> pd.Series:
    s = df_tr.copy()
    s["Time"] = pd.to_datetime(s["datahora"])
    s = s.set_index("Time").sort_index()

    full_index = pd.date_range(s.index.min().normalize(), s.index.max().normalize(), freq="D")
    s = s.reindex(full_index)

    s = s.loc[pd.to_datetime(START_DATE):]
    if END_DATE is not None:
        s = s.loc[:pd.to_datetime(END_DATE)]

    # target
    s["Smax"] = s["Smax"].interpolate(method="time").ffill().bfill()
    return s["Smax"]

def add_calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    cal = pd.DataFrame(index=idx)
    cal["weekday"] = idx.weekday
    cal["is_weekend"] = (cal["weekday"] >= 5).astype(int)
    cal["month"] = idx.month
    cal["dayofyear"] = idx.dayofyear
    cal["year"] = idx.year
    return cal

def load_weather() -> pd.DataFrame:
    w = pd.read_csv(WEATHER_PATH, sep=";", encoding="latin-1")
    w["DATA"] = pd.to_datetime(w["DATA"])
    w = w.set_index("DATA").sort_index()

    cols = ["temp_max_anos", "temp_min_anos", "temp_media_anos", "precipitacao_anos"]
    w = w[cols].copy()
    return w


# EDA PLOTS
def plot_series(y, tid):
    plt.figure(figsize=(13,4))
    y.plot(linewidth=1)
    plt.title(f"{tid} - Daily Smax (continuous)")
    plt.xlabel("Date"); plt.ylabel("Smax")
    savefig(os.path.join(OUT_DIR, f"{tid}_01_series.svg"))

def plot_hist_box(y, tid):
    fig = plt.figure(figsize=(12,4))
    plt.hist(y.values, bins=60)
    plt.title(f"{tid} - Histogram of Smax")
    plt.xlabel("Smax"); plt.ylabel("Count")
    savefig(os.path.join(OUT_DIR, f"{tid}_02_hist.svg"))

    plt.figure(figsize=(10,3))
    plt.boxplot(y.values, vert=False)
    plt.title(f"{tid} - Boxplot of Smax")
    plt.xlabel("Smax")
    savefig(os.path.join(OUT_DIR, f"{tid}_03_box.svg"))

def plot_decomposition(y, tid, period=365):
    res = seasonal_decompose(y, model="additive", period=period)
    fig = res.plot()
    fig.set_size_inches(12,8)
    plt.suptitle(f"{tid} - Seasonal Decomposition (period={period})", y=1.02)
    savefig(os.path.join(OUT_DIR, f"{tid}_04_decomposition.svg"))

def plot_acf_pacf(y, tid, nlags=60):
    plt.figure(figsize=(10,4))
    plot_acf(y, lags=nlags)
    plt.title(f"{tid} - ACF (lags={nlags})")
    savefig(os.path.join(OUT_DIR, f"{tid}_05_acf.svg"))

    plt.figure(figsize=(10,4))
    plot_pacf(y, lags=nlags, method="ywm")
    plt.title(f"{tid} - PACF (lags={nlags})")
    savefig(os.path.join(OUT_DIR, f"{tid}_06_pacf.svg"))

def plot_monthly_box(y, tid):
    df = pd.DataFrame({"Smax": y.values}, index=y.index)
    df["month"] = df.index.month
    data = [df.loc[df["month"] == m, "Smax"].values for m in range(1,13)]

    plt.figure(figsize=(12,4))
    plt.boxplot(data, labels=list(range(1,13)), showfliers=False)
    plt.title(f"{tid} - Monthly distribution (boxplot)")
    plt.xlabel("Month"); plt.ylabel("Smax")
    savefig(os.path.join(OUT_DIR, f"{tid}_07_monthly_boxplot.svg"))

def plot_weekday_profile(y, tid):
    df = pd.DataFrame({"Smax": y.values}, index=y.index)
    df["weekday"] = df.index.weekday
    means = df.groupby("weekday")["Smax"].mean()

    plt.figure(figsize=(10,4))
    plt.plot(means.index, means.values, marker="o")
    plt.xticks(range(7), ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    plt.title(f"{tid} - Mean Smax by weekday")
    plt.xlabel("Weekday"); plt.ylabel("Mean Smax")
    savefig(os.path.join(OUT_DIR, f"{tid}_08_weekday_mean.svg"))

def plot_year_day_heatmap(y, tid):
    """Heatmap ano × dia-do-ano para ver sazonalidade anual."""
    df = pd.DataFrame({"Smax": y.values}, index=y.index)
    df["year"] = df.index.year
    df["doy"] = df.index.dayofyear
    pivot = df.pivot_table(index="year", columns="doy", values="Smax", aggfunc="mean")

    plt.figure(figsize=(14,4))
    plt.imshow(pivot.values, aspect="auto")
    plt.title(f"{tid} - Heatmap (Year x Day-of-Year)")
    plt.ylabel("Year"); plt.xlabel("Day of year")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Smax")
    savefig(os.path.join(OUT_DIR, f"{tid}_09_heatmap_year_doy.svg"))

def plot_weekday_vs_sunday_15days(y, tid, start_date=None, window_days=15):
    s = y.copy()

    if start_date is None:
        start = s.index.max() - pd.Timedelta(days=window_days-1)
    else:
        start = pd.to_datetime(start_date)
    end = start + pd.Timedelta(days=window_days-1)

    w = s.loc[start:end]
    if len(w) < 5:
        return

    df = pd.DataFrame({"Smax": w.values}, index=w.index)
    df["weekday"] = df.index.weekday
    df["is_sunday"] = (df["weekday"] == 6).astype(int)

    weekday_curve = df.loc[df["is_sunday"] == 0, "Smax"].copy()
    sunday_curve = df.loc[df["is_sunday"] == 1, "Smax"].copy()

    plt.figure(figsize=(12,4))
    weekday_curve.plot(label="Weekdays (non-Sun)", linewidth=2)
    if len(sunday_curve) > 0:
        sunday_curve.plot(label="Sundays", linewidth=2)
    plt.title(f"{tid} - Weekdays vs Sundays (window={window_days} days)")
    plt.xlabel("Date"); plt.ylabel("Smax")
    plt.legend()
    savefig(os.path.join(OUT_DIR, f"{tid}_10_weekdays_vs_sundays_{window_days}d.svg"))

def plot_weather_relations(y, weather, tid):
    df = pd.DataFrame({"Smax": y})
    df = df.join(weather, how="left").ffill().bfill()

    # correlação
    corr = df.corr(numeric_only=True)["Smax"].drop("Smax").sort_values(ascending=False)
    corr.to_csv(os.path.join(OUT_DIR, f"{tid}_weather_corr.csv"))

    # scatter básicos
    for col in ["temp_media_anos", "precipitacao_anos", "temp_max_anos", "temp_min_anos"]:
        if col not in df.columns:
            continue
        plt.figure(figsize=(5,4))
        plt.scatter(df[col].values, df["Smax"].values, s=8)
        plt.title(f"{tid} - Smax vs {col}")
        plt.xlabel(col); plt.ylabel("Smax")
        savefig(os.path.join(OUT_DIR, f"{tid}_11_scatter_{col}.svg"))

    plt.figure(figsize=(13,4))
    df["Smax"].plot(label="Smax", linewidth=1)
    df["temp_media_anos"].plot(label="temp_media_anos", linewidth=1)
    plt.title(f"{tid} - Smax and temperature (scaled axes not applied)")
    plt.legend()
    savefig(os.path.join(OUT_DIR, f"{tid}_12_series_smax_temp.svg"))


def compute_summary(y, tid):
    s = pd.Series(y.values)
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outlier_rate = ((s < low) | (s > high)).mean()

    return {
        "id": tid,
        "start": str(y.index.min().date()),
        "end": str(y.index.max().date()),
        "n_days": int(len(y)),
        "mean": float(np.mean(s)),
        "std": float(np.std(s)),
        "min": float(np.min(s)),
        "p25": float(q1),
        "median": float(np.median(s)),
        "p75": float(q3),
        "max": float(np.max(s)),
        "iqr": float(iqr),
        "outlier_rate_iqr": float(outlier_rate),
    }


def main():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")

    df = df[df["id"].isin(TRANSFORMERS)].copy()
    if df.empty:
        raise ValueError("Nenhum dado encontrado para os TRANSFORMERS informados. Verifique os IDs.")

    weather = load_weather() if USE_WEATHER else None

    summaries = []

    for tid, df_tr in df.groupby("id"):
        print(f"EDA -> {tid}")

        y = preparar_serie_diaria(df_tr)
        summaries.append(compute_summary(y, tid))

        plot_series(y, tid)
        plot_hist_box(y, tid)
        plot_decomposition(y, tid, period=PERIOD)
        plot_acf_pacf(y, tid, nlags=60)
        plot_monthly_box(y, tid)
        plot_weekday_profile(y, tid)
        plot_year_day_heatmap(y, tid)
        plot_weekday_vs_sunday_15days(y, tid, start_date=None, window_days=15)

        if weather is not None:
            plot_weather_relations(y, weather, tid)

    df_sum = pd.DataFrame(summaries).sort_values("id")
    df_sum.to_csv(os.path.join(OUT_DIR, "summary_stats_3_transformers.csv"), index=False)

if __name__ == "__main__":
    main()
