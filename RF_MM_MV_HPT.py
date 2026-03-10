import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures

import holidays

DATA_PATH = "bases_tratadas/daily_peak_transformers_dataset.csv"
WEATHER_PATH = "bases_tratadas/weather_dataset_2022-2024.csv"

MAPA_PLOT = {
    "T21a": "T1",
    "T19a": "T2",
    "T20a": "T3",
}

TRANSFORMADORES_ALVO = ['T21a', 'T19a', 'T20a']


TRANSFORMADORES_ALVO = list(MAPA_PLOT.keys())

START_DATE = "2021-01-01"
END_VALIDATION = "2023-12-31"

LAGS = 365
STEPS = 365

OUTPUT_DIR = "resultados_MV"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mape, r2

def carregar_weather():
    weather = pd.read_csv(WEATHER_PATH, sep=";", encoding="latin-1")
    weather["DATA"] = pd.to_datetime(weather["DATA"])
    weather = weather.set_index("DATA").sort_index()
    weather = weather[[
        "temp_max_anos",
        "temp_min_anos",
        "temp_media_anos",
        "precipitacao_anos"
    ]]
    return weather


def preparar_serie_diaria(df, weather):
    serie = df.copy()
    serie["Time"] = pd.to_datetime(serie["datahora"])
    serie = serie.set_index("Time").sort_index()

    full_index = pd.date_range(serie.index.min(), serie.index.max(), freq="D")
    serie = serie.reindex(full_index)

    serie = serie.join(weather, how="left")
    serie = serie.loc[START_DATE:]

    # TARGET
    serie["Smax"] = serie["Smax"].interpolate(method="time").ffill()

    br_holidays = holidays.Brazil()

    serie["weekday"] = serie.index.weekday
    serie["weekday_sin"] = np.sin(2 * np.pi * serie["weekday"] / 7)
    serie["weekday_cos"] = np.cos(2 * np.pi * serie["weekday"] / 7)

    serie["dayofyear"] = serie.index.dayofyear
    serie["dayofyear_sin"] = np.sin(2 * np.pi * serie["dayofyear"] / 365)
    serie["dayofyear_cos"] = np.cos(2 * np.pi * serie["dayofyear"] / 365)

    serie["month"] = serie.index.month
    serie["month_sin"] = np.sin(2 * np.pi * serie["month"] / 12)
    serie["month_cos"] = np.cos(2 * np.pi * serie["month"] / 12)

    serie["is_weekend"] = (serie.index.weekday >= 5).astype(int)
    serie["is_holiday"] = serie.index.isin(br_holidays).astype(int)

    # EXÓGENAS
    exog_cols = [
        "weekday", "is_weekend", "month",
        "dayofyear", "is_holiday",
        "temp_max_anos", "temp_min_anos",
        "temp_media_anos", "precipitacao_anos"
    ]
    serie[exog_cols] = serie[exog_cols].ffill().bfill()

    y = serie["Smax"]
    exog = serie[exog_cols]
    return y, exog

# MODELOS + HYPERPARAMS

def criar_modelos():
    return {
        "LGBM": {
            "estimator": LGBMRegressor(random_state=100, n_jobs=-1),
            "params": [
                {"n_estimators": 100, "learning_rate": 0.1, "max_depth": -1},
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": -1},
            ]
        },
        "XGBoost": {
            "estimator": XGBRegressor(objective="reg:squarederror", random_state=100),
            "params": [
                {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
            ]
        },
        "GradientBoosting": {
            "estimator": GradientBoostingRegressor(random_state=100),
            "params": [
                {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
            ]
        }
    }

# PLOTS

def plot_forecast(y, pred, transformer_id, modelo):
    split = pd.to_datetime(END_VALIDATION)

    nome_plot = MAPA_PLOT.get(transformer_id, transformer_id)

    plt.figure(figsize=(13,5))
    # y.loc[:split].plot(label="Train")
    y.loc[split:].plot(label="Test",  fontsize=14)
    pred["pred"].plot(label="Forecast",  fontsize=14)
    plt.axvline(split, linestyle="--")
    plt.title(f"{nome_plot} - {modelo} - Multivariate", fontsize=16)
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/forecast_{transformer_id}_{modelo}_MV.svg")
    plt.close()

def plot_importance(model, exog_columns, path):

    if not hasattr(model, "feature_importances_"):
        return

    # importância das exógenas + Smax
    n_exog = len(exog_columns)
    imp_values = model.feature_importances_[-n_exog:]
    imp = pd.Series(imp_values, index=exog_columns)
    imp.sort_values().tail(20).plot.barh(figsize=(6,4))
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# TREINO + HYPERPARAM

def treinar(df, weather):
    resultados = []
    modelos_dict = criar_modelos()

    for tid, df_tr in df.groupby("id"):
        print(f"\n>>> {tid}")

        y, exog = preparar_serie_diaria(df_tr, weather)
        initial_train_size = (y.index <= END_VALIDATION).sum()
        cv = TimeSeriesFold(
            steps=STEPS,
            initial_train_size=initial_train_size,
            refit=False
        )

        for nome_modelo, info in modelos_dict.items():
            best_mae = np.inf
            best_params = None
            best_forecaster = None
            best_pred = None

            for params in info["params"]:
                est = info["estimator"].set_params(**params)
                forecaster = ForecasterRecursive(
                    estimator=est,
                    lags=LAGS,
                    window_features=RollingFeatures(
                        stats=["mean"],
                        window_sizes=[365]
                    )
                )

                # backtesting
                _, pred = backtesting_forecaster(
                    forecaster=forecaster,
                    y=y,
                    exog=exog,
                    cv=cv,
                    metric="mean_absolute_error",
                    verbose=False
                )

                mae = np.mean(np.abs(y.loc[pred.index] - pred["pred"]))
                if mae < best_mae:
                    best_mae = mae
                    best_params = params
                    best_forecaster = forecaster
                    best_pred = pred

            # Métricas do melhor
            y_true = y.loc[best_pred.index]
            y_pred = best_pred["pred"]
            rmse, mape, r2 = compute_metrics(y_true, y_pred)
            resultados.append([tid, nome_modelo, best_params, rmse, mape, r2])

            plot_forecast(y, best_pred, tid, nome_modelo)

            best_forecaster.fit(y=y, exog=exog)
            plot_importance(best_forecaster.estimator, exog.columns, f"{OUTPUT_DIR}/importance_{tid}_{nome_modelo}_MV.svg")

    return pd.DataFrame(resultados, columns=["id","modelo","best_params","RMSE","MAPE","R2"])


def main():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES_ALVO)]
    weather = carregar_weather()
    resultados = treinar(df, weather)
    resultados.to_csv(f"{OUTPUT_DIR}/metricas_MV.csv", index=False)
    print(resultados)

if __name__ == "__main__":
    main()
