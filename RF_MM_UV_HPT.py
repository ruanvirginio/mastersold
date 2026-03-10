import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures
from statsmodels.tsa.seasonal import seasonal_decompose

DATA_PATH = "bases_tratadas/daily_peak_transformers_dataset.csv"

MAPA_PLOT = {
    "T21a": "T1",
    "T19a": "T2",
    "T20a": "T3",
}

TRANSFORMADORES_ALVO = ['T21a', 'T19a', 'T20a']
START_DATE = "2021-01-01"
END_VALIDATION = "2023-12-31"

LAGS = 365
STEPS = 365

OUTPUT_DIR = "resultados_UV"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return rmse, mape, r2

def preparar_serie_diaria(df):
    serie = df.copy()
    serie["Time"] = pd.to_datetime(serie["datahora"])
    serie = serie.set_index("Time").sort_index()
    full_index = pd.date_range(serie.index.min(), serie.index.max(), freq="D")
    serie = serie.reindex(full_index)
    serie = serie.loc[START_DATE:]
    serie["Smax"] = serie["Smax"].interpolate(method="time").ffill()
    return serie["Smax"]

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
            "estimator": XGBRegressor(objective="reg:squarederror", random_state=100, n_jobs=-1),
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


def plot_split_real(y, transformer_id):
    split = pd.to_datetime(END_VALIDATION)
    plt.figure(figsize=(13, 5))
    y.loc[:split].plot(label="Train",  fontsize=14)
    y.loc[split:].plot(label="Test",  fontsize=14)
    plt.axvline(split, linestyle="--")
    plt.title(f"{transformer_id} - Train/Test split",  fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_split_{transformer_id}.svg", dpi=150)
    plt.close()

def plot_decomposition(y, transformer_id):
    result = seasonal_decompose(y, model="additive", period=365)
    fig = result.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_decomposition_{transformer_id}.svg", dpi=150)
    plt.close()

def plot_forecast(y, pred, transformer_id, modelo):
    split = pd.to_datetime(END_VALIDATION)

    nome_plot = MAPA_PLOT.get(transformer_id, transformer_id)

    plt.figure(figsize=(13, 5))
    # y.loc[:split].plot(label="Train")
    y.loc[split:].plot(label="Test", fontsize=14)
    pred["pred"].plot(label="Forecast", fontsize=14)
    plt.axvline(split, linestyle="--")
    plt.title(f"{nome_plot} - {modelo} - Univariate", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot_forecast_{transformer_id}_{modelo}.svg", dpi=150)
    plt.close()

# TREINO + HYPERPARAM

def treinar_multimodelo(df_filtrado):
    modelos_dict = criar_modelos()
    resultados = []
    predicoes = {}

    for transformer_id, df_tr in df_filtrado.groupby("id"):
        print(f"\n>>> {transformer_id}")
        y = preparar_serie_diaria(df_tr)

        # EDA
        plot_split_real(y, transformer_id)
        plot_decomposition(y, transformer_id)

        initial_train_size = (y.index <= END_VALIDATION).sum()
        cv = TimeSeriesFold(steps=STEPS, initial_train_size=initial_train_size, refit=False)

        for nome_modelo, info in modelos_dict.items():
            print(f"   -> {nome_modelo}")
            best_mae = np.inf
            best_params = None
            best_forecaster = None
            best_pred = None

            # Hiperparametrização manual
            for params in info["params"]:
                est = info["estimator"].set_params(**params)
                forecaster = ForecasterRecursive(
                    estimator=est,
                    lags=LAGS,
                    window_features=RollingFeatures(stats=["mean"], window_sizes=[365])
                )

                # backtesting
                _, pred = backtesting_forecaster(
                    forecaster=forecaster,
                    y=y,
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

            # Métricas do melhor modelo
            y_true = y.loc[best_pred.index]
            y_pred = best_pred["pred"]
            rmse, mape, r2 = compute_metrics(y_true, y_pred)
            resultados.append({
                "id": transformer_id,
                "modelo": nome_modelo,
                "best_params": best_params,
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2
            })

            predicoes[(transformer_id, nome_modelo)] = best_pred
            plot_forecast(y, best_pred, transformer_id, nome_modelo)

    return pd.DataFrame(resultados), predicoes

def main():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin-1")
    df = df[df["id"].isin(TRANSFORMADORES_ALVO)]
    resultados, _ = treinar_multimodelo(df)
    resultados.to_csv(f"{OUTPUT_DIR}/metricas_UV.csv", index=False)
    print(resultados.sort_values(["id", "RMSE"]))

if __name__ == "__main__":
    main()