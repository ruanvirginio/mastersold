# Previsão de Potência Aparente — Mestrado em TI
## Subestações da Paraíba · Random Forest / Boosting / LSTM / CNN / SVR / SARIMAX / Transformer

---

## Estrutura do Projeto

```
masters/
├── config.py                  ← EDITE AQUI: IDs, datas, hiperparâmetros
├── utils.py                   ← Funções compartilhadas (métricas, plots, pré-proc.)
├── EDA.py                     ← Análise Exploratória
├── models_boosting.py         ← XGBoost · LightGBM · GradientBoosting (UV + MV)
├── models_svr.py              ← SVR com seleção de features
├── models_sarimax.py          ← SARIMAX (baseline estatístico)
├── models_dl.py               ← LSTM · Stacked LSTM · CNN-LSTM
├── models_transformer.py      ← Transformer (modo lite e full)
├── report_generator.py        ← Relatório HTML completo automático
│
├── bases_tratadas/
│   ├── daily_peak_transformers_dataset.csv
│   └── weather_dataset_2022-2024.csv
│
├── eda_out/                   ← Gráficos EDA
├── resultados_UV/             ← Boosting univariado
├── resultados_MV/             ← Boosting multivariado
├── resultados_DL/             ← LSTM / CNN-LSTM
├── resultados_SVR/            ← SVR
├── resultados_SARIMAX/        ← SARIMAX
├── resultados_Transformer/    ← Transformer
├── resultados_comparacao/     ← Tabela CSV consolidada
├── previsao_futura/           ← CSVs + gráficos de previsão 2025-2027
└── relatorio/                 ← relatorio_final.html + tabela_resultados.tex
```

---

## Instalação

```bash
pip install numpy pandas matplotlib scikit-learn lightgbm xgboost \
            skforecast statsmodels holidays tensorflow scipy
```

---

## Ordem de Execução

```bash
# 1. Análise exploratória
python EDA.py

# 2. Modelos de Boosting (UV e MV) — mais rápido, ~5-15 min
python models_boosting.py

# 3. SVR — ~10-20 min
python models_svr.py

# 4. SARIMAX — ~15-30 min (pode variar com convergência)
python models_sarimax.py

# 5. LSTM / CNN-LSTM — ~20-60 min no Ryzen 5700G
python models_dl.py

# 6. Transformer — escolha o modo:
# python models_transformer.py --mode lite    # ~20-40 min CPU (recomendado para testes)
python models_transformer.py --mode full    # ~2-4h CPU (melhores resultados)
# python models_transformer.py --mode both    # roda os dois e compara

# 7. Relatório final (após rodar todos os modelos)
python report_generator.py
```

---

## Metodologia sem Data Leakage

```
┌─────────────────┬──────────────┬─────────────┐
│    TREINO       │  VALIDAÇÃO   │   TESTE     │
│  2021 – 2022    │    2023      │    2024     │
│  ajuste modelo  │  HPT / ES    │ avaliação   │
│                 │              │  final única│
└─────────────────┴──────────────┴─────────────┘
```

**Garantias implementadas:**
1. HPT realizado SOMENTE sobre validação — teste nunca tocado durante HPT
2. Conjunto de teste acessado UMA única vez, após HPT encerrado
3. Scalers (DL) ajustados SOMENTE nos dados de treino
4. Early stopping monitora val_loss — nunca test_loss
5. Previsão recursiva: o modelo usa suas próprias previsões como lags — nunca valores reais futuros
6. Previsão futura (2025-2027): modelo retreinado em TODOS os dados, projeta iterativamente para frente

---

## Previsão de 2025, 2026, 2027

**Backtesting (métricas no artigo):**
Simula como o modelo teria previsto 2024, usando apenas dados até 2023.
Produz métricas reais (RMSE, R², sMAPE) que validam a qualidade do modelo.

**Previsão futura real (aplicação prática):**
O modelo é retreinado em todos os dados (2021-2024) e projeta para frente.
Sem ground truth → sem métricas, com Intervalo de Confiança 95% (bootstrap residual).

**Sobre prever 3 anos:**
Com 4 anos de histórico, prever 3 anos à frente é possível mas com incerteza crescente.
Para o Plano de Negócios da distribuidora, apresente:
- Previsão central (linha)
- IC 95% (faixa) que aumenta com o tempo
- Discussão de limitações (sem dados de crescimento de carga, conexões novas, etc.)

---

## Hardware — Ryzen 5700G

| Modelo              | Tempo estimado (CPU) | Observação |
|---------------------|---------------------|------------|
| Boosting (UV+MV)    | 5–15 min            | n_jobs=-1 usa todos os cores |
| SVR                 | 10–20 min           | Escala com n_amostras² |
| SARIMAX             | 15–30 min           | Por transformador |
| LSTM / CNN-LSTM     | 20–60 min           | TensorFlow CPU |
| Transformer (lite)  | 20–40 min           | d_model=64 |
| Transformer (full)  | 2–4 horas           | d_model=256 |

**Dica:** Para desenvolvimento/testes, use `--mode lite` no Transformer.
Para o resultado final do artigo, use `--mode full` (deixe rodando overnight).

---

## Métricas

| Métrica | Descrição | Recomendação |
|---------|-----------|-------------|
| RMSE    | Raiz do erro quadrático médio | Penaliza erros grandes |
| MAE     | Erro absoluto médio | Robusto a outliers |
| MAPE    | Erro percentual médio absoluto | Instável com valores próximos de zero |
| **sMAPE** | **MAPE simétrico** | **Métrica primária recomendada** |
| R²      | Coeficiente de determinação | Variância explicada |

---

## Referências Essenciais para o Artigo

- Box, G. E. P. et al. (2015). *Time Series Analysis* (5th ed.). Wiley. → SARIMAX
- Hochreiter & Schmidhuber (1997). LSTM. *Neural Computation, 9*(8). → LSTM
- Vaswani et al. (2017). Attention is all you need. *NeurIPS*. → Transformer
- Lim et al. (2021). TFT. *International Journal of Forecasting, 37*(4). → Transformer
- Bergmeir & Benítez (2012). CV for time series. *Information Sciences, 191*. → Metodologia
- Diebold & Mariano (1995). DM test. *JBES, 13*(3). → Comparação estatística
- Makridakis et al. (2022). M5 competition. *IJF*. → Benchmark
