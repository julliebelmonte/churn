# Churn Prediction — End-to-End ML Pipeline

Pipeline de machine learning para predição de probabilidade de churn em varejo online. A solução cobre todas as etapas de um projeto de ML em produção: extração, pré-processamento, feature engineering, treinamento, inferência, explicabilidade e monitoramento.

---

## Resultados
 
| Métrica | Valor |
|---|---|
| ROC-AUC | — |
| PR-AUC | — |
| Recall (churn) | — |
| F1-score (churn) | — |
| Captured Churn Rate @ 20% | — |
 
> Os valores são preenchidos após o treinamento. O Captured Churn Rate (CCR) é a métrica principal de negócio: de todos os clientes que foram churnar, qual % o modelo capturou dentro da capacidade operacional do time de retenção.
 
---

## 📂 Estrutura do Projeto

```
.
├── data/
│   ├── raw/
│   │   └── online_retail_customer_churn.csv
│   └── processed/
│
├── src/
│   ├── extraction/
│   │   └── extractor.py          # leitura, validação de schema, tipagem mínima
│   │
│   ├── preprocessing/
│   │   ├── window_builder.py     # recorte temporal (janelas de observação e predição)
│   │   ├── cleaner.py            # limpeza, imputação, encoding, split estratificado
│   │   └── segmentation.py       # segmentos comportamentais + risk score heurístico
│   │
│   ├── features/
│   │   └── engineer.py           # features contínuas baseadas em desvio comportamental
│   │
│   ├── training/
│   │   ├── model_config.py       # hiperparâmetros e configuração do experimento
│   │   ├── trainer.py            # orquestração do treino, early stopping, artefatos
│   │   └── evaluator.py          # métricas estatísticas, métricas de negócio, SHAP
│   │
│   ├── inference/
│   │   ├── schemas.py            # contratos de entrada e saída (CustomerInput, ChurnPrediction)
│   │   ├── predictor.py          # scoring batch e single
│   │   ├── explainer.py          # explicação SHAP on-demand por cliente
│   │   └── api.py                # FastAPI — endpoint REST de serving
│   │
│   ├── monitoring/
│   │   ├── drift_detector.py     # KS-test (data drift) + PSI no score (concept drift)
│   │   └── performance_tracker.py # degradação de ROC-AUC, PR-AUC e CCR com histórico
│   │
│   └── pipeline/
│       └── pipeline.py           # orquestrador: extraction → preprocessing → features
│
├── artifacts/                    # modelos serializados, métricas e feature importance
├── monitoring/
│   └── history/                  # snapshots de performance por janela temporal
├── tests/
│   └── test_preprocessing.py
│       
├── config.yaml
└── README.md
```
 
---
 

## Dataset

**Online Retail Customer Churn** — 1.000 clientes com 15 features comportamentais e transacionais.
 
| Coluna | Tipo | Descrição |
|---|---|---|
| `Customer_ID` | int | Identificador (removido antes do modelo) |
| `Age` | int | Idade do cliente |
| `Gender` | str | Gênero |
| `Annual_Income` | float | Renda anual estimada |
| `Total_Spend` | float | Gasto total histórico |
| `Years_as_Customer` | int | Tempo de relacionamento |
| `Num_of_Purchases` | int | Volume de compras |
| `Average_Transaction_Amount` | float | Ticket médio |
| `Num_of_Returns` | int | Volume de devoluções |
| `Num_of_Support_Contacts` | int | Contatos com suporte |
| `Satisfaction_Score` | int | Satisfação (1–5) |
| `Last_Purchase_Days_Ago` | int | Recência em dias |
| `Email_Opt_In` | bool | Aceite de comunicação |
| `Promotion_Response` | str | Resposta a promoções |
| `Target_Churn` | bool | **Target** — 1 = churnou |

## 📈 Diferenciais do Projeto

- Arquitetura modular e escalável
- Separação clara entre etapas do pipeline
- Uso de janelas temporais
- Monitoramento de drift integrado
- Estrutura pronta para produção