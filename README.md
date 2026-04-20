# Churn Prediction — End-to-End ML Pipeline

Pipeline completo de machine learning para predição de probabilidade de churn em varejo online. A solução cobre todas as etapas de um projeto em produção: extração, pré-processamento, feature engineering, treinamento, inferência, explicabilidade e monitoramento de drift.

---

## Resultados
 
| Métrica | Valor |
|---|---|
| ROC-AUC | **0.581** |
| PR-AUC | **0.564** |
| Recall (churn) | **1.000** |
| F1-score (churn) | **0.649** |
| **Captured Churn Rate @ top-20%** | **25.0%** |
| Threshold ótimo (F-beta=2) | **0.510** |

> **Captured Churn Rate (CCR) é a métrica principal de negócio:** de todos os clientes que iriam churnar, o modelo capturou 25% deles usando apenas os 20% mais arriscados da base — respeitando a capacidade operacional do time de retenção.

> **Nota sobre o dataset:** a base sintética tem distribuição quase equilibrada (52.6% churn vs 47.4% retidos) e features com baixo poder discriminativo — diferenças entre churners e retidos chegam a apenas 3% em variáveis-chave. Em um dataset real com sinal mais forte, o modelo tende a performar muito acima desses números.
---

## Contexto de Negócio

### Definição do evento de Churn

O churn é definido por janelas temporais explícitas, separando o período de observação do período de predição:

```
|←————— 90 dias (observação) —————→|←— 30 dias (predição) —→|
         features são extraídas aqui        evento de churn acontece aqui
```

- **Janela de observação (90 dias):** período do qual as features comportamentais são extraídas. Captura pelo menos 3 ciclos mensais completos — suficiente para distinguir comportamento consistente de variação pontual.
- **Janela de predição (30 dias):** período futuro onde o evento de churn pode acontecer. Curto o suficiente para o time de retenção agir antes do cancelamento.

Essa separação é fundamental para evitar data leakage: o modelo nunca "vê" eventos do futuro durante o treino.

### Segmentação Comportamental

Clientes diferentes cancelam por motivos diferentes. O pipeline cria 3 segmentos automáticos que o modelo usa como features:

| Segmento | Critério | Insight de Negócio |
|---|---|---|
| `segment_tenure` | Tempo como cliente (new / early / mature / veteran) | Clientes novos cancelam por expectativa frustrada; veteranos por inércia quebrada |
| `segment_recency` | Dias desde última compra (active / warm / cold / dormant) | Recência é o sinal mais forte de abandono iminente |
| `segment_engagement` | Volume de compras (low / medium / high / vip) | Clientes VIP que param de comprar são o maior risco de receita |

Além dos segmentos, o pipeline calcula um `risk_score_heuristic` — pontuação de risco pré-modelagem que combina tenure curto, recência alta e baixo volume de compras. Serve como feature para o modelo e como fallback de regra de negócio quando o modelo não está disponível.

---

## Como Executar

### Pré-requisitos

```bash
# Python 3.9+
pip install lightgbm shap scikit-learn pandas numpy pyyaml pytest
```

### Estrutura esperada dos dados

Coloque o CSV em `data/raw/` antes de executar:

```
data/raw/online_retail_customer_churn.csv
```

### Executar o pipeline completo

```bash
python run_pipeline.py
```

Saída esperada:

```
06:13:37 | INFO | CHURN PREDICTION PIPELINE — INÍCIO
06:13:37 | INFO | Etapa 1/3 — Extração, pré-processamento e feature engineering
06:13:38 | INFO | Partições: train=800 | val=100 | test=100
06:13:38 | INFO | Features criadas (12): [...]
06:13:38 | INFO | Etapa 2/3 — Treinamento do modelo LightGBM
06:13:38 | INFO | RESULTADOS NO CONJUNTO DE TESTE
06:13:38 | INFO |   ROC-AUC              : 0.5815
06:13:38 | INFO |   Captured Churn Rate  : 25.00% (top-20% da base)
06:13:38 | INFO | TOP 10 FEATURES MAIS IMPORTANTES:
06:13:38 | INFO |   feat_recency_vs_population    ████████████ 23.3%
...
```

## Estrutura do Projeto

```
.
├── run_pipeline.py               # ← ENTRYPOINT: python run_pipeline.py
│
├── data/
│   └── raw/
│       └── online_retail_customer_churn.csv
│
├── src/
│   ├── extraction/
│   │   └── extractor.py          # leitura, validação de schema, tipagem mínima
│   │
│   ├── preprocessing/
│   │   ├── window_builder.py     # janelas de observação e predição
│   │   ├── cleaner.py            # limpeza, imputação, encoding, split estratificado
│   │   └── segmentation.py       # segmentos comportamentais + risk score heurístico
│   │
│   ├── features/
│   │   └── engineer.py           # features de desvio comportamental (feat_*)
│   │
│   ├── training/
│   │   ├── model_config.py       # hiperparâmetros e configuração do experimento
│   │   ├── trainer.py            # orquestração do treino, early stopping, artefatos
│   │   └── evaluator.py          # métricas estatísticas, de negócio e SHAP
│   │
│   ├── inference/
│   │   ├── schemas.py            # contratos de entrada/saída (CustomerInput, ChurnPrediction)
│   │   ├── predictor.py          # scoring batch e single
│   │   ├── explainer.py          # explicação SHAP on-demand por cliente
│   │   └── api.py                # FastAPI — endpoint REST de serving
│   │
│   ├── monitoring/
│   │   ├── drift_detector.py     # KS-test (data drift) + PSI (concept drift)
│   │   └── performance_tracker.py# degradação de ROC-AUC, PR-AUC e CCR
│   │
│   └── pipeline/
│       └── pipeline.py           # orquestrador: extraction → preprocessing → features
│
├── artifacts/                    # modelos, métricas e feature importance (gerados no treino)
├── 3. tests/
│   └── test_preprocessing.py
├── config.yaml
└── README.md
```
### Separação de responsabilidades

Cada módulo tem uma e apenas uma responsabilidade. Isso garante testabilidade e facilita manutenção:

| Módulo | O que faz | O que NÃO faz |
|---|---|---|
| `extractor.py` | Lê, valida schema, faz tipagem mínima | Não limpa, não transforma |
| `window_builder.py` | Cria colunas de janela temporal | Não faz encoding nem imputação |
| `cleaner.py` | Imputa, encoda, faz split | Não cria features novas |
| `segmentation.py` | Cria segmentos e risk score | Não limpa nem modela |
| `engineer.py` | Cria features `feat_*` | Não acessa dados brutos sem passar pelo cleaner |
| `trainer.py` | Orquestra treino e persiste artefatos | Não faz preprocessing |
| `evaluator.py` | Calcula métricas e SHAP | Não treina, não prediz |
| `predictor.py` | Serving batch e single | Não retreina |
| `drift_detector.py` | Detecta data drift e concept drift | Não retreina, não alerta diretamente |

---

## Feature Engineering

O princípio central é: **o modelo não aprende regras, aprende desvios**.

Em vez de usar valores absolutos (ex: "comprou 5 vezes"), o pipeline cria features relativas ao comportamento esperado de cada cliente e à população. Isso torna o modelo robusto a diferentes perfis de uso.

### Features criadas (prefixo `feat_`)

| Feature | Fórmula | Por que importa |
|---|---|---|
| `feat_recency_vs_population` | `recência_cliente / mediana_população` | **Mais importante (23.3%)** — cliente comprando mais tarde que a base é sinal forte |
| `feat_purchases_vs_population` | `compras_por_ano / mediana_população` | **2° lugar (20%)** — queda relativa de engajamento |
| `feat_support_per_purchase` | `n_suporte / n_compras` | **3° lugar (13.3%)** — fricção por transação |
| `feat_spend_per_year` | `gasto_total / anos_como_cliente` | Intensidade de gasto normalizada pelo tempo |
| `feat_return_rate` | `devoluções / compras` | Taxa de insatisfação com produto |
| `feat_recency_anomaly` | `dias_sem_compra / intervalo_esperado` | **CORE:** um cliente semanal com 14 dias parado é muito mais anômalo que um mensal |
| `feat_satisfaction_gap` | `5 - satisfaction_score` | Distância do máximo de satisfação |
| `feat_value_risk_proxy` | `gasto_total × recency_anomaly` | Clientes de alto valor em risco = maior impacto financeiro |
| `feat_support_pressure` | `n_suporte × satisfaction_gap` | Insatisfação ativa: reclama E está insatisfeito |
| `feat_support_silence` | `flag: suporte=0 E gap>0` | **Abandono silencioso:** insatisfeito mas não reclama — sinal de desistência |

> `feat_support_silence` é uma feature de design cuidadoso: sem ela, um cliente satisfeito sem contatos e um insatisfeito sem contatos geram o mesmo valor (`feat_support_pressure = 0`). São comportamentos opostos — o modelo precisa vê-los separados.

---

## Métricas de Negócio

### Por que não usar apenas ROC-AUC?

ROC-AUC mede discriminação global. Em churn, o problema real é diferente: **o time de retenção não consegue acionar todos os clientes**. Se a capacidade operacional é de 200 contatos por mês em uma base de 10.000 clientes, o modelo precisa ser bom no top-2% — não em toda a distribuição. ROC-AUC não captura isso.

### Captured Churn Rate (CCR) — a métrica que a área entende

```
CCR = churners capturados no top-N% / total de churners reais
```

**Exemplo concreto com os resultados do modelo:**
- Base de teste: 100 clientes, 48 são churners reais
- Capacidade do CRM: top-20% = 20 clientes para acionar
- CCR = 25% → o modelo colocou 12 dos 48 churners reais nesses 20 slots

**Como apresentar para o gerente:** "A cada R$100k em receita em risco de churn, o modelo consegue que o time de retenção salve R$25k — usando apenas 20% da capacidade disponível."

### Tiers de risco para o CRM

| Tier | Score | Ação recomendada |
|---|---|---|
| `high` | ≥ 0.65 | Acionar imediatamente — ligação ou oferta personalizada |
| `medium` | 0.35–0.65 | Fila de retenção — e-mail ou push com incentivo |
| `low` | < 0.35 | Monitorar — sem ação imediata |

---

## Explicabilidade (SHAP)

Quando a área de negócios precisa entender **por que um cliente específico foi classificado como alto risco**, o pipeline usa SHAP values — não feature importance global.

**Feature importance global** diz quais features são importantes para o modelo em média.
**SHAP** diz por que *este* cliente foi classificado desta forma — que é o que o analista de retenção precisa para construir a abordagem.

### Exemplo de output SHAP

```
Cliente #361 | churn_score: 0.54 | tier: medium

feature                     shap_value   direção
feat_support_per_purchase     +0.035      risco      ← muitos contatos de suporte por compra
feat_spend_per_year           +0.030      risco      ← gasto anual elevado (alto impacto)
feat_value_risk_proxy         +0.007      risco      ← cliente de valor em anomalia de recência
feat_recency_vs_population    -0.004      proteção   ← recência ainda dentro da média
```

**Tradução para o CRM:** "Cliente de alto ticket com histórico de fricção no suporte. Abordagem recomendada: proativa, focada em resolver problemas anteriores antes de fazer oferta de retenção."

---
 
 ## Resiliência a Edge Cases

### Caso: cliente VIP que some por férias

Um cliente VIP que compra semanalmente e fica 3 semanas sem comprar vai disparar `feat_recency_anomaly = 3.0` — 3x o intervalo esperado. Isso aumenta o score de churn, mas não necessariamente é churn real.

Como o pipeline lida com isso:

1. **`_no_purchase_in_window`**: flag binária que indica ausência total na janela de 90 dias. Férias de 3 semanas em uma janela de 90 dias não ativa essa flag.

2. **`feat_recency_anomaly` é normalizado pelo comportamento do próprio cliente**, não por threshold fixo. O modelo aprende que um desvio de 3x é menos raro para clientes com histórico irregular.

3. **`feat_purchases_vs_population`**: se o cliente tem histórico VIP sólido, essa feature fica alta mesmo com a pausa — dando sinal de proteção ao modelo.

4. **Regra de negócio recomendada em produção**: criar uma feature `days_since_last_high_value_event` que captura a última compra acima do ticket médio do cliente, não a última compra qualquer. Isso diferencia "sumiu de vez" de "está em pausa".

---

## Arquitetura de Deploy

### Serving recomendado para produção

```
┌─────────────────────────────────────────────────────────────┐
│                     DUAS MODALIDADES                        │
│                                                             │
│  BATCH (diário via Airflow/cron)                            │
│  └─ ChurnPredictor.score_batch()                            │
│     → scores para toda a base                               │
│     → output: CSV/tabela com score + tier + timestamp       │
│     → consumido pelo CRM para fila de retenção do dia       │
│                                                             │
│  SINGLE (on-demand via API REST)                            │
│  └─ FastAPI → ChurnPredictor.score_one()                    │
│     → score para um cliente específico em < 200ms           │
│     → consumido pelo atendente durante ligação no CRM       │
│     → endpoint /explain retorna SHAP para aquele cliente    │
└─────────────────────────────────────────────────────────────┘
```

### Endpoints da API (FastAPI)

```
POST /predict          → score individual (CustomerInput → ChurnPrediction)
POST /predict/batch    → scoring em batch (CSV → BatchPredictionResult)
GET  /explain/{id}     → SHAP on-demand para um cliente específico
GET  /health           → healthcheck para load balancer
```

**Por que duas modalidades e não só API?** Batch é otimizado para throughput (10k+ clientes de uma vez, aceita latência de minutos). Single é otimizado para latência (um cliente, resposta em < 200ms durante atendimento). Usar batch para on-demand ou API para processar toda a base seriam escolhas ineficientes.

---

## Monitoramento (MLOps)

### Diferenciando Data Drift de Concept Drift

Esta é a distinção mais importante para não reagir errado a um alerta:

| Situação | Diagnóstico | Ação |
|---|---|---|
| Features deslocadas, CCR estável | **Data Drift** (ex: sazonalidade de natal) | Monitorar. Pode ser temporário. Retreinar preventivo. |
| Features estáveis, CCR caiu | **Concept Drift** (ex: crise econômica muda perfil de cancelamento) | Retreinar com dados recentes. Urgente. |
| Features deslocadas, CCR caiu | **Ambos** | Retreinar urgente + investigar mudança na base. |
| Features e CCR estáveis | **OK** | Continuar monitorando. |

### Como é detectado

```python
# Data Drift — KS-test por feature
# KS statistic > 0.10 → feature com distribuição diferente do treino
for cada feature:
    ks_stat = kolmogorov_smirnov(feature_treino, feature_atual)
    se ks_stat > 0.10 → alerta de data drift

# Concept Drift — PSI no score de saída do modelo
# PSI < 0.10 → estável
# PSI 0.10–0.20 → mudança moderada, investigar
# PSI > 0.20 → instabilidade severa, retreinar
psi = population_stability_index(scores_treino, scores_atuais)

# CCR real (disponível T+30 dias após scoring)
# Queda > 8 pp → concept drift confirmado
ccr_drop = ccr_baseline - ccr_atual
```

### Schedule de monitoramento

```
Diário:    PSI nos scores + KS nas features → alerta automático no Slack/PagerDuty
Semanal:   CCR real (com labels T+7) → relatório para o time de retenção
Mensal:    Revisão completa de métricas + decisão de retreino
```

---

## Testes

```bash
pytest "3. tests/test_preprocessing.py" -v
```

### Cobertura atual

| Classe | Testes | Status |
|---|---|---|
| `DataCleaner` | Remove identificadores, binariza target, sem colunas objeto, split consistente, requer fit | 4/5 ✅ |
| `WindowBuilder` | Adiciona metadata de janela, cria flag de ausência, não modifica input | 3/3 ✅ |
| `CustomerSegmenter` | Cria segmentos, lógica de tenure, ordenação de risk score, sem nulls | 4/4 ✅ |
| `Integração` | Pipeline completo roda, sem nulls após processamento | 2/2 ✅ |

**13/14 passando.**

### Falha conhecida e correção

**`test_split_is_consistent`** usa `Index.isdisjoint()`, removido no pandas 2.x.

```python
# Antes (quebrado no pandas 2.x)
assert train.index.isdisjoint(test.index)

# Depois (compatível)
assert len(set(train.index) & set(test.index)) == 0
```

---

## Decisões de Design

### Por que LightGBM?

- Nativo em dados tabulares desbalanceados (`scale_pos_weight` automático)
- Early stopping integrado sem configuração extra
- SHAP values nativos — explicabilidade sem custo adicional de infraestrutura
- Suporta missing values nativamente (crítico em produção onde dados chegam incompletos)

### Por que threshold 0.35 como padrão (e não 0.5)?

Em churn, os custos são assimétricos:
- **Falso Negativo** (não detectar quem vai sair): perde o MRR do cliente para sempre
- **Falso Positivo** (abordar quem ficaria): custo do incentivo de retenção (~R$20–50)

Com ratio FN/FP de 5:1, o threshold ótimo é mais baixo que 0.5. O `Evaluator` encontra o threshold ótimo automaticamente via curva Precision-Recall maximizando F-beta com beta=2 (recall vale o dobro da precision).

### Por que PR-AUC além de ROC-AUC?

ROC-AUC é insensível ao desbalanceamento. Um modelo que classifica tudo como negativo pode ter ROC-AUC alto quando churn < 10%. PR-AUC foca na classe positiva — exatamente o que importa para Retenção/CRM.

### Por que split 80/10/10 e não 70/15/15?

Com 1.000 registros, reduzir o treino para 700 prejudica mais o aprendizado do que ganhar 50 exemplos extras em validação. O val set (100) é suficiente para otimizar threshold e fazer early stopping. O test set (100) é suficiente para estimativa não enviesada de performance.

---

## Próximos Passos

Em um projeto com dados reais e mais tempo, as evoluções naturais seriam:

1. **Calibração de probabilidade** com `CalibratedClassifierCV(method='isotonic')` — garante que score 0.7 significa realmente 70% de chance de churn
2. **Features de sequência temporal** — variação de comportamento mês a mês, não só snapshot
3. **Modelo por segmento** — um LightGBM para clientes "new", outro para "veteran", já que os padrões de churn são diferentes
4. **MLflow** para rastreamento de experimentos e versionamento de modelos
5. **Feature store** para reaproveitar features calculadas entre pipelines de retenção, oferta e engajamento

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
