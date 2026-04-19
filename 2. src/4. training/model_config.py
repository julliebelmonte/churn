"""
model_config.py — configurações centralizadas de modelo e treinamento

Por que um arquivo de config separado?
    Hiperparâmetros hardcoded dentro do trainer são um problema de manutenção:
    qualquer experimento exige alterar código, não configuração. Centralizando
    aqui, o trainer fica agnóstico ao modelo — você troca LightGBM por XGBoost
    ou muda o threshold sem tocar na lógica de treinamento.

    Em produção, este dicionário pode ser substituído por um arquivo YAML
    carregado via Hydra ou MLflow, sem alterar o trainer.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# -------------------------------------------------------------------
# THRESHOLD DE CLASSIFICAÇÃO
# -------------------------------------------------------------------
# O threshold padrão de 0.5 é otimizado para acurácia, não para negócio.
# Em churn, falso negativo (não detectar quem vai sair) custa muito mais
# que falso positivo (acionar retenção para quem ficaria). Por isso,
# começamos com 0.35 e ajustamos via curva Precision-Recall no Evaluator.
DEFAULT_THRESHOLD: float = 0.35


# -------------------------------------------------------------------
# CONFIGURAÇÃO DO MODELO
# -------------------------------------------------------------------
@dataclass
class ModelConfig:
    """
    Configuração completa de um experimento de treinamento.

    Separar config de código permite:
    - Versionamento de experimentos (cada config = um run reproduzível)
    - Troca de modelo sem alterar o trainer
    - Integração futura com MLflow / Hydra sem refatoração
    """

    # Identificação do experimento
    experiment_name: str = "churn_lgbm_baseline"
    random_state:    int = 42

    # Threshold de decisão para classificação binária
    # Ajustar via find_optimal_threshold() no Evaluator antes de usar em prod
    threshold: float = DEFAULT_THRESHOLD

    # Hiperparâmetros do LightGBM
    # scale_pos_weight compensa o desbalanceamento de classes (churn ~10-20%).
    # O valor aqui é um ponto de partida — deve ser calibrado como
    # (n_negativos / n_positivos) no conjunto de treino real.
    lgbm_params: dict = field(default_factory=lambda: {
        "objective":        "binary",
        "metric":           "binary_logloss",
        "n_estimators":     500,
        "learning_rate":    0.05,
        "num_leaves":       31,
        "max_depth":        -1,
        "min_child_samples": 20,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "scale_pos_weight": 5,       # ajustar para n_neg / n_pos do treino
        "n_jobs":           -1,
        "verbose":          -1,
        "random_state":     42,
    })

    # Early stopping — evita overfitting sem busca manual de n_estimators
    early_stopping_rounds: int = 50

    # Colunas a excluir do X (não são features do modelo)
    # Prefixos de controle interno do pipeline + target
    non_feature_prefixes: tuple = ("_obs_", "_pred_")
    target_col:           str   = "Target_Churn"