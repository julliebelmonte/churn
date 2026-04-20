"""
engineer.py — feature engineering baseado em comportamento relativo

PRINCÍPIO:
O modelo não aprende regras.
Ele aprende DESVIOS.

Nada aqui deve ser uma decisão.
Tudo aqui deve ser uma representação.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constante nomeada para o teto da escala de satisfação.
# Se a fonte mudar a escala, há um único ponto a alterar.
SATISFACTION_MAX: int = 5


class FeatureEngineer:
    """
    Cria features contínuas e interpretáveis a partir do comportamento do cliente.

    Correções aplicadas:
    - SATISFACTION_MAX extraído como constante nomeada. Antes o valor 5 estava
      hardcoded em _interaction_features — se a escala da fonte mudar, a feature
      vira lixo sem nenhum erro.

    - get_feature_names() reescrito para derivar os nomes diretamente do
      DataFrame após o transform, em vez de manter uma lista manual desacoplada.
      Lista manual = drift silencioso toda vez que uma feature é adicionada ou
      removida.

    - Adicionada flag feat_support_silence para distinguir dois perfis que
      feat_support_pressure colapsava em zero:
        * cliente satisfeito sem contatos  (bom sinal)
        * cliente insatisfeito sem contatos (sinal de abandono silencioso)
      São comportamentos opostos — o modelo precisa vê-los separados.
    """

    FEATURE_PREFIX = "feat_"

    def __init__(self):
        self._pop_stats: dict = {}
        self._feature_names: list[str] = []
        self._fitted = False

    # =========================================================
    # FIT — aprende baseline da população (TRAIN ONLY)
    # =========================================================
    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        self._pop_stats = {
            "median_purchases_per_year": (
                df["Num_of_Purchases"] / df["Years_as_Customer"].clip(lower=1)
            ).median(),

            "p75_purchases":    df["Num_of_Purchases"].quantile(0.75),
            "median_recency":   df["Last_Purchase_Days_Ago"].median(),
            "median_spend":     df["Total_Spend"].median(),
            "median_support":   df["Num_of_Support_Contacts"].median(),
            "satisfaction_max": SATISFACTION_MAX,
        }

        self._fitted = True
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # =========================================================
    # TRANSFORM
    # =========================================================
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de transform().")

        df = df.copy()

        df = self._intensity_features(df)
        df = self._relative_features(df)
        df = self._anomaly_features(df)
        df = self._interaction_features(df)

        # Deriva os nomes das features a partir do que foi criado,
        # eliminando a lista manual propensa a drift.
        self._feature_names = [
            c for c in df.columns if c.startswith(self.FEATURE_PREFIX)
        ]

        return df

    # =========================================================
    # 1. Intensidade (volume normalizado)
    # =========================================================
    def _intensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        tenure    = df["Years_as_Customer"].clip(lower=1)
        purchases = df["Num_of_Purchases"].clip(lower=1)

        df["feat_spend_per_year"]       = df["Total_Spend"] / tenure
        df["feat_purchases_per_year"]   = purchases / tenure
        df["feat_return_rate"]          = (df["Num_of_Returns"] / purchases).clip(0, 1)
        df["feat_support_per_purchase"] = df["Num_of_Support_Contacts"] / purchases

        return df

    # =========================================================
    # 2. Features relativas à população
    # =========================================================
    def _relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["feat_purchases_vs_population"] = (
            df["feat_purchases_per_year"]
            / (self._pop_stats["median_purchases_per_year"] + 1e-6)
        )

        df["feat_spend_vs_population"] = (
            df["Total_Spend"]
            / (self._pop_stats["median_spend"] + 1e-6)
        )

        df["feat_recency_vs_population"] = (
            df["Last_Purchase_Days_Ago"]
            / (self._pop_stats["median_recency"] + 1e-6)
        )

        return df

    # =========================================================
    # 3. Anomaly de comportamento (CORE DO MODELO)
    # =========================================================
    def _anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Principal sinal de churn:
        "ausência relativa ao comportamento esperado do cliente"

        Normaliza recência pelo intervalo de compra esperado daquele
        cliente específico — não por um threshold fixo de negócio.
        Um cliente semanal com 14 dias ausente é mais anômalo que
        um cliente mensal com 14 dias ausente.
        """
        annual            = df["feat_purchases_per_year"].clip(lower=0.1)
        expected_interval = 365 / annual

        df["feat_recency_anomaly"] = (
            df["Last_Purchase_Days_Ago"] / expected_interval
        ).clip(0, 20)

        return df

    # =========================================================
    # 4. Interações simples (sem heurística de decisão)
    # =========================================================
    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        sat_max = self._pop_stats["satisfaction_max"]

        df["feat_satisfaction_gap"] = sat_max - df["Satisfaction_Score"]

        df["feat_value_risk_proxy"] = (
            df["Total_Spend"] * df["feat_recency_anomaly"]
        )

        df["feat_support_pressure"] = (
            df["Num_of_Support_Contacts"] * df["feat_satisfaction_gap"]
        )

        # Flag que distingue dois zeros de feat_support_pressure:
        #   - cliente satisfeito sem contatos  → bom sinal
        #   - cliente insatisfeito sem contatos → abandono silencioso
        # Sem essa flag, o modelo trata os dois perfis como idênticos.
        df["feat_support_silence"] = (
            (df["Num_of_Support_Contacts"] == 0) &
            (df["feat_satisfaction_gap"] > 0)
        ).astype(int)

        return df

    # =========================================================
    # util
    # =========================================================
    def get_feature_names(self) -> list[str]:
        """
        Retorna os nomes das features criadas no último transform().

        Antes: lista manual hardcoded — desacoplada do código real.
        Agora: derivada do DataFrame após o transform, sempre sincronizada.

        Raises RuntimeError se chamado antes de transform().
        """
        if not self._fitted:
            raise RuntimeError("Chame fit_transform() antes de get_feature_names().")
        return self._feature_names