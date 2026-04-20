"""
pipeline.py — orquestrador do pipeline de pré-processamento + feature engineering

Responsabilidade:
    Conectar as camadas em ordem correta e expor uma interface única
    para training/ e inference/.

Ordem de execução:
    1. WindowBuilder  — recorte temporal (cria colunas de janela)
    2. DataCleaner    — limpeza, imputação, encoding
    3. CustomerSegmenter — segmentos comportamentais + risk score
    4. FeatureEngineer   — features contínuas e relativas

Por que essa ordem?
    WindowBuilder deve ser o primeiro porque cria colunas (_recency_clipped,
    _no_purchase_in_window) que o DataCleaner vai processar (imputar e encodar).
    Inverter Cleaner e Window significa que o cleaner opera sobre dados sem
    as colunas de janela, e o window opera sobre dados já transformados —
    quebrando a semântica de ambos.

    FeatureEngineer vem por último porque depende de colunas brutas originais
    (Total_Spend, Num_of_Purchases etc.) e de colunas criadas pelo segmenter
    (risk_score_heuristic). Ele nunca deve rodar antes da limpeza.

Correções em relação à versão original:
    - Removido código solto no escopo do módulo (fe = FeatureEngineer(),
      df = pipeline.fit_transform(...)) que causava NameError na importação.
    - Corrigida a ordem: estava Cleaner → Window → Segmenter.
      Correto: Window → Cleaner → Segmenter → FeatureEngineer.
    - FeatureEngineer integrado à classe — antes ficava de fora da pipeline.
    - WindowBuilder movido para __init__ com configuração padrão e aceito
      como argumento opcional em fit/transform para manter flexibilidade.
"""

from __future__ import annotations

import pandas as pd

from src.extraction.extractor import DataExtractor
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.window_builder import WindowBuilder, WindowConfig
from src.preprocessing.segmentation import CustomerSegmenter
from src.features.engineer import FeatureEngineer


class ChurnPipeline:
    """
    Pipeline completo de churn: extração → preprocessing → features.

    Uso básico
    ----------
    >>> pipeline = ChurnPipeline()
    >>> train, val, test = pipeline.fit("data/churn.csv")
    >>> df_inference = pipeline.transform("data/new_batch.csv")

    Uso com configuração de janela customizada
    ------------------------------------------
    >>> config = WindowConfig(observation_days=60, prediction_days=30)
    >>> pipeline = ChurnPipeline(window_config=config)
    >>> train, val, test = pipeline.fit("data/churn.csv")
    """

    def __init__(self, window_config: WindowConfig | None = None):
        self.window_builder  = WindowBuilder(window_config or WindowConfig())
        self.cleaner         = DataCleaner()
        self.segmenter       = CustomerSegmenter()
        self.feature_engineer = FeatureEngineer()
        self._fitted = False

    # ---------------------------------------------------
    # FIT — recebe caminho do arquivo, devolve partições
    # ---------------------------------------------------
    def fit(
        self,
        source: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Lê, aprende os parâmetros do treino e devolve train / val / test
        já transformados e com features.

        Parâmetros
        ----------
        source : str
            Caminho para o arquivo CSV de treino.

        Retorna
        -------
        train, val, test : pd.DataFrame
            Partições prontas para o training/.
        """
        df_raw = DataExtractor(source).extract()

        # Split ANTES de qualquer fit para garantir zero data leakage.
        # O cleaner aprende mediana só do treino bruto.
        train_raw, val_raw, test_raw = DataCleaner().split(df_raw)

        # Fit de todos os transformers no treino bruto
        self.cleaner.fit(train_raw)
        self.segmenter.fit(train_raw)

        # Transform das três partições
        train = self._transform_partition(train_raw)
        val   = self._transform_partition(val_raw)
        test  = self._transform_partition(test_raw)

        # FeatureEngineer aprende apenas no treino transformado
        self.feature_engineer.fit(train)
        train = self.feature_engineer.transform(train)
        val   = self.feature_engineer.transform(val)
        test  = self.feature_engineer.transform(test)

        self._fitted = True
        return train, val, test

    # ---------------------------------------------------
    # TRANSFORM — para inferência em produção
    # ---------------------------------------------------
    def transform(self, source: str) -> pd.DataFrame:
        """
        Aplica o pipeline treinado sobre um novo batch de dados.
        Não aprende nada — usa apenas os parâmetros do fit().

        Parâmetros
        ----------
        source : str
            Caminho para o arquivo CSV do batch de inferência.
        """
        if not self._fitted:
            raise RuntimeError(
                "Pipeline não treinado. Chame fit() antes de transform()."
            )

        df_raw = DataExtractor(source).extract()
        df     = self._transform_partition(df_raw)
        df     = self.feature_engineer.transform(df)
        return df

    # ---------------------------------------------------
    # INTERNO — aplica preprocessing em uma partição
    # ---------------------------------------------------
    def _transform_partition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ordem canônica: Window → Cleaner → Segmenter.
        Nunca chama fit() — só transform().
        """
        df = self.window_builder.apply(df)
        df = self.cleaner.transform(df)
        df = self.segmenter.transform(df)
        return df

    # ---------------------------------------------------
    # UTIL
    # ---------------------------------------------------
    def get_feature_names(self) -> list[str]:
        """Delega para o FeatureEngineer — fonte única de verdade."""
        return self.feature_engineer.get_feature_names()