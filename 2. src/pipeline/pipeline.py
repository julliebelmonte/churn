import pandas as pd

from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.window_builder import WindowBuilder
from src.preprocessing.segmentation import CustomerSegmenter
from src.features.engineer import FeatureEngineer

fe = FeatureEngineer()

df = pipeline.fit_transform(df, window)
df = fe.fit_transform(df)

class ChurnPreprocessingPipeline:
    """
    Pipeline único de pré-processamento.

    Responsabilidade:
        Orquestrar preprocessing SEM misturar responsabilidades internas.

    Ordem fixa:
        1. Cleaner
        2. Window Builder
        3. Segmenter
    """

    def __init__(self):
        self.cleaner = DataCleaner()
        self.segmenter = CustomerSegmenter()
        self.window_builder = None  # configurado no fit

    # ---------------------------------------------------
    # FIT
    # ---------------------------------------------------
    def fit(self, df: pd.DataFrame):
        """
        Fit apenas no cleaner (único estado real necessário).
        """
        self.cleaner.fit(df)
        self.segmenter.fit(df)
        return self

    # ---------------------------------------------------
    # TRANSFORM
    # ---------------------------------------------------
    def transform(self, df: pd.DataFrame, window_builder):
        """
        Aplica pipeline completo.
        WindowBuilder é stateless → passado como argumento.
        """

        # 1. CLEAN
        df = self.cleaner.transform(df)

        # 2. WINDOW
        df = window_builder.apply(df)

        # 3. SEGMENTATION
        df = self.segmenter.transform(df)

        return df

    # ---------------------------------------------------
    # FIT + TRANSFORM
    # ---------------------------------------------------
    def fit_transform(self, df: pd.DataFrame, window_builder):
        self.fit(df)
        return self.transform(df, window_builder)