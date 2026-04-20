import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataCleaner:
    """
    Limpeza e preparação estrutural do dataset.
    NÃO faz feature engineering nem segmentação.
    """

    def __init__(self):
        self._numeric_medians = None
        self._fitted = False

    # ----------------------------
    # FIT
    # ----------------------------
    def fit(self, df: pd.DataFrame):
        df = df.copy()

        numeric_cols = df.select_dtypes(include=np.number).columns
        self._numeric_medians = df[numeric_cols].median()
        self._fitted = True

        return self

    # ----------------------------
    # TRANSFORM
    # ----------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        df = df.copy()

        # Remove identificadores
        if "Customer_ID" in df.columns:
            df = df.drop(columns=["Customer_ID"])

        # Target binarization segura
        if "Target_Churn" in df.columns:
            df["Target_Churn"] = df["Target_Churn"].astype(int)

        # Imputação numérica baseada no TREINO
        for col, median in self._numeric_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)

        # Encoding simples de bool → int
        bool_cols = df.select_dtypes(include="bool").columns
        for col in bool_cols:
            df[col] = df[col].astype(int)

        # Encoding categórico leve (one-hot simples)
        cat_cols = df.select_dtypes(include="object").columns
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        return df

    # ----------------------------
    # FIT + TRANSFORM
    # ----------------------------
    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    # ----------------------------
    # SPLIT
    # ----------------------------
    def split(self, df: pd.DataFrame, test_size=0.2, val_size=0.1):
        df = df.copy()

        train, temp = train_test_split(df, test_size=test_size, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        return train, val, test