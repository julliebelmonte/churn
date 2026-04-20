import pandas as pd
import numpy as np


class CustomerSegmenter:
    """
    Segmentação comportamental.
    NÃO faz limpeza nem transformação estrutural.
    """

    def __init__(self):
        self.fitted = False

    # ----------------------------
    # FIT
    # ----------------------------
    def fit(self, df: pd.DataFrame):
        self.fitted = True
        return self

    # ----------------------------
    # TRANSFORM
    # ----------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ----------------------------
        # TENURE SEGMENT
        # ----------------------------
        if "Years_as_Customer" in df.columns:
            df["segment_tenure"] = pd.cut(
                df["Years_as_Customer"],
                bins=[0, 2, 5, 10, 999],
                labels=["new", "early", "mature", "veteran"]
            )

        # ----------------------------
        # RECENCY SEGMENT
        # ----------------------------
        if "Last_Purchase_Days_Ago" in df.columns:
            df["segment_recency"] = pd.cut(
                df["Last_Purchase_Days_Ago"],
                bins=[0, 30, 90, 180, 999],
                labels=["active", "warm", "cold", "dormant"]
            )

        # ----------------------------
        # ENGAGEMENT SEGMENT
        # ----------------------------
        if "Num_of_Purchases" in df.columns:
            df["segment_engagement"] = pd.cut(
                df["Num_of_Purchases"],
                bins=[0, 10, 30, 70, 999],
                labels=["low", "medium", "high", "vip"]
            )

        # ----------------------------
        # RISK SCORE HEURÍSTICO
        # ----------------------------
        df["risk_score_heuristic"] = self._compute_risk(df)

        # ----------------------------
        # ENCODING SIMPLES
        # ----------------------------
        self._encode(df)

        return df

    # ----------------------------
    # FIT + TRANSFORM
    # ----------------------------
    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    # ----------------------------
    # RISK MODEL (heurístico simples e consistente)
    # ----------------------------
    def _compute_risk(self, df):
        score = np.zeros(len(df))

        if "Years_as_Customer" in df:
            score += (df["Years_as_Customer"] <= 2).astype(int) * 0.4

        if "Last_Purchase_Days_Ago" in df:
            score += (df["Last_Purchase_Days_Ago"] > 180).astype(int) * 0.4

        if "Num_of_Purchases" in df:
            score += (df["Num_of_Purchases"] < 10).astype(int) * 0.2

        return score

    # ----------------------------
    # ENCODING INTERNO
    # ----------------------------
    def _encode(self, df):
        enc_map = {
            "segment_tenure": {"new": 0, "early": 1, "mature": 2, "veteran": 3},
            "segment_recency": {"active": 0, "warm": 1, "cold": 2, "dormant": 3},
            "segment_engagement": {"low": 0, "medium": 1, "high": 2, "vip": 3},
        }

        for col, mapping in enc_map.items():
            if col in df.columns:
                df[col + "_enc"] = df[col].map(mapping).fillna(-1)