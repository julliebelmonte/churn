import pandas as pd


class WindowConfig:
    def __init__(self, observation_days=90, prediction_days=30):
        self.observation_days = observation_days
        self.prediction_days = prediction_days

    def validate(self):
        if self.observation_days < 30:
            raise ValueError("observation_days must be >= 30")

        if not (7 <= self.prediction_days <= 90):
            raise ValueError("prediction_days must be between 7 and 90")


class WindowBuilder:
    """
    Cria recorte temporal do dataset.
    NÃO faz limpeza nem encoding.
    """

    def __init__(self, config: WindowConfig):
        config.validate()
        self.config = config

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        obs = self.config.observation_days

        # metadata de janela
        df["_obs_window_days"] = obs
        df["_pred_window_days"] = self.config.prediction_days

        # flag crítica: cliente sem evento na janela
        if "Last_Purchase_Days_Ago" in df.columns:
            df["_no_purchase_in_window"] = (
                df["Last_Purchase_Days_Ago"] > obs
            ).astype(int)

        # recência truncada (evita outliers extremos)
        if "Last_Purchase_Days_Ago" in df.columns:
            df["_recency_clipped"] = df["Last_Purchase_Days_Ago"].clip(0, obs)

        return df