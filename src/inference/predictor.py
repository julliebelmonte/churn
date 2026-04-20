"""
predictor.py — serving de predições de churn

Responsabilidade:
    Carregar o trainer serializado e expor uma interface limpa para
    dois modos de uso distintos:
        - Batch scoring: processa um CSV inteiro (cron diário / Airflow)
        - Single scoring: pontua um cliente individual (API REST)

Por que dois modos?
    Batch é o caso principal: a área de retenção recebe a lista diária
    de clientes em risco para planejar as ações da semana. É otimizado
    para throughput — processa milhares de linhas de uma vez.

    Single é para casos on-demand: o atendente abre o CRM, vê um cliente
    específico e quer saber o score e a explicação daquele momento.
    É otimizado para latência — responde em < 200ms.

Separação de responsabilidades:
    predictor.py  → score (0–1) e tier (low/medium/high)
    explainer.py  → por que esse score (SHAP, on-demand)
    schemas.py    → contratos de entrada e saída

Serving architecture:
    Batch  → ChurnPredictor.score_batch()  ← chamado pelo Airflow/cron
    Single → ChurnPredictor.score_one()    ← chamado pela API REST (FastAPI)
"""

from __future__ import annotations

import logging
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from src.training.trainer import ChurnTrainer
from src.pipeline.pipeline import ChurnPipeline
from src.inference.schemas import (
    CustomerInput,
    ChurnPrediction,
    BatchPredictionResult,
)

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Interface de serving do modelo de churn.

    Uso batch (cron/Airflow)
    ------------------------
    >>> predictor = ChurnPredictor.load("artifacts/trainer.pkl")
    >>> result = predictor.score_batch("data/batch_hoje.csv")
    >>> df = pd.DataFrame(result.to_records())
    >>> df.to_csv("outputs/scores_hoje.csv", index=False)

    Uso single (API REST)
    ---------------------
    >>> predictor = ChurnPredictor.load("artifacts/trainer.pkl")
    >>> prediction = predictor.score_one(customer_input)
    """

    def __init__(
        self,
        trainer:  ChurnTrainer,
        pipeline: ChurnPipeline,
        model_version: str = "unknown",
    ):
        self.trainer       = trainer
        self.pipeline      = pipeline
        self.model_version = model_version
        self._threshold    = trainer.evaluator.threshold

    # -------------------------------------------------------------------
    # FACTORY — carrega de artefatos salvos
    # -------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        trainer_path:  str | Path,
        pipeline_path: str | Path | None = None,
        model_version: str = "unknown",
    ) -> "ChurnPredictor":
        """
        Carrega predictor a partir dos artefatos de treinamento.

        Parâmetros
        ----------
        trainer_path : str | Path
            Caminho para o trainer.pkl salvo pelo ChurnTrainer.
        pipeline_path : str | Path | None
            Caminho para o pipeline serializado. Se None, cria um novo
            (usado quando o pipeline é stateless entre runs).
        model_version : str
            Identificador do modelo para rastreabilidade nas predições.
        """
        trainer = ChurnTrainer.load(trainer_path)

        if pipeline_path:
            import pickle
            with open(pipeline_path, "rb") as f:
                pipeline = pickle.load(f)
        else:
            # Pipeline sem estado salvo — será usado apenas para transform()
            # O fit() já foi feito durante o treino; aqui só serve inferência
            pipeline = ChurnPipeline()

        logger.info(f"ChurnPredictor carregado | versão={model_version}")
        return cls(trainer, pipeline, model_version)

    # -------------------------------------------------------------------
    # BATCH SCORING — throughput otimizado
    # -------------------------------------------------------------------
    def score_batch(self, source: str) -> BatchPredictionResult:
        """
        Pontua todos os clientes de um arquivo CSV.

        Ordem:
            1. Pipeline.transform() — preprocessing + features
            2. Trainer.predict_proba() — score individual
            3. Empacota em BatchPredictionResult com metadados

        Parâmetros
        ----------
        source : str
            Caminho para o CSV do batch. Deve ter as colunas do extractor.
        """
        logger.info(f"Iniciando batch scoring | source={source}")

        df_processed = self.pipeline.transform(source)
        scores       = self.trainer.predict_proba(df_processed)

        # Recupera customer_id se existir no processed (pode ter sido dropado)
        # Usa índice como fallback
        ids = (
            df_processed["Customer_ID"].tolist()
            if "Customer_ID" in df_processed.columns
            else list(df_processed.index)
        )

        predictions = [
            ChurnPrediction.from_score(
                customer_id   = cid,
                score         = float(score),
                threshold     = self._threshold,
                model_version = self.model_version,
            )
            for cid, score in zip(ids, scores)
        ]

        total_high = sum(1 for p in predictions if p.risk_tier == "high")

        result = BatchPredictionResult(
            predictions     = predictions,
            total_scored    = len(predictions),
            total_high_risk = total_high,
            model_version   = self.model_version,
            scored_at       = datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            f"Batch concluído | total={result.total_scored} "
            f"high_risk={total_high} ({result.high_risk_rate:.1%})"
        )
        return result

    # -------------------------------------------------------------------
    # SINGLE SCORING — latência otimizada
    # -------------------------------------------------------------------
    def score_one(self, customer: CustomerInput) -> ChurnPrediction:
        """
        Pontua um único cliente.

        Converte o CustomerInput em DataFrame de uma linha,
        aplica o pipeline e retorna a predição.

        Raises
        ------
        ValueError
            Se o CustomerInput falhar na validação de domínio.
        """
        customer.validate()

        df = self._input_to_dataframe(customer)
        df_processed = self._transform_single(df)
        score = float(self.trainer.predict_proba(df_processed)[0])

        return ChurnPrediction.from_score(
            customer_id   = customer.customer_id,
            score         = score,
            threshold     = self._threshold,
            model_version = self.model_version,
        )

    # -------------------------------------------------------------------
    # INTERNOS
    # -------------------------------------------------------------------
    def _input_to_dataframe(self, customer: CustomerInput) -> pd.DataFrame:
        """Converte CustomerInput em DataFrame de uma linha."""
        return pd.DataFrame([{
            "Customer_ID":                customer.customer_id,
            "Age":                        customer.age,
            "Gender":                     customer.gender,
            "Annual_Income":              customer.annual_income,
            "Total_Spend":                customer.total_spend,
            "Years_as_Customer":          customer.years_as_customer,
            "Num_of_Purchases":           customer.num_of_purchases,
            "Average_Transaction_Amount": customer.average_transaction_amount,
            "Num_of_Returns":             customer.num_of_returns,
            "Num_of_Support_Contacts":    customer.num_of_support_contacts,
            "Satisfaction_Score":         customer.satisfaction_score,
            "Last_Purchase_Days_Ago":     customer.last_purchase_days_ago,
            "Email_Opt_In":               customer.email_opt_in,
            "Promotion_Response":         customer.promotion_response,
        }])

    def _transform_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica pipeline em um DataFrame de uma linha.

        Chama diretamente os transformers internos do pipeline
        (sem re-fit) para garantir consistência com o treino.
        """
        df = self.pipeline.window_builder.apply(df)
        df = self.pipeline.cleaner.transform(df)
        df = self.pipeline.segmenter.transform(df)
        df = self.pipeline.feature_engineer.transform(df)
        return df
    
class Predictor:
    def __init__(self):
        self.model = joblib.load("model/trainer.pkl")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]