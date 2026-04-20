"""
explainer.py — explicação SHAP on-demand por cliente

Responsabilidade:
    Gerar explicações individuais de por que um cliente foi classificado
    como alto risco. Separado do predictor porque:
        - SHAP é computacionalmente mais caro que scoring
        - Nem toda chamada de scoring precisa de explicação
        - A área de negócios consome scoring em batch mas explicação on-demand

Quando usar:
    - Atendente abre ficha do cliente no CRM e clica "Por que alto risco?"
    - Gestão questiona um caso específico
    - Debug de predições inesperadas

Quando NÃO usar:
    - Batch scoring diário (usar predictor.score_batch())
    - Monitoramento de drift (usar drift_detector.py)
"""

from __future__ import annotations

import logging
import pandas as pd

from src.training.trainer import ChurnTrainer
from src.inference.predictor import ChurnPredictor
from src.inference.schemas import CustomerInput, ExplanationOutput

logger = logging.getLogger(__name__)


class ChurnExplainer:
    """
    Explicação SHAP on-demand para clientes individuais.

    Uso
    ---
    >>> explainer = ChurnExplainer(predictor)
    >>> output = explainer.explain(customer_input, top_n=5)
    >>> print(output.to_dict())
    """

    def __init__(self, predictor: ChurnPredictor):
        self.predictor = predictor

    # -------------------------------------------------------------------
    # INTERFACE PÚBLICA
    # -------------------------------------------------------------------
    def explain(
        self,
        customer: CustomerInput,
        top_n:    int = 5,
    ) -> ExplanationOutput:
        """
        Retorna os top-N fatores de risco para um cliente específico.

        Fluxo:
            1. score_one() — obtém o score (sem duplicar lógica)
            2. _transform_for_shap() — prepara X para o TreeExplainer
            3. trainer.explain_customer() — calcula SHAP values
            4. Empacota em ExplanationOutput

        Parâmetros
        ----------
        customer : CustomerInput
            Dados brutos do cliente a explicar.
        top_n : int
            Número de fatores a retornar. Default 5.
        """
        customer.validate()

        # Score primeiro — reutiliza a lógica do predictor
        prediction = self.predictor.score_one(customer)

        # Prepara X no mesmo formato que o trainer espera
        df_raw       = self.predictor._input_to_dataframe(customer)
        df_processed = self.predictor._transform_single(df_raw)

        feature_names = self.predictor.trainer.get_feature_names()
        X_customer    = df_processed[feature_names]

        # SHAP via trainer (delega para evaluator.explain_customer)
        shap_df = self.predictor.trainer.explain_customer(
            X_customer, top_n=top_n
        )

        top_factors = shap_df.to_dict(orient="records")

        logger.info(
            f"Explicação gerada | customer_id={customer.customer_id} "
            f"score={prediction.churn_score:.3f} top_n={top_n}"
        )

        return ExplanationOutput(
            customer_id   = customer.customer_id,
            churn_score   = prediction.churn_score,
            top_factors   = top_factors,
            model_version = self.predictor.model_version,
        )

    def explain_from_df(
        self,
        df_processed:  pd.DataFrame,
        customer_id:   int,
        churn_score:   float,
        top_n:         int = 5,
    ) -> ExplanationOutput:
        """
        Variante para quando o DataFrame já foi processado pelo pipeline.
        Útil no contexto de batch quando se quer explicar casos específicos
        sem reprocessar os dados do zero.

        Parâmetros
        ----------
        df_processed : pd.DataFrame
            Uma linha do DataFrame já transformado pelo pipeline.
        customer_id : int
            ID do cliente (para rastreabilidade).
        churn_score : float
            Score já calculado pelo predictor (evita recalcular).
        top_n : int
            Número de fatores a retornar.
        """
        feature_names = self.predictor.trainer.get_feature_names()
        X_customer    = df_processed[feature_names]

        shap_df     = self.predictor.trainer.explain_customer(X_customer, top_n=top_n)
        top_factors = shap_df.to_dict(orient="records")

        return ExplanationOutput(
            customer_id   = customer_id,
            churn_score   = churn_score,
            top_factors   = top_factors,
            model_version = self.predictor.model_version,
        )