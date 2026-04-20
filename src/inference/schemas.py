"""
schemas.py — contratos de entrada e saída da camada de inferência

Por que schemas explícitos?
    Sem validação de contrato, um campo faltando ou com tipo errado
    chega silenciosamente até o modelo e gera predição inválida ou erro
    críptico no meio do pipeline. Schemas explícitos:
    - Falham rápido e com mensagem clara na borda do sistema
    - Documentam o contrato entre quem chama e o modelo
    - Permitem versionamento da API sem quebrar consumidores antigos
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomerInput:
    """
    Contrato de entrada para predição de um único cliente.
    Espelha as colunas brutas do extractor — antes de qualquer transformação.
    """
    customer_id:                int
    age:                        int
    gender:                     str
    annual_income:              float
    total_spend:                float
    years_as_customer:          int
    num_of_purchases:           int
    average_transaction_amount: float
    num_of_returns:             int
    num_of_support_contacts:    int
    satisfaction_score:         int
    last_purchase_days_ago:     int
    email_opt_in:               bool
    promotion_response:         str

    def validate(self) -> None:
        """Validações mínimas de domínio antes de entrar no pipeline."""
        if self.satisfaction_score not in range(1, 6):
            raise ValueError(
                f"satisfaction_score deve ser entre 1 e 5, "
                f"recebido: {self.satisfaction_score}"
            )
        if self.last_purchase_days_ago < 0:
            raise ValueError("last_purchase_days_ago não pode ser negativo.")
        if self.years_as_customer < 0:
            raise ValueError("years_as_customer não pode ser negativo.")
        if self.num_of_purchases < 0:
            raise ValueError("num_of_purchases não pode ser negativo.")


@dataclass
class ChurnPrediction:
    """
    Contrato de saída de uma predição individual.
    Inclui o score bruto, a classe e os metadados de rastreabilidade.
    """
    customer_id:    int
    churn_score:    float          # probabilidade 0–1
    churn_label:    int            # 0 ou 1 após aplicação do threshold
    threshold_used: float
    risk_tier:      str            # "low" | "medium" | "high" (para o CRM)
    model_version:  str

    @classmethod
    def from_score(
        cls,
        customer_id:   int,
        score:         float,
        threshold:     float,
        model_version: str,
    ) -> "ChurnPrediction":
        return cls(
            customer_id    = customer_id,
            churn_score    = round(score, 4),
            churn_label    = int(score >= threshold),
            threshold_used = threshold,
            risk_tier      = _score_to_tier(score),
            model_version  = model_version,
        )

    def to_dict(self) -> dict:
        return {
            "customer_id":    self.customer_id,
            "churn_score":    self.churn_score,
            "churn_label":    self.churn_label,
            "threshold_used": self.threshold_used,
            "risk_tier":      self.risk_tier,
            "model_version":  self.model_version,
        }


@dataclass
class BatchPredictionResult:
    """
    Resultado de uma rodada de scoring em batch.
    Agrega metadados operacionais além das predições individuais.
    """
    predictions:       list[ChurnPrediction]
    total_scored:      int
    total_high_risk:   int
    model_version:     str
    scored_at:         str    # ISO timestamp

    def to_records(self) -> list[dict]:
        return [p.to_dict() for p in self.predictions]

    @property
    def high_risk_rate(self) -> float:
        if self.total_scored == 0:
            return 0.0
        return round(self.total_high_risk / self.total_scored, 4)


@dataclass
class ExplanationOutput:
    """
    Contrato de saída da explicação SHAP para um cliente.
    Usado pela API on-demand quando o CRM precisa entender um caso específico.
    """
    customer_id: int
    churn_score: float
    top_factors: list[dict]   # [{"feature": ..., "shap_value": ..., "direction": ...}]
    model_version: str

    def to_dict(self) -> dict:
        return {
            "customer_id":  self.customer_id,
            "churn_score":  self.churn_score,
            "top_factors":  self.top_factors,
            "model_version": self.model_version,
        }


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def _score_to_tier(score: float) -> str:
    """
    Converte score contínuo em tier categórico para o CRM.

    Tiers comunicam prioridade de acionamento sem expor o score bruto
    para equipes não-técnicas. Os thresholds devem ser alinhados com
    a capacidade operacional do time de retenção.
    """
    if score >= 0.65:
        return "high"
    elif score >= 0.35:
        return "medium"
    return "low"