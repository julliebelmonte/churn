"""
api.py — serving FastAPI do modelo de churn

Dois endpoints:
    GET  /health  — liveness + readiness (modelo carregado?)
    POST /predict — score individual com explicação opcional

Decisões de arquitetura:
    - Lifespan para carregar o predictor uma única vez no startup,
      não no escopo do módulo. Carregar no módulo significa que o modelo
      é instanciado no import — antes do servidor estar pronto para aceitar
      requests — e sem nenhum artefato carregado.

    - score_one() em vez de predict() + predict_proba() separados.
      Duas chamadas ao pipeline para o mesmo cliente é desperdício e
      abre janela para inconsistência (scores diferentes entre chamadas
      se houver estado mutável).

    - explain=false por padrão no /predict. SHAP é mais caro que scoring.
      O CRM só pede explicação quando o atendente abre a ficha — não
      em cada request de scoring em batch.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from src.inference.predictor import ChurnPredictor
from src.inference.explainer import ChurnExplainer
from src.inference.schemas import CustomerInput

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Estado global da aplicação
# -------------------------------------------------------------------
# Dicionário mutável — permite injeção em testes sem monkey-patching
_state: dict = {
    "predictor": None,
    "explainer": None,
}

TRAINER_PATH  = Path("artifacts/trainer.pkl")
MODEL_VERSION = "1.0.0"


# -------------------------------------------------------------------
# LIFESPAN — carrega modelo no startup, libera no shutdown
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carrega o predictor uma única vez quando o servidor sobe.
    Sem lifespan, qualquer import desse módulo instanciaria o modelo —
    inclusive em testes unitários que não precisam do modelo carregado.
    """
    logger.info(f"Carregando modelo de {TRAINER_PATH}...")

    if not TRAINER_PATH.exists():
        raise RuntimeError(
            f"Artefato não encontrado: {TRAINER_PATH}. "
            "Execute o training antes de subir a API."
        )

    predictor = ChurnPredictor.load(
        trainer_path  = TRAINER_PATH,
        model_version = MODEL_VERSION,
    )
    explainer = ChurnExplainer(predictor)

    _state["predictor"] = predictor
    _state["explainer"] = explainer

    logger.info(f"Modelo carregado | versão={MODEL_VERSION} | threshold={predictor._threshold:.2f}")

    yield  # servidor ativo

    # Shutdown
    _state["predictor"] = None
    _state["explainer"] = None
    logger.info("Modelo descarregado.")


# -------------------------------------------------------------------
# APP
# -------------------------------------------------------------------
app = FastAPI(
    title       = "Churn Prediction API",
    description = "Scoring e explicação de risco de churn por cliente.",
    version     = MODEL_VERSION,
    lifespan    = lifespan,
)


# -------------------------------------------------------------------
# SCHEMAS DE REQUEST / RESPONSE (Pydantic v2)
# -------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    Espelha CustomerInput mas como modelo Pydantic para validação HTTP.
    Separado do CustomerInput (dataclass interna) para não acoplar
    a camada de inferência ao contrato HTTP.
    """
    customer_id:                int
    age:                        int   = Field(..., ge=0, le=120)
    gender:                     str
    annual_income:              float = Field(..., ge=0)
    total_spend:                float = Field(..., ge=0)
    years_as_customer:          int   = Field(..., ge=0)
    num_of_purchases:           int   = Field(..., ge=0)
    average_transaction_amount: float = Field(..., ge=0)
    num_of_returns:             int   = Field(..., ge=0)
    num_of_support_contacts:    int   = Field(..., ge=0)
    satisfaction_score:         int   = Field(..., ge=1, le=5)
    last_purchase_days_ago:     int   = Field(..., ge=0)
    email_opt_in:               bool
    promotion_response:         str

    @field_validator("gender")
    @classmethod
    def gender_valid(cls, v: str) -> str:
        allowed = {"male", "female", "other"}
        if v.lower() not in allowed:
            raise ValueError(f"gender deve ser um de: {allowed}")
        return v.lower()

    def to_customer_input(self) -> CustomerInput:
        """Converte para o contrato interno da camada de inferência."""
        return CustomerInput(**self.model_dump())


class PredictResponse(BaseModel):
    customer_id:    int
    churn_score:    float
    churn_label:    int
    risk_tier:      str
    threshold_used: float
    model_version:  str


class PredictWithExplanationResponse(PredictResponse):
    top_factors: list[dict]


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str | None
    threshold:     float | None


# -------------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """
    Liveness + readiness em um único endpoint.
    Retorna model_loaded=false se o predictor não foi inicializado —
    útil para o load balancer não rotear requests antes do modelo subir.
    """
    predictor = _state["predictor"]
    return HealthResponse(
        status        = "ok",
        model_loaded  = predictor is not None,
        model_version = MODEL_VERSION if predictor else None,
        threshold     = predictor._threshold if predictor else None,
    )


@app.post("/predict", tags=["scoring"])
def predict(
    data:    PredictRequest,
    explain: bool = Query(default=False, description="Inclui explicação SHAP na resposta"),
) -> PredictResponse | PredictWithExplanationResponse:
    """
    Pontua um cliente e retorna o risco de churn.

    - explain=false (padrão): retorna score + tier. Otimizado para latência.
    - explain=true: inclui os top-5 fatores SHAP. Mais lento — usar on-demand.
    """
    predictor: ChurnPredictor | None = _state["predictor"]
    explainer: ChurnExplainer | None = _state["explainer"]

    if predictor is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Modelo ainda não carregado. Tente novamente em instantes.",
        )

    # Converte e valida no domínio (CustomerInput.validate())
    try:
        customer_input = data.to_customer_input()
        prediction     = predictor.score_one(customer_input)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(f"Erro interno no scoring | customer_id={data.customer_id}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")

    base = PredictResponse(
        customer_id    = prediction.customer_id,
        churn_score    = prediction.churn_score,
        churn_label    = prediction.churn_label,
        risk_tier      = prediction.risk_tier,
        threshold_used = prediction.threshold_used,
        model_version  = prediction.model_version,
    )

    if not explain:
        return base

    # Explicação SHAP — só executa se solicitada
    try:
        explanation = explainer.explain(customer_input, top_n=5)
    except Exception as e:
        logger.exception(f"Erro na explicação SHAP | customer_id={data.customer_id}")
        raise HTTPException(status_code=500, detail="Erro ao gerar explicação.")

    return PredictWithExplanationResponse(
        **base.model_dump(),
        top_factors = explanation.top_factors,
    )