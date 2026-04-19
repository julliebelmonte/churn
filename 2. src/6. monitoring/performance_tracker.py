"""
performance_tracker.py — rastreamento de performance do modelo em produção

Responsabilidade:
    Registrar métricas de performance a cada ciclo de scoring e alertar
    quando houver degradação em relação ao baseline de deploy.

Diferença em relação ao drift_detector:
    drift_detector  → detecta mudança na DISTRIBUIÇÃO (features e score)
    performance_tracker → detecta mudança na PERFORMANCE REAL (quando labels chegam)

Labels em produção chegam com atraso:
    O modelo classifica hoje. O churn real só é observável 30 dias depois
    (quando o cliente cancela ou não). Por isso o tracker opera em dois momentos:
        T+0  → scoring (predictor.score_batch)
        T+30 → avaliação (performance_tracker.evaluate_window)

    Esse delay é o "prediction_days" da WindowConfig — os dois devem ser iguais.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)

# Thresholds de alerta de performance
ROC_AUC_DROP_ALERT:  float = 0.05   # queda > 5 pp no ROC-AUC
PR_AUC_DROP_ALERT:   float = 0.05   # queda > 5 pp no PR-AUC
CCR_DROP_ALERT:      float = 0.08   # queda > 8 pp no CCR


@dataclass
class PerformanceSnapshot:
    """
    Snapshot de performance de um ciclo de scoring específico.
    Persistido em disco para histórico e comparação entre janelas.
    """
    window_id:      str           # ex: "2025-04-01"
    n_scored:       int
    n_churners:     int
    roc_auc:        float
    pr_auc:         float
    ccr_at_20pct:   float         # CCR com capacidade de 20%
    evaluated_at:   str           # ISO timestamp

    def to_dict(self) -> dict:
        return {
            "window_id":    self.window_id,
            "n_scored":     self.n_scored,
            "n_churners":   self.n_churners,
            "roc_auc":      self.roc_auc,
            "pr_auc":       self.pr_auc,
            "ccr_at_20pct": self.ccr_at_20pct,
            "evaluated_at": self.evaluated_at,
        }


@dataclass
class PerformanceAlert:
    """Alerta gerado quando uma métrica cai abaixo do threshold."""
    metric:          str
    baseline_value:  float
    current_value:   float
    drop:            float
    threshold:       float
    window_id:       str

    @property
    def message(self) -> str:
        return (
            f"[ALERTA] {self.metric} caiu {self.drop:.2%} "
            f"(baseline={self.baseline_value:.4f} → atual={self.current_value:.4f}) "
            f"na janela {self.window_id}"
        )


class PerformanceTracker:
    """
    Rastreia e compara métricas de performance ao longo do tempo.

    Uso
    ---
    # No deploy — registra baseline
    >>> tracker = PerformanceTracker(output_dir="monitoring/history")
    >>> tracker.set_baseline(roc_auc=0.82, pr_auc=0.61, ccr_at_20pct=0.74)

    # 30 dias após cada scoring — quando labels ficam disponíveis
    >>> snapshot, alerts = tracker.evaluate_window(
    ...     window_id="2025-04-01",
    ...     y_true=labels,
    ...     y_proba=scores,
    ... )
    >>> for alert in alerts:
    ...     send_alert(alert.message)
    """

    def __init__(self, output_dir: str | Path = "monitoring/history"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._baseline: dict = {}
        self._history:  list[PerformanceSnapshot] = []

    # -------------------------------------------------------------------
    # BASELINE — registrado no deploy
    # -------------------------------------------------------------------
    def set_baseline(
        self,
        roc_auc:      float,
        pr_auc:       float,
        ccr_at_20pct: float,
    ) -> None:
        """
        Registra as métricas do modelo no momento do deploy.
        Serve de referência para todos os alertas futuros.
        """
        self._baseline = {
            "roc_auc":      roc_auc,
            "pr_auc":       pr_auc,
            "ccr_at_20pct": ccr_at_20pct,
        }
        logger.info(
            f"Baseline registrado | ROC-AUC={roc_auc:.4f} "
            f"PR-AUC={pr_auc:.4f} CCR@20%={ccr_at_20pct:.4f}"
        )

    # -------------------------------------------------------------------
    # EVALUATE — chamado quando labels chegam (T+30)
    # -------------------------------------------------------------------
    def evaluate_window(
        self,
        window_id:    str,
        y_true:       np.ndarray,
        y_proba:      np.ndarray,
        capacity_pct: float = 0.20,
    ) -> tuple[PerformanceSnapshot, list[PerformanceAlert]]:
        """
        Avalia performance de uma janela de scoring e gera alertas.

        Parâmetros
        ----------
        window_id : str
            Identificador da janela (ex: "2025-04-01").
        y_true : np.ndarray
            Labels reais observados 30 dias após o scoring.
        y_proba : np.ndarray
            Scores que o modelo gerou na época do scoring.
            Devem ser armazenados junto às predições para esse match posterior.
        capacity_pct : float
            Capacidade operacional do CRM para cálculo do CCR.
        """
        if not self._baseline:
            raise RuntimeError(
                "Baseline não definido. Chame set_baseline() após o deploy."
            )

        y_true  = np.array(y_true)
        y_proba = np.array(y_proba)

        roc_auc      = roc_auc_score(y_true, y_proba)
        pr_auc       = average_precision_score(y_true, y_proba)
        ccr_at_20pct = self._ccr(y_true, y_proba, capacity_pct)

        snapshot = PerformanceSnapshot(
            window_id    = window_id,
            n_scored     = len(y_true),
            n_churners   = int(y_true.sum()),
            roc_auc      = round(roc_auc, 4),
            pr_auc       = round(pr_auc, 4),
            ccr_at_20pct = round(ccr_at_20pct, 4),
            evaluated_at = datetime.now(timezone.utc).isoformat(),
        )

        self._history.append(snapshot)
        self._save_snapshot(snapshot)

        alerts = self._check_alerts(snapshot)
        for alert in alerts:
            logger.warning(alert.message)

        return snapshot, alerts

    # -------------------------------------------------------------------
    # HISTÓRICO
    # -------------------------------------------------------------------
    def get_history(self) -> pd.DataFrame:
        """Retorna o histórico completo de snapshots como DataFrame."""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self._history])

    def load_history(self) -> None:
        """Carrega snapshots salvos em disco para self._history."""
        for path in sorted(self.output_dir.glob("snapshot_*.json")):
            with open(path) as f:
                data = json.load(f)
            self._history.append(PerformanceSnapshot(**data))
        logger.info(f"{len(self._history)} snapshots carregados de {self.output_dir}")

    # -------------------------------------------------------------------
    # INTERNOS
    # -------------------------------------------------------------------
    def _ccr(
        self,
        y_true:       np.ndarray,
        y_proba:      np.ndarray,
        capacity_pct: float,
    ) -> float:
        n_capacity    = max(1, int(len(y_true) * capacity_pct))
        top_idx       = np.argsort(y_proba)[::-1][:n_capacity]
        total_churn   = y_true.sum()
        if total_churn == 0:
            return 0.0
        return float(y_true[top_idx].sum() / total_churn)

    def _check_alerts(self, snapshot: PerformanceSnapshot) -> list[PerformanceAlert]:
        alerts = []
        checks = [
            ("roc_auc",      snapshot.roc_auc,      ROC_AUC_DROP_ALERT),
            ("pr_auc",       snapshot.pr_auc,        PR_AUC_DROP_ALERT),
            ("ccr_at_20pct", snapshot.ccr_at_20pct,  CCR_DROP_ALERT),
        ]
        for metric, current, threshold in checks:
            baseline = self._baseline.get(metric, 0.0)
            drop     = baseline - current
            if drop > threshold:
                alerts.append(PerformanceAlert(
                    metric         = metric,
                    baseline_value = baseline,
                    current_value  = current,
                    drop           = drop,
                    threshold      = threshold,
                    window_id      = snapshot.window_id,
                ))
        return alerts

    def _save_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        path = self.output_dir / f"snapshot_{snapshot.window_id}.json"
        with open(path, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)