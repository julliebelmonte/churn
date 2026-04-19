"""
drift_detector.py — detecção de data drift e concept drift

DISTINÇÃO CRÍTICA:

    Data Drift (covariates shift):
        A distribuição das features mudou, mas a relação entre features
        e target pode ser a mesma. Exemplo: sazonalidade de compras no
        natal faz Last_Purchase_Days_Ago subir para toda a base — o modelo
        ainda funciona, mas as features estão fora da distribuição de treino.
        Sinal: distribuição de X muda, performance pode ou não degradar.
        Diagnóstico: KS-test (Kolmogorov-Smirnov) por feature.

    Concept Drift (posterior shift):
        O comportamento que define churn mudou — a relação P(churn | features)
        se alterou. Exemplo: uma crise econômica faz clientes satisfeitos
        cancelarem por motivo financeiro, não por insatisfação. O modelo
        não consegue capturar esse novo padrão porque não estava no treino.
        Sinal: performance real degradou (CCR caiu) sem mudança nas features.
        Diagnóstico: PSI no score + queda de CCR quando labels ficam disponíveis.

Como diferenciar na prática:
    1. Data drift sem concept drift → features deslocadas, CCR estável
       Ação: monitorar. Pode ser sazonal. Retreinar preventivamente.
    2. Data drift com concept drift → features deslocadas, CCR caiu
       Ação: retreinar com dados recentes urgente.
    3. Concept drift sem data drift → features estáveis, CCR caiu
       Ação: investigar mudança de comportamento. Retreinar.
    4. Nenhum drift → features e CCR estáveis
       Ação: continuar monitorando.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats

logger = logging.getLogger(__name__)

# Thresholds de alerta — ajustar conforme tolerância do negócio
KS_ALERT_THRESHOLD:  float = 0.10   # KS statistic > 10% = data drift
PSI_ALERT_THRESHOLD: float = 0.20   # PSI > 0.20 = instabilidade severa no score
CCR_DROP_THRESHOLD:  float = 0.08   # queda > 8 pp no CCR = concept drift


@dataclass
class DriftReport:
    """
    Resultado consolidado de uma rodada de detecção de drift.
    Diferencia explicitamente os dois tipos de drift.
    """
    # Data drift — por feature
    feature_ks:         dict[str, float] = field(default_factory=dict)
    drifted_features:   list[str]        = field(default_factory=list)
    data_drift_detected: bool            = False

    # Concept drift — via score PSI e CCR
    score_psi:              float = 0.0
    ccr_current:            float | None = None
    ccr_baseline:           float | None = None
    ccr_drop:               float | None = None
    concept_drift_detected: bool         = False

    # Diagnóstico consolidado
    @property
    def alert_level(self) -> str:
        if self.concept_drift_detected:
            return "critical"   # retreinar urgente
        if self.data_drift_detected:
            return "warning"    # monitorar / retreinar preventivo
        return "ok"

    def summary(self) -> str:
        lines = [
            f"Alert level   : {self.alert_level.upper()}",
            f"Data drift    : {self.data_drift_detected} "
            f"({len(self.drifted_features)} features)",
            f"Concept drift : {self.concept_drift_detected} "
            f"(PSI={self.score_psi:.3f})",
        ]
        if self.ccr_drop is not None:
            lines.append(f"CCR drop      : {self.ccr_drop:.2%}")
        if self.drifted_features:
            lines.append(f"Drifted feats : {', '.join(self.drifted_features)}")
        return "\n".join(lines)


class DriftDetector:
    """
    Monitora drift após o deploy, comparando a distribuição atual com
    a distribuição de referência (treino).

    Uso
    ---
    >>> detector = DriftDetector()
    >>> detector.fit_reference(df_train, train_scores)
    >>> report = detector.detect(df_current, current_scores)
    >>> if report.alert_level != "ok":
    ...     send_alert(report.summary())
    """

    def __init__(
        self,
        ks_threshold:  float = KS_ALERT_THRESHOLD,
        psi_threshold: float = PSI_ALERT_THRESHOLD,
        ccr_threshold: float = CCR_DROP_THRESHOLD,
    ):
        self.ks_threshold  = ks_threshold
        self.psi_threshold = psi_threshold
        self.ccr_threshold = ccr_threshold

        self._ref_features: pd.DataFrame | None = None
        self._ref_scores:   np.ndarray | None   = None
        self._ccr_baseline: float | None        = None
        self._fitted = False

    # -------------------------------------------------------------------
    # FIT — aprende distribuição de referência no treino
    # -------------------------------------------------------------------
    def fit_reference(
        self,
        df_train:      pd.DataFrame,
        train_scores:  np.ndarray,
        ccr_baseline:  float | None = None,
    ) -> "DriftDetector":
        """
        Memoriza a distribuição de referência do conjunto de treino.

        Parâmetros
        ----------
        df_train : pd.DataFrame
            DataFrame de treino com as features (saída do pipeline).
        train_scores : np.ndarray
            Scores do modelo no treino (para calcular PSI de referência).
        ccr_baseline : float | None
            CCR baseline do treino. Se None, PSI sozinho detecta concept drift.
        """
        self._ref_features  = df_train.copy()
        self._ref_scores    = np.array(train_scores)
        self._ccr_baseline  = ccr_baseline
        self._fitted        = True

        logger.info(
            f"DriftDetector ajustado | n_ref={len(df_train)} "
            f"features={list(df_train.columns[:5])}..."
        )
        return self

    # -------------------------------------------------------------------
    # DETECT — compara batch atual com referência
    # -------------------------------------------------------------------
    def detect(
        self,
        df_current:     pd.DataFrame,
        current_scores: np.ndarray,
        y_true:         np.ndarray | None = None,
        ccr_current:    float | None      = None,
    ) -> DriftReport:
        """
        Executa detecção de data drift e concept drift.

        Parâmetros
        ----------
        df_current : pd.DataFrame
            DataFrame do batch atual (mesmas features do treino).
        current_scores : np.ndarray
            Scores do modelo no batch atual.
        y_true : np.ndarray | None
            Labels reais, se disponíveis (ex: 30 dias após scoring).
            Quando disponível, permite calcular CCR real.
        ccr_current : float | None
            CCR já calculado externamente. Alternativa a y_true.
        """
        if not self._fitted:
            raise RuntimeError("Chame fit_reference() antes de detect().")

        report = DriftReport()

        # ---------------------------
        # 1. DATA DRIFT — KS-test por feature
        # ---------------------------
        common_cols = [
            c for c in self._ref_features.columns
            if c in df_current.columns and
            self._ref_features[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        for col in common_cols:
            ks_stat, _ = stats.ks_2samp(
                self._ref_features[col].dropna(),
                df_current[col].dropna(),
            )
            report.feature_ks[col] = round(float(ks_stat), 4)
            if ks_stat > self.ks_threshold:
                report.drifted_features.append(col)

        report.data_drift_detected = len(report.drifted_features) > 0

        # ---------------------------
        # 2. CONCEPT DRIFT — PSI no score
        # ---------------------------
        report.score_psi = self._compute_psi(self._ref_scores, current_scores)

        # Se CCR real estiver disponível, compara com baseline
        if ccr_current is not None and self._ccr_baseline is not None:
            report.ccr_current  = ccr_current
            report.ccr_baseline = self._ccr_baseline
            report.ccr_drop     = self._ccr_baseline - ccr_current

            # Concept drift = PSI alto OU queda significativa no CCR
            report.concept_drift_detected = (
                report.score_psi > self.psi_threshold or
                report.ccr_drop > self.ccr_threshold
            )
        else:
            # Sem CCR real, usa PSI sozinho como proxy de concept drift
            report.concept_drift_detected = report.score_psi > self.psi_threshold

        logger.info(f"\n{report.summary()}")
        return report

    # -------------------------------------------------------------------
    # PSI — Population Stability Index
    # -------------------------------------------------------------------
    def _compute_psi(
        self,
        reference: np.ndarray,
        current:   np.ndarray,
        n_bins:    int = 10,
    ) -> float:
        """
        Calcula o PSI entre a distribuição de referência e a atual.

        PSI < 0.10 → estável
        PSI 0.10–0.20 → mudança moderada, monitorar
        PSI > 0.20 → instabilidade severa, retreinar

        Usa os bins da referência para garantir comparabilidade.
        Adiciona epsilon para evitar log(0).
        """
        eps = 1e-6
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)   # remove duplicatas em distribuições concentradas

        ref_counts, _  = np.histogram(reference, bins=bins)
        cur_counts, _  = np.histogram(current,   bins=bins)

        ref_pct = ref_counts / (ref_counts.sum() + eps)
        cur_pct = cur_counts / (cur_counts.sum() + eps)

        # Evita log(0)
        ref_pct = np.where(ref_pct == 0, eps, ref_pct)
        cur_pct = np.where(cur_pct == 0, eps, cur_pct)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(round(psi, 4))