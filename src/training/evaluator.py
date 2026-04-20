"""
evaluator.py — métricas de negócio, explicabilidade e otimização de threshold

PRINCÍPIO:
    ROC-AUC mede o modelo.
    As métricas aqui medem o valor do modelo para o negócio.

Três responsabilidades:
    1. Métricas de negócio (além do ROC-AUC padrão)
    2. Explicabilidade por cliente (SHAP values)
    3. Otimização de threshold via curva Precision-Recall
"""

from __future__ import annotations

import logging
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Avalia o modelo em duas dimensões:
        - Estatística: ROC-AUC, PR-AUC, classification report
        - Negócio:     Captured Churn Rate, Expected Saved Revenue,
                       threshold ótimo por F-beta

    Por que PR-AUC além de ROC-AUC?
        ROC-AUC é insensível ao desbalanceamento — um modelo que classifica
        tudo como negativo pode ter ROC-AUC alto em datasets com churn < 10%.
        PR-AUC (Average Precision) foca no desempenho na classe positiva,
        que é exatamente o que importa para a área de Retenção/CRM.

    Métrica de negócio principal — Captured Churn Rate (CCR):
        "De todos os clientes que foram churnar, qual % o modelo capturou
        dentro da capacidade operacional do time de retenção?"

        O time de CRM não consegue acionar todos os clientes. Se a capacidade
        é de 200 contatos/mês e a base tem 10.000 clientes, o modelo precisa
        ser bom no top-2% — não em toda a distribuição.
        CCR mede exatamente isso: recall dentro de uma capacidade operacional.
    """

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    # -------------------------------------------------------------------
    # AVALIAÇÃO COMPLETA
    # -------------------------------------------------------------------
    def evaluate(
        self,
        y_true:  np.ndarray | pd.Series,
        y_proba: np.ndarray,
        capacity_pct: float = 0.20,
        beta: float = 2.0,
    ) -> dict:
        """
        Retorna dicionário completo de métricas estatísticas e de negócio.

        Parâmetros
        ----------
        y_true : array-like
            Labels reais (0/1).
        y_proba : array-like
            Probabilidades preditas para a classe positiva.
        capacity_pct : float
            Fração da base que o time de retenção consegue acionar.
            Default 0.20 = top-20% de risco.
        beta : float
            Peso do recall no F-beta para otimização de threshold.
            beta=2 penaliza mais falso negativo (churn não detectado)
            do que falso positivo (retenção desnecessária).
        """
        y_true  = np.array(y_true)
        y_proba = np.array(y_proba)
        y_pred  = (y_proba >= self.threshold).astype(int)

        metrics = {}

        # ---------------------------
        # Métricas estatísticas
        # ---------------------------
        metrics["roc_auc"]  = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"]   = average_precision_score(y_true, y_proba)
        metrics["threshold"] = self.threshold

        report = classification_report(y_true, y_pred, output_dict=True)
        metrics["precision_churn"] = report.get("1", {}).get("precision", 0)
        metrics["recall_churn"]    = report.get("1", {}).get("recall", 0)
        metrics["f1_churn"]        = report.get("1", {}).get("f1-score", 0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["true_positives"]  = int(tp)
        metrics["false_negatives"] = int(fn)
        metrics["false_positives"] = int(fp)

        # ---------------------------
        # Métricas de negócio
        # ---------------------------
        metrics["captured_churn_rate"] = self._captured_churn_rate(
            y_true, y_proba, capacity_pct
        )
        metrics["capacity_pct_used"] = capacity_pct

        # ---------------------------
        # Threshold ótimo por F-beta
        # ---------------------------
        metrics["optimal_threshold"] = self._find_optimal_threshold(
            y_true, y_proba, beta=beta
        )
        metrics["fbeta_beta"] = beta

        self._log_summary(metrics)
        return metrics

    # -------------------------------------------------------------------
    # CAPTURED CHURN RATE — métrica principal de negócio
    # -------------------------------------------------------------------
    def _captured_churn_rate(
        self,
        y_true:       np.ndarray,
        y_proba:      np.ndarray,
        capacity_pct: float,
    ) -> float:
        """
        Calcula o recall do modelo dentro da capacidade operacional.

        Ordena a base pelo score de risco (maior → menor), pega os top-N%
        e calcula qual fração dos churners reais foram capturados nesse grupo.

        Exemplo:
            Base: 10.000 clientes, 800 churners reais (8%)
            Capacidade: 20% → top 2.000 clientes pelo score
            Se 600 dos 800 churners estão nos top-2.000:
            CCR = 600 / 800 = 75%

        Essa é a métrica que a área de Retenção/CRM entende e cobra.
        """
        n_total   = len(y_true)
        n_capacity = max(1, int(n_total * capacity_pct))

        top_indices = np.argsort(y_proba)[::-1][:n_capacity]

        churners_total    = y_true.sum()
        churners_captured = y_true[top_indices].sum()

        if churners_total == 0:
            return 0.0

        return float(churners_captured / churners_total)

    # -------------------------------------------------------------------
    # THRESHOLD ÓTIMO — F-beta
    # -------------------------------------------------------------------
    def _find_optimal_threshold(
        self,
        y_true:  np.ndarray,
        y_proba: np.ndarray,
        beta:    float = 2.0,
    ) -> float:
        """
        Encontra o threshold que maximiza o F-beta score.

        beta=2 → recall vale o dobro da precision.
        Justificativa: custo de perder um churner (FN) é maior que
        custo de acionar um cliente que ficaria (FP).
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        # Evita divisão por zero
        denom = (beta ** 2 * precision + recall)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fbeta = np.where(
                denom > 0,
                (1 + beta ** 2) * precision * recall / denom,
                0,
            )

        best_idx = np.argmax(fbeta[:-1])  # thresholds tem len = len(precision) - 1
        return float(thresholds[best_idx])

    # -------------------------------------------------------------------
    # EXPLICABILIDADE — SHAP por cliente
    # -------------------------------------------------------------------
    def explain_customer(
        self,
        model,
        X_customer: pd.DataFrame,
        feature_names: list[str],
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Retorna os top-N fatores de risco para um cliente específico via SHAP.

        Por que SHAP e não feature importance global?
            Feature importance global diz quais features são importantes
            para o modelo em média. SHAP diz por que ESTE cliente específico
            foi classificado como alto risco — que é o que a área de negócios
            precisa para construir a abordagem de retenção.

        Exemplo de output:
            feature                   shap_value  direction
            feat_recency_anomaly          0.42       risco
            feat_satisfaction_gap         0.31       risco
            feat_purchases_vs_population -0.18      proteção

        Parâmetros
        ----------
        model : LightGBM Booster ou sklearn-compatible
            Modelo treinado com suporte a SHAP.
        X_customer : pd.DataFrame
            Uma linha com as features do cliente a explicar.
        feature_names : list[str]
            Lista de features na mesma ordem das colunas de X_customer.
        top_n : int
            Quantos fatores retornar (os de maior |shap_value|).
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP não instalado. Execute: pip install shap"
            )

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_customer)

        # LightGBM binário retorna lista com [neg, pos] — pega a classe positiva
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_series = pd.Series(
            shap_values[0],
            index=feature_names,
        )

        top = shap_series.abs().nlargest(top_n).index
        result = pd.DataFrame({
            "feature":    top,
            "shap_value": shap_series[top].values,
            "direction":  ["risco" if v > 0 else "proteção"
                           for v in shap_series[top].values],
        })

        return result.reset_index(drop=True)

    # -------------------------------------------------------------------
    # LOG
    # -------------------------------------------------------------------
    def _log_summary(self, metrics: dict) -> None:
        logger.info("=" * 50)
        logger.info("AVALIAÇÃO DO MODELO")
        logger.info(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
        logger.info(f"  PR-AUC             : {metrics['pr_auc']:.4f}")
        logger.info(f"  Precision (churn)  : {metrics['precision_churn']:.4f}")
        logger.info(f"  Recall    (churn)  : {metrics['recall_churn']:.4f}")
        logger.info(f"  F1        (churn)  : {metrics['f1_churn']:.4f}")
        logger.info(
            f"  Captured Churn Rate: {metrics['captured_churn_rate']:.2%} "
            f"(capacidade={metrics['capacity_pct_used']:.0%})"
        )
        logger.info(f"  Threshold usado    : {metrics['threshold']:.2f}")
        logger.info(f"  Threshold ótimo    : {metrics['optimal_threshold']:.2f} "
                    f"(F-beta={metrics['fbeta_beta']})")
        logger.info("=" * 50)