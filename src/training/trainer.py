"""
trainer.py — orquestrador do treinamento do modelo de churn

Responsabilidade:
    Receber as partições prontas do pipeline, treinar o modelo,
    avaliar e persistir artefatos. NÃO faz preprocessing nem
    feature engineering — isso é responsabilidade do pipeline/.

Fluxo:
    ChurnPipeline.fit()  →  train, val, test
    ChurnTrainer.train() →  modelo treinado + métricas + artefatos salvos

Separação de responsabilidades:
    trainer.py    — orquestra treino, early stopping, persistência
    evaluator.py  — calcula métricas estatísticas e de negócio
    model_config.py — hiperparâmetros e configuração do experimento
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.model_config import ModelConfig
from src.training.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


class ChurnTrainer:
    """
    Treina e avalia o modelo de churn a partir das partições do pipeline.

    Uso
    ---
    >>> pipeline = ChurnPipeline()
    >>> train, val, test = pipeline.fit("data/churn.csv")
    >>>
    >>> trainer = ChurnTrainer()
    >>> results = trainer.train(train, val, test, pipeline.get_feature_names())

    O que o trainer persiste em artifacts/:
        - modelo serializado (.pkl)
        - métricas de avaliação (.json)
        - feature importance (.csv)
        - threshold ótimo calculado no val set

    Por que LightGBM?
        - Nativo em dados tabulares desbalanceados (scale_pos_weight)
        - Early stopping integrado sem configuração extra
        - SHAP values nativos — explicabilidade sem custo adicional
        - Muito mais rápido que XGBoost em datasets < 1M linhas
        - Suporta missing values nativamente (útil em produção)
    """

    def __init__(
        self,
        config:     ModelConfig | None = None,
        output_dir: str | Path = "artifacts",
    ):
        self.config     = config or ModelConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model:     object | None = None
        self.evaluator: ModelEvaluator = ModelEvaluator(
            threshold=self.config.threshold
        )
        self._feature_names: list[str] = []

    # -------------------------------------------------------------------
    # INTERFACE PÚBLICA
    # -------------------------------------------------------------------
    def train(
        self,
        train:         pd.DataFrame,
        val:           pd.DataFrame,
        test:          pd.DataFrame,
        feature_names: list[str],
    ) -> dict:
        """
        Treina o modelo e retorna o dicionário completo de resultados.

        Parâmetros
        ----------
        train, val, test : pd.DataFrame
            Partições prontas (saída do ChurnPipeline.fit()).
        feature_names : list[str]
            Features a usar no treinamento (saída de pipeline.get_feature_names()).

        Retorna
        -------
        dict com:
            - val_metrics:  métricas no conjunto de validação
            - test_metrics: métricas no conjunto de teste (avaliação final)
            - feature_importance: DataFrame com importância das features
            - artifact_paths: caminhos dos artefatos salvos
        """
        self._feature_names = feature_names
        logger.info(f"Iniciando treinamento | features={len(feature_names)}")

        X_train, y_train = self._split_xy(train)
        X_val,   y_val   = self._split_xy(val)
        X_test,  y_test  = self._split_xy(test)

        # Ajusta scale_pos_weight com base na proporção real do treino
        self.config.lgbm_params["scale_pos_weight"] = self._compute_pos_weight(y_train)

        # Treino com early stopping no val set
        self.model = self._fit_lgbm(X_train, y_train, X_val, y_val)

        # Avaliação no val — usado para escolher threshold ótimo
        y_val_proba  = self._predict_proba(X_val)
        val_metrics  = self.evaluator.evaluate(y_val, y_val_proba)

        # Atualiza threshold com o ótimo encontrado no val
        optimal_threshold = val_metrics["optimal_threshold"]
        self.evaluator.threshold = optimal_threshold
        logger.info(f"Threshold atualizado para {optimal_threshold:.3f} (F-beta ótimo no val)")

        # Avaliação final no test — nunca usado para decisões de treino
        y_test_proba = self._predict_proba(X_test)
        test_metrics = self.evaluator.evaluate(y_test, y_test_proba)

        # Feature importance
        importance_df = self._get_feature_importance()

        # Persistência
        artifact_paths = self._save_artifacts(
            val_metrics, test_metrics, importance_df, optimal_threshold
        )

        logger.info("Treinamento concluído.")

        return {
            "val_metrics":        val_metrics,
            "test_metrics":       test_metrics,
            "feature_importance": importance_df,
            "artifact_paths":     artifact_paths,
            "optimal_threshold":  optimal_threshold,
        }

    # -------------------------------------------------------------------
    # PREDICT — para uso pelo inference/
    # -------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades para um DataFrame de features."""
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Chame train() primeiro.")
        return self._predict_proba(X[self._feature_names])

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Retorna classes binárias (0/1) usando o threshold configurado."""
        t = threshold or self.evaluator.threshold
        return (self.predict_proba(X) >= t).astype(int)

    def explain_customer(self, X_customer: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """Delega para o Evaluator — SHAP values para um cliente específico."""
        return self.evaluator.explain_customer(
            self.model,
            X_customer[self._feature_names],
            self._feature_names,
            top_n=top_n,
        )

    # -------------------------------------------------------------------
    # PERSISTÊNCIA
    # -------------------------------------------------------------------
    def save(self, path: str | Path | None = None) -> Path:
        """Serializa o trainer completo (modelo + config + threshold)."""
        if self.model is None:
            raise RuntimeError("Nada para salvar — treine o modelo primeiro.")

        path = Path(path) if path else self.output_dir / "trainer.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Trainer salvo em {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ChurnTrainer":
        """Carrega um trainer serializado."""
        with open(path, "rb") as f:
            trainer = pickle.load(f)
        logger.info(f"Trainer carregado de {path}")
        return trainer

    # -------------------------------------------------------------------
    # INTERNOS
    # -------------------------------------------------------------------
    def _split_xy(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separa features de target.

        Exclui o target e as colunas de controle interno do pipeline
        (prefixos _obs_ e _pred_) que não devem entrar no modelo.
        """
        exclude = {self.config.target_col} | {
            c for c in df.columns
            if any(c.startswith(p) for p in self.config.non_feature_prefixes)
        }
        feature_cols = [
            c for c in self._feature_names if c in df.columns
        ]
        X = df[feature_cols]
        y = df[self.config.target_col]
        return X, y

    def _compute_pos_weight(self, y: pd.Series) -> float:
        """
        Calcula scale_pos_weight = n_negativos / n_positivos.

        Compensa o desbalanceamento de classes de forma dinâmica,
        em vez de usar um valor fixo na config.
        """
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos == 0:
            raise ValueError("Nenhum exemplo positivo (churn=1) no treino.")
        weight = n_neg / n_pos
        logger.info(
            f"Desbalanceamento: {n_pos} positivos / {n_neg} negativos "
            f"→ scale_pos_weight={weight:.2f}"
        )
        return weight

    def _fit_lgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
    ):
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM não instalado. Execute: pip install lightgbm"
            )

        params = {**self.config.lgbm_params}
        n_estimators = params.pop("n_estimators")

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.config.early_stopping_rounds,
                verbose=False,
            ),
            lgb.log_evaluation(period=50),
        ]

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            **params,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

        best_iter = model.best_iteration_
        logger.info(f"Early stopping: melhor iteração = {best_iter}")
        return model

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def _get_feature_importance(self) -> pd.DataFrame:
        importance = pd.DataFrame({
            "feature":    self._feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        importance["importance_pct"] = (
            importance["importance"] / importance["importance"].sum()
        ).round(4)

        return importance

    def _save_artifacts(
        self,
        val_metrics:   dict,
        test_metrics:  dict,
        importance_df: pd.DataFrame,
        threshold:     float,
    ) -> dict:
        import json

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Modelo
        model_path = self.output_dir / f"model_{ts}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Métricas
        metrics_path = self.output_dir / f"metrics_{ts}.json"
        payload = {
            "experiment":     self.config.experiment_name,
            "timestamp":      ts,
            "threshold":      threshold,
            "val_metrics":    {k: round(float(v), 6) if isinstance(v, float) else v
                               for k, v in val_metrics.items()},
            "test_metrics":   {k: round(float(v), 6) if isinstance(v, float) else v
                               for k, v in test_metrics.items()},
        }
        with open(metrics_path, "w") as f:
            json.dump(payload, f, indent=2)

        # Feature importance
        importance_path = self.output_dir / f"feature_importance_{ts}.csv"
        importance_df.to_csv(importance_path, index=False)

        paths = {
            "model":              str(model_path),
            "metrics":            str(metrics_path),
            "feature_importance": str(importance_path),
        }

        logger.info(f"Artefatos salvos em {self.output_dir}/")
        return paths