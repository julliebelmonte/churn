#!/usr/bin/env python
"""
run_pipeline.py — entrypoint para execução do pipeline de churn

Uso:
    python run_pipeline.py

Saída:
    artifacts/model_<ts>.pkl
    artifacts/metrics_<ts>.json
    artifacts/feature_importance_<ts>.csv
"""

import sys, logging, warnings, json
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

DATA_PATH = "data/raw/online_retail_customer_churn.csv"


def main():
    from src.pipeline.pipeline import ChurnPipeline
    from src.training.trainer import ChurnTrainer

    logger.info("=" * 60)
    logger.info("CHURN PREDICTION PIPELINE — INÍCIO")
    logger.info("=" * 60)

    # 1. Pipeline de dados
    logger.info("Etapa 1/3 — Extração, pré-processamento e feature engineering")
    pipeline = ChurnPipeline()
    train, val, test = pipeline.fit(DATA_PATH)
    feature_names = pipeline.get_feature_names()
    logger.info(
        f"Partições: train={len(train)} | val={len(val)} | test={len(test)}"
    )
    logger.info(f"Features criadas ({len(feature_names)}): {feature_names}")

    # 2. Treinamento
    logger.info("Etapa 2/3 — Treinamento do modelo LightGBM")
    trainer = ChurnTrainer(output_dir="artifacts")
    results = trainer.train(train, val, test, feature_names)

    # 3. Resumo final
    logger.info("Etapa 3/3 — Resultados finais")
    tm = results["test_metrics"]
    logger.info("=" * 60)
    logger.info("RESULTADOS NO CONJUNTO DE TESTE")
    logger.info(f"  ROC-AUC              : {tm['roc_auc']:.4f}")
    logger.info(f"  PR-AUC               : {tm['pr_auc']:.4f}")
    logger.info(f"  Recall (churn)       : {tm['recall_churn']:.4f}")
    logger.info(f"  Precision (churn)    : {tm['precision_churn']:.4f}")
    logger.info(f"  F1 (churn)           : {tm['f1_churn']:.4f}")
    logger.info(
        f"  Captured Churn Rate  : {tm['captured_churn_rate']:.2%} "
        f"(top-{tm['capacity_pct_used']:.0%} da base)"
    )
    logger.info(f"  Threshold ótimo      : {tm['optimal_threshold']:.3f}")
    logger.info("=" * 60)

    logger.info("\nTOP 10 FEATURES MAIS IMPORTANTES:")
    fi = results["feature_importance"]
    for _, row in fi.head(10).iterrows():
        bar = "█" * int(row["importance_pct"] * 60)
        logger.info(f"  {row['feature']:<35} {bar} {row['importance_pct']:.1%}")

    logger.info("\nArtefatos salvos:")
    for k, v in results["artifact_paths"].items():
        logger.info(f"  {k:<20}: {v}")

    logger.info("\nPipeline concluído com sucesso.")
    return results


if __name__ == "__main__":
    main()
