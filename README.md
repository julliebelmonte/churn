churn_prediction/
├── data/ # dados brutos e processados
│ ├── raw/
│ └── processed/
├── src/ # código fonte principal
│ ├── extraction/ # leitura de fontes 
│ │ └── extractor.py
│ ├── preprocessing/ # limpeza, validação, janelas
│ │ ├── window_builder.py # lógica de obs/pred windows
│ │ └── cleaner.py
│ ├── features/ # engenharia de features
│ │ ├── engagement.py
│ │ ├── financial.py
│ │ └── support.py
│ ├── training/ # treino, validação, calibração
│ │ ├── trainer.py
│ │ └── evaluator.py
│ ├── inference/ # scoring batch e API
│ │ ├── batch_scorer.py
│ │ └── api.py
│ └── monitoring/ # drift detection, alertas
│ ├── drift_detector.py
│ └── metrics_logger.py
├── tests/ # testes por camada
│ ├── test_window_builder.py
│ ├── test_features.py
│ └── test_inference.py
├── notebooks/ # exploração 
├── config.yaml # parâmetros centralizados
└── README.md