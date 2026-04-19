"""
test_preprocessing.py — testes do módulo preprocessing

Camada testada:
  1. cleaner.py
  2. window_builder.py
  3. segmentation.py

Regra:
  Este arquivo NÃO testa features nem model training.
"""

import pytest
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.window_builder import WindowConfig, WindowBuilder
from src.preprocessing.segmentation import CustomerSegmenter
from src.preprocessing.cleaner import DataCleaner


# =============================================================================
# FIXTURE BASE
# =============================================================================

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        "Customer_ID": range(n),
        "Age": np.random.randint(18, 70, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Annual_Income": np.random.uniform(20, 150, n),
        "Total_Spend": np.random.uniform(500, 50000, n),
        "Years_as_Customer": np.random.randint(1, 20, n),
        "Num_of_Purchases": np.random.randint(1, 100, n),
        "Average_Transaction_Amount": np.random.uniform(10, 500, n),
        "Num_of_Returns": np.random.randint(0, 10, n),
        "Num_of_Support_Contacts": np.random.randint(0, 5, n),
        "Satisfaction_Score": np.random.randint(1, 6, n),
        "Last_Purchase_Days_Ago": np.random.randint(1, 365, n),
        "Email_Opt_In": np.random.choice([True, False], n),
        "Promotion_Response": np.random.choice(["Responded", "Ignored"], n),
        "Target_Churn": np.random.choice([0, 1], n),
    })


@pytest.fixture
def window_config():
    return WindowConfig(observation_days=90, prediction_days=30)


# =============================================================================
# 1. CLEANER — responsabilidade isolada
# =============================================================================

class TestDataCleaner:

    def test_removes_identifier_column(self, sample_df):
        cleaner = DataCleaner()
        df = cleaner.fit_transform(sample_df)

        assert "Customer_ID" not in df.columns

    def test_target_is_binary_int(self, sample_df):
        cleaner = DataCleaner()
        df = cleaner.fit_transform(sample_df)

        assert df["Target_Churn"].dtype in [np.int32, np.int64, int]

    def test_no_object_columns_remain(self, sample_df):
        cleaner = DataCleaner()
        df = cleaner.fit_transform(sample_df)

        assert df.select_dtypes(include="object").shape[1] == 0

    def test_split_is_consistent(self, sample_df):
        cleaner = DataCleaner()

        train, val, test = cleaner.split(sample_df)

        assert len(train) + len(val) + len(test) == len(sample_df)
        assert train.index.isdisjoint(test.index)
        assert train.index.isdisjoint(val.index)
        assert val.index.isdisjoint(test.index)

    def test_transform_requires_fit(self, sample_df):
        cleaner = DataCleaner()

        with pytest.raises(RuntimeError):
            cleaner.transform(sample_df)


# =============================================================================
# 2. WINDOW BUILDER — recorte temporal
# =============================================================================

class TestWindowBuilder:

    def test_window_metadata_added(self, sample_df, window_config):
        builder = WindowBuilder(window_config)

        df = builder.apply(sample_df)

        assert "_obs_window_days" in df.columns
        assert "_pred_window_days" in df.columns

    def test_no_purchase_flag_exists(self, sample_df, window_config):
        df = sample_df.copy()
        df.loc[:10, "Last_Purchase_Days_Ago"] = 200

        builder = WindowBuilder(window_config)
        df = builder.apply(df)

        assert "_no_purchase_in_window" in df.columns
        assert df["_no_purchase_in_window"].sum() > 0

    def test_input_dataframe_not_modified(self, sample_df, window_config):
        original_cols = set(sample_df.columns)

        builder = WindowBuilder(window_config)
        _ = builder.apply(sample_df)

        assert set(sample_df.columns) == original_cols


# =============================================================================
# 3. SEGMENTATION — comportamento do cliente
# =============================================================================

class TestCustomerSegmenter:

    def test_segments_are_created(self, sample_df):
        seg = CustomerSegmenter()
        df = seg.fit_transform(sample_df)

        assert "segment_tenure" in df.columns
        assert "segment_recency" in df.columns
        assert "risk_score_heuristic" in df.columns

    def test_tenure_bucket_logic(self):
        seg = CustomerSegmenter()

        df = pd.DataFrame({
            "Years_as_Customer": [1, 5, 15],
            "Num_of_Purchases": [10, 50, 90],
            "Last_Purchase_Days_Ago": [10, 50, 200],
        })

        df = seg.fit_transform(df)

        assert df.loc[0, "segment_tenure"] in ["new", "early"]
        assert df.loc[2, "segment_tenure"] in ["mature", "veteran"]

    def test_risk_score_ordering(self):
        seg = CustomerSegmenter()

        df = pd.DataFrame({
            "Years_as_Customer": [1, 15],
            "Num_of_Purchases": [5, 100],
            "Last_Purchase_Days_Ago": [300, 10],
        })

        df = seg.fit_transform(df)

        assert df.loc[0, "risk_score_heuristic"] > df.loc[1, "risk_score_heuristic"]

    def test_no_null_encoded_columns(self, sample_df):
        seg = CustomerSegmenter()
        df = seg.fit_transform(sample_df)

        encoded_cols = [c for c in df.columns if c.endswith("_enc")]

        for col in encoded_cols:
            assert df[col].isnull().sum() == 0


# =============================================================================
# 4. INTEGRAÇÃO LEVE (apenas preprocessing)
# =============================================================================

class TestPreprocessingIntegration:

    def test_preprocessing_pipeline_runs(self, sample_df, window_config):
        cleaner = DataCleaner()
        builder = WindowBuilder(window_config)
        seg = CustomerSegmenter()

        df = cleaner.fit_transform(sample_df)
        df = builder.apply(df)
        df = seg.fit_transform(df)

        assert len(df) > 0
        assert "Target_Churn" in df.columns
        assert "Customer_ID" not in df.columns

    def test_no_nulls_after_preprocessing(self, sample_df, window_config):
        cleaner = DataCleaner()
        builder = WindowBuilder(window_config)
        seg = CustomerSegmenter()

        df = cleaner.fit_transform(sample_df)
        df = builder.apply(df)
        df = seg.fit_transform(df)

        assert df.isnull().sum().sum() == 0