"""
SmartFoodAI - Data Preprocessing & Validation Module
Handles loading, validation, cleaning, feature engineering, and Scikit-Learn preprocessing pipelines.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 1. Feature Definitions
CATEGORICAL_FEATURES = [
    'Type of Food',
    'Event Type',
    'Storage Conditions',
    'Purchase History',
    'Seasonality',
    'Preparation Method',
    'Geographical Location',
    'Pricing'
]

NUMERICAL_FEATURES = [
    'Number of Guests',
    'Quantity of Food',
    'Food per Guest'  # Engineered feature (safe for ML input)
]

TARGET_VARIABLE = 'Wastage Food Amount'

# EDA-only target-derived metrics (DO NOT use as ML model inputs to prevent target leakage)
TARGET_DERIVED_METRICS = [
    'Wastage Rate',
    'Waste per Guest'
]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load food wastage dataset from CSV file.
    Validates file existence and readability.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    return pd.read_csv(filepath)


def validate_schema(df: pd.DataFrame) -> dict:
    """
    Verify expected columns and data types in the dataset.
    Returns a validation summary dictionary.
    """
    base_required = set([
        'Type of Food', 'Number of Guests', 'Event Type', 'Quantity of Food',
        'Storage Conditions', 'Purchase History', 'Seasonality',
        'Preparation Method', 'Geographical Location', 'Pricing',
        TARGET_VARIABLE
    ])
    actual_columns = set(df.columns)
    missing_columns = list(base_required - actual_columns)
    
    return {
        "is_valid": len(missing_columns) == 0,
        "missing_columns": missing_columns,
        "total_rows": len(df),
        "total_columns": len(df.columns)
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset:
    - Strips whitespace from string columns
    - Handles missing values
    - Removes duplicate records to prevent train-test contamination
    - Validates physical/logical constraints
    """
    df_clean = df.copy()

    # Strip whitespace in string columns
    str_cols = df_clean.select_dtypes(include=['object', 'string']).columns
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # Impute missing values if any exist
    for col in ['Number of Guests', 'Quantity of Food', TARGET_VARIABLE]:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # Enforce physical constraints
    df_clean = df_clean[df_clean['Number of Guests'] > 0]
    df_clean = df_clean[df_clean['Quantity of Food'] > 0]
    if TARGET_VARIABLE in df_clean.columns:
        df_clean[TARGET_VARIABLE] = df_clean[TARGET_VARIABLE].clip(lower=0)
        df_clean[TARGET_VARIABLE] = np.minimum(
            df_clean[TARGET_VARIABLE],
            df_clean['Quantity of Food']
        )

    return df_clean


def engineer_features(df: pd.DataFrame, include_target_metrics: bool = True) -> pd.DataFrame:
    """
    Feature engineering:
    - 'Food per Guest': Safe input feature (Quantity of Food / Number of Guests)
    
    If include_target_metrics is True (for EDA and reporting only):
    - 'Wastage Rate': Percentage of prepared food wasted (Wastage / Quantity * 100)
    - 'Waste per Guest': Waste per attendee (Wastage / Number of Guests)
    
    CRITICAL: Never include target-derived metrics in ML training to avoid target leakage!
    """
    df_fe = df.copy()

    # Safe predictor feature: Food per Guest
    df_fe['Food per Guest'] = np.round(
        df_fe['Quantity of Food'] / np.maximum(df_fe['Number of Guests'], 1), 3
    )

    if include_target_metrics and TARGET_VARIABLE in df_fe.columns:
        # Wastage Rate (%)
        df_fe['Wastage Rate'] = np.round(
            (df_fe[TARGET_VARIABLE] / np.maximum(df_fe['Quantity of Food'], 0.001)) * 100, 2
        )
        # Waste per Guest
        df_fe['Waste per Guest'] = np.round(
            df_fe[TARGET_VARIABLE] / np.maximum(df_fe['Number of Guests'], 1), 3
        )

    return df_fe


def build_preprocessor() -> ColumnTransformer:
    """
    Creates Scikit-learn ColumnTransformer:
    - Categorical pipeline: One-Hot Encoding (handles unknown categories gracefully)
    - Numerical pipeline: StandardScaler for uniform feature scaling
    """
    num_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extract transformed feature names after fitting preprocessor.
    Essential for SHAP explainability and feature importance analysis.
    """
    feature_names = []
    # Numerical features
    feature_names.extend(NUMERICAL_FEATURES)
    # One-hot encoded categorical features
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(cat_names.tolist())
    return feature_names


def prepare_train_test_data(cleaned_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepares X (features) and y (target), ensuring target leakage prevention.
    Splits into 80% train and 20% test subsets.
    """
    # Ensure engineered input features exist
    if 'Food per Guest' not in cleaned_df.columns:
        cleaned_df = engineer_features(cleaned_df, include_target_metrics=False)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    # Strictly verify target-derived metrics are NOT in feature_cols
    for leaked_col in TARGET_DERIVED_METRICS:
        assert leaked_col not in feature_cols, f"Target leakage detected: {leaked_col} in features!"

    X = cleaned_df[feature_cols].copy()
    y = cleaned_df[TARGET_VARIABLE].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
